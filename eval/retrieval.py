"""검색 평가: query → z-embedding cosine similarity로 세그먼트 검색.

각 세그먼트는 독립적으로 학습된 단위. 193개 세그먼트 중에서 검색.

실행: python -m eval.retrieval --checkpoint runs/.../best.pt [--config ...]
"""

import argparse, ast, csv, json, sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from model import ZModel
from data import load_csv, tokenize_and_segment


def get_query_embedding(query, tokenizer, llm, device):
    """query 텍스트 → LLM input embedding mean pool."""
    tokens = tokenizer(query, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"].to(device)
    with torch.no_grad():
        token_embeds = llm.get_input_embeddings()(input_ids)
    return token_embeds.mean(dim=1).squeeze(0).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--top_k", default="1,5,10,20", help="top-k values (comma-separated)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    top_k = [int(k) for k in args.top_k.split(",")]

    # 데이터 로드 (tokenizer만 따로 로드해서 모델 1회만 생성)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    texts = load_csv(cfg["csv_path"], cfg["text_column"], cfg["num_docs"])
    seg_ids, seg_texts, seg_to_doc = tokenize_and_segment(texts, tokenizer, cfg["segment_len"])

    # 모델 + 체크포인트 (1회만 로드)
    model = ZModel(cfg["llm_name"], num_segments=len(seg_ids))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.z_embeddings.load_state_dict(ckpt["z_embeddings"])
    model.eval()

    # z-embedding matrix (normalized)
    z_matrix = model.z_embeddings.weight.data.float()
    z_matrix = F.normalize(z_matrix, dim=1)

    # query-doc pairs 로드
    pairs = []  # (doc_idx, query_str)
    with open(cfg["csv_path"], encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= cfg["num_docs"]:
                break
            queries = ast.literal_eval(row["queries"])
            for q in queries:
                pairs.append((i, q))

    print(f"{len(pairs)} queries, {len(seg_ids)} segments, top_k={top_k}")

    # 검색 평가
    top_counts = {k: 0 for k in top_k}
    mrr_total = 0.0
    examples = []

    for doc_idx, query in pairs:
        q_embed = get_query_embedding(query, tokenizer, model.llm, "cuda")
        q_embed = F.normalize(q_embed, dim=0)

        scores = z_matrix @ q_embed
        ranked = torch.argsort(scores, descending=True)

        # 해당 doc의 세그먼트 중 가장 높은 rank 찾기
        doc_seg_set = set(si for si, d in enumerate(seg_to_doc) if d == doc_idx)
        rank = None
        for r, seg_idx in enumerate(ranked.tolist()):
            if seg_idx in doc_seg_set:
                rank = r
                break

        if rank is not None:
            for k in top_k:
                if rank < k:
                    top_counts[k] += 1
            mrr_total += 1.0 / (rank + 1)

        examples.append({
            "doc_idx": doc_idx,
            "query": query,
            "rank": rank,
            "top5": ranked[:5].tolist(),
        })
        print(f"  doc={doc_idx} rank={rank} q={query[:60]}")

    # 결과
    total = max(len(pairs), 1)
    metrics = {f"top{k}_acc": top_counts[k] / total for k in top_k}
    metrics["mrr"] = mrr_total / total
    metrics["num_queries"] = len(pairs)
    metrics["num_segments"] = len(seg_ids)

    output_dir = Path(args.checkpoint).parent
    json.dump({"metrics": metrics, "examples": examples},
              open(output_dir / "retrieval.json", "w"), indent=2, ensure_ascii=False)

    print(f"\nMRR={metrics['mrr']:.4f}", end="")
    for k in top_k:
        print(f"  Top{k}={metrics[f'top{k}_acc']:.4f}", end="")
    print()


if __name__ == "__main__":
    main()
