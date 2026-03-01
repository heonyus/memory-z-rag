"""검색 평가: query → z-embedding cosine similarity로 세그먼트 검색.

Answer-based hit 기준 (DPR/Contriever 표준):
리트리브된 세그먼트에 정답 텍스트가 포함되어 있어야 hit.

실행: python -m eval.retrieval --checkpoint runs/.../best.pt [--config ...]
"""

import argparse, ast, csv, json, re, sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from eval.model_loader import load_eval_model

_TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_answer(text):
    lowered = text.lower().strip()
    lowered = _TOKEN_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def _segment_contains_answer(seg_text, answers):
    """DPR 표준: 세그먼트 텍스트에 정답 텍스트가 포함되어 있는지 확인."""
    seg_norm = _normalize_answer(seg_text)
    for alias in answers:
        alias_norm = _normalize_answer(alias)
        if alias_norm and alias_norm in seg_norm:
            return True
    return False


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

    # 모델 + 데이터 로드
    model, tokenizer, z_matrix, seg_ids, seg_texts, seg_to_doc = load_eval_model(args.checkpoint, cfg)

    # query-doc pairs + answers 로드
    pairs = []  # (doc_idx, query_str)
    all_answers = {}  # doc_idx → list[str]
    with open(cfg["csv_path"], encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= cfg["num_docs"]:
                break
            queries = ast.literal_eval(row["queries"])
            answers = ast.literal_eval(row["answers"])
            all_answers[i] = answers
            for q in queries:
                pairs.append((i, q))

    print(f"{len(pairs)} queries, {len(seg_ids)} segments, top_k={top_k}")

    # 검색 평가 (answer-based hit)
    top_counts = {k: 0 for k in top_k}
    mrr_total = 0.0
    examples = []

    for doc_idx, query in pairs:
        q_embed = get_query_embedding(query, tokenizer, model.llm, "cuda")
        q_embed = F.normalize(q_embed, dim=0)

        scores = z_matrix @ q_embed
        ranked = torch.argsort(scores, descending=True)

        # answer-based hit: 정답 텍스트가 포함된 세그먼트의 rank 찾기
        answers = all_answers.get(doc_idx, [])
        rank = None
        for r, seg_idx in enumerate(ranked.tolist()):
            if _segment_contains_answer(seg_texts[seg_idx], answers):
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
