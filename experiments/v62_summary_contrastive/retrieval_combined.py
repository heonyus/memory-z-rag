"""v62+v56 combined retrieval: summary z vs text z 별도 집계.

각 query에 대해:
  - summary pool (v62) 에서의 best rank / score
  - text pool (v56) 에서의 best rank / score
  - combined pool 에서의 best rank + 출처

실행:
  cd ~/data/memory-z-rag
  python experiments/v62_summary_contrastive/retrieval_combined.py \
    --sum_ckpt experiments/v62_summary_contrastive/runs/20260226_074017/best.pt \
    --text_ckpt experiments/v56_triviaqa_recon/runs/20260226_073923/best.pt
"""

import argparse, ast, csv, json, sys, torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import load_config
from model import ZModel
from data import load_csv, tokenize_and_segment


def load_z_matrix(ckpt_path, config_path, tokenizer):
    """체크포인트에서 z_matrix (normalized) + seg_to_doc 로드."""
    cfg = load_config(config_path)
    texts = load_csv(cfg["csv_path"], cfg["text_column"], cfg["num_docs"])
    _, _, seg_to_doc = tokenize_and_segment(texts, tokenizer, cfg["segment_len"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    z = ckpt["z_embeddings"]["weight"].float()
    z = F.normalize(z, dim=1)
    return z, seg_to_doc


def find_best_rank(scores, seg_to_doc, doc_idx):
    """scores 벡터에서 해당 doc의 세그먼트 중 best rank + score 반환."""
    ranked = torch.argsort(scores, descending=True)
    doc_segs = set(si for si, d in enumerate(seg_to_doc) if d == doc_idx)
    for r, si in enumerate(ranked.tolist()):
        if si in doc_segs:
            return r, scores[si].item()
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sum_ckpt", required=True, help="v62 summary checkpoint")
    parser.add_argument("--sum_config", default="experiments/v62_summary_contrastive/config.py")
    parser.add_argument("--text_ckpt", required=True, help="v56 text checkpoint")
    parser.add_argument("--text_config", default="experiments/v56_triviaqa_recon/config.py")
    args = parser.parse_args()

    # tokenizer + LLM (1회만 로드)
    sum_cfg = load_config(args.sum_config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(sum_cfg["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ZModel은 query embedding 용으로만 사용 (아무 크기나 OK)
    model = ZModel(sum_cfg["llm_name"], num_segments=1)
    model.eval()

    # z-matrices 로드
    sum_z, sum_s2d = load_z_matrix(args.sum_ckpt, args.sum_config, tokenizer)
    text_z, text_s2d = load_z_matrix(args.text_ckpt, args.text_config, tokenizer)
    sum_z, text_z = sum_z.to("cuda"), text_z.to("cuda")

    # merged pool: 두 z-matrix를 concat, seg_to_doc도 합침
    merged_z = torch.cat([sum_z, text_z], dim=0)  # (50+193, 3072)
    merged_s2d = list(sum_s2d) + list(text_s2d)

    print(f"summary pool: {sum_z.shape[0]} seg  |  text pool: {text_z.shape[0]} seg  |  merged: {merged_z.shape[0]} seg")

    # query-doc pairs
    pairs = []
    with open(sum_cfg["csv_path"], encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= sum_cfg["num_docs"]:
                break
            for q in ast.literal_eval(row["queries"]):
                pairs.append((i, q))

    # 집계
    top_k = [1, 5, 10, 20]
    methods = ["sum", "text", "by_score", "by_rank", "merged"]
    stats = {m: {"mrr": 0.0, **{k: 0 for k in top_k}} for m in methods}
    examples = []

    for doc_idx, query in pairs:
        tokens = tokenizer(query, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            q_embed = model.llm.get_input_embeddings()(tokens["input_ids"].to("cuda"))
        q_embed = F.normalize(q_embed.mean(dim=1).squeeze(0).float(), dim=0)

        # 각 pool 별도 검색
        sum_rank, sum_score = find_best_rank(sum_z @ q_embed, sum_s2d, doc_idx)
        text_rank, text_score = find_best_rank(text_z @ q_embed, text_s2d, doc_idx)
        merged_rank, merged_score = find_best_rank(merged_z @ q_embed, merged_s2d, doc_idx)

        # by_score: cosine score가 높은 쪽 채택
        # by_rank: rank가 낮은(좋은) 쪽 채택
        if sum_rank is not None and text_rank is not None:
            score_src = "sum" if sum_score >= text_score else "text"
            rank_src = "sum" if sum_rank <= text_rank else "text"
        elif sum_rank is not None:
            score_src = rank_src = "sum"
        elif text_rank is not None:
            score_src = rank_src = "text"
        else:
            score_src = rank_src = "?"

        score_rank = sum_rank if score_src == "sum" else text_rank
        rank_rank = sum_rank if rank_src == "sum" else text_rank

        # 집계
        for name, rank in [("sum", sum_rank), ("text", text_rank),
                           ("by_score", score_rank), ("by_rank", rank_rank),
                           ("merged", merged_rank)]:
            if rank is not None:
                stats[name]["mrr"] += 1.0 / (rank + 1)
                for k in top_k:
                    if rank < k:
                        stats[name][k] += 1

        examples.append({
            "doc": doc_idx,
            "sum_rank": sum_rank, "sum_score": round(sum_score, 4) if sum_score else None,
            "text_rank": text_rank, "text_score": round(text_score, 4) if text_score else None,
            "by_score": score_src, "by_rank": rank_src,
            "merged_rank": merged_rank, "merged_score": round(merged_score, 4) if merged_score else None,
            "query": query[:80],
        })
        print(f"  doc={doc_idx:2d}  sum={str(sum_rank):>4s}({sum_score:.4f})"
              f"  text={str(text_rank):>4s}({text_score:.4f})"
              f"  by_score={score_src:4s} by_rank={rank_src:4s}  q={query[:50]}")

    # 결과 출력
    total = max(len(pairs), 1)
    print(f"\n{'':15s} {'MRR':>7s} {'Top1':>7s} {'Top5':>7s} {'Top10':>7s} {'Top20':>7s}")
    print("-" * 60)
    for name in methods:
        s = stats[name]
        mrr = s["mrr"] / total
        row = f"  {name:13s} {mrr:7.4f}"
        for k in top_k:
            row += f" {s[k]/total:7.4f}"
        print(row)

    # winner 분포
    score_wins = {"sum": 0, "text": 0}
    rank_wins = {"sum": 0, "text": 0}
    for e in examples:
        if e["by_score"] in score_wins: score_wins[e["by_score"]] += 1
        if e["by_rank"] in rank_wins: rank_wins[e["by_rank"]] += 1
    print(f"\nby_score 분포: sum={score_wins['sum']}, text={score_wins['text']}")
    print(f"by_rank  분포: sum={rank_wins['sum']}, text={rank_wins['text']}")

    # JSON 저장
    output_path = Path(args.sum_ckpt).parent / "retrieval_combined.json"
    result = {
        "metrics": {},
        "winner_distribution": {"by_score": score_wins, "by_rank": rank_wins},
        "examples": examples,
    }
    for name in methods:
        s = stats[name]
        result["metrics"][name] = {
            "mrr": s["mrr"] / total,
            **{f"top{k}_acc": s[k] / total for k in top_k},
        }
    json.dump(result, open(output_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved → {output_path}")


if __name__ == "__main__":
    main()
