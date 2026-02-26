"""Baseline 검색 평가: BM25 / E5 / Contriever / DPR → MRR + Top-k.

실행:
  python -m eval.retrieval_baseline --retriever bm25
  python -m eval.retrieval_baseline --retriever e5
  python -m eval.retrieval_baseline --retriever contriever
  python -m eval.retrieval_baseline --retriever dpr
"""

import argparse, ast, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from data import load_csv, tokenize_and_segment
from eval.retrievers import build_retriever, PRESETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True, choices=list(PRESETS.keys()))
    parser.add_argument("--config", default=None)
    parser.add_argument("--top_k", default="1,5,10,20")
    args = parser.parse_args()
    cfg = load_config(args.config)
    top_k = [int(k) for k in args.top_k.split(",")]

    # 데이터 로드
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    texts = load_csv(cfg["csv_path"], cfg["text_column"], cfg["num_docs"])
    _, seg_texts, seg_to_doc = tokenize_and_segment(texts, tokenizer, cfg["segment_len"])

    # retriever 생성
    print(f"Building {args.retriever} retriever for {len(seg_texts)} segments...")
    retriever = build_retriever(args.retriever, seg_texts)

    # query-doc pairs
    pairs = []
    with open(cfg["csv_path"], encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= cfg["num_docs"]:
                break
            for q in ast.literal_eval(row["queries"]):
                pairs.append((i, q))

    print(f"{len(pairs)} queries, top_k={top_k}")

    # 검색 평가
    top_counts = {k: 0 for k in top_k}
    mrr_total = 0.0
    examples = []

    for doc_idx, query in pairs:
        ranked = retriever.rank(query)

        doc_seg_set = set(si for si, d in enumerate(seg_to_doc) if d == doc_idx)
        rank = None
        for r, si in enumerate(ranked):
            if si in doc_seg_set:
                rank = r
                break

        if rank is not None:
            for k in top_k:
                if rank < k:
                    top_counts[k] += 1
            mrr_total += 1.0 / (rank + 1)

        examples.append({"doc_idx": doc_idx, "query": query, "rank": rank, "top5": ranked[:5]})
        print(f"  doc={doc_idx} rank={rank} q={query[:60]}")

    # 결과
    total = max(len(pairs), 1)
    metrics = {f"top{k}_acc": top_counts[k] / total for k in top_k}
    metrics["mrr"] = mrr_total / total
    metrics["num_queries"] = len(pairs)
    metrics["num_segments"] = len(seg_texts)

    output_dir = Path("experiments/v64_baselines") / args.retriever
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "retrieval.json"
    json.dump({"metrics": metrics, "examples": examples},
              open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n[{args.retriever}] MRR={metrics['mrr']:.4f}", end="")
    for k in top_k:
        print(f"  Top{k}={metrics[f'top{k}_acc']:.4f}", end="")
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
