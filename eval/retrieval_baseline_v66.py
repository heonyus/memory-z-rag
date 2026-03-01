"""v66: Answer-based hit 기준 Baseline 검색 평가.

v48(mz-RAG)과 동일한 평가 방식: 리트리브된 세그먼트에 정답 텍스트가 포함되어야 hit.
기존 retrieval_baseline.py는 gold doc의 아무 세그먼트면 hit (doc-based) → 과대평가 문제.

실행:
  python -m eval.retrieval_baseline_v66 --retriever bm25
  python -m eval.retrieval_baseline_v66 --retriever e5
  python -m eval.retrieval_baseline_v66 --retriever contriever
  python -m eval.retrieval_baseline_v66 --retriever dpr
"""

import argparse, ast, csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from data import load_csv, tokenize_and_segment
from eval.retrievers import build_retriever, PRESETS

_TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_answer(text):
    lowered = text.lower().strip()
    lowered = _TOKEN_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def _segment_contains_answer(seg_text, answers):
    """v48 방식: 세그먼트 텍스트에 정답 텍스트가 포함되어 있는지 확인."""
    seg_norm = _normalize_answer(seg_text)
    for alias in answers:
        alias_norm = _normalize_answer(alias)
        if alias_norm and alias_norm in seg_norm:
            return True
    return False


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

    # answers 로드
    all_answers = {}  # doc_idx → list[str]
    with open(cfg["csv_path"], encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= cfg["num_docs"]:
                break
            all_answers[i] = ast.literal_eval(row["answers"])

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
    print(f"[v66] answer-based hit criterion (v48 방식)")

    # 검색 평가 (answer-based hit)
    top_counts = {k: 0 for k in top_k}
    mrr_total = 0.0
    examples = []

    for doc_idx, query in pairs:
        ranked = retriever.rank(query)

        # answer-based hit: 정답 텍스트가 포함된 세그먼트의 rank 찾기
        answers = all_answers.get(doc_idx, [])
        rank = None
        for r, si in enumerate(ranked):
            if _segment_contains_answer(seg_texts[si], answers):
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

    output_dir = Path("experiments/v66_answer_hit") / args.retriever
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "retrieval.json"
    json.dump({"metrics": metrics, "examples": examples},
              open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n[v66 {args.retriever}] MRR={metrics['mrr']:.4f}", end="")
    for k in top_k:
        print(f"  Top{k}={metrics[f'top{k}_acc']:.4f}", end="")
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
