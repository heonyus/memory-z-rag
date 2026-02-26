"""Baseline QA 평가: retriever별 검색 → Gemini QA → EM/F1.

실행:
  python -m eval.qa_baseline --retriever bm25
  python -m eval.qa_baseline --retriever e5
  python -m eval.qa_baseline --retriever contriever
  python -m eval.qa_baseline --retriever dpr
"""

import argparse, ast, csv, json, sys, time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from data import load_csv, tokenize_and_segment
from eval.retrievers import build_retriever, PRESETS
from eval.qa_utils import best_metrics, build_qa_prompt, call_gemini_with_retry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True, choices=list(PRESETS.keys()))
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model")
    parser.add_argument("--top_k", default="1,5,10,20")
    args = parser.parse_args()
    cfg = load_config(args.config)
    top_k_list = [int(k) for k in args.top_k.split(",")]

    # 데이터 로드
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    texts = load_csv(cfg["csv_path"], cfg["text_column"], cfg["num_docs"])
    _, seg_texts, seg_to_doc = tokenize_and_segment(texts, tokenizer, cfg["segment_len"])

    # retriever 생성
    print(f"Building {args.retriever} retriever for {len(seg_texts)} segments...")
    retriever = build_retriever(args.retriever, seg_texts)
    print(f"Ready. top_k={top_k_list}, model={args.model}")

    # query-doc-answers 로드
    entries = []
    with open(cfg["csv_path"], encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= cfg["num_docs"]:
                break
            queries = ast.literal_eval(row["queries"])
            answers = ast.literal_eval(row["answers"])
            for q in queries:
                entries.append({"doc_idx": i, "query": q, "answers": answers})

    print(f"{len(entries)} queries")

    # QA 평가
    results_by_k = {k: {"em": [], "f1": [], "examples": []} for k in top_k_list}

    for entry in entries:
        doc_idx, query, gold_answers = entry["doc_idx"], entry["query"], entry["answers"]

        # retrieval
        ranked = retriever.rank(query)

        # best rank for gold doc
        doc_segs = set(si for si, d in enumerate(seg_to_doc) if d == doc_idx)
        rank = next((r for r, si in enumerate(ranked) if si in doc_segs), None)

        # 각 top_k에 대해 Gemini QA
        for k in top_k_list:
            context = "\n".join(seg_texts[si] for si in ranked[:k])
            prompt = build_qa_prompt(context, query)
            pred = call_gemini_with_retry(prompt, model_name=args.model, k=k)

            em, f1 = best_metrics(pred, gold_answers)
            results_by_k[k]["em"].append(em)
            results_by_k[k]["f1"].append(f1)
            results_by_k[k]["examples"].append({
                "doc_idx": doc_idx, "query": query, "rank": rank,
                "retrieved_seg": ranked[0], "pred": pred,
                "gold": gold_answers[0] if gold_answers else "",
                "em": em, "f1": round(f1, 4),
            })
            time.sleep(0.3)

        print(f"  doc={doc_idx:2d} rank={rank} "
              + " ".join(f"k{k}={results_by_k[k]['em'][-1]:.0f}/{results_by_k[k]['f1'][-1]:.2f}"
                         for k in top_k_list))

    # 결과 저장
    output_dir = Path("experiments/v64_baselines") / args.retriever
    output_dir.mkdir(parents=True, exist_ok=True)

    total = max(len(entries), 1)
    all_metrics = {}
    for k in top_k_list:
        r = results_by_k[k]
        all_metrics[f"top{k}"] = {
            "em": sum(r["em"]) / total,
            "f1": sum(r["f1"]) / total,
        }

    result = {
        "retriever": args.retriever,
        "metrics": all_metrics,
        "num_queries": len(entries),
        "num_segments": len(seg_texts),
        "model": args.model,
        "examples_by_k": {str(k): results_by_k[k]["examples"] for k in top_k_list},
    }
    out_path = output_dir / "qa.json"
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n[{args.retriever}]")
    print(f"{'':>6s} {'EM':>7s} {'F1':>7s}")
    print("-" * 25)
    for k in top_k_list:
        m = all_metrics[f"top{k}"]
        print(f"  k={k:<3d} {m['em']:7.4f} {m['f1']:7.4f}")
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
