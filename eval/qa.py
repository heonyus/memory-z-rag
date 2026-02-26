"""QA 평가: z-embedding retrieval → Gemini QA → EM/F1.

실행: python -m eval.qa --checkpoint runs/.../best.pt [--config ...]
"""

import argparse, ast, csv, json, sys, time, torch
import torch.nn.functional as F
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from model import ZModel
from data import load_csv, tokenize_and_segment
from eval.qa_utils import best_metrics, build_qa_prompt, call_gemini_with_retry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="실험별 config.py 경로")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model")
    parser.add_argument("--top_k", default="1,5,10,20", help="top-k values (comma-separated)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    top_k_list = [int(k) for k in args.top_k.split(",")]

    # 데이터 로드
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    texts = load_csv(cfg["csv_path"], cfg["text_column"], cfg["num_docs"])
    seg_ids, seg_texts, seg_to_doc = tokenize_and_segment(texts, tokenizer, cfg["segment_len"])

    # 모델 + 체크포인트
    model = ZModel(cfg["llm_name"], num_segments=len(seg_ids))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.z_embeddings.load_state_dict(ckpt["z_embeddings"])
    model.eval()

    z_matrix = F.normalize(model.z_embeddings.weight.data.float(), dim=1)

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

    print(f"{len(entries)} queries, {len(seg_ids)} segments, top_k={top_k_list}, model={args.model}")

    # top_k 별 집계
    results_by_k = {k: {"em": [], "f1": [], "examples": []} for k in top_k_list}

    for entry in entries:
        doc_idx, query, gold_answers = entry["doc_idx"], entry["query"], entry["answers"]

        # retrieval (1회)
        q_tok = tokenizer(query, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            q_embed = model.llm.get_input_embeddings()(q_tok["input_ids"].to("cuda"))
        q_embed = F.normalize(q_embed.mean(dim=1).squeeze(0).float(), dim=0)
        scores = z_matrix @ q_embed
        ranked = torch.argsort(scores, descending=True).tolist()

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

        # 로그 (top_k=1 기준)
        r1 = results_by_k[top_k_list[0]]
        print(f"  doc={doc_idx:2d} rank={rank} "
              + " ".join(f"k{k}={results_by_k[k]['em'][-1]:.0f}/{results_by_k[k]['f1'][-1]:.2f}"
                         for k in top_k_list))

    # 결과 저장
    output_dir = Path(args.checkpoint).parent
    total = max(len(entries), 1)
    all_metrics = {}
    for k in top_k_list:
        r = results_by_k[k]
        all_metrics[f"top{k}"] = {
            "em": sum(r["em"]) / total,
            "f1": sum(r["f1"]) / total,
        }

    result = {
        "metrics": all_metrics,
        "num_queries": len(entries),
        "num_segments": len(seg_ids),
        "model": args.model,
        "examples_by_k": {str(k): results_by_k[k]["examples"] for k in top_k_list},
    }
    json.dump(result, open(output_dir / "qa.json", "w"), indent=2, ensure_ascii=False)

    print(f"\n{'':>6s} {'EM':>7s} {'F1':>7s}")
    print("-" * 25)
    for k in top_k_list:
        m = all_metrics[f"top{k}"]
        print(f"  k={k:<3d} {m['em']:7.4f} {m['f1']:7.4f}")
    print(f"\nsaved → {output_dir / 'qa.json'}")


if __name__ == "__main__":
    main()
