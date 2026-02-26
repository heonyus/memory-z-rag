"""v62+v56 combined QA: by_score / by_rank 방식으로 retrieval → Gemini QA.

실행:
  cd ~/data/memory-z-rag
  uv run python experiments/v62_summary_contrastive/qa_combined.py \
    --sum_ckpt experiments/v62_summary_contrastive/runs/20260226_074017/best.pt \
    --text_ckpt experiments/v56_triviaqa_recon/runs/20260226_073923/best.pt \
    --method by_score
"""

import argparse, ast, csv, json, sys, time, torch
import torch.nn.functional as F
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import load_config
from model import ZModel
from data import load_csv, tokenize_and_segment
from eval.qa_utils import best_metrics, build_qa_prompt, call_gemini_with_retry


def load_z_and_texts(ckpt_path, config_path, tokenizer):
    """체크포인트에서 z_matrix + seg_texts + seg_to_doc 로드."""
    cfg = load_config(config_path)
    texts = load_csv(cfg["csv_path"], cfg["text_column"], cfg["num_docs"])
    seg_ids, seg_texts, seg_to_doc = tokenize_and_segment(texts, tokenizer, cfg["segment_len"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    z = F.normalize(ckpt["z_embeddings"]["weight"].float(), dim=1)
    return z, seg_texts, seg_to_doc


def find_best(scores, seg_to_doc, doc_idx):
    """scores에서 해당 doc의 best rank + score + 전체 ranked list."""
    ranked = torch.argsort(scores, descending=True).tolist()
    doc_segs = set(si for si, d in enumerate(seg_to_doc) if d == doc_idx)
    for r, si in enumerate(ranked):
        if si in doc_segs:
            return r, scores[si].item(), ranked
    return None, None, ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sum_ckpt", required=True)
    parser.add_argument("--sum_config", default="experiments/v62_summary_contrastive/config.py")
    parser.add_argument("--text_ckpt", required=True)
    parser.add_argument("--text_config", default="experiments/v56_triviaqa_recon/config.py")
    parser.add_argument("--method", required=True, choices=["by_score", "by_rank"])
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--top_k", default="1,5,10,20", help="top-k values (comma-separated)")
    args = parser.parse_args()
    top_k_list = [int(k) for k in args.top_k.split(",")]

    # tokenizer + LLM
    sum_cfg = load_config(args.sum_config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(sum_cfg["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = ZModel(sum_cfg["llm_name"], num_segments=1)
    model.eval()

    # z-matrices + seg_texts
    sum_z, sum_texts, sum_s2d = load_z_and_texts(args.sum_ckpt, args.sum_config, tokenizer)
    text_z, text_texts, text_s2d = load_z_and_texts(args.text_ckpt, args.text_config, tokenizer)
    sum_z, text_z = sum_z.to("cuda"), text_z.to("cuda")

    print(f"summary: {sum_z.shape[0]} seg | text: {text_z.shape[0]} seg | "
          f"method={args.method} | top_k={top_k_list}")

    # query-doc-answers
    entries = []
    with open(sum_cfg["csv_path"], encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= sum_cfg["num_docs"]:
                break
            for q in ast.literal_eval(row["queries"]):
                entries.append({
                    "doc_idx": i,
                    "query": q,
                    "answers": ast.literal_eval(row["answers"]),
                })

    results_by_k = {k: {"em": [], "f1": [], "examples": [], "wins": {"sum": 0, "text": 0}}
                    for k in top_k_list}

    for entry in entries:
        doc_idx, query, gold_answers = entry["doc_idx"], entry["query"], entry["answers"]

        # query embedding
        q_tok = tokenizer(query, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            q_embed = model.llm.get_input_embeddings()(q_tok["input_ids"].to("cuda"))
        q_embed = F.normalize(q_embed.mean(dim=1).squeeze(0).float(), dim=0)

        # 각 pool 검색 (1회)
        sum_rank, sum_score, sum_ranked = find_best(sum_z @ q_embed, sum_s2d, doc_idx)
        text_rank, text_score, text_ranked = find_best(text_z @ q_embed, text_s2d, doc_idx)

        # winner 결정
        if sum_rank is not None and text_rank is not None:
            if args.method == "by_score":
                winner = "sum" if sum_score >= text_score else "text"
            else:
                winner = "sum" if sum_rank <= text_rank else "text"
        elif sum_rank is not None:
            winner = "sum"
        elif text_rank is not None:
            winner = "text"
        else:
            winner = "text"

        chosen_rank = sum_rank if winner == "sum" else text_rank

        # 각 top_k에 대해 Gemini QA
        for k in top_k_list:
            if winner == "sum":
                context = "\n".join(sum_texts[si] for si in sum_ranked[:k])
            else:
                context = "\n".join(text_texts[si] for si in text_ranked[:k])

            prompt = build_qa_prompt(context, query)
            pred = call_gemini_with_retry(prompt, model_name=args.model, k=k)

            em, f1 = best_metrics(pred, gold_answers)
            results_by_k[k]["em"].append(em)
            results_by_k[k]["f1"].append(f1)
            results_by_k[k]["wins"][winner] += 1
            results_by_k[k]["examples"].append({
                "doc_idx": doc_idx, "query": query, "winner": winner,
                "sum_rank": sum_rank, "text_rank": text_rank,
                "pred": pred, "gold": gold_answers[0],
                "em": em, "f1": round(f1, 4),
            })
            time.sleep(0.3)

        print(f"  doc={doc_idx:2d} winner={winner:4s} rank={chosen_rank} "
              + " ".join(f"k{k}={results_by_k[k]['em'][-1]:.0f}/{results_by_k[k]['f1'][-1]:.2f}"
                         for k in top_k_list))

    # 결과
    total = max(len(entries), 1)
    all_metrics = {}
    for k in top_k_list:
        r = results_by_k[k]
        all_metrics[f"top{k}"] = {
            "em": sum(r["em"]) / total,
            "f1": sum(r["f1"]) / total,
        }

    output_dir = Path(args.sum_ckpt).parent
    output_file = f"qa_{args.method}.json"
    result = {
        "metrics": all_metrics,
        "method": args.method,
        "num_queries": len(entries),
        "model": args.model,
        "winner_distribution": {str(k): results_by_k[k]["wins"] for k in top_k_list},
        "examples_by_k": {str(k): results_by_k[k]["examples"] for k in top_k_list},
    }
    json.dump(result, open(output_dir / output_file, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'':>6s} {'EM':>7s} {'F1':>7s}")
    print("-" * 25)
    for k in top_k_list:
        m = all_metrics[f"top{k}"]
        print(f"  k={k:<3d} {m['em']:7.4f} {m['f1']:7.4f}")
    print(f"\nwinner 분포 (k={top_k_list[0]}): {results_by_k[top_k_list[0]]['wins']}")
    print(f"saved → {output_dir / output_file}")


if __name__ == "__main__":
    main()
