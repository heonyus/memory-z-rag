"""TriviaQA에서 문서를 선택하고 CSV로 저장한다.

사용법:
    python data/prepare_data.py                        # 세그먼트 3~5개인 문서 50개
    python data/prepare_data.py --num_docs 200         # 200개
    python data/prepare_data.py --output data/out.csv  # 출력 경로 지정
"""

import argparse
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from config import CONFIG


def extract_text(example):
    """entity_pages에서 가장 긴 wiki_context를 꺼낸다."""
    contexts = example.get("entity_pages", {}).get("wiki_context", [])
    texts = [t for t in contexts if isinstance(t, str) and len(t) >= 50]
    return max(texts, key=len).strip() if texts else None


def count_segments(num_tokens, seg_len):
    if num_tokens <= 0:
        return 0
    return (num_tokens + seg_len - 1) // seg_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=CONFIG["num_docs"])
    parser.add_argument("--segment_len", type=int, default=CONFIG["segment_len"])
    parser.add_argument("--min_segments", type=int, default=3)
    parser.add_argument("--max_segments", type=int, default=5)
    parser.add_argument("--output", type=str, default="data/docs.csv")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("trivia_qa", "rc.wikipedia", split="train")
    seen = set()
    candidates = []

    for ex in tqdm(dataset, desc="TriviaQA 스캔"):
        text = extract_text(ex)
        if not text or text in seen:
            continue

        question = ex.get("question", "")
        aliases = ex.get("answer", {}).get("aliases", [])
        if not question or not aliases:
            continue

        seen.add(text)

        n_seg = count_segments(
            len(tokenizer(text, add_special_tokens=False)["input_ids"]),
            args.segment_len,
        )
        if args.min_segments <= n_seg <= args.max_segments:
            answers = list({a.strip() for a in aliases if isinstance(a, str) and a.strip()})
            candidates.append((text, question.strip(), answers))

        if len(candidates) >= args.num_docs:
            break

    print(f"선택: {len(candidates)}개 문서")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "input_ids", "queries", "answers"])
        w.writeheader()
        for text, query, answers in candidates:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            w.writerow({
                "text": text,
                "input_ids": json.dumps(ids),
                "queries": json.dumps([query], ensure_ascii=False),
                "answers": json.dumps(answers, ensure_ascii=False),
            })

    print(f"저장: {out} ({len(candidates)}개)")


if __name__ == "__main__":
    main()
