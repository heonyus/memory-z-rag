"""기존 CSV에 Gemini 요약(doc_sum) 컬럼을 추가한다.

사용법:
    python data/add_summary.py data/docs.csv
    python data/add_summary.py data/docs.csv --output data/docs_with_summary.csv
"""

import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


def generate_summary(client, model_name, text):
    prompt = f"""Summarize the following document into a single concise paragraph of approximately 100 words. Preserve all key entities (names, dates, numbers, locations) and the main topic. Do not add any information not present in the original document.

    Document:
    {text}

    Summary:"""
    resp = client.models.generate_content(
        model=model_name, contents=prompt,
        config={"max_output_tokens": 256},
    )
    return resp.text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="입력 CSV (text 컬럼 필요)")
    parser.add_argument("--output", default=None, help="출력 CSV (기본: 입력 파일 덮어쓰기)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY가 .env에 없습니다.")
        return

    from google import genai
    client = genai.Client(api_key=api_key)

    # CSV 읽기
    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{len(rows)}개 문서 요약 생성 중...")
    for row in tqdm(rows, desc="요약 생성"):
        row["doc_sum"] = generate_summary(client, args.model, row["text"])

    # CSV 쓰기
    out = Path(args.output or args.input)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"저장: {out}")


if __name__ == "__main__":
    main()
