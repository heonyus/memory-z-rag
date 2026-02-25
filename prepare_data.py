"""TriviaQA에서 문서를 선택하고 CSV로 저장한다.

사용법:
    python prepare_data.py                    # 50개 문서
    python prepare_data.py --num_docs 200     # 200개 문서
    python prepare_data.py --with_summary     # Gemini 요약 포함
"""

import argparse
import csv
import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from config import CONFIG


# ── 세그먼트 분할 ──

def build_segments(length, segment_len):
    """토큰 길이를 segment_len 단위로 나눈다. [(start, end), ...] 반환."""
    if length <= 0:
        return [(0, 1)]
    if length <= segment_len:
        return [(0, length)]
    ranges = []
    start = 0
    while start < length:
        end = min(start + segment_len, length)
        ranges.append((start, end))
        if end >= length:
            break
        start += segment_len
    return ranges


# ── TriviaQA 문서 추출 ──

def extract_text(example):
    """TriviaQA 예제에서 가장 긴 wiki_context 텍스트를 추출한다."""
    entity_pages = example.get("entity_pages", {})
    contexts = entity_pages.get("wiki_context", [])
    if not contexts:
        return None
    # 가장 긴 텍스트 선택
    texts = [t for t in contexts if isinstance(t, str) and t.strip()]
    if not texts:
        return None
    return max(texts, key=len).strip()


def extract_query(example):
    """질문 텍스트 추출."""
    q = example.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return None


def extract_answers(example):
    """정답 리스트 추출 (aliases + normalized_aliases)."""
    answer = example.get("answer", {})
    aliases = answer.get("aliases", [])
    normalized = answer.get("normalized_aliases", [])
    # 중복 제거
    seen = set()
    result = []
    for a in aliases + normalized:
        if isinstance(a, str) and a.strip() and a.strip() not in seen:
            seen.add(a.strip())
            result.append(a.strip())
    return result


# ── Gemini 요약 생성 ──

def generate_summary(client, model_name, text, max_tokens=256):
    """Gemini API로 문서 요약을 생성한다."""
    prompt = (
        "Summarize the following text in 2-3 sentences. "
        "Keep all key facts and names.\n\n"
        f"Text: {text[:3000]}\n\nSummary:"
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"max_output_tokens": max_tokens},
    )
    return response.text.strip()


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="TriviaQA → CSV 데이터 준비")
    parser.add_argument("--num_docs", type=int, default=CONFIG["num_docs"])
    parser.add_argument("--segment_len", type=int, default=CONFIG["segment_len"])
    parser.add_argument("--min_segments", type=int, default=3)
    parser.add_argument("--max_segments", type=int, default=5)
    parser.add_argument("--with_summary", action="store_true",
                        help="Gemini로 요약 생성")
    parser.add_argument("--output", type=str, default="data/docs.csv")
    args = parser.parse_args()

    # 토크나이저 로드
    print(f"토크나이저 로드: {CONFIG['llm_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # TriviaQA 데이터셋 로드
    print("TriviaQA 로드 중...")
    dataset = load_dataset("trivia_qa", "rc.wikipedia", split="train")

    # 문서 추출 + 필터링
    print("문서 추출 및 필터링...")
    docs = []
    seen_texts = set()

    for example in tqdm(dataset, desc="문서 필터링"):
        if len(docs) >= args.num_docs:
            break

        text = extract_text(example)
        if text is None or text in seen_texts:
            continue

        query = extract_query(example)
        if query is None:
            continue

        answers = extract_answers(example)

        # 토큰화
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        token_len = len(encoded["input_ids"])

        # 세그먼트 수 필터링 (3~5개인 문서만)
        num_segs = len(build_segments(token_len, args.segment_len))
        if num_segs < args.min_segments or num_segs > args.max_segments:
            continue

        seen_texts.add(text)

        # 이미 같은 텍스트의 문서가 있으면 query/answers만 추가
        existing = next((d for d in docs if d["text"] == text), None)
        if existing:
            existing["queries"].append(query)
            existing["answers"].extend(answers)
            continue

        docs.append({
            "text": text,
            "queries": [query],
            "answers": answers,
            "input_ids": json.dumps(encoded["input_ids"]),
        })

    print(f"선택된 문서: {len(docs)}개")

    # (선택) Gemini 요약 생성
    if args.with_summary:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("경고: GEMINI_API_KEY가 없어서 요약 생성을 건너뜁니다.")
            for doc in docs:
                doc["summary"] = ""
        else:
            from google import genai
            client = genai.Client(api_key=api_key)
            print("Gemini 요약 생성 중...")
            for doc in tqdm(docs, desc="요약 생성"):
                doc["summary"] = generate_summary(
                    client, "gemini-2.5-flash-lite", doc["text"]
                )
    else:
        for doc in docs:
            doc["summary"] = ""

    # CSV 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "input_ids", "queries",
                                                "answers", "summary"])
        writer.writeheader()
        for doc in docs:
            writer.writerow({
                "text": doc["text"],
                "input_ids": doc["input_ids"],
                "queries": json.dumps(doc["queries"], ensure_ascii=False),
                "answers": json.dumps(doc["answers"], ensure_ascii=False),
                "summary": doc["summary"],
            })

    print(f"저장 완료: {output_path} ({len(docs)}개 문서)")


if __name__ == "__main__":
    main()
