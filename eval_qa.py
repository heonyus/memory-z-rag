"""QA 평가. 검색 결과를 context로 Gemini에 답변을 생성하고 EM/F1을 측정한다.

사용법:
    python eval_qa.py --checkpoint runs/.../best.pt
"""

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
from transformers import AutoTokenizer

from config import CONFIG, get_run_dir
from model import ZModel


# ── 유틸 함수 ──

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_segments(length, segment_len):
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


# ── QA 평가 지표 ──

_TOKEN_RE = re.compile(r"[^a-z0-9 ]")


def normalize_answer(text):
    """답변 정규화: 소문자, 특수문자 제거, 공백 정리."""
    text = text.lower().strip()
    text = _TOKEN_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred, gold_list):
    """정답 리스트 중 하나라도 일치하면 1."""
    pred_norm = normalize_answer(pred)
    return int(any(normalize_answer(g) == pred_norm for g in gold_list))


def f1_score(pred, gold_list):
    """정답 리스트 중 가장 높은 F1."""
    pred_tokens = normalize_answer(pred).split()
    best = 0.0
    for gold in gold_list:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


# ── 데이터 로딩 ──

def load_docs_and_queries(csv_path, text_column, num_docs, tokenizer):
    docs, queries = [], []
    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= num_docs:
                break
            text = row[text_column]
            if not text or not text.strip():
                continue
            encoded = tokenizer(text, add_special_tokens=False, truncation=False)
            doc_idx = len(docs)
            docs.append({"text": text, "input_ids": encoded["input_ids"]})
            raw_queries = json.loads(row.get("queries", "[]"))
            raw_answers = json.loads(row.get("answers", "[]"))
            if raw_queries:
                queries.append((doc_idx, raw_queries[0], raw_answers))
    return docs, queries


def split_into_segments(docs, tokenizer, segment_len):
    seg_ids_list, seg_to_doc, doc_to_segs, seg_texts = [], [], [], []
    for doc_idx, doc in enumerate(docs):
        ids = doc["input_ids"]
        segs = []
        for start, end in build_segments(len(ids), segment_len):
            seg_idx = len(seg_ids_list)
            seg_ids_list.append(torch.tensor(ids[start:end], dtype=torch.long))
            seg_to_doc.append(doc_idx)
            segs.append(seg_idx)
            seg_texts.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        doc_to_segs.append(segs)
    return seg_ids_list, seg_to_doc, doc_to_segs, seg_texts


# ── Gemini 답변 생성 ──

def ask_gemini(client, model_name, query, context, max_tokens=64):
    """Gemini에 query + context를 보내서 답변을 받는다."""
    prompt = (
        f"Answer the question based on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (short, factual):"
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"max_output_tokens": max_tokens},
            )
            return response.text.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  Gemini 오류: {e}")
                return ""


# ── 검색 함수 ──

def get_rankings(model, queries, seg_ids_list, seg_texts, tokenizer, device):
    """z-token과 BM25 검색 결과를 반환한다."""
    from rank_bm25 import BM25Okapi

    num_segs = len(seg_ids_list)
    results = {}

    # z-token 검색
    with torch.no_grad():
        z_matrix = F.normalize(model.z_embeddings.weight.data.float(), dim=1)

    z_rankings = []
    for doc_idx, query, answers in queries:
        tokens = tokenizer(query, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            q_embed = model.llm.get_input_embeddings()(tokens["input_ids"].to(device))
            q_embed = F.normalize(q_embed.mean(dim=1).squeeze(0).float(), dim=0)
        scores = z_matrix @ q_embed
        z_rankings.append(scores.argsort(descending=True).tolist())
    results["z_token"] = z_rankings

    # BM25 검색
    tokenized = [text.lower().split() for text in seg_texts]
    bm25 = BM25Okapi(tokenized)
    bm25_rankings = []
    for doc_idx, query, answers in queries:
        scores = bm25.get_scores(query.lower().split())
        bm25_rankings.append(sorted(range(len(scores)), key=lambda i: scores[i],
                                     reverse=True))
    results["bm25"] = bm25_rankings

    return results


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="QA 평가")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text_column", type=str, default=CONFIG["text_column"])
    parser.add_argument("--num_docs", type=int, default=CONFIG["num_docs"])
    parser.add_argument("--csv_path", type=str, default=CONFIG["csv_path"])
    parser.add_argument("--top_k", type=int, default=5,
                        help="context로 사용할 상위 세그먼트 수")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY가 필요합니다. .env 파일에 설정하세요.")
        return

    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로딩
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    docs, queries = load_docs_and_queries(
        args.csv_path, args.text_column, args.num_docs, tokenizer
    )
    seg_ids_list, seg_to_doc, doc_to_segs, seg_texts = split_into_segments(
        docs, tokenizer, CONFIG["segment_len"]
    )
    print(f"문서: {len(docs)}개, 세그먼트: {len(seg_ids_list)}개, 쿼리: {len(queries)}개")

    if not queries:
        print("쿼리가 없어서 QA 평가를 건너뜁니다.")
        return

    # 모델 로드
    model = ZModel(CONFIG["llm_name"], len(seg_ids_list), CONFIG["num_z_tokens"],
                    CONFIG["quantization"], device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.z_embeddings.load_state_dict(ckpt["z_embeddings"])

    # 검색
    print("\n=== 검색 수행 ===")
    all_rankings = get_rankings(model, queries, seg_ids_list, seg_texts, tokenizer, device)

    # Gemini QA
    client = genai.Client(api_key=api_key)
    model_name = CONFIG["gemini_model"]
    max_tokens = CONFIG["gemini_max_tokens"]
    run_dir = get_run_dir(prefix="qa")
    results = {}

    for method, rankings in all_rankings.items():
        print(f"\n=== {method} QA ===")
        em_scores, f1_scores_list = [], []
        examples = []

        for q_idx, (doc_idx, query, answers) in enumerate(tqdm(queries, desc=f"{method}")):
            # 상위 k개 세그먼트를 context로 사용
            top_segs = rankings[q_idx][:args.top_k]
            context = "\n".join(seg_texts[s] for s in top_segs)

            # Gemini에 질문
            pred = ask_gemini(client, model_name, query, context, max_tokens)

            em = exact_match(pred, answers)
            f1 = f1_score(pred, answers)
            em_scores.append(em)
            f1_scores_list.append(f1)

            if len(examples) < 10:
                examples.append({
                    "query": query,
                    "pred": pred,
                    "answers": answers,
                    "em": em,
                    "f1": f1,
                })

        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        avg_f1 = sum(f1_scores_list) / len(f1_scores_list) if f1_scores_list else 0.0

        results[method] = {
            "em": avg_em,
            "f1": avg_f1,
            "num_queries": len(queries),
        }
        print(f"  EM={avg_em:.3f}, F1={avg_f1:.3f}")

        with open(run_dir / f"qa_{method}_examples.json", "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    # 결과 요약
    print(f"\n=== QA 평가 요약 ===")
    print(f"{'Method':<15} {'EM':>8} {'F1':>8}")
    print("-" * 31)
    for method, m in results.items():
        print(f"{method:<15} {m['em']:>8.3f} {m['f1']:>8.3f}")

    with open(run_dir / "qa_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  결과 저장: {run_dir}")


if __name__ == "__main__":
    main()
