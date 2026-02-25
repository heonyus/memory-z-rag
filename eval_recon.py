"""복원 평가. z-token에서 생성한 텍스트와 원문을 비교한다.

사용법:
    python eval_recon.py --checkpoint runs/.../best.pt
"""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

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


# ── 평가 지표 ──

def lcs_length(a, b):
    """최장 공통 부분수열(LCS) 길이."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l_f1(gen_ids, ref_ids):
    """ROUGE-L F1 (token ID 기반)."""
    if not gen_ids or not ref_ids:
        return 0.0
    lcs = lcs_length(gen_ids, ref_ids)
    precision = lcs / len(gen_ids)
    recall = lcs / len(ref_ids)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(gen_ids, ref_ids):
    """Token F1 (unigram overlap 기반)."""
    if not gen_ids or not ref_ids:
        return 0.0
    gen_counts = Counter(gen_ids)
    ref_counts = Counter(ref_ids)
    overlap = sum((gen_counts & ref_counts).values())
    precision = overlap / len(gen_ids)
    recall = overlap / len(ref_ids)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── 데이터 로딩 ──

def load_docs_from_csv(csv_path, text_column, num_docs, tokenizer):
    docs = []
    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= num_docs:
                break
            text = row[text_column]
            if not text or not text.strip():
                continue
            encoded = tokenizer(text, add_special_tokens=False, truncation=False)
            docs.append({"text": text, "input_ids": encoded["input_ids"]})
    return docs


def split_into_segments(docs, tokenizer, segment_len):
    seg_ids_list, seg_to_doc, seg_texts = [], [], []
    for doc_idx, doc in enumerate(docs):
        ids = doc["input_ids"]
        for start, end in build_segments(len(ids), segment_len):
            seg_ids_list.append(torch.tensor(ids[start:end], dtype=torch.long))
            seg_to_doc.append(doc_idx)
            seg_texts.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
    return seg_ids_list, seg_to_doc, seg_texts


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="복원 평가")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text_column", type=str, default=CONFIG["text_column"])
    parser.add_argument("--num_docs", type=int, default=CONFIG["num_docs"])
    parser.add_argument("--csv_path", type=str, default=CONFIG["csv_path"])
    parser.add_argument("--gen_samples", type=int, default=50,
                        help="생성 평가할 세그먼트 수")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로딩
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    docs = load_docs_from_csv(args.csv_path, args.text_column, args.num_docs, tokenizer)
    seg_ids_list, seg_to_doc, seg_texts = split_into_segments(
        docs, tokenizer, CONFIG["segment_len"]
    )
    num_segments = len(seg_ids_list)
    print(f"문서: {len(docs)}개, 세그먼트: {num_segments}개")

    # 모델 로드
    model = ZModel(CONFIG["llm_name"], num_segments, CONFIG["num_z_tokens"],
                    CONFIG["quantization"], device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.z_embeddings.load_state_dict(ckpt["z_embeddings"])
    print(f"체크포인트 로드: {args.checkpoint}")

    # 생성 평가할 세그먼트 샘플링
    gen_count = min(args.gen_samples, num_segments)
    gen_set = set(random.sample(range(num_segments), gen_count))

    # 평가
    nlls, ppls = [], []
    rouge_scores, f1_scores = [], []
    examples = []
    max_tokens = CONFIG["max_new_tokens"]
    max_ref = CONFIG["max_ref_tokens"]

    model.eval()
    with torch.no_grad():
        for seg_idx in tqdm(range(num_segments), desc="복원 평가"):
            seg_input = seg_ids_list[seg_idx].unsqueeze(0).to(device)
            seg_mask = torch.ones_like(seg_input, device=device)
            idx_tensor = torch.tensor([seg_idx], device=device)

            outputs = model(idx_tensor, seg_input, seg_mask)
            nll = outputs["loss"].item()
            nlls.append(nll)
            ppls.append(math.exp(nll))

            # 생성 평가 (샘플만)
            if seg_idx in gen_set:
                generated = model.generate(seg_idx, max_new_tokens=max_tokens)
                gen_ids = tokenizer.encode(generated, add_special_tokens=False)[:max_ref]
                ref_ids = seg_ids_list[seg_idx].tolist()[:max_ref]

                r = rouge_l_f1(gen_ids, ref_ids)
                f = token_f1(gen_ids, ref_ids)
                rouge_scores.append(r)
                f1_scores.append(f)

                if len(examples) < 20:
                    examples.append({
                        "seg_idx": seg_idx,
                        "nll": nll,
                        "rouge_l": r,
                        "token_f1": f,
                        "generated": generated[:300],
                        "reference": seg_texts[seg_idx][:300],
                    })

    # 결과 요약
    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "nll_mean": avg(nlls),
        "ppl_mean": avg(ppls),
        "rouge_l_mean": avg(rouge_scores),
        "token_f1_mean": avg(f1_scores),
        "num_segments": num_segments,
        "gen_samples": gen_count,
    }

    print(f"\n=== 복원 평가 결과 ===")
    print(f"  NLL:       {summary['nll_mean']:.4f}")
    print(f"  PPL:       {summary['ppl_mean']:.4f}")
    print(f"  ROUGE-L:   {summary['rouge_l_mean']:.4f}")
    print(f"  Token F1:  {summary['token_f1_mean']:.4f}")

    # 저장
    run_dir = get_run_dir(prefix="recon")
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "examples.json", "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"  결과 저장: {run_dir}")


if __name__ == "__main__":
    main()
