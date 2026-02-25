"""CSV 로드 + 토큰화 + 세그먼트 분할."""

import csv
import torch
from .segment import build_segments


def load_csv(csv_path, text_column, num_docs):
    """CSV 파일에서 텍스트를 읽어 list[str]로 반환."""
    texts = []
    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= num_docs:
                break
            text = row[text_column].strip()
            if text:
                texts.append(text)
    return texts


def tokenize_and_segment(texts, tokenizer, segment_len):
    """텍스트 리스트를 토큰화하고 세그먼트로 분할.

    Returns:
        seg_ids: list[Tensor] - 각 세그먼트의 token id
        seg_texts: list[str] - 각 세그먼트의 디코딩된 텍스트
        seg_to_doc: list[int] - 각 세그먼트가 속한 문서 인덱스
    """
    seg_ids = []
    seg_texts = []
    seg_to_doc = []

    eos_id = tokenizer.eos_token_id

    for doc_idx, text in enumerate(texts):
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        segments = build_segments(len(token_ids), segment_len)
        for i, (start, end) in enumerate(segments):
            ids = token_ids[start:end] + [eos_id]
            seg_ids.append(torch.tensor(ids, dtype=torch.long))
            seg_texts.append(tokenizer.decode(token_ids[start:end]))
            seg_to_doc.append(doc_idx)

    return seg_ids, seg_texts, seg_to_doc
