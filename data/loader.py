"""CSV 로드 + 토큰화 + 세그먼트 분할."""

import csv
import torch
from .segment import build_segments


def load_csv(csv_path, text_column, num_docs, new_text_column=None, base_num_docs=None,
             append_column=None, append_num_docs=None):
    """CSV 파일에서 텍스트를 읽어 list[str]로 반환.

    new_text_column + base_num_docs가 설정되면:
      - doc 0..base_num_docs-1 → text_column
      - doc base_num_docs..num_docs-1 → new_text_column

    append_column이 설정되면:
      - 먼저 num_docs개를 text_column으로 로드
      - 그 다음 같은 CSV 행 0..append_num_docs-1을 append_column으로 다시 로드해서 뒤에 추가
    """
    texts = []
    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= num_docs:
                break
            if new_text_column and base_num_docs is not None and i >= base_num_docs:
                text = row[new_text_column].strip()
            else:
                text = row[text_column].strip()
            if text:
                texts.append(text)

    # append: 같은 CSV 행을 다른 컬럼으로 다시 읽어 추가
    if append_column:
        n_append = append_num_docs or num_docs
        with open(csv_path, encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i >= n_append:
                    break
                text = row[append_column].strip()
                if text:
                    texts.append(text)

    return texts


def tokenize_and_segment(texts, tokenizer, segment_len,
                         new_segment_len=None, base_num_docs=None):
    """텍스트 리스트를 토큰화하고 세그먼트로 분할.

    new_segment_len + base_num_docs가 설정되면:
      - doc 0..base_num_docs-1 → segment_len
      - doc base_num_docs.. → new_segment_len

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
        if new_segment_len and base_num_docs is not None and doc_idx >= base_num_docs:
            sl = new_segment_len
        else:
            sl = segment_len
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        segments = build_segments(len(token_ids), sl)
        for i, (start, end) in enumerate(segments):
            ids = token_ids[start:end] + [eos_id]
            seg_ids.append(torch.tensor(ids, dtype=torch.long))
            seg_texts.append(tokenizer.decode(token_ids[start:end]))
            seg_to_doc.append(doc_idx)

    return seg_ids, seg_texts, seg_to_doc
