"""검색 평가. python -m eval.retrieval --checkpoint runs/.../best.pt [--config ...]"""

import argparse, json, sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config
from model import ZModel
from data import load_csv, build_segments
from retrieval import BM25Index, DenseEncoder


def mrr(ranks):
    return sum(1.0 / r for r in ranks) / len(ranks) if ranks else 0

def top_k_acc(ranks, k):
    return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="실험별 config.py 경로")
    args = parser.parse_args()
    cfg = load_config(args.config)

    texts = load_csv(cfg["csv_path"], cfg["text_column"], cfg["num_docs"])
    model = ZModel(cfg["llm_name"], num_segments=1)
    tok = model.tok

    seg_ids, seg_texts, seg_to_doc = [], [], []
    for di, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for s, e in build_segments(len(ids), cfg["segment_len"]):
            seg_ids.append(torch.tensor(ids[s:e], dtype=torch.long))
            seg_texts.append(tok.decode(ids[s:e]))
            seg_to_doc.append(di)

    del model; torch.cuda.empty_cache()
    model = ZModel(cfg["llm_name"], num_segments=len(seg_ids))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.z.load_state_dict(ckpt["z"])
    model.eval()

    # TODO: query 데이터 로드 (CSV의 queries 컬럼)
    # 지금은 placeholder — query가 있는 데이터로 실행할 때 채울 것
    print("retrieval eval: query 데이터 필요. config에 query 경로 추가 후 실행.")


if __name__ == "__main__":
    main()
