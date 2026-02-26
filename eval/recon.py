"""복원 평가: NLL + 생성 + ROUGE-L + Token F1.

실행: python -m eval.recon --checkpoint runs/.../best.pt [--config experiments/v55/config.py]
"""

import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from model import ZModel
from data import load_csv, tokenize_and_segment
from metrics import rouge_l_f1, token_f1, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="실험별 config.py 경로")
    args = parser.parse_args()
    config = load_config(args.config)

    # 데이터 로드 + 세그먼트 분할
    from transformers import AutoTokenizer
    texts = load_csv(config["csv_path"], config["text_column"], config["num_docs"])
    tokenizer = AutoTokenizer.from_pretrained(config["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    seg_ids, seg_texts, _ = tokenize_and_segment(texts, tokenizer, config["segment_len"])

    # 모델 + 체크포인트 로드
    model = ZModel(config["llm_name"], num_segments=len(seg_ids))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.z_embeddings.load_state_dict(checkpoint["z_embeddings"])
    model.eval()

    # 각 세그먼트 평가
    all_nlls, all_rouges, all_f1s = [], [], []
    samples = []
    max_tokens = config["max_new_tokens"]

    for seg_idx in range(len(seg_ids)):
        input_ids = seg_ids[seg_idx].unsqueeze(0).to("cuda")
        idx_tensor = torch.tensor([seg_idx], device="cuda")

        # NLL (teacher forcing)
        with torch.no_grad():
            nll_loss, _ = model(idx_tensor, input_ids, torch.ones_like(input_ids))

        # 생성 + 메트릭
        gen_ids = model.generate(seg_idx, max_new_tokens=max_tokens)
        ref_ids = seg_ids[seg_idx].tolist()[:max_tokens]
        gen_trimmed = gen_ids[:max_tokens]

        rouge = rouge_l_f1(gen_trimmed, ref_ids)
        f1 = token_f1(gen_trimmed, ref_ids)

        all_nlls.append(nll_loss.item())
        all_rouges.append(rouge)
        all_f1s.append(f1)

        samples.append({
            "seg_idx": seg_idx,
            "nll": nll_loss.item(),
            "rouge_l": rouge,
            "generated": tokenizer.decode(gen_ids),
            "reference": seg_texts[seg_idx],
        })
        print(f"seg {seg_idx}: NLL={nll_loss.item():.4f}  ROUGE-L={rouge:.4f}")

    # 결과 저장
    output_dir = Path(args.checkpoint).parent
    result = {
        "nll": summarize(all_nlls),
        "rouge_l": summarize(all_rouges),
        "token_f1": summarize(all_f1s),
        "num_segments": len(seg_ids),
    }
    json.dump(result, open(output_dir / "recon.json", "w"), indent=2)
    json.dump(samples, open(output_dir / "recon_samples.json", "w"),
              indent=2, ensure_ascii=False)

    print(f"\nNLL={result['nll']['mean']:.4f}  ROUGE-L={result['rouge_l']['mean']:.4f}")


if __name__ == "__main__":
    main()
