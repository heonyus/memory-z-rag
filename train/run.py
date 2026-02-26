"""z-token 학습.

실행: python -m train.run [--config experiments/v55/config.py]
"""

import argparse
import json
import logging
import random
import sys
import torch
from datetime import datetime
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from model import ZModel
from data import load_csv, tokenize_and_segment
from loss import contrastive_loss, build_content_matrix


def save_checkpoint(model, config, path, epoch=0):
    torch.save({"z_embeddings": model.z_embeddings.state_dict(), "config": config, "epoch": epoch}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="실험별 config.py 경로")
    parser.add_argument("--resume", default=None, help="체크포인트 경로 (이어서 학습)")
    args = parser.parse_args()

    config = load_config(args.config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # 실행 디렉토리
    if args.resume:
        run_dir = Path(args.resume).parent
    else:
        run_dir = Path(config["save_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True)
    json.dump(config, open(run_dir / "config.json", "w"), indent=2)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)

    # 데이터 로드 + 세그먼트 분할
    texts = load_csv(config["csv_path"], config["text_column"], config["num_docs"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    seg_ids, seg_texts, seg_to_doc = tokenize_and_segment(texts, tokenizer, config["segment_len"])

    # 모델 생성 (정확한 세그먼트 수로)
    model = ZModel(config["llm_name"], num_segments=len(seg_ids))
    log.info(f"{len(texts)} docs → {len(seg_ids)} segments")

    # resume: 체크포인트에서 z_embeddings 로드 + start_epoch 설정
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.z_embeddings.load_state_dict(ckpt["z_embeddings"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    # contrastive loss용 content matrix
    content_matrix = None
    if config["contrastive_lambda"] > 0:
        content_matrix = build_content_matrix(model.llm, seg_ids)

    # optimizer
    OptimizerClass = torch.optim.AdamW if config["optimizer"] == "adamw" else torch.optim.SGD
    optimizer = OptimizerClass(
        model.z_embeddings.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    # warmup scheduler
    scheduler = None
    warmup_iters = config.get("warmup_iters", 0)
    if warmup_iters > 0:
        start_factor = config.get("warmup_start_factor", 0.2)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters,
        )

    best_nll = float("inf")
    patience_counter = 0
    patience = config["early_stop_patience"]

    # 학습 루프
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        indices = list(range(len(seg_ids)))
        random.shuffle(indices)
        epoch_nll_losses = []
        epoch_con_losses = []

        for seg_idx in tqdm(indices, desc=f"epoch {epoch}", leave=False):
            input_ids = seg_ids[seg_idx].unsqueeze(0).to("cuda")
            idx_tensor = torch.tensor([seg_idx], device="cuda")

            optimizer.zero_grad(set_to_none=True)
            nll_loss, _ = model(idx_tensor, input_ids, torch.ones_like(input_ids))

            # contrastive loss 추가
            total_loss = nll_loss
            if content_matrix is not None:
                z_embed = model.z_embeddings(idx_tensor).squeeze(0)
                con_loss = contrastive_loss(
                    z_embed.unsqueeze(0), content_matrix, config["contrastive_temperature"],
                    label_idx=seg_idx,
                )
                total_loss = nll_loss + config["contrastive_lambda"] * con_loss
                epoch_con_losses.append(con_loss.item())

            total_loss.backward()
            clip_grad_norm_(model.z_embeddings.parameters(), config["grad_clip"])
            optimizer.step()
            epoch_nll_losses.append(nll_loss.item())

        if scheduler is not None:
            scheduler.step()

        # epoch 로깅 + 체크포인트
        avg_nll = sum(epoch_nll_losses) / len(epoch_nll_losses)
        msg = f"[epoch {epoch}/{config['epochs']}] nll={avg_nll:.4f}  best={best_nll:.4f}"
        if epoch_con_losses:
            avg_con = sum(epoch_con_losses) / len(epoch_con_losses)
            msg += f"  con={avg_con:.4f}"
        log.info(msg)

        if avg_nll < best_nll:
            best_nll = avg_nll
            patience_counter = 0
            save_checkpoint(model, config, run_dir / "best.pt", epoch)
        else:
            patience_counter += 1

        latest_path = run_dir / f"latest_epoch{epoch}.pt"
        for old in run_dir.glob("latest_epoch*.pt"):
            old.unlink()
        save_checkpoint(model, config, latest_path, epoch)

        if config["checkpoint_every"] and (epoch + 1) % config["checkpoint_every"] == 0:
            save_checkpoint(model, config, run_dir / f"epoch{epoch+1}.pt", epoch)

        # early stopping
        if patience > 0 and patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break
    log.info(f"Done → {run_dir}")


if __name__ == "__main__":
    main()
