"""z-token 학습. SGD lr=5.0으로 z_embeddings만 학습한다.

사용법:
    python train.py                             # 원문으로 학습
    python train.py --text_column summary       # 요약문으로 학습
    python train.py --epochs 5                  # 5 에포크만 (테스트)
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
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
    """토큰 길이를 segment_len 단위로 나눈다."""
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


# ── 데이터 로딩 ──

def load_docs_from_csv(csv_path, text_column, num_docs, tokenizer):
    """CSV에서 문서를 읽고 토큰화한다."""
    docs = []
    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= num_docs:
                break
            text = row[text_column]
            if not text or not text.strip():
                continue
            encoded = tokenizer(text, add_special_tokens=False, truncation=False)
            ids = encoded["input_ids"]
            docs.append({
                "text": text,
                "input_ids": ids,
                "queries": row.get("queries", "[]"),
                "answers": row.get("answers", "[]"),
            })
    return docs


def split_into_segments(docs, tokenizer, segment_len):
    """문서들을 segment_len 토큰 단위로 분할한다.

    Returns:
        seg_ids_list: 각 세그먼트의 token ids (list of tensors)
        seg_to_doc: 세그먼트 → 문서 매핑
        doc_to_segs: 문서 → 세그먼트 리스트 매핑
        seg_texts: 각 세그먼트의 텍스트
    """
    seg_ids_list = []
    seg_to_doc = []
    doc_to_segs = []
    seg_texts = []

    for doc_idx, doc in enumerate(docs):
        ids = doc["input_ids"]
        ranges = build_segments(len(ids), segment_len)
        segs_for_doc = []

        for start, end in ranges:
            seg_idx = len(seg_ids_list)
            segment = torch.tensor(ids[start:end], dtype=torch.long)
            seg_ids_list.append(segment)
            seg_to_doc.append(doc_idx)
            segs_for_doc.append(seg_idx)
            seg_texts.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))

        doc_to_segs.append(segs_for_doc)

    return seg_ids_list, seg_to_doc, doc_to_segs, seg_texts


# ── Content embedding 캐시 (contrastive loss용) ──

def build_content_matrix(model, seg_ids_list, device):
    """각 세그먼트의 mean-pooled token embedding을 미리 계산한다."""
    embeddings = []
    with torch.no_grad():
        for seg_ids in seg_ids_list:
            ids = seg_ids.unsqueeze(0).to(device)
            token_embed = model.llm.get_input_embeddings()(ids)  # (1, L, H)
            mean_embed = token_embed.mean(dim=1).squeeze(0)       # (H,)
            embeddings.append(mean_embed.float())
    matrix = torch.stack(embeddings, dim=0)  # (N, H)
    # L2 정규화 (cosine similarity용)
    return F.normalize(matrix, dim=1)


# ── Contrastive loss ──

def contrastive_loss(z_embed, seg_idx, content_matrix, temperature):
    """z-token embedding과 content embedding 간 contrastive loss.

    positive: 같은 세그먼트의 content embedding
    negative: 나머지 모든 세그먼트의 content embedding
    """
    z = F.normalize(z_embed.float().reshape(1, -1), dim=1)    # (1, H)
    scores = (z @ content_matrix.T).squeeze(0) / temperature   # (N,)

    # target = seg_idx (자기 자신이 positive)
    target = torch.tensor([seg_idx], device=scores.device)
    return F.cross_entropy(scores.unsqueeze(0), target)


# ── 체크포인트 저장/로드 ──

def save_checkpoint(path, model, epoch, config, history):
    torch.save({
        "epoch": epoch,
        "z_embeddings": model.z_embeddings.state_dict(),
        "config": config,
        "history": history,
    }, path)
    print(f"  체크포인트 저장: {path}")


# ── 메인 학습 루프 ──

def main():
    parser = argparse.ArgumentParser(description="z-token 학습")
    parser.add_argument("--text_column", type=str, default=CONFIG["text_column"])
    parser.add_argument("--num_docs", type=int, default=CONFIG["num_docs"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--csv_path", type=str, default=CONFIG["csv_path"])
    parser.add_argument("--resume", type=str, default=None, help="체크포인트 경로")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 데이터 로딩
    print(f"\n=== 데이터 로딩: {args.csv_path} (column={args.text_column}) ===")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    docs = load_docs_from_csv(args.csv_path, args.text_column, args.num_docs, tokenizer)
    print(f"  문서 수: {len(docs)}")

    # 2. 세그먼트 분할
    seg_ids_list, seg_to_doc, doc_to_segs, seg_texts = split_into_segments(
        docs, tokenizer, CONFIG["segment_len"]
    )
    num_segments = len(seg_ids_list)
    print(f"  세그먼트 수: {num_segments}")
    print(f"  문서당 평균 세그먼트: {num_segments / len(docs):.1f}")

    # 3. 모델 생성
    print(f"\n=== 모델 생성 ===")
    model = ZModel(
        llm_name=CONFIG["llm_name"],
        num_segments=num_segments,
        num_z_tokens=CONFIG["num_z_tokens"],
        quantization=CONFIG["quantization"],
        device=device,
    )

    # 4. Content matrix (contrastive loss용)
    contrastive_lambda = CONFIG["contrastive_lambda"]
    content_matrix = None
    if contrastive_lambda > 0:
        print("\n=== Content matrix 생성 ===")
        content_matrix = build_content_matrix(model, seg_ids_list, device)
        print(f"  shape: {content_matrix.shape}")

    # 5. Optimizer (SGD, z_embeddings만)
    optimizer = SGD(model.z_embeddings.parameters(), lr=args.lr)

    # 6. Resume
    start_epoch = 0
    history = {"nll": [], "contrastive": [], "total": []}
    if args.resume:
        print(f"\n체크포인트에서 재개: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.z_embeddings.load_state_dict(ckpt["z_embeddings"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        del ckpt

    # 7. 실행 디렉토리 생성
    run_dir = get_run_dir(prefix="train")
    with open(run_dir / "config.json", "w") as f:
        json.dump({**CONFIG, "text_column": args.text_column,
                    "num_docs": args.num_docs, "lr": args.lr}, f, indent=2)

    # 8. 학습 루프
    print(f"\n=== 학습 시작 (epoch {start_epoch}~{args.epochs-1}) ===")
    print(f"  optimizer=SGD, lr={args.lr}, contrastive_lambda={contrastive_lambda}")

    best_nll = min(history["nll"]) if history["nll"] else float("inf")
    temperature = CONFIG["contrastive_temperature"]
    grad_clip = CONFIG["grad_clip"]
    checkpoint_every = CONFIG["checkpoint_every"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        indices = list(range(num_segments))
        random.shuffle(indices)

        epoch_nll, epoch_con, epoch_total = [], [], []
        optimizer.zero_grad(set_to_none=True)

        for seg_idx in tqdm(indices, desc=f"Epoch {epoch:03d}", leave=False):
            seg_input = seg_ids_list[seg_idx].unsqueeze(0).to(device)
            seg_mask = torch.ones_like(seg_input, device=device)
            idx_tensor = torch.tensor([seg_idx], device=device)

            # Forward: NLL loss
            outputs = model(idx_tensor, seg_input, seg_mask)
            nll = outputs["loss"]

            # Contrastive loss
            con = torch.tensor(0.0, device=device)
            if contrastive_lambda > 0 and content_matrix is not None:
                z_embed = model.z_embeddings(idx_tensor).squeeze(0)
                con = contrastive_loss(z_embed, seg_idx, content_matrix, temperature)

            total_loss = nll + contrastive_lambda * con
            total_loss.backward()

            clip_grad_norm_(model.z_embeddings.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_nll.append(nll.item())
            epoch_con.append(con.item())
            epoch_total.append(total_loss.item())

        # 에포크 평균
        avg_nll = sum(epoch_nll) / len(epoch_nll)
        avg_con = sum(epoch_con) / len(epoch_con)
        avg_total = sum(epoch_total) / len(epoch_total)
        z_norm = model.z_embeddings.weight.data.norm().item()

        history["nll"].append(avg_nll)
        history["contrastive"].append(avg_con)
        history["total"].append(avg_total)

        print(f"[Epoch {epoch:03d}] nll={avg_nll:.4f} | con={avg_con:.4f} | "
              f"total={avg_total:.4f} | z_norm={z_norm:.1f}")

        # 체크포인트 저장
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch{epoch+1}.pt",
                            model, epoch, CONFIG, history)

        if avg_nll < best_nll:
            best_nll = avg_nll
            save_checkpoint(run_dir / "best.pt", model, epoch, CONFIG, history)

    # 최종 저장
    save_checkpoint(run_dir / "latest.pt", model, args.epochs - 1, CONFIG, history)

    # 샘플 생성
    print("\n=== 샘플 생성 ===")
    sample_count = min(10, num_segments)
    samples = random.sample(range(num_segments), sample_count)
    sample_results = []
    for seg_idx in samples:
        generated = model.generate(seg_idx, max_new_tokens=CONFIG["max_new_tokens"])
        reference = seg_texts[seg_idx]
        sample_results.append({
            "seg_idx": seg_idx,
            "doc_idx": seg_to_doc[seg_idx],
            "generated": generated[:300],
            "reference": reference[:300],
        })
        print(f"\n[seg {seg_idx}] doc={seg_to_doc[seg_idx]}")
        print(f"  생성: {generated[:100]}...")
        print(f"  원문: {reference[:100]}...")

    with open(run_dir / "samples.json", "w") as f:
        json.dump(sample_results, f, indent=2, ensure_ascii=False)

    # 메타데이터 저장
    metadata = {
        "num_docs": len(docs),
        "num_segments": num_segments,
        "seg_to_doc": seg_to_doc,
        "doc_to_segs": doc_to_segs,
        "best_nll": best_nll,
        "text_column": args.text_column,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== 학습 완료 ===")
    print(f"  결과 디렉토리: {run_dir}")
    print(f"  best NLL: {best_nll:.4f}")


if __name__ == "__main__":
    main()
