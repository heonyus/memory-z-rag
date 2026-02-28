"""체크포인트 저장 / 로드 / resume 로직."""

import torch
import torch.nn as nn


def save_checkpoint(model, config, path, epoch=0, num_segments=0, frozen_prefix=0):
    """z_embeddings + projector + 메타 정보 저장."""
    torch.save({
        "z_embeddings": model.z_embeddings.state_dict(),
        "projector": model.projector.state_dict(),
        "config": config,
        "epoch": epoch,
        "num_segments": num_segments,
        "frozen_prefix": frozen_prefix,
    }, path)


def load_checkpoint(path, model, config):
    """체크포인트에서 모델 복원. z expansion + projector + freeze 처리.

    Returns: (start_epoch, frozen_prefix)
    """
    ckpt = torch.load(path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})

    # projector 관련 config를 체크포인트에서 복원
    for key in ["projector_hidden", "projector_layers", "projector_dropout",
                 "contrastive_loss_type", "use_projected_z",
                 "contrastive_lambda", "contrastive_temperature"]:
        if key in ckpt_cfg:
            config[key] = ckpt_cfg[key]

    # z_embeddings 로드 (z expansion 지원)
    z_weight = ckpt["z_embeddings"]["weight"]
    old_n = z_weight.shape[0]
    new_n = model.z_embeddings.weight.shape[0]

    if old_n > new_n:
        raise ValueError(f"checkpoint has {old_n} segments, but current dataset has {new_n}")

    model.z_embeddings.weight.data[:old_n].copy_(z_weight.to(model.z_embeddings.weight.device))
    if old_n < new_n:
        nn.init.normal_(model.z_embeddings.weight.data[old_n:], std=0.02)

    # projector 로드
    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"])

    # frozen prefix (continual learning: 기존 z 고정)
    frozen_prefix = 0
    if config.get("freeze_base_z", False):
        frozen_prefix = old_n

    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch, frozen_prefix
