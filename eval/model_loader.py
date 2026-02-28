"""평가용 모델 + 데이터 로딩.

eval/retrieval.py와 eval/qa.py에서 공통으로 사용하는 로딩 패턴.
"""

import torch
import torch.nn.functional as F

from config import load_config
from model import ZModel
from data import load_csv, tokenize_and_segment


def load_eval_model(checkpoint_path, config):
    """체크포인트에서 모델 + 데이터 로드.

    Returns: (model, tokenizer, z_matrix, seg_ids, seg_texts, seg_to_doc)
        z_matrix: [num_segments, hidden_size] normalized (projected if configured).
    """
    # 데이터 로드 + 세그먼트 분할
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["llm_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    texts = load_csv(config["csv_path"], config["text_column"], config["num_docs"])
    seg_ids, seg_texts, seg_to_doc = tokenize_and_segment(texts, tokenizer, config["segment_len"])

    # 체크포인트에서 projector 설정 읽기
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})

    model = ZModel(
        config["llm_name"], num_segments=len(seg_ids),
        projector_hidden=ckpt_cfg.get("projector_hidden", config.get("projector_hidden", 0)),
        projector_layers=ckpt_cfg.get("projector_layers", config.get("projector_layers", 1)),
        projector_dropout=ckpt_cfg.get("projector_dropout", config.get("projector_dropout", 0.0)),
    )
    model.z_embeddings.load_state_dict(ckpt["z_embeddings"])
    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"])
    model.eval()

    # z_matrix: projected z 사용 여부에 따라 분기
    use_projected = ckpt_cfg.get("use_projected_z", config.get("use_projected_z", False))
    z_matrix = model.z_embeddings.weight.data.float()
    if use_projected:
        with torch.no_grad():
            z_matrix = model.project_z(z_matrix)
    z_matrix = F.normalize(z_matrix, dim=1)

    return model, tokenizer, z_matrix, seg_ids, seg_texts, seg_to_doc
