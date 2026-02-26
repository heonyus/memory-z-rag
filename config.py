"""설정. 기본값 + 실험별 오버라이드."""

import importlib.util
from pathlib import Path


def load_config(config_path=None):
    """기본 CONFIG에 실험별 config를 오버라이드해서 반환.

    config_path: Python 파일 경로 (CONFIG dict 포함).
    None이면 기본값만 사용.
    """
    config = dict(CONFIG)  # 기본값 복사

    if config_path is not None:
        spec = importlib.util.spec_from_file_location("exp_config", config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        config.update(mod.CONFIG)

    return config


CONFIG = {
    # 모델
    "llm_name": "meta-llama/Llama-3.2-3B",

    # 데이터
    "csv_path": "data/docs_with_summary.csv",
    "text_column": "text",       # "text" = 원문, "doc_sum" = 요약
    "num_docs": 50,
    "segment_len": 128,

    # 학습
    "optimizer": "adamw",        # "adamw" or "sgd"
    "lr": 1e-2,
    "epochs": 300,
    "grad_clip": 1.0,
    "contrastive_lambda": 0.1,
    "contrastive_temperature": 0.07,
    "checkpoint_every": 50,
    "early_stop_patience": 0,   # 0이면 비활성화

    # 평가
    "max_new_tokens": 128,

    # 기타
    "seed": 42,
    "save_dir": "runs",
}
