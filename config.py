"""설정 파일. 모든 설정을 CONFIG dict 하나로 관리한다."""

from datetime import datetime
from pathlib import Path

CONFIG = {
    # ── 모델 ──
    "llm_name": "meta-llama/Llama-3.2-3B",
    "quantization": "4bit",       # "4bit", "8bit", or "none"
    "num_z_tokens": 1,            # 세그먼트당 z-token 수

    # ── 데이터 ──
    "csv_path": "data/docs.csv",  # prepare_data.py가 생성
    "text_column": "text",        # "text" (원문) 또는 "summary" (요약문)
    "num_docs": 50,
    "segment_len": 128,           # 세그먼트 길이 (토큰 수)

    # ── 학습 ──
    "lr": 5.0,                    # SGD learning rate
    "epochs": 300,
    "grad_clip": 1.0,
    "contrastive_lambda": 0.1,    # contrastive loss 가중치
    "contrastive_temperature": 0.07,
    "checkpoint_every": 50,       # N 에포크마다 체크포인트 저장

    # ── 복원 평가 ──
    "max_new_tokens": 128,        # 생성 최대 토큰 수
    "max_ref_tokens": 128,        # 참조 텍스트 최대 토큰 수

    # ── 검색 평가 ──
    "retrieval_top_k": [1, 5, 10, 20],

    # ── QA 평가 (Gemini) ──
    "gemini_model": "gemini-2.5-flash-lite",
    "gemini_max_tokens": 64,

    # ── 기타 ──
    "seed": 42,
    "save_dir": "runs",
}


def get_run_dir(prefix: str = "") -> Path:
    """타임스탬프 기반 실행 디렉토리 생성."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}" if prefix else timestamp
    run_dir = Path(CONFIG["save_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
