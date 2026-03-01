"""v68: v65와 동일 세팅, LLM만 Llama-3.2-3B-Instruct로 변경 (ablation).

비교 대상: v65 (Llama-3.1-8B-Instruct)
"""

CONFIG = {
    # 모델
    "llm_name": "meta-llama/Llama-3.2-3B-Instruct",

    # projector (MLP) — 3B hidden_size=3072 → 768 → GELU → 3072
    "projector_hidden": 768,
    "projector_layers": 2,
    "projector_dropout": 0.0,
    "use_projected_z": True,

    # 학습
    "optimizer": "sgd",
    "lr": 5.0,
    "epochs": 3000,

    # contrastive
    "contrastive_loss_type": "sigmoid",
    "contrastive_lambda": 0.1,

    # 저장
    "save_dir": "experiments/v68_3b_ablation/runs",
}
