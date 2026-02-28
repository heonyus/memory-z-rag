"""v65: MLP projector + sigmoid contrastive loss.

MLP가 z를 input embedding 공간으로 매핑.
Sigmoid (SigLIP) loss로 학습 → continual learning 대비.
"""

CONFIG = {
    # 모델
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",

    # projector (MLP)
    "projector_hidden": 1024,
    "projector_layers": 2,       # 4096 → 1024 → GELU → 4096
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
    "save_dir": "experiments/v65_mlp_projector/runs",
}
