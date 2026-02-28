"""v65: MLP projector + sigmoid contrastive loss.

MLP가 z를 input embedding 공간으로 매핑.
Sigmoid (SigLIP) loss로 학습 → continual learning 대비.
"""

CONFIG = {
    # projector (MLP)
    "projector_hidden": 1024,
    "projector_layers": 2,       # 3072 → 1024 → GELU → 3072
    "projector_dropout": 0.0,
    "use_projected_z": True,

    # contrastive
    "contrastive_loss_type": "sigmoid",
    "contrastive_lambda": 0.1,

    # 저장
    "save_dir": "experiments/v65_mlp_projector/runs",
}
