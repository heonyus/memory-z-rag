"""MLP projector: z_embedding → input embedding 공간 매핑.

hidden_dim > 0이면 MLP를 생성하고, 0이면 nn.Identity()를 반환.
"""

import torch.nn as nn


def build_projector(hidden_size, hidden_dim=0, num_layers=1, dropout=0.0):
    """MLP projector 생성. hidden_dim=0이면 Identity (하위 호환)."""
    if hidden_dim <= 0:
        return nn.Identity()

    num_layers = max(1, num_layers)
    layers = []
    in_dim = hidden_size

    for _ in range(num_layers - 1):
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    # 마지막 레이어: hidden_size로 복원
    layers.append(nn.Linear(in_dim, hidden_size))
    return nn.Sequential(*layers)
