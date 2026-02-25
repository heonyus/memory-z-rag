"""NLL loss (cross-entropy with ignore_index=-100)."""

import torch.nn.functional as F


def nll_loss(logits, labels):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
