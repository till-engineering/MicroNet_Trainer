"""Loss des Trainings: BCE + Dice (Dice über den ganzen Batch – stabil, auch wenn Patches keine Pore haben)."""

import torch.nn.functional as F
import torch


def bce_dice(logits: torch.Tensor, target: torch.Tensor, bce_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    overlap = (prob * target).sum()
    dice = 1 - (2 * overlap + 1) / (prob.sum() + target.sum() + 1)
    return bce_weight * bce + (1 - bce_weight) * dice
