from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance.

    For multi-class classification:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the correct class,
    and alpha_t is a weighting factor for the correct class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute standard cross entropy loss (reduction='none' to retain batch shape)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # Gather alpha values matching the target classes, ensuring correct device placement
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(
    class_counts: Union[List[int], Dict[int, int]],
    method: str = "balanced",
) -> torch.Tensor:
    """Compute class weights based on dataset statistics.

    Supported methods:
      - "balanced": weight_c = total_samples / (num_classes * count_c)
      - "inverse": weight_c = 1.0 / count_c, normalized to sum to num_classes
      - "none": uniform weights (1.0 for all classes)
    """
    if isinstance(class_counts, dict):
        # Ensure we cover all indices up to max class index to prevent shape mismatch in loss (H4 Fix)
        max_class_id = max(class_counts.keys()) if class_counts else 0
        counts = [class_counts.get(i, 0) for i in range(max_class_id + 1)]
    else:
        counts = list(class_counts)

    # Number of active classes (classes with samples)
    active_num_classes = sum(1 for c in counts if c > 0)
    active_num_classes = max(1, active_num_classes)
    total_samples = sum(counts)

    if total_samples == 0:
        return torch.ones(len(counts), dtype=torch.float32)

    if method == "balanced":
        weights = []
        for c in counts:
            if c > 0:
                weights.append(total_samples / (active_num_classes * c))
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)

    elif method == "inverse":
        inv_counts = [1.0 / c if c > 0 else 0.0 for c in counts]
        sum_inv = sum(inv_counts)
        if sum_inv > 0:
            weights = [active_num_classes * ic / sum_inv for ic in inv_counts]
        else:
            weights = [1.0] * len(counts)
        return torch.tensor(weights, dtype=torch.float32)

    else:
        return torch.ones(len(counts), dtype=torch.float32)
