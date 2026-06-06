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
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute standard cross entropy loss (reduction='none' to retain batch shape)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # Move alpha to the same device as logits
            alpha_device = self.alpha.to(logits.device)
            # Gather alpha values matching the target classes
            alpha_t = alpha_device[targets]
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
        # Ensure classes are sorted by index
        sorted_keys = sorted(class_counts.keys())
        counts = [class_counts[k] for k in sorted_keys]
    else:
        counts = list(class_counts)

    num_classes = len(counts)
    total_samples = sum(counts)

    if total_samples == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    if method == "balanced":
        weights = []
        for c in counts:
            if c > 0:
                weights.append(total_samples / (num_classes * c))
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)

    elif method == "inverse":
        inv_counts = [1.0 / c if c > 0 else 0.0 for c in counts]
        sum_inv = sum(inv_counts)
        if sum_inv > 0:
            weights = [num_classes * ic / sum_inv for ic in inv_counts]
        else:
            weights = [1.0] * num_classes
        return torch.tensor(weights, dtype=torch.float32)

    else:
        return torch.ones(num_classes, dtype=torch.float32)
