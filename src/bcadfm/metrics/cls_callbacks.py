from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import TrainerCallback


@dataclass
class SaveTwoBestClsModelsConfig:
    """Configuration for saving the best classification models.

    Saves two separate model state dicts during training:
    - best_loss.pt: best according to lowest eval_loss.
    - best_f1.pt: best according to highest eval_f1.
    """

    filename_best_loss: str = "best_loss.pt"
    filename_best_f1: str = "best_f1.pt"


class SaveTwoBestClsModelsCallback(TrainerCallback):
    """Trainer callback to save best models by loss and F1.

    On each evaluation step, this callback checks eval_loss and eval_f1
    from the metrics dict. If either improves over the best value seen
    so far, the current model state dict is saved under run_dir.
    """

    def __init__(
        self,
        run_dir: str,
        config: Optional[SaveTwoBestClsModelsConfig] = None,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.config = config or SaveTwoBestClsModelsConfig()

        self.best_loss = float("inf")
        self.best_f1 = float("-inf")

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # type: ignore[override]
        model = kwargs.get("model")
        if model is None:
            return control

        loss = metrics.get("eval_loss")
        f1 = metrics.get("eval_f1")

        # Save best by eval_loss (lower is better)
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
            path_loss = os.path.join(self.run_dir, self.config.filename_best_loss)
            torch.save(model.state_dict(), path_loss)

        # Save best by eval_f1 (higher is better)
        if f1 is not None and f1 > self.best_f1:
            self.best_f1 = f1
            path_f1 = os.path.join(self.run_dir, self.config.filename_best_f1)
            torch.save(model.state_dict(), path_f1)

        return control
