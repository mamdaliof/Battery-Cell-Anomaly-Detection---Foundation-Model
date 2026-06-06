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
        # Only save checkpoints on the main process to avoid DDP race conditions
        if not state.is_world_process_zero:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        loss = metrics.get("eval_loss")
        f1 = metrics.get("eval_f1")

        improved_loss = loss is not None and loss < self.best_loss
        improved_f1   = f1  is not None and f1  > self.best_f1

        # Only pay the cost of state_dict() when at least one metric improved
        if improved_loss or improved_f1:
            unwrapped_model = model.module if hasattr(model, "module") else model
            state_dict = unwrapped_model.state_dict()   # one copy, shared if both improve

            if improved_loss:
                self.best_loss = loss
                path_loss = os.path.join(self.run_dir, self.config.filename_best_loss)
                torch.save(state_dict, path_loss)

            if improved_f1:
                self.best_f1 = f1
                path_f1 = os.path.join(self.run_dir, self.config.filename_best_f1)
                torch.save(state_dict, path_f1)

        return control


class BeautifulLoggingCallback(TrainerCallback):
    """Trainer callback to pretty-print metrics and evaluations in a human-friendly format."""

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return control

        if logs:
            if "loss" in logs:
                epoch = logs.get("epoch", 0.0)
                step = state.global_step
                loss = logs["loss"]
                lr = logs.get("learning_rate", 0.0)
                print(f"📈 [Epoch {epoch:.2f} | Step {step}] Loss: {loss:.4f} | LR: {lr:.2e}")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return control

        if metrics:
            print("\n📊 EVALUATION RESULTS:")
            print("─" * 50)
            for k, v in sorted(metrics.items()):
                if k.startswith("eval_"):
                    name = k[5:].replace("_", " ").title()
                    # Check if v is float and check for NaN (v != v)
                    if isinstance(v, float):
                        if v != v:
                            print(f"  🔹 {name:<25}: NaN")
                        else:
                            print(f"  🔹 {name:<25}: {v:.4f}")
                    else:
                        print(f"  🔹 {name:<25}: {v}")
            print("─" * 50 + "\n")
        return control
