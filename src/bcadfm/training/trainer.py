from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, List
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer

from bcadfm.training.losses import FocalLoss, compute_class_weights


class ImbalanceTrainer(Trainer):
    """Custom Trainer that implements class imbalance handling strategies.

    This includes loss-level adjustments (class-weighted cross-entropy, Focal Loss)
    and data-level adjustments (WeightedRandomSampler).
    """

    def __init__(
        self,
        *args,
        imbalance_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.imbalance_config = imbalance_config or {}
        
        # Determine class weights from training dataset if requested
        self.class_weights: Optional[torch.Tensor] = None
        if self.train_dataset is not None:
            self._prepare_imbalance_handling()

    def _prepare_imbalance_handling(self) -> None:
        # Get training labels
        labels = self._get_train_labels()
        if not labels:
            return

        # Count occurrences of each class
        class_counts = {}
        for l in labels:
            class_counts[l] = class_counts.get(l, 0) + 1

        # Check configuration for class weights
        class_weights_method = self.imbalance_config.get("class_weights", "none")
        if class_weights_method != "none" and len(class_counts) > 1:
            self.class_weights = compute_class_weights(class_counts, method=class_weights_method)
            
            # Print class weights on the main process
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"📊 Computed class weights ({class_weights_method}): {self.class_weights.tolist()} for counts {class_counts}")

        # Initialize the loss function
        self.loss_fn = self._init_loss_fn()

    def _get_train_labels(self) -> List[int]:
        if self.train_dataset is None:
            return []
        if hasattr(self.train_dataset, "samples"):
            return [s.label for s in self.train_dataset.samples]
        
        # Fallback if dataset is wrapped or custom
        try:
            labels = []
            for i in range(len(self.train_dataset)):
                item = self.train_dataset[i]
                if "labels" in item:
                    labels.append(item["labels"].item())
            return labels
        except Exception:
            return []

    def _init_loss_fn(self) -> nn.Module:
        loss_type = self.imbalance_config.get("loss_type", "cross_entropy")
        
        if loss_type == "focal":
            gamma = self.imbalance_config.get("focal_gamma", 2.0)
            alpha_val = self.imbalance_config.get("focal_alpha", None)
            
            alpha_tensor = None
            if alpha_val is not None:
                if isinstance(alpha_val, (int, float)):
                    # Binary classification: [1 - alpha, alpha]
                    alpha_tensor = torch.tensor([1.0 - alpha_val, alpha_val], dtype=torch.float32)
                else:
                    alpha_tensor = torch.tensor(alpha_val, dtype=torch.float32)
            else:
                # Fallback to calculated class weights
                alpha_tensor = self.class_weights

            return FocalLoss(gamma=gamma, alpha=alpha_tensor)
            
        elif loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _get_train_sampler(self, *args, **kwargs) -> torch.utils.data.Sampler | None:
        dataset = args[0] if args else kwargs.get("dataset", self.train_dataset)
        if dataset is None:
            return None

        oversampling_method = self.imbalance_config.get("oversampling_method", "none")
        if oversampling_method == "weighted_sampler":
            labels = self._get_train_labels()
            if not labels:
                return super()._get_train_sampler(*args, **kwargs)

            class_counts = {}
            for l in labels:
                class_counts[l] = class_counts.get(l, 0) + 1

            if len(class_counts) <= 1:
                return super()._get_train_sampler(*args, **kwargs)

            # Compute balanced weights specifically for sampler
            weights_tensor = compute_class_weights(class_counts, method="balanced")
            sample_weights = [weights_tensor[l].item() for l in labels]

            generator = torch.Generator()
            if hasattr(self.args, "seed"):
                generator.manual_seed(self.args.seed)

            # Warning under DDP
            if self.args.world_size > 1:
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(
                        "⚠️ WARNING: WeightedRandomSampler is used under DDP training. "
                        "This will not partition samples correctly across GPUs. "
                        "Consider using oversampling_method='data_level' instead."
                    )

            return WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=generator,
            )

        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using custom loss function and imbalance strategies."""
        # Ensure loss_fn is initialized
        if not hasattr(self, "loss_fn"):
            self._prepare_imbalance_handling()
            if not hasattr(self, "loss_fn"):
                self.loss_fn = nn.CrossEntropyLoss()

        labels = inputs.get("labels")
        outputs = model(**inputs)

        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            # Fallback
            if return_outputs:
                return outputs.get("loss"), outputs
            return outputs.get("loss")

        loss = self.loss_fn(logits, labels)

        if isinstance(outputs, dict):
            outputs["loss"] = loss

        return (loss, outputs) if return_outputs else loss
