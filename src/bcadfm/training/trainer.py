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
        self._train_labels_cached = None

        # Determine class weights from training dataset if requested
        self.class_weights: Optional[torch.Tensor] = None
        if self.train_dataset is not None:
            self._prepare_imbalance_handling()

    def _prepare_imbalance_handling(self) -> None:
        # Validate that oversampling and class weighting are not applied concurrently to prevent double-correction
        oversampling_method = self.imbalance_config.get("oversampling_method", "none")
        class_weights_method = self.imbalance_config.get("class_weights", "none")
        focal_alpha = self.imbalance_config.get("focal_alpha", None)

        if oversampling_method != "none":
            if class_weights_method != "none" or focal_alpha is not None:
                raise ValueError(
                    f"Invalid imbalance configuration: both oversampling_method='{oversampling_method}' and "
                    f"class_weights='{class_weights_method}'/focal_alpha are enabled. "
                    "Apply only one strategy (data-level or loss-level) to avoid double-correction."
                )

        # Get training labels
        labels = self._get_train_labels()
        if not labels:
            return

        # Check for DDP incompatibility with WeightedRandomSampler (C3 Fix)
        if oversampling_method == "weighted_sampler" and self.args.world_size > 1:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    "⚠️ WARNING: oversampling_method='weighted_sampler' is incompatible with DDP. "
                    "Automatically falling back to 'data_level' oversampling by modifying the training dataset in-place."
                )
            if hasattr(self.train_dataset, "oversample_dataset"):
                self.train_dataset.oversample_dataset()
            self.imbalance_config["oversampling_method"] = "data_level"

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
        if self._train_labels_cached is not None:
            return self._train_labels_cached

        if self.train_dataset is None:
            return []

        # Traverse wrapper datasets (like Subset) to find base dataset (H6 Fix)
        curr_ds = self.train_dataset
        indices = None
        while hasattr(curr_ds, "dataset"):
            if hasattr(curr_ds, "indices"):
                indices = curr_ds.indices
            curr_ds = curr_ds.dataset

        labels = []
        if hasattr(curr_ds, "samples"):
            if indices is not None:
                labels = [curr_ds.samples[i].label for i in indices]
            else:
                labels = [s.label for s in curr_ds.samples]
        elif hasattr(curr_ds, "labels"):
            raw_labels = curr_ds.labels
            if indices is not None:
                labels = [raw_labels[i] for i in indices]
            else:
                labels = list(raw_labels)
        elif hasattr(curr_ds, "targets"):
            raw_targets = curr_ds.targets
            if indices is not None:
                labels = [raw_targets[i] for i in indices]
            else:
                labels = list(raw_targets)
        else:
            # Fallback if dataset is wrapped or custom (runs once and caches)
            try:
                for i in range(len(self.train_dataset)):
                    item = self.train_dataset[i]
                    if "labels" in item:
                        if isinstance(item["labels"], torch.Tensor):
                            labels.append(item["labels"].item())
                        else:
                            labels.append(int(item["labels"]))
            except Exception:
                labels = []

        self._train_labels_cached = labels
        return labels

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

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, Any]] = None) -> None:
        try:
            super()._save(output_dir, state_dict)
        except Exception as e:
            # Fall back to standard PyTorch serialization if safetensors fails
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            if state_dict is None:
                state_dict = self.model.state_dict()
            
            # Save standard pytorch weights
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            # Save training arguments
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            
            # Print fallback message
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"⚠️ [WARN] safetensors save failed ({e}). Successfully saved checkpoint using PyTorch fallback (pytorch_model.bin).")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using custom loss function and imbalance strategies."""
        # Ensure loss_fn is initialized
        if not hasattr(self, "loss_fn"):
            self._prepare_imbalance_handling()
            if not hasattr(self, "loss_fn"):
                self.loss_fn = nn.CrossEntropyLoss()

        # Copy inputs and pop labels to prevent model from computing redundant unweighted loss (C6 Fix)
        inputs_copy = inputs.copy()
        labels = inputs_copy.pop("labels", None)
        outputs = model(**inputs_copy)

        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            # Fallback
            if return_outputs:
                return outputs.get("loss"), outputs
            return outputs.get("loss")

        # Move loss function weights/buffers to correct device (C1 & C2 helper)
        if hasattr(self.loss_fn, "weight") and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(logits.device)
        if hasattr(self.loss_fn, "alpha") and self.loss_fn.alpha is not None:
            self.loss_fn.alpha = self.loss_fn.alpha.to(logits.device)

        loss = self.loss_fn(logits, labels)

        if isinstance(outputs, dict):
            outputs["loss"] = loss

        return (loss, outputs) if return_outputs else loss
