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
        
        # Profiling variables (C7 Fix)
        self._last_step_time = None
        self._last_forward_time = 0.0
        self._step_count = 0
        self._accumulated_data_time = 0.0
        self._accumulated_forward_time = 0.0
        self._accumulated_step_time = 0.0
        
        # Determine class weights from training dataset if requested
        self.class_weights: Optional[torch.Tensor] = None
        if self.train_dataset is not None:
            self._prepare_imbalance_handling()

    def _prepare_imbalance_handling(self) -> None:
        # Get training labels
        labels = self._get_train_labels()
        if not labels:
            return

        # Check for DDP incompatibility with WeightedRandomSampler (C3 Fix)
        oversampling_method = self.imbalance_config.get("oversampling_method", "none")
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

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        import time
        t_start = time.perf_counter()
        
        # Measure time elapsed since last training step ended (data loading and prep)
        if self._last_step_time is not None:
            data_time = t_start - self._last_step_time
        else:
            data_time = 0.0
            
        self._last_forward_time = 0.0  # Reset for this step
        
        # Execute forward pass, loss computation, and backpropagation via HF Trainer
        loss = super().training_step(model, inputs)
        
        step_time = time.perf_counter() - t_start
        backward_time = max(0.0, step_time - self._last_forward_time)
        
        self._step_count += 1
        self._accumulated_data_time += data_time
        self._accumulated_forward_time += self._last_forward_time
        self._accumulated_step_time += step_time
        
        # Log profiling results every 50 steps
        if self._step_count % 50 == 0:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                avg_data = self._accumulated_data_time / 50
                avg_forward = self._accumulated_forward_time / 50
                avg_step = self._accumulated_step_time / 50
                avg_backward = max(0.0, avg_step - avg_forward)
                print(
                    f"\n⏱️ [Trainer Profiler] Step {self._step_count} bottleneck diagnostics (avg of last 50 steps):\n"
                    f"  - Data Prep / Loader (CPU→GPU): {avg_data:.4f}s ({(avg_data/avg_step)*100:.1f}% of step)\n"
                    f"  - Model Forward + Loss:          {avg_forward:.4f}s ({(avg_forward/avg_step)*100:.1f}% of step)\n"
                    f"  - Model Backward + Opt:          {avg_backward:.4f}s ({(avg_backward/avg_step)*100:.1f}% of step)\n"
                    f"  - Total Step Time:               {avg_step:.4f}s\n"
                )
                # Reset accumulators
                self._accumulated_data_time = 0.0
                self._accumulated_forward_time = 0.0
                self._accumulated_step_time = 0.0
                
        self._last_step_time = time.perf_counter()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using custom loss function and imbalance strategies."""
        import time
        t_start = time.perf_counter()
        
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
            self._last_forward_time = time.perf_counter() - t_start
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

        self._last_forward_time = time.perf_counter() - t_start
        return (loss, outputs) if return_outputs else loss
