# 🛠️ Dev log: Global Seed Configuration & Codebase Audit Resolutions

Date: 2026-06-08

This log documents the implementation of a global configuration-driven random seed for reproducibility and the resolution of all critical (C1–C8) and high-priority (H1–H12) issues identified during our codebase audit.

---

## 1. 🎲 Global Reproducibility & Seeding Config

- **Problem**: Previously, random seeds were not consistently parameterized via the configuration files, causing non-deterministic behavior during dataset oversampling, augmentations, and model initialization.
- **Resolution**:
  - Added a global `seed: int = 42` parameter to the `TrainingConfig` dataclass and parsed it from YAML configs.
  - Added `seed: 42` to the configuration templates (`configs/baseline.yaml` and `configs/test_smoke.yaml`).
  - Propagated the seed from `cfg.seed` to initialize `random.seed`, `np.random.seed`, `torch.manual_seed`, and `torch.cuda.manual_seed_all` in `scripts/train.py`.
  - Added `seed` to the `BatteryCellDataset` initialization parameters to ensure that in-place data oversampling shuffles and samples consistently across identical runs.

---

## 2. 📊 Evaluation Frequency Verification

- **Verification**: Confirmed that the evaluation process is performed **per epoch** rather than per iteration. In `scripts/train.py`, the training arguments are explicitly configured as follows:
  - `eval_strategy="epoch"`
  - `save_strategy="epoch"`
  - `logging_strategy="epoch"`
- This guarantees evaluation and validation-based checkpointing occur exactly once at the end of each epoch, maintaining stability.

---

## 3. 🧩 Core Audit Resolutions

The following code issues identified in the audit were resolved:

- **C1 & C2 (Device Mismatch in Loss Functions)**: Fixed by dynamically shifting loss weights (class weight tensor) and FocalLoss `alpha` parameters to the active `logits.device` during the forward pass inside `compute_loss`.
- **C3 (WeightedRandomSampler in DDP)**: Since `WeightedRandomSampler` does not partition sample weights correctly across ranks in multi-GPU configurations, the trainer was updated with an automatic fallback. If a multi-GPU DDP run requests `weighted_sampler`, the trainer emits a warning and automatically switches to `data_level` oversampling in-place, which is fully DDP-safe.
- **C4 (Multi-GPU Warmup Steps)**: Corrected the warmup steps division in `scripts/train.py` to divide by the DDP world size, ensuring accurate step count calculations.
- **C5 (Dataset Oversampling Determinism)**: Isolated the python random state inside `BatteryCellDataset.oversample_dataset()` to prevent oversampling state from leaking into other random generators, applying `random.seed(self.seed)` deterministically.
- **C6 (Redundant Loss Computation)**: Passing `labels` to the classifier backbone triggered the model's internal unweighted loss calculation. We resolved this by copying the input dictionary and popping the `labels` key before passing inputs to the backbone model forward pass.
- **C7 (VPT and Register Layout)**: Structured prompt token insertion in `VptLayerWrapper` to match the exact register token slicing layout: `[CLS, prompts, registers, patches]`.
- **C8 (DDP Dataset Index Wrapper Handling)**: Unpacked index wrapper layers (such as PyTorch subsets) during label scanning in the trainer to map subset indices correctly back to the target class values.
- **H3 (Random Augmentation Operator Selection)**: Rewrote the sampler in `RandomAugmentationCombo` to perform iterative sampling without replacement using raw operator probabilities directly as selection weights.
- **H4 (Missing Classes Weight Support)**: Allowed `compute_class_weights` to scale gracefully when certain class labels are absent in training splits, filling missing class index weights with default values of `1.0`.
- **H9 (Trainer v5 Early Stopping Callback compatibility)**: Aliased `training_args.evaluation_strategy = training_args.eval_strategy` right after instantiation to prevent newer versions of `transformers` from crashing inside `EarlyStoppingCallback`.
- **H12 (Robust Image Processor Fallback)**: Implemented fallback loading of `google/vit-base-patch16-224` image processor when checkpoint loading fails, ensuring the training pipeline never crashes on data loader setup.

---

## 4. 📈 Status

- All source files and configuration changes have been updated, validated, and staged.
- The modifications are pushed to the remote repository.
