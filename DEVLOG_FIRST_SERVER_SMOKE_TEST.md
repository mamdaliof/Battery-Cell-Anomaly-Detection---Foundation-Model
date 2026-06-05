# Dev log: first server smoke test

Date: 2026-06-05

This log documents the sequence of changes and debugging steps performed to get the first end-to-end training smoke test running on the remote server.

## 1. Initial training script and YAML config

- Added a config-driven training entry point in `scripts/train.py` that:
  - Loads a `TrainingConfig` from a YAML file via `load_yaml_config`.
  - Builds `BatteryCellDataset` instances for `train` and `val` splits.
  - Instantiates `DinoV3Classifier` with a configurable head.
  - Wraps everything in Hugging Face `Trainer` + `TrainingArguments`.
- Introduced:
  - `src/bcadfm/utils/config.py` with `TrainingConfig`, `SchedulerConfig`, and `AmpConfig`.
  - `configs/baseline.yaml` as the main experiment config.
  - `configs/test_smoke.yaml` as a minimal one-epoch, small-batch smoke-test config.

## 2. Augmentation control

- Extended `DataConfig` to support detailed augmentation control:
  - Global switch: `augmentations_enabled`.
  - Global probability: `aug_global_prob` (probability to apply any augmentation).
  - Max transforms per image: `aug_max_transforms`.
  - Per-transform probabilities and parameters for:
    - Random resized crop.
    - Horizontal flip.
    - Rotation.
    - Color jitter.
    - Gaussian noise.
- Reworked `build_augmentation_pipeline` to build a `RandomAugmentationCombo` transform that:
  - Samples whether to augment at all (using `aug_global_prob`).
  - Samples up to `aug_max_transforms` distinct transforms, weighted by per-transform probabilities.
  - Applies the selected transforms sequentially.

## 3. Metrics and best-model saving

- Added a classification metrics module in `src/bcadfm/metrics/cls_metrics.py`:
  - `compute_cls_metrics(eval_pred)` returns accuracy, precision, recall, F1, AUROC, and confusion matrix counts (TN, FP, FN, TP).
- Wired `compute_cls_metrics` into `Trainer` via `compute_metrics=compute_cls_metrics`.
- Added a callback in `src/bcadfm/metrics/cls_callbacks.py`:
  - `SaveTwoBestClsModelsCallback` monitors `eval_loss` and `eval_f1` and saves:
    - `best_loss.pt` when `eval_loss` reaches a new minimum.
    - `best_f1.pt` when `eval_f1` reaches a new maximum.
- Registered `SaveTwoBestClsModelsCallback` in `scripts/train.py` so every run produces these two snapshots in the run directory.

## 4. Run directory naming and config copy

- Updated run directory naming to avoid local folder names colliding with Hugging Face repo IDs:
  - Run directory pattern is now:
    - `outputs/{task_name}__{safe_model_name}/{timestamp}/`,
    - where `task_name = "cls"` and `safe_model_name` is the model name with `/` replaced by `-`.
- Each run now:
  - Creates the run directory.
  - Copies the used YAML config into `config.yaml` inside that directory for reproducibility.

## 5. Server-side issues and fixes

### 5.1. PYTHONPATH and package import

- Server initially failed with `ModuleNotFoundError: No module named 'bcadfm'`.
- Fixed at runtime by setting:
  - `export PYTHONPATH=$(pwd)/src:$PYTHONPATH` from the repo root.

### 5.2. Torchvision dependency

- `AutoImageProcessor.from_pretrained` raised an error about missing `torchvision`.
- Resolved by installing `torchvision` in the `pytorch_env` environment.

### 5.3. Hugging Face authentication and gated DINOv3

- Attempting to load `facebook/dinov3-vitb16-pretrain-lvd1689m` produced a `GatedRepoError` / 401.
- Logged in using `hf auth login` and a fine-grained read token.
- After authentication, the processor still failed to load for this specific checkpoint in this environment.
- Decision: to unblock the pipeline, temporarily switch to a public model for smoke tests.

### 5.4. Switch to a public ViT backbone for smoke tests

- Updated `configs/test_smoke.yaml` to use:
  - `model_name: "google/vit-base-patch16-224"`.
- Confirmed that `AutoImageProcessor` and `ViTModel` load correctly with this backbone.

### 5.5. Transformers 5.x API changes (TrainingArguments)

- The environment uses `transformers==5.10.2`, which differs from 4.x APIs.
- Initial usage of `evaluation_strategy`, `save_strategy`, and `logging_strategy` caused:
  - `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`.
- Iteratively adapted `TrainingArguments` to this environment:
  - Removed explicit `evaluation_strategy`, `save_strategy`, and `logging_strategy`.
  - Replaced `warmup_ratio` with `warmup_steps=0`.
  - Temporarily disabled `load_best_model_at_end`.
- EarlyStopping integration attempts led to additional assertions from `EarlyStoppingCallback` about:
  - `metric_for_best_model` needing to be defined.
  - `eval_strategy` not being `NO` and matching `save_strategy`.
- To avoid fighting these API changes in this environment, the final decision for now:
  - **Disable `EarlyStoppingCallback` entirely** in `scripts/train.py`.
  - Rely on:
    - `num_epochs` from YAML to control run length.
    - `SaveTwoBestClsModelsCallback` to save `best_loss.pt` and `best_f1.pt` during evaluation.

### 5.6. Data layout and empty dataset

- Initial server runs failed with `ValueError: num_samples should be a positive integer value, but got num_samples=0`.
- Cause: `data/train` and `data/val` did not yet exist; only `data/split_base` was present.
- Fix: run the conversion script to create the classification layout:

  ```bash
  python scripts/convert_split_base_to_classification.py \  
    --source-root data/split_base \  
    --target-root data \  
    --abnormal-labels burnt crack
  ```

- After conversion, `BatteryCellDataset` sees non-zero samples for both `train` and `val`.

## 6. First successful runs

### 6.1. Single-process CPU smoke test

- Command:

  ```bash
  export PYTHONPATH=$(pwd)/src:$PYTHONPATH
  CUDA_VISIBLE_DEVICES= \
  python scripts/train.py --config configs/test_smoke.yaml
  ```

- Result:
  - Model and processor loaded for `google/vit-base-patch16-224`.
  - One epoch of training completed on CPU.
  - Metrics computed via `compute_cls_metrics`.
  - Run directory created under `outputs/cls__google-vit-base-patch16-224/<timestamp>/` with:
    - `config.yaml`.
    - `best_loss.pt` and `best_f1.pt` from `SaveTwoBestClsModelsCallback`.
    - At least one `checkpoint-*` folder.

### 6.2. Two-process (intended 2-GPU) run via torchrun

- Command:

  ```bash
  export PYTHONPATH=$(pwd)/src:$PYTHONPATH
  torchrun --nproc_per_node=2 scripts/train.py --config configs/test_smoke.yaml
  ```

- Observations:
  - Two processes launched; both loaded the ViT backbone and ran 32 steps (dataset + config work correctly in distributed launch).
  - Because the NVIDIA driver is too old, PyTorch logs a CUDA warning and effectively runs on CPU only (no real GPU acceleration yet).
  - Training reports per-process stats (runtime, samples/sec, loss) for epoch 1.

## 7. Current status and next steps

- The codebase now supports:
  - YAML-driven configuration for model, data, head, scheduler, and AMP.
  - Controlled augmentations with global and per-transform probabilities.
  - Classification metrics and best-model snapshots (`best_loss.pt`, `best_f1.pt`).
  - Clean run directories with config copies and task/model-prefixed naming.
  - Single-process and torchrun-based multi-process training in the current server environment (CPU-bound due to GPU driver).
- For future work on this server:
  - Re-enable early stopping once the exact `TrainingArguments` API in use is stable and compatible.
  - Revisit DINOv3 backbones once the environment and `transformers` versions match the DINOv3 image-processor expectations.
  - Upgrade GPU drivers to enable real multi-GPU, mixed-precision training.
