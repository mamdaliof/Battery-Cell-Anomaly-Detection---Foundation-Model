# Dev log: Ablation Study Framework & Performance Optimization

Date: 2026-06-06

This log documents the implementation of the full ablation study framework, parallel training infrastructure, and training pipeline performance optimizations.

---

## 1. Ablation Grid Generation

- Created `scripts/generate_ablation_grid.py` to produce a combinatorial grid of 58 YAML configurations.
- Grid covers:
  - 2 backbones: DINOv3 ViT-S/16, DINOv3 ViT-B/16.
  - 5 PEFT methods: frozen baseline, LoRA, Bottleneck Adapters, VPT Shallow, VPT Deep.
  - Multiple hyperparameter settings per method (rank, bottleneck dimension, token count).
  - Layer targeting strategies: all layers, last-4, last-2.
  - 2 learning rates: 3e-4, 5e-4.
- All configs share base settings: 300 epochs, batch size 64, cosine LR schedule, focal loss with class weights, dataset-level oversampling.
- Configs written to `configs/ablations/` with sequential naming: `01_baseline_vits16.yaml` through `58_...`.

---

## 2. Configuration Validation

- Built `scripts/validate_ablation_configs.py` to dry-run all configs.
- Validation loads model, applies PEFT wrappers, counts parameters — no GPU or data needed.
- Result: all 67 configs (58 ablation + 9 template configs) passed.
- Key output per config: PEFT type, model name, total params, trainable params, trainable %.

---

## 3. Parallel Training Infrastructure

- Built `scripts/run_parallel_ablations.py` to distribute training across 8 GPUs.
- Architecture:
  - Job queue of pending configs, pre-sorted by name.
  - GPU slot manager assigns one config per GPU.
  - Subprocess launched with `CUDA_VISIBLE_DEVICES={gpu_id}` isolation.
  - Reader threads parse stdout with regex to extract: epoch, step, total steps, loss, F1, status.
  - Thread-safe `_slots` dictionary guarded by `threading.Lock`.
- Terminal dashboard: fixed 8-line in-place display using ANSI escape codes.
- Full logs saved to `outputs/logs/<config_name>.log`.
- Resume support: skips configs with existing output directories.

### 3.1. Dashboard Evolution

- Initial version: printed every subprocess line with `[GPU_ID|config]` prefix — too noisy.
- Final version: in-place ANSI dashboard with one line per GPU, updating values via regex-parsed tqdm output.
- Shows: GPU ID, config name, epoch, progress bar, speed (it/s), loss, F1, status emoji (⏳🚀✅❌).

---

## 4. Training Pipeline Performance Optimizations

### 4.1. Data Loading

- Added `dataloader_num_workers=4` to parallelize image loading across CPU workers.
- Added `pin_memory=True` for faster CPU→GPU tensor transfers via DMA.
- Set in `scripts/train.py` via `TrainingArguments`.

### 4.2. Augmentation Pipeline (dataset.py)

- **Problem**: Transform objects were being re-instantiated on every `__getitem__` call.
- **Fix**: Pre-build all transform objects once in `build_augmentation_pipeline()` and reuse.
- **Problem**: Gaussian noise augmentation used PIL↔Tensor round-trip conversions.
- **Fix**: Reimplemented using `np.random.normal` on float arrays directly, avoiding expensive format conversions.

### 4.3. Checkpoint Saving (cls_callbacks.py)

- **Problem**: `model.state_dict()` was called on every evaluation step, copying full model weights to RAM.
- **Fix**: Deferred `state_dict()` to only execute when a new best metric is achieved.
- Impact: eliminates redundant multi-MB RAM copies on non-improving evaluation epochs.

### 4.4. Training Arguments Cleanup

- Resolved `warmup_ratio` deprecation by switching to `warmup_steps`.
- Set `eval_strategy='epoch'` and `save_strategy='epoch'` explicitly.
- Removed redundant `model.to(device)` calls that conflicted with Trainer's internal device placement.

---

## 5. DINOv3 Architecture Discovery

- Discovered DINOv3 uses `q_proj`, `v_proj`, `k_proj`, `o_proj` naming (not `query`/`value`/`key`).
- Layer hierarchy: `model.layer.{i}.attention.{proj_name}`.
- Updated LoRA target modules from `['query', 'value']` to `['q_proj', 'v_proj']`.
- Dynamic layer routing in `dinov3_classifier.py` now handles: `encoder.layer`, `model.layer`, `layer`, `layers`.

---

## 6. Status

- All 58 ablation configs generated and validated.
- Parallel runner ready for launch on 8-GPU server.
- Command:

  ```bash
  python3 scripts/run_parallel_ablations.py
  ```

- Expected runtime: ~24–48 hours for the full 58-config sweep (300 epochs each).
