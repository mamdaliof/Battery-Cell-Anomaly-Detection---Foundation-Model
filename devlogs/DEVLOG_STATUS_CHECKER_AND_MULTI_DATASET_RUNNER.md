# Dev log: Status Checker & Multi-Dataset Parallel Ablation Runner

Date: 2026-06-10

This log documents the implementation of the updated training status checker, multi-dataset detection ablation grid runners, restoration of custom architecture configs, and cleanup of legacy runner files.

---

## 1. Multi-Dataset Detection Ablation Study Support

### 1.1. Dataset Split Strategies
Object detection runs across three specific dataset splits in `data/det_v1.0/`:
* **All Labels**: `data/battery_detection_all.yaml` (classes: abnormal, cell, text)
* **No Cell**: `data/battery_detection_no_cell.yaml` (classes: abnormal, text)
* **Abnormal Only**: `data/battery_detection_abnormal_only.yaml` (classes: abnormal)

### 1.2. Architecture Config Restoration
* Restored the custom architecture file [yolo26_dino.yaml](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/configs/det/yolo26_dino.yaml) which was deleted in previous cleanup commits. This layout maps the self-supervised self-attention layers of DINOv3 with transpose/pooling blocks (SFP neck) to standard YOLO26 detection heads.

### 1.3. Grid Generation overhaul (`generate_det_ablation_grid.py`)
* Modified the script to load [peft_smoke_all_label.yaml](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/configs/det/peft_smoke_all_label.yaml) as the template config (since baseline configs were deleted).
* Added argparse option `--strategy` (`all_label`, `no_cell`, or `abnormal_only`) to write configs to separate configuration folders (`ablations_all_label`, `ablations_no_cell`, and `ablations_abnormal_only`) and configure their datasets (`yolo_data_yaml`) and output directories (`output_dir`) dynamically.
* Removed legacy shell runner (`.sh`) generation.

---

## 2. Multi-Dataset Parallel Detection Runner (`run_parallel_det_ablations.py`)

* Updated `scripts/run_parallel_det_ablations.py` to accept a custom `--config_dir` argument so that the user can train any of the three datasets in parallel:
  * `python scripts/run_parallel_det_ablations.py --config_dir configs/det/ablations_all_label`
  * `python scripts/run_parallel_det_ablations.py --config_dir configs/det/ablations_no_cell`
  * `python scripts/run_parallel_det_ablations.py --config_dir configs/det/ablations_abnormal_only`
* Fixed completion validation: updated comparison logic inside `_equiv` to check for matching `yolo_data_yaml` and `yolo_model_config` strings, preventing runs from different datasets from colliding or skipping each other during scan checks.

---

## 3. Training Run Status Check Overhaul (`check_ablation_status.py`)

### 3.1. Arbitrary Path Depth Scan Fix
* **Problem**: The status check script only inspected directories at a fixed relative path depth of 3 (`outputs/{task}/{run}/{timestamp}`). By moving detection runs under strategy subfolders (e.g. `outputs/no_cell/det/...`), their depth increased to 4, making them invisible to the status checker.
* **Fix**: Rewrote scanning logic to match run folders by checking if the leaf directory is named as a timestamp (`YYYYMMDD_HHMMSS`) and dynamically parsing task names based on whether `"cls"` or `"det"` is present in the parent path.

### 3.2. Failed vs. Interrupted Run Partitioning
* Enhanced completion checks to verify:
  1. Presence of a `DONE` file.
  2. Non-empty weights (`verify_weights`).
  3. Valid `trainer_state.json` file.
* Runs that completed but are missing weights or have corrupt state files are now explicitly grouped and reported as **Failed / Incorrect Runs** rather than incomplete runs.

### 3.3. Completed Runs Argument (`--show-completed`)
* By default, the script suppresses the lengthy completed runs table, printing only the overview statistics and the list of incorrect or interrupted runs (as well as configs to run again).
* Added `--show-completed` argument to show details of successfully completed runs.

---

## 4. Helper Script Refactoring

* **[cleanup_unfinished_runs.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/cleanup_unfinished_runs.py)**: Updated directory depth scan logic to align with `check_ablation_status.py`.
* **[generate_ablation_grid.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/generate_ablation_grid.py)**: Switched to `peft_smoke_all_label.yaml` template config and updated target folder to `configs/cls/ablations_all_label/`.
* **[generate_yolo26_configs.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/generate_yolo26_configs.py)**: Switched template loader paths to point to `ablations_all_label/yolo11n_train.yaml`.
* **[generate_yolo_variant_configs.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/generate_yolo_variant_configs.py)**: Overhauled to loop over all three strategies (`all_label`, `no_cell`, `abnormal_only`) and generate the configs with scale-specific safe batch sizes to prevent out-of-memory issues.
* **[check_missing_ablations.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/check_missing_ablations.py) & [validate_ablation_configs.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/validate_ablation_configs.py)**: Updated search directories to target the renamed config folders.

---

## 5. Incomplete Training Run Troubleshooting & Fixes

### 5.1. CUDA Out of Memory (OOM) Resolution
* **Problem**: Training yolov8/11 medium (`m`), large (`l`), and largest (`x`) models with a batch size of 64 on 15GB VRAM GPUs led to CUDA OOM crashes.
* **Fix**: Programmed scale-specific batch size limits into the config generators:
  * Nano (`n`) / Small (`s`): batch size 64
  * Medium (`m`): batch size 32
  * Large (`l`): batch size 16
  * Largest (`x`): batch size 8
  * Custom YOLO26 variants: batch size 64 (nano/small/medium), 32 (large), 16 (largest).

### 5.2. Missing Model Architecture Configs
* **Problem**: Runs using custom YOLO26 variants failed with a `FileNotFoundError` as the `configs/det/yolo_variants/` directory was deleted.
* **Fix**: Regenerated all custom architecture files (`yolo26{scale}.yaml` and `yolo26{scale}_dino.yaml`) under `configs/det/yolo_variants/`.

### 5.3. Strategy Mapping Collision in Status Checker
* **Problem**: The status check script walk logic mapped all runs (e.g. `outputs/det_all_label/...`) to the first matching config filename (which belonged to `ablations_no_cell`).
* **Fix**: Updated `check_ablation_status.py` to parse the split strategy (e.g. `all_label`, `no_cell`, `abnormal_only`) from the directory path and target config lookups to the correct subfolder.

### 5.4. Log File Overwriting in Parallel Runner
* **Problem**: Runs across different strategies wrote logs to the same path under `outputs/det/logs/`, overwriting each other.
* **Fix**: Updated `run_parallel_det_ablations.py` to write logs to strategy-specific log directories (e.g. `outputs/{strategy}/det/logs/`).

### 5.5. Clean Up Obsolete Configs
* Removed legacy classification smoke configurations (`peft_smoke_abnormal_only.yaml`, etc.) pointing to non-existent dataset paths.
