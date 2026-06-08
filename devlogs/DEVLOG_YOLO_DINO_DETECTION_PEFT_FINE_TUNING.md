# Dev log: YOLO Detection Fine-Tuning & PEFT Integration

Date: 2026-06-08

This log documents the implementation, details, and resolutions during the integration of Parameter-Efficient Fine-Tuning (PEFT) on the DINOv3 SFP object detection model, simple train/test bench scripts, parallel ablation study setups, and test file relocations.

---

## 1. PEFT Wrapper for YOLO Backbone

- **Goal**: Support low-rank adaptation (LoRA), Pfeiffer bottleneck adapters, and Visual Prompt Tuning (VPT) Shallow/Deep modes directly on the frozen backbone of our custom registered YOLO26 model.
- **Key Changes**:
  - `src/bcadfm/models/yolo_dino.py` (`DinoV3Backbone`):
    - Added `peft_config` parameter to the constructor.
    - Wrapped the underlying pretrained model using PEFT configs (similar to classification).
    - Enabled gradient tracking dynamically during the forward pass: when PEFT is active and the model is in training mode, gradients are kept enabled to update the adapter parameters.
    - Fixed the **VPT Slicing Loophole** by adjusting `start_idx` dynamically: `start_idx = 1 + num_prompts + num_registers`. Previously, VPT prepended tokens were not accounted for, causing spatial feature misalignment.

---

## 2. Active PEFT Config Registry

- **Goal**: Pass the active PEFT config schema dynamically from the training script to the Ultralytics model parser parsing layers at runtime.
- **Key Changes**:
  - `src/bcadfm/utils/yolo_utils.py`:
    - Implemented a thread-safe global registry with `set_active_peft_config()` and `get_active_peft_config()`.
    - Patched the wrapper `custom_parse_model` to retrieve the active `peft_config` and pass it to `DinoV3Backbone` during layer parsing and instantiation.

---

## 3. Simple Train & Test Bench for Detection

- **Goal**: Implement a training script that loads unified configs (comparable to classification) and translates them to YOLO overrides.
- **Key Changes**:
  - `src/bcadfm/utils/config.py`: Extended `TrainingConfig` to support optional detection fields (`yolo_model_config` and `yolo_data_yaml`).
  - `scripts/train_detection.py`:
    - Loads the unified configuration.
    - Sets active PEFT config state.
    - Translates top-level hyperparameters (learning rate, batch size, epochs, cosine scheduler, etc.) to YOLO trainer overrides.
    - Instantiates and launches `CustomDetectionTrainer` which prints standard box metrics and custom validation report metrics.
  - Created baseline and PEFT smoke configs under `configs/det/`:
    - `configs/det/test_smoke.yaml`
    - `configs/det/peft_smoke.yaml`
    - `configs/det/benchmark_baseline.yaml`

---

## 4. Parallel Detection Ablation Studies

- **Goal**: Setup parallel sweeps for detection.
- **Key Changes**:
  - `scripts/generate_det_ablation_grid.py`: Generates 58 configuration YAML files under `configs/det/ablations/` and a shell sequence script `run_det_ablations.sh`.
  - `scripts/run_parallel_det_ablations.py`: Subprocess reader/writer parallel GPU manager customized for detection training logs, mapping validation `mAP50` and training box/class loss to the slot monitor.

---

## 5. Relocation of Test Files to `tests/`

- **Goal**: Move all verification and test codes under `tests/` to maintain clean codebase separation.
- **Moved Files**:
  - `scripts/test_yolo_registration.py` ➔ `tests/test_yolo_registration.py`
  - `scripts/verify_peft.py` ➔ `tests/verify_peft.py`
  - `scripts/gpu_alloc_test.py` ➔ `tests/gpu_alloc_test.py`
  - `scripts/ddp_alloc_test.py` ➔ `tests/ddp_alloc_test.py`
- Re-routed all reference paths in design documentation (`PROJECT_PLAN.md`, `README.md`, `PEFT_IMBALANCE_REPORT.md`, `devlogs/DEVLOG_PEFT_IMBALANCE_INTEGRATION.md`, and `docs/technical_details.md`) to the new paths under `tests/`.
