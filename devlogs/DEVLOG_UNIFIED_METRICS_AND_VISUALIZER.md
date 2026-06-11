# 🛠️ Dev log: Unified Metric Tracking & Interactive Visualizer Dashboard

Date: 2026-06-09

This devlog documents the integration of unified metric tracking across classification and object detection tasks, the consolidation of training directories to eliminate redundant folders, the historical data migration for 58 completed runs, and the multi-task enhancements made to the interactive Streamlit visualizer dashboard.

---

## 1. 📂 Consolidated Output Directories

- **Problem**: YOLO training outputs were previously saved in two different locations: the configurations/logs were written to `outputs/det/`, while standard Ultralytics outputs (plots, results.csv, and model weights) were saved in `runs/detect/`. This directory mismatch cluttered the workspace and made loading results difficult.
- **Resolution**:
  - Configured Ultralytics global settings dynamically in `src/bcadfm/training/yolo_trainer.py` to use `outputs/det` as the default `runs_dir`.
  - Updated `scripts/train_detection.py` to resolve the `project` override using its absolute path (`run_dir.parent.resolve()`). This prevents Ultralytics from prepending the default `runs_dir/detect` when resolving relative paths, forcing YOLO to save all weights, plots, and logs directly under `outputs/det/{model_name}__{config_name}/{timestamp}/`.

---

## 2. 🧮 Unified Metric Backend & Callback

- **Problem**: Detection models only output bounding boxes and standard detection metrics (precision, recall, mAP50, mAP50-95), preventing direct comparison against image-level classification models on the core anomaly detection objective. Additionally, YOLO training saved results in CSV format, whereas classification runs saved Hugging Face-compliant JSON states.
- **Resolution**:
  - **Image-Level Conversion**: Formulated a box-to-image conversion layer during validation. Bounding box predictions are evaluated at a decision threshold (0.25 confidence) to yield image-level abnormal and text predictions, enabling calculation of Accuracy, Precision, Recall, F1, and AUROC side-by-side with classification runs.
  - **Unified State Callback**: Implemented a custom callback `save_yolo_trainer_state_callback` registered to `on_fit_epoch_end` in `yolo_trainer.py`. The callback compiles learning rates, training losses, validation losses, standard YOLO metrics, and custom metrics (mapped under `eval_custom_...` keys) into a unified `trainer_state.json` file.
  - **DDP-Safe Validation**: Implemented `ddp_gather_list` using `torch.distributed.all_gather_object` to ensure validation collectors (lists containing prediction tuples, labels, and match overlaps) are globally synchronized across GPU ranks prior to metric computation.
  - **Collector Reset**: Updated `init_metrics()` to reset validation statistics lists at the start of each validation epoch to prevent historical metric accumulation.

---

## 3. 🖥️ Enhanced Streamlit Results Dashboard

- **Problem**: The dashboard only supported loading Hugging Face classification checkpoints and could not render detection runs, custom matched box overlaps, or comparative metrics.
- **Resolution**:
  - Overhauled `load_results()` in `visualize.py` to recursively load results uniformly from `trainer_state.json` files for both classification and detection runs.
  - Added columns to the Leaderboard showing converted image-level F1 and AUROC side-by-side.
  - Updated Trajectory Curves to plot YOLO losses, mAP50, mAP50-95, mean IoU/Dice, and image-level classification trajectories.
  - Added a detailed Single Run Inspector for detection runs, displaying standard mAP, custom box matching stats (mean IoU/Dice), per-class box metrics, and converted image-level metrics.
  - Developed a Model Comparison Tab that extracts the best classification model and best converted detection model, rendering their metrics and abnormal confusion matrices side-by-side.
  - Resolved a PyArrow serialization `ArrowTypeError` by casting the `epochs_configured` value to a string in the details table.

---

## 4. 🔄 Historical Runs Migration

- **Mechanism**:
  - Since the 58 completed detection runs were executed before the output consolidation fix, their results were stored in `runs/detect`.
  - Wrote and executed a script `scripts/migrate_yolo_runs.py` to locate all completed runs under `outputs/det/`, map them to the old results in `runs/detect/`, convert their `results.csv` files into Hugging Face-compliant `trainer_state.json` files, copy all weights and visual plots, and cleanly delete the legacy `runs` folder.
  - The Streamlit app successfully parsed all 58 migrated runs without error.

---

## 5. 📈 Status

- Both backend modifications and dashboard enhancements have been verified.
- The Streamlit server hot-reloads dynamically, and the unified metrics dashboard is fully operational on port `8501`.
