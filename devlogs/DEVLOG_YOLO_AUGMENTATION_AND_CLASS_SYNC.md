# 🛠️ Dev log: YOLO Data Augmentations & Class Names Synchronization

Date: 2026-06-09

This log documents the implementation of config-driven YOLO data augmentations mapping and class names synchronization between the classification config and object detection pipeline.

---

## 1. 🎨 YAML-driven YOLO Augmentation Mapping
- **Problem**: In the detection pipeline, data augmentations defined under the `data:` block of the configuration YAML files were ignored by YOLO training (acting as schema placeholders).
- **Solution**: Updated [config.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/data/config.py) and [train_detection.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/train_detection.py) to connect configuration variables to YOLO overrides:
  - **Disabled Augmentations**: If `augmentations_enabled: false` is configured, we explicitly zero out all YOLO internal augmentations (`mosaic=0.0`, `fliplr=0.0`, `degrees=0.0`, etc.) to guarantee clean, unaugmented training baselines.
  - **Direct Overrides**: Added `yolo_augmentations` optional dictionary to `DataConfig`. If defined in the YAML, these keys are passed directly to YOLO overrides (e.g., configuring `mosaic`, `mixup`, `degrees`, etc.).
  - **Fallback Mapping**: If `yolo_augmentations` is omitted but augmentations are enabled, standard classification parameters (`horizontal_flip_prob`, `rotation_degrees`, `color_jitter_brightness`/`contrast`/`saturation`/`hue`, and `random_resized_crop_scale`) are mapped to YOLO equivalents (`fliplr`, `degrees`, `hsv_h`/`hsv_s`/`hsv_v`, and `scale`).

---

## 2. ⚖️ Class Names Synchronization
- **Problem**: Mismatches between classification folder names (`normal`, `abnormal`) and detection class names (`abnormality`, `cell`, `text`) caused metric matching failures in visualizer displays and logs (which previously hardcoded `abnormality` as the target class).
- **Solution**:
  - Pass the configured `normal_class_name` and `abnormal_class_name` from the config YAML down to `CustomDetectionTrainer` and `CustomDetectionValidator` in [yolo_trainer.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/training/yolo_trainer.py).
  - In `CustomDetectionValidator`, dynamically resolve the index of the abnormality class based on config names.
  - In `get_stats()`, duplicate metrics logged under `/abnormality` to also be logged under `/{abnormal_class_name}`. This maintains full backwards compatibility with visualizer features expecting `/abnormality` while supporting config-driven name alignment.
  - Updated [visualize.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/visualize.py) to search for metric keys dynamically under both names.

---

## 3. 📈 Status
- All features have been successfully implemented, verified, and unit-tested.
- Reinforcing the test suite, we added:
  - **DDP Mock Oversampling Test** in [test_dataset.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/tests/test_dataset.py) to assert dataset alignment and ordering across mock multi-GPU ranks.
  - **VPT Deep Layer Prompt Wrapper Test** in [test_models.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/tests/test_models.py) to assert correct prompt token replacement logic in `VptLayerWrapper`.
- The expanded unittest suite compiles and runs successfully: **32 tests passed (100% success rate)**.
