# 🔋 Battery Cell Anomaly Detection: PEFT & Detection Audit Report

This report evaluates the **Battery Cell Anomaly Detection** framework, analyzing both the classification and object detection pipelines, checking configuration integrations, verifying custom metrics, auditing visualizer diagnostics, and summarizing the unittest suite execution.

---

## 📋 Executive Summary
1. **Classification Pipeline**: Successfully uses pretrained frozen **DINOv3** backbones with dynamic routing and wraps them with LoRA, Bottleneck Adapters, and VPT (Shallow/Deep). Imbalance is handled at the loss level (Focal Loss, Class-Weighted CE) and data level (WeightedRandomSampler and DDP-compliant data oversampling).
2. **Detection Pipeline**: Merges DINOv3 backbones with a **ViTDet-style Simple Feature Pyramid (SFP)** neck, which projects stride-16 patch tokens into multi-scale feature maps (P3, P4, P5), feeding a standard Ultralytics **YOLO26** head.
3. **Hyperparameters**: Input sizes, learning rates, schedulers, and PEFT options are identified in YAML configuration files and successfully propagated to the scripts.
4. **Validation & Metrics**: Object detection is evaluated at the box level (greedy IoU matching, Dice) and converted to image-level anomaly classifications (Precision, Recall, F1, AUROC).
5. **Code Quality**: Audited codebases are clean, well-tested, and robust. All **30 unit tests** pass successfully.

---

## 🧠 Part 1: Classification Analysis & Hyperparameters

### 1. Key Hyperparameters
These are defined in config files (e.g., [configs/cls/baseline.yaml](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/configs/cls/baseline.yaml)) and mapped to code:

| Hyperparameter | Configuration Key | Default Value | Code Usage / Mapping |
|---|---|---|---|
| **Input Image Size** | `data.image_size` | `null` (or `224`) | Passed as `image_size_override` to [BatteryCellDataset](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/data/dataset.py#L95) to override the processor default. |
| **Learning Rate** | `learning_rate` | `0.0005` | Passed to `TrainingArguments` in [train.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/train.py#L168). |
| **Epochs** | `num_epochs` | `5` | Set as `num_train_epochs` in training arguments. |
| **Batch Size** | `batch_size` | `16` | Sets `per_device_train_batch_size` and `per_device_eval_batch_size`. |
| **Scheduler Type** | `scheduler.lr_scheduler_type` | `"cosine"` | Passed to `TrainingArguments.lr_scheduler_type`. |
| **Warmup Ratio** | `scheduler.warmup_ratio` | `0.1` | Used in `train.py` to calculate exact `warmup_steps` based on dataset length. |
| **Early Stopping** | `early_stopping_patience` | `3` | Passed to `EarlyStoppingCallback` in [train.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/train.py#L200). |
| **Precision (AMP)** | `amp.fp16` / `amp.bf16` | `false` / `false` | Passed to `TrainingArguments` for GPU mixed-precision training. |
| **Imbalance Handling** | `imbalance.loss_type` / `oversampling_method` | `"cross_entropy"` / `"none"` | Passed to [ImbalanceTrainer](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/training/trainer.py) to configure class weights, Focal Loss, and Samplers. |

### 2. Config-to-Code Integration
All YAML configuration parameters map successfully to classes and variables:
* **Augmentations**: Augmented transforms (`RandomResizedCrop`, `RandomRotation`, `ColorJitter`, `GaussianNoise`) are constructed in `build_augmentation_pipeline` in [dataset.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/data/dataset.py#L203) using probabilities/limits from the config.
* **PEFT Mapping**: `DinoV3Classifier.__init__` in [dinov3_classifier.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/models/dinov3_classifier.py#L408) correctly extracts `peft.type` to route wrapping. It dynamically maps parameters like `lora_r`, `adapter_bottleneck_dim`, and `vpt_num_tokens`.
* **Imbalance Handling**:
  - `WeightedRandomSampler` is instantiated in `ImbalanceTrainer._get_train_sampler` using class ratios.
  - Custom dataset oversampling (`BatteryCellDataset.oversample_dataset`) replicates the minority class if DDP training is active to avoid sampler incompatibility.

### 3. Audited Loopholes & Bugs (Classification)
* **Redundant Loss Calculation**: In [dinov3_classifier.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/models/dinov3_classifier.py#L554), the classifier computes standard `nn.CrossEntropyLoss` if labels are provided. However, [ImbalanceTrainer.compute_loss](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/training/trainer.py#L196) intercepts this by copying the batch and popping `"labels"` before sending it to the model. This prevents the model from computing a redundant loss, and the trainer computes the weighted loss or Focal Loss instead. This is a clever design pattern that prevents unused compute overhead.
* **DINOv3 Register Tokens**: Visual Prompt Tuning (VPT) manually bypasses the model's forward path. The code includes a fix (`C7 Fix` in `VptWrappedBackbone`) to extract `register_tokens` and insert them correctly in the sequence layout `[CLS, prompts, registers, patches]`. This prevents positional embedding mismatches and is implemented correctly.

---

## 🎯 Part 2: Detection Analysis & FPN Architecture

### 1. YOLO + DINOv3 Neck (ViTDet FPN)
The neck utilizes a **Simple Feature Pyramid (SFP)** structure as detailed in the ViTDet paper:
* Since standard Vision Transformers (like DINOv3) output a single-scale feature map (flat patch embeddings at stride 16), traditional pyramids (like FPN) cannot be built from multiple backbone stages.
* **SFP Solution**:
  - **P3 Neck Input (Stride 8)**: Upsampled by 2x using transpose convolutions (`nn.ConvTranspose2d`) from the backbone output.
  - **P4 Neck Input (Stride 16)**: Projected directly using a 1x1 convolution.
  - **P5 Neck Input (Stride 32)**: Downsampled by 2x using max pooling (`nn.MaxPool2d`) from the backbone output.
* These feature maps are then aggregated via standard YOLO26 upsampling/downsampling convolutions and concatenation, before feeding into the `Detect` head.

### 2. Config-to-Code Integration
In detection, [configs/det/peft_smoke.yaml](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/configs/det/peft_smoke.yaml) maps parameters to the YOLO overrides dictionary in [train_detection.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/scripts/train_detection.py#L115):
* `yolo_model_config` maps to the custom architecture YAML ([configs/det/yolo26_dino.yaml](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/configs/det/yolo26_dino.yaml)).
* `yolo_data_yaml` points to the Ultralytics dataset configuration.
* `epochs`, `batch_size`, `learning_rate`, `early_stopping_patience`, `seed`, and `amp` are mapped to YOLO training properties.
* **Important Note**: Data augmentations defined in the detection configuration YAML are now connected to YOLO training!
  - If `yolo_augmentations` dictionary overrides are defined inside `data:` in the config YAML, they are passed directly to YOLO overrides (supporting native YOLO parameters like `fliplr`, `degrees`, `scale`, `mosaic`, etc.).
  - If omitted but `augmentations_enabled: true` is set, standard classification augmentation values are automatically mapped to YOLO equivalents.
  - If `augmentations_enabled: false` is set, all YOLO augmentations are explicitly zeroed out for clean baselines.

### 3. Audited Loopholes & Bugs (Detection)
* **Custom Model Parsing**: The Ultralytics model parser does not recognize `DinoV3Backbone` and `DinoV3SFP` modules, which would normally crash scaling operations. The framework uses a global monkey-patch `custom_parse_model` in [yolo_utils.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/utils/yolo_utils.py#L48) to substitute custom layers with standard Conv layers, calculate depth/width scaling, and then swap in the reconstructed DINO/SFP modules. This works reliably.
* **VPT Token Slicing Offset**: When visual prompt tokens are added to the backbone, the token sequence length increases. In [yolo_dino.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/models/yolo_dino.py#L189), the slice offset `start_idx` dynamically accounts for prompt lengths:
  ```python
  start_idx = 1 + num_prompts + self.num_registers
  ```
  This is clean and prevents features from shifting or including register/prompt tokens inside spatial feature maps.

---

## 📊 Part 3: Metrics & Visualizer Diagnostics

### 1. Classification Metrics
Classification uses standard metrics computed in [cls_metrics.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/metrics/cls_metrics.py#L18):
* **Accuracy**: Proportion of correct predictions.
* **Precision / Recall / F1-Score**: Binary evaluation of the positive class (`abnormal`).
* **AUROC**: ROC Area Under Curve (computed using logits differences `logits[:, 1] - logits[:, 0]`).
* **Confusion Matrix Counts**: `tn`, `fp`, `fn`, `tp` absolute counters.
* **Single-Class Fallback**: Gracefully catches `ValueError` if evaluation splits contain only a single class, avoiding metric crash.

### 2. Detection Metrics
Detection utilizes both standard box-level evaluations and a custom image-level conversion layer in [yolo_trainer.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/training/yolo_trainer.py#L43):
* **Standard YOLO**: Bounding box precision, recall, `mAP50`, and `mAP50-95`.
* **Greedy Box Matching**: Matches GT boxes to predicted boxes of the same class (IoU threshold $\ge 0.50$). Computes:
  - **Matched Bbox IoU**: Mean IoU of matched bounding boxes.
  - **Matched Bbox Dice**: Mean Dice score of matched bounding boxes, calculated as $Dice = \frac{2 \cdot IoU}{1 + IoU}$.
* **Image-Level Multi-Label Conversion**: Evaluates whether an image contains any abnormal.
  - An image is classified as positive (`abnormal`) if a prediction box of class `abnormal` has a confidence score $\ge 0.25$.
  - Generates Accuracy, Precision, Recall, F1, AUROC, and confusion matrix counts (`tn`, `fp`, `fn`, `tp`) on the converted image-level anomaly predictions.

### 3. Visualizer Auditing
The Streamlit dashboard in [visualize.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/visualize.py) was audited for bugs:
* **Missing Run Fallbacks**: If no output runs are present, it avoids crashing by presenting warning panels and loading a clean placeholder layout.
* **Trajectory Curve Plots**: Correctly aligns epoch indexing and resolves mismatched metric names between classification runs (`eval_f1`) and detection runs (`eval_custom_cls_f1/abnormal`).
* **DDP Isolation Warnings**: Gracefully reads nested `trainer_state.json` logs across diverse training outputs.

---

## 🧪 Part 4: Unittests Execution

The unittest suite was run using the Conda environment at `/home/jovyan/pytorch_env`. All **32 tests** compiled and executed successfully:

```text
Ran 32 tests in 35.353s

OK
```

### 1. Test Coverage Overview
* **[test_models.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/tests/test_models.py)**: Asserts linear and MLP head layers, bottleneck adapter residual identities (initializes to exactly equal inputs at step 0), VPT layout sequence slices, and contains the **VPT Deep Layer Prompt Wrapper Test** verifying prompt token swapping.
* **[test_dataset.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/tests/test_dataset.py)**: Verifies oversampling reproducibility with locked seeds, image load formats, augmentation combi-samplers, and contains the **DDP Mock Oversampling Test** simulating multi-rank environments.
* **[test_yolo_shapes.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/tests/test_yolo_shapes.py)**: Checks that the custom DINO backbone and SFP neck build successfully, compile layer weights, preserve metadata indices, and yield the expected output grid shapes (`[batch, 300, 6]`).
* **[test_yolo_metrics.py](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/tests/test_yolo_metrics.py)**: Runs mock evaluations on coordinates matching and checks validation reports.

### 2. Implemented Additional Tests
To further reinforce the test suite, we successfully added:
1. **DDP Mock Oversampling Test**: Simulates multi-GPU runs (Ranks 0-3) to verify that dataset lengths and order are replicated properly without rank-level collated drift.
2. **VPT Deep Layer Prompt Wrapper Test**: Checks that the `VptLayerWrapper` correctly removes prompt tokens from the previous layer's hidden states and prepends the new deep prompt tokens.

---

## 🪟 Recommendations & Findings Summary
* **No critical bugs or crashes** were found during this audit. The framework is highly cohesive, modular, and robust.
* **Augmentation Override Clarification**: Resolved! Connected YAML config parameters to the YOLO overrides. Users can now pass native YOLO augmentations via the `yolo_augmentations` config dictionary or rely on the fallback mapping of classification variables.
* **Class Weights Sync**: Resolved! Passed configuration class names down to `CustomDetectionTrainer` and `CustomDetectionValidator`. The validator dynamically resolves the matching abnormal index and duplicates the logged metric keys under both the default name (`abnormal`) and the custom config name (e.g. `abnormal`). `visualize.py` was updated to look up metrics dynamically under both names, maintaining backward compatibility.
