# Battery Cell Anomaly Detection with DINOv3 and PEFT

## Project overview

This project uses DINOv3 vision transformers as a frozen backbone and fine-tunes lightweight classification heads with and without parameter-efficient fine-tuning (PEFT) methods to detect abnormal battery cells from images. The dataset is highly imbalanced (normal vs abnormal), and the main goal is to maximize F1 score on the abnormal class while also monitoring accuracy, precision, recall, AUROC, and confusion matrix counts.

## High-level components

1. **Data module (HF-compatible)**
   - Load train/validation image datasets in a format compatible with Hugging Face `Trainer`.
   - Apply consistent preprocessing and augmentations for all experiments.
   - Provide class labels and metadata required for imbalance handling (class counts/weights).

2. **Model module (DINOv3 + classifier)**
   - Load a DINOv3 backbone from `transformers` (start with a smaller variant, e.g. ViT-B/16).[web:3]
   - Freeze the DINOv3 backbone by default and add a configurable classification head (`DinoV3Classifier`).
   - Implement a pure baseline: frozen DINOv3 + configurable head, trained without PEFT.

3. **PEFT integration module**
   - Integrate Hugging Face PEFT for parameter-efficient fine-tuning.
   - **Design Decision**: Wrap only the underlying DINOv3 backbone (`self.backbone`) using the HF `peft` library, leaving the classification head (`self.classifier`) as standard trainable PyTorch parameters.
   - Support multiple PEFT methods via configuration:
     - LoRA: targeting selected attention/MLP modules (e.g. q, v, or full qkv; number of blocks configurable).
     - Adapters: small bottleneck/adapter modules inserted in selected transformer blocks (leveraging HF `peft` library's third-party implementation).
     - Visual prompt tuning: learnable visual tokens prepended to patch embeddings.
   - Ensure that PEFT configuration (type, target modules, ranks, number of prompt tokens, etc.) is fully driven by a config file.

4. **Training pipeline (Trainer + Accelerate)**
   - Use Hugging Face `Trainer` as the main training interface, relying on its built-in integration with `accelerate` for multi-GPU and mixed precision training.[web:7]
   - Configure training hyperparameters through a YAML config file (epochs, batch size, learning rate, scheduler, AMP, etc.).
   - Each run creates a unique output directory and stores the exact config used.

5. **Imbalance handling module**
   - Implement multiple imbalance handling strategies, all configurable:
     - Class weighting in the loss (implemented via a custom loss module).
     - Data-level strategies (e.g. oversampling minority class via standard sampler).
     - Focal loss (implemented via a custom loss module).
   - Allow enabling/disabling or combining these strategies via the config.

6. **Evaluation and metrics module**
   - Implement custom `compute_metrics` for `Trainer` to return:
     - Accuracy.
     - Precision, recall, F1 (with F1 as the main selection metric).
     - AUROC.
     - Confusion matrix cell counts (TN, FP, FN, TP) as absolute values.
   - Optionally support class-wise metrics if useful.

7. **Ablation study framework**
   - Design experiments to compare:
     - Baseline: frozen DINOv3 + configurable head (no PEFT).
     - LoRA variants (different ranks, different sets of targeted blocks/modules).
     - Adapter variants (different bottleneck sizes and placement).
     - Visual prompt tuning with different numbers of prompt tokens.
     - [Optional] Different imbalance strategies (class weights vs focal loss vs none).
   - Fix random seed, DINOv3 backbone, and preprocessing pipeline for all runs.
   - Use config files to define each experiment setting so runs are reproducible.

8. **Experiment management and logging**
   - Central YAML configuration system to specify:
     - Dataset paths and splits.
     - Model and backbone options.
     - PEFT method and its hyperparameters.
     - Training hyperparameters and scheduler/AMP options.
     - Imbalance handling options.
   - Logging of metrics and configuration for each run (e.g. via run directories and optional external loggers).

## TODOs and implementation steps

### 1. Repository and Environment Setup
- [x] Define a standard Python package/layout structure (e.g. `src/` with `data/`, `models/`, `training/`, `configs/`).
- [x] Add a `requirements.txt` listing core dependencies: `transformers`, image libraries, `peft`, `accelerate`, `torch`, metric libraries.
- [x] Expand `README.md` with a short description, design decisions, dataset conversion instructions, baseline model description, and config-driven training.
- [x] **Fix DINOv3 Gated Access**: Set up access token and test download.
  - [x] Set up fine-grained Hugging Face read token with permission to access the `facebook/dinov3-vitb16-pretrain-lvd1689m` gated repository.
  - [x] Authenticate on the target training server using `huggingface-cli login`.
  - [x] Run a test python command to verify `AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")` works without a 403 or GatedRepoError.

### 2. Data Pipeline & Baseline Verification
- [x] Implement a data loading module (`src/bcadfm/data/dataset.py`) compatible with Hugging Face `Trainer`.
- [x] Implement conversion script (`scripts/convert_split_base_to_classification.py`).
- [x] Add dataset config section in configs.
- [x] **Data Pipeline Robustness & Processor Fallback**:
  - [x] Inspect how `AutoImageProcessor` loads processor for DINOv3 vs ViT. If processor loading fails on the training machine due to library versions, implement a try/except fallback that constructs manual PyTorch transforms (Resize, CenterCrop, Normalize using standard ImageNet stats). (Handled by removing manual fallback per request and strictly utilizing HF processor).
- [x] **Stabilize Baseline**:
  - [x] Ensure `configs/baseline.yaml` and `configs/test_smoke.yaml` load and run properly.
  - [x] Execute a smoke-test run using `google/vit-base-patch16-224` to confirm image loading, batching, and basic forward/backward passes work end-to-end on CPU/GPU.
 
### 3. Baseline Model & Metrics Stabilization
- [x] Implement `DinoV3Classifier` module wrapping the backbone and a configurable classification head.
- [x] Implement `compute_cls_metrics` returning Accuracy, Precision, Recall, F1, AUROC, and confusion matrix counts (TN, FP, FN, TP).
- [x] Implement `SaveTwoBestClsModelsCallback` for tracking and saving `best_loss.pt` and `best_f1.pt`.
- [ ] **Turn Baseline into a Benchmark**:
  - [ ] Set up a canonical training config for the baseline benchmark (frozen backbone, classification head depth=1, specified epochs, batch size, learning rate).
  - [ ] Run the baseline model under this canonical config multiple times with different seeds (e.g. `seed=42`, `seed=100`) to measure baseline metric variance.
  - [ ] Re-evaluate early stopping in Hugging Face Trainer v5.x API. If early stopping is desired, resolve the `evaluation_strategy`/`save_strategy` deprecation or keep early stopping disabled and rely on `SaveTwoBestClsModelsCallback` + manual epoch limits.
 
### 4. PEFT Integration Module (LoRA, Adapters, VPT)
- [x] **Define PEFT Configuration Schema**:
  - [x] Add config classes under `src/bcadfm/utils/config.py` for PEFT settings.
- [x] **Implement PEFT Wrapping Logic**:
  - [x] Update `src/bcadfm/models/dinov3_classifier.py` or a utility helper to wrap the underlying backbone (`self.backbone`) using HF's `peft` library. **Note**: Leave the classification head (`self.classifier`) as standard trainable PyTorch parameters (do not wrap the entire classifier model).
  - [x] Implement LoRA configuration logic using `LoraConfig` and `get_peft_model`.
  - [x] Implement Adapter module insertions utilizing the HF `peft` third-party library.
  - [x] Implement Visual Prompt Tuning (VPT) block wrapping (prepending learned prompt tokens to patch embeddings and modifying attention masks if necessary).
- [x] **Verify Freezing & Trainable Parameters**:
  - [x] Implement a utility in `src/bcadfm/utils/model_utils.py` that counts and lists trainable parameters.
  - [x] Log the exact list of trainable parameters when initializing the model, verifying that only PEFT adapter/LoRA weights and the classification head have `requires_grad = True` while the backbone remains fully frozen.
 
### 5. Imbalance Handling Module
- [x] **Class-Weighted Cross-Entropy**:
  - [x] Compute class frequencies from the training split.
  - [x] Implement weighted cross-entropy as a **custom loss class** in `src/bcadfm/training/losses.py`, scaling the loss based on the inverse class frequencies.
- [x] **Focal Loss**:
  - [x] Implement focal loss as a **custom loss class** in `src/bcadfm/training/losses.py` with configurable `alpha` and `gamma` parameters.
  - [x] Allow switching loss function via `imbalance.loss_type` in the YAML config, passing the custom loss class to the HF Trainer.
- [x] **Minority-Class Oversampling**:
  - [x] Implement a custom data loader sampler (e.g. PyTorch `WeightedRandomSampler`) that oversamples the minority class during training.
  - [x] Add a config toggle under `imbalance.sampler` to enable/disable oversampling.
- [x] **Config Integration**:
  - [x] Expose an `imbalance` section in the YAML schema to choose and combine strategies.
 
### 6. Training & Multi-GPU Infrastructure
- [x] Configure `TrainingArguments` to support batch size, lr, mixed precision, and schedulers via YAML.
- [x] **Confirm Multi-GPU Execution**:
  - [x] Verify that `torchrun --nproc_per_node=N` correctly distributes training across available GPUs.
  - [x] Test mixed-precision (`fp16: true` or `bf16: true`) under multi-GPU execution.
  - [x] Verify that the `SaveTwoBestClsModelsCallback` behaves correctly under DDP (i.e. only rank 0 writes checkpoints, no lockouts or race conditions).

### 7. Ablation Study Framework & Layered Experiments
- [ ] **Layer 1: PEFT vs Non-PEFT Baseline**:
  - [ ] Keep the imbalance strategy fixed (e.g., class-weighted cross-entropy).
  - [ ] Run and compare:
    1. Baseline (Frozen backbone, classification head only, no PEFT).
    2. LoRA with small rank (e.g., r=8) applied to `q, v` modules in last 4 blocks.
    3. LoRA with medium rank (e.g., r=16) applied to all blocks.
    4. Adapters with bottleneck dimension=64 in last 4 blocks.
    5. Visual Prompt Tuning (VPT) with 8 tokens.
- [ ] **Layer 2: Imbalance Strategies**:
  - [ ] Select the best-performing PEFT configuration from Layer 1.
  - [ ] Run and compare imbalance handling techniques:
    1. No imbalance handling (standard cross-entropy).
    2. Class-weighted cross-entropy.
    3. Focal loss.
    4. Minority-class oversampling.
- [ ] **Variance & Seed Control**:
  - [ ] Run each ablation candidate with at least 2 distinct seeds to ensure statistical significance.
  - [ ] Document validation metrics for each run.
- [ ] **Final Test Evaluation**:
  - [ ] Set aside a small test dataset split. Only run evaluation on this split at the very end on the final chosen configuration to prevent validation leakage.

### 8. Future Plans & Advanced Ablations
- [ ] Experiment with partially unfreezing the top-most layers of the backbone (e.g., last 2 blocks) alongside PEFT.
- [ ] Integrate hyperparameter search for learning rate, LoRA rank, and focal loss parameters.

---

## YOLO26 + DINOv3 (ViTDet) & Custom Loss Implementation Details

### 1. ViTDet-Style Simple Feature Pyramid (SFP) Wrapper Design
Based on Detectron2's `SimpleFeaturePyramid` implementation:
- **`DinoV3Backbone`**:
  - Load the DINOv3 model (`facebook/dinov3-vitb16-pretrain-lvd1689m`).
  - Extract the patch tokens from the final encoder layer.
  - Reshape from `(B, H_patch * W_patch, D)` to 2D feature grid `(B, D, H_patch, W_patch)`.
- **`DinoV3SFP_P3` (Stride 8)**:
  - Input: `(B, D, H_patch, W_patch)` (stride 16 relative to input).
  - Layers:
    - `nn.ConvTranspose2d(D, D // 2, kernel_size=2, stride=2)` -> Stride 8.
    - `nn.LayerNorm` (or `BatchNorm2d`) + `nn.GELU()`.
    - `nn.Conv2d(D // 2, out_channels, kernel_size=1)` (project to neck channel dimension, e.g., 256).
    - `nn.LayerNorm` (or `BatchNorm2d`).
    - `nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)` (spatial smoothing).
    - `nn.LayerNorm` (or `BatchNorm2d`).
- **`DinoV3SFP_P4` (Stride 16)**:
  - Input: `(B, D, H_patch, W_patch)` (stride 16).
  - Layers:
    - `nn.Conv2d(D, out_channels, kernel_size=1)` (channel projection, e.g., 512).
    - `nn.LayerNorm` (or `BatchNorm2d`).
    - `nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)` (spatial smoothing).
    - `nn.LayerNorm` (or `BatchNorm2d`).
- **`DinoV3SFP_P5` (Stride 32)**:
  - Input: `(B, D, H_patch, W_patch)` (stride 16).
  - Layers:
    - `nn.MaxPool2d(kernel_size=2, stride=2)` -> Stride 32.
    - `nn.Conv2d(D, out_channels, kernel_size=1)` (channel projection, e.g., 1024).
    - `nn.LayerNorm` (or `BatchNorm2d`).
    - `nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)` (spatial smoothing).
    - `nn.LayerNorm` (or `BatchNorm2d`).

### 2. Dynamic Namespace Patching (Monkey-Patching)
To register the custom layers inside the `ultralytics` package without altering vendor code:
```python
import ultralytics.nn.tasks
from src.bcadfm.models.yolo_dino import DinoV3Backbone, DinoV3SFP_P3, DinoV3SFP_P4, DinoV3SFP_P5

# Dynamically inject the custom classes into the globals of ultralytics.nn.tasks
# so that eval("DinoV3Backbone") resolves correctly during YAML configuration parsing.
setattr(ultralytics.nn.tasks, "DinoV3Backbone", DinoV3Backbone)
setattr(ultralytics.nn.tasks, "DinoV3SFP_P3", DinoV3SFP_P3)
setattr(ultralytics.nn.tasks, "DinoV3SFP_P4", DinoV3SFP_P4)
setattr(ultralytics.nn.tasks, "DinoV3SFP_P5", DinoV3SFP_P5)
```

### 3. Custom Loss Integration (Class Imbalance Handling)
To apply class weighting or focal loss within the YOLO26 training pipeline:
- **`YOLODetectionLoss`**:
  - Subclasses the standard Ultralytics detection loss class.
  - Overrides the classification loss component (typically computed using `nn.BCEWithLogitsLoss`) to inject custom class weights or focal loss parameters.
- **`YOLODetectionTrainer`**:
  - Inherits from `ultralytics.models.yolo.detect.DetectionTrainer`.
  - Overrides `init_criterion(self)` to return our custom `YOLODetectionLoss`.
- **Training Invocation**:
  - Pass the custom trainer class explicitly to `model.train()`:
    ```python
    model = YOLO("configs/yolo26_dino.yaml")
    model.train(data="data/battery_detection.yaml", trainer=YOLODetectionTrainer, ...)
    ```

---

## Object Detection Pipeline (YOLO26 + DINOv3) TODOs

### Sub-Task 1: Dynamic Module Registration & Env Verification
- [x] **Conceptualize dynamic registration wrapper**: Map custom modules to the `ultralytics.nn.tasks` namespace at runtime to avoid modifying vendor code. (Done)
- [x] **Conceptualize verification script layout**: Structure a mock script that loads a dummy YAML configuration to verify layer instantiation. (Done)
- [x] Implement the dynamic module registration helper in `src/bcadfm/utils/yolo_utils.py`.
- [x] Implement the registration verification script in `tests/test_yolo_registration.py`.
- [x] Execute `python tests/test_yolo_registration.py` and resolve any package import or runtime configuration issues.

### Sub-Task 2: DinoV3 & SFP PyTorch Modules
- [x] **Conceptualize DinoV3Backbone**: Design patch token sequence extraction, CLS token removal, and 2D tensor reshaping. (Done)
- [x] **Conceptualize DinoV3SFP_P3/P4/P5 Layers**: Design stride-8, stride-16, and stride-32 convolutional/pooling modules with Channel-wise GroupNorm (as LayerNorm) and spatial smoothing blocks. (Done)
- [x] Implement `DinoV3Backbone` module in `src/bcadfm/models/yolo_dino.py`.
- [x] Implement `DinoV3SFP_P3`, `DinoV3SFP_P4`, and `DinoV3SFP_P5` modules in `src/bcadfm/models/yolo_dino.py`.
- [x] Write shapes unit tests in `tests/test_yolo_shapes.py` to ensure feature maps align with strides 8, 16, and 32 on a $640 \times 640$ input tensor.

### Sub-Task 3: Custom YOLO26 Config (configs/yolo26_dino.yaml)
- [x] **Conceptualize custom network architecture mapping**: Map custom backbone layers (P3, P4, P5 at layers 1, 2, 3) to neck upsamplers, concats, and detection heads. (Done)
- [x] Write the custom network architecture YAML file `configs/yolo26_dino.yaml`.
- [x] Verify initialization by loading the configuration via the Ultralytics model class (`model = YOLO("configs/yolo26_dino.yaml")`) in a validation script.

### Sub-Task 4: Custom Loss and Trainer (Imbalance Handling)
- [x] **Conceptualize custom loss subclassing**: Design class-weighted BCE / Focal Loss integration in YOLO detection loss. (Done)
- [x] **Conceptualize custom trainer subclassing**: Design custom trainer overriding `init_criterion` to inject weighted detection loss. (Done)
- [x] **De-prioritize custom loss/trainer components**: To isolate DINOv3 backbone comparative performance, use standard YOLO losses and trainers natively as requested by user. (Cancelled/Deferred)
- [x] Create dataset variant yaml configurations under `data/` referencing train/val splits. (Done)
- [x] Implement custom validator (`CustomDetectionValidator`) and trainer (`CustomDetectionTrainer`) to compute custom class-wise stats, matched box IoU/Dice, and multi-label classification conversions. (Done)


### Sub-Task 5: Training Pipeline & Ablations
- [x] **Conceptualize DDP training execution**: Design script with multi-process dynamic module registration and argument parsers for multi-GPU training. (Done)
- [x] **Conceptualize ablation study configurations**: Detail the training recipes for standard YOLO26, YOLO26 + DINOv3 + SFP, and YOLO26 + DINOv3-LoRA + SFP. (Done)
- [ ] Run standard YOLO26 baseline training.
- [ ] Run YOLO26 + Frozen DINOv3 + SFP training.
- [ ] Run YOLO26 + Fine-Tuned DINOv3 (LoRA) + SFP training.
- [ ] Log and compare mAP@0.5 and abnormal class F1 metrics across all runs.

---

## Ablation Analysis & Visualization (Classification & Detection) TODOs

### Step 1: Folder Completion Checker & Configuration Comparer
- [x] **Conceptualize folder scan & configuration comparer**: Design logic to detect incomplete runs (missing `DONE` files) and compile parameters from `config.yaml` files. (Done)
- [x] Implement `scripts/check_runs.py` (completed as `scripts/check_ablation_status.py`) to identify incomplete folders, list completed runs, and output a summary of completed hyperparameters.

### Step 2: Interactive Jupyter Notebook Visualizer
- [x] **Conceptualize Jupyter notebook visualizer**: Design `ipywidgets` + `plotly` selection logic to select and display metrics interactively inside a notebook. (Done)
- [x] Create `notebooks/visualize_results.ipynb` containing the interactive training curves and leaderboard tables.

### Step 3: Streamlit Web Visualizer Enhancement (`visualize.py`)
- [x] **Conceptualize Streamlit/Plotly visualizer enhancements**: Compare with the existing `visualize.py` code, and outline updates to handle Hugging Face `trainer_state.json` formats, anomaly-specific metrics (F1, AUROC, confusion matrix counts), and PEFT configs. (Done)
- [x] Refactor `visualize.py` at the root of the project to add native support for Hugging Face trainer files, PEFT parameters, and battery anomaly detection metrics.





