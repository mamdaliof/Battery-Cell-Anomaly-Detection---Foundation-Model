# Battery Cell Anomaly Detection with DINOv3 and PEFT

## Project overview

This project uses DINOv3 vision transformers as a frozen backbone and fine-tunes lightweight classification heads with and without parameter-efficient fine-tuning (PEFT) methods to detect abnormal battery cells from images. The dataset is highly imbalanced (normal vs abnormal), and the main goal is to maximize F1 score on the abnormal class while also monitoring accuracy, precision, recall, AUROC, and confusion matrix counts.

## High-level components

1. **Data module (HF-compatible)**
   - Load train/validation image datasets in a format compatible with Hugging Face `Trainer`.
   - Apply consistent preprocessing and augmentations for all experiments.
   - Provide class labels and metadata required for imbalance handling (class counts/weights).

2. **Model module (DINOv3 + classifier)**
   - Load a DINOv3 backbone from `transformers` (start with a smaller variant, e.g. ViT-B/16).
   - Freeze the DINOv3 backbone by default and add a configurable classification head (`DinoV3Classifier`).
   - Implement a pure baseline: frozen DINOv3 + configurable head, trained without PEFT.

3. **PEFT integration module**
   - Integrate Hugging Face PEFT for parameter-efficient fine-tuning.
   - **Design Decision**: Wrap only the underlying DINOv3 backbone (`self.backbone`) using the HF `peft` library, leaving the classification head (`self.classifier`) as standard trainable PyTorch parameters.
   - Support multiple PEFT methods via configuration:
     - LoRA: targeting selected attention/MLP modules (e.g. q, v, or full qkv; number of blocks configurable).
     - Adapters: small bottleneck/adapter modules inserted in selected transformer blocks.
     - Visual prompt tuning: learnable visual tokens prepended to patch embeddings.
   - **LoRA block targeting**: Apply `LoraConfig` without `layers_to_transform`/`layers_pattern` (unreliable for DINOv3), then post-wrap freeze `lora_*` weights in non-target blocks via `requires_grad = False`. Block discovery uses the `_get_transformer_blocks()` helper.
   - Ensure that PEFT configuration (type, target modules, ranks, number of prompt tokens, etc.) is fully driven by a config file.

4. **Training pipeline (Trainer + Accelerate)**
   - Use Hugging Face `Trainer` as the main training interface, relying on its built-in integration with `accelerate` for multi-GPU and mixed precision training.
   - Configure training hyperparameters through a YAML config file (epochs, batch size, learning rate, scheduler, AMP, etc.).
   - Each run creates a unique output directory and stores the exact config used.
   - **`ImbalanceTrainer`** extends `Trainer`; no `training_step` override — standard HF Trainer step is used.

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
  - [x] Update `src/bcadfm/models/dinov3_classifier.py` or a utility helper to wrap the underlying backbone (`self.backbone`) using HF's `peft` library. **Note**: Leave the classification head (`self.classifier`) as standard trainable PyTorch parameters.
  - [x] Implement LoRA configuration logic using `LoraConfig` and `get_peft_model`.
  - [x] Implement Adapter module insertions utilizing the HF `peft` third-party library.
  - [x] Implement Visual Prompt Tuning (VPT) block wrapping.
- [x] **Fix LoRA Block Targeting for DINOv3** (2026-06-09):
  - [x] Replace `layers_to_transform`/`layers_pattern` PEFT mechanism with post-wrap `requires_grad = False` on non-target block LoRA weights.
  - [x] Implement architecture-agnostic `_get_transformer_blocks()` helper.
  - [x] Verify correct trainable parameter counts per ablation config.
- [x] **Verify Freezing & Trainable Parameters**:
  - [x] Implement a utility in `src/bcadfm/utils/model_utils.py` that counts and lists trainable parameters.
  - [x] Log the exact list of trainable parameters when initializing the model.
 
### 5. Imbalance Handling Module
- [x] **Class-Weighted Cross-Entropy**:
  - [x] Compute class frequencies from the training split.
  - [x] Implement weighted cross-entropy as a **custom loss class** in `src/bcadfm/training/losses.py`.
- [x] **Focal Loss**:
  - [x] Implement focal loss as a **custom loss class** in `src/bcadfm/training/losses.py` with configurable `alpha` and `gamma` parameters.
  - [x] Allow switching loss function via `imbalance.loss_type` in the YAML config.
- [x] **Minority-Class Oversampling**:
  - [x] Implement a custom data loader sampler (e.g. PyTorch `WeightedRandomSampler`) that oversamples the minority class during training.
  - [x] Add a config toggle under `imbalance.sampler` to enable/disable oversampling.
- [x] **Config Integration**:
  - [x] Expose an `imbalance` section in the YAML schema to choose and combine strategies.
- [x] **Trainer Stability** (2026-06-09):
  - [x] Remove `training_step` timing override from `ImbalanceTrainer` — was causing `TypeError` under HF Trainer v5.x.
  - [x] Confirm standard `Trainer.training_step` is used.
 
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

## ✅ Review Checklist for 2026-06-10

> **These are the items changed on 2026-06-09. Check each one before running ablations.**

### 🔴 Must Verify Before Running

- [ ] **`src/bcadfm/training/trainer.py` — `training_step` removed**
  - Open the file and confirm there is NO `training_step` method in `ImbalanceTrainer`.
  - Confirm the six timing variables (`_last_step_time`, `_step_count`, etc.) are NOT in `__init__`.
  - Run: `grep -n "training_step\|_step_count\|_accumulated" src/bcadfm/training/trainer.py` → should return empty.

- [ ] **`src/bcadfm/models/dinov3_classifier.py` — LoRA block targeting**
  - Open `_apply_lora` (or the PEFT section of `__init__`) and confirm:
    - No `layers_to_transform` or `layers_pattern` keys are passed to `LoraConfig`.
    - There is a post-wrap loop that sets `param.requires_grad = False` for `lora_*` weights in non-target blocks.
  - Confirm `_get_transformer_blocks()` helper function exists near the top of the file.
  - Run: `grep -n "layers_to_transform\|layers_pattern" src/bcadfm/models/dinov3_classifier.py` → should return empty.

- [ ] **End-to-end smoke test (most important)**
  ```bash
  torchrun --nproc_per_node=2 scripts/train.py --config configs/peft_smoke.yaml
  ```
  Expected: trains without `TypeError`, logs trainable parameter summary.

### 🟡 Verify If Running LoRA Ablations

- [ ] **Trainable parameter counts match expectations**
  - For a LoRA config targeting blocks `[8, 9, 10, 11]` with `r=8`, only those 4 blocks' `lora_A`/`lora_B` weights should appear in the trainable params log.
  - Run `python scripts/validate_ablation_configs.py` and inspect the printed param tables.

- [ ] **`_get_transformer_blocks()` resolves correctly for DINOv3**
  - Quick check in a Python shell:
    ```python
    from src.bcadfm.models.dinov3_classifier import _get_transformer_blocks, DinoV3Classifier
    m = DinoV3Classifier(...)  # your config
    blocks = _get_transformer_blocks(m.backbone)
    print(len(blocks))  # should be 12 for ViT-B
    ```

### 🟢 Documentation / No Code Impact

- [ ] Devlog `devlogs/DEVLOG_LORA_BLOCK_TARGETING_FIX.md` exists and is readable.
- [ ] `PEFT_IMBALANCE_REPORT.md` §3 and §4 are updated.
- [ ] `README.md` LoRA bullet reflects new mechanism.
- [ ] `PROJECT_PLAN.md` §4 and §5 have new `[x]` items for the 2026-06-09 fixes.

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

setattr(ultralytics.nn.tasks, "DinoV3Backbone", DinoV3Backbone)
setattr(ultralytics.nn.tasks, "DinoV3SFP_P3", DinoV3SFP_P3)
setattr(ultralytics.nn.tasks, "DinoV3SFP_P4", DinoV3SFP_P4)
setattr(ultralytics.nn.tasks, "DinoV3SFP_P5", DinoV3SFP_P5)
```

### 3. Custom Loss Integration (Class Imbalance Handling)
To apply class weighting or focal loss within the YOLO26 training pipeline:
- **`YOLODetectionLoss`**:
  - Subclasses the standard Ultralytics detection loss class.
  - Overrides the classification loss component to inject custom class weights or focal loss parameters.
- **`YOLODetectionTrainer`**:
  - Inherits from `ultralytics.models.yolo.detect.DetectionTrainer`.
  - Overrides `init_criterion(self)` to return our custom `YOLODetectionLoss`.
- **Training Invocation**:
  ```python
  model = YOLO("configs/yolo26_dino.yaml")
  model.train(data="data/battery_detection.yaml", trainer=YOLODetectionTrainer, ...)
  ```

---

## Object Detection Pipeline (YOLO26 + DINOv3) TODOs

### Sub-Task 1: Dynamic Module Registration & Env Verification
- [x] Conceptualize dynamic registration wrapper.
- [x] Implement the dynamic module registration helper in `src/bcadfm/utils/yolo_utils.py`.
- [x] Verify custom modules load correctly from YAML config parsing.

### Sub-Task 2: SFP Feature Pyramid Implementation
- [x] Implement `DinoV3Backbone`, `DinoV3SFP_P3`, `DinoV3SFP_P4`, `DinoV3SFP_P5` in `src/bcadfm/models/yolo_dino.py`.
- [x] Write shape unit tests in `tests/test_yolo_shapes.py`.
- [x] Verify all pyramid output shapes are correct for standard input resolutions.

### Sub-Task 3: YOLO26 Integration & Training
- [x] Create `configs/yolo26_dino.yaml` integrating SFP backbone with YOLO26 head.
- [x] Verify end-to-end forward pass produces correct loss shapes.
- [ ] Run full training on battery detection data and compare against YOLOv8/YOLO11 baseline.

### Sub-Task 4: PEFT for Detection
- [x] Integrate LoRA, Adapters, and VPT into `YoloDinoClassifier` for the detection backbone.
- [x] Verify PEFT parameter counts for detection configs.
- [ ] Run detection ablations comparing frozen vs LoRA-tuned backbone.
