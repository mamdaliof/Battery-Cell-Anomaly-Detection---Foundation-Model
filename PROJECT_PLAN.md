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


