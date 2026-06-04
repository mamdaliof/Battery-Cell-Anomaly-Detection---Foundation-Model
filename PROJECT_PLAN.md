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
   - Freeze the DINOv3 backbone by default and add a linear classification head for binary classification.
   - Implement a pure baseline: frozen DINOv3 + linear head, trained without PEFT.

3. **PEFT integration module**
   - Integrate Hugging Face PEFT for parameter-efficient fine-tuning.
   - Support multiple PEFT methods via configuration:
     - LoRA: targeting selected attention/MLP modules (e.g. q, v, or full qkv; number of blocks configurable).
     - Adapters: small bottleneck modules inserted in selected transformer blocks.
     - Visual prompt tuning: learnable visual tokens prepended to patch embeddings.
     - [Placeholder] Additional PEFT methods to be added later.
   - Ensure that PEFT configuration (type, target modules, ranks, number of prompt tokens, etc.) is fully driven by a config file.

4. **Training pipeline (Trainer + Accelerate)**
   - Use Hugging Face `Trainer` as the main training interface, relying on its built-in integration with `accelerate` for multi-GPU and mixed precision training.
   - Configure training hyperparameters through a config file (learning rate, batch size, epochs, optimizer, scheduler, gradient accumulation, fp16/bf16, etc.).
   - Support multi-GPU training on up to 8 GPUs (12 GB each) without writing a custom training loop.

5. **Imbalance handling module**
   - Implement multiple imbalance handling strategies, all configurable:
     - Class weighting in the loss (e.g. weighted cross-entropy using class frequencies).
     - Data-level strategies (e.g. oversampling minority class or targeted augmentations).
     - Focal loss as an alternative to standard cross-entropy.
   - Allow enabling/disabling or combining these strategies via the config.

6. **Evaluation and metrics module**
   - Implement custom `compute_metrics` for `Trainer` to return:
     - Accuracy.
     - Precision, recall, F1 (with focus on abnormal class if needed).
     - AUROC.
     - Confusion matrix cell counts (TN, FP, FN, TP) as absolute values.
   - Optionally support class-wise metrics if useful.

7. **Ablation study framework**
   - Design experiments to compare:
     - Baseline: frozen DINOv3 + linear head (no PEFT).
     - LoRA variants (different ranks, different sets of targeted blocks/modules).
     - Adapter variants (different bottleneck sizes and placement).
     - Visual prompt tuning with different numbers of prompt tokens.
     - [Optional] Different imbalance strategies (class weights vs focal loss vs none).
   - Fix random seed, DINOv3 backbone, and preprocessing pipeline for all runs.
   - Use config files to define each experiment setting so runs are reproducible.

8. **Experiment management and logging**
   - Central configuration system (YAML/JSON/pyconfig) to specify:
     - Dataset paths and splits.
     - Model and backbone options.
     - PEFT method and its hyperparameters.
     - Training hyperparameters.
     - Imbalance handling options.
   - Logging of metrics and configuration for each run (e.g. to local files, TensorBoard, or WandB).

## TODOs and implementation steps

### 1. Repository setup

- [ ] Define a standard Python package/layout structure (e.g. `src/` with `data/`, `models/`, `training/`, `configs/`).
- [ ] Add a `pyproject.toml` or `requirements.txt` listing core dependencies: `transformers`, `datasets` or `image` libraries, `peft`, `accelerate`, `torch`, metric libraries.
- [ ] Expand `README.md` with a short description and a link to this project spec.

### 2. Data pipeline

- [ ] Implement a data loading module that:
  - [ ] Reads train/val splits from the battery cell dataset.
  - [ ] Applies DINOv3-compatible image preprocessing (resize, normalization, etc.).
  - [ ] Adds optional augmentations (config-controlled).
  - [ ] Exposes a `Dataset` compatible with Hugging Face `Trainer`.
- [ ] Add a config section for dataset parameters (paths, resolution, augmentations, batch size).

### 3. Baseline model (no PEFT)

- [ ] Implement a DINOv3 classifier module that:
  - [ ] Loads a pre-trained DINOv3 backbone from `transformers`.
  - [ ] Freezes all backbone weights.
  - [ ] Adds a trainable linear classification head for 2 classes.
- [ ] Integrate this model into `Trainer` with a standard cross-entropy loss.
- [ ] Run a first baseline experiment and log metrics (accuracy, precision, recall, F1, AUROC, confusion matrix).

### 4. PEFT integration

- [ ] Add PEFT configuration support:
  - [ ] Create a config section for `peft.type` (e.g. `none`, `lora`, `adapter`, `visual_prompt`).
  - [ ] For LoRA: config parameters for rank, alpha, dropout, targeted modules (e.g. `q`, `k`, `v`, `out`), and which transformer blocks to apply it to.
  - [ ] For adapters: config parameters for bottleneck size and which blocks to insert them into.
  - [ ] For visual prompt tuning: config for number of prompt tokens and where to insert them.
- [ ] Implement model-wrapping functions that apply the chosen PEFT method to the base DINOv3 classifier.
- [ ] Ensure all PEFT parameters are correctly registered as trainable while the backbone remains frozen (unless the config says otherwise).

### 5. Imbalance handling

- [ ] Implement class-weighted cross-entropy based on dataset statistics.
- [ ] Implement focal loss and allow switching between focal and standard cross-entropy via config.
- [ ] (Optional, but recommended) Implement an oversampling sampler or a simple minority-class oversampling mechanism.
- [ ] Add config options to select and combine these strategies.

### 6. Training and multi-GPU

- [ ] Define `TrainingArguments` for `Trainer`, including:
  - [ ] Learning rate, weight decay, epochs, warmup, scheduler.
  - [ ] Mixed precision (fp16/bf16) and gradient accumulation.
  - [ ] Logging and evaluation frequency.
- [ ] Confirm multi-GPU training works by launching with `accelerate launch` or `torchrun` and verifying that all 8 GPUs are used.
- [ ] Add gradient checkpointing or other memory optimizations if needed when scaling to larger DINOv3 models.

### 7. Metrics and evaluation

- [ ] Implement a `compute_metrics` function for `Trainer` that:
  - [ ] Computes accuracy.
  - [ ] Computes precision, recall, F1 (with F1 as the main selection metric).
  - [ ] Computes AUROC.
  - [ ] Computes confusion matrix and returns TN, FP, FN, TP.
- [ ] Add support for saving best checkpoints according to F1.

### 8. Ablation study setup

- [ ] Define a set of initial ablation experiments in configs, for example:
  - [ ] Baseline: frozen DINOv3 + linear head, no PEFT, standard cross-entropy.
  - [ ] LoRA with small rank on q/v in the last N blocks.
  - [ ] LoRA with medium rank or more blocks.
  - [ ] Adapters with small bottleneck in the last N blocks.
  - [ ] Visual prompt tuning with a small number of prompt tokens.
- [ ] Keep the backbone, preprocessing, and core training hyperparameters fixed across these runs.
- [ ] Optionally create configs that vary imbalance handling (class weights vs focal loss vs none) while keeping PEFT fixed.

### 9. Future placeholders

- [ ] [Placeholder] Experiment with partially unfreezing upper DINOv3 blocks in combination with PEFT.
- [ ] [Placeholder] Add support for more PEFT methods as they become available for vision models.
- [ ] [Placeholder] Add automated hyperparameter search for a small set of critical parameters (e.g. LoRA rank, learning rate).

