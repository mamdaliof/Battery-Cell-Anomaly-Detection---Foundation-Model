# Battery Cell Anomaly Detection with Foundation Models

This repository explores **battery cell anomaly detection** using **DINOv3** vision transformers as frozen backbones, combined with **classifier heads** and **parameter-efficient fine-tuning (PEFT)** methods.

## Current design decisions

- **Data handling**
  - All image data is handled **locally** due to privacy constraints.
  - Raw data is expected in a detection-style format under a directory such as `split_base/` with `train/` and `val/` subfolders containing paired `*.png` and `*.xml` files.
  - A conversion script transforms this raw layout into a classification dataset under `data/`.
  - The `.gitignore` file is configured to ignore `data/`, `raw_data/`, `processed_data`, and other local artifacts (checkpoints, logs, wandb, etc.).

- **Backbone and preprocessing**
  - The backbone is a **DINOv3** model loaded from Hugging Face `transformers` (starting with a smaller variant, e.g. `facebook/dinov3-vitb16-pretrain-lvd1689m`).[web:3]
  - Preprocessing (resize, normalization, RGB handling) is delegated to the corresponding **DINOv3 image processor** from `transformers`. By default, the processor’s **native resolution and normalization** are used.[web:3]
  - The configuration includes an optional `image_size` placeholder so input resolution can be overridden later if necessary.

- **Augmentations**
  - Augmentations are applied **only on the training split**, on top of the DINOv3-compatible preprocessing.
  - Augmentations are configured via a YAML file and support:
    - A global probability (`aug_global_prob`) that any augmentation is applied.
    - A maximum number of transforms per image (`aug_max_transforms`).
    - Per-transform probabilities and parameters for:
      - Random resized crop (scale/ratio).
      - Horizontal flip.
      - Small rotation.
      - Color jitter (brightness, contrast, saturation, hue).
      - Gaussian noise.

- **Baseline model**
  - A generic `DinoV3Classifier` module lives in `src/bcadfm/models/dinov3_classifier.py`.
  - It loads a pretrained DINOv3 backbone via `AutoModel.from_pretrained`, freezes it by default, and attaches a configurable classification head.
  - The classification head depth is configurable through `HeadConfig`:
    - `depth = 1` → single linear layer.
    - `depth > 1` → multi-layer MLP with GELU activations and optional dropout.
  - The classifier expects `pixel_values` from the DINOv3 image processor and outputs logits (and, if labels are provided, a cross-entropy loss) suitable for use with `Trainer`.

- **Training configuration and orchestration**
  - Training is driven by a YAML configuration file (see `configs/baseline.yaml`) loaded into a `TrainingConfig` dataclass.
  - Configuration covers:
    - Model name and output directory.
    - Data settings (`DataConfig`), including paths and augmentations.
    - Classification head settings (`HeadConfig`).
    - Training hyperparameters: epochs, batch size, learning rate.
    - Early stopping and best model selection (`metric_for_best`, `greater_is_better`).
    - Learning rate scheduler (`lr_scheduler_type`, `warmup_ratio`).
    - Automatic mixed precision (`fp16`, `bf16`).
  - Each training run creates a unique run directory `outputs/{model_name}/{timestamp}/` and copies the used YAML config into that directory as `config.yaml` for reproducibility.

- **Metrics and objective**
  - Target task: binary classification (normal vs abnormal) on highly imbalanced battery cell data.
  - Primary optimization metric: **F1 score** (with emphasis on the abnormal class).
  - Additional metrics to compute and log (to be added via `compute_metrics`):
    - Accuracy.
    - Precision and recall.
    - AUROC.
    - Confusion matrix absolute counts: TN, FP, FN, TP.

- **Imbalance handling (to be implemented)**
  - Class-weighted cross-entropy based on label frequencies.
  - Focal loss as an alternative loss function.
  - Potential data-level strategies (e.g. oversampling) via custom samplers or collators.
  - All imbalance strategies will be toggled/configured via a central config file.

- **Training stack**
  - Use **Hugging Face Trainer** as the main training interface.
  - Rely on Trainer’s integration with **Accelerate** for multi-GPU and mixed-precision training (no custom training loop planned at this stage).[web:7]
  - Initial experiments use a **frozen DINOv3 backbone** plus a **configurable classification head** as the baseline (no PEFT).

- **PEFT plans (later stages)**
  - Integrate PEFT methods such as **LoRA**, **adapters**, and **visual prompt tuning** on top of the DINOv3 backbone.[web:12][web:15]
  - LoRA will likely target selected attention/MLP modules (e.g. q/v or full qkv) and possibly only the last few transformer blocks.
  - Visual prompt tuning will use learnable visual tokens prepended to patch embeddings.
  - All PEFT hyperparameters (type, ranks, target blocks, number of prompt tokens, etc.) will be defined in configs.

## Dataset conversion and usage

The raw dataset is assumed to live under a directory like `split_base/` with the following structure:

```text
split_base/
  train/
    c44_5.png
    c44_5.xml
    ...
  val/
    ...
```

Each `*.xml` file contains bounding box annotations for the corresponding image. Some object labels (e.g. `burnt`, `crack`, etc.) indicate **abnormal** cells.

To convert this detection-style dataset into a classification dataset compatible with the training pipeline, use the conversion script:

```bash
python scripts/convert_split_base_to_classification.py \
  --source-root /path/to/split_base \
  --target-root data \
  --abnormal-labels burnt crack \
  # --use-symlinks  # optional: use symlinks instead of copying
```

This will create a directory structure like:

```text
data/
  train/
    normal/
    abnormal/
  val/
    normal/
    abnormal/
```

where each image is assigned to `normal` or `abnormal` depending on whether any of its XML labels match the provided abnormal labels.[cite:26]

## Running training

With the classification dataset prepared under `data/` and a YAML config file defined (for example `configs/baseline.yaml`), run training with:

```bash
python scripts/train.py --config configs/baseline.yaml
```

This will:

- Load model, data, and training settings from the YAML file.
- Create a run directory under `outputs/{model_name}/{timestamp}/`.
- Copy the used config into that directory.
- Train `DinoV3Classifier` using `Trainer` with early stopping and `load_best_model_at_end=True`.

For DDP/multi-GPU, launch the same script with `torchrun` or `accelerate launch`.

## Project planning

A more detailed project specification and TODO list is maintained in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

As the project evolves, this README will be updated with setup instructions, usage examples, and experiment summaries.
