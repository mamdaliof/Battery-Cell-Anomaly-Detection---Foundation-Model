# Battery Cell Anomaly Detection with Foundation Models

This repository explores **battery cell anomaly detection** using **DINOv3** vision transformers as frozen backbones, combined with **classifier heads** and **parameter-efficient fine-tuning (PEFT)** methods.

## Current design decisions

- **Data handling**
  - All image data is handled **locally** due to privacy constraints.
  - Raw data is expected in a detection-style format under a directory such as `split_base/` with `train/` and `val/` subfolders containing paired `*.png` and `*.xml` files.
  - A conversion script transforms this raw layout into a classification dataset under `data/`.
  - The `.gitignore` file is configured to ignore `data/`, `raw_data/`, `processed_data/`, and other local artifacts (checkpoints, logs, wandb, etc.).

- **Backbone and preprocessing**
  - The backbone is a **DINOv3** model loaded from Hugging Face `transformers` (starting with a smaller variant, e.g. `facebook/dinov3-vitb16-pretrain-lvd1689m`).
  - Preprocessing (resize, normalization, RGB handling) is delegated to the corresponding **DINOv3 image processor** from `transformers`. By default, the processor’s **native resolution and normalization** are used.[web:3]
  - The configuration will include an optional `image_size` placeholder so input resolution can be overridden later if necessary.

- **Augmentations**
  - Augmentations are applied **only on the training split**, on top of the DINOv3-compatible preprocessing.
  - Planned augmentations (all to be controlled via config):
    - Horizontal flip.
    - Small random rotation.
    - Slight random resize / random resized crop.
    - Color jitter (brightness, contrast, saturation, hue).
    - HSV-like adjustment (implemented via color jitter or custom transform).
    - Light Gaussian noise.

- **Metrics and objective**
  - Target task: binary classification (normal vs abnormal) on highly imbalanced battery cell data.
  - Primary optimization metric: **F1 score** (with emphasis on the abnormal class).
  - Additional metrics to compute and log:
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
  - Rely on Trainer’s integration with **Accelerate** for multi-GPU and mixed-precision training (no custom training loop planned at this stage).
  - Initial experiments will use a **frozen DINOv3 backbone** plus a **trainable linear classification head** as the baseline (no PEFT).

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

where each image is assigned to `normal` or `abnormal` depending on whether any of its XML labels match the provided abnormal labels.

## Project planning

A more detailed project specification and TODO list is maintained in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

As the project evolves, this README will be updated with setup instructions, usage examples, and experiment summaries.
