# Battery Cell Anomaly Detection with Foundation Models

This repository explores **battery cell anomaly detection** using **DINOv3** vision transformers as frozen backbones, combined with **classifier heads** and **parameter-efficient fine-tuning (PEFT)** methods.

## Current design decisions

- **Data handling**
  - All image data is handled **locally** due to privacy constraints.
  - Raw data is expected in a detection-style format under a directory such as `split_base/` with `train/` and `val` subfolders containing paired `*.png` and `*.xml` files.
  - A conversion script transforms this raw layout into a classification dataset under `data/`.
  - The `.gitignore` file is configured to ignore `data/`, `raw_data/`, `processed_data`, and other local artifacts (checkpoints, logs, wandb, etc.).

- **Backbone and preprocessing**
  - The backbone is a **DINOv3** model loaded from Hugging Face `transformers` (starting with a smaller variant, e.g. `facebook/dinov3-vitb16-pretrain-lvd1689m`).
  - Preprocessing (resize, normalization, RGB handling) is delegated to the corresponding **image processor** from `transformers`. By default, the processor’s **native resolution and normalization** are used.
  - The configuration includes an optional `image_size` placeholder so input resolution can be overridden later if necessary.

- **Augmentations**
  - Augmentations are applied **only on the training split**, on top of the backbone-compatible preprocessing.
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
  - A generic classifier module lives in `src/bcadfm/models/dinov3_classifier.py` (for DINOv3 backbones) and can also be instantiated with other ViT-style backbones such as `google/vit-base-patch16-224`.
  - It loads a pretrained backbone via `AutoModel.from_pretrained`, freezes it by default, and attaches a configurable classification head.
  - The classification head depth is configurable through `HeadConfig`:
    - `depth = 1` → single linear layer.
    - `depth > 1` → multi-layer MLP with GELU activations and optional dropout.
  - The classifier expects `pixel_values` from the corresponding image processor and outputs logits (and, if labels are provided, a cross-entropy loss) suitable for use with `Trainer`.

- **Training configuration and orchestration**
  - Training is driven by a YAML configuration file (see `configs/baseline.yaml`) loaded into a `TrainingConfig` dataclass.
  - Configuration covers:
    - Model name and output directory.
    - Data settings (`DataConfig`), including paths and augmentations.
    - Classification head settings (`HeadConfig`).
    - Training hyperparameters: epochs, batch size, learning rate.
    - Best model selection via a custom callback that saves both `best_loss.pt` and `best_f1.pt`.
    - Learning rate scheduler (`lr_scheduler_type`, `warmup_steps`).
    - Automatic mixed precision (`fp16`, `bf16`) when GPUs are available.
  - Each training run creates a unique run directory `outputs/{task_name}__{model_name}/{timestamp}/` and copies the used YAML config into that directory as `config.yaml` for reproducibility.

- **Metrics and objective**
  - Target task: binary classification (normal vs abnormal) on highly imbalanced battery cell data.
  - Primary optimization metric: **F1 score** (with emphasis on the abnormal class).
  - Metrics are computed via a custom `compute_cls_metrics` function and include:
    - Accuracy.
    - Precision and recall.
    - F1.
    - AUROC.
    - Confusion matrix absolute counts: TN, FP, FN, TP.

- **Imbalance handling (to be implemented)**
  - Class-weighted cross-entropy based on label frequencies.
  - Focal loss as an alternative loss function.
  - Potential data-level strategies (e.g. oversampling) via custom samplers or collators.
  - All imbalance strategies will be toggled/configured via a central config file.

- **Training stack**
  - Use **Hugging Face Trainer** as the main training interface.
  - Use `torchrun` for multi-GPU training with DistributedDataParallel (DDP); no custom low-level DDP loop is required.
  - Initial experiments use a **frozen DINOv3 or ViT backbone** plus a **configurable classification head** as the baseline (no PEFT).

- **PEFT plans (later stages)**
  - Integrate PEFT methods such as **LoRA**, **adapters**, and **visual prompt tuning** on top of the backbone.
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

## How to run the code

### 1. Create and activate the environment

Create a Python 3.10+ environment and install dependencies, for example with `pip`:

```bash
python -m venv pytorch_env
source pytorch_env/bin/activate

pip install -r requirements.txt

# Install a CUDA-enabled PyTorch build compatible with your driver
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You can verify CUDA visibility with:

```bash
python - << 'EOF'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
EOF
```

### 2. Prepare the dataset

Convert the detection-style dataset into the classification layout under `data/`:

```bash
python scripts/convert_split_base_to_classification.py \
  --source-root /path/to/split_base \
  --target-root data \
  --abnormal-labels burnt crack
```

### 3. Single-process (CPU or 1 GPU) smoke test

From the repository root:

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Option A: force CPU
CUDA_VISIBLE_DEVICES= \
python scripts/train.py --config configs/test_smoke.yaml

# Option B: use a single GPU (e.g. GPU 0)
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py --config configs/test_smoke.yaml
```

This will run a short, one-epoch training and create an output directory under:

```text
outputs/cls__{model_name}/{timestamp}/
```

containing `config.yaml`, checkpoints, and the two best-model snapshots `best_loss.pt` and `best_f1.pt`.

### 4. Multi-GPU training with torchrun (DDP)

If you have 2 GPUs available (for example on a machine with two NVIDIA A10s), you can launch distributed training via `torchrun`:

```bash
cd /path/to/Battery-Cell-Anomaly-Detection---Foundation-Model
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

# Quick multi-GPU smoke test
torchrun --nproc_per_node=2 scripts/train.py --config configs/test_smoke.yaml

# Full baseline training
torchrun --nproc_per_node=2 scripts/train.py --config configs/baseline.yaml
```

Notes:

- `nproc_per_node` should match the number of GPUs you are using on that node.
- The `batch_size` in the YAML config is **per GPU**. For example, `batch_size: 64` with 2 GPUs results in an effective global batch size of 128.
- Training metrics and model checkpoints are written to the run directory under `outputs/` as described above.

## Project planning

A more detailed project specification and TODO list is maintained in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

As the project evolves, this README will be updated with setup instructions, usage examples, and experiment summaries.
