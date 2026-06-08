# 🔋 Battery Cell Anomaly Detection with Foundation Models

This repository explores **battery cell anomaly detection** using **DINOv3** vision transformers as frozen backbones, combined with **classifier heads** and **parameter-efficient fine-tuning (PEFT)** methods.

## 🎯 Current design decisions

- **📦 Data handling**
  - All image data is handled **locally** due to privacy constraints.
  - Raw data is expected in a detection-style format under a directory such as `split_base/` with `train/` and `val` subfolders containing paired `*.png` and `*.xml` files.
  - A conversion script transforms this raw layout into a classification dataset under `data/`.
  - The `.gitignore` file is configured to ignore `data/`, `raw_data/`, `processed_data`, and other local artifacts (checkpoints, logs, wandb, etc.).

- **🧠 Backbone and preprocessing**
  - The backbone is a **DINOv3** model loaded from Hugging Face `transformers` (starting with a smaller variant, e.g. `facebook/dinov3-vitb16-pretrain-lvd1689m`).
  - Preprocessing (resize, normalization, RGB handling) is delegated to the corresponding **image processor** from `transformers`. By default, the processor’s **native resolution and normalization** are used.
  - The configuration includes an optional `image_size` placeholder so input resolution can be overridden later if necessary.

- **🎨 Augmentations**
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

- **🧱 Baseline model**
  - A generic classifier module lives in `src/bcadfm/models/dinov3_classifier.py` (for DINOv3 backbones) and can also be instantiated with other ViT-style backbones such as `google/vit-base-patch16-224`.
  - It loads a pretrained backbone via `AutoModel.from_pretrained`, freezes it by default, and attaches a configurable classification head.
  - The classification head depth is configurable through `HeadConfig`:
    - `depth = 1` → single linear layer.
    - `depth > 1` → multi-layer MLP with GELU activations and optional dropout.
    - `hidden_dim` can be defined as an absolute size (e.g. `256`), a multiplier float (e.g. `0.5` times DINOv3's dimension), or a multiplier string (e.g. `"1.1X"`).
  - The classifier expects `pixel_values` from the corresponding image processor and outputs logits (and, if labels are provided, a cross-entropy loss) suitable for use with `Trainer`.

- **🚀 Training configuration and orchestration**
  - Training is driven by a YAML configuration file (see `configs/baseline.yaml`) loaded into a `TrainingConfig` dataclass.
  - Configuration covers:
    - Model name and output directory.
    - Data settings (`DataConfig`), including paths and augmentations.
    - Classification head settings (`HeadConfig`).
    - Training hyperparameters: epochs, batch size, learning rate.
    - Best model selection via a custom callback that saves both `best_loss.pt` and `best_f1.pt`.
    - Learning rate scheduler (`lr_scheduler_type`, `warmup_ratio`).
    - Automatic mixed precision (`fp16`, `bf16`) when GPUs are available.
    - **Global Seed**: A global configuration seed (`seed`) that governs PyTorch, CUDA, NumPy, and Python's random state for reproducible training runs, data oversampling, and augmentation selections.
  - **Data loading performance**: `dataloader_num_workers=4` and `pin_memory=True` are enabled for efficient GPU data feeding.
  - **Efficient augmentations**: Gaussian noise augmentation uses direct **NumPy array injection**, avoiding costly PIL↔Tensor round-trips.
  - **Deferred checkpointing**: `state_dict()` is only called inside callbacks when an actual metric improvement is detected, reducing unnecessary I/O overhead.
  - **Evaluation & saving**: Evaluation, checkpointing, and logging are performed **per epoch** (`eval_strategy='epoch'`, `save_strategy='epoch'`, and `logging_strategy='epoch'`) to ensure consistent, stable checkpoint evaluation.
  - Each training run creates a unique run directory `outputs/{task_name}__{safe_model_name}__{cfg_stem}/{timestamp}/` to prevent concurrent folder write collisions when running parallel GPU runs, and copies the used YAML config into that directory as `config.yaml` for reproducibility.

- **📊 Metrics and objective**
  - Target task: binary classification (normal vs abnormal) on highly imbalanced battery cell data.
  - Primary optimization metric: **F1 score** (with emphasis on the abnormal class).
  - Metrics are computed via a custom `compute_cls_metrics` function and include:
    - Accuracy.
    - Precision and recall.
    - F1.
    - AUROC.
    - Confusion matrix absolute counts: TN, FP, FN, TP.

- **⚖️ Imbalance handling (Implemented)**
  - Class-weighted cross-entropy based on label frequencies.
  - Focal loss incorporating gamma and alpha coefficients.
  - Dynamic dataset-level minority class oversampling (fully compatible with multi-GPU DDP training).
  - Sampler-level oversampling (`WeightedRandomSampler`).
  - Configured and activated through the central `imbalance` configuration section.

- **⚡ PEFT Integration (Implemented)**
  - **LoRA**: Parameter-efficient fine-tuning via Hugging Face `peft` targeting specific attention projections (`q_proj`, `v_proj`).
  - **Bottleneck Adapters**: Pfeiffer-style bottleneck adapters wrapping transformer feed-forward blocks.
  - **Visual Prompt Tuning (VPT)**: Support for Shallow (input-level prompt parameters) and Deep (layer-wise prompt replacement wrappers) prompt tuning.
  - Supports dynamic model structure routing (handles standard `encoder.layer`/`layers` and DINOv3's `model.layer` format).
  - VPT (Visual Prompt Tuning) includes a sequential block execution fallback for architectures without a nested `encoder` module (such as DINOv3).

- **🖥️ Parallel Training Dashboard**
  - `run_parallel_ablations.py` manages a job queue distributed across **8 GPUs**.
  - Uses **threading** to parse subprocess stdout in real-time.
  - Fixed **8-line in-place ANSI terminal display** (no scrolling) for live monitoring.
  - Shows: GPU ID, config name, epoch progress, tqdm bar, loss, F1, and completion status.
  - Full output per config is logged to `outputs/logs/<config_name>.log`.

- **🎯 YOLO26 + DINOv3 SFP Object Detection**
  - Integrated the frozen DINOv3 vision backbone and a **Simple Feature Pyramid (SFP)** neck with standard **Ultralytics YOLO26** object detection head and native losses.
  - Custom layers (`DinoV3Backbone` with ImageNet normalization, `DinoV3SFP_P3/P4/P5` projections) are implemented in [yolo_dino.py](file:///home/mamdaliof/Documents/GitHub/mamdaliof-obsidian/02-Projects/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/models/yolo_dino.py).
  - Dynamic class registration and tasks parser wrapping (supporting width/depth channel scaling and metadata attribute preservation) is managed in [yolo_utils.py](file:///home/mamdaliof/Documents/GitHub/mamdaliof-obsidian/02-Projects/Battery-Cell-Anomaly-Detection---Foundation-Model/src/bcadfm/utils/yolo_utils.py).
  - Configured via [yolo26_dino.yaml](file:///home/mamdaliof/Documents/GitHub/mamdaliof-obsidian/02-Projects/Battery-Cell-Anomaly-Detection---Foundation-Model/configs/yolo26_dino.yaml).
  - Fully verified and tested via shapes unit test suite [test_yolo_shapes.py](file:///home/mamdaliof/Documents/GitHub/mamdaliof-obsidian/02-Projects/Battery-Cell-Anomaly-Detection---Foundation-Model/tests/test_yolo_shapes.py).

- **💾 Local Model Caching**
  - Hugging Face cache is redirected to the workspace folder `models/hf_cache` via dynamic environment injection (`os.environ["HF_HOME"]`).
  - This prevents redundant model downloads from the internet during validation, unit testing, and parallel training.
  - The local `models/` directory is registered in `.gitignore` to prevent large model weight binaries from being tracked in the repository.


## 📂 Dataset conversion and usage

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

### 🎯 Object Detection Dataset Conversion

To convert the detection-style `split_base/` dataset into a YOLO format dataset compatible with the object detection pipeline:

```bash
python scripts/prepare_yolo_detection_data.py \
  --source-root /path/to/split_base \
  --target-root data/battery_detection \
  --detection-labels abnormality \
  # --use-symlinks  # optional: use symlinks instead of copying images
```

This generates a structured output under `data/battery_detection/` containing `images/train/`, `images/val/`, `labels/train/`, and `labels/val/` directories, with absolute bounding boxes converted to normalized `[0, 1]` YOLO coordinate format (`class_idx x_center y_center width height`).

## 💻 How to run the code

### 1. ⚙️ Create and activate the environment

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

### 2. 🔄 Prepare the dataset

Convert the detection-style dataset into the classification layout under `data/`:

```bash
python scripts/convert_split_base_to_classification.py \
  --source-root /path/to/split_base \
  --target-root data \
  --abnormal-labels burnt crack
```

### 3. 🧪 Single-process (CPU or 1 GPU) smoke test

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

### 4. 🔗 Distributed multi-GPU training with torchrun (DDP)

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

### 5. 🧪 Ablation Study (58 Configs)

A comprehensive ablation study framework automates exploration of backbone, PEFT method, and hyperparameter combinations.

#### 📐 Grid generation

`scripts/generate_ablation_grid.py` generates **58 YAML configs** under `configs/ablations/` covering a combinatorial grid of:

| Axis | Variants |
|------|----------|
| **Backbones** | ViT-S/16 (`dinov3-vits16-pretrain-lvd1689m`), ViT-B/16 (`dinov3-vitb16-pretrain-lvd1689m`) |
| **PEFT methods** | Frozen baseline (no PEFT), LoRA (ranks 8/16 × all/last-4/last-2 layers), Bottleneck Adapters (dims 32/64 × all/last-4/last-2), VPT Shallow (8/16/32 tokens), VPT Deep (8/16/32 tokens × all/last-4/last-2) |
| **Learning rates** | 3e-4, 5e-4 |

#### ✅ Config validation

`scripts/validate_ablation_configs.py` validates that **all 58 configs** (plus template configs, 67 total) can successfully load the model + processor without errors.

#### 🚀 Parallel execution

`scripts/run_parallel_ablations.py` distributes training across **8 GPUs** with a real-time in-place terminal dashboard showing per-GPU status, progress bars, and metrics.

```bash
# Generate ablation configs
python scripts/generate_ablation_grid.py

# Validate all configs
python scripts/validate_ablation_configs.py

# Launch parallel training on 8 GPUs
python scripts/run_parallel_ablations.py
```

### 📊 Results Visualization Suite

To analyze and compare results from completed and in-progress ablation runs, we provide two interactive tools:

1. **Streamlit Dashboard** ([`visualize.py`](file:///home/mamdaliof/Documents/GitHub/mamdaliof-obsidian/02-Projects/Battery-Cell-Anomaly-Detection---Foundation-Model/visualize.py)):
   A feature-rich web dashboard containing an F1-prioritized leaderboard, confusion matrix heatmaps (TP/FP/TN/FN) computed at the best epoch, multi-run trajectory line plot comparisons, and PEFT parameter analysis charts.
   Launch from the root folder:
   ```bash
   streamlit run visualize.py
   ```
2. **Jupyter Notebook Analyzer** ([`notebooks/visualize_results.ipynb`](file:///home/mamdaliof/Documents/GitHub/mamdaliof-obsidian/02-Projects/Battery-Cell-Anomaly-Detection---Foundation-Model/notebooks/visualize_results.ipynb)):
   An interactive notebook with widgets and Plotly diagrams for quick local results exploration.

### 🎯 YOLO26 + DINOv3 SFP Object Detection Verification

To verify that the custom registered DINOv3 backbone, SFP neck, and the Ultralytics tasks parser work correctly, run the shapes unit test suite:
```bash
python3 tests/test_yolo_shapes.py
```
This script compiles the YOLO26 + DINOv3 model, runs a dummy forward pass, and verifies that the output prediction tensor shape matches expected resolutions.

### 🔬 GPU VRAM & DDP Isolation Verification

Two lightweight helper scripts are available to verify GPU visibility, isolation, and process group setups without loading datasets or full models:
1. **Single-GPU Isolation**: Allocates 4 GB of VRAM on local `cuda:0` of the visible device:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python3 scripts/gpu_alloc_test.py --duration 60
   ```
2. **Multi-GPU DDP NCCL Group**: Initializes NCCL distributed group and allocates 4 GB on each participating device. Specify `--master_port` for parallel launches to avoid port conflict:
   ```bash
   CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 --master_port=29501 scripts/ddp_alloc_test.py
   ```

## 📅 Project planning

A more detailed project specification and TODO list is maintained in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

Additional documentation resources:

- 📓 **Dev logs**: Detailed development logs are maintained in the [`devlogs/`](./devlogs/) directory.
  - [`devlogs/DEVLOG_YOLO_DINO_DETECTION_CUSTOM_METRICS.md`](./devlogs/DEVLOG_YOLO_DINO_DETECTION_CUSTOM_METRICS.md) (Custom evaluation & multi-label classification validation metrics)
  - [`devlogs/DEVLOG_YOLO_DINO_DETECTION_DATA_PREP.md`](./devlogs/DEVLOG_YOLO_DINO_DETECTION_DATA_PREP.md) (YOLO dataset preparation and cleaning for object detection)
  - [`devlogs/DEVLOG_YOLO_DINO_DETECTION_INTEGRATION.md`](./devlogs/DEVLOG_YOLO_DINO_DETECTION_INTEGRATION.md) (YOLO26 + DINOv3 object detection integration)
  - [`devlogs/DEVLOG_VPT_FIX_AND_COLLISION_RESOLUTION.md`](./devlogs/DEVLOG_VPT_FIX_AND_COLLISION_RESOLUTION.md) (VPT compatibility & run directory collision fix)
  - [`devlogs/DEVLOG_RESULTS_VISUALIZATION_SUITE.md`](./devlogs/DEVLOG_RESULTS_VISUALIZATION_SUITE.md) (Interactive Jupyter & Streamlit results visualization suite)
  - [`devlogs/DEVLOG_LOCAL_MODEL_CACHING.md`](./devlogs/DEVLOG_LOCAL_MODEL_CACHING.md) (Local model caching and offline setup)
- 📘 **Technical reference**: In-depth implementation details are documented in [`docs/technical_details.md`](./docs/technical_details.md).
- 📊 **PEFT & imbalance report**: Integration analysis and results are captured in [`PEFT_IMBALANCE_REPORT.md`](./PEFT_IMBALANCE_REPORT.md).


As the project evolves, this README will be updated with setup instructions, usage examples, and experiment summaries.

