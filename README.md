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
  - Preprocessing (resize, normalization, RGB handling) is delegated to the corresponding **image processor** from `transformers`. By default, the processor's **native resolution and normalization** are used.
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
  - Each training run creates a unique run directory under `outputs/cls/` or `outputs/det/` (e.g. `outputs/cls/{safe_model_name}__{cfg_stem}/{timestamp}/`) to prevent concurrent folder write collisions when running parallel GPU runs, and copies the used YAML config into that directory as `config.yaml` for reproducibility.

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
  - Parameter-efficient fine-tuning is supported for both **classification** (MLP head) and **object detection** (YOLO26 detector).
  - **LoRA**: Targets specific attention projections (`q_proj`, `v_proj`). LoRA is applied to all blocks via `get_peft_model`, then `lora_*` weights in non-target blocks are frozen using `requires_grad = False` — replacing the previous fragile `layers_to_transform`/`layers_pattern` PEFT mechanism.
  - **Bottleneck Adapters**: Pfeiffer-style bottleneck adapters wrapping transformer feed-forward blocks.
  - **Visual Prompt Tuning (VPT)**: Support for Shallow (input-level prompt parameters) and Deep (layer-wise prompt replacement wrappers) prompt tuning.
  - Supports dynamic model structure routing via a new `_get_transformer_blocks()` helper that probes common attribute paths (`model.encoder.layer`, `model.layer`, `encoder.layer`, `layers`, `layer`) in order — works correctly across DINOv3, standard ViT, and other transformer variants.
  - VPT (Visual Prompt Tuning) includes a sequential block execution fallback for architectures without a nested `encoder` module (such as DINOv3).
  - The SFP (Simple Feature Pyramid) neck and the backbone gradients are automatically routed to permit training the PEFT layers when active, adjusting feature extraction slice indices to account for prepended prompt tokens.

- **🖥️ Parallel Training Dashboard**
  - `run_parallel_ablations.py` manages a job queue distributed across **8 GPUs**.
  - Uses **threading** to parse subprocess stdout in real-time.
  - Fixed **8-line in-place ANSI terminal display** (no scrolling) for live monitoring.
  - Shows: GPU ID, config name, epoch progress, tqdm bar, loss, F1, and completion status.
  - Full output per config is logged to `outputs/logs/<config_name>.log`.

- **🎯 YOLO26 + DINOv3 SFP Object Detection**
  - Integrated the DINOv3 vision backbone and a **Simple Feature Pyramid (SFP)** neck with standard **Ultralytics YOLO26** object detection head and native losses.
  - Supports both **fully frozen backbone** and **Parameter-Efficient Fine-Tuning (PEFT)** via LoRA, Pfeiffer Bottleneck Adapters, and Visual Prompt Tuning (VPT).
  - Custom layers and PEFT configurations are implemented in `src/bcadfm/models/yolo_dino.py`.
  - Dynamic class registration and tasks parser wrapping (supporting width/depth channel scaling and metadata attribute preservation) is managed in `src/bcadfm/utils/yolo_utils.py`.
  - Configured via `configs/yolo26_dino.yaml`.
  - **YAML-driven Augmentation Mapping**: Augmentation parameters in config YAML files are fully mapped to YOLO overrides. Supports passing a custom `yolo_augmentations` dict or automatically mapping standard classification equivalents (or disabling them if `augmentations_enabled` is false).
  - **Class Names Mapping Alignment**: Resolves configured `normal` / `abnormal` class names dynamically inside the detection pipeline, synchronizing metric logging keys and visualizer tab displays.
  - Fully verified and tested via shapes unit test suite `tests/test_yolo_shapes.py`.

- **💾 Local Model Caching**
  - Hugging Face cache is redirected to the workspace folder `models/hf_cache` via dynamic environment injection (`os.environ["HF_HOME"]`).
  - This prevents redundant model downloads from the internet during validation, unit testing, and parallel training.
  - The local `models/` directory is registered in `.gitignore` to prevent large model weight binaries from being tracked in the repository.


- **🖥️ Streamlit Visualization Dashboard (Enhanced)**
  - An interactive dashboard (`visualize.py`) loads and parses classification and detection training results uniformly.
  - **Leaderboard**: Compares runs across backbones, PEFT configurations, and training parameters, ranked by image-level anomaly classification F1.
  - **Trajectory Curves**: Plots training/validation loss, learning rates, standard detection (mAP50, mAP50-95), and custom metrics (IoU, Dice, image-level F1).
  - **Single Run Inspector**: Shows config properties, bbox details, per-class metrics, and interactive confusion matrices.
  - **Comparison Tab**: Compares the best classification model directly with the best detection model on image-level anomaly classification, complete with side-by-side confusion matrices and performance metrics.
  - Driven by the unified `trainer_state.json` file generated at the end of each training epoch.

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
  --input_dir split_base \
  --output_dir data
```

This creates `data/train/normal/`, `data/train/abnormal/`, `data/val/normal/`, and `data/val/abnormal/` directories.

## 🏋️ Training

### Single GPU
```bash
python scripts/train.py --config configs/baseline.yaml
```

### Multi-GPU (DDP via torchrun)
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/baseline.yaml
```

### Parallel Ablations (8-GPU dashboard)
```bash
python run_parallel_ablations.py
```

## 🔬 PEFT Training Example
```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/peft_smoke.yaml
```

## 🧪 Testing & Verification
```bash
# PEFT integration verification
python tests/verify_peft.py

# YOLO26+DINOv3 shape tests
python -m pytest tests/test_yolo_shapes.py

# Ablation config validation
python scripts/validate_ablation_configs.py

# Launch visualizer dashboard
streamlit run visualize.py --server.port 8501
```

## 🛠️ Development Logs

The evolution and detailed implementation steps of the codebase are recorded in the developer logs:
- [DEVLOG_YOLO_AUGMENTATION_AND_CLASS_SYNC.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_YOLO_AUGMENTATION_AND_CLASS_SYNC.md): Connects YOLO augmentations to YAML configuration, maps classification default parameters, and aligns class names across dashboards and metrics validator.
- [DEVLOG_YOLO_DINO_PEFT_INTEGRATION_AND_REORGANIZATION.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_YOLO_DINO_PEFT_INTEGRATION_AND_REORGANIZATION.md): Integrates PEFT inside the YOLO detection model, config/output folder reorganization, and gradient routing.
- [DEVLOG_LORA_BLOCK_TARGETING_FIX.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_LORA_BLOCK_TARGETING_FIX.md): Rewrote LoRA block-targeting using an architecture-agnostic freezing strategy.
- [DEVLOG_VPT_FIX_AND_COLLISION_RESOLUTION.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_VPT_FIX_AND_COLLISION_RESOLUTION.md): Fixed VPT for DINOv3 architectures and resolved file collisions.
- [DEVLOG_UNIFIED_METRICS_AND_VISUALIZER.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_UNIFIED_METRICS_AND_VISUALIZER.md): Integrated class indicators and confusion matrices, unified training state logging.
- [DEVLOG_RESULTS_VISUALIZATION_SUITE.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_RESULTS_VISUALIZATION_SUITE.md): Enhanced Streamlit leaderboard, run comparing, and curve plots.
- [DEVLOG_YOLO_DINO_DETECTION_CUSTOM_METRICS.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_YOLO_DINO_DETECTION_CUSTOM_METRICS.md): Bbox IoU, Dice coefficients, and image-level metrics.
- [DEVLOG_YOLO_DINO_DETECTION_PEFT_FINE_TUNING.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_YOLO_DINO_DETECTION_PEFT_FINE_TUNING.md): Details of tuning adapter blocks in object detection.
- [DEVLOG_YOLO_DINO_DETECTION_INTEGRATION.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_YOLO_DINO_DETECTION_INTEGRATION.md): Integration of YOLOv8 loss targets and DINOv3 backbone SFP neck.
- [DEVLOG_YOLO_DINO_DETECTION_DATA_PREP.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_YOLO_DINO_DETECTION_DATA_PREP.md): Custom detection collate functions and shapes tests.
- [DEVLOG_GLOBAL_SEED_AND_AUDIT_FIXES.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_GLOBAL_SEED_AND_AUDIT_FIXES.md): Global seeds, early stopping callbacks, and dataloader optimization.
- [DEVLOG_LOCAL_MODEL_CACHING.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_LOCAL_MODEL_CACHING.md): Hugging Face local cache redirection.
- [DEVLOG_PEFT_IMBALANCE_INTEGRATION.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_PEFT_IMBALANCE_INTEGRATION.md): Imbalance handling and initial classification PEFT hooks.
- [DEVLOG_ABLATION_STUDY_AND_OPTIMIZATION.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_ABLATION_STUDY_AND_OPTIMIZATION.md): Parallel run schedulers and GPU allocation.
- [DEVLOG_FIRST_SERVER_SMOKE_TEST.md](file:///home/jovyan/Battery-Cell-Anomaly-Detection---Foundation-Model/devlogs/DEVLOG_FIRST_SERVER_SMOKE_TEST.md): Initial server setup and DINOv3 classifier verify scripts.

## 📁 Repository Layout

```text
├── configs/              # YAML experiment configs
├── devlogs/              # Development logs per feature/session
├── docs/                 # Extended technical documentation
├── scripts/              # Training, conversion, and validation scripts
├── src/bcadfm/
│   ├── data/             # Dataset loading and augmentation
│   ├── models/           # DinoV3Classifier, YOLO+DINOv3 models
│   ├── training/         # ImbalanceTrainer, losses, callbacks
│   └── utils/            # Config schemas, model_utils, yolo_utils
├── tests/                # Unit and integration tests
├── outputs/              # Training run outputs (gitignored)
└── models/               # HF model cache (gitignored)
```
