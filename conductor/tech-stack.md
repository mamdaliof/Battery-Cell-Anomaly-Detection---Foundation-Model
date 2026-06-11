# Tech Stack - Battery Cell Anomaly Detection - Foundation Model

## Programming Language
- **Python**: Core programming language (typically 3.10+ / 3.11).

## Deep Learning & Model Frameworks
- **PyTorch**: Deep learning framework (`torch==2.4.0`, `torchvision==0.19.0`).
- **Hugging Face Transformers**: Model loading, feature extraction, and pre-processing pipeline for foundation models (`transformers>=5.10.0`).
- **Hugging Face PEFT**: Parameter-efficient fine-tuning wrapper supporting LoRA, Bottleneck Adapters, and Visual Prompt Tuning (`peft>=0.11.0`).
- **Hugging Face Accelerate**: Distributed training wrapper for multi-GPU configurations (`accelerate>=0.30.0`).
- **Ultralytics**: Object detection pipeline containing YOLOv8/YOLO11 and YOLO26 structures.

## Utilities & Metrics
- **scikit-learn**: Evaluation metric computations (accuracy, precision, recall, F1, AUROC, confusion matrices).
- **Pillow**: Image processing and transformations.
- **matplotlib & plotly**: Visual analysis of performance, training loss, and metrics.
- **streamlit**: Interactive dashboards and visualization demos.

## Infrastructure & Tooling

### 1. Centralized Local Model Caching
- **Directory**: `models/hf_cache` at the project's root.
- **Mechanism**: Dynamic environment injection (`os.environ["HF_HOME"] = "models/hf_cache"`) during package initialization (`src/bcadfm/__init__.py`) and entry point scripts.
- **Purpose**: Enables offline training, prevents Hugging Face Hub API rate-limiting, and ensures zero redundant downloads during parallel sweeps.
- **Git Strategy**: The `models/` directory is registered in `.gitignore` to prevent tracking large weight binaries in the repository.

### 2. Multi-GPU Concurrency Runner (`run_parallel_ablations.py`)
- **Capacity**: Manages a training queue distributed across up to 8 GPUs concurrently.
- **Architecture**: Isolated subprocess spawning with `CUDA_VISIBLE_DEVICES` environment variable routing, synchronized using Python's `threading` modules.
- **Status Dashboard**: Multi-threaded regex scanner parsing outputs dynamically to render an in-place terminal status table (ANSI controls).

### 3. Streamlit Web Dashboard (`visualize.py`)
- **Features**: Interactive training leaderboard, trajectory curve overlay, validation confusion matrix inspector, and parallel coordinates plot correlating hyperparameters (rank, LR, bottleneck dims) with F1.
