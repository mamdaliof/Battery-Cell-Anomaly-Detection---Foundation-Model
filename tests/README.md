# 🧪 Battery Cell Anomaly Detection: Unit Testing Suite & Test Benches

This directory contains a comprehensive suite of unit tests and validation test benches designed to ensure the stability, correctness, and DDP-safety of all model components, datasets, custom loss functions, evaluation metrics, and end-to-end processing pipelines.

---

## 🏛️ Test Benches Overview

### 1. `test_env.py` (Environment & Diagnostics Check)
- **Why We Have It**: Verifies that the host server environment has correct PyTorch, CUDA, and third-party dependencies (such as `transformers`, `peft`, `ultralytics`, and `accelerate`) configured before running resource-heavy sweeps.
- **How It Should Behave**: Checks for active library versions, lists available CUDA devices, prints CUDA names, and validates that local `bcadfm` source modules can import successfully.

### 2. `test_dataset.py` (Dataset & Data Augmentations)
- **Why We Have It**: Guarantees that data loading, parsing, and class sorting operate correctly under extreme imbalance, and that data-level oversampling is reproducible.
- **How It Should Behave**: Builds a temporary dataset containing normal and abnormal samples, verifies that binary classification mapping matches targets, checks that `RandomAugmentationCombo` samples unique operations without replacement (no duplicates), and validates that oversampling shuffles and replicates deterministically when using identical seeds.

### 3. `test_models.py` (DINOv3 Classifier & PEFT Modules)
- **Why We Have It**: Validates that frozen backbones, classification heads, bottleneck adapters, and visual prompts wrap correctly without parameter leaks.
- **How It Should Behave**: Loads a DINOv3 model from Hugging Face (`facebook/dinov3-vits16-pretrain-lvd1689m`), verifies that classification heads parse hidden dim multipliers (e.g., `"0.5X"`), verifies that `BottleneckAdapter` maps to identity at training step 0, and checks that Visual Prompt Tuning (VPT) layouts structure tokens safely to match the DINOv3 registers slicing.

### 4. `test_yolo_shapes.py` (YOLO26 + DINOv3 Object Detection)
- **Why We Have It**: Verifies that the custom DINOv3 backbone and Simple Feature Pyramid (SFP) neck register correctly with the Ultralytics parser and output expected bounding box shapes.
- **How It Should Behave**: Verifies that Custom modules preserve vital parser metadata (`i`, `f`, `type`, `np`) when swapping layers, checks that dummy forward passes compile with dynamic resolution position embedding interpolation, and verifies that spatial feature maps project down to P3, P4, and P5 grids.

### 5. `test_trainer.py` (Custom Trainer & Losses)
- **Why We Have It**: Validates that custom training modifications, device alignments, label scanning, and loss scaling compute correctly under DDP partitioning.
- **How It Should Behave**: Verifies that training label scanning unwraps PyTorch Subsets to locate ground-truth samples, tests that FocalLoss scales mathematically with gamma/alpha parameters, checks that `compute_loss` pops labels before model forwarding (preventing redundant double loss builds), and verifies that `WeightedRandomSampler` fallback switches to DDP-safe `data_level` oversampling under multi-GPU runs.

### 6. `test_metrics.py` (Validation Metrics & Callbacks)
- **Why We Have It**: Validates classification statistics (F1, AUROC, Confusion Matrix) and checkpoint saving operations.
- **How It Should Behave**: Verifies classification score calculations under standard conditions and tests that single-class splits (e.g. all normal) handle division-by-zero boundaries without crashing. Verifies that `SaveBestModelCallback` tracks metrics improvements correctly.

### 7. `test_utils.py` (Configurations & Helper Utilities)
- **Why We Have It**: Validates config loading and parameter counting.
- **How It Should Behave**: Checks that default config parameters and the global seed parsing resolve correctly, counts model parameters, and registers custom modules in the Ultralytics namespace.

### 8. `test_pipelines.py` (End-to-End Pipeline Scripts)
- **Why We Have It**: Verifies command-line utilities.
- **How It Should Behave**: Simulates XML detection parsing in `convert_split_base_to_classification.py` to check target folder mapping, and verifies config sweep generation and validation.

---

## 🏃 How to Run the Test Suite on Your Server

### 1. Prerequisite Environment Setup
Ensure your python environment is activated and all requirements are installed:
```bash
# Activate your conda or virtual environment
conda activate pytorch

# Install required packages
pip install -r requirements.txt
```

### 2. Run All Tests Simultaneously
To discover and run all unit tests in the suite, execute:
```bash
python -m unittest discover -s tests
```

### 3. Run a Specific Test Bench
To execute a single test file (e.g. environment check or datasets), run:
```bash
# Run environment diagnostics
python -m unittest tests/test_env.py

# Run trainer and loss tests
python -m unittest tests/test_trainer.py
```
