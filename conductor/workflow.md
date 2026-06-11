# Workflow Guidelines - Battery Cell Anomaly Detection - Foundation Model

## Development Methodology
- **Spec-driven development with checklist checkpoints**: For every feature, bug fix, or major change, create a spec (`spec.md`) and a checklist plan (`plan.md`) under a new track folder (`conductor/tracks/<track-id>/`).
- **Permanent and growing unit test suite**: Ensure that unit tests and validation code written during verification are kept permanently in the `tests/` directory to prevent future regressions.

## Git Branching & Commit Strategy
- **Feature Branches**: Work on feature branches corresponding to the active track (e.g., `track/ablation-framework`, `track/yolo26-integration`).
- **PR Merge Policy**: Merge into the `main` branch only after verification steps are fully completed and all unit tests pass on the branch.
- **Commit Messages**: Use clear and descriptive semantic logs when checking in work.

## Verification & Computation Policies

> [!IMPORTANT]
> **Environment-Specific Execution Policy**
> 
> * **Local PC (Farhad's machine)**:
>   * Environment: Activate using `conda activate pytorch`.
>   * Restriction: **DO NOT** run any heavy training, full multi-GPU runs, or extensive ablation studies locally. Only run fast unit tests and quick smoke checks.
> * **Training Server**:
>   * Environment: Activate using `conda activate /home/jovyan/pytorch_env`.
>   * Capabilities: Full access to GPU/multi-GPU compute. All heavy training jobs, smoke configs (e.g., `configs/cls/peft_smoke.yaml`), and ablation grids must be run here.

## DDP Multi-GPU Orchestration Rules

### 1. Master Port Conflict Prevention
- When running multiple independent DDP training loops or sweeps in parallel on the server, you must assign unique communication ports via `torchrun` to avoid port conflict lockouts:
  ```bash
  torchrun --nproc_per_node=2 --master_port=29501 scripts/train.py --config configs/cls/peft_smoke.yaml
  torchrun --nproc_per_node=2 --master_port=29502 scripts/train_detection.py --config configs/det/peft_smoke.yaml
  ```

### 2. DDP-Safe Dataset Oversampling
- Data-level oversampling (`BatteryCellDataset.oversample_dataset`) is fully compatible with DDP partitioning.
- **Sampler Fallback Audit**: If `oversampling_method="weighted_sampler"` is requested in a DDP run, the trainer must log a warning and automatically fall back to data-level oversampling.
- **Index Traversal**: The trainer must handle subset wrappers (e.g. `Subset`) dynamically, traversing the nested hierarchy to locate correct indices and labels.

## Verification Commands

Before merging any feature branch, execute:

### 1. Local Verification (Safe to run on Farhad's PC)
- **Unit and Shape Tests**:
  ```bash
  pytest tests/
  ```
- **PEFT Parameter Allocation Test**:
  ```bash
  python tests/verify_peft.py
  ```
- **Ablation Configurations Check**:
  ```bash
  python scripts/validate_ablation_configs.py
  ```

### 2. Server-Side Verification (Only run on Server)
- **Classification Smoke Test**:
  ```bash
  torchrun --nproc_per_node=2 --master_port=29501 scripts/train.py --config configs/cls/peft_smoke.yaml
  ```
- **Detection Smoke Test**:
  ```bash
  torchrun --nproc_per_node=2 --master_port=29502 scripts/train_detection.py --config configs/det/peft_smoke.yaml
  ```
