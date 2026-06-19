# Dev log: 5-Fold Cross-Validation Framework

Date: 2026-06-19

This log documents the implementation of 5-fold cross-validation support across both classification and object detection pipelines, along with dynamic dataset path resolving, fold-specific seeds, directory isolation, and dynamic YOLO dataset configuration rewrites.

---

## 1. The Problem

To evaluate foundation models and Parameter-Efficient Fine-Tuning (PEFT) configurations robustly, we need to train and validate models across a 5-fold cross-validation split (e.g., directory structures where datasets are placed in `fold_0` to `fold_4` subdirectories under the main data path).

Implementing cross-validation across parallel GPU processes presented several technical challenges:
1. **Filesystem Collision Risk**: Parallel training runs targeting different folds simultaneously could execute at the exact same second, causing timestamped output directories to collide if they shared the same configuration base name.
2. **YOLO Data Configuration Race Conditions**: Ultralytics YOLO requires a single data description YAML file defining dataset paths and classes. Running sweeps across different folds concurrently would cause race conditions if multiple subprocesses tried to edit or read a single shared data YAML file.
3. **Reproducibility and Random States**: For robust cross-validation, each fold needs its own independent seed to ensure distinct and reproducible initial model states, dataset shuffles, and oversampling operations.

---

## 2. The Solution

We extended the config schema, modified training entrypoints, and updated generation, scheduling, and validation scripts to support isolated 5-fold training.

### 2.1. Config Schema Extension (`src/bcadfm/utils/config.py`)
We added the `fold` parameter to the top-level `TrainingConfig` dataclass:
```python
@dataclass
class TrainingConfig:
    # ...
    seed: int = 42
    fold: Optional[int | str] = None
```
This allows YAML configurations to explicitly declare the fold they target.

### 2.2. Fold-Aware Dataset Loading and Path Resolution
In both classification (`scripts/train.py`) and object detection (`scripts/train_detection.py`), we added paths resolution logic to append the fold folder dynamically to the base data directory:
```python
if cfg.fold is not None:
    fold_str = f"fold_{cfg.fold}" if isinstance(cfg.fold, int) or (isinstance(cfg.fold, str) and cfg.fold.isdigit()) else str(cfg.fold)
    cfg.data.data_dir = str(Path(cfg.data.data_dir) / fold_str)
```
This forces the training dataset loaders to read from the correct fold subset.

### 2.3. Dynamic YOLO Dataset Config Rewriting (`scripts/train_detection.py`)
To prevent different parallel GPU training runs from colliding when using Ultralytics YOLO, the detection training script dynamically overrides the dataset path and writes a temporary YAML config within each run's isolated directory:
```python
yolo_data_path = cfg.yolo_data_yaml or "data/battery_detection_all.yaml"
if cfg.fold is not None:
    fold_str = f"fold_{cfg.fold}" if isinstance(cfg.fold, int) or (isinstance(cfg.fold, str) and cfg.fold.isdigit()) else str(cfg.fold)
    with open(yolo_data_path, "r") as f:
        yolo_data_dict = yaml.safe_load(f)
    orig_path = yolo_data_dict.get("path", "")
    yolo_data_dict["path"] = str(Path(orig_path) / fold_str)
    
    # Write temporary data YAML inside the run directory to prevent parallel job collision
    temp_data_yaml = run_dir / "yolo_data_fold.yaml"
    with open(temp_data_yaml, "w") as f:
        yaml.safe_dump(yolo_data_dict, f)
    yolo_data_path = str(temp_data_yaml)
```
The YOLO training engine is then invoked using this localized copy of the data config.

### 2.4. Fold-Specific Seeds
To keep each partition separate and reproducible, the grid configuration generator scripts (`generate_ablation_grid.py` and `generate_det_ablation_grid.py`) now compute fold-specific seeds dynamically:
$$\text{seed} = 30 + \text{fold} \times 10$$
- Fold 0 gets seed 30
- Fold 1 gets seed 40
- Fold 2 gets seed 50
- Fold 3 gets seed 60
- Fold 4 gets seed 70

### 2.5. Collison-Free Output Directory isolation
Run directories now incorporate the fold index in their directory names:
```python
if cfg.fold is not None:
    cfg_stem = f"{cfg_stem}_fold_{cfg.fold}"
# outputs/{strategy}/{model_name}__{cfg_stem}/{timestamp}
```
This guarantees separate directories for all folds even if they start concurrently.

### 2.6. Equivalence Verification in Runners and Status Checker
The parallel training runner schedulers (`scripts/run_parallel_ablations.py` and `scripts/run_parallel_det_ablations.py`) and the status auditing script (`scripts/check_ablation_status.py`) were modified to compare `fold` and `seed` variables:
```python
def _equiv(a: Dict, b: Dict) -> bool:
    for k in ("model_name", "data", "head", "peft",
              "learning_rate", "num_epochs", "imbalance", "fold", "seed"):
        if a.get(k) != b.get(k):
            return False
    return True
```
This prevents the schedulers from incorrectly identifying different folds of the same configuration as already completed, while correctly skipping folds that have already run.

---

## 3. Impact & Verification

- **Parallel Safety**: Parallel ablation sweeps across all 5 folds can be executed concurrently on 8 GPUs without directory write conflicts or dataset path overrides colliding.
- **Robust Evaluation**: Enables computing average metrics and standard deviations across the 5 folds for all architectures and PEFT configurations.
- **Unit Tested**: The logic is fully covered by tests in `tests/test_utils.py` and verified as passing within the conda `pytorch` environment.
