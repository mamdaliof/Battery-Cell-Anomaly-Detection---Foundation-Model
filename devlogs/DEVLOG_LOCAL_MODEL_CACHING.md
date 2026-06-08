# Dev log: Local Model Caching & Offline Speedup

Date: 2026-06-08

This log documents the implementation of local model caching to prevent redundant Hugging Face downloads, speed up training initialization, and support offline model execution.

---

## 1. The Problem
During development, testing, and scaling of parallel ablation sweeps, foundation models (e.g. DINOv3 backbones like `facebook/dinov3-vits16-pretrain-lvd1689m` and `facebook/dinov3-vitb16-pretrain-lvd1689m`) were being downloaded repeatedly, or resolving files via the network on every run.
- This consumed significant bandwidth.
- It delayed the startup time of training and verification processes.
- It caused network-dependent vulnerabilities or rate-limiting risk during parallel job validation.

---

## 2. The Solution: Workspace Caching (`models/hf_cache`)
We redirected the Hugging Face hub cache directory to a local folder named `models/hf_cache` under the project workspace root. 

### 2.1. Git Exclusion
To ensure large pre-trained model checkpoint binaries (which can be several gigabytes) are not checked into Git, we appended the local models directory to `.gitignore`:
```text
# Local cached models
/models/
```

### 2.2. Dynamic Environment Injection
Instead of hardcoding `cache_dir` arguments in all `AutoModel.from_pretrained` and `AutoImageProcessor.from_pretrained` calls, we dynamically inject the `HF_HOME` environment variable at script startup. This ensures that *any* third-party package dependency (such as `transformers` or `huggingface_hub`) automatically points to the local folder.

We added the following runtime block to:
1. `src/bcadfm/__init__.py` (executes on any local package import)
2. `scripts/train.py`
3. `scripts/run_parallel_ablations.py` (propagates to all concurrent subprocess slots)
4. `scripts/validate_ablation_configs.py`
5. `scripts/check_model_init.py`
6. `scripts/diagnose_backbone.py`

```python
import os
from pathlib import Path

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    # Resolves to workspace root (e.g., parents[1] for scripts, parents[2] for bcadfm init)
    workspace_root = Path(__file__).resolve().parents[1] 
    hf_cache_dir = workspace_root / "models" / "hf_cache"
    os.environ["HF_HOME"] = str(hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
```

---

## 3. Impact
- **Zero Redundant Downloads**: Once a DINOv3 or ViT model is downloaded by any script or unit test, it is cached locally in `models/hf_cache`.
- **Fast Startup**: Subsequent runs of unit tests, single trainings, or parallel ablation sweeps load models instantly from the local SSD/HDD.
- **Robustness**: Ensures the codebase can run offline or in network-isolated server environments once the cache directory is populated.
