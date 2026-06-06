# Dev log: PEFT & Imbalance Handling Integration

Date: 2026-06-06

This log documents the sequence of changes, implementation details, debugging steps, and successful verification of Task 4 (PEFT Integration Module) and Task 5 (Imbalance Handling Module).

---

## 1. PEFT Integration Module (Task 4)

- **Goal**: Integrate LoRA, Pfeiffer Bottleneck Adapters, and Visual Prompt Tuning (VPT) Shallow & Deep modes as parameter-efficient options on top of the frozen DINOv3 (or standard ViT) backbone.
- **Key Files**:
  - `src/bcadfm/utils/config.py`: Added configuration schemas (`PeftConfigSchema`) parsing PEFT types, rank, target modules, bottleneck dimension, and number of prompt tokens.
  - `src/bcadfm/models/dinov3_classifier.py`:
    - **LoRA**: Wraps underlying backbone using Hugging Face `peft` library.
    - **Bottleneck Adapters**: Standard Pfeiffer bottleneck adapter (`Input -> Down -> Act -> Drop -> Up -> Residual`) inserted dynamically after FFN/MLP blocks.
    - **VPT (Shallow & Deep)**: Custom `VptWrappedBackbone` and `VptLayerWrapper` to prepend and replace learnable prompt parameters.
  - `src/bcadfm/utils/model_utils.py`: Added count parameter helper functions (`count_parameters` and `log_parameter_summary`) to track trainable/non-trainable weight distribution.
  - `scripts/verify_peft.py`: Unit test verifying parameters and activations on all 5 tuning modes (None, LoRA, Adapters, Shallow VPT, Deep VPT) with a standard backbone.

---

## 2. Imbalance Handling Module (Task 5)

- **Goal**: Address severe class imbalance via loss-level and data-level strategies.
- **Key Files**:
  - `src/bcadfm/training/losses.py`: Added custom class-weighted cross-entropy loss and `FocalLoss` support for binary/multi-class targets.
  - `src/bcadfm/training/trainer.py`: Added `ImbalanceTrainer` subclass of `Trainer` that dynamically calculates dataset statistics, loads weights, supports `WeightedRandomSampler`, and overrides `compute_loss`.
  - `src/bcadfm/data/dataset.py`: Implemented in-place dataset oversampling of minority class samples. Highly recommended for multi-GPU training to bypass distributed sampler constraints.

---

## 3. Debugging & Portability Issues Resolved

### 3.1. DDP Warning Cleanups
- **find_unused_parameters Warning**: Added `ddp_find_unused_parameters=False` in `TrainingArguments` to prevent unnecessary traversals.
- **NCCL Cleanup Warning**: Unconditionally destroy process group using `torch.distributed.destroy_process_group()` at the end of training.

### 3.2. Duplicate prints in DDP
- Checking `torch.distributed.is_initialized()` at startup is `False` because `Trainer` initializes distributed loop later. This caused rank 0 and rank 1 to print headers twice.
- **Fix**: Check `LOCAL_RANK` environment variable directly at startup: `int(os.environ.get("LOCAL_RANK", "0")) == 0`.

### 3.3. PEFT Target Module & Architecture Mismatches
- **Problem**: `google/vit-base-patch16-224-in21k` and `facebook/dinov3-vitb16-pretrain-lvd1689m` linear layers end with `q_proj` and `v_proj` (instead of standard `query` and `value`).
- **Problem**: DINOv3 model structures layers under `model.layer` or `layers` (instead of `encoder.layer` or standard `layer`).
- **Fix**:
  - Updated target modules in config schemas to `q_proj`/`v_proj`.
  - Implemented dynamic layer mapping inside `dinov3_classifier.py` matching `encoder.layer`, `model.layer`, `layer`, or `layers` attributes.

### 3.4. HF Trainer Sampler Signature Mismatch
- **Problem**: Newer versions of HF Trainer invoke `_get_train_sampler` with dataset arguments (e.g. `sampler_fn(dataset)`).
- **Fix**: Updated `_get_train_sampler` signature in `ImbalanceTrainer` to accept `*args, **kwargs` and pass them correctly to the parent constructor.

---

## 4. Successful Verification Runs

- **Automated PEFT tests**: `python3 scripts/verify_peft.py` executed successfully, passing Baseline, LoRA, Bottleneck Adapters, VPT Shallow, and VPT Deep parameter count assertions.
- **DINOv3 LoRA smoke test**: `torchrun --nproc_per_node=2 scripts/train.py --config configs/peft_smoke.yaml` completed 1 epoch of training successfully on the server.
