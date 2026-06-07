# 🛠️ Dev log: VPT DINOv3 Compatibility & Parallel Run Directory Collision Resolution

Date: 2026-06-07

This log documents the identification and resolution of two major issues discovered during the execution of the ablation study sweep: VPT model wrapping failures and parallel directory write conflicts.

---

## 1. 🧩 VPT Encoder-Less Backbone Crash

- **Problem**: Runs 43 to 58 (VPT configurations) failed immediately on epoch 1 with the following error:
  `ValueError: Could not find encoder module in backbone.`
- **Cause**: The DINOv3 backbone model loaded via `AutoModel.from_pretrained` does not have a nested `.encoder` module like standard ViT or DINOv2. The layer blocks reside directly under `self.layer`. Therefore, checking for `hasattr(self.original_backbone, "encoder")` failed.
- **Fix**: Reimplemented the forward pass in `VptWrappedBackbone` (`src/bcadfm/models/dinov3_classifier.py`). When no encoder module is present, it executes layers sequentially in a loop:
  - Iterates over `self.layers` manually.
  - Extracts intermediate hidden states and self-attentions if requested.
  - Constructs and returns a Hugging Face `BaseModelOutput` object containing the final layer hidden states, satisfying the classification head's expectations.

---

## 2. 🔀 Parallel Output Directory Collisions (Race Condition)

- **Problem**: Several LoRA, Adapter, and Baseline runs (e.g., runs 2, 7, 39) failed midway with:
  `safetensors_rust.SafetensorError: Error while serializing: I/O error: No such file or directory (os error 2)`
- **Cause**: `run_parallel_ablations.py` launches up to 8 configs simultaneously. Since the sub-processes are spawned in millisecond intervals, they all computed the exact same timestamp down to the second (e.g., `20260606_203526`). Because they used the same backbone, they mapped to the same output directory path:
  `outputs/cls__facebook-dinov3-vits16-pretrain-lvd1689m/20260606_203526`
  This caused processes to overwrite each other's checkpoint directories and configs, leading to I/O read/write conflicts during model saving and early stopping.
- **Fix**: Modified `scripts/train.py` to append the config file's stem name (`cfg_stem`) to the output path:
  `outputs/cls__facebook-dinov3-vits16-pretrain-lvd1689m__07_lora_vits16_r8_last2_lr0.0003/20260606_203526`
  This guarantees complete isolation for each ablation configuration, even if launched at the exact same second.

---

## 3. 🔍 Status Checker Improvements

- **Problem**: The status checker `check_ablation_status.py` listed runs as "to run again" if it found an older interrupted run folder, even if a subsequent attempt had completed successfully.
- **Fix**: Updated `scripts/check_ablation_status.py` to also map completed runs to their original configs. Now, if a configuration is found in the list of completed runs, it is filtered out of the "Config files to run again" list, preventing redundant training.

---

## 4. 📈 Status

- All fixes pushed to git.
- The 16 failed VPT configs and other interrupted runs can now be resumed or rerun cleanly without directory conflict.
