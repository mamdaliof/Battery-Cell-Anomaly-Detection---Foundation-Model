# 🛠️ Dev log: YOLO26 + DINOv3 SFP PEFT Integration & Folder Reorganization

Date: 2026-06-08

This log documents the implementation of Parameter-Efficient Fine-Tuning (PEFT) inside the YOLO-DINO detection model (`YoloDinoModel`), the folder structure reorganization separating classification (`cls`) and detection (`det`) configs/outputs, and the relocation of GPU validation utilities.

---

## 1. 🔌 YOLO-DINO PEFT Integration

- **Problem**: Previously, `YoloDinoModel` only supported a fully frozen backbone. To improve detection accuracy on battery cell defects, we needed to support parameter-efficient fine-tuning (PEFT) inside the YOLO object detection backbone.
- **Solution**: Updated `src/bcadfm/models/yolo_dino.py` to accept and parse a `peft_config`:
  - **LoRA**: Wraps the backbone using Hugging Face `peft` targeting specific attention projections (`q_proj`, `v_proj`).
  - **Bottleneck Adapters**: Dynamically inserts Pfeiffer-style bottleneck adapters after transformer FFN/MLP blocks.
  - **Visual Prompt Tuning (VPT)**: Wraps the backbone with `VptWrappedBackbone` supporting Shallow and Deep prompt parameters.

---

## 2. 🔄 Gradient Routing during PEFT Training

- **Problem**: To save GPU memory and prevent gradient calculation overhead, the backbone's forward pass in `yolo_dino.py` runs inside `with torch.no_grad():`. However, if PEFT is active during training, executing inside `torch.no_grad()` freezes the trainable adapter weights, making training impossible.
- **Solution**: Conditioned the backbone's gradient tracking context on the active PEFT state and training mode:
  ```python
  if self.peft_type != "none" and self.training:
      outputs = self.model(x_norm)
  else:
      with torch.no_grad():
          outputs = self.model(x_norm)
  ```
  This ensures that gradients are only computed and tracked through the trainable adapter layers when training, while keeping the rest of the backbone fully frozen.

---

## 3. 📐 VPT Token Slicing Offset in Detection

- **Problem**: When Visual Prompt Tuning (VPT) is active, the layout of sequence outputs from the DINOv3 model shifts:
  $$\text{Output Sequence} = [\text{CLS}] \mathbin{\Vert} [\text{Prompts}] \mathbin{\Vert} [\text{Registers}] \mathbin{\Vert} [\text{Patches}]$$
  If we extract patch tokens using the standard CLS + Register offset index, we end up extracting prompt tokens instead of patch tokens, misaligning the spatial feature grids and causing shape mismatch errors.
- **Solution**: Adjusted the patch feature extractor in `yolo_dino.py` to calculate `start_idx` dynamically based on the number of prompt tokens:
  ```python
  num_prompts = 0
  if self.peft_type == "visual_prompt":
      num_prompts = getattr(self.model, "num_tokens", 0)
      
  start_idx = 1 + num_prompts + self.num_registers
  ```
  This skips prompt tokens correctly and extracts only the spatial patch features for 2D feature grid reconstruction.

---

## 4. 📂 Folder Reorganization & Test Relocations

- **Goal**: Organize the repository structure to cleanly isolate classification (`cls`) and detection (`det`) tasks, configs, and outputs.
- **Modifications**:
  - **Configs**: Moved all classification configs to `configs/cls/` (e.g. `configs/cls/ablations/`).
  - **Outputs**: Re-routed classification runs to `outputs/cls/{safe_model_name}__{cfg_stem}/{timestamp}/`.
  - **Validation tests**: Relocated `gpu_alloc_test.py` and `ddp_alloc_test.py` from `scripts/` to `tests/` and updated `docs/technical_details.md`.
  - **Ablation runner**: Updated `validate_ablation_configs.py` and `run_parallel_ablations.py` to use the updated config and output directories.

---

## 5. 📈 Status

- All PEFT modifications, folder reorganizations, and script relocations are pushed to git and verified.
- The unit test suite compiles and runs cleanly, passing all 30 validation tests.
