# Specification: Align Conductor Context

## Requirements

1. **Pipeline & Metrics Synchronization**:
   - The DINOv3 + PEFT classification pipeline and the DINOv3 SFP + YOLO26 detection pipeline must be clearly defined in `conductor/product.md`.
   - Relevant metrics (F1-score for classification; mAP50, mAP50-95, and bbox IoU/Dice for object detection) must be listed.

2. **Guidelines Alignment**:
   - Document device-safe loss layer initialization (e.g. GPU device placement for loss class weight buffers).
   - Document NumPy-based PIL-free image augmentations to optimize data loading throughput.
   - Document seed reproducibility parameter (`seed: 42`).

3. **Tech Stack Updates**:
   - Pinned dependency versions (`torch==2.4.0`, `torchvision==0.19.0`, etc.) must be added.
   - Centralized local model caching directory (`models/hf_cache`) and the dynamic redirection environment logic (`HF_HOME`) must be documented.
   - parallel training dashboard (`run_parallel_ablations.py`) and visualizer dashboard (`visualize.py`) details must be recorded.

4. **Workflow Policies**:
   - Local validation steps must list pytest shape test execution and check_model_init.
   - Multi-GPU DDP master port selection (conflict prevention) must be specified.
   - In-place oversampling DDP-safety checks and traversal behavior must be outlined.

## Acceptance Criteria
- All modified markdown files (`product.md`, `product-guidelines.md`, `tech-stack.md`, `workflow.md`) are successfully updated.
- Automated tests (`pytest tests/` and `python tests/verify_peft.py`) run successfully.
