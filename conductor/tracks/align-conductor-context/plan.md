# Plan: Align Conductor Context

## Phase 1: Update Product & Guidelines Context
- [ ] Update `conductor/product.md` with multi-pipeline descriptions and key metrics.
  - **Verification**: Ensure classification and detection systems are represented.
- [ ] Update `conductor/product-guidelines.md` with seeding, device-safe buffers, and numpy augmentations.
  - **Verification**: Confirm design principles cover deep learning specific rules.

## Phase 2: Update Tech Stack & Workflow Context
- [ ] Update `conductor/tech-stack.md` with package versions, local caching, parallel runners, and visualizers.
  - **Verification**: Check version numbers match `requirements.txt`.
- [ ] Update `conductor/workflow.md` with local testing commands, server execution commands, DDP master port overrides, and oversampling rules.
  - **Verification**: Verify local conda environment (`pytorch`) and server conda environment (`/home/jovyan/pytorch_env`) details are clear.

## Phase 3: Final Verification
- [ ] Run pytest validation locally.
  - **Command**: `pytest tests/`
- [ ] Run PEFT parameter verification locally.
  - **Command**: `python tests/verify_peft.py`
