# Product Guidelines - Battery Cell Anomaly Detection - Foundation Model

## Voice and Tone
- **Concise and direct**: Communication, logs, documentation, and user interfaces should be straightforward, concise, and technically precise.

## Design Principles
1. **Modular and extensible architecture**: Ensure the codebase is structured to easily scale to other vision tasks (such as detection and multi-label classification), new models, and other datasets.
2. **Strict reproducibility**: Control and fix random seeds, and drive all experiments through structured configuration files (e.g., YAML) to ensure every run can be perfectly replicated.
3. **Performance and resource efficiency**: Rely on parameter-efficient fine-tuning (PEFT) methods (LoRA, Adapters, VPT) and frozen foundation model backbones to minimize computational footprint and training time.
4. **Comprehensive metric reporting**: Explicitly report classification metrics (F1-score, accuracy, precision, recall), detection metrics (mAP), and raw confusion matrix cell counts to carefully monitor performance under severe class imbalance.
5. **Mandatory testing and validation**: Write unit tests and validation codes for each module to verify correctness and prevent regression.

## Implementation & Code Constraints

### 1. Seeding and Determinism
- A global configuration parameter `seed` (defaulting to `42`) must be set across all execution files.
- Random states must be set globally during training initialization:
  ```python
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
  ```
- Random states in local helpers (such as dataset oversampling) must be isolated to guarantee deterministic run-to-run variance.

### 2. Device-Safe Tensor & Buffer Placements
- **Audit Rule**: To avoid PyTorch device mismatch exceptions on multi-GPU nodes, ensure all custom loss weights, focal alpha parameters, and other dynamic tensors are dynamically initialized or copied to the target GPU device inside the forward pass:
  ```python
  # Align target tensors with incoming prediction device dynamically
  weights = self.weights.to(logits.device)
  ```
- Backbones should register their normalization bounds (ImageNet mean/std) as PyTorch buffers (`self.register_buffer("mean", ...)`) to leverage GPU-side image normalization and ensure dynamic device safety.

### 3. PIL-Free & Efficient Augmentations
- Data preprocessing and custom transforms must avoid converting images back and forth between PIL and Tensors.
- Implement random operations (like Gaussian noise) using native NumPy-based injections:
  $$\text{np.array (float32)} \xrightarrow{\text{np.random.normal}} \text{np.array (float32)} \xrightarrow{\text{clip}} \text{np.array (uint8)}$$
- Instantiation of transformation pipelines must be performed once (reused across all `__getitem__` calls) to minimize CPU/GPU memory footprint and CPU thread overhead.

### 4. Deferred Checkpointing
- To minimize I/O overhead on multi-GPU environments, model serialization (`model.state_dict()`) must be deferred. Only execute model checkpointing when a metric improvement (e.g., lower validation loss or higher F1) is actually achieved.
