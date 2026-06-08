# Dev Log: LoRA Block Targeting Fix & training_step Rollback

Date: 2026-06-09

This log documents two closely related fixes applied in sequence on 2026-06-09:

1. **Rollback of `training_step` timing/profiling code** (v1.0 restoration)
2. **LoRA block-targeting rewrite** (architecture-agnostic freezing strategy)

---

## 1. training_step Rollback (Commit 6b5307b)

### Problem
A `training_step` override had been introduced with timing/profiling instrumentation. This override used a 3-argument signature (`self, model, inputs, num_items_in_batch`) which conflicts with the HF Trainer v5.x API, causing a `TypeError` at the start of every training run.

The override also introduced six new instance variables in `__init__`:
```python
self._last_step_time = None
self._last_forward_time = 0.0
self._step_count = 0
self._accumulated_data_time = 0.0
self._accumulated_forward_time = 0.0
self._accumulated_step_time = 0.0
```

### Fix
- Removed the entire `training_step` override from `ImbalanceTrainer`.
- Removed all six timing instance variables from `__init__`.
- All other fixes (DDP oversampling fallback C3, subset/wrapper traversal H6, `inputs.copy()` + label pop C6, device placement C1/C2, `num_items_in_batch` parameter in `compute_loss`) were preserved.

### Result
Trainer reverts to the standard HF `Trainer.training_step` implementation. No functional change to training behaviour — the profiling was passive instrumentation only.

---

## 2. LoRA Block Targeting Rewrite (Commit ec2dd85)

### Problem
The `_apply_lora` path in `dinov3_classifier.py` used PEFT's `layers_to_transform` + `layers_pattern` mechanism to inject LoRA weights only into a user-specified subset of transformer blocks. This mechanism is fragile for non-standard architectures:

- For DINOv3, the module tree uses `model.layer` or `layers` (not `encoder.layer`).
- PEFT's pattern-matching internally resolves layer indices against `layers_pattern` as a substring path match. When the path doesn't resolve correctly, PEFT either silently applies LoRA to all layers or raises a `KeyError`.
- The auto-detection heuristic (`detected_pattern = next(...)`) could return `None` or an incorrect path depending on the model variant, making behaviour non-deterministic across architectures.

### Solution
Replace the PEFT-native targeting approach with a two-phase PyTorch approach:

**Phase 1 – Apply LoRA to all blocks (no PEFT filtering):**
```python
# No layers_to_transform / layers_pattern
peft_lora_config = LoraConfig(
    r=r,
    lora_alpha=alpha,
    target_modules=target_modules,
    lora_dropout=dropout,
    bias="none",
)
self.backbone = get_peft_model(self.backbone, peft_lora_config)
```

**Phase 2 – Freeze `lora_*` weights in non-target blocks:**
```python
if target_blocks is not None:
    blocks = _get_transformer_blocks(self.backbone)  # new helper
    target_set = set(target_blocks)
    for idx, block in enumerate(blocks):
        if idx not in target_set:
            for name, param in block.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
```

**`_get_transformer_blocks` helper:**
Finds the transformer block list by probing common attribute paths in order:
1. `model.encoder.layer`
2. `model.layer`
3. `encoder.layer`
4. `layers`
5. `layer`
6. Falls back to walking `named_modules()` for any `nn.ModuleList` whose children are `nn.Module` instances with `attention` sub-modules.

This makes the function architecture-agnostic and works correctly for DINOv3, standard ViT, and other transformer variants.

### Net Effect
The training outcome is **identical** — only the target blocks contribute LoRA gradients. The difference is purely in *how* that filtering is enforced: PyTorch `requires_grad = False` instead of relying on PEFT's path resolution, which was unreliable for this architecture.

---

## 3. Files Changed

| File | Change |
|------|--------|
| `src/bcadfm/training/trainer.py` | Removed `training_step` override and 6 timing `__init__` vars |
| `src/bcadfm/models/dinov3_classifier.py` | Replaced PEFT `layers_to_transform`/`layers_pattern` with post-wrap freezing; added `_get_transformer_blocks()` helper |

---

## 4. Verification

After applying both fixes:
```bash
git pull origin master
python scripts/validate_ablation_configs.py
```

Expected: all ablation configs load without `TypeError` and LoRA-only ablations show correct trainable parameter counts (only target block LoRA weights have `requires_grad=True`).
