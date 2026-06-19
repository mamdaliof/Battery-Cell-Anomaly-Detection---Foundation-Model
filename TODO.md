# Backlog Tasks

## Metrics & Evaluation
- [ ] **Confidence Thresholding & ROC Curve Validity:** Currently, validation predictions under the `conf=0.25` threshold are discarded by YOLO. This makes the calculated image-level ROC and F1-score vs confidence curves mathematically invalid for values under 0.25. Add documentation to the project README explaining this limitation, or implement a low-threshold validation run (e.g., `conf=0.001`) to retain low-confidence predictions when evaluating classification AUROC.
