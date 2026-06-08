# 🛠️ Dev log: Custom Object Detection Validation Metrics

Date: 2026-06-08

This log documents the design, implementation, and verification of custom evaluation metrics for the YOLO26 + DINOv3 SFP object detection pipeline.

---

## 1. 📐 Subclassing Ultralytics Validation Loop

- **Problem**: Standard YOLO validation calculates only global/aggregate metrics (Precision, Recall, mAP50, mAP50-95) and doesn't expose class-specific true positive/false positive counts, matched bounding box overlaps (IoU/Dice), or map detection outputs back to image-level classification predictions.
- **Solution**: Created `src/bcadfm/training/yolo_trainer.py`:
  - **`CustomDetectionValidator`**: Inherits from `DetectionValidator` and overrides `update_metrics`, `get_stats`, and `finalize_metrics`.
  - **`CustomDetectionTrainer`**: Inherits from `DetectionTrainer` and overrides `get_validator()` to return our custom validator.

---

## 2. 🔌 Custom Metrics Computation

The custom validator implements three layers of custom metrics:

### 2.1. Per-Class Bbox Metrics
- Extends standard metrics to compute exact **True Positives (TP)**, **False Positives (FP)**, and **False Negatives (FN)** at `IoU=0.50` for each class dynamically:
  $$\text{TP}_c = \sum (\text{tp\_mask} \land (\text{pred\_cls} == c))$$
  $$\text{FP}_c = \sum (\neg\text{tp\_mask} \land (\text{pred\_cls} == c))$$
  $$\text{FN}_c = \text{GroundTruthCount}_c - \text{TP}_c$$

### 2.2. Bbox Overlap IoU & Dice Coefficient
- Matches predicted bounding boxes to ground truths of the same class using a greedy search at `IoU >= 0.50`.
- Calculates individual box-level IoU and computes the **Dice Coefficient**:
  $$\text{Dice} = \frac{2 \times \text{IoU}}{1 + \text{IoU}}$$
- Computes global average IoU and Dice across all matched box pairs in the validation set.

### 2.3. Image-Level Multi-Label Classification Conversion
- Constructs binary classification indicator vectors `[has_abnormality, has_text]` for each validation image:
  - **Ground Truth**: Set to 1 if there is at least one ground-truth bounding box of that class, else 0.
  - **Prediction**: Set to 1 if there is at least one predicted bounding box of that class with confidence $\ge 0.25$, else 0.
  - **Probability**: Set to the maximum confidence score of all predicted boxes of that class (for ROC-AUC).
- Uses `scikit-learn` to calculate standard classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **AUROC** independently for `abnormality` and `text`.

---

## 3. 🧪 Verification & Unit Tests

- **Implementation**: Created `tests/test_yolo_metrics.py` to dry-run the custom validation metrics using mock prediction and target tensors.
- **Test cases**:
  - `test_coordinate_matching_and_classification`: Verifies coordinate matching, IoU/Dice calculation, and binary classification mapping indices are correct.
  - `test_get_stats_formatting`: Verifies sklearn metrics computation and formatting in the stats output dictionary.
- **Results**: Executed the unit test suite under the `pytorch` conda environment:
  ```bash
  PYTHONPATH=src python tests/test_yolo_metrics.py
  ```
  Both tests passed successfully (`OK`).

---

## 4. Status

- Custom validator and trainer committed to `src/bcadfm/training/yolo_trainer.py`.
- Validation tests committed to `tests/test_yolo_metrics.py`.
- Verified and ready for object detection training runs.
