# 🛠️ Dev log: YOLO Dataset Preparation & Cleaning for Object Detection (Renamed & Variants)

Date: 2026-06-08

This log documents the implementation of the dataset cleaning and conversion pipeline from the raw `split_base/` format to normalized YOLO object detection format with support for three different annotation variants.

---

## 1. 📐 Pascal VOC to YOLO Format Conversion

- **Problem**: The source dataset at `split_base/` stores annotations in Pascal VOC format (XML files containing absolute `(xmin, ymin, xmax, ymax)` pixel coordinates). Downstream Ultralytics YOLO26 training expects labels in standard YOLO format (normalized space `[0, 1]` with layout `class_idx x_center y_center width height`).
- **Solution**: Developed `scripts/convert_split_base_to_detection.py` (matching the nomenclature and style of classification converter `convert_split_base_to_classification.py`):
  - Parses each image-XML pair in the `train/` and `val/` splits.
  - Extracts the image dimensions `(width, height)` from the XML size header, falling back to PIL image attributes if missing.
  - Normalizes absolute bounding box coordinates to the range `[0.0, 1.0]`.
  - Automatically handles negative coordinates, coordinate boundaries exceeding image dimensions, and degenerate/empty bounding box dimensions.

---

## 2. 🧹 Annotation Variants

- **Problem**: The raw annotations contain three distinct object classes: `cell`, `text`, and `abnormality`. The user requires different dataset variants to study model behavior across subsets of classes.
- **Solution**: The conversion script automatically outputs three distinct dataset layouts and configuration YAML files in a single run:

### Variant 1: `all`
- **Goal**: Localize all annotated objects in the dataset (`abnormality`, `cell`, and `text`).
- **Classes**: `abnormality` (idx 0), `cell` (idx 1), `text` (idx 2).
- **Target Dir**: `data/battery_detection_all/`
- **Target YAML**: `data/battery_detection_all.yaml`
- **Statistics**:
  - Train split: 253 images with targets, 577 bounding boxes total.
  - Val split: 79 images with targets, 169 bounding boxes total.

### Variant 2: `no_cell`
- **Goal**: Localize abnormality and text, dropping cell boxes.
- **Classes**: `abnormality` (idx 0), `text` (idx 1).
- **Target Dir**: `data/battery_detection_no_cell/`
- **Target YAML**: `data/battery_detection_no_cell.yaml`
- **Statistics**:
  - Train split: 221 images with targets, 324 bounding boxes total.
  - Val split: 70 images with targets, 90 bounding boxes total.

### Variant 3: `only_cell`
- **Goal**: Localize only cell boundaries, dropping abnormality and text boxes.
- **Classes**: `cell` (idx 0).
- **Target Dir**: `data/battery_detection_only_cell/`
- **Target YAML**: `data/battery_detection_only_cell.yaml`
- **Statistics**:
  - Train split: 252 images with targets, 253 bounding boxes total.
  - Val split: 79 images with targets, 79 bounding boxes total.

---

## 3. 💾 File Copying and Symlinking

- **Problem**: Copying large image directories wastes disk space and slows down data generation.
- **Solution**: Implemented a `--use-symlinks` option mimicking the behavior of the classification conversion script. This creates relative symlinks pointing to the original image files, preserving portability while maintaining zero disk-space overhead for images.

---

## 4. 📊 Dataset Statistics & Validation

- **Execution**: The conversion script was executed against the source dataset:
  - Validated that all output label files match their corresponding image file names across splits.
  - Asserted that all coordinates are strictly bounded within `[0.0, 1.0]`.
  - Asserted that class indices match each variant's respective mapping.

---

## 5. Status

- Preparation script committed to `scripts/convert_split_base_to_detection.py`.
- Old script `scripts/prepare_yolo_detection_data.py` removed.
- Dataset YAML files written to `data/battery_detection_{all,no_cell,only_cell}.yaml`.
- Dataset splits validated and verified.
