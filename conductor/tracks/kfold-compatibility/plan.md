# Plan: K-Fold Compatibility

## Phase 1: Statistical Analysis Script
- [x] Create `scripts/analyze_kfold_stats.py` to parse XMLs and count image distributions per fold dynamically.
- [x] Save metrics as a CSV file and output to terminal.
- [x] Verify script outputs correct statistics for all folds and overall dataset.

## Phase 2: K-Fold Dataset Conversion
- [x] Create and update `scripts/convert_kfold_to_classification.py` with `--kfold`.
- [x] Create and update `scripts/convert_kfold_to_detection.py` with `--kfold`.
- [x] Delete old conversion scripts: `convert_split_base_to_classification.py` and `convert_split_base_to_detection.py`.
- [x] Verify conversion scripts work end-to-end on the 5-fold dataset.

## Phase 3: Configs, Unit Tests & Visualization Update
- [x] Update classification and detection configs to use K-Fold targets and specify fold field.
- [x] Create `tests/test_kfold_conversion.py` and update existing tests.
- [x] Update `visualize.py` to support "Average over Folds" grouping, averaged curves, fold-selector in run inspector, and display K-fold CSV stats.
- [x] Verify visualizer works with both single runs and grouped K-fold runs.

## Phase 4: Final Validation
- [x] Run pytest validation suite in conda environment.
