# Specification: K-Fold Compatibility

## Requirements

1. **Statistical Analysis Script**:
   - Must calculate normal/abnormal image distributions (classification) and bounding box counts (cell-wise/box-wise) for train/val/test splits per fold and entire dataset.
   - Bounding box classes must be determined dynamically from the dataset instead of being hardcoded.
   - Output must be saved as a CSV file and printed to the terminal.

2. **K-Fold Dataset Conversion**:
   - Rename conversion scripts to:
     - `scripts/convert_kfold_to_classification.py`
     - `scripts/convert_kfold_to_detection.py`
   - Add a `--kfold` flag to support processing all folds.
   - Output format:
     - Classification: `data/kfold_classification/fold_{idx}/(train|val)/(normal|abnormal)`
     - Detection: `data/kfold_detection/battery_detection_{variant}/fold_{idx}/(images|labels)/(train|val)`

3. **Config Updates**:
   - Default classification and detection smoke/ablation configs must point to new K-fold conversion outputs.
   - Detection configs must document/include the `fold: null` option.

4. **Visualizer updates**:
   - Streamlit visualizer must support a checkbox to average metrics over folds.
   - Grouping runs based on identical hyperparameters, extracting fold index from config or path name.
   - Plotting averaged training curves.
   - Supporting inspecting either overall group average or selecting a specific fold in the inspector.
   - Load and display the K-Fold statistics CSV file if present.

5. **Unit Tests**:
   - Include unit tests for the conversion scripts and the visualizer grouping/averaging logic.
   - Run verification in the local `pytorch` conda environment.

## Acceptance Criteria
- Statistical analysis script runs successfully and reports stats.
- Conversion scripts generate the folded datasets correctly.
- Streamlit visualizer runs and correctly aggregates metrics.
- All unit tests pass successfully in the local conda environment.
