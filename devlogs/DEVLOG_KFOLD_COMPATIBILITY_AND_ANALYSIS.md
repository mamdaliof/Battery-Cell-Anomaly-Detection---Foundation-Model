# Dev log: K-Fold Compatibility and Analysis - CAVE MAN WRITE THIS!

Date: 2026-06-21

ME ANTIGRAVITY. ME WRITE LOG FOR TODAY WORK. TODAY ME WORK ON KFOLD DATA CONVERT AND ANALYSIS STUFF!

---

## 1. THE PROBLEM (WHAT MAKE CAVE MAN MAD)

1. **NO FOLD CONVERT SCRIPTS FOR ALL FOLDS**: Old scripts only split one piece of rock. We need to split 5 folds of rocks!
2. **NO STATS BONES**: Caveman did not know how many normal image rocks or box bones exist in each fold partition.
3. **VISUALIZER NOT AGGREGATE**: Streamlit dashboard did not show averaged curves across folds. Caveman had to look at 5 different plots.
4. **NO TESTS**: We had no tests to prove that the conversion scripts worked correctly.

---

## 2. THE SOLUTION (ME SMASH WITH CLUB)

Me make new scripts, new tests, and update dashboard!

### 2.1. Dynamic Conversion Scripts (`scripts/convert_kfold_to_classification.py` & `scripts/convert_kfold_to_detection.py`)
- Me rename old scripts.
- Me add `--kfold` flag. If caveman set flag, script find all `fold_*` dirs in source root and convert all of them.
- Me support `--use-symlinks` to not copy heavy image rocks, only make links! Saving space in cave.

### 2.2. Stats Counting Script (`scripts/analyze_kfold_stats.py`)
- Me build new script to count image rocks and box bones dynamically.
- XML object labels are found dynamically (no hardcoding).
- Write all counts to `data/kfold_structured_dataset/kfold_stats.csv` and print nice tables to console.

### 2.3. Dashboard Updates (`visualize.py`)
- Me add checkbox: "Average Metrics Over Folds".
- Dashboard group runs with identical hyperparameters, average curves, and average metrics.
- Single Run Inspector let caveman choose to see averaged group or inspect a specific fold run.
- Dashboard show `kfold_stats.csv` table.

### 2.4. Tests & Verification (`tests/test_kfold_conversion.py` & `tests/test_visualizer.py`)
- Me add `test_kfold_conversion.py` to check fold splits.
- Me add `test_visualizer.py` to check run grouping and averaging.
- All 39 tests pass in `pytorch_env` environment!
