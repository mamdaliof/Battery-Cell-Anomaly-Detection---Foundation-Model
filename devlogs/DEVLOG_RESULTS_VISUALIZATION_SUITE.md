# 📊 Dev log: Results Visualization Suite (Jupyter Notebook & Streamlit Dashboard)

Date: 2026-06-07

This log documents the design, implementation, and local verification of the DINOv3 + PEFT ablation study visualization suite.

---

## 1. 📓 Jupyter Notebook Analyzer (`notebooks/visualize_results.ipynb`)

- **Objective**: Create a robust local analysis notebook for quick research access.
- **Features**:
  - **Recursive Loader**: Scans the `outputs/` directory structure for `config.yaml` and `trainer_state.json`. If a run is in progress or was interrupted, it scans for `checkpoint-*` folders and loads the most recent `trainer_state.json` file.
  - **Interactive Leaderboard**: Built with `ipywidgets` to filter and sort runs. Sorted primarily by validation F1 score.
  - **Plotly Trajectory Comparator**: Lets users select multiple runs from a list to plot their training loss and evaluation metric curves (F1, accuracy, loss, etc.) over epochs.

---

## 2. 🖥️ Streamlit Web Dashboard (`visualize.py`)

- **Objective**: Build an interactive, premium web application for results communication, dashboard analysis, and hyperparameter exploration.
- **Implementation**:
  - **Layout & Aesthetics**: Developed a sleek layout using custom CSS styles (dark card motifs, gradient headers, HSL color-coded KPIs).
  - **Dynamic Sidebar**:
    - Includes filters for backbone models, PEFT types, learning rates, and imbalance strategies.
    - Features placeholder selections for future tasks (e.g., Anomaly Localization, Object Detection, Segmentation) and future hyperparameters (e.g., ScheduleFree Optimizer, custom weight decay rates).
  - **Core Tabs**:
    - **🏆 Leaderboard**: Interactive DataFrame sorted by validation F1 score. The highest-performing configuration is highlighted using cell backgrounds.
    - **📈 Trajectory Curves**: Multi-run curve comparator allowing users to select multiple runs and plot train/val loss, accuracy, precision, recall, AUROC, and F1 trajectories. Includes an option to truncate the plots at the best validation epoch.
    - **🔬 Single Run Inspector**: Displays a complete breakdown of configuration settings, detailed epoch metrics, and a dynamic **Confusion Matrix** (TP, TN, FP, FN) generated via a Plotly heatmap from the best validation epoch.
    - **📊 PEFT & Hyperparameter Analysis**: Provides aggregated comparisons across PEFT methods and learning rates, as well as a **Parallel Coordinates Plot** mapping the relationships between learning rate, PEFT size (LoRA rank, adapter dim, VPT tokens), and final validation F1 score.

---

## 3. 🔍 Local Verification & Performance

- **Environment**: Ran local smoke tests under the Miniconda `pytorch` environment.
- **Results**:
  - The recursive results parser successfully located and parsed **53 directories** containing completed and active training runs.
  - Handled missing dictionary keys and varying checklist lengths gracefully (fail-safe parsing).
  - Calculated exact confusion matrices for completed configurations.
  - Verified compilation and execution of Streamlit components with no syntax or library mismatch errors.

---

## 4. 📈 Status

- Both `notebooks/visualize_results.ipynb` and `visualize.py` have been pushed to the remote repository.
- Added necessary visualization dependencies (`streamlit`, `plotly`, `ipywidgets`) to `requirements.txt`.
- Running local instance command:
  ```bash
  streamlit run visualize.py
  ```
