# Plan: Visualizer Enhancements

## Phase 1: Implement Visualizer Unit Test Suite
- [x] Create `tests/test_visualizer.py` with mock folder utilities.
- [x] Implement checks for model config parsing, history extraction, and dynamic best epoch calculations.
  - **Verification**: Run `pytest tests/test_visualizer.py` (should fail or pass depending on current code state).

## Phase 2: Refactor load_results & Implement Dynamic Selection
- [x] Refactor `load_results()` in `visualize.py` to keep the raw history log lists.
- [x] Add the dynamic selection dropdown selector in the Streamlit sidebar.
- [x] Implement the `resolve_best_epoch(history, metric, mode)` logic to parse metrics at the chosen epoch dynamically.
- [x] Update dashboard leaderboard, curve overlays, and comparative tabs.
  - **Verification**: Verify that changing selection dynamically updates all displayed metrics.

## Phase 3: Verification & Polish
- [x] Execute `pytest tests/` to confirm all shape, PEFT, and visualizer tests pass.
- [x] Launch Streamlit dashboard to perform visual sanity checks on the UI.
