# Specification: Visualizer Enhancements

## Requirements

1. **Dynamic Benchmark Epoch Selection**:
   - Add a UI selector in the Streamlit dashboard to pick the target benchmark metric (e.g., `Validation Loss (Min)`, `Validation F1 (Max)`, `mAP50 (Max)`, `Training Loss (Min)`).
   - Dynamically compute the best epoch for each run based on the selected benchmark.
   - Report and compare all other metrics at that specific chosen epoch in the Leaderboard, inspector, and comparison tabs.

2. **Code Robustness & Debugging**:
   - Scan and parse all trainer states safely. Handle empty or interrupted histories without raising exceptions.
   - Refactor `load_results` to expose clean Pandas columns for sorting and filtering.

3. **Visualizer Testing**:
   - Write a unit test suite `tests/test_visualizer.py` verifying config loading, JSON parsing, and dynamic epoch selection logic.

## Acceptance Criteria
- Streamlit visualizer runs without syntax or runtime exceptions.
- Changing the benchmark metric updates the reported F1, loss, and confusion matrices dynamically in the UI.
- `pytest tests/test_visualizer.py` passes successfully in the `pytorch` environment.
