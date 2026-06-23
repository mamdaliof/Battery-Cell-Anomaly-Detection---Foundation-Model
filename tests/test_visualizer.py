import os
import tempfile
import yaml
import json
import pandas as pd
from pathlib import Path
from visualize import load_results

def test_load_results():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create mock classification run
        cls_run_dir = tmp_path / "cls__vit-base__smoke_run" / "20260611_120000"
        cls_run_dir.mkdir(parents=True)
        
        cls_config = {
            "model_name": "google/vit-base-patch16-224",
            "num_epochs": 3,
            "learning_rate": 0.0003,
            "peft": {
                "type": "lora",
                "lora_r: ": 8
            },
            "imbalance": {
                "oversampling_method": "weighted_sampler",
                "loss_type": "focal"
            }
        }
        with open(cls_run_dir / "config.yaml", "w") as f:
            yaml.dump(cls_config, f)
            
        cls_state = {
            "log_history": [
                {"epoch": 1.0, "step": 10, "loss": 0.5, "eval_loss": 0.4, "eval_f1": 0.7},
                {"epoch": 2.0, "step": 20, "loss": 0.3, "eval_loss": 0.2, "eval_f1": 0.8},
                {"epoch": 3.0, "step": 30, "loss": 0.2, "eval_loss": 0.3, "eval_f1": 0.75}
            ]
        }
        with open(cls_run_dir / "trainer_state.json", "w") as f:
            json.dump(cls_state, f)

        # 2. Create mock detection run
        det_run_dir = tmp_path / "det__yolo26__smoke_run" / "20260611_120500"
        det_run_dir.mkdir(parents=True)
        
        det_config = {
            "model_name": "yolo26_dino",
            "yolo_model_config": "yolo26_dino.yaml",
            "yolo_data_yaml": "data/det_v1.0/battery_detection_abnormal_only.yaml",
            "num_epochs": 2,
            "learning_rate": 0.0001,
            "peft": {
                "type": "adapter",
                "adapter_bottleneck_dim": 64
            }
        }
        with open(det_run_dir / "config.yaml", "w") as f:
            yaml.dump(det_config, f)
            
        det_state = {
            "log_history": [
                {"epoch": 1.0, "step": 5, "loss": 1.2, "eval_loss": 0.9, "eval_mAP50": 0.5, "eval_custom_cls_f1/abnormal": 0.45},
                {"epoch": 2.0, "step": 10, "loss": 0.7, "eval_loss": 0.6, "eval_mAP50": 0.8, "eval_custom_cls_f1/abnormal": 0.75}
            ]
        }
        with open(det_run_dir / "trainer_state.json", "w") as f:
            json.dump(det_state, f)

        # 3. Call load_results
        df = load_results(tmp_dir)
        
        # 4. Asserts
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) == 2, f"Should load exactly 2 runs, found {len(df)}"
        
        # Check classification run parsing
        cls_row = df[df["task"] == "Classification"].iloc[0]
        assert cls_row["model"] == "google/vit-base-patch16-224"
        assert cls_row["peft_type"] == "lora"
        assert cls_row["loss_type"] == "focal"
        assert cls_row["lr"] == 0.0003
        assert cls_row["dataset"] == "cls_v1.0"
        assert "total_params" in cls_row
        assert "trainable_params" in cls_row
        
        # Check detection run parsing
        det_row = df[df["task"] == "Detection"].iloc[0]
        assert det_row["model"] == "yolo26_dino"
        assert det_row["peft_type"] == "adapter"
        assert det_row["lr"] == 0.0001
        assert det_row["dataset"] == "battery_detection_abnormal_only"
        assert "total_params" in det_row
        assert "trainable_params" in det_row
        
        # Validate that history list is present (to be added in refactoring)
        assert "history" in cls_row, "history should be preserved in df rows"
        assert len(cls_row["history"]) == 3

        # 5. Test Dynamic Benchmark Selection: eval_loss (Min)
        from visualize import get_best_epoch_metrics, update_best_metrics_inplace
        
        # Test get_best_epoch_metrics directly
        best_step_cls = get_best_epoch_metrics(cls_row["history"], "eval_loss", "min")
        assert best_step_cls["epoch"] == 2.0
        assert best_step_cls["eval_f1"] == 0.8
        
        # Test update_best_metrics_inplace on DataFrame
        df_eval_loss = df.copy()
        update_best_metrics_inplace(df_eval_loss, "eval_loss", "min")
        
        cls_row_updated = df_eval_loss[df_eval_loss["task"] == "Classification"].iloc[0]
        assert cls_row_updated["best_metrics"]["epoch"] == 2.0
        assert cls_row_updated["best_eval_f1"] == 0.8
        assert cls_row_updated["best_eval_loss"] == 0.2
        
        det_row_updated = df_eval_loss[df_eval_loss["task"] == "Detection"].iloc[0]
        assert det_row_updated["best_metrics"]["epoch"] == 2.0
        assert det_row_updated["best_eval_f1"] == 0.75
        assert det_row_updated["best_eval_loss"] == 0.6

        # 6. Test Dynamic Benchmark Selection: loss (Min Train Loss)
        df_train_loss = df.copy()
        update_best_metrics_inplace(df_train_loss, "loss", "min")
        
        cls_row_train = df_train_loss[df_train_loss["task"] == "Classification"].iloc[0]
        assert cls_row_train["best_metrics"]["epoch"] == 3.0
        assert cls_row_train["best_eval_f1"] == 0.75
        assert cls_row_train["best_eval_loss"] == 0.3

def test_fold_parsing_and_averaging():
    from visualize import parse_fold_and_base_name, group_results_by_fold
    
    # 1. Test parse_fold_and_base_name
    fold, base = parse_fold_and_base_name("vit__peft_smoke_all_label_fold_0", "peft_smoke_all_label_fold_0")
    assert fold == "0"
    assert base == "peft_smoke_all_label"

    fold, base = parse_fold_and_base_name("vit__peft-smoke-fold-3", "peft-smoke-fold-3")
    assert fold == "3"
    assert base == "peft-smoke"

    fold, base = parse_fold_and_base_name("vit__peft_smoke_all_label", "peft_smoke_all_label", config_fold=2)
    assert fold == "2"
    assert base == "peft_smoke_all_label"

    # 2. Test grouping/averaging
    data = [
        {
            "task": "Classification", "model": "vit", "peft_type": "lora", "peft_detail": "r=8",
            "imbalance_strategy": "none", "loss_type": "ce", "lr": 0.001, "epochs_configured": 5,
            "dataset": "ds", "short_cfg_name": "cfg_fold_0", "display_name": "cfg_fold_0",
            "fold": "0", "base_cfg_name": "cfg", "completed": True,
            "best_eval_f1": 0.8, "best_eval_loss": 0.2, "img_abnormal_f1": 0.8, "img_abnormal_auroc": 0.85,
            "final_train_loss": 0.1, "total_params": 100, "trainable_params": 10, "pct_trainable": 10.0
        },
        {
            "task": "Classification", "model": "vit", "peft_type": "lora", "peft_detail": "r=8",
            "imbalance_strategy": "none", "loss_type": "ce", "lr": 0.001, "epochs_configured": 5,
            "dataset": "ds", "short_cfg_name": "cfg_fold_1", "display_name": "cfg_fold_1",
            "fold": "1", "base_cfg_name": "cfg", "completed": True,
            "best_eval_f1": 0.9, "best_eval_loss": 0.1, "img_abnormal_f1": 0.9, "img_abnormal_auroc": 0.95,
            "final_train_loss": 0.08, "total_params": 100, "trainable_params": 10, "pct_trainable": 10.0
        }
    ]
    df = pd.DataFrame(data)
    df_grouped = group_results_by_fold(df)
    
    assert not df_grouped.empty
    assert len(df_grouped) == 1
    row = df_grouped.iloc[0]
    assert row["short_cfg_name"] == "cfg"
    assert row["completed_folds_count"] == 2
    assert row["total_folds_count"] == 2
    assert abs(row["best_eval_f1"] - 0.85) < 1e-5
    assert abs(row["best_eval_loss"] - 0.15) < 1e-5
    assert abs(row["img_abnormal_f1"] - 0.85) < 1e-5
    assert abs(row["img_abnormal_auroc"] - 0.9) < 1e-5
    assert abs(row["final_train_loss"] - 0.09) < 1e-5


def test_hex_to_rgba():
    from visualize import hex_to_rgba
    rgba = hex_to_rgba("#4facfe", 0.2)
    assert rgba == "rgba(79, 172, 254, 0.2)"
    rgba_default = hex_to_rgba("00f2fe")
    assert rgba_default == "rgba(0, 242, 254, 0.15)"

def test_estimate_model_params():
    from visualize import estimate_model_params
    
    # 1. Classification, LoRA
    est1 = estimate_model_params(
        model_name="vitb16",
        task="Classification",
        peft_type="lora",
        peft_config={"lora_r": 8, "lora_target_blocks": [0, 1, 2, 3]}
    )
    # vit_b_params = 85955328, d = 768, target_blocks count = 4
    # head_params = 1538
    # peft = 4 * (4 * 8 * 768) = 98304
    assert est1["trainable"] == 1538 + 98304
    
    # 2. Classification, Visual Prompt Tuning (VPT)
    est2 = estimate_model_params(
        model_name="vit-base",
        task="Classification",
        peft_type="visual_prompt",
        peft_config={"vpt_num_tokens": 10, "vpt_deep": True, "vpt_target_blocks": [0, 1]}
    )
    # vpt_deep with 2 targeted blocks: 2 * 10 * 768 = 15360
    assert est2["trainable"] == 1538 + 15360

    # 3. Detection, standard YOLO
    est3 = estimate_model_params(
        model_name="yolo11n.pt",
        task="Detection",
        peft_type="none",
        peft_config={}
    )
    assert est3["total"] == 2600000
    assert est3["trainable"] == 2600000

def test_early_stopping_detection():
    # Simulate a run that stopped early
    run_data = {
        "epochs_configured": 10,
        "history": [
            {"epoch": 1.0, "step": 10, "loss": 0.5},
            {"epoch": 2.0, "step": 20, "loss": 0.4},
            {"epoch": 3.0, "step": 30, "loss": 0.35}
        ]
    }
    
    epochs_configured = run_data.get("epochs_configured", 0)
    history = run_data.get("history", [])
    run_epochs = [entry.get("epoch") for entry in history if "epoch" in entry]
    max_run_epoch = max(run_epochs) if run_epochs else 0
    
    stopped_early = max_run_epoch > 0 and epochs_configured > 0 and max_run_epoch < epochs_configured
    assert stopped_early is True

def test_caching_decorator():
    from visualize import load_results
    assert hasattr(load_results, "clear"), "load_results should be a cached function with a .clear() method"

def test_main_execution():
    from unittest.mock import MagicMock, patch
    import sys
    
    # Create mock streamlit
    mock_st = MagicMock()
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.text_input.return_value = "outputs"
    mock_st.sidebar.radio.return_value = "All Runs"
    mock_st.sidebar.checkbox.return_value = True
    
    def selectbox_side_effect(label, options, *args, **kwargs):
        if "Metric Source Mode" in label:
            return "Custom Classification Metrics (by Class)"
        elif "Target Class Label" in label:
            return "abnormal"
        elif "Benchmark Metric" in label:
            return options[0]
        elif "Task Profile" in label:
            return "All Tasks"
        elif "Future Optimizer Type" in label:
            return "AdamW (Active)"
        elif "Select Class Label for Confusion Matrix" in label:
            return "abnormal"
        return options[0]

    mock_st.sidebar.selectbox.side_effect = selectbox_side_effect
    mock_st.selectbox.side_effect = selectbox_side_effect

    def columns_side_effect(spec, *args, **kwargs):
        if isinstance(spec, int):
            return [MagicMock() for _ in range(spec)]
        elif isinstance(spec, list):
            return [MagicMock() for _ in range(len(spec))]
        return [MagicMock(), MagicMock()]

    mock_st.columns.side_effect = columns_side_effect

    def tabs_side_effect(tab_names, *args, **kwargs):
        return [MagicMock() for _ in range(len(tab_names))]

    mock_st.tabs.side_effect = tabs_side_effect
    
    # Mock load_results to return a small mock dataframe with duplicate configuration names
    mock_df = pd.DataFrame([
        {
            "dir": "run_0",
            "display_name": "run_0_fold_0_fold_0",
            "short_cfg_name": "run_0",
            "base_cfg_name": "run_0",
            "fold": "0",
            "task": "Classification",
            "model": "vit",
            "peft_type": "lora",
            "peft_detail": "r=8",
            "imbalance_strategy": "none",
            "loss_type": "ce",
            "lr": 0.0003,
            "epochs_configured": 5,
            "custom_param": "default",
            "best_eval_f1": 0.8,
            "best_eval_loss": 0.2,
            "final_train_loss": 0.1,
            "completed": True,
            "best_metrics": {"epoch": 4, "eval_loss": 0.2, "eval_f1": 0.8},
            "history": [{"epoch": 1, "step": 10, "eval_loss": 0.4, "eval_f1": 0.6}, {"epoch": 4, "step": 40, "eval_loss": 0.2, "eval_f1": 0.8}],
            "img_abnormal_f1": 0.8,
            "img_abnormal_auroc": 0.85,
            "abnormal_class_name": "abnormal",
            "dataset": "ds_v1",
            "total_params": 1000,
            "trainable_params": 100,
            "pct_trainable": 10.0,
            "fold_runs": []
        },
        {
            "dir": "run_1",
            "display_name": "run_0_fold_1_fold_1",
            "short_cfg_name": "run_0",  # Duplicate Configuration name to trigger deduplication
            "base_cfg_name": "run_0",
            "fold": "1",
            "task": "Classification",
            "model": "vit",
            "peft_type": "lora",
            "peft_detail": "r=8",
            "imbalance_strategy": "none",
            "loss_type": "ce",
            "lr": 0.0003,
            "epochs_configured": 5,
            "custom_param": "default",
            "best_eval_f1": 0.85,
            "best_eval_loss": 0.18,
            "final_train_loss": 0.09,
            "completed": True,
            "best_metrics": {"epoch": 4, "eval_loss": 0.18, "eval_f1": 0.85},
            "history": [{"epoch": 1, "step": 10, "eval_loss": 0.4, "eval_f1": 0.6}, {"epoch": 4, "step": 40, "eval_loss": 0.18, "eval_f1": 0.85}],
            "img_abnormal_f1": 0.85,
            "img_abnormal_auroc": 0.90,
            "abnormal_class_name": "abnormal",
            "dataset": "ds_v1",
            "total_params": 1000,
            "trainable_params": 100,
            "pct_trainable": 10.0,
            "fold_runs": []
        }
    ])
    
    try:
        from visualize import main
        with patch('visualize.load_results', return_value=mock_df), patch('visualize.st', mock_st):
            main()
    except Exception as e:
        raise e


if __name__ == "__main__":
    test_load_results()
    test_fold_parsing_and_averaging()
    test_hex_to_rgba()
    test_estimate_model_params()
    test_early_stopping_detection()
    test_caching_decorator()
    test_main_execution()
    print("ALL TESTS PASSED!")

