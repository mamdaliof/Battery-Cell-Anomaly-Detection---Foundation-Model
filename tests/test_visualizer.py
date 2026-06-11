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

if __name__ == "__main__":
    test_load_results()
    print("ALL TESTS PASSED!")
