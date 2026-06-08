import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from bcadfm.metrics.cls_metrics import compute_cls_metrics
from bcadfm.metrics.cls_callbacks import SaveTwoBestClsModelsCallback

class TestMetricsAndCallbacks(unittest.TestCase):
    """
    Unit tests for evaluation metrics (compute_cls_metrics) and training callbacks
    (SaveTwoBestClsModelsCallback).

    Why We Have It:
    These tests ensure that anomaly detection classification statistics (F1, AUROC, confusion matrix)
    are calculated correctly under extreme imbalance and handle zero-division edge cases gracefully.
    """

    def test_compute_cls_metrics_standard(self):
        """
        Verify that compute_cls_metrics computes metrics correctly for balanced predictions.
        """
        # Create mock predictions: 4 samples
        # logits: shape [B, C]
        logits = np.array([
            [1.5, -1.5],  # Pred: 0, True: 0 (TN)
            [-0.5, 0.5],  # Pred: 1, True: 1 (TP)
            [0.2, -0.2],  # Pred: 0, True: 1 (FN)
            [-1.0, 1.0]   # Pred: 1, True: 0 (FP)
        ])
        labels = np.array([0, 1, 1, 0])
        
        from transformers.trainer_utils import EvalPrediction
        eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
        
        metrics = compute_cls_metrics(eval_pred)
        
        # Accuracy should be 0.5 (2 correct out of 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        # Confusion matrix checks
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertIn("f1", metrics)
        self.assertIn("auroc", metrics)

    def test_compute_cls_metrics_single_class_edge_case(self):
        """
        Verify that compute_cls_metrics does not crash and returns fallback values
        when validation splits contain only a single target class (e.g. all normal).
        """
        logits = np.array([
            [2.0, -2.0],
            [1.5, -1.5],
            [3.0, -3.0]
        ])
        labels = np.array([0, 0, 0])  # Only class 0 present
        
        from transformers.trainer_utils import EvalPrediction
        eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
        
        # This should execute and fallback for AUROC gracefully (returning 0.5 or 0.0 instead of crashing)
        metrics = compute_cls_metrics(eval_pred)
        
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["tn"], 3)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"], 0)
        # F1 and AUROC should be handled gracefully
        self.assertTrue("f1" in metrics)
        self.assertTrue("auroc" in metrics)

    def test_best_model_callback_comparison(self):
        """
        Verify that SaveTwoBestClsModelsCallback tracks metric improvements correctly
        for eval_loss and eval_f1.
        """
        from transformers import TrainingArguments, TrainerState, TrainerControl
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            callback = SaveTwoBestClsModelsCallback(run_dir=temp_dir)
            
            # Create dummy args, state, control
            args = TrainingArguments(output_dir=temp_dir, report_to="none")
            state = TrainerState()
            state.is_world_process_zero = True
            control = TrainerControl()
            
            # Verify initial values
            self.assertEqual(callback.best_loss, float("inf"))
            self.assertEqual(callback.best_f1, float("-inf"))
            
            # 1. First evaluation: metrics improve
            metrics1 = {"eval_loss": 0.5, "eval_f1": 0.8}
            # We mock the model and torch.save to prevent actual writing during this check
            class DummyModel:
                def state_dict(self):
                    return {"weight": torch.tensor([1.0])}
            
            import unittest.mock as mock
            with mock.patch("torch.save") as mock_save:
                callback.on_evaluate(args, state, control, metrics1, model=DummyModel())
                self.assertEqual(callback.best_loss, 0.5)
                self.assertEqual(callback.best_f1, 0.8)
                self.assertEqual(mock_save.call_count, 2)  # saves both loss and f1

            # 2. Second evaluation: loss improves, f1 does not
            metrics2 = {"eval_loss": 0.4, "eval_f1": 0.75}
            with mock.patch("torch.save") as mock_save:
                callback.on_evaluate(args, state, control, metrics2, model=DummyModel())
                self.assertEqual(callback.best_loss, 0.4)
                self.assertEqual(callback.best_f1, 0.8)  # f1 remains 0.8
                self.assertEqual(mock_save.call_count, 1)  # only saves loss

            # 3. Third evaluation: f1 improves, loss does not
            metrics3 = {"eval_loss": 0.45, "eval_f1": 0.85}
            with mock.patch("torch.save") as mock_save:
                callback.on_evaluate(args, state, control, metrics3, model=DummyModel())
                self.assertEqual(callback.best_loss, 0.4)  # loss remains 0.4
                self.assertEqual(callback.best_f1, 0.85)
                self.assertEqual(mock_save.call_count, 1)  # only saves f1

        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()
