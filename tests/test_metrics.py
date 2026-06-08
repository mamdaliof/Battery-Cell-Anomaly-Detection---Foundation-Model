import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from bcadfm.metrics.cls_metrics import compute_cls_metrics
from bcadfm.metrics.cls_callbacks import SaveBestModelCallback

class TestMetricsAndCallbacks(unittest.TestCase):
    """
    Unit tests for evaluation metrics (compute_cls_metrics) and training callbacks
    (SaveBestModelCallback).

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
        Verify that SaveBestModelCallback tracks metric improvements correctly
        for both higher-is-better and lower-is-better configurations.
        """
        # Test 1: Higher is better (F1)
        callback_f1 = SaveBestModelCallback(
            metric_for_best="eval_f1",
            greater_is_better=True,
            output_dir="temp_best"
        )
        
        # Initial improvement check
        self.assertTrue(callback_f1._is_improvement(current_val=0.8, best_val=None))
        # Better value
        self.assertTrue(callback_f1._is_improvement(current_val=0.85, best_val=0.8))
        # Worse value
        self.assertFalse(callback_f1._is_improvement(current_val=0.75, best_val=0.8))

        # Test 2: Lower is better (Loss)
        callback_loss = SaveBestModelCallback(
            metric_for_best="eval_loss",
            greater_is_better=False,
            output_dir="temp_best"
        )
        
        # Initial check
        self.assertTrue(callback_loss._is_improvement(current_val=0.5, best_val=None))
        # Better value (lower)
        self.assertTrue(callback_loss._is_improvement(current_val=0.4, best_val=0.5))
        # Worse value (higher)
        self.assertFalse(callback_loss._is_improvement(current_val=0.6, best_val=0.5))

if __name__ == "__main__":
    unittest.main()
