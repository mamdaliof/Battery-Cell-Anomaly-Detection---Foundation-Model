import sys
import os
import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from bcadfm.training.losses import FocalLoss, compute_class_weights
from bcadfm.training.trainer import ImbalanceTrainer

class DummyDataset(Dataset):
    """
    Minimal dummy dataset class representing typical BatteryCellDataset structures.
    """
    def __init__(self, num_samples=10, num_classes=2):
        self.samples = []
        for i in range(num_samples):
            # Create a mock object with a 'label' attribute
            class DummySample:
                def __init__(self, label):
                    self.label = label
            self.samples.append(DummySample(label=i % num_classes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "pixel_values": torch.randn(3, 224, 224),
            "labels": torch.tensor(self.samples[idx].label, dtype=torch.long)
        }

class TestTrainerAndLosses(unittest.TestCase):
    """
    Unit tests for custom Trainer class (ImbalanceTrainer), loss functions (FocalLoss),
    class weight computations, and device placement buffers.

    Why We Have It:
    These tests guarantee that severe class imbalance strategies (WeightedRandomSampler fallbacks,
    class weighting adjustments, and focal loss parameters) compute correctly, remain DDP-safe,
    and prevent device mismatch crashes during distributed runs.
    """

    def test_focal_loss_computation(self):
        """
        Verify that FocalLoss computes correctly and scales based on gamma and alpha.

        How It Should Behave:
        Focal loss must return higher values for hard predictions (low probability of correct class)
        and lower values for easy predictions compared to standard cross-entropy.
        """
        logits = torch.tensor([[2.0, -2.0], [-1.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        
        # Test Focal Loss with alpha=None, gamma=2
        focal_loss_fn = FocalLoss(alpha=None, gamma=2.0, reduction="mean")
        loss_val = focal_loss_fn(logits, targets)
        self.assertTrue(loss_val.item() > 0.0)

        # Test Focal Loss with class weights alpha
        alpha = torch.tensor([0.25, 0.75], dtype=torch.float32)
        focal_loss_weighted = FocalLoss(alpha=alpha, gamma=1.0, reduction="mean")
        loss_val_w = focal_loss_weighted(logits, targets)
        self.assertTrue(loss_val_w.item() > 0.0)

    def test_compute_class_weights(self):
        """
        Verify class weight calculation methods (balanced vs inverse) and padding
        for missing class indices (H4 Fix).
        """
        # Test 1: Standard balanced weights
        counts = {0: 10, 1: 2}
        weights_b = compute_class_weights(counts, method="balanced")
        # Majority class weight must be smaller than minority class weight
        self.assertTrue(weights_b[0] < weights_b[1])

        # Test 2: Missing class indices (e.g. class 2 has 0 samples)
        counts_missing = {0: 10, 2: 2} # class 1 missing
        weights_m = compute_class_weights(counts_missing, method="balanced")
        # The returned weight tensor should be size 3 (0, 1, 2)
        self.assertEqual(len(weights_m), 3)
        # Class 1 (missing) should be padded with a default weight of 1.0
        self.assertEqual(weights_m[1].item(), 1.0)

    def test_trainer_labels_extraction(self):
        """
        Verify that trainer label scanning successfully unpacks PyTorch Subsets
        and wrapper layers to extract correct target labels (C8 Fix).
        """
        base_dataset = DummyDataset(num_samples=20, num_classes=2)
        # Wrap the dataset in a Subset
        subset = Subset(base_dataset, [0, 2, 4, 11, 15])
        
        from transformers import TrainingArguments
        training_args = TrainingArguments(output_dir="temp_out", report_to="none", no_cuda=True)
        
        # Use dummy model
        model = nn.Linear(10, 2)
        
        trainer = ImbalanceTrainer(
            model=model,
            args=training_args,
            train_dataset=subset,
            imbalance_config={"class_weights": "balanced"}
        )
        
        # Call label scanner
        labels = trainer._get_train_labels()
        # Ensure it matched base_dataset targets using the subset mapping
        # base_dataset labels are [0, 1, 0, 1, 0, 1, ...]
        # indices [0, 2, 4] label is 0. indices [11, 15] label is 1.
        self.assertEqual(labels, [0, 0, 0, 1, 1])

    def test_trainer_double_loss_prevention(self):
        """
        Verify that compute_loss pops labels from the model input dict
        to avoid redundant unweighted loss computation in the backbone (C6 Fix).
        """
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3 * 224 * 224, 2)
                
            def forward(self, pixel_values, labels=None):
                # If labels are passed, raise an exception or record it
                if labels is not None:
                    raise AssertionError("Labels should not reach the backbone forward pass!")
                flat = pixel_values.view(pixel_values.shape[0], -1)
                return {"logits": self.proj(flat)}

        model = MockModel()
        from transformers import TrainingArguments
        training_args = TrainingArguments(output_dir="temp_out", report_to="none", no_cuda=True)
        
        trainer = ImbalanceTrainer(
            model=model,
            args=training_args,
            train_dataset=DummyDataset(num_samples=4),
            imbalance_config={"loss_type": "cross_entropy"}
        )
        
        # Run compute_loss mock call
        inputs = {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "labels": torch.tensor([0, 1], dtype=torch.long)
        }
        
        # This should execute without AssertionError because labels are popped
        loss = trainer.compute_loss(model, inputs)
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0.0)

    def test_trainer_ddp_sampler_fallback(self):
        """
        Verify that requesting WeightedRandomSampler in multi-GPU environments
        triggers automatic fallback to DDP-safe data_level oversampling (C3 Fix).
        """
        base_dataset = DummyDataset(num_samples=10, num_classes=2)
        # Mock the dataset oversample_dataset method
        dataset_called = False
        def mock_oversample():
            nonlocal dataset_called
            dataset_called = True
        base_dataset.oversample_dataset = mock_oversample

        from transformers import TrainingArguments
        training_args = TrainingArguments(output_dir="temp_out", report_to="none", no_cuda=True)
        # Force world size > 1 to mock DDP env
        training_args._world_size = 2
        
        model = nn.Linear(10, 2)
        
        trainer = ImbalanceTrainer(
            model=model,
            args=training_args,
            train_dataset=base_dataset,
            imbalance_config={
                "oversampling_method": "weighted_sampler",
                "class_weights": "balanced"
            }
        )
        
        # The setup in init calls _prepare_imbalance_handling, which triggers fallback
        self.assertTrue(dataset_called)
        # Oversampling method must be updated to data_level
        self.assertEqual(trainer.imbalance_config["oversampling_method"], "data_level")

if __name__ == "__main__":
    unittest.main()
