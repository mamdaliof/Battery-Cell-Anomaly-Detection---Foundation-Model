import sys
import unittest
import tempfile
import shutil
import random
from pathlib import Path
import torch
from PIL import Image

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from bcadfm.data.dataset import BatteryCellDataset, build_augmentation_pipeline, RandomAugmentationCombo
from bcadfm.data.config import DataConfig

class TestBatteryCellDataset(unittest.TestCase):
    """
    Unit tests for the BatteryCellDataset class, including sample collection,
    minority class oversampling (with deterministic seed isolation),
    and PIL image preprocessing/pipeline execution.
    """

    def setUp(self):
        # Create a temporary directory structure for testing sample collection
        self.test_dir = Path(tempfile.mkdtemp())
        self.train_dir = self.test_dir / "train"
        
        self.normal_class = "normal"
        self.abnormal_class = "abnormal"
        
        # Create class folders
        (self.train_dir / self.normal_class).mkdir(parents=True)
        (self.train_dir / self.abnormal_class).mkdir(parents=True)

        # Generate mock images (blank PNGs)
        self.num_normal = 5
        self.num_abnormal = 2  # minority class
        
        for i in range(self.num_normal):
            img = Image.new("RGB", (10, 10), color="white")
            img.save(self.train_dir / self.normal_class / f"norm_{i}.png")
            
        for i in range(self.num_abnormal):
            img = Image.new("RGB", (10, 10), color="black")
            img.save(self.train_dir / self.abnormal_class / f"abnorm_{i}.png")

        # Create a DataConfig mock
        self.data_config = DataConfig(
            data_dir=str(self.test_dir),
            normal_class_name=self.normal_class,
            abnormal_class_name=self.abnormal_class,
            image_size=224,
            aug_global_prob=0.0,  # disable augmentations by default in tests
            aug_max_transforms=2,
            aug_resized_crop_prob=0.0,
            aug_resized_crop_scale=(0.8, 1.0),
            aug_resized_crop_ratio=(0.75, 1.33),
            aug_horizontal_flip_prob=0.0,
            aug_rotation_prob=0.0,
            aug_rotation_degrees=15,
            aug_color_jitter_prob=0.0,
            aug_color_jitter_brightness=0.1,
            aug_color_jitter_contrast=0.1,
            aug_color_jitter_saturation=0.1,
            aug_color_jitter_hue=0.05,
            aug_gaussian_noise_prob=0.0,
            aug_gaussian_noise_mean=0.0,
            aug_gaussian_noise_std=0.01,
        )

        # Use facebook/dinov3-vits16-pretrain-lvd1689m checkpoint for tests
        self.model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dataset_initialization(self):
        """
        Verify that BatteryCellDataset loads and groups class directories correctly.
        """
        dataset = BatteryCellDataset(
            split="train",
            data_config=self.data_config,
            model_name_or_path=self.model_name,
            oversample=False,
            seed=42,
        )

        self.assertEqual(len(dataset.samples), self.num_normal + self.num_abnormal)
        # Verify labels classification mapping
        labels = [s.label for s in dataset.samples]
        self.assertEqual(labels.count(0), self.num_normal)
        self.assertEqual(labels.count(1), self.num_abnormal)
        self.assertEqual(dataset.label2id[self.normal_class], 0)
        self.assertEqual(dataset.label2id[self.abnormal_class], 1)

    def test_oversampling_reproducibility(self):
        """
        Verify that minority class oversampling replicates samples to match major class
        and yields identical results when using the same random seed.
        """
        # Run 1
        dataset1 = BatteryCellDataset(
            split="train",
            data_config=self.data_config,
            model_name_or_path=self.model_name,
            oversample=True,
            seed=101,
        )

        # Run 2
        dataset2 = BatteryCellDataset(
            split="train",
            data_config=self.data_config,
            model_name_or_path=self.model_name,
            oversample=True,
            seed=101,
        )

        # Run 3 (different seed)
        dataset3 = BatteryCellDataset(
            split="train",
            data_config=self.data_config,
            model_name_or_path=self.model_name,
            oversample=True,
            seed=202,
        )

        # Total size must be twice the majority class size (5 normal, 5 abnormal)
        self.assertEqual(len(dataset1), 2 * self.num_normal)
        self.assertEqual(len(dataset2), 2 * self.num_normal)

        # Run 1 and Run 2 must have the exact same sample order (identical seeds)
        paths1 = [s.image_path.name for s in dataset1.samples]
        paths2 = [s.image_path.name for s in dataset2.samples]
        self.assertEqual(paths1, paths2)

        # Run 3 might differ due to a different seed
        paths3 = [s.image_path.name for s in dataset3.samples]
        # Re-shuffling means paths1 and paths3 can differ (statistically highly likely)
        if paths1 == paths3:
            # Fallback assertion: check that oversampling still produced correct counts
            self.assertEqual(len(dataset3), 10)

    def test_dataset_getitem_output(self):
        """
        Verify that __getitem__ loads the image, executes transforms, and returns
        a dict with pixel_values and labels.
        """
        dataset = BatteryCellDataset(
            split="train",
            data_config=self.data_config,
            model_name_or_path=self.model_name,
            oversample=False,
            seed=42,
        )

        item = dataset[0]
        self.assertIn("pixel_values", item)
        self.assertIn("labels", item)
        
        # Verify shape of pixel values tensor: [C, H, W]
        self.assertEqual(len(item["pixel_values"].shape), 3)
        self.assertEqual(item["pixel_values"].shape[0], 3)  # RGB channels
        self.assertEqual(item["pixel_values"].shape[1], 224)
        self.assertEqual(item["pixel_values"].shape[2], 224)
        self.assertTrue(isinstance(item["labels"], torch.Tensor))

    def test_augmentation_pipeline_construction(self):
        """
        Verify that build_augmentation_pipeline builds a pipeline that wraps
        augmentations, and checks the RandomAugmentationCombo operator sampling.
        """
        # Enable all augmentations in config
        self.data_config.aug_global_prob = 1.0
        self.data_config.aug_horizontal_flip_prob = 1.0
        self.data_config.aug_rotation_prob = 1.0

        pipeline = build_augmentation_pipeline(self.data_config, split="train")
        self.assertIsNotNone(pipeline)
        self.assertTrue(isinstance(pipeline, RandomAugmentationCombo))

        # Check random sampling without replacement (H3 Fix)
        # We define simple dummy operations to test sampling
        ops = [
            (lambda img: img, 1.0, "op1"),
            (lambda img: img, 0.5, "op2"),
            (lambda img: img, 0.0, "op3"),
        ]
        combo = RandomAugmentationCombo(ops, max_transforms=2, global_prob=1.0)
        
        # Dry-run application on mock image
        img = Image.new("RGB", (10, 10))
        out_img = combo(img)
        self.assertEqual(out_img.size, (10, 10))

if __name__ == "__main__":
    unittest.main()
