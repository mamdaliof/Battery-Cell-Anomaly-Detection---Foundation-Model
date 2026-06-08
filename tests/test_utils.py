import sys
import unittest
import tempfile
import yaml
from pathlib import Path
import torch
import torch.nn as nn

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from bcadfm.utils.config import load_yaml_config, TrainingConfig
from bcadfm.utils.model_utils import count_parameters
from bcadfm.utils.yolo_utils import register_yolo_dino

class TestConfigAndUtilities(unittest.TestCase):
    """
    Unit tests for configuration loaders, parameter counting utility methods,
    and YOLO monkey-patching hooks.

    Why We Have It:
    These tests ensure that configuration defaults (including seeds) parse correctly
    and that dynamic registration modules are successfully injected without breaking
    vendor namespaces.
    """

    def setUp(self):
        # Create a temporary config file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        self.config_dict = {
            "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "output_dir": "outputs/test_run",
            "num_epochs": 5,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "early_stopping_patience": 3,
            "metric_for_best": "eval_f1",
            "greater_is_better": True,
            "seed": 1234,  # custom seed check
            "data": {
                "data_dir": "data",
                "normal_class_name": "normal",
                "abnormal_class_name": "abnormal",
                "image_size": 224,
                "aug_global_prob": 0.5,
                "aug_max_transforms": 2,
                "random_resized_crop_prob": 0.5,
                "random_resized_crop_scale": [0.8, 1.0],
                "random_resized_crop_ratio": [0.75, 1.33],
                "horizontal_flip_prob": 0.5,
                "rotation_prob": 0.5,
                "rotation_degrees": 15.0,
                "color_jitter_prob": 0.5,
                "color_jitter_brightness": 0.1,
                "color_jitter_contrast": 0.1,
                "color_jitter_saturation": 0.1,
                "color_jitter_hue": 0.05,
                "gaussian_noise_prob": 0.5,
                "gaussian_noise_std": 0.01,
            },
            "peft": {
                "type": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lora_target_modules": ["q_proj", "v_proj"],
                "lora_target_blocks": None,
                "adapter_bottleneck_dim": 64,
                "adapter_dropout": 0.1,
                "vpt_num_tokens": 10,
                "vpt_deep": False,
            },
            "head": {
                "num_labels": 2,
                "depth": 2,
                "hidden_dim": "0.5X",
                "dropout": 0.1,
            },
            "imbalance": {
                "oversampling_method": "data_level",
                "class_weights": "balanced",
                "loss_type": "focal",
                "focal_gamma": 2.0,
                "focal_alpha": None,
            },
            "scheduler": {
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
            },
            "amp": {
                "fp16": False,
                "bf16": True,
            }
        }
        with open(self.temp_file.name, "w") as f:
            yaml.safe_dump(self.config_dict, f)

    def tearDown(self):
        import os
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_load_yaml_config_parsing(self):
        """
        Verify that YAML properties are correctly parsed into TrainingConfig structures
        and that the global seed is successfully resolved (H8 Fix).
        """
        cfg = load_yaml_config(self.temp_file.name)
        self.assertTrue(isinstance(cfg, TrainingConfig))
        self.assertEqual(cfg.seed, 1234)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.peft.type, "lora")
        self.assertEqual(cfg.imbalance.loss_type, "focal")
        self.assertTrue(cfg.amp.bf16)

    def test_count_parameters(self):
        """
        Verify count_parameters returns correct values for trainable and total weights.
        """
        model = nn.Sequential(
            nn.Linear(10, 5),  # 10*5 + 5 = 55 params
            nn.Linear(5, 2)    # 5*2 + 2 = 12 params (total = 67 params)
        )
        
        # All trainable
        params = count_parameters(model)
        total, trainable = params["total"], params["trainable"]
        self.assertEqual(total, 67)
        self.assertEqual(trainable, 67)

        # Freeze first layer
        for p in model[0].parameters():
            p.requires_grad = False
            
        params = count_parameters(model)
        total, trainable = params["total"], params["trainable"]
        self.assertEqual(total, 67)
        self.assertEqual(trainable, 12)

    def test_yolo_registration_hook(self):
        """
        Verify that register_yolo_dino monkey-patches tasks parser and binds custom modules
        to standard Ultralytics namespaces.
        """
        register_yolo_dino()
        
        # Check if modules are successfully registered in sys.modules and namespaces
        import sys
        tasks_module = sys.modules.get("ultralytics.nn.tasks")
        self.assertIsNotNone(tasks_module)
        
        self.assertTrue(hasattr(tasks_module, "DinoV3Backbone"))
        self.assertTrue(hasattr(tasks_module, "DinoV3SFP_P3"))
        self.assertTrue(hasattr(tasks_module, "DinoV3SFP_P4"))
        self.assertTrue(hasattr(tasks_module, "DinoV3SFP_P5"))

if __name__ == "__main__":
    unittest.main()
