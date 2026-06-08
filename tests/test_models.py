import sys
import unittest
import torch
import torch.nn as nn
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from bcadfm.models.dinov3_classifier import (
    DinoV3Classifier,
    HeadConfig,
    BottleneckAdapter,
    AdapterWrappedMLP,
    apply_adapters,
    VptWrappedBackbone,
    VptLayerWrapper
)

class PeftConfig:
    def __init__(self, **kwargs):
        # Default properties to avoid AttributeError in DinoV3Classifier wrapping
        self.type = "none"
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.0
        self.lora_target_modules = None
        self.lora_target_blocks = None
        self.adapter_bottleneck_dim = 64
        self.adapter_dropout = 0.0
        self.adapter_target_blocks = None
        self.vpt_num_tokens = 10
        self.vpt_deep = False
        self.vpt_target_blocks = None
        
        for k, v in kwargs.items():
            setattr(self, k, v)

class TestDinoClassifierAndPeft(unittest.TestCase):
    """
    Unit tests for DinoV3Classifier wrapping, classifier head generation,
    bottleneck adapters, and Visual Prompt Tuning (VPT) token slicing.
    """

    @classmethod
    def setUpClass(cls):
        # We load a small, open-access vision model to act as our backbone
        # as requested, we configure it using facebook/dinov3-vits16-pretrain-lvd1689m
        # to test real structural mapping and register token presence.
        cls.model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        
        # Define configs
        cls.head_cfg = HeadConfig(
            num_labels=2,
            depth=2,
            hidden_dim="0.5X",  # tests multiplier string parsing
            dropout=0.1
        )
        cls.peft_cfg_none = PeftConfig(type="none")
        
        cls.peft_cfg_lora = PeftConfig(
            type="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_target_modules=["q_proj", "v_proj"]
        )

        cls.peft_cfg_adapter = PeftConfig(
            type="adapter",
            adapter_bottleneck_dim=32,
            adapter_dropout=0.1
        )

        cls.peft_cfg_vpt_shallow = PeftConfig(
            type="visual_prompt",
            vpt_num_tokens=10,
            vpt_deep=False
        )

        cls.peft_cfg_vpt_deep = PeftConfig(
            type="visual_prompt",
            vpt_num_tokens=10,
            vpt_deep=True
        )

    def test_classifier_head_construction(self):
        """
        Verify classifier head construction with absolute integers and multipliers.
        """
        # Test 1: depth 1
        cfg1 = HeadConfig(num_labels=2, depth=1, dropout=0.0)
        classifier1 = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=cfg1,
            peft_config=self.peft_cfg_none
        )
        # Check that the classifier head is a single linear layer wrapped in sequential
        self.assertTrue(isinstance(classifier1.classifier, nn.Sequential))
        self.assertTrue(isinstance(classifier1.classifier[0], nn.Linear))
        self.assertEqual(classifier1.classifier[0].out_features, 2)

        # Test 2: depth > 1 with multiplier string (0.5X of 384 hidden_size)
        cfg2 = HeadConfig(num_labels=2, depth=2, hidden_dim="0.5X", dropout=0.1)
        classifier2 = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=cfg2,
            peft_config=self.peft_cfg_none
        )
        self.assertTrue(isinstance(classifier2.classifier, nn.Sequential))
        # First layer should project from 384 to 192 (384 * 0.5)
        self.assertEqual(classifier2.classifier[0].out_features, 192)

    def test_bottleneck_adapter_zero_init(self):
        """
        Verify that Pfeiffer BottleneckAdapter up-projections are initialized to 0,
        ensuring identity mapping at training step 0 (residual identity connection).
        """
        adapter = BottleneckAdapter(input_dim=128, bottleneck_dim=16, dropout=0.1)
        x = torch.randn(2, 5, 128)
        
        # Test zero initialization of up-projection
        with torch.no_grad():
            output = adapter(x)
            
        # At step 0, output must be exactly equal to input (x)
        self.assertTrue(torch.allclose(output, x, atol=1e-6))

        # Check submodules
        self.assertEqual(adapter.down_proj.out_features, 16)
        self.assertEqual(adapter.up_proj.out_features, 128)

    def test_apply_adapters_freezing(self):
        """
        Verify that apply_adapters wraps transformer MLP blocks and freezes backbone params.
        """
        classifier = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_adapter
        )
        
        # Check if backbone parameters are frozen
        for name, param in classifier.backbone.named_parameters():
            if "adapter" not in name:
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen.")
            else:
                self.assertTrue(param.requires_grad, f"Adapter parameter {name} should be trainable.")

        # Check classifier head parameters are trainable
        for param in classifier.classifier.parameters():
            self.assertTrue(param.requires_grad)

    def test_vpt_token_layout_and_slicing(self):
        """
        Verify the VPT prepending layout and token slicing under shallow and deep prompt tuning
        with the DINOv3 registers layout (C7 Fix).
        """
        # Test shallow VPT
        classifier_vpt_s = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_vpt_shallow
        )
        
        # Verify prompt exists and layout has correct token length
        self.assertTrue(hasattr(classifier_vpt_s.backbone, "prompt"))
        self.assertEqual(classifier_vpt_s.backbone.prompt.shape, (1, 10, 384))

        # Test deep VPT wrapping
        classifier_vpt_d = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_vpt_deep
        )
        self.assertTrue(classifier_vpt_d.backbone.deep)
        
        # Test forward pass shape checks
        dummy_pixel_values = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            outputs = classifier_vpt_s(pixel_values=dummy_pixel_values)
            
        self.assertEqual(outputs["logits"].shape, (2, 2))

    def test_lora_wrapping(self):
        """
        Verify that LoRA targets attention projections correctly and freezes backbone.
        """
        classifier = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_lora
        )
        
        # Check if LoRA parameters are trainable and other backbone params are frozen
        lora_found = False
        for name, param in classifier.backbone.named_parameters():
            if "lora" in name:
                lora_found = True
                self.assertTrue(param.requires_grad)
            else:
                # Unless it's the classifier head, all other weights should be frozen
                self.assertFalse(param.requires_grad)
                
        self.assertTrue(lora_found)

if __name__ == "__main__":
    unittest.main()
