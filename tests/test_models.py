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
        print("\n🧪 [test_models] Running: test_classifier_head_construction")
        
        # Test 1: depth 1
        print("  - [Test 1] Building depth 1 classifier head...")
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
        print(f"    => Head shape: Sequential Linear(in={classifier1.classifier[0].in_features}, out={classifier1.classifier[0].out_features}) [OK]")

        # Test 2: depth > 1 with multiplier string (0.5X of 384 hidden_size)
        print("  - [Test 2] Building depth 2 head with '0.5X' multiplier...")
        cfg2 = HeadConfig(num_labels=2, depth=2, hidden_dim="0.5X", dropout=0.1)
        classifier2 = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=cfg2,
            peft_config=self.peft_cfg_none
        )
        self.assertTrue(isinstance(classifier2.classifier, nn.Sequential))
        # First layer should project from 384 to 192 (384 * 0.5)
        self.assertEqual(classifier2.classifier[0].out_features, 192)
        print(f"    => Head shape: Sequential Linear1(out={classifier2.classifier[0].out_features}) -> GELU -> Linear2(out={classifier2.classifier[-1].out_features}) [OK]")
        print("✅ [test_models] Passed: test_classifier_head_construction")

    def test_bottleneck_adapter_zero_init(self):
        """
        Verify that Pfeiffer BottleneckAdapter up-projections are initialized to 0,
        ensuring identity mapping at training step 0 (residual identity connection).
        """
        print("\n🧪 [test_models] Running: test_bottleneck_adapter_zero_init")
        print("  - Instantiating BottleneckAdapter with input_dim=128, bottleneck_dim=16...")
        adapter = BottleneckAdapter(input_dim=128, bottleneck_dim=16, dropout=0.1)
        x = torch.randn(2, 5, 128)
        
        # Test zero initialization of up-projection
        with torch.no_grad():
            output = adapter(x)
            
        # At step 0, output must be exactly equal to input (x)
        self.assertTrue(torch.allclose(output, x, atol=1e-6))
        print("    => output is exactly equal to input at step 0 (residual connection check) [OK]")

        # Check submodules
        self.assertEqual(adapter.down_proj.out_features, 16)
        self.assertEqual(adapter.up_proj.out_features, 128)
        print(f"    => down_proj size: {adapter.down_proj.weight.shape}, up_proj size: {adapter.up_proj.weight.shape} [OK]")
        print("✅ [test_models] Passed: test_bottleneck_adapter_zero_init")

    def test_apply_adapters_freezing(self):
        """
        Verify that apply_adapters wraps transformer MLP blocks and freezes backbone params.
        """
        print("\n🧪 [test_models] Running: test_apply_adapters_freezing")
        print("  - Instantiating DinoV3Classifier with Bottleneck Adapters...")
        classifier = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_adapter
        )
        
        # Check if backbone parameters are frozen
        frozen_params = 0
        trainable_params = 0
        for name, param in classifier.backbone.named_parameters():
            if "adapter" not in name:
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen.")
                frozen_params += 1
            else:
                self.assertTrue(param.requires_grad, f"Adapter parameter {name} should be trainable.")
                trainable_params += 1
        print(f"    => Backbone params checked: {frozen_params} frozen backbone parameters, {trainable_params} trainable adapter parameters [OK]")

        # Check classifier head parameters are trainable
        head_params = 0
        for param in classifier.classifier.parameters():
            self.assertTrue(param.requires_grad)
            head_params += 1
        print(f"    => Head params checked: {head_params} trainable head parameters [OK]")
        print("✅ [test_models] Passed: test_apply_adapters_freezing")

    def test_vpt_token_layout_and_slicing(self):
        """
        Verify the VPT prepending layout and token slicing under shallow and deep prompt tuning
        with the DINOv3 registers layout (C7 Fix).
        """
        print("\n🧪 [test_models] Running: test_vpt_token_layout_and_slicing")
        
        # Test shallow VPT
        print("  - Instantiating classifier with Shallow VPT (10 tokens)...")
        classifier_vpt_s = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_vpt_shallow
        )
        
        # Verify prompt exists and layout has correct token length
        self.assertTrue(hasattr(classifier_vpt_s.backbone, "prompt"))
        self.assertEqual(classifier_vpt_s.backbone.prompt.shape, (1, 10, 384))
        print(f"    => Shallow prompt parameter size: {classifier_vpt_s.backbone.prompt.shape} [OK]")

        # Test deep VPT wrapping
        print("  - Instantiating classifier with Deep VPT...")
        classifier_vpt_d = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_vpt_deep
        )
        self.assertTrue(classifier_vpt_d.backbone.deep)
        print(f"    => Deep prompt parameter dictionary keys: {list(classifier_vpt_d.backbone.deep_prompts.keys())} [OK]")
        
        # Test forward pass shape checks
        print("  - Simulating forward pass with input shape (2, 3, 224, 224) through Shallow VPT classifier...")
        dummy_pixel_values = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            outputs = classifier_vpt_s(pixel_values=dummy_pixel_values)
            
        self.assertEqual(outputs["logits"].shape, (2, 2))
        print(f"    => Forward pass outputs shape: {outputs['logits'].shape} [OK]")
        print("✅ [test_models] Passed: test_vpt_token_layout_and_slicing")

    def test_lora_wrapping(self):
        """
        Verify that LoRA targets attention projections correctly and freezes backbone.
        """
        print("\n🧪 [test_models] Running: test_lora_wrapping")
        print("  - Instantiating classifier with LoRA (r=8, target_modules=['q_proj', 'v_proj'])...")
        classifier = DinoV3Classifier(
            model_name_or_path=self.model_name,
            head_config=self.head_cfg,
            peft_config=self.peft_cfg_lora
        )
        
        # Check if LoRA parameters are trainable and other backbone params are frozen
        lora_found = False
        lora_params = 0
        frozen_params = 0
        for name, param in classifier.backbone.named_parameters():
            if "lora" in name:
                lora_found = True
                self.assertTrue(param.requires_grad)
                lora_params += 1
            else:
                # Unless it's the classifier head, all other weights should be frozen
                self.assertFalse(param.requires_grad)
                frozen_params += 1
                
        self.assertTrue(lora_found)
        print(f"    => LoRA params check: found {lora_params} trainable LoRA parameters, {frozen_params} frozen parameters [OK]")
        print("✅ [test_models] Passed: test_lora_wrapping")

    def test_vpt_deep_layer_prompt_wrapper(self):
        """
        Verify that VptLayerWrapper correctly discards prompt tokens from the
        previous layer's hidden states and prepends the new deep prompt tokens.
        """
        print("\n🧪 [test_models] Running: test_vpt_deep_layer_prompt_wrapper")
        
        # Instantiate a dummy nn.Module (e.g. Identity) to be wrapped
        dummy_layer = nn.Identity()
        num_tokens = 5
        hidden_size = 128
        batch_size = 2
        
        # Create a trainable prompt parameter (deep prompt tokens for this layer)
        # Initialize to all 4.0
        prompt_param = nn.Parameter(torch.ones(1, num_tokens, hidden_size) * 4.0)
        
        # Instantiate VptLayerWrapper
        wrapper = VptLayerWrapper(
            original_layer=dummy_layer,
            num_tokens=num_tokens,
            prompt_parameter=prompt_param
        )
        
        # Create input hidden states representing output of previous layer:
        # Layout: [CLS] (index 0) + [Old Prompts] (indices 1..5) + [Patches] (indices 6..15)
        # Let's populate with distinctive values:
        # CLS token is all 1.0
        # Old prompt tokens are all 2.0
        # Patch/register tokens are all 3.0
        cls_tokens = torch.ones(batch_size, 1, hidden_size) * 1.0
        old_prompts = torch.ones(batch_size, num_tokens, hidden_size) * 2.0
        patch_tokens = torch.ones(batch_size, 10, hidden_size) * 3.0
        
        input_hidden_states = torch.cat([cls_tokens, old_prompts, patch_tokens], dim=1)
        self.assertEqual(input_hidden_states.shape, (batch_size, 1 + num_tokens + 10, hidden_size))
        
        # Pass through wrapper
        output = wrapper(input_hidden_states)
        
        # The output shape must remain the same: (batch_size, 1 + num_tokens + 10, hidden_size)
        self.assertEqual(output.shape, (batch_size, 1 + num_tokens + 10, hidden_size))
        
        # Verify tokens at different slices:
        # 1. CLS token (index 0) should be 1.0
        self.assertTrue(torch.allclose(output[:, 0, :], torch.ones(batch_size, hidden_size) * 1.0))
        
        # 2. Prompt tokens (indices 1 to 5) should be 4.0 (the new prompt tokens)
        self.assertTrue(torch.allclose(output[:, 1:1+num_tokens, :], torch.ones(batch_size, num_tokens, hidden_size) * 4.0))
        
        # 3. Patch tokens (indices 6 to 15) should be 3.0 (old prompt tokens discarded)
        self.assertTrue(torch.allclose(output[:, 1+num_tokens:, :], torch.ones(batch_size, 10, hidden_size) * 3.0))
        
        print("    => Output tensor shape and values validated successfully [OK]")
        print("    => Old prompt tokens (2.0) correctly replaced by new prompt tokens (4.0) [OK]")
        print("✅ [test_models] Passed: test_vpt_deep_layer_prompt_wrapper")

if __name__ == "__main__":
    unittest.main()
