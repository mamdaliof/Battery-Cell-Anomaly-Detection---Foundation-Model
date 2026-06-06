import sys
import torch
import os

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bcadfm.models.dinov3_classifier import DinoV3Classifier, HeadConfig
from bcadfm.utils.model_utils import count_parameters, log_parameter_summary
from bcadfm.utils.config import PeftConfigSchema

def run_verification():
    # Use a small standard HF ViT backbone for verification
    model_name = "google/vit-base-patch16-224-in21k"
    head_config = HeadConfig(num_labels=2, depth=2, hidden_dim=256, dropout=0.1)

    print("=================================================================")
    print("🧪 RUNNING PEFT INTEGRATION VERIFICATION")
    print("=================================================================\n")

    # 1. Test Baseline (No PEFT)
    print("🔹 Testing Baseline (No PEFT)...")
    peft_none = PeftConfigSchema(type="none")
    model_none = DinoV3Classifier(
        model_name_or_path=model_name,
        head_config=head_config,
        peft_config=peft_none,
        freeze_backbone=True
    )
    summary_none = count_parameters(model_none)
    log_parameter_summary(model_none, "Baseline Classifier")
    
    assert summary_none["trainable"] > 0, "Trainable parameters should be > 0"
    print("✅ Baseline Test Passed!\n")

    # 2. Test LoRA PEFT
    print("🔹 Testing LoRA PEFT...")
    peft_lora = PeftConfigSchema(
        type="lora",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["q_proj", "v_proj"]
    )
    model_lora = DinoV3Classifier(
        model_name_or_path=model_name,
        head_config=head_config,
        peft_config=peft_lora,
        freeze_backbone=True
    )
    summary_lora = count_parameters(model_lora)
    log_parameter_summary(model_lora, "LoRA Classifier")
    
    assert summary_lora["trainable"] > summary_none["trainable"], "LoRA should have more trainable parameters than baseline"
    backbone_trainable = sum(p.numel() for p in model_lora.backbone.parameters() if p.requires_grad)
    assert backbone_trainable > 0, "LoRA adapters should be trainable"
    print("✅ LoRA Test Passed!\n")

    # 3. Test Adapters PEFT
    print("🔹 Testing Bottleneck Adapters...")
    peft_adapter = PeftConfigSchema(
        type="adapter",
        adapter_bottleneck_dim=32,
        adapter_dropout=0.1
    )
    model_adapter = DinoV3Classifier(
        model_name_or_path=model_name,
        head_config=head_config,
        peft_config=peft_adapter,
        freeze_backbone=True
    )
    summary_adapter = count_parameters(model_adapter)
    log_parameter_summary(model_adapter, "Adapter Classifier")
    
    assert summary_adapter["trainable"] > summary_none["trainable"], "Adapter should have more trainable parameters than baseline"
    print("✅ Bottleneck Adapter Test Passed!\n")

    # 4. Test VPT (Visual Prompt Tuning) - Shallow
    print("🔹 Testing Shallow Visual Prompt Tuning...")
    peft_vpt_shallow = PeftConfigSchema(
        type="visual_prompt",
        vpt_num_tokens=10,
        vpt_deep=False
    )
    model_vpt_shallow = DinoV3Classifier(
        model_name_or_path=model_name,
        head_config=head_config,
        peft_config=peft_vpt_shallow,
        freeze_backbone=True
    )
    summary_vpt_shallow = count_parameters(model_vpt_shallow)
    log_parameter_summary(model_vpt_shallow, "Shallow VPT Classifier")
    
    vpt_trainable = summary_vpt_shallow["trainable"] - summary_none["trainable"]
    print(f"Calculated prompt trainable parameters: {vpt_trainable:,}")
    assert vpt_trainable == 7680, f"Expected 7,680 prompt parameters, got {vpt_trainable:,}"
    print("✅ Shallow VPT Test Passed!\n")

    # 5. Test VPT (Visual Prompt Tuning) - Deep
    print("🔹 Testing Deep Visual Prompt Tuning...")
    peft_vpt_deep = PeftConfigSchema(
        type="visual_prompt",
        vpt_num_tokens=10,
        vpt_deep=True
    )
    model_vpt_deep = DinoV3Classifier(
        model_name_or_path=model_name,
        head_config=head_config,
        peft_config=peft_vpt_deep,
        freeze_backbone=True
    )
    summary_vpt_deep = count_parameters(model_vpt_deep)
    log_parameter_summary(model_vpt_deep, "Deep VPT Classifier")
    
    assert summary_vpt_deep["trainable"] > summary_vpt_shallow["trainable"], "Deep VPT should have more trainable parameters than Shallow VPT"
    print("✅ Deep VPT Test Passed!\n")

    print("=================================================================")
    print("🎉 ALL PEFT INTEGRATION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_verification()
