#!/usr/bin/env python3
import os
import yaml
from pathlib import Path

def main():
    # 1. Paths
    configs_dir = Path("configs/det")
    variants_dir = configs_dir / "yolo_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate YOLO26 model architecture configs (with scale specifiers)
    yolo26_base_path = configs_dir / "yolo26_dino.yaml"
    if not yolo26_base_path.exists():
        print(f"❌ Error: {yolo26_base_path} not found.")
        return

    # Read the base YOLO26 architecture template
    with open(yolo26_base_path, "r") as f:
        yolo26_arch_content = f.read()

    scales = ["n", "s", "m", "l", "x"]
    
    # We will write the custom architecture files under configs/det/yolo_variants/
    for scale in scales:
        arch_filename = f"yolo26{scale}_dino.yaml"
        arch_filepath = variants_dir / arch_filename
        
        # Prepend the scale specifier at the top of the YAML
        scaled_content = f"# Scale Specifier\nscale: {scale}\n\n" + yolo26_arch_content
        with open(arch_filepath, "w") as f:
            f.write(scaled_content)
        print(f"📝 Created YOLO26 model architecture: {arch_filepath}")

    # 3. Base Training Template (adapted from configs/det/ablations/03_baseline_standard_yolov8n.yaml)
    # Let's read configs/det/ablations/03_baseline_standard_yolov8n.yaml as the template
    template_path = configs_dir / "ablations" / "03_baseline_standard_yolov8n.yaml"
    if not template_path.exists():
        # Fallback to general benchmark baseline
        template_path = configs_dir / "benchmark_baseline.yaml"
        
    with open(template_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Standardize hyperparameters across all variants
    base_cfg["batch_size"] = 64
    base_cfg["num_epochs"] = 300
    base_cfg["early_stopping_patience"] = 20
    base_cfg["amp"]["fp16"] = True
    base_cfg["amp"]["bf16"] = False
    base_cfg["yolo_data_yaml"] = "data/battery_detection_all.yaml"

    # Reset PEFT to none for baseline training of standard models
    base_cfg["peft"] = {
        "type": "none",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_target_blocks": None,
        "adapter_bottleneck_dim": 64,
        "adapter_dropout": 0.0,
        "adapter_target_blocks": None,
        "vpt_num_tokens": 10,
        "vpt_deep": False,
        "vpt_target_blocks": None
    }

    # Sizes mappings for YOLO models
    size_names = {
        "n": "nano",
        "s": "small",
        "m": "medium",
        "l": "large",
        "x": "largest"
    }

    # Generate training configs
    # A. YOLOv8 Variants
    print("\n--- Generating YOLOv8 Training Configs ---")
    for scale in scales:
        model_key = f"yolov8{scale}"
        cfg = base_cfg.copy()
        cfg["model_name"] = model_key
        cfg["yolo_model_config"] = f"{model_key}.pt"
        cfg["data"] = base_cfg["data"].copy()
        cfg["data"]["image_size"] = 640  # Standard YOLOv8 size
        
        cfg_filepath = variants_dir / f"yolov8{scale}_train.yaml"
        with open(cfg_filepath, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        print(f"✅ Created training config: {cfg_filepath} (Size: {size_names[scale]})")

    # B. YOLO11 Variants
    print("\n--- Generating YOLO11 Training Configs ---")
    for scale in scales:
        model_key = f"yolo11{scale}"
        cfg = base_cfg.copy()
        cfg["model_name"] = model_key
        cfg["yolo_model_config"] = f"{model_key}.pt"
        cfg["data"] = base_cfg["data"].copy()
        cfg["data"]["image_size"] = 640  # Standard YOLO11 size
        
        cfg_filepath = variants_dir / f"yolo11{scale}_train.yaml"
        with open(cfg_filepath, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        print(f"✅ Created training config: {cfg_filepath} (Size: {size_names[scale]})")

    # C. YOLO26 Variants (DINOv3 backbone + SFP + YOLO26 Head)
    print("\n--- Generating YOLO26 (DINOv3 + SFP) Training Configs ---")
    for scale in scales:
        cfg = base_cfg.copy()
        
        # For YOLO26, the model_name is the backbone model name
        cfg["model_name"] = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        cfg["yolo_model_config"] = f"configs/det/yolo_variants/yolo26{scale}_dino.yaml"
        cfg["data"] = base_cfg["data"].copy()
        cfg["data"]["image_size"] = 256  # standard size for DINOv3 backbone in this project
        
        cfg_filepath = variants_dir / f"yolo26{scale}_dino_train.yaml"
        with open(cfg_filepath, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        print(f"✅ Created training config: {cfg_filepath} (Size: {size_names[scale]})")

    # Also generate a special config for yolo26n.pt standard baseline checkpoint if they want to load it
    cfg = base_cfg.copy()
    cfg["model_name"] = "yolo26n"
    cfg["yolo_model_config"] = "yolo26n.pt"
    cfg["data"] = base_cfg["data"].copy()
    cfg["data"]["image_size"] = 640
    cfg_filepath = variants_dir / "yolo26n_pretrained_train.yaml"
    with open(cfg_filepath, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    print(f"✅ Created training config for pretrained checkpoint: {cfg_filepath}")

if __name__ == "__main__":
    main()
