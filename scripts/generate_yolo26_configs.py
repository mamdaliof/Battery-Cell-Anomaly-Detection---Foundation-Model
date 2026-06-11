#!/usr/bin/env python3
import os
import yaml
from pathlib import Path

def main():
    workspace = Path(__file__).resolve().parents[1]
    
    # 1. Load the original yolo26_dino.yaml content
    original_yaml_path = workspace / "configs" / "det" / "yolo26_dino.yaml"
    with open(original_yaml_path, "r") as f:
        original_yaml_content = f.read()

    yolo_variants_dir = workspace / "configs" / "det" / "yolo_variants"
    ablations_dir = workspace / "configs" / "det" / "ablations_all_label"
    
    yolo_variants_dir.mkdir(parents=True, exist_ok=True)
    ablations_dir.mkdir(parents=True, exist_ok=True)

    scales = ["n", "s", "m", "l", "x"]

    # Load template config from one of standard yolo training configs
    template_config_path = ablations_dir / "yolo11n_train.yaml"
    with open(template_config_path, "r") as f:
        train_template = yaml.safe_load(f)

    # Adjust training settings for yolo26 dino model
    train_template["model_name"] = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    train_template["data"]["image_size"] = 256  # DINO standard resolution is 256

    for scale in scales:
        # A. Create the architecture yolo26<scale>.yaml file
        arch_filename = f"yolo26{scale}.yaml"
        arch_path = yolo_variants_dir / arch_filename
        arch_content = f"# Scale Specifier\nscale: {scale}\n\n" + original_yaml_content
        with open(arch_path, "w") as f:
            f.write(arch_content)
        print(f"Generated architecture config: {arch_path.relative_to(workspace)}")

        # B. Create the training yolo26<scale>_train.yaml file
        train_cfg = train_template.copy()
        train_cfg["yolo_model_config"] = f"configs/det/yolo_variants/{arch_filename}"
        
        # Set batch size to 32 for 'x' scale, 64 for others
        train_cfg["batch_size"] = 32 if scale == "x" else 64

        train_filename = f"yolo26{scale}_train.yaml"
        
        # Write to configs/det/yolo_variants/
        with open(yolo_variants_dir / train_filename, "w") as f:
            yaml.safe_dump(train_cfg, f)
            
        # Write to configs/det/ablations_all_label/
        with open(ablations_dir / train_filename, "w") as f:
            yaml.safe_dump(train_cfg, f)

        print(f"Generated training config: configs/det/ablations_all_label/{train_filename}")

if __name__ == "__main__":
    main()
