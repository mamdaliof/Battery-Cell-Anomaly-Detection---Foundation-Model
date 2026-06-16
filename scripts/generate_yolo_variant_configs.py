#!/usr/bin/env python3
import os
import yaml
from pathlib import Path

def save_config_with_validation(cfg: dict, path: Path) -> None:
    """Validate and save training configuration to file, preventing concurrent imbalance strategies."""
    imbalance = cfg.get("imbalance", {})
    oversampling_method = imbalance.get("oversampling_method", "none")
    class_weights = imbalance.get("class_weights", "none")
    focal_alpha = imbalance.get("focal_alpha", None)
    if oversampling_method != "none" and (class_weights != "none" or focal_alpha is not None):
        raise ValueError(
            f"Config generator constraint violation: "
            f"cannot generate config with both oversampling_method='{oversampling_method}' and "
            f"class_weights='{class_weights}'/focal_alpha active."
        )
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

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
        # Also write without _dino suffix
        with open(variants_dir / f"yolo26{scale}.yaml", "w") as f:
            f.write(scaled_content)
        print(f"📝 Created YOLO26 model architecture: {arch_filepath} and yolo26{scale}.yaml")

    # 3. Base Training Template (adapted from configs/det/peft_smoke_all_label.yaml)
    template_path = configs_dir / "peft_smoke_all_label.yaml"
    with open(template_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Standardize hyperparameters across all variants
    base_cfg["num_epochs"] = 300
    base_cfg["early_stopping_patience"] = 20
    base_cfg["amp"]["fp16"] = True
    base_cfg["amp"]["bf16"] = False

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
    
    # Safe batch size per scale (due to GPU OOM limits on 15GB VRAM)
    scale_batch_sizes = {
        "n": 64,
        "s": 64,
        "m": 32,
        "l": 16,
        "x": 8
    }

    # Loop through the three dataset split strategies
    strategies = ["all_label", "no_cell", "abnormal_only"]
    for strategy in strategies:
        data_name = "all" if strategy == "all_label" else strategy
        yolo_data_yaml = f"data/det_v1.0/battery_detection_{data_name}.yaml"
        
        if strategy == "all_label":
            output_dir = "outputs/det_all"
        elif strategy == "no_cell":
            output_dir = "outputs/det_no_cell"
        elif strategy == "abnormal_only":
            output_dir = "outputs/det_abnormal"
        else:
            output_dir = f"outputs/{strategy}"
        
        ablations_dir = configs_dir / f"ablations_{strategy}"
        ablations_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 --- Generating training configs for strategy: {strategy} ---")
        
        # A. YOLOv8 Variants
        for scale in scales:
            model_key = f"yolov8{scale}"
            cfg = base_cfg.copy()
            cfg["model_name"] = model_key
            cfg["yolo_model_config"] = f"{model_key}.pt"
            cfg["yolo_data_yaml"] = yolo_data_yaml
            cfg["output_dir"] = output_dir
            cfg["batch_size"] = scale_batch_sizes[scale]
            cfg["data"] = base_cfg["data"].copy()
            cfg["data"]["image_size"] = 640  # Standard YOLOv8 size
            
            cfg_filepath = ablations_dir / f"yolov8{scale}_train.yaml"
            save_config_with_validation(cfg, cfg_filepath)
            print(f"  ✅ YOLOv8{scale} (batch={cfg['batch_size']}) -> {cfg_filepath}")

        # B. YOLO11 Variants
        for scale in scales:
            model_key = f"yolo11{scale}"
            cfg = base_cfg.copy()
            cfg["model_name"] = model_key
            cfg["yolo_model_config"] = f"{model_key}.pt"
            cfg["yolo_data_yaml"] = yolo_data_yaml
            cfg["output_dir"] = output_dir
            cfg["batch_size"] = scale_batch_sizes[scale]
            cfg["data"] = base_cfg["data"].copy()
            cfg["data"]["image_size"] = 640  # Standard YOLO11 size
            
            cfg_filepath = ablations_dir / f"yolo11{scale}_train.yaml"
            save_config_with_validation(cfg, cfg_filepath)
            print(f"  ✅ YOLO11{scale} (batch={cfg['batch_size']}) -> {cfg_filepath}")

        # C. YOLO26 Variants (DINOv3 backbone + SFP + YOLO26 Head)
        for scale in scales:
            # We generate both `yolo26{scale}_dino_train.yaml` and `yolo26{scale}_train.yaml`
            # to remain compatible with any legacy paths or status checker conventions.
            for suffix in ["_dino_train.yaml", "_train.yaml"]:
                cfg = base_cfg.copy()
                cfg["model_name"] = "facebook/dinov3-vitb16-pretrain-lvd1689m"
                
                # Architecture path reference
                arch_filename = f"yolo26{scale}_dino.yaml" if "dino" in suffix else f"yolo26{scale}.yaml"
                cfg["yolo_model_config"] = f"configs/det/yolo_variants/{arch_filename}"
                cfg["yolo_data_yaml"] = yolo_data_yaml
                cfg["output_dir"] = output_dir
                
                # YOLO26 image size is 256, allowing slightly higher batch size on large/x than standard YOLO 640 size
                # but we will still reduce it for large/x to be completely safe from OOM
                if scale == "x":
                    cfg["batch_size"] = 16
                elif scale == "l":
                    cfg["batch_size"] = 32
                else:
                    cfg["batch_size"] = 64
                
                cfg["data"] = base_cfg["data"].copy()
                cfg["data"]["image_size"] = 256  # DINO standard size
                
                cfg_filename = f"yolo26{scale}{suffix}"
                cfg_filepath = ablations_dir / cfg_filename
                
                save_config_with_validation(cfg, cfg_filepath)
            print(f"  ✅ YOLO26{scale} (batch={cfg['batch_size']}) -> {ablations_dir}/yolo26{scale}_train.yaml")

        # Also generate a special config for yolo26n.pt standard baseline checkpoint if they want to load it
        cfg = base_cfg.copy()
        cfg["model_name"] = "yolo26n"
        cfg["yolo_model_config"] = "yolo26n.pt"
        cfg["yolo_data_yaml"] = yolo_data_yaml
        cfg["output_dir"] = output_dir
        cfg["batch_size"] = 64
        cfg["data"] = base_cfg["data"].copy()
        cfg["data"]["image_size"] = 640
        cfg_filepath = ablations_dir / "yolo26n_pretrained_train.yaml"
        save_config_with_validation(cfg, cfg_filepath)
        print(f"  ✅ Pretrained YOLO26n (batch=64) -> {cfg_filepath}")

if __name__ == "__main__":
    main()
