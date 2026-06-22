#!/usr/bin/env python3
import os
import yaml
import argparse
from pathlib import Path
from typing import Any, Dict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO detection ablation study configs.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all_label",
        choices=["all_label", "no_cell", "abnormal_only"],
        help="Dataset split strategy to generate configs for."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Map strategy to datasets and outputs
    strategy = args.strategy
    data_name = "all" if strategy == "all_label" else strategy
    yolo_data_yaml = f"data/kfold_detection/battery_detection_{data_name}.yaml"
    
    if strategy == "all_label":
        output_dir = "outputs/det_all"
    elif strategy == "no_cell":
        output_dir = "outputs/det_no_cell"
    elif strategy == "abnormal_only":
        output_dir = "outputs/det_abnormal"
    else:
        output_dir = f"outputs/{strategy}"
    
    out_dir = Path(f"configs/det/ablations_{strategy}")
    
    # Load base configuration as template (use peft_smoke_all_label as template)
    base_config_path = Path("configs/det/peft_smoke_all_label.yaml")
    if not base_config_path.exists():
        print(f"❌ Error: {base_config_path} does not exist.")
        return

    with open(base_config_path, "r") as f:
        base_cfg: Dict[str, Any] = yaml.safe_load(f)

    # Apply updated global configs (aligned with classification ablations)
    base_cfg["batch_size"] = 64
    base_cfg["num_epochs"] = 300
    base_cfg["early_stopping_patience"] = 20
    base_cfg["amp"]["fp16"] = True
    base_cfg["amp"]["bf16"] = False
    
    # YOLO specific defaults
    base_cfg["yolo_model_config"] = "configs/det/yolo26_dino.yaml"
    base_cfg["yolo_data_yaml"] = yolo_data_yaml
    base_cfg["output_dir"] = output_dir
    base_cfg["data"]["data_dir"] = "data/kfold_detection"

    # Clean previous generated yaml files to avoid mixing old/new runs
    if out_dir.exists():
        for f in out_dir.glob("*.yaml"):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generated_configs = []
    run_id = 1

    # Swept models
    models = [
        ("vits16", "facebook/dinov3-vits16-pretrain-lvd1689m")
    ]

    # 1. Baseline Runs
    for model_key, model_name in models:
        for fold in range(5):
            cfg = base_cfg.copy()
            cfg["model_name"] = model_name
            cfg["learning_rate"] = 0.0003
            cfg["fold"] = fold
            cfg["seed"] = 30 + fold * 10
            cfg["peft"] = {
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
            
            path = out_dir / f"{run_id:02d}_baseline_{model_key}_fold_{fold}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            generated_configs.append(path)
            run_id += 1

    # 1b. Standard YOLO Baseline Runs (compare yolo26n and yolo26s)
    standard_yolos = [
        ("yolo26n", "yolo26n.pt", 640),
        ("yolo26s", "yolo26s.pt", 640)
    ]
    for model_key, model_config, img_size in standard_yolos:
        for fold in range(5):
            cfg = base_cfg.copy()
            cfg["model_name"] = model_key
            cfg["yolo_model_config"] = model_config
            cfg["learning_rate"] = 0.0003
            cfg["fold"] = fold
            cfg["seed"] = 30 + fold * 10
            cfg["data"] = base_cfg["data"].copy()
            cfg["data"]["image_size"] = img_size
            cfg["peft"] = {
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
            
            path = out_dir / f"{run_id:02d}_baseline_standard_{model_key}_fold_{fold}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            generated_configs.append(path)
            run_id += 1

    # 2. LoRA Runs
    lora_ranks = [8, 16]
    lora_blocks = [
        ("last4", [8, 9, 10, 11]),
        ("last2", [10, 11])
    ]
    lora_lrs = [0.0003]

    for model_key, model_name in models:
        for rank in lora_ranks:
            for block_name, blocks in lora_blocks:
                for lr in lora_lrs:
                    for fold in range(5):
                        cfg = base_cfg.copy()
                        cfg["model_name"] = model_name
                        cfg["learning_rate"] = lr
                        cfg["fold"] = fold
                        cfg["seed"] = 30 + fold * 10
                        cfg["peft"] = {
                            "type": "lora",
                            "lora_r": rank,
                            "lora_alpha": rank * 2,
                            "lora_dropout": 0.1,
                            "lora_target_modules": ["q_proj", "v_proj"],
                            "lora_target_blocks": blocks,
                            "adapter_bottleneck_dim": 64,
                            "adapter_dropout": 0.0,
                            "adapter_target_blocks": None,
                            "vpt_num_tokens": 10,
                            "vpt_deep": False,
                            "vpt_target_blocks": None
                        }
                        
                        path = out_dir / f"{run_id:02d}_lora_{model_key}_r{rank}_{block_name}_lr{lr}_fold_{fold}.yaml"
                        with open(path, "w") as f:
                            yaml.safe_dump(cfg, f)
                        generated_configs.append(path)
                        run_id += 1

    # 3. Adapter Runs
    adapter_dims = [32, 64]
    adapter_blocks = [
        ("last4", [8, 9, 10, 11]),
        ("last2", [10, 11])
    ]
    adapter_lrs = [0.0003]

    for model_key, model_name in models:
        for dim in adapter_dims:
            for block_name, blocks in adapter_blocks:
                for lr in adapter_lrs:
                    for fold in range(5):
                        cfg = base_cfg.copy()
                        cfg["model_name"] = model_name
                        cfg["learning_rate"] = lr
                        cfg["fold"] = fold
                        cfg["seed"] = 30 + fold * 10
                        cfg["peft"] = {
                            "type": "adapter",
                            "lora_r": 8,
                            "lora_alpha": 16,
                            "lora_dropout": 0.0,
                            "lora_target_modules": ["q_proj", "v_proj"],
                            "lora_target_blocks": None,
                            "adapter_bottleneck_dim": dim,
                            "adapter_dropout": 0.1,
                            "adapter_target_blocks": blocks,
                            "vpt_num_tokens": 10,
                            "vpt_deep": False,
                            "vpt_target_blocks": None
                        }
                        
                        path = out_dir / f"{run_id:02d}_adapter_{model_key}_d{dim}_{block_name}_lr{lr}_fold_{fold}.yaml"
                        with open(path, "w") as f:
                            yaml.safe_dump(cfg, f)
                        generated_configs.append(path)
                        run_id += 1

    # 4. VPT Runs
    vpt_types = [
        ("shallow", False, None),
        ("deep_last2", True, [10, 11]),
        ("deep_last4", True, [8, 9, 10, 11])
    ]
    vpt_tokens = [10]
    vpt_lrs = [0.0003]

    for model_key, model_name in models:
        for type_name, is_deep, blocks in vpt_types:
            for tokens in vpt_tokens:
                for lr in vpt_lrs:
                    for fold in range(5):
                        cfg = base_cfg.copy()
                        cfg["model_name"] = model_name
                        cfg["learning_rate"] = lr
                        cfg["fold"] = fold
                        cfg["seed"] = 30 + fold * 10
                        cfg["peft"] = {
                            "type": "visual_prompt",
                            "lora_r": 8,
                            "lora_alpha": 16,
                            "lora_dropout": 0.0,
                            "lora_target_modules": ["q_proj", "v_proj"],
                            "lora_target_blocks": None,
                            "adapter_bottleneck_dim": 64,
                            "adapter_dropout": 0.0,
                            "adapter_target_blocks": None,
                            "vpt_num_tokens": tokens,
                            "vpt_deep": is_deep,
                            "vpt_target_blocks": blocks
                        }
                        
                        path = out_dir / f"{run_id:02d}_vpt_{model_key}_{type_name}_t{tokens}_lr{lr}_fold_{fold}.yaml"
                        with open(path, "w") as f:
                            yaml.safe_dump(cfg, f)
                        generated_configs.append(path)
                        run_id += 1

    print(f"✅ Generated {len(generated_configs)} detection configuration files under {out_dir}/")

if __name__ == "__main__":
    main()
