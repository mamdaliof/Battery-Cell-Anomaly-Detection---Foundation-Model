#!/usr/bin/env python3
import os
import yaml
from pathlib import Path
from typing import Any, Dict

def main():
    # Load base configuration as template
    base_config_path = Path("configs/benchmark_baseline.yaml")
    if not base_config_path.exists():
        print(f"❌ Error: {base_config_path} does not exist.")
        return

    with open(base_config_path, "r") as f:
        base_cfg: Dict[str, Any] = yaml.safe_load(f)

    # Apply updated global configs
    base_cfg["data"]["data_dir"] = "data/cls_v1.0"
    base_cfg["data"]["abnormal_class_name"] = "abnormal"
    base_cfg["batch_size"] = 64
    base_cfg["num_epochs"] = 300
    base_cfg["early_stopping_patience"] = 20
    base_cfg["amp"]["fp16"] = True
    base_cfg["amp"]["bf16"] = False

    # Output directory
    out_dir = Path("configs/ablations")
    # Clean previous generated yaml files to avoid mixing old/new runs
    if out_dir.exists():
        for f in out_dir.glob("*.yaml"):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generated_configs = []
    run_id = 1

    # Swept models
    models = [
        ("vits16", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        ("vitb16", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    ]

    # 1. Baseline Runs (2 runs: Small, Base)
    for model_key, model_name in models:
        cfg = base_cfg.copy()
        cfg["model_name"] = model_name
        cfg["peft"] = {
            "type": "none",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules": ["query", "value"],
            "lora_target_blocks": None,
            "adapter_bottleneck_dim": 64,
            "adapter_dropout": 0.0,
            "adapter_target_blocks": None,
            "vpt_num_tokens": 10,
            "vpt_deep": False,
            "vpt_target_blocks": None
        }
        
        path = out_dir / f"{run_id:02d}_baseline_{model_key}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        generated_configs.append(path)
        run_id += 1

    # 2. LoRA Runs (24 runs: 2 models x 2 ranks x 3 blocks x 2 LRs)
    lora_ranks = [8, 16]
    lora_blocks = [
        ("all", None),
        ("last4", [8, 9, 10, 11]),
        ("last2", [10, 11])
    ]
    lora_lrs = [0.0003, 0.0005]

    for model_key, model_name in models:
        for rank in lora_ranks:
            for block_name, blocks in lora_blocks:
                for lr in lora_lrs:
                    cfg = base_cfg.copy()
                    cfg["model_name"] = model_name
                    cfg["learning_rate"] = lr
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
                    
                    path = out_dir / f"{run_id:02d}_lora_{model_key}_r{rank}_{block_name}_lr{lr}.yaml"
                    with open(path, "w") as f:
                        yaml.safe_dump(cfg, f)
                    generated_configs.append(path)
                    run_id += 1

    # 3. Adapter Runs (16 runs: 2 models x 2 dims x 2 blocks x 2 LRs)
    adapter_dims = [32, 64]
    adapter_blocks = [
        ("last4", [8, 9, 10, 11]),
        ("last2", [10, 11])
    ]
    adapter_lrs = [0.0003, 0.0005]

    for model_key, model_name in models:
        for dim in adapter_dims:
            for block_name, blocks in adapter_blocks:
                for lr in adapter_lrs:
                    cfg = base_cfg.copy()
                    cfg["model_name"] = model_name
                    cfg["learning_rate"] = lr
                    cfg["peft"] = {
                        "type": "adapter",
                        "lora_r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.0,
                        "lora_target_modules": ["query", "value"],
                        "lora_target_blocks": None,
                        "adapter_bottleneck_dim": dim,
                        "adapter_dropout": 0.1,
                        "adapter_target_blocks": blocks,
                        "vpt_num_tokens": 10,
                        "vpt_deep": False,
                        "vpt_target_blocks": None
                    }
                    
                    path = out_dir / f"{run_id:02d}_adapter_{model_key}_d{dim}_{block_name}_lr{lr}.yaml"
                    with open(path, "w") as f:
                        yaml.safe_dump(cfg, f)
                    generated_configs.append(path)
                    run_id += 1

    # 4. VPT Runs (16 runs: 2 models x 2 types x 2 tokens x 2 LRs)
    vpt_types = [
        ("shallow", False, None),
        ("deep_last4", True, [8, 9, 10, 11])
    ]
    vpt_tokens = [10, 20]
    vpt_lrs = [0.0005, 0.001]

    for model_key, model_name in models:
        for type_name, is_deep, blocks in vpt_types:
            for tokens in vpt_tokens:
                for lr in vpt_lrs:
                    cfg = base_cfg.copy()
                    cfg["model_name"] = model_name
                    cfg["learning_rate"] = lr
                    cfg["peft"] = {
                        "type": "visual_prompt",
                        "lora_r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.0,
                        "lora_target_modules": ["query", "value"],
                        "lora_target_blocks": None,
                        "adapter_bottleneck_dim": 64,
                        "adapter_dropout": 0.0,
                        "adapter_target_blocks": None,
                        "vpt_num_tokens": tokens,
                        "vpt_deep": is_deep,
                        "vpt_target_blocks": blocks
                    }
                    
                    path = out_dir / f"{run_id:02d}_vpt_{model_key}_{type_name}_t{tokens}_lr{lr}.yaml"
                    with open(path, "w") as f:
                        yaml.safe_dump(cfg, f)
                    generated_configs.append(path)
                    run_id += 1

    # Write runner shell script
    runner_path = Path("run_ablations.sh")
    with open(runner_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Exit immediately on failure\nset -e\n\n")
        f.write("export PYTHONPATH=$(pwd)/src:$PYTHONPATH\n\n")
        f.write(f"echo \"🚀 Starting Ablation Study execution sequence ({len(generated_configs)} runs)...\"\n\n")
        
        for config_file in generated_configs:
            f.write(f"echo \"----------------------------------------------------------------\"\n")
            f.write(f"echo \"🏃 Running config: {config_file}\"\n")
            f.write(f"python scripts/train.py --config {config_file}\n\n")
            
        f.write(f"echo \"🎉 All {len(generated_configs)} ablation runs completed successfully!\"\n")
        
    os.chmod(runner_path, 0o755)

    print(f"✅ Generated {len(generated_configs)} configuration files under {out_dir}/")
    print(f"✅ Generated execution script: {runner_path}")

if __name__ == "__main__":
    main()
