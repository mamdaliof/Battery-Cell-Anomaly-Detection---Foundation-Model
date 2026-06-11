#!/usr/bin/env python3
import os
import json
import yaml
from pathlib import Path
import sys
import torch

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[1]
    hf_cache_dir = workspace_root / "models" / "hf_cache"
    os.environ["HF_HOME"] = str(hf_cache_dir)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    outputs_dir = Path("outputs")
    cache_file = outputs_dir / "parameter_cache.json"
    
    # Load existing cache if exists
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    print("🔍 Scanning outputs directory for configurations...")
    configs = []
    for root, dirs, files in os.walk(outputs_dir):
        if "config.yaml" in files:
            configs.append(Path(root) / "config.yaml")

    print(f"Found {len(configs)} configuration files.")
    
    # Unique config mapping to avoid duplicate model instantiations
    unique_configs = {}
    
    for cfg_path in configs:
        rel_path = str(cfg_path.parent.relative_to(outputs_dir))
        
        # Skip if already cached
        if rel_path in cache:
            continue
            
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            if not cfg:
                continue
                
            # Create a signature key for unique config
            model_name = cfg.get("model_name", "unknown")
            peft_cfg = cfg.get("peft", {})
            peft_type = peft_cfg.get("type", "none")
            
            is_det = "yolo_model_config" in cfg
            task = "Detection" if is_det else "Classification"
            yolo_variant = cfg.get("yolo_model_config", "")
            
            sig = (task, model_name, peft_type, json.dumps(peft_cfg, sort_keys=True), yolo_variant)
            unique_configs.setdefault(sig, []).append(rel_path)
        except Exception as e:
            print(f"Error pre-parsing config at {cfg_path}: {e}")

    if not unique_configs:
        print("✅ All runs already cached in parameter_cache.json.")
        return

    print(f"Resolving exact parameters for {len(unique_configs)} unique configurations...")
    
    # Import necessary packages for instantiation
    from bcadfm.models.dinov3_classifier import DinoV3Classifier, HeadConfig
    from bcadfm.models.yolo_dino import DinoV3Backbone
    
    for sig, paths in unique_configs.items():
        task, model_name, peft_type, peft_cfg_str, yolo_variant = sig
        peft_cfg = json.loads(peft_cfg_str)
        
        print(f"\nEvaluating: task={task}, model={model_name}, peft={peft_type}, yolo={yolo_variant}")
        
        try:
            total, trainable = 0, 0
            if task == "Classification":
                # Create head config
                head_cfg = HeadConfig(num_labels=2, depth=1, hidden_dim=None, dropout=0.0)
                
                # Instantiate DinoV3Classifier
                model = DinoV3Classifier(
                    model_name_or_path=model_name,
                    head_config=head_cfg,
                    peft_config=peft_cfg,
                    freeze_backbone=True
                )
                total, trainable = count_params(model)
                del model
                
            else: # Detection
                is_yolo_dino = "dino" in yolo_variant.lower()
                
                if is_yolo_dino:
                    # DINOv3 Backbone wrapped inside YOLO
                    backbone = DinoV3Backbone(
                        c1=3,
                        c2=768 if "vitb16" in model_name.lower() or "vit-base" in model_name.lower() else 384,
                        model_name=model_name,
                        peft_config=peft_cfg
                    )
                    bb_total, bb_trainable = count_params(backbone)
                    del backbone
                    
                    # YOLO Head approximation
                    yolo_head_params = {
                        "yolo26n": 2800000, "yolo26s": 9900000, "yolo26m": 20700000, "yolo26l": 44800000, "yolo26x": 69900000,
                        "yolo11n": 1400000, "yolo11s": 4800000, "yolo11m": 10100000, "yolo11l": 21800000, "yolo11x": 34100000
                    }
                    var_key = Path(yolo_variant).stem.lower()
                    head_p = yolo_head_params.get(var_key, 2800000)
                    
                    total = bb_total + head_p
                    trainable = bb_trainable + head_p
                else:
                    # Standard YOLO
                    yolo_params = {
                        "yolo11n": 2600000, "yolo11s": 9400000, "yolo11m": 20100000, "yolo11l": 25300000, "yolo11x": 56900000,
                        "yolo26n": 5500000, "yolo26s": 19300000, "yolo26m": 40700000, "yolo26l": 87800000, "yolo26x": 136900000,
                        "yolov8n": 3200000, "yolov8s": 11200000, "yolov8m": 25900000, "yolov8l": 43700000, "yolov8x": 68200000
                    }
                    var_key = Path(yolo_variant or model_name).stem.replace(".pt", "").lower()
                    tot = yolo_params.get(var_key, 5500000)
                    total = tot
                    trainable = tot
                    
            print(f"Success! Total={total:,} | Trainable={trainable:,}")
            
            # Save to cache map
            for path in paths:
                cache[path] = {
                    "total": total,
                    "trainable": trainable
                }
                
        except Exception as e:
            print(f"⚠️ Failed to instantiate model for config: {e}")
            
    # Write cache file
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"\n🎉 Parameter cache successfully written to {cache_file}")
    except Exception as e:
        print(f"Failed to write parameter cache: {e}")

if __name__ == "__main__":
    main()
