#!/usr/bin/env python

"""Training entry point for YOLO detection using a unified YAML config.

Usage (single GPU):

    python scripts/train_detection.py --config configs/det/baseline.yaml
"""

from __future__ import annotations

# Monkeypatch torch for platforms/versions lacking float8_e8m0fnu (needed by transformers dev branch)
try:
    import torch
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float16)
except ImportError:
    pass

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import os
import random
import numpy as np

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[1]
    hf_cache_dir = workspace_root / "models" / "hf_cache"
    os.environ["HF_HOME"] = str(hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # If the model is cached in the default home directory, copy it to the local workspace cache
    default_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if default_cache.exists():
        for p in default_cache.glob("models--facebook--dino*"):
            if p.is_dir():
                target_hub_dir = hf_cache_dir / "hub"
                target_dir = target_hub_dir / p.name
                if not target_dir.exists():
                    try:
                        import shutil
                        target_hub_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(p, target_dir, symlinks=True)
                    except Exception:
                        pass

import torch

from bcadfm.utils.config import TrainingConfig, load_yaml_config
from bcadfm.utils.yolo_utils import register_yolo_dino, set_active_peft_config
from bcadfm.training.yolo_trainer import CustomDetectionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO Detector with unified YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to unified YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load unified config schema (mirrors classification config)
    cfg: TrainingConfig = load_yaml_config(args.config)

    # Set random seeds for reproducibility
    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Register custom DinoV3 + SFP layers dynamically inside Ultralytics
    # Call this on all processes before any model parsing or initialization
    register_yolo_dino()

    # Pass the active PEFT config schema to the model parser registry
    set_active_peft_config(cfg.peft)

    task_name = "det"  # detection task name
    
    # Handle dataset fold path combination
    if cfg.fold is not None:
        fold_str = f"fold_{cfg.fold}" if isinstance(cfg.fold, int) or (isinstance(cfg.fold, str) and cfg.fold.isdigit()) else str(cfg.fold)
        cfg.data.data_dir = str(Path(cfg.data.data_dir) / fold_str)

    # Create run-specific output directory under outputs/det
    base_out = Path(cfg.output_dir)
    safe_model_name = cfg.model_name.replace("/", "-")
    cfg_stem = Path(args.config).stem
    if cfg.fold is not None:
        cfg_stem = f"{cfg_stem}_fold_{cfg.fold}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"{safe_model_name}__{cfg_stem}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the used config file into the run directory for reproducibility
    shutil.copy2(args.config, run_dir / "config.yaml")

    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if is_main_process:
        print("\n" + "=" * 80)
        print("🚀 STARTING DETECTION TRAINING PIPELINE (YOLO)")
        print("=" * 80)
        print(f"📄 Config: {args.config}")
        print(f"📦 Backbone Model: {cfg.model_name}")
        print(f"🛠️ YOLO Model YAML: {cfg.yolo_model_config}")
        print(f"📂 Output: {run_dir}")
        print(f"🧪 PEFT Type: {cfg.peft.type}")
        print("ℹ️  [INFO] Data augmentations defined in the YAML config are placeholders kept for schema compatibility.")
        print("ℹ️  [INFO] YOLO training will utilize standard internal Ultralytics augmentations.")
        print("=" * 80 + "\n")

    # Load and potentially rewrite the YOLO data config YAML to include the fold directory
    import yaml
    yolo_data_path = cfg.yolo_data_yaml or "data/battery_detection_all.yaml"
    if cfg.fold is not None:
        fold_str = f"fold_{cfg.fold}" if isinstance(cfg.fold, int) or (isinstance(cfg.fold, str) and cfg.fold.isdigit()) else str(cfg.fold)
        with open(yolo_data_path, "r") as f:
            yolo_data_dict = yaml.safe_load(f)
        orig_path = yolo_data_dict.get("path", "")
        yolo_data_dict["path"] = str(Path(orig_path) / fold_str)
        
        # Write temporary data YAML inside the run directory to prevent parallel job collision
        temp_data_yaml = run_dir / "yolo_data_fold.yaml"
        with open(temp_data_yaml, "w") as f:
            yaml.safe_dump(yolo_data_dict, f)
        yolo_data_path = str(temp_data_yaml)

    # Translate unified configuration schema into YOLO training parameters
    yolo_overrides = {
        "model": cfg.yolo_model_config or "configs/det/yolo26_dino.yaml",
        "data": yolo_data_path,
        "epochs": cfg.num_epochs,
        "batch": cfg.batch_size,
        "imgsz": cfg.data.image_size or 224,  # Configurable input image size, defaults to DINO standard 224
        "lr0": cfg.learning_rate,
        "seed": cfg.seed,
        "amp": cfg.amp.fp16 or cfg.amp.bf16,
        "patience": cfg.early_stopping_patience,
        "project": str(run_dir.parent.resolve()),
        "name": run_dir.name,
        "exist_ok": True,
        "val": True,
        "cos_lr": cfg.scheduler.lr_scheduler_type == "cosine",
    }

    # Apply augmentation overrides from config to YOLO overrides
    if not cfg.data.augmentations_enabled:
        # Scenario A: Augmentations disabled, zero out all YOLO augmentations
        yolo_overrides.update({
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
            "fliplr": 0.0, "flipud": 0.0, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0
        })
    else:
        # Scenario B: Direct overrides present in YAML
        if cfg.data.yolo_augmentations is not None:
            yolo_overrides.update(cfg.data.yolo_augmentations)
        else:
            # Scenario C: Fallback to mapping classification augmentations to YOLO equivalents
            # Mapping probability values
            yolo_overrides["fliplr"] = cfg.data.horizontal_flip_prob
            
            # Mapping degrees (only if rotation probability is > 0)
            yolo_overrides["degrees"] = cfg.data.rotation_degrees if cfg.data.rotation_prob > 0 else 0.0
            
            # Mapping color jitter to HSV shifts (only if color jitter probability is > 0)
            if cfg.data.color_jitter_prob > 0:
                yolo_overrides["hsv_h"] = cfg.data.color_jitter_hue
                yolo_overrides["hsv_s"] = cfg.data.color_jitter_saturation
                yolo_overrides["hsv_v"] = cfg.data.color_jitter_brightness
            else:
                yolo_overrides["hsv_h"] = 0.0
                yolo_overrides["hsv_s"] = 0.0
                yolo_overrides["hsv_v"] = 0.0
                
            # Mapping random resized crop scale to YOLO scale range
            if cfg.data.random_resized_crop_prob > 0 and cfg.data.random_resized_crop_scale is not None:
                min_scale = cfg.data.random_resized_crop_scale[0]
                scale_dev = max(0.0, 1.0 - min_scale)
                yolo_overrides["scale"] = scale_dev

    # Instantiate CustomDetectionTrainer subclassing Ultralytics DetectionTrainer
    trainer = CustomDetectionTrainer(
        overrides=yolo_overrides,
        normal_class_name=cfg.data.normal_class_name,
        abnormal_class_name=cfg.data.abnormal_class_name
    )

    # Run the training loop
    trainer.train()

    if is_main_process:
        print("\n" + "=" * 80)
        print("🎉 DETECTION TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"📂 Output: {run_dir}")
        print("=" * 80 + "\n")
        
        # Verify that training outputs are present and non-empty
        weights_dir = run_dir / "weights"
        best_weights = weights_dir / "best.pt"
        last_weights = weights_dir / "last.pt"
        
        weights_exist = (
            (best_weights.exists() and best_weights.stat().st_size > 0) or
            (last_weights.exists() and last_weights.stat().st_size > 0)
        )
        if weights_exist:
            # Touch DONE file to mark successful completion
            (run_dir / "DONE").touch()
        else:
            raise RuntimeError(f"Training finished but YOLO model weights were not successfully saved in {weights_dir}.")


if __name__ == "__main__":
    main()
