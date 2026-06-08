#!/usr/bin/env python

"""Training entry point using a YAML config.

Usage (single GPU):

    python scripts/train.py --config configs/baseline.yaml

For DDP/multi-GPU, launch with `torchrun` or `accelerate launch`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
from dataclasses import asdict
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
        for p in default_cache.glob("models--facebook--dinov3*"):
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

from transformers import EarlyStoppingCallback, TrainingArguments

from bcadfm.data.dataset import BatteryCellDataset, build_augmentation_pipeline
from bcadfm.metrics.cls_metrics import compute_cls_metrics
from bcadfm.metrics.cls_callbacks import BeautifulLoggingCallback, SaveTwoBestClsModelsCallback
from bcadfm.models.dinov3_classifier import DinoV3Classifier
from bcadfm.training import ImbalanceTrainer
from bcadfm.utils.config import TrainingConfig, load_yaml_config
from bcadfm.utils.model_utils import log_parameter_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DinoV3Classifier with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg: TrainingConfig = load_yaml_config(args.config)

    # Set random seeds for reproducibility (H8 Fix)
    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Task name prefix to avoid local directory name conflicts with HF repos
    task_name = "cls"  # classification; later other tasks (e.g. seg, det) can use different prefixes

    # Create run-specific output directory: outputs/{task_name}__{safe_model_name}__{cfg_stem}/{timestamp}
    base_out = Path(cfg.output_dir)
    safe_model_name = cfg.model_name.replace("/", "-")
    cfg_stem = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"{task_name}__{safe_model_name}__{cfg_stem}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the used config file into the run directory for reproducibility
    shutil.copy2(args.config, run_dir / "config.yaml")

    # Build datasets
    train_transform = build_augmentation_pipeline(cfg.data, split="train")
    val_transform = None

    oversample_data = cfg.imbalance.oversampling_method == "data_level"

    train_dataset = BatteryCellDataset(
        split="train",
        data_config=cfg.data,
        model_name_or_path=cfg.model_name,
        transform=train_transform,
        image_size_override=cfg.data.image_size,
        oversample=oversample_data,
        seed=cfg.seed,
    )
    eval_dataset = BatteryCellDataset(
        split="val",
        data_config=cfg.data,
        model_name_or_path=cfg.model_name,
        transform=val_transform,
        image_size_override=cfg.data.image_size,
    )

    # Model
    model = DinoV3Classifier(
        model_name_or_path=cfg.model_name,
        head_config=cfg.head,
        peft_config=cfg.peft,
        freeze_backbone=True,
        id2label={0: "normal", 1: "abnormal"},
        label2id={"normal": 0, "abnormal": 1},
    )
    # Note: model.to(device) is intentionally omitted — the HF Trainer
    # manages device placement internally.

    # Log parameters summary
    log_parameter_summary(model, "DinoV3Classifier")

    # Check preprocessor type and prepare logs
    is_official = train_dataset.processor is not None
    preprocessor_str = (
        "🤖 Official Hugging Face AutoImageProcessor" if is_official
        else "🛠️ Fallback manual torchvision preprocessor (DINOv3-style)"
    )

    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if is_main_process:
        print("\n" + "=" * 80)
        print("🚀 STARTING TRAINING PIPELINE")
        print("=" * 80)
        print(f"📄 Config: {args.config}")
        print(f"📦 Model:  {cfg.model_name}")
        print(f"📂 Output: {run_dir}")
        print(f"⚙️ Preprocessor: {preprocessor_str}")
        print("=" * 80 + "\n")

    # Training arguments (scheduler + AMP included)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.scheduler.lr_scheduler_type,
        # Compute warmup_steps from actual dataset size (warmup_ratio is deprecated in v5.2) (C4 Fix)
        warmup_steps=int(
            (len(train_dataset) / (cfg.batch_size * max(1, int(os.environ.get("WORLD_SIZE", "1")))))
            * cfg.num_epochs
            * cfg.scheduler.warmup_ratio
        ),
        # ── Evaluation & checkpointing: once per epoch ─────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best,
        greater_is_better=cfg.greater_is_better,
        seed=cfg.seed,
        save_total_limit=2,
        # ── Data loading ───────────────────────────────────────────────────
        dataloader_num_workers=4,         # parallel CPU workers; eliminates GPU idle time
        dataloader_pin_memory=True,       # faster CPU→GPU transfer
        # ── Misc ───────────────────────────────────────────────────────────
        remove_unused_columns=False,
        report_to=[],
        fp16=cfg.amp.fp16,
        bf16=cfg.amp.bf16,
        ddp_find_unused_parameters=False,
    )
    # Map deprecated evaluation_strategy to eval_strategy to ensure EarlyStoppingCallback compatibility (H9 Fix)
    training_args.evaluation_strategy = training_args.eval_strategy

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
        ),
        SaveTwoBestClsModelsCallback(run_dir=str(run_dir)),
        BeautifulLoggingCallback(),
    ]

    trainer = ImbalanceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_cls_metrics,
        callbacks=callbacks,
        imbalance_config=asdict(cfg.imbalance),
    )

    train_result = trainer.train()

    if is_main_process:
        # Save final model state and trainer state to the root run directory
        trainer.save_model()
        trainer.save_state()

        metrics = train_result.metrics
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"⏱️ Runtime:            {metrics.get('train_runtime', 0.0):.2f} seconds")
        print(f"📊 Samples/sec:        {metrics.get('train_samples_per_second', 0.0):.2f}")
        print(f"📉 Final Train Loss:   {metrics.get('train_loss', 0.0):.4f}")
        print(f"🔁 Total Epochs:       {metrics.get('epoch', 0.0):.1f}")
        print("=" * 80 + "\n")
        
        # Touch DONE file to mark successful completion
        (run_dir / "DONE").touch()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
