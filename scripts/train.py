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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Task name prefix to avoid local directory name conflicts with HF repos
    task_name = "cls"  # classification; later other tasks (e.g. seg, det) can use different prefixes

    # Create run-specific output directory: outputs/{task_name}__{safe_model_name}/{timestamp}
    base_out = Path(cfg.output_dir)
    safe_model_name = cfg.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"{task_name}__{safe_model_name}" / timestamp
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
    model.to(device)

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
        warmup_ratio=cfg.scheduler.warmup_ratio,
        # ── Evaluation & checkpointing: once per epoch ─────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best,
        greater_is_better=True,
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
