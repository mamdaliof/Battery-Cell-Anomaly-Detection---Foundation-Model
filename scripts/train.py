#!/usr/bin/env python

"""Training entry point (temporary, config-less version).

This script wires together:
- Local datasets converted under `data/train/{normal,abnormal}` and `data/val/{normal,abnormal}`.
- The DinoV3Classifier baseline model.
- Hugging Face Trainer for a simple training run.

Later, this will be driven entirely by a config file; for now, key
hyperparameters are hardcoded for smoke-testing on a small subset.

DDP/multi-GPU is handled by launching this script with `torchrun` or
`accelerate launch`. Early stopping and saving the best model are
enabled via Trainer callbacks and arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from bcadfm.data.config import DataConfig
from bcadfm.data.dataset import BatteryCellDataset, build_augmentation_pipeline
from bcadfm.models.dinov3_classifier import DinoV3Classifier, HeadConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DinoV3Classifier on battery cell data (baseline)")
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face model name or path for the DINOv3 backbone",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory for classification data (contains train/ and val/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline",
        help="Where to save checkpoints and logs",
    )
    # Very basic head options for now
    parser.add_argument("--head-depth", type=int, default=1, help="Number of linear layers in the classification head")
    parser.add_argument("--head-hidden-dim", type=int, default=512, help="Hidden size for head when depth > 1")
    parser.add_argument("--head-dropout", type=float, default=0.1, help="Dropout probability in the classification head")

    # Lightweight training hyperparameters (will be moved to config later)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)

    # Early stopping / best model
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of evaluation steps (epochs here) with no improvement before stopping",
    )
    parser.add_argument(
        "--metric-for-best",
        type=str,
        default="eval_loss",
        help="Metric name to monitor for early stopping and best model (e.g. eval_loss, eval_f1)",
    )
    parser.add_argument(
        "--greater-is-better",
        action="store_true",
        help="Set if higher metric values are better (e.g. F1); default assumes lower is better (e.g. loss)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data configuration (minimal for now)
    data_cfg = DataConfig(data_dir=args.data_dir)

    # Build datasets
    train_transform = build_augmentation_pipeline(data_cfg, split="train")
    val_transform = None  # no augmentations for validation

    train_dataset = BatteryCellDataset(
        split="train",
        data_config=data_cfg,
        model_name_or_path=args.model_name,
        transform=train_transform,
        image_size_override=data_cfg.image_size,
    )
    eval_dataset = BatteryCellDataset(
        split="val",
        data_config=data_cfg,
        model_name_or_path=args.model_name,
        transform=val_transform,
        image_size_override=data_cfg.image_size,
    )

    # Model with configurable head depth
    head_cfg = HeadConfig(
        num_labels=2,
        depth=args.head_depth,
        hidden_dim=(args.head_hidden_dim if args.head_depth > 1 else None),
        dropout=args.head_dropout,
    )

    model = DinoV3Classifier(
        model_name_or_path=args.model_name,
        head_config=head_cfg,
        freeze_backbone=True,
        id2label={0: "normal", 1: "abnormal"},
        label2id={"normal": 0, "abnormal": 1},
    )

    model.to(device)

    # Basic training arguments (will later come from config)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best,
        greater_is_better=args.greater_is_better,
        remove_unused_columns=False,  # important for custom vision models
        report_to=[],  # disable W&B etc. by default
    )

    # Early stopping callback (patience measured in evaluation steps, here epochs)
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
        )
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    main()
