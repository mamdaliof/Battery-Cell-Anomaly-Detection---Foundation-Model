#!/usr/bin/env python

"""Training entry point using a YAML config.

Usage (single GPU):

    python scripts/train.py --config configs/baseline.yaml

For DDP/multi-GPU, launch with `torchrun` or `accelerate launch`.
"""

from __future__ import annotations

import argparse

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from bcadfm.data.dataset import BatteryCellDataset, build_augmentation_pipeline
from bcadfm.models.dinov3_classifier import DinoV3Classifier
from bcadfm.utils.config import TrainingConfig, load_yaml_config


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

    # Build datasets
    train_transform = build_augmentation_pipeline(cfg.data, split="train")
    val_transform = None

    train_dataset = BatteryCellDataset(
        split="train",
        data_config=cfg.data,
        model_name_or_path=cfg.model_name,
        transform=train_transform,
        image_size_override=cfg.data.image_size,
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
        freeze_backbone=True,
        id2label={0: "normal", 1: "abnormal"},
        label2id={"normal": 0, "abnormal": 1},
    )
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best,
        greater_is_better=cfg.greater_is_better,
        remove_unused_columns=False,
        report_to=[],
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
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
