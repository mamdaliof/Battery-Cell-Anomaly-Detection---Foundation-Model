#!/usr/bin/env python3
"""
check_model_init.py

Loads a single ablation config, initialises the full DinoV3Classifier
(backbone + PEFT + head), and prints a one-line PASS/FAIL summary.
Exit code 0 = success, 1 = failure.

Usage:
    python3 scripts/check_model_init.py --config configs/ablations/03_lora_vits16_r8_all_lr0.0003.yaml
"""
import argparse
import sys
import traceback

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = args.config

    try:
        # ── Load config ──────────────────────────────────────────────────────
        from bcadfm.utils.config import load_yaml_config
        cfg = load_yaml_config(config_path)

        # ── Build label maps ─────────────────────────────────────────────────
        id2label = {0: cfg.data.normal_class_name, 1: cfg.data.abnormal_class_name}
        label2id = {v: k for k, v in id2label.items()}

        # ── Instantiate model ─────────────────────────────────────────────────
        from bcadfm.models.dinov3_classifier import DinoV3Classifier
        model = DinoV3Classifier(
            model_name_or_path=cfg.model_name,
            head_config=cfg.head,
            peft_config=cfg.peft,
            id2label=id2label,
            label2id=label2id,
        )

        # ── Count parameters ──────────────────────────────────────────────────
        total      = sum(p.numel() for p in model.parameters())
        trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct        = 100.0 * trainable / total if total > 0 else 0.0

        peft_type  = cfg.peft.type
        model_name = cfg.model_name.split("/")[-1]

        print(
            f"PASS | {config_path} | peft={peft_type} | model={model_name} | "
            f"total={total:,} | trainable={trainable:,} ({pct:.4f}%)"
        )
        return 0

    except Exception:
        print(f"FAIL | {config_path}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
