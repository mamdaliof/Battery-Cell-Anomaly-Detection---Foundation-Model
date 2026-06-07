#!/usr/bin/env python3
"""
Script to check training run status inside the outputs directory.
Identifies completed runs (with a DONE file) and interrupted/incomplete runs.
"""

import os
import sys
import yaml
import argparse

def check_outputs(base_path="outputs"):
    incomplete_dirs = []
    completed_runs = []

    # 1. Scan subdirectories under base_path
    if not os.path.exists(base_path):
        print(f"❌ Error: Path '{base_path}' does not exist.")
        return None

    for root, dirs, files in os.walk(base_path):
        # Skip the standard 'log' or 'tb' log folders
        if "log" in root or "runs" in root:
            continue
            
        # We check experiment run folders containing training outputs
        if "config.yaml" in files or "args.yaml" in files:
            # Check if training finished successfully
            if "DONE" not in files:
                incomplete_dirs.append(root)
            else:
                # 2. Parse config.yaml to extract run parameters
                config_path = os.path.join(root, "config.yaml")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            cfg = yaml.safe_load(f)
                        imb_cfg = cfg.get("imbalance", {})
                        imb_strategy = imb_cfg.get("strategy") or imb_cfg.get("oversampling_method") or "none"
                        
                        completed_runs.append({
                            "dir": root,
                            "model": cfg.get("model_name"),
                            "peft_type": cfg.get("peft", {}).get("type", "none"),
                            "imbalance_strategy": imb_strategy,
                            "lr": cfg.get("learning_rate"),
                            "epochs": cfg.get("num_epochs")
                        })
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to parse config at {config_path}: {e}")

    # Display results
    print("\n" + "=" * 80)
    print("📋 ABLATION STUDY RUN STATUS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Completed Runs: {len(completed_runs)}")
    print(f"❌ Incomplete / Interrupted Runs: {len(incomplete_dirs)}")
    
    if incomplete_dirs:
        print("\n🛑 List of Incomplete / Interrupted Runs:")
        for directory in sorted(incomplete_dirs):
            print(f"  - {directory}")
    else:
        print("\n🎉 No incomplete runs found!")

    # Attempt to print completed runs table
    if completed_runs:
        print("\n✨ Completed Runs Details:")
        try:
            import pandas as pd
            df_summary = pd.DataFrame(completed_runs)
            print(df_summary.to_string(index=False))
            return df_summary
        except ImportError:
            # Fallback to standard library formatting if pandas is not installed
            headers = ["Model", "PEFT", "Imbalance", "LR", "Epochs"]
            # Extract basic info for quick terminal table
            rows = []
            for run in completed_runs:
                model_short = run["model"].split("/")[-1] if run["model"] else "none"
                rows.append([
                    model_short[:25],
                    str(run["peft_type"]),
                    str(run["imbalance_strategy"]),
                    str(run["lr"]),
                    str(run["epochs"])
                ])
            
            # Print simple text table
            col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
            fmt_str = "  ".join(f"{{:<{w}}}" for w in col_widths)
            print("  " + fmt_str.format(*headers))
            print("  " + "  ".join("-" * w for w in col_widths))
            for row in rows:
                print("  " + fmt_str.format(*row))
    
    return completed_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check output run status.")
    parser.add_argument(
        "--base-path",
        type=str,
        default="outputs",
        help="Base outputs path to scan (default: outputs)"
    )
    args = parser.parse_args()
    check_outputs(args.base_path)
