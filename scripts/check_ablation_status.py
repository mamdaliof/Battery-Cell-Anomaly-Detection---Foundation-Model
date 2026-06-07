#!/usr/bin/env python3
"""
Script to check training run status inside the outputs directory.
Identifies completed runs (with a DONE file) and interrupted/incomplete runs.
"""

import os
import sys
import yaml
import argparse

def find_matching_config(run_cfg, configs_dir="configs/ablations"):
    if not os.path.exists(configs_dir):
        return None
        
    def is_equiv(a, b):
        for k in ("model_name", "data", "head", "peft",
                  "learning_rate", "num_epochs", "imbalance"):
            if a.get(k) != b.get(k):
                return False
        return True

    for file in sorted(os.listdir(configs_dir)):
        if file.endswith(".yaml"):
            cfg_path = os.path.join(configs_dir, file)
            try:
                with open(cfg_path, "r") as f:
                    ablation_cfg = yaml.safe_load(f)
                if ablation_cfg and is_equiv(run_cfg, ablation_cfg):
                    return os.path.join(configs_dir, file)
            except Exception:
                pass
    return None

def check_outputs(base_path="outputs", configs_dir="configs/ablations"):
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
                config_path = os.path.join(root, "config.yaml")
                matched_config = None
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            run_cfg = yaml.safe_load(f)
                        matched_config = find_matching_config(run_cfg, configs_dir)
                    except Exception:
                        pass
                incomplete_dirs.append((root, matched_config))
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
        for directory, matched_cfg in sorted(incomplete_dirs, key=lambda x: x[0]):
            if matched_cfg:
                print(f"  - {directory}  ->  {matched_cfg}")
            else:
                print(f"  - {directory}  ->  (No matching ablation config found)")
        
        print("\n🛠️ Config files to run again:")
        configs_to_run = sorted(list(set(item[1] for item in incomplete_dirs if item[1])))
        for cfg_file in configs_to_run:
            print(f"  {cfg_file}")
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
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs/ablations",
        help="Directory containing original ablation configs (default: configs/ablations)"
    )
    args = parser.parse_args()
    check_outputs(args.base_path, args.configs_dir)
