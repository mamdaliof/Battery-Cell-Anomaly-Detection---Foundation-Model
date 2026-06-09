#!/usr/bin/env python3
"""
Script to check training run status inside the outputs directory.
Identifies completed runs (with a DONE file and verified weights) and interrupted/incomplete runs.
"""

import os
import sys
import yaml
import argparse

def verify_weights(root, task):
    if task == "cls":
        return any(
            os.path.exists(os.path.join(root, name)) and os.path.getsize(os.path.join(root, name)) > 0
            for name in ("model.safetensors", "pytorch_model.bin", "best_f1.pt", "best_loss.pt")
        )
    elif task == "det":
        weights_dir = os.path.join(root, "weights")
        return any(
            os.path.exists(os.path.join(weights_dir, name)) and os.path.getsize(os.path.join(weights_dir, name)) > 0
            for name in ("best.pt", "last.pt")
        )
    return False

def find_config_by_stem(cfg_stem, task, custom_configs_dir=None):
    if not cfg_stem:
        return None
        
    search_dirs = []
    if custom_configs_dir:
        search_dirs.append(custom_configs_dir)
    else:
        # Default task-specific directories
        search_dirs.append(os.path.join("configs", task))
        
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for r, d, f_names in os.walk(search_dir):
            if f"{cfg_stem}.yaml" in f_names:
                return os.path.join(r, f"{cfg_stem}.yaml")
                
    return None

def find_matching_config_by_content(run_cfg, task, custom_configs_dir=None):
    search_dirs = []
    if custom_configs_dir:
        search_dirs.append(custom_configs_dir)
    else:
        search_dirs.append(os.path.join("configs", task))
        
    def is_equiv(a, b):
        for k in ("model_name", "data", "head", "peft",
                  "learning_rate", "num_epochs", "imbalance"):
            if a.get(k) != b.get(k):
                return False
        return True

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for r, d, f_names in os.walk(search_dir):
            for file in sorted(f_names):
                if file.endswith(".yaml"):
                    cfg_path = os.path.join(r, file)
                    try:
                        with open(cfg_path, "r") as f:
                            ablation_cfg = yaml.safe_load(f)
                        if ablation_cfg and is_equiv(run_cfg, ablation_cfg):
                            return cfg_path
                    except Exception:
                        pass
    return None

def check_outputs(base_path="outputs", configs_dir=None):
    incomplete_dirs = []
    completed_runs = []

    # 1. Scan subdirectories under base_path
    if not os.path.exists(base_path):
        print(f"❌ Error: Path '{base_path}' does not exist.")
        return None

    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        parts = [p for p in rel_path.split(os.sep) if p and p != "."]
        
        # We only care about directories at depth 3 under outputs (e.g. outputs/{cls|det}/{run_name}/{timestamp})
        if len(parts) != 3:
            continue
            
        task = parts[0]
        run_name = parts[1]
        timestamp = parts[2]
        
        if task not in ("cls", "det"):
            continue
            
        if any(p in ("log", "runs", ".ipynb_checkpoints") for p in parts):
            continue

        # Check if training finished successfully and weights were saved
        has_done = "DONE" in files
        if has_done:
            if not verify_weights(root, task):
                has_done = False

        # Extract config stem from folder name suffix
        cfg_stem = None
        if "__" in run_name:
            cfg_stem = run_name.split("__")[-1]

        # Try to parse config to extract run parameters
        cfg = None
        config_path = os.path.join(root, "config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(root, "args.yaml")
            
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                pass
                
        # Find matched config file
        matched_config = find_config_by_stem(cfg_stem, task, configs_dir)
        if not matched_config and cfg:
            matched_config = find_matching_config_by_content(cfg, task, configs_dir)
            
        # Fallback to reading parameters from matched_config if cfg is None
        if cfg is None and matched_config:
            try:
                with open(matched_config, "r") as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                pass

        if cfg:
            model_name = cfg.get("model_name") or "unknown"
            peft_type = cfg.get("peft", {}).get("type") or cfg.get("peft_type") or "none"
            
            imb_cfg = cfg.get("imbalance", {})
            imb_strategy = (
                imb_cfg.get("strategy") or 
                imb_cfg.get("oversampling_method") or 
                cfg.get("imbalance_strategy") or 
                "none"
            )
            
            lr = cfg.get("learning_rate") or cfg.get("lr") or "unknown"
            epochs = cfg.get("num_epochs") or cfg.get("epochs") or "unknown"
        else:
            model_name = "unknown"
            peft_type = "unknown"
            imb_strategy = "unknown"
            lr = "unknown"
            epochs = "unknown"

        if has_done:
            completed_runs.append({
                "dir": root,
                "task": task,
                "model": model_name,
                "peft_type": peft_type,
                "imbalance_strategy": imb_strategy,
                "lr": lr,
                "epochs": epochs,
                "matched_cfg": matched_config
            })
        else:
            incomplete_dirs.append((root, matched_config))

    # Display results
    print("\n" + "=" * 80)
    print("📋 ABLATION STUDY RUN STATUS SUMMARY")
    print("=" * 80)
    
    # Track which original ablation configs have already completed at least once
    completed_configs = set(run["matched_cfg"] for run in completed_runs if run.get("matched_cfg"))
    
    print(f"\n📊 Completed Runs: {len(completed_runs)}")
    print(f"❌ Incomplete / Interrupted Runs: {len(incomplete_dirs)}")
    
    if incomplete_dirs:
        print("\n🛑 List of Incomplete / Interrupted Runs:")
        for directory, matched_cfg in sorted(incomplete_dirs, key=lambda x: x[0]):
            if matched_cfg:
                suffix = " (Already completed in another run)" if matched_cfg in completed_configs else ""
                print(f"  - {directory}  ->  {matched_cfg}{suffix}")
            else:
                print(f"  - {directory}  ->  (No matching ablation config found)")
        
        print("\n🛠️ Config files to run again:")
        configs_to_run = sorted(list(set(item[1] for item in incomplete_dirs if item[1] and item[1] not in completed_configs)))
        if configs_to_run:
            for cfg_file in configs_to_run:
                print(f"  {cfg_file}")
        else:
            print("  (All interrupted runs have already been successfully completed in subsequent runs!)")
    else:
        print("\n🎉 No incomplete runs found!")

    # Attempt to print completed runs table (clean display without matched_cfg column)
    if completed_runs:
        print("\n✨ Completed Runs Details:")
        # Create a display list without internal columns
        completed_runs_display = []
        for run in completed_runs:
            d = run.copy()
            d.pop("matched_cfg", None)
            completed_runs_display.append(d)
            
        try:
            import pandas as pd
            df_summary = pd.DataFrame(completed_runs_display)
            print(df_summary.to_string(index=False))
            return pd.DataFrame(completed_runs)
        except ImportError:
            # Fallback to standard library formatting if pandas is not installed
            headers = ["Task", "Model", "PEFT", "Imbalance", "LR", "Epochs"]
            # Extract basic info for quick terminal table
            rows = []
            for run in completed_runs:
                model_short = run["model"].split("/")[-1] if run["model"] else "none"
                rows.append([
                    run["task"],
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
        default=None,
        help="Directory containing original ablation configs (default: auto-detect based on task)"
    )
    args = parser.parse_args()
    check_outputs(args.base_path, args.configs_dir)
