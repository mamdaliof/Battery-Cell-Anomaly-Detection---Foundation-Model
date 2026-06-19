#!/usr/bin/env python3
"""
Script to check training run status inside the outputs directory.
Identifies completed runs (with a DONE file and verified weights) and interrupted/incomplete runs.
"""

import os
import sys
import yaml
import argparse
import re
import json

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

def verify_trainer_state(root):
    state_path = os.path.join(root, "trainer_state.json")
    if not os.path.exists(state_path) or os.path.getsize(state_path) == 0:
        return False
    try:
        with open(state_path, "r") as f:
            json.load(f)
        return True
    except Exception:
        return False

def find_config_by_stem(cfg_stem, task, custom_configs_dir=None, strategy=None):
    if not cfg_stem:
        return None
        
    search_dirs = []
    if custom_configs_dir:
        search_dirs.append(custom_configs_dir)
    else:
        # Default task-specific directories
        if task == "det" and strategy:
            search_dirs.append(os.path.join("configs", "det", f"ablations_{strategy}"))
        elif task == "cls" and strategy:
            search_dirs.append(os.path.join("configs", "cls", f"ablations_{strategy}"))
        else:
            search_dirs.append(os.path.join("configs", task))
        
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for r, d, f_names in os.walk(search_dir):
            if f"{cfg_stem}.yaml" in f_names:
                return os.path.join(r, f"{cfg_stem}.yaml")
                
    return None

def find_matching_config_by_content(run_cfg, task, custom_configs_dir=None, strategy=None):
    search_dirs = []
    if custom_configs_dir:
        search_dirs.append(custom_configs_dir)
    else:
        if task == "det" and strategy:
            search_dirs.append(os.path.join("configs", "det", f"ablations_{strategy}"))
        elif task == "cls" and strategy:
            search_dirs.append(os.path.join("configs", "cls", f"ablations_{strategy}"))
        else:
            search_dirs.append(os.path.join("configs", task))
        
    def is_equiv(a, b):
        for k in ("model_name", "data", "head", "peft",
                  "learning_rate", "num_epochs", "imbalance",
                  "yolo_model_config", "yolo_data_yaml", "fold", "seed"):
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

def check_outputs(base_path="outputs", configs_dir=None, show_completed=False):
    incomplete_dirs = []
    failed_runs = []
    completed_runs = []

    # 1. Scan subdirectories under base_path
    if not os.path.exists(base_path):
        print(f"❌ Error: Path '{base_path}' does not exist.")
        return None

    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        parts = [p for p in rel_path.split(os.sep) if p and p != "."]
        
        if any(p in ("log", "runs", ".ipynb_checkpoints") for p in parts):
            continue

        # Identify run directories by the timestamp format (e.g. 20260609_222030)
        timestamp = parts[-1] if parts else ""
        if not re.match(r"^\d{8}_\d{6}$", timestamp):
            continue
            
        run_name = parts[-2] if len(parts) >= 2 else ""

        # Determine the task (cls or det) based on path or contents
        task = "unknown"
        for part in parts:
            if "cls" in part:
                task = "cls"
                break
            elif "det" in part or "no_cell" in part or "only_cell" in part or "abnormal_only" in part:
                task = "det"
                break

        if task == "unknown":
            # Check config contents for yolo or dino detection
            config_path = os.path.join(root, "config.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(root, "args.yaml")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = yaml.safe_load(f)
                    if cfg and ("yolo_model_config" in cfg or "yolo_data_yaml" in cfg):
                        task = "det"
                    else:
                        task = "cls"
                except Exception:
                    task = "cls"
            else:
                task = "cls"

        # Check successful completion conditions
        has_done = "DONE" in files
        weights_ok = verify_weights(root, task)
        state_ok = verify_trainer_state(root)
        
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
                
        # Determine strategy (all_label, no_cell, abnormal_only)
        strategy = None
        for part in parts:
            if "all_label" in part or "det_all" in part or "cls_all" in part:
                strategy = "all_label"
                break
            elif "no_cell" in part or "det_no_cell" in part:
                strategy = "no_cell"
                break
            elif "only_cell" in part or "abnormal_only" in part or "det_abnormal" in part:
                strategy = "abnormal_only"
                break

        # Find matched config file
        matched_config = find_config_by_stem(cfg_stem, task, configs_dir, strategy)
        if not matched_config and cfg:
            matched_config = find_matching_config_by_content(cfg, task, configs_dir, strategy)
            
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

        run_info = {
            "dir": root,
            "task": task,
            "model": model_name,
            "peft_type": peft_type,
            "imbalance_strategy": imb_strategy,
            "lr": lr,
            "epochs": epochs,
            "matched_cfg": matched_config
        }

        if has_done and weights_ok and state_ok:
            completed_runs.append(run_info)
        elif has_done:
            # Done file exists but weights or trainer state is invalid
            reasons = []
            if not weights_ok:
                reasons.append("Missing/invalid weights")
            if not state_ok:
                reasons.append("Missing/corrupt trainer_state.json")
            run_info["fail_reason"] = ", ".join(reasons)
            failed_runs.append(run_info)
        else:
            # No Done file -> Interrupted
            incomplete_dirs.append(run_info)

    # Display results
    print("\n" + "=" * 80)
    print("📋 ABLATION STUDY RUN STATUS SUMMARY")
    print("=" * 80)
    
    # Track which original ablation configs have already completed successfully at least once
    completed_configs = set(run["matched_cfg"] for run in completed_runs if run.get("matched_cfg"))
    
    print(f"\n📊 Successfully Completed Runs: {len(completed_runs)}")
    print(f"❌ Completed but Failed/Incorrect Runs: {len(failed_runs)}")
    print(f"🛑 Interrupted / Incomplete Runs: {len(incomplete_dirs)}")
    
    # Print Incorrect runs
    if failed_runs:
        print("\n❌ List of Failed / Incorrect Runs (Completed but corrupt):")
        for run in sorted(failed_runs, key=lambda x: x["dir"]):
            cfg_suffix = f" -> {run['matched_cfg']}" if run['matched_cfg'] else " -> (No matching config)"
            already_done = " (Successfully completed in another run)" if run['matched_cfg'] in completed_configs else ""
            print(f"  - {run['dir']}{cfg_suffix} [Reason: {run['fail_reason']}]{already_done}")

    # Print Incomplete runs
    if incomplete_dirs:
        print("\n🛑 List of Interrupted / Incomplete Runs (Missing DONE file):")
        for run in sorted(incomplete_dirs, key=lambda x: x["dir"]):
            cfg_suffix = f" -> {run['matched_cfg']}" if run['matched_cfg'] else " -> (No matching config)"
            already_done = " (Successfully completed in another run)" if run['matched_cfg'] in completed_configs else ""
            print(f"  - {run['dir']}{cfg_suffix}{already_done}")
            
    # Config files to run again
    re_run_configs = set()
    for run in failed_runs + incomplete_dirs:
        if run['matched_cfg'] and run['matched_cfg'] not in completed_configs:
            re_run_configs.add(run['matched_cfg'])
            
    if re_run_configs:
        print("\n🛠️ Config files to run again (Not yet completed successfully):")
        for cfg_file in sorted(re_run_configs):
            print(f"  {cfg_file}")

    if not show_completed and completed_runs:
        print(f"\n💡 Info: {len(completed_runs)} runs completed successfully. Use the --show-completed argument to list them.")

    # Attempt to print completed runs table (if show_completed is True)
    if show_completed and completed_runs:
        print("\n✨ Successfully Completed Runs Details:")
        completed_runs_display = []
        for run in completed_runs:
            d = run.copy()
            d.pop("matched_cfg", None)
            completed_runs_display.append(d)
            
        try:
            import pandas as pd
            df_summary = pd.DataFrame(completed_runs_display)
            print(df_summary.to_string(index=False))
        except ImportError:
            headers = ["Task", "Model", "PEFT", "Imbalance", "LR", "Epochs"]
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
    parser.add_argument(
        "--show-completed",
        action="store_true",
        help="Display the table of successfully completed runs"
    )
    args = parser.parse_args()
    check_outputs(args.base_path, args.configs_dir, args.show_completed)
