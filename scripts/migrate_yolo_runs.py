#!/usr/bin/env python3
import os
import csv
import json
import shutil
from pathlib import Path

def main():
    workspace_dir = Path("/home/mamdaliof/Documents/GitHub/mamdaliof-obsidian/02-Projects/Battery-Cell-Anomaly-Detection---Foundation-Model")
    outputs_det_dir = workspace_dir / "outputs" / "det"
    runs_detect_dir = workspace_dir / "runs" / "detect" / "outputs" / "det"

    if not runs_detect_dir.exists():
        print(f"❌ runs/detect directory not found at {runs_detect_dir}")
        return

    print("🔍 Scanning for completed runs with DONE files in outputs/det...")
    # Find all timestamp directories under outputs/det
    timestamp_dirs = []
    for root, dirs, files in os.walk(outputs_det_dir):
        if "DONE" in files:
            timestamp_dirs.append(Path(root))

    print(f"Found {len(timestamp_dirs)} completed run timestamp directories.")
    
    migrated_count = 0
    for target_dir in timestamp_dirs:
        # target_dir looks like: outputs/det/{model__config}/{timestamp}
        run_name = target_dir.parent.name
        timestamp = target_dir.name
        
        # Source directory under runs/detect
        src_run_dir = runs_detect_dir / run_name
        
        if not src_run_dir.exists():
            print(f"⚠️ Source directory {src_run_dir} does not exist. Skipping.")
            continue
            
        csv_file = src_run_dir / "results.csv"
        if not csv_file.exists():
            print(f"⚠️ results.csv not found in {src_run_dir}. Skipping.")
            continue

        print(f"📦 Migrating: {run_name} ({timestamp})")

        # 1. Parse results.csv and build trainer_state.json
        log_history = []
        best_metric_val = 0.0
        best_epoch = 0
        
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                # Strip spaces from column headers
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
                for row in reader:
                    # Clean row keys and convert values
                    cleaned_row = {}
                    for k, v in row.items():
                        k_clean = k.strip()
                        try:
                            # Convert to float or int
                            cleaned_row[k_clean] = float(v) if '.' in v or 'e' in v.lower() else int(v)
                        except ValueError:
                            cleaned_row[k_clean] = v
                    
                    epoch = float(cleaned_row.get("epoch", 1.0))
                    step = int(epoch)
                    
                    # Compute training loss (sum of train/box_loss, train/cls_loss, train/dfl_loss)
                    t_loss = (
                        cleaned_row.get("train/box_loss", 0.0) +
                        cleaned_row.get("train/cls_loss", 0.0) +
                        cleaned_row.get("train/dfl_loss", 0.0)
                    )
                    
                    # Compute validation loss (sum of val/box_loss, val/cls_loss, val/dfl_loss)
                    v_loss = (
                        cleaned_row.get("val/box_loss", 0.0) +
                        cleaned_row.get("val/cls_loss", 0.0) +
                        cleaned_row.get("val/dfl_loss", 0.0)
                    )
                    
                    # Retrieve learning rate from pg0, pg1, or pg2
                    lr = cleaned_row.get("lr/pg0") or cleaned_row.get("lr/pg1") or cleaned_row.get("lr/pg2") or 0.0
                    
                    # Write training log history entry
                    train_entry = {
                        "epoch": epoch,
                        "learning_rate": float(lr),
                        "loss": float(t_loss),
                        "step": step
                    }
                    log_history.append(train_entry)
                    
                    # Write evaluation log history entry
                    eval_entry = {
                        "epoch": epoch,
                        "step": step,
                        "eval_loss": float(v_loss),
                        "eval_precision": float(cleaned_row.get("metrics/precision(B)", 0.0)),
                        "eval_recall": float(cleaned_row.get("metrics/recall(B)", 0.0)),
                        "eval_mAP50": float(cleaned_row.get("metrics/mAP50(B)", 0.0)),
                        "eval_mAP50-95": float(cleaned_row.get("metrics/mAP50-95(B)", 0.0)),
                    }
                    
                    # Add all custom metrics to eval_entry mapping them to eval_custom_...
                    for k, val in cleaned_row.items():
                        if k.startswith("metrics/custom_"):
                            new_key = k.replace("metrics/custom_", "eval_custom_")
                            eval_entry[new_key] = float(val)
                            
                    log_history.append(eval_entry)
                    
                    # Determine best epoch (prioritize abnormal F1, fallback to mAP50)
                    metric_for_best = "eval_custom_cls_f1/abnormal"
                    current_best_val = eval_entry.get(metric_for_best, 0.0)
                    if current_best_val == 0.0:
                        metric_for_best = "eval_mAP50"
                        current_best_val = eval_entry.get(metric_for_best, 0.0)
                        
                    if current_best_val >= best_metric_val:
                        best_metric_val = current_best_val
                        best_epoch = step
                        
            # Compile trainer_state JSON structure
            trainer_state = {
                "best_global_step": best_epoch,
                "best_metric": float(best_metric_val),
                "best_model_checkpoint": f"outputs/det/{run_name}/{timestamp}/weights/best.pt",
                "epoch": float(best_epoch),
                "global_step": best_epoch,
                "log_history": log_history
            }
            
            # Write trainer_state.json to target timestamp directory
            state_file = target_dir / "trainer_state.json"
            with open(state_file, "w") as f:
                json.dump(trainer_state, f, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to parse or write state for {run_name}: {e}")
            continue

        # 2. Copy plots and images from source to target directory
        for item in src_run_dir.iterdir():
            if item.is_file() and item.suffix in [".png", ".jpg", ".yaml"] and item.name != "results.csv":
                try:
                    shutil.copy2(item, target_dir / item.name)
                except Exception as e:
                    print(f"⚠️ Failed to copy file {item.name}: {e}")

        # Also copy weights folder if it exists
        src_weights_dir = src_run_dir / "weights"
        if src_weights_dir.exists():
            try:
                shutil.copytree(src_weights_dir, target_dir / "weights", dirs_exist_ok=True)
            except Exception as e:
                print(f"⚠️ Failed to copy weights folder: {e}")

        migrated_count += 1

    print(f"🎉 Successfully migrated {migrated_count} runs.")

if __name__ == "__main__":
    main()
