#!/usr/bin/env python3
"""
Utility script to clean up unfinished/interrupted training run folders inside the outputs directory.
Deletes folders that do not have a DONE file or are missing weight files.
"""

import os
import shutil
import sys
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

def main():
    base_path = "outputs"
    if not os.path.exists(base_path):
        print(f"❌ Error: Path '{base_path}' does not exist.")
        sys.exit(1)

    print("🧹 Scanning outputs directory for unfinished runs...")
    to_delete = []
    completed_count = 0

    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        parts = [p for p in rel_path.split(os.sep) if p and p != "."]
        
        if any(p in ("log", "runs", ".ipynb_checkpoints") for p in parts):
            continue

        # Identify run directories by the timestamp format (e.g. 20260609_222030)
        timestamp = parts[-1] if parts else ""
        if not re.match(r"^\d{8}_\d{6}$", timestamp):
            continue

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

        # Check if training finished successfully and weights/trainer_state were saved correctly
        has_done = "DONE" in files
        is_valid = has_done and verify_weights(root, task) and verify_trainer_state(root)

        if not is_valid:
            to_delete.append(root)
        else:
            completed_count += 1

    print(f"📊 Completed/Valid Runs: {completed_count}")
    print(f"❌ Unfinished/Invalid Runs Found: {len(to_delete)}")

    if not to_delete:
        print("🎉 No unfinished runs to clean up!")
        return

    print("\n🛑 The following directories will be permanently DELETED:")
    for directory in sorted(to_delete):
        print(f"  - {directory}")

    print("\n🗑️ Deleting directories...")
    deleted_count = 0
    for directory in to_delete:
        try:
            shutil.rmtree(directory)
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Failed to delete {directory}: {e}")

    print(f"✅ Successfully cleaned up {deleted_count} unfinished runs.")

if __name__ == "__main__":
    main()
