#!/usr/bin/env python3
"""
Utility script to clean up unfinished/interrupted training run folders inside the outputs directory.
Deletes folders that do not have a DONE file or are missing weight files.
"""

import os
import shutil
import sys

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
        is_valid = has_done and verify_weights(root, task)

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
