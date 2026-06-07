#!/usr/bin/env python3
"""
Script to collect and package ablation training results.
It scans the 'outputs' directory, copies only 'config.yaml', 'DONE', and any '*.json' files
(maintaining the original subfolder structure) into a temporary folder, zips it,
and cleans up the temporary folder.
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

def package_results(output_dir: str = "outputs", zip_name: str = "ablation_results"):
    out_path = Path(output_dir)
    if not out_path.exists():
        print(f"❌ Error: Output directory '{output_dir}' does not exist.")
        return

    temp_dir = Path(f"{zip_name}_temp")
    
    # Clean up any leftover temp dir from previous runs
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Scanning '{out_path}' for results files...")
    
    copied_count = 0
    # Walk the outputs directory
    for root, dirs, files in os.walk(out_path):
        for file in files:
            # We only care about config.yaml, DONE, and *.json files (like trainer_state.json)
            if file in ("config.yaml", "DONE") or file.endswith(".json"):
                file_path = Path(root) / file
                # Compute relative path to maintain directory structure
                rel_path = file_path.relative_to(out_path)
                dest_path = temp_dir / rel_path
                
                # Create parent directories in the temp folder if they don't exist
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(file_path, dest_path)
                copied_count += 1

    if copied_count == 0:
        print("⚠️ No matching results files found (config.yaml, DONE, or *.json).")
        # Clean up temp dir and exit
        shutil.rmtree(temp_dir)
        return

    print(f"✅ Copied {copied_count} files to temporary directory '{temp_dir}'.")
    
    # Zip the temporary directory
    zip_file_path = Path(f"{zip_name}.zip")
    print(f"📦 Zipping temporary directory to '{zip_file_path}'...")
    
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                # Archive path should be relative to the temp_dir so the zip doesn't have the temp folder name inside it
                archive_name = file_path.relative_to(temp_dir)
                zipf.write(file_path, archive_name)

    print("🧹 Cleaning up temporary directory...")
    shutil.rmtree(temp_dir)
    
    print(f"🎉 Packaging complete! Created: {zip_file_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package ablation training results config and metrics files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to the outputs directory containing run results (default: outputs)"
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="ablation_results",
        help="Name of the output zip file (without extension, default: ablation_results)"
    )
    args = parser.parse_args()
    
    package_results(output_dir=args.output_dir, zip_name=args.zip_name)
