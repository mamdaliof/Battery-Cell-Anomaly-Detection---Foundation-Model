#!/usr/bin/env python3
"""
Script to:
1. Recursively walk through a target dataset directory.
2. Find all XML files, replace '<name>abnormality</name>' with '<name>abnormal</name>' in-place.
3. Remove target classification folder.
4. Run scripts/convert_kfold_to_classification.py to regenerate classification splits.
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description="Fix annotation label abnormality -> abnormal and run conversion script.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/kfold_structured_dataset"),
        help="Path to structured dataset directory containing XMLs (default: data/kfold_structured_dataset)"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/kfold_classification"),
        help="Path to classification output directory (default: data/kfold_classification)"
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        default=True,
        help="Use symlinks for conversion (default: True)"
    )
    return parser.parse_args()

def fix_xml_file(filepath: Path) -> bool:
    modified = False
    try:
        # First try string replace to preserve formatting/indentation/comments
        content = filepath.read_text(encoding="utf-8")
        if "<name>abnormality</name>" in content:
            new_content = content.replace("<name>abnormality</name>", "<name>abnormal</name>")
            filepath.write_text(new_content, encoding="utf-8")
            modified = True
            
        # Parse XML to check if we missed any weird formatting of the name element
        tree = ET.parse(filepath)
        root = tree.getroot()
        xml_modified = False
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text is not None:
                if name_el.text.strip() == "abnormality":
                    name_el.text = "abnormal"
                    xml_modified = True
        
        if xml_modified:
            if not modified:
                # Format might have changed, write using standard tree write
                tree.write(filepath, encoding="utf-8", xml_declaration=True)
                modified = True
    except Exception as e:
        print(f"⚠️ Error processing XML file {filepath}: {e}")
    return modified

def main():
    args = parse_args()
    
    dataset_dir = args.dataset_dir.resolve()
    target_dir = args.target_dir.resolve()
    
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)
        
    print(f"📂 Scanning '{dataset_dir}' for XML files...")
    
    xml_files = list(dataset_dir.rglob("*.xml"))
    print(f"🔎 Found {len(xml_files)} XML files.")
    
    modified_count = 0
    for xml_file in xml_files:
        if fix_xml_file(xml_file):
            modified_count += 1
            print(f"  ✍️ Fixed label in: {xml_file.relative_to(dataset_dir.parent.parent)}")
            
    print(f"📊 Completed XML fix: Modified {modified_count} out of {len(xml_files)} files.")
    
    # Clean up the output classification folder if it exists
    if target_dir.exists():
        print(f"🧹 Removing old classification output folder '{target_dir}' to avoid stale normal class files...")
        shutil.rmtree(target_dir)
        
    # Re-run classification dataset conversion script
    convert_script = Path("scripts/convert_kfold_to_classification.py").resolve()
    if not convert_script.exists():
        print(f"❌ Error: Conversion script '{convert_script}' not found.")
        sys.exit(1)
        
    cmd = [
        sys.executable,
        str(convert_script),
        "--source-root", str(dataset_dir),
        "--target-root", str(target_dir),
        "--abnormal-labels", "abnormal",
        "--kfold"
    ]
    if args.use_symlinks:
        cmd.append("--use-symlinks")
        
    print(f"🚀 Running conversion script: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("\n--- stdout from convert_kfold_to_classification.py ---")
    print(result.stdout)
    
    if result.stderr:
        print("--- stderr from convert_kfold_to_classification.py ---")
        print(result.stderr)
        
    if result.returncode == 0:
        print("🎉 Successfully converted dataset!")
    else:
        print(f"❌ Failed to run conversion script (exit code {result.returncode})")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
