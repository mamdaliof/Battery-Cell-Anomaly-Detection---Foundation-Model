#!/usr/bin/env python3
"""Statistical analysis script for K-Fold datasets.

Analyzes image-wise (normal vs abnormal) and cell-wise/box-wise (dynamic classes)
distributions per fold and split (train/val), and saves them to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze K-Fold dataset distributions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/kfold_structured_dataset"),
        help="Path to kfold structured dataset containing fold_* directories",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/kfold_structured_dataset/kfold_stats.csv"),
        help="Path to save stats as CSV",
    )
    return parser.parse_args()


def get_xml_classes(xml_path: Path) -> List[str]:
    """Extract object names from a Pascal VOC XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError):
        return []

    classes = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is not None and name_el.text is not None:
            classes.append(name_el.text.strip())
    return classes


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    output_csv: Path = args.output_csv

    if not data_dir.exists():
        print(f"❌ Error: Data directory {data_dir} does not exist.")
        return

    # Find all fold directories
    fold_dirs = sorted([d for d in data_dir.glob("fold_*") if d.is_dir()])
    if not fold_dirs:
        print(f"❌ Error: No fold_* directories found in {data_dir}.")
        return

    print(f"ℹ️ Found {len(fold_dirs)} folds in {data_dir}")

    # Gather stats
    # Structure: stats[fold_name][split][metric_type][class_name] = count
    stats: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(int)
            )
        )
    )

    all_box_classes: Set[str] = set()

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        for split in ("train", "val"):
            split_dir = fold_dir / split
            if not split_dir.exists():
                continue

            # Image-wise classification stats are based on folder names
            for label in ("normal", "abnormality"):
                class_dir = split_dir / label
                if not class_dir.exists():
                    continue

                # Count unique image files
                images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                stats[fold_name][split]["image_count"][label] = len(images)

                # Parse XMLs in the directory to get box-wise distributions
                for xml_path in class_dir.glob("*.xml"):
                    classes = get_xml_classes(xml_path)
                    for cls in classes:
                        stats[fold_name][split]["box_count"][cls] += 1
                        all_box_classes.add(cls)

    # Let's aggregate overall/entire dataset stats
    overall: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(int)
        )
    )
    for fold_name, split_dict in stats.items():
        for split, metric_dict in split_dict.items():
            for metric_type, class_dict in metric_dict.items():
                for class_name, count in class_dict.items():
                    overall[split][metric_type][class_name] += count

    # Write stats to CSV
    # Headers: fold, split, metric_type, class_name, count
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fold", "split", "metric_type", "class_name", "count"])

        # Write fold-specific stats
        for fold_name in sorted(stats.keys()):
            for split in ("train", "val"):
                for metric_type in ("image_count", "box_count"):
                    class_counts = stats[fold_name][split][metric_type]
                    for class_name, count in sorted(class_counts.items()):
                        writer.writerow([fold_name, split, metric_type, class_name, count])

        # Write overall stats
        for split in ("train", "val"):
            for metric_type in ("image_count", "box_count"):
                class_counts = overall[split][metric_type]
                for class_name, count in sorted(class_counts.items()):
                    writer.writerow(["overall", split, metric_type, class_name, count])

    print(f"✅ Statistics saved to {output_csv}")

    # Print a beautiful table to the console
    print("\n📊 --- K-FOLD CLASSIFICATION SUMMARY (IMAGE-WISE) ---")
    print(f"{'Fold':<10} | {'Split':<6} | {'Normal':<10} | {'Abnormal':<10} | {'Total':<10}")
    print("-" * 55)

    def print_row(fold: str, split: str, norm: int, abnorm: int) -> None:
        tot = norm + abnorm
        print(f"{fold:<10} | {split:<6} | {norm:<10} | {abnorm:<10} | {tot:<10}")

    for fold_name in sorted(stats.keys()):
        for split in ("train", "val"):
            norm = stats[fold_name][split]["image_count"]["normal"]
            abnorm = stats[fold_name][split]["image_count"]["abnormality"]
            print_row(fold_name, split, norm, abnorm)
    print("-" * 55)
    for split in ("train", "val"):
        norm = overall[split]["image_count"]["normal"]
        abnorm = overall[split]["image_count"]["abnormality"]
        print_row("overall", split, norm, abnorm)

    print("\n📦 --- K-FOLD DETECTION SUMMARY (BOX-WISE) ---")
    sorted_box_classes = sorted(list(all_box_classes))
    headers = f"{'Fold':<10} | {'Split':<6} | " + " | ".join(f"{cls.capitalize():<12}" for cls in sorted_box_classes)
    print(headers)
    print("-" * len(headers))

    for fold_name in sorted(stats.keys()):
        for split in ("train", "val"):
            row_str = f"{fold_name:<10} | {split:<6} | "
            counts = [stats[fold_name][split]["box_count"][cls] for cls in sorted_box_classes]
            row_str += " | ".join(f"{c:<12}" for c in counts)
            print(row_str)
    print("-" * len(headers))
    for split in ("train", "val"):
        row_str = f"{'overall':<10} | {split:<6} | "
        counts = [overall[split]["box_count"][cls] for cls in sorted_box_classes]
        row_str += " | ".join(f"{c:<12}" for c in counts)
        print(row_str)
    print()


if __name__ == "__main__":
    main()
