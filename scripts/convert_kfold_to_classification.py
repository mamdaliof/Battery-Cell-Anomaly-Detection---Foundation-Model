#!/usr/bin/env python3

"""Convert detection-style XML annotations into a classification dataset.

Supports both single split directories and multi-fold cross-validation directories.
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert detection-style XML annotations to classification folders.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to source root containing train/val or fold_* directories",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Path where classification data will be written (e.g., data/kfold_classification)",
    )
    parser.add_argument(
        "--abnormal-labels",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of XML object labels that should be treated as abnormal. "
            "If any of these labels appear in an image's XML, the image is marked abnormal."
        ),
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        help="If set, create symlinks instead of copying image files.",
    )
    parser.add_argument(
        "--kfold",
        action="store_true",
        help="If set, loops through all fold_* directories in the source root.",
    )
    return parser.parse_args()


def find_image_xml_pairs(split_dir: Path) -> List[tuple[Path, Path]]:
    """Return (image_path, xml_path) pairs for all .png images in split_dir.

    Assumes that for each image `name.png` there is a corresponding `name.xml`.
    """
    pairs: List[tuple[Path, Path]] = []
    for img_path in split_dir.glob("*.png"):
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            print(f"⚠️ [WARN] XML not found for image: {img_path}")
            continue
        pairs.append((img_path, xml_path))
    return pairs


def extract_labels_from_xml(xml_path: Path) -> List[str]:
    """Parse XML and extract object labels."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"⚠️ [WARN] Failed to parse XML {xml_path}: {e}")
        return []

    labels: List[str] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is not None and name_el.text is not None:
            labels.append(name_el.text.strip())
    return labels


def is_abnormal(labels: Iterable[str], abnormal_set: Set[str]) -> bool:
    return any(lbl in abnormal_set for lbl in labels)


def copy_or_link(src: Path, dst: Path, use_symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if use_symlinks:
        if dst.exists():
            dst.unlink()
        # Create relative symlink for portability
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)
    else:
        shutil.copy2(src, dst)


def convert_split(
    split_name: str,
    source_dir: Path,
    target_dir: Path,
    abnormal_labels: Set[str],
    use_symlinks: bool,
) -> None:
    # Under source_dir we can have standard splits like train/val,
    # or the abnormal/normal subdirectory structure directly!
    # In data/kfold_structured_dataset/fold_0/train/, images are in 'abnormal' and 'normal' subfolders.
    # So we must search recursively or check subdirectories.
    print(f"ℹ️ [INFO] Processing split: {split_name} in {source_dir}")

    # Check if files are directly in split_dir or in subfolders (abnormal/normal)
    pairs = []
    if (source_dir / "normal").exists() or (source_dir / "abnormal").exists():
        for sub in ("normal", "abnormal"):
            sub_dir = source_dir / sub
            if sub_dir.exists():
                pairs.extend(find_image_xml_pairs(sub_dir))
    else:
        pairs.extend(find_image_xml_pairs(source_dir))

    print(f"ℹ️ [INFO] Found {len(pairs)} image-XML pairs in {source_dir}")

    for img_path, xml_path in pairs:
        labels = extract_labels_from_xml(xml_path)
        abnormal = is_abnormal(labels, abnormal_labels)
        label_str = "abnormal" if abnormal else "normal"

        dst_dir = target_dir / split_name / label_str
        dst_path = dst_dir / img_path.name

        copy_or_link(img_path, dst_path, use_symlinks)

    print(f"✅ [INFO] Finished split: {split_name}")


def main() -> None:
    args = parse_args()
    source_root: Path = args.source_root
    target_root: Path = args.target_root
    abnormal_labels: Set[str] = {lbl.strip() for lbl in args.abnormal_labels}

    print(f"ℹ️ [INFO] Source root: {source_root}")
    print(f"ℹ️ [INFO] Target root: {target_root}")
    print(f"ℹ️ [INFO] Abnormal labels: {sorted(abnormal_labels)}")
    print(f"ℹ️ [INFO] Using symlinks: {args.use_symlinks}")
    print(f"ℹ️ [INFO] K-Fold mode: {args.kfold}")

    if args.kfold:
        fold_dirs = sorted([d for d in source_root.glob("fold_*") if d.is_dir()])
        if not fold_dirs:
            print(f"❌ [ERROR] No fold_* directories found in {source_root}")
            return
        
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            print(f"\n📂 Processing fold: {fold_name}")
            for split_name in ("train", "val"):
                convert_split(
                    split_name=split_name,
                    source_dir=fold_dir / split_name,
                    target_dir=target_root / fold_name,
                    abnormal_labels=abnormal_labels,
                    use_symlinks=bool(args.use_symlinks),
                )
    else:
        for split_name in ("train", "val"):
            convert_split(
                split_name=split_name,
                source_dir=source_root / split_name,
                target_dir=target_root,
                abnormal_labels=abnormal_labels,
                use_symlinks=bool(args.use_symlinks),
            )

    print("\n🎉 [INFO] Classification dataset conversion completed.")


if __name__ == "__main__":
    main()
