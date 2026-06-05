#!/usr/bin/env python

"""Convert detection-style XML annotations into a classification dataset.

Source layout (depth 1):

    split_base/
      train/
        c44_5.png
        c44_5.xml
        ...
      val/
        ...

Each XML file is assumed to contain one or more annotated objects. If any
object label is in the set of "abnormal" labels, the corresponding image is
classified as abnormal; otherwise it is classified as normal.

Target layout (for BatteryCellDataset):

    target_root/
      train/
        normal/
        abnormal/
      val/
        normal/
        abnormal/

By default, files are COPIED. You can optionally use symlinks instead.
"""

from __future__ import annotations

import argparse
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
        help="Path to split_base root containing train/ and val/",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Path where classification data will be written (data/)",
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
    return parser.parse_args()


def find_image_xml_pairs(split_dir: Path) -> List[tuple[Path, Path]]:
    """Return (image_path, xml_path) pairs for all .png images in split_dir.

    Assumes that for each image `name.png` there is a corresponding `name.xml`.
    """

    pairs: List[tuple[Path, Path]] = []
    for img_path in split_dir.glob("*.png"):
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            # You can choose to warn or skip strictly
            print(f"[WARN] XML not found for image: {img_path}")
            continue
        pairs.append((img_path, xml_path))
    return pairs


def extract_labels_from_xml(xml_path: Path) -> List[str]:
    """Parse XML and extract object labels.

    This assumes a standard VOC-like structure where object names are under
    `object/name`. Adjust if your schema differs.
    """

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[WARN] Failed to parse XML {xml_path}: {e}")
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
        rel_src = src.relative_to(dst.parent)
        dst.symlink_to(rel_src)
    else:
        shutil.copy2(src, dst)


def convert_split(
    split_name: str,
    source_root: Path,
    target_root: Path,
    abnormal_labels: Set[str],
    use_symlinks: bool,
) -> None:
    split_src = source_root / split_name
    if not split_src.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_src}")

    print(f"[INFO] Processing split: {split_name}")

    pairs = find_image_xml_pairs(split_src)
    print(f"[INFO] Found {len(pairs)} image-XML pairs in {split_src}")

    for img_path, xml_path in pairs:
        labels = extract_labels_from_xml(xml_path)
        abnormal = is_abnormal(labels, abnormal_labels)
        label_str = "abnormal" if abnormal else "normal"

        dst_dir = target_root / split_name / label_str
        dst_path = dst_dir / img_path.name

        copy_or_link(img_path, dst_path, use_symlinks)

    print(f"[INFO] Finished split: {split_name}")


def main() -> None:
    args = parse_args()
    source_root: Path = args.source_root
    target_root: Path = args.target_root
    abnormal_labels: Set[str] = {lbl.strip() for lbl in args.abnormal_labels}

    print(f"[INFO] Source root: {source_root}")
    print(f"[INFO] Target root: {target_root}")
    print(f"[INFO] Abnormal labels: {sorted(abnormal_labels)}")
    print(f"[INFO] Using symlinks: {args.use_symlinks}")

    for split_name in ("train", "val"):
        convert_split(
            split_name=split_name,
            source_root=source_root,
            target_root=target_root,
            abnormal_labels=abnormal_labels,
            use_symlinks=bool(args.use_symlinks),
        )

    print("[INFO] Conversion completed.")


if __name__ == "__main__":
    main()
