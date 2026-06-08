#!/usr/bin/env python

"""Convert detection-style XML annotations into YOLO-compatible dataset variants.

Source layout (depth 1):

    split_base/
      train/
        c44_5.png
        c44_5.xml
        ...
      val/
        ...

Target layout (for Ultralytics YOLO variants under target_root):

    target_root/
      battery_detection_{variant}/
        images/
          train/
          val/
        labels/
          train/
          val/
      battery_detection_{variant}.yaml

Each XML file is parsed to extract target bounding boxes depending on the selected
variant, and converted to the normalized YOLO format: `class_idx x_center y_center width height`.

Variants:
1. 'all': keeps cell, text, abnormality -> class indices: abnormality=0, cell=1, text=2
2. 'no_cell': drops cell, keeps text and abnormality -> class indices: abnormality=0, text=1
3. 'only_cell': keeps only cell -> class indices: cell=0

By default, image files are COPIED. You can optionally use symlinks instead.
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set


# Define target labels and mapping indices for each variant
VARIANTS = {
    "all": {
        "labels": {"abnormality", "cell", "text"},
        "mapping": {"abnormality": 0, "cell": 1, "text": 2}
    },
    "no_cell": {
        "labels": {"abnormality", "text"},
        "mapping": {"abnormality": 0, "text": 1}
    },
    "only_cell": {
        "labels": {"cell"},
        "mapping": {"cell": 0}
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert detection-style XML annotations to YOLO detection variants.",
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
        help="Path where YOLO formatted variant dirs and YAML configs will be written (data/)",
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
            print(f"⚠️ [WARN] XML not found for image: {img_path}")
            continue
        pairs.append((img_path, xml_path))
    return pairs


def extract_yolo_bboxes_from_xml(
    xml_path: Path,
    target_labels: Set[str],
    label_to_idx: Dict[str, int]
) -> List[tuple[int, float, float, float, float]]:
    """Parse XML and extract normalized YOLO bounding boxes for target labels.

    YOLO format: (class_idx, x_center, y_center, width, height) normalized.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"⚠️ [WARN] Failed to parse XML {xml_path}: {e}")
        return []

    # Get width and height of the image
    size_node = root.find("size")
    if size_node is None:
        print(f"⚠️ [WARN] Missing size node in XML {xml_path}")
        return []
    
    width_el = size_node.find("width")
    height_el = size_node.find("height")
    if width_el is None or height_el is None or not width_el.text or not height_el.text:
        print(f"⚠️ [WARN] Invalid width/height in XML {xml_path}")
        return []
        
    width = float(width_el.text)
    height = float(height_el.text)
    if width <= 0 or height <= 0:
        print(f"⚠️ [WARN] Width/height <= 0 in XML {xml_path}")
        return []

    yolo_boxes: List[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is not None and name_el.text is not None:
            name = name_el.text.strip()
            if name in target_labels:
                class_idx = label_to_idx[name]
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                # Clip coordinates to image boundary
                xmin = max(0.0, min(xmin, width))
                ymin = max(0.0, min(ymin, height))
                xmax = max(0.0, min(xmax, width))
                ymax = max(0.0, min(ymax, height))
                
                if xmax <= xmin or ymax <= ymin:
                    print(f"⚠️ [WARN] Invalid bbox dimensions in {xml_path}")
                    continue

                # Compute normalized YOLO coordinates
                dw = 1.0 / width
                dh = 1.0 / height
                x_center = (xmin + xmax) / 2.0 * dw
                y_center = (ymin + ymax) / 2.0 * dh
                w = (xmax - xmin) * dw
                h = (ymax - ymin) * dh
                
                # Final check to guarantee range [0, 1]
                if 0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                    yolo_boxes.append((class_idx, x_center, y_center, w, h))
                else:
                    print(f"⚠️ [WARN] Bbox out of range in {xml_path}: {(x_center, y_center, w, h)}")
                    
    return yolo_boxes


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


def convert_split_variant(
    split_name: str,
    source_root: Path,
    target_root: Path,
    variant_name: str,
    use_symlinks: bool,
) -> None:
    split_src = source_root / split_name
    if not split_src.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_src}")

    variant_info = VARIANTS[variant_name]
    target_labels = variant_info["labels"]
    label_to_idx = variant_info["mapping"]

    print(f"ℹ️ [INFO] Processing split '{split_name}' for variant '{variant_name}'")

    variant_dir = target_root / f"battery_detection_{variant_name}"
    target_images_dir = variant_dir / "images" / split_name
    target_labels_dir = variant_dir / "labels" / split_name
    
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_image_xml_pairs(split_src)
    print(f"ℹ️ [INFO] Found {len(pairs)} image-XML pairs in {split_src}")

    image_with_detection_count = 0
    total_boxes = 0

    for img_path, xml_path in pairs:
        # Process and write boxes
        boxes = extract_yolo_bboxes_from_xml(xml_path, target_labels, label_to_idx)
        
        # Copy or link image
        dst_image_path = target_images_dir / img_path.name
        copy_or_link(img_path, dst_image_path, use_symlinks)
        
        # Write labels file (empty if no detections)
        dst_label_path = target_labels_dir / img_path.with_suffix(".txt").name
        
        lines = []
        for box in boxes:
            class_idx, xc, yc, w, h = box
            lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        with open(dst_label_path, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
                image_with_detection_count += 1
                total_boxes += len(lines)

    print(f"✅ [INFO] Finished split '{split_name}' (variant '{variant_name}')")
    print(f"   - Images with targets: {image_with_detection_count}")
    print(f"   - Total bounding boxes: {total_boxes}")


def write_dataset_yaml(target_root: Path, variant_name: str) -> None:
    variant_info = VARIANTS[variant_name]
    label_to_idx = variant_info["mapping"]
    
    # Sort class names by index
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    names_dict = {idx: idx_to_label[idx] for idx in sorted(idx_to_label.keys())}

    yaml_path = target_root / f"battery_detection_{variant_name}.yaml"
    variant_abs_path = target_root.resolve() / f"battery_detection_{variant_name}"
    
    lines = [
        f"# Battery Cell Anomaly Detection Dataset Config - Variant '{variant_name}'",
        f"# Generated automatically for YOLO26 + DINOv3 SFP object detection model",
        "",
        f"path: {variant_abs_path}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(names_dict)}",
        "names:",
    ]
    for idx, name in names_dict.items():
        lines.append(f"  {idx}: {name}")
    
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"✅ [INFO] Wrote configuration YAML to: {yaml_path}")


def main() -> None:
    args = parse_args()
    source_root: Path = args.source_root
    target_root: Path = args.target_root

    print(f"ℹ️ [INFO] Source root: {source_root}")
    print(f"ℹ️ [INFO] Target root: {target_root}")
    print(f"ℹ️ [INFO] Target variants: {list(VARIANTS.keys())}")
    print(f"ℹ️ [INFO] Using symlinks: {args.use_symlinks}")

    # Generate each variant
    for variant in VARIANTS.keys():
        variant_dir = target_root / f"battery_detection_{variant}"
        if variant_dir.exists():
            print(f"ℹ️ [INFO] Cleaning existing folder for variant '{variant}'...")
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "val"):
            convert_split_variant(
                split_name=split_name,
                source_root=source_root,
                target_root=target_root,
                variant_name=variant,
                use_symlinks=bool(args.use_symlinks),
            )
            
        # Write config YAML file for this variant
        write_dataset_yaml(target_root, variant)

    print("🎉 [INFO] Conversion completed for all variants.")


if __name__ == "__main__":
    main()
