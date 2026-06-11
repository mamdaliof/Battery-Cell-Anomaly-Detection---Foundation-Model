#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
from PIL import Image, ImageDraw

# Add src/ to path so bcadfm can be imported
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bcadfm.models.dinov3_classifier import DinoV3Classifier, HeadConfig
from bcadfm.data.config import DataConfig
from bcadfm.data.dataset import BatteryCellDataset

def find_classification_weights(run_dir: Path) -> Path:
    # Check direct weights in run_dir
    for name in ["best_f1.pt", "best_loss.pt", "model.safetensors", "pytorch_model.bin"]:
        p = run_dir / name
        if p.exists():
            return p
            
    # Check in subdirectories / checkpoints
    checkpoint_dirs = sorted(run_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]), reverse=True)
    for ckpt in checkpoint_dirs:
        for name in ["model.safetensors", "pytorch_model.bin"]:
            p = ckpt / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find classification weights in {run_dir}")

def find_yolo_weights(run_dir: Path) -> Path:
    # Search weights/best.pt
    p = run_dir / "weights" / "best.pt"
    if p.exists():
        return p
    for best_pt in run_dir.rglob("best.pt"):
        return best_pt
    for pt in run_dir.rglob("*.pt"):
        if pt.name not in ["last.pt"]:
            return pt
    raise FileNotFoundError(f"Could not find YOLO weights in {run_dir}")

def main():
    parser = argparse.ArgumentParser(description="Perform validation set inference and save outputs")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory or weights file")
    parser.add_argument("--output_dir", type=str, default="visual_inference", help="Path to save predictions and ground truths")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # 1. Resolve paths and configs
    if model_path.is_file():
        weights_path = model_path
        run_dir = model_path.parent
        if run_dir.name.startswith("checkpoint-"):
            run_dir = run_dir.parent
    else:
        run_dir = model_path
        weights_path = None

    # Try to locate config.yaml
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        # Fallback to parents
        for parent in model_path.parents:
            config_path = parent / "config.yaml"
            if config_path.exists():
                run_dir = parent
                break
                
    if not config_path.exists():
        raise FileNotFoundError(f"Could not locate config.yaml for model {args.model}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Determine Task Type
    is_det = "yolo_model_config" in cfg
    
    # Create directories
    pred_dir = output_dir / "predictions"
    gt_dir = output_dir / "ground_truth"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Output Directories:\n   Predictions:  {pred_dir}\n   Ground Truth: {gt_dir}")

    if is_det:
        print("🔍 Detected Task: YOLO Object Detection")
        if weights_path is None:
            weights_path = find_yolo_weights(run_dir)
        print(f"🎯 Loading weights: {weights_path}")
        
        from ultralytics import YOLO
        import cv2
        
        model = YOLO(str(weights_path))
        
        yolo_data_yaml = run_dir / cfg.get("yolo_data_yaml", "data/det_v1.0/battery_detection_all.yaml")
        if not yolo_data_yaml.exists():
            # Fallback to root path
            yolo_data_yaml = Path(cfg.get("yolo_data_yaml", "data/det_v1.0/battery_detection_all.yaml")).resolve()
            
        with open(yolo_data_yaml, "r") as f:
            det_cfg = yaml.safe_load(f)
            
        dataset_base_path = Path(det_cfg["path"])
        val_images_rel = det_cfg["val"]
        val_images_dir = dataset_base_path / val_images_rel
        val_labels_dir = dataset_base_path / val_images_rel.replace("images", "labels")
        class_names = det_cfg.get("names", {0: "abnormal", 1: "cell", 2: "text"})
        
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        val_images = []
        for ext in image_extensions:
            val_images.extend(val_images_dir.glob(f"*{ext}"))
            
        print(f"🖼️ Found {len(val_images)} validation images for detection.")
        
        for img_path in val_images:
            # Predict
            results = model(str(img_path))
            plotted = results[0].plot() # Returns BGR numpy array
            
            # Save Prediction
            cv2.imwrite(str(pred_dir / f"{img_path.stem}_pred.jpg"), plotted)
            
            # Draw Ground Truth
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            label_path = val_labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, "r") as lf:
                    lines = lf.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        
                        img_w, img_h = img.size
                        x1 = (x_c - w / 2) * img_w
                        y1 = (y_c - h / 2) * img_h
                        x2 = (x_c + w / 2) * img_w
                        y2 = (y_c + h / 2) * img_h
                        
                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                        color = colors[cls_id % len(colors)]
                        
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        cls_name = class_names.get(cls_id, f"Class {cls_id}")
                        draw.text((x1 + 5, y1 + 5), cls_name, fill=color)
            
            # Save Ground Truth
            img.save(gt_dir / f"{img_path.stem}_gt.jpg")

    else:
        print("🔍 Detected Task: DINOv3 + PEFT Image Classification")
        if weights_path is None:
            weights_path = find_classification_weights(run_dir)
        print(f"🎯 Loading weights: {weights_path}")
        
        # Load classification configuration details
        head_cfg = HeadConfig(**cfg.get("head", {}))
        peft_cfg = cfg.get("peft", {})
        model_name = cfg.get("model_name")
        
        model = DinoV3Classifier(
            model_name_or_path=model_name,
            head_config=head_cfg,
            peft_config=peft_cfg,
            id2label={0: "normal", 1: "abnormal"},
            label2id={"normal": 0, "abnormal": 1}
        )
        
        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
        else:
            state_dict = torch.load(str(weights_path), map_location="cpu")
            
        model.load_state_dict(state_dict)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Instantiate dataset to leverage resizing/normalization logic
        data_cfg = DataConfig(**cfg["data"])
        val_dataset = BatteryCellDataset(
            split="val",
            data_config=data_cfg,
            model_name_or_path=model_name,
            image_size_override=cfg["data"].get("image_size", 224)
        )
        
        print(f"🖼️ Found {len(val_dataset)} validation images for classification.")
        
        for idx in range(len(val_dataset)):
            sample = val_dataset.samples[idx]
            img_path = sample.image_path
            gt_label = sample.label
            gt_class_name = val_dataset.id2label[gt_label]
            
            inputs = val_dataset[idx]
            pixel_values = inputs["pixel_values"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs["logits"]
                pred_idx = logits.argmax(dim=-1).item()
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[0, pred_idx].item()
                
            pred_class_name = val_dataset.id2label[pred_idx]
            
            # Draw Prediction
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            pred_color = (0, 255, 0) if pred_class_name == "normal" else (255, 0, 0)
            draw.text((10, 10), f"Pred: {pred_class_name} ({confidence*100:.1f}%)", fill=pred_color)
            img.save(pred_dir / f"{img_path.stem}_pred.jpg")
            
            # Draw Ground Truth
            img_gt = Image.open(img_path).convert("RGB")
            draw_gt = ImageDraw.Draw(img_gt)
            gt_color = (0, 255, 0) if gt_class_name == "normal" else (255, 0, 0)
            draw_gt.text((10, 10), f"GT: {gt_class_name}", fill=gt_color)
            img_gt.save(gt_dir / f"{img_path.stem}_gt.jpg")

    print("✅ Validation Inference and saving completed successfully!")

if __name__ == "__main__":
    main()
