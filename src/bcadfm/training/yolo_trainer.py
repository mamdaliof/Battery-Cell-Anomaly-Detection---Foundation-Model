from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import ultralytics.nn.tasks
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou


class CustomDetectionValidator(DetectionValidator):
    """Custom YOLO Validation class that calculates advanced metrics.

    Calculates matched box-level IoU and Dice scores, per-class metrics
    (Recall, Precision, TP, FP, FN, F1-score, mAP50, mAP50-95), and maps bboxes
    to image-level multi-label classification metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bbox IoU and Dice collectors
        self.custom_iou_dice_stats: List[tuple[float, float]] = []
        
        # Image-level abnormality classification collectors
        self.cls_gt_abnormality: List[int] = []
        self.cls_pred_abnormality: List[int] = []
        self.cls_prob_abnormality: List[float] = []

        # Image-level text classification collectors
        self.cls_gt_text: List[int] = []
        self.cls_pred_text: List[int] = []
        self.cls_prob_text: List[float] = []

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Accumulates standard metrics and extracts custom verification metrics."""
        # 1. Update standard Ultralytics metrics
        super().update_metrics(preds, batch)

        # Class names to indices map
        abnormality_idx = None
        text_idx = None
        for idx, name in self.names.items():
            if name == "abnormality":
                abnormality_idx = idx
            elif name == "text":
                text_idx = idx

        # 2. Extract matched box stats and classification labels
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            gt_classes = pbatch["cls"].cpu().numpy().astype(int) if pbatch["cls"].shape[0] > 0 else np.zeros(0)
            pred_classes = predn["cls"].cpu().numpy().astype(int) if predn["cls"].shape[0] > 0 else np.zeros(0)
            pred_confs = predn["conf"].cpu().numpy() if predn["cls"].shape[0] > 0 else np.zeros(0)

            # --- Image-level Multi-Label Classification Labels ---
            gt_has_abn = int(abnormality_idx in gt_classes) if abnormality_idx is not None else 0
            gt_has_txt = int(text_idx in gt_classes) if text_idx is not None else 0

            # Find predictions above decision threshold (standard 0.25 confidence)
            pred_has_abn = 0
            prob_abn = 0.0
            pred_has_txt = 0
            prob_txt = 0.0

            for c_idx, conf in zip(pred_classes, pred_confs):
                if abnormality_idx is not None and c_idx == abnormality_idx:
                    prob_abn = max(prob_abn, float(conf))
                    if conf >= 0.25:
                        pred_has_abn = 1
                elif text_idx is not None and c_idx == text_idx:
                    prob_txt = max(prob_txt, float(conf))
                    if conf >= 0.25:
                        pred_has_txt = 1

            if abnormality_idx is not None:
                self.cls_gt_abnormality.append(gt_has_abn)
                self.cls_pred_abnormality.append(pred_has_abn)
                self.cls_prob_abnormality.append(prob_abn)

            if text_idx is not None:
                self.cls_gt_text.append(gt_has_txt)
                self.cls_pred_text.append(pred_has_txt)
                self.cls_prob_text.append(prob_txt)

            # --- Bounding Box-level matched IoU and Dice ---
            if pbatch["bboxes"].shape[0] > 0 and predn["bboxes"].shape[0] > 0:
                # Calculate pairwise IoU matrix between gt and pred boxes
                iou_matrix = box_iou(pbatch["bboxes"], predn["bboxes"]).cpu().numpy()
                matched_pred_indices = set()

                # Greedy matching per ground truth box
                for gt_i in range(pbatch["bboxes"].shape[0]):
                    gt_c = gt_classes[gt_i]
                    best_iou = -1.0
                    best_pred_idx = -1

                    for pred_i in range(predn["bboxes"].shape[0]):
                        if pred_i in matched_pred_indices:
                            continue
                        if pred_classes[pred_i] != gt_c:
                            continue
                        
                        curr_iou = iou_matrix[gt_i, pred_i]
                        if curr_iou > best_iou:
                            best_iou = curr_iou
                            best_pred_idx = pred_i

                    # Record matched boxes if overlap matches threshold
                    if best_pred_idx != -1 and best_iou >= 0.50:
                        matched_pred_indices.add(best_pred_idx)
                        dice = (2.0 * best_iou) / (1.0 + best_iou)
                        self.custom_iou_dice_stats.append((best_iou, dice))

    def get_stats(self) -> Dict[str, Any]:
        """Calculates and returns custom stats injected alongside YOLO stats."""
        stats = {}
        try:
            stats = super().get_stats()
        except Exception as e:
            print(f"⚠️ [WARN] Failed to calculate standard YOLO metrics (possibly zero detections): {e}")
            stats = {
                "metrics/precision(B)": 0.0,
                "metrics/recall(B)": 0.0,
                "metrics/mAP50(B)": 0.0,
                "metrics/mAP50-95(B)": 0.0,
            }

        names_list = sorted(self.names.keys())

        # 1. Bbox class-specific TP, FP, FN, Precision, Recall, F1, maps
        try:
            if hasattr(self.metrics, "stats") and self.metrics.stats:
                concatenated_stats = {}
                for k, v in self.metrics.stats.items():
                    if v and len(v) > 0:
                        concatenated_stats[k] = np.concatenate(v, 0)
                
                if concatenated_stats and "tp" in concatenated_stats and concatenated_stats["tp"].size > 0:
                    tp_iou_05 = concatenated_stats["tp"][:, 0]
                    pred_cls = concatenated_stats["pred_cls"]
                    target_cls = concatenated_stats["target_cls"]

                    for idx in names_list:
                        name = self.names[idx]
                        tp_c = int(np.sum(tp_iou_05 & (pred_cls == idx)))
                        fp_c = int(np.sum((~tp_iou_05) & (pred_cls == idx)))
                        fn_c = int(np.sum(target_cls == idx)) - tp_c

                        stats[f"metrics/custom_TP/{name}"] = tp_c
                        stats[f"metrics/custom_FP/{name}"] = fp_c
                        stats[f"metrics/custom_FN/{name}"] = fn_c

                        # Map metrics from ap_class_index mapping
                        results_idx = np.where(self.metrics.ap_class_index == idx)[0]
                        if len(results_idx) > 0:
                            r_idx = results_idx[0]
                            stats[f"metrics/custom_P/{name}"] = float(self.metrics.box.p[r_idx])
                            stats[f"metrics/custom_R/{name}"] = float(self.metrics.box.r[r_idx])
                            stats[f"metrics/custom_F1/{name}"] = float(self.metrics.box.f1[r_idx])
                            stats[f"metrics/custom_mAP50/{name}"] = float(self.metrics.box.all_ap[r_idx, 0])
                            stats[f"metrics/custom_mAP50-95/{name}"] = float(self.metrics.box.all_ap[r_idx].mean())
        except Exception as e:
            print(f"⚠️ [WARN] Error calculating per-class stats: {e}")

        # 2. Matched bbox IoU and Dice averages
        try:
            if self.custom_iou_dice_stats:
                ious = [x[0] for x in self.custom_iou_dice_stats]
                dices = [x[1] for x in self.custom_iou_dice_stats]
                stats["metrics/custom_mean_bbox_IoU"] = float(np.mean(ious))
                stats["metrics/custom_mean_bbox_Dice"] = float(np.mean(dices))
            else:
                stats["metrics/custom_mean_bbox_IoU"] = 0.0
                stats["metrics/custom_mean_bbox_Dice"] = 0.0
        except Exception as e:
            stats["metrics/custom_mean_bbox_IoU"] = 0.0
            stats["metrics/custom_mean_bbox_Dice"] = 0.0

        # 3. Image-level Multi-Label Classification Metrics
        try:
            for cls_name, gt, pred, prob in [
                ("abnormality", self.cls_gt_abnormality, self.cls_pred_abnormality, self.cls_prob_abnormality),
                ("text", self.cls_gt_text, self.cls_pred_text, self.cls_prob_text)
            ]:
                if gt:
                    gt_arr = np.array(gt)
                    pred_arr = np.array(pred)
                    prob_arr = np.array(prob)
                    stats[f"metrics/custom_cls_accuracy/{cls_name}"] = float(accuracy_score(gt_arr, pred_arr))
                    stats[f"metrics/custom_cls_precision/{cls_name}"] = float(precision_score(gt_arr, pred_arr, zero_division=0))
                    stats[f"metrics/custom_cls_recall/{cls_name}"] = float(recall_score(gt_arr, pred_arr, zero_division=0))
                    stats[f"metrics/custom_cls_f1/{cls_name}"] = float(f1_score(gt_arr, pred_arr, zero_division=0))
                    
                    if len(np.unique(gt_arr)) > 1:
                        stats[f"metrics/custom_cls_auroc/{cls_name}"] = float(roc_auc_score(gt_arr, prob_arr))
                    else:
                        stats[f"metrics/custom_cls_auroc/{cls_name}"] = 0.5
        except Exception as e:
            print(f"⚠️ [WARN] Error calculating classification metrics: {e}")

        return stats

    def finalize_metrics(self) -> None:
        """Sets final standard metrics and prints a clean validation summary table."""
        try:
            super().finalize_metrics()
        except Exception as e:
            print(f"⚠️ [WARN] Failed to finalize standard YOLO metrics: {e}")
        
        # Display the custom stats summary in terminal
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("📊 CUSTOM VALIDATION METRICS REPORT")
        print("=" * 80)
        
        # Matched box IoU/Dice
        print(f"📐 Matched Bbox IoU:   {stats.get('metrics/custom_mean_bbox_IoU', 0.0):.4f}")
        print(f"📐 Matched Bbox Dice:  {stats.get('metrics/custom_mean_bbox_Dice', 0.0):.4f}")
        
        # Per-class metrics
        print("\n📦 Bbox Metrics Per-Class (IoU=0.50):")
        print(f"{'Class':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'mAP50':<6}")
        print("-" * 80)
        for idx, name in self.names.items():
            tp = stats.get(f"metrics/custom_TP/{name}", 0)
            fp = stats.get(f"metrics/custom_FP/{name}", 0)
            fn = stats.get(f"metrics/custom_FN/{name}", 0)
            p = stats.get(f"metrics/custom_P/{name}", 0.0)
            r = stats.get(f"metrics/custom_R/{name}", 0.0)
            f1 = stats.get(f"metrics/custom_F1/{name}", 0.0)
            map50 = stats.get(f"metrics/custom_mAP50/{name}", 0.0)
            print(f"{name:<15} | {tp:<5} | {fp:<5} | {fn:<5} | {p:.3f}  | {r:.3f}  | {f1:.3f}  | {map50:.3f}")

        # Classification metrics
        print("\n🖥️ Image-Level Multi-Label Classification Conversion:")
        print(f"{'Class Indicator':<18} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'AUROC':<6}")
        print("-" * 80)
        for cls_name in ["abnormality", "text"]:
            acc = stats.get(f"metrics/custom_cls_accuracy/{cls_name}")
            if acc is not None:
                p = stats.get(f"metrics/custom_cls_precision/{cls_name}", 0.0)
                r = stats.get(f"metrics/custom_cls_recall/{cls_name}", 0.0)
                f1 = stats.get(f"metrics/custom_cls_f1/{cls_name}", 0.0)
                auc = stats.get(f"metrics/custom_cls_auroc/{cls_name}", 0.5)
                print(f"{cls_name:<18} | {acc:.3f}    | {p:.3f}     | {r:.3f}   | {f1:.3f} | {auc:.3f}")
        print("=" * 80 + "\n")


class CustomDetectionTrainer(DetectionTrainer):
    """Custom YOLO Trainer class that registers CustomDetectionValidator."""

    def get_validator(self) -> CustomDetectionValidator:
        """Returns the custom validator instance."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomDetectionValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)
