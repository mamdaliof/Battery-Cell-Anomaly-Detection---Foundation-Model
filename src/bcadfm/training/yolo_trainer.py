from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import ultralytics.nn.tasks
from ultralytics import settings
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou

# Force Ultralytics to use outputs as the default run directory
try:
    settings.update({"runs_dir": "outputs"})
except Exception as e:
    print(f"⚠️ [WARN] Failed to set Ultralytics runs_dir setting: {e}")


from typing import Any, Dict, List, Union

def ddp_gather_list(data_list: List[Any]) -> List[Any]:
    """Gather lists of arbitrary picklable objects from all DDP ranks to Rank 0."""
    if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
        return data_list
    try:
        gathered_data = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_data, data_list)
        flat_list = []
        for sublist in gathered_data:
            if sublist is not None:
                flat_list.extend(sublist)
        return flat_list
    except Exception as e:
        print(f"⚠️ [WARN] Failed to gather data across DDP ranks: {e}")
        return data_list


def ddp_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a 1D tensor from all DDP ranks to Rank 0, concatenating them."""
    if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
        return tensor
    try:
        world_size = torch.distributed.get_world_size()
        local_size = torch.tensor([tensor.numel()], dtype=torch.long, device=tensor.device)
        sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
        torch.distributed.all_gather(sizes, local_size)
        
        max_size = max(int(s.item()) for s in sizes)
        # Pad tensor if needed to make sizes equal for all_gather
        padded = tensor
        if tensor.numel() < max_size:
            padded = torch.cat([tensor, torch.zeros(max_size - tensor.numel(), dtype=tensor.dtype, device=tensor.device)])
        
        gathered_tensors = [torch.zeros(max_size, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, padded)
        
        # Unpad and concatenate
        results = []
        for i, s in enumerate(sizes):
            size_val = int(s.item())
            results.append(gathered_tensors[i][:size_val])
        return torch.cat(results)
    except Exception as e:
        print(f"⚠️ [WARN] Failed to gather tensor across DDP ranks: {e}")
        return tensor


def ddp_gather_tensor_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a 2D tensor of shape (N, D) from all DDP ranks to Rank 0."""
    if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
        return tensor
    try:
        world_size = torch.distributed.get_world_size()
        local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
        sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
        torch.distributed.all_gather(sizes, local_size)
        
        max_rows = max(int(s.item()) for s in sizes)
        D = tensor.shape[1]
        
        # Pad tensor if needed to make sizes equal
        padded = tensor
        if tensor.shape[0] < max_rows:
            padding = torch.zeros(max_rows - tensor.shape[0], D, dtype=tensor.dtype, device=tensor.device)
            padded = torch.cat([tensor, padding], dim=0)
            
        gathered_tensors = [torch.zeros(max_rows, D, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, padded)
        
        results = []
        for i, s in enumerate(sizes):
            size_val = int(s.item())
            results.append(gathered_tensors[i][:size_val])
        return torch.cat(results, dim=0)
    except Exception as e:
        print(f"⚠️ [WARN] Failed to gather 2D tensor across DDP ranks: {e}")
        return tensor


class CustomDetectionValidator(DetectionValidator):
    """Custom YOLO Validation class that calculates advanced metrics.

    Calculates matched box-level IoU and Dice scores, per-class metrics
    (Recall, Precision, TP, FP, FN, F1-score, mAP50, mAP50-95), and maps bboxes
    to image-level multi-label classification metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_class_name = "normal"
        self.abnormal_class_name = "abnormal"
        # Bbox IoU and Dice collectors (stores tuple of: class_idx, best_iou, dice)
        self.custom_iou_dice_stats: List[tuple[int, Union[float, torch.Tensor], Union[float, torch.Tensor]]] = []
        
        # Initialize internal dynamic dictionary collectors
        self.cls_gt = {}
        self.cls_pred = {}
        self.cls_prob = {}

    def _get_class_idx(self, name: str) -> int | None:
        if not hasattr(self, "names") or not self.names:
            return None
        for idx, val in self.names.items():
            if val.lower() == name.lower():
                return idx
        return None

    # Backward compatibility properties
    @property
    def cls_gt_abnormal(self) -> List[int]:
        idx = self._get_class_idx(self.abnormal_class_name)
        if idx is not None and idx in self.cls_gt:
            return [int(x.item() if isinstance(x, torch.Tensor) else x) for x in self.cls_gt[idx]]
        return []

    @cls_gt_abnormal.setter
    def cls_gt_abnormal(self, val: List[int]):
        idx = self._get_class_idx(self.abnormal_class_name)
        if idx is not None:
            self.cls_gt[idx] = val

    @property
    def cls_pred_abnormal(self) -> List[int]:
        idx = self._get_class_idx(self.abnormal_class_name)
        if idx is not None and idx in self.cls_pred:
            return [int(x.item() if isinstance(x, torch.Tensor) else x) for x in self.cls_pred[idx]]
        return []

    @cls_pred_abnormal.setter
    def cls_pred_abnormal(self, val: List[int]):
        idx = self._get_class_idx(self.abnormal_class_name)
        if idx is not None:
            self.cls_pred[idx] = val

    @property
    def cls_prob_abnormal(self) -> List[float]:
        idx = self._get_class_idx(self.abnormal_class_name)
        if idx is not None and idx in self.cls_prob:
            return [float(x.item() if isinstance(x, torch.Tensor) else x) for x in self.cls_prob[idx]]
        return []

    @cls_prob_abnormal.setter
    def cls_prob_abnormal(self, val: List[float]):
        idx = self._get_class_idx(self.abnormal_class_name)
        if idx is not None:
            self.cls_prob[idx] = val

    @property
    def cls_gt_text(self) -> List[int]:
        idx = self._get_class_idx("text")
        if idx is not None and idx in self.cls_gt:
            return [int(x.item() if isinstance(x, torch.Tensor) else x) for x in self.cls_gt[idx]]
        return []

    @cls_gt_text.setter
    def cls_gt_text(self, val: List[int]):
        idx = self._get_class_idx("text")
        if idx is not None:
            self.cls_gt[idx] = val

    @property
    def cls_pred_text(self) -> List[int]:
        idx = self._get_class_idx("text")
        if idx is not None and idx in self.cls_pred:
            return [int(x.item() if isinstance(x, torch.Tensor) else x) for x in self.cls_pred[idx]]
        return []

    @cls_pred_text.setter
    def cls_pred_text(self, val: List[int]):
        idx = self._get_class_idx("text")
        if idx is not None:
            self.cls_pred[idx] = val

    @property
    def cls_prob_text(self) -> List[float]:
        idx = self._get_class_idx("text")
        if idx is not None and idx in self.cls_prob:
            return [float(x.item() if isinstance(x, torch.Tensor) else x) for x in self.cls_prob[idx]]
        return []

    @cls_prob_text.setter
    def cls_prob_text(self, val: List[float]):
        idx = self._get_class_idx("text")
        if idx is not None:
            self.cls_prob[idx] = val

    def init_metrics(self, model):
        """Initializes/resets metric collectors at the start of each validation epoch."""
        super().init_metrics(model)
        self.custom_iou_dice_stats = []
        self.cls_gt = {idx: [] for idx in self.names.keys()}
        self.cls_pred = {idx: [] for idx in self.names.keys()}
        self.cls_prob = {idx: [] for idx in self.names.keys()}

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Accumulates standard metrics and extracts custom verification metrics."""
        # 1. Update standard Ultralytics metrics
        super().update_metrics(preds, batch)

        # 2. Extract matched box stats and classification labels on GPU
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            device = predn["cls"].device if predn["cls"].shape[0] > 0 else (pbatch["cls"].device if pbatch["cls"].shape[0] > 0 else torch.device("cpu"))
            
            gt_classes = pbatch["cls"].reshape(-1).long() if pbatch["cls"].shape[0] > 0 else torch.empty(0, dtype=torch.long, device=device)
            pred_classes = predn["cls"].reshape(-1).long() if predn["cls"].shape[0] > 0 else torch.empty(0, dtype=torch.long, device=device)
            pred_confs = predn["conf"] if predn["cls"].shape[0] > 0 else torch.empty(0, dtype=torch.float32, device=device)

            # --- Image-level Multi-Label Classification Labels ---
            for idx in self.names.keys():
                gt_has = (gt_classes == idx).any().int() if gt_classes.numel() > 0 else torch.tensor(0, dtype=torch.int32, device=device)
                
                pred_has = torch.tensor(0, dtype=torch.int32, device=device)
                prob = torch.tensor(0.0, dtype=torch.float32, device=device)
                
                if pred_classes.numel() > 0:
                    class_mask = (pred_classes == idx)
                    if class_mask.any():
                        prob = pred_confs[class_mask].max()
                        pred_has = (prob >= 0.25).int()

                if idx not in self.cls_gt:
                    self.cls_gt[idx] = []
                    self.cls_pred[idx] = []
                    self.cls_prob[idx] = []
                self.cls_gt[idx].append(gt_has)
                self.cls_pred[idx].append(pred_has)
                self.cls_prob[idx].append(prob)

            # --- Bounding Box-level matched IoU and Dice ---
            if pbatch["bboxes"].shape[0] > 0 and predn["bboxes"].shape[0] > 0:
                iou_matrix = box_iou(pbatch["bboxes"], predn["bboxes"])
                matched_pred_indices = set()

                for gt_i in range(pbatch["bboxes"].shape[0]):
                    gt_c = gt_classes[gt_i].item()
                    best_iou = torch.tensor(-1.0, dtype=torch.float32, device=device)
                    best_pred_idx = -1

                    for pred_i in range(predn["bboxes"].shape[0]):
                        if pred_i in matched_pred_indices:
                            continue
                        if pred_classes[pred_i].item() != gt_c:
                            continue
                        
                        curr_iou = iou_matrix[gt_i, pred_i]
                        if curr_iou > best_iou:
                            best_iou = curr_iou
                            best_pred_idx = pred_i

                    if best_pred_idx != -1 and best_iou >= 0.50:
                        matched_pred_indices.add(best_pred_idx)
                        dice = (2.0 * best_iou) / (1.0 + best_iou)
                        self.custom_iou_dice_stats.append((int(gt_c), best_iou, dice))

    def get_stats(self) -> Dict[str, Any]:
        """Calculates and returns custom stats injected alongside YOLO stats."""
        concatenated_stats = {}
        try:
            if hasattr(self.metrics, "stats") and self.metrics.stats:
                for k, v in self.metrics.stats.items():
                    if v and len(v) > 0:
                        concatenated_stats[k] = np.concatenate(v, 0)
        except Exception as e:
            print(f"⚠️ [WARN] Error copying self.metrics.stats: {e}")

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
        device = self.device if hasattr(self, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Gather custom validator metrics across all DDP ranks on GPU
        gathered_gt = {}
        gathered_pred = {}
        gathered_prob = {}
        
        for idx in self.names.keys():
            if idx in self.cls_gt and len(self.cls_gt[idx]) > 0:
                gt_t = torch.stack([x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.int32, device=device) for x in self.cls_gt[idx]])
                pred_t = torch.stack([x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.int32, device=device) for x in self.cls_pred[idx]])
                prob_t = torch.stack([x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32, device=device) for x in self.cls_prob[idx]])
            else:
                gt_t = torch.empty(0, dtype=torch.int32, device=device)
                pred_t = torch.empty(0, dtype=torch.int32, device=device)
                prob_t = torch.empty(0, dtype=torch.float32, device=device)

            gathered_gt[idx] = ddp_gather_tensor(gt_t).cpu().numpy()
            gathered_pred[idx] = ddp_gather_tensor(pred_t).cpu().numpy()
            gathered_prob[idx] = ddp_gather_tensor(prob_t).cpu().numpy()

        if self.custom_iou_dice_stats:
            iou_dice_tensor = torch.stack([
                torch.stack([
                    torch.tensor(c, dtype=torch.float32, device=device) if not isinstance(c, torch.Tensor) else c.to(device),
                    iou.to(device) if isinstance(iou, torch.Tensor) else torch.tensor(iou, dtype=torch.float32, device=device),
                    dice.to(device) if isinstance(dice, torch.Tensor) else torch.tensor(dice, dtype=torch.float32, device=device)
                ])
                for c, iou, dice in self.custom_iou_dice_stats
            ])
        else:
            iou_dice_tensor = torch.empty((0, 3), dtype=torch.float32, device=device)

        gathered_iou_dice = ddp_gather_tensor_2d(iou_dice_tensor).cpu().numpy()

        # 2. Bbox class-specific TP, FP, FN, Precision, Recall, F1, maps
        try:
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
            if gathered_iou_dice.size > 0:
                ious = gathered_iou_dice[:, 1]
                dices = gathered_iou_dice[:, 2]
                stats["metrics/custom_mean_bbox_IoU"] = float(np.mean(ious))
                stats["metrics/custom_mean_bbox_Dice"] = float(np.mean(dices))

                for idx in names_list:
                    name = self.names[idx]
                    class_mask = (gathered_iou_dice[:, 0] == idx)
                    if class_mask.any():
                        class_ious = gathered_iou_dice[class_mask, 1]
                        class_dices = gathered_iou_dice[class_mask, 2]
                        stats[f"metrics/custom_mean_bbox_IoU/{name}"] = float(np.mean(class_ious))
                        stats[f"metrics/custom_mean_bbox_Dice/{name}"] = float(np.mean(class_dices))
                    else:
                        stats[f"metrics/custom_mean_bbox_IoU/{name}"] = 0.0
                        stats[f"metrics/custom_mean_bbox_Dice/{name}"] = 0.0
            else:
                stats["metrics/custom_mean_bbox_IoU"] = 0.0
                stats["metrics/custom_mean_bbox_Dice"] = 0.0
                for idx in names_list:
                    name = self.names[idx]
                    stats[f"metrics/custom_mean_bbox_IoU/{name}"] = 0.0
                    stats[f"metrics/custom_mean_bbox_Dice/{name}"] = 0.0
        except Exception as e:
            stats["metrics/custom_mean_bbox_IoU"] = 0.0
            stats["metrics/custom_mean_bbox_Dice"] = 0.0
            for idx in names_list:
                name = self.names[idx]
                stats[f"metrics/custom_mean_bbox_IoU/{name}"] = 0.0
                stats[f"metrics/custom_mean_bbox_Dice/{name}"] = 0.0

        # 3. Image-level Multi-Label Classification Metrics for all classes
        try:
            for idx, cls_name in self.names.items():
                gt = gathered_gt.get(idx, np.array([]))
                pred = gathered_pred.get(idx, np.array([]))
                prob = gathered_prob.get(idx, np.array([]))

                if gt.size > 0:
                    stats[f"metrics/custom_cls_accuracy/{cls_name}"] = float(accuracy_score(gt, pred))
                    stats[f"metrics/custom_cls_precision/{cls_name}"] = float(precision_score(gt, pred, zero_division=0))
                    stats[f"metrics/custom_cls_recall/{cls_name}"] = float(recall_score(gt, pred, zero_division=0))
                    stats[f"metrics/custom_cls_f1/{cls_name}"] = float(f1_score(gt, pred, zero_division=0))
                    
                    if len(np.unique(gt)) > 1:
                        stats[f"metrics/custom_cls_auroc/{cls_name}"] = float(roc_auc_score(gt, prob))
                    else:
                        stats[f"metrics/custom_cls_auroc/{cls_name}"] = 0.5

                    cm = confusion_matrix(gt, pred, labels=[0, 1])
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    else:
                        tn = fp = fn = tp = 0
                    stats[f"metrics/custom_cls_tn/{cls_name}"] = float(tn)
                    stats[f"metrics/custom_cls_fp/{cls_name}"] = float(fp)
                    stats[f"metrics/custom_cls_fn/{cls_name}"] = float(fn)
                    stats[f"metrics/custom_cls_tp/{cls_name}"] = float(tp)
        except Exception as e:
            print(f"⚠️ [WARN] Error calculating classification metrics: {e}")

        # Duplicate abnormal metric keys for config abnormal_class_name compatibility
        cfg_abnormal = getattr(self, "abnormal_class_name", "abnormal")
        if cfg_abnormal != "abnormal":
            extra_stats = {}
            for k, v in stats.items():
                if "/abnormal" in k:
                    new_key = k.replace("/abnormal", f"/{cfg_abnormal}")
                    extra_stats[new_key] = v
            stats.update(extra_stats)

        return stats

    def finalize_metrics(self) -> None:
        """Sets final standard metrics and prints a clean validation summary table."""
        try:
            super().finalize_metrics()
        except Exception as e:
            print(f"⚠️ [WARN] Failed to finalize standard YOLO metrics: {e}")
        
        stats = self.get_stats()
        is_short = getattr(self, "training", True)
        
        if is_short:
            iou = stats.get('metrics/custom_mean_bbox_IoU', 0.0)
            dice = stats.get('metrics/custom_mean_bbox_Dice', 0.0)
            
            cfg_abnormal = getattr(self, "abnormal_class_name", "abnormal")
            abn_f1 = stats.get(f"metrics/custom_F1/{cfg_abnormal}", stats.get("metrics/custom_F1/abnormal", 0.0))
            txt_f1 = stats.get("metrics/custom_F1/text", 0.0)
            abn_img_f1 = stats.get(f"metrics/custom_cls_f1/{cfg_abnormal}", stats.get("metrics/custom_cls_f1/abnormal", 0.0))
            txt_img_f1 = stats.get("metrics/custom_cls_f1/text", 0.0)
            cell_f1 = stats.get("metrics/custom_F1/cell", 0.0)
            cell_img_f1 = stats.get("metrics/custom_cls_f1/cell", 0.0)
            
            images_count = getattr(self, "seen", 0)
            instances_count = 0
            if hasattr(self, "nt_per_class") and self.nt_per_class is not None:
                instances_count = int(self.nt_per_class.sum())
                
            print(f"\n{'Class':<15} {'Images':<8} {'Targets':<8} {'Mean-IoU':<10} {'Mean-Dice':<10} {'Abn-BoxF1':<10} {'Txt-BoxF1':<10} {'Cell-BoxF1':<11} {'Abn-ImgF1':<10} {'Txt-ImgF1':<10}")
            print(f"{'custom-stats':<15} {images_count:<8} {instances_count:<8} {iou:<10.4f} {dice:<10.4f} {abn_f1:<10.3f} {txt_f1:<10.3f} {cell_f1:<11.3f} {abn_img_f1:<10.3f} {txt_img_f1:<10.3f}\n")
        else:
            print("\n" + "=" * 80)
            print("📊 CUSTOM VALIDATION METRICS REPORT (FINAL EVALUATION)")
            print("=" * 80)
            
            print(f"📐 Matched Bbox IoU:   {stats.get('metrics/custom_mean_bbox_IoU', 0.0):.4f}")
            print(f"📐 Matched Bbox Dice:  {stats.get('metrics/custom_mean_bbox_Dice', 0.0):.4f}")
            
            print("\n📦 Bbox Metrics Per-Class (IoU=0.50):")
            print(f"{'Class':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'mAP50':<6} | {'IoU':<6} | {'Dice':<6}")
            print("-" * 105)
            for idx, name in self.names.items():
                tp = stats.get(f"metrics/custom_TP/{name}", 0)
                fp = stats.get(f"metrics/custom_FP/{name}", 0)
                fn = stats.get(f"metrics/custom_FN/{name}", 0)
                p = stats.get(f"metrics/custom_P/{name}", 0.0)
                r = stats.get(f"metrics/custom_R/{name}", 0.0)
                f1 = stats.get(f"metrics/custom_F1/{name}", 0.0)
                map50 = stats.get(f"metrics/custom_mAP50/{name}", 0.0)
                iou = stats.get(f"metrics/custom_mean_bbox_IoU/{name}", 0.0)
                dice = stats.get(f"metrics/custom_mean_bbox_Dice/{name}", 0.0)
                print(f"{name:<15} | {tp:<5} | {fp:<5} | {fn:<5} | {p:.3f}  | {r:.3f}  | {f1:.3f}  | {map50:.3f}  | {iou:.3f}  | {dice:.3f}")

            print("\n🖥️ Image-Level Multi-Label Classification Conversion:")
            print(f"{'Class Indicator':<18} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'AUROC':<6}")
            print("-" * 80)
            printed_classes = []
            for idx, cls_name in self.names.items():
                if cls_name in printed_classes:
                    continue
                acc = stats.get(f"metrics/custom_cls_accuracy/{cls_name}")
                if acc is not None:
                    printed_classes.append(cls_name)
                    p = stats.get(f"metrics/custom_cls_precision/{cls_name}", 0.0)
                    r = stats.get(f"metrics/custom_cls_recall/{cls_name}", 0.0)
                    f1 = stats.get(f"metrics/custom_cls_f1/{cls_name}", 0.0)
                    auc = stats.get(f"metrics/custom_cls_auroc/{cls_name}", 0.5)
                    print(f"{cls_name:<18} | {acc:.3f}    | {p:.3f}     | {r:.3f}   | {f1:.3f} | {auc:.3f}")
            print("=" * 80 + "\n")



class CustomDetectionTrainer(DetectionTrainer):
    """Custom YOLO Trainer class that registers CustomDetectionValidator."""

    def __init__(self, *args, **kwargs):
        # Extract configurations for normal and abnormal class names
        self.normal_class_name = kwargs.pop("normal_class_name", "normal")#TODO: what if I have more classes?
        self.abnormal_class_name = kwargs.pop("abnormal_class_name", "abnormal")#TODO: what if I have more classes?
        super().__init__(*args, **kwargs)
        # Register the callback that runs at the end of each validation epoch to update progress metrics
        self.add_callback("on_fit_epoch_end", save_yolo_trainer_state_callback)

    def get_validator(self) -> CustomDetectionValidator:
        """Returns the custom validator instance."""
        # Configure model losses and instantiate the custom validator with custom class name rules
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = CustomDetectionValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)
        validator.normal_class_name = self.normal_class_name
        validator.abnormal_class_name = self.abnormal_class_name
        return validator


def save_yolo_trainer_state_callback(trainer) -> None:
    """Callback to save YOLO training progress in HF trainer_state.json format."""
    # Only save on the main process to avoid DDP write collisions
    if not (getattr(trainer, "is_world_process_zero", True) or int(os.environ.get("LOCAL_RANK", "0")) == 0):
        return

    save_dir = Path(trainer.save_dir)
    state_file = save_dir / "trainer_state.json"
    
    epoch = float(trainer.epoch + 1)
    step = int(trainer.epoch + 1)
    
    # 1. Load existing state or initialize new
    # EXAMPLE format of trainer_state.json:
    # {
    #   "best_global_step": 3,
    #   "best_metric": 0.885,
    #   "best_model_checkpoint": "outputs/train/weights/best.pt",
    #   "epoch": 3.0,
    #   "global_step": 3,
    #   "log_history": [...]
    # }
    state = {
        "best_global_step": 0,
        "best_metric": 0.0,
        "best_model_checkpoint": "",
        "epoch": epoch,
        "global_step": step,
        "log_history": []
    }
    
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
        except Exception:
            pass
            
    # Update current epoch and step
    state["epoch"] = epoch
    state["global_step"] = step
    
    # Remove previous log_history entries for this epoch to support resumption or safety overrides
    state["log_history"] = [entry for entry in state["log_history"] if entry.get("epoch") != epoch]
    
    # 2. Get training loss
    # EXAMPLE train_loss: 2.345 (sum of box_loss, cls_loss, dfl_loss)
    train_loss = 0.0
    if hasattr(trainer, "tloss") and trainer.tloss is not None:
        if isinstance(trainer.tloss, (list, tuple, np.ndarray)):
            train_loss = float(sum(trainer.tloss))
        elif torch.is_tensor(trainer.tloss):
            train_loss = float(trainer.tloss.sum().item())
            
    # Get current learning rate
    # EXAMPLE lr: 0.001
    lr = 0.0
    if hasattr(trainer, "lr") and trainer.lr is not None:
        if isinstance(trainer.lr, dict):
            lr = float(next(iter(trainer.lr.values())))
        elif isinstance(trainer.lr, (list, tuple, np.ndarray)) and len(trainer.lr) > 0:
            lr = float(trainer.lr[0])
        else:
            lr = float(trainer.lr)
            
    # Add training entry
    train_entry = {
        "epoch": epoch,
        "learning_rate": lr,
        "loss": train_loss,
        "step": step
    }
    state["log_history"].append(train_entry)
    
    # 3. Get evaluation metrics
    eval_entry = {
        "epoch": epoch,
        "step": step
    }
    
    # Get validation loss
    # EXAMPLE val_loss: 1.876
    val_loss = 0.0
    if hasattr(trainer, "validator") and trainer.validator is not None:
        if hasattr(trainer.validator, "loss") and trainer.validator.loss is not None:
            if isinstance(trainer.validator.loss, (list, tuple, np.ndarray)):
                val_loss = float(sum(trainer.validator.loss))
            elif torch.is_tensor(trainer.validator.loss):
                val_loss = float(trainer.validator.loss.sum().item())
                
    if val_loss == 0.0 and trainer.metrics:
        # Fallback from metrics dict
        val_loss = float(
            trainer.metrics.get("val/box_loss", 0.0) +
            trainer.metrics.get("val/cls_loss", 0.0) +
            trainer.metrics.get("val/dfl_loss", 0.0)
        )
        
    eval_entry["eval_loss"] = val_loss
    
    # Copy and map validation metrics
    # EXAMPLE eval_entry populated:
    # {
    #   "epoch": 3.0,
    #   "step": 3,
    #   "eval_loss": 1.876,
    #   "eval_precision": 0.762,
    #   "eval_custom_cls_f1/abnormal": 0.885,
    #   ...
    # }
    if trainer.metrics:
        # Map standard YOLO metrics
        eval_entry["eval_precision"] = float(trainer.metrics.get("metrics/precision(B)", 0.0))
        eval_entry["eval_recall"] = float(trainer.metrics.get("metrics/recall(B)", 0.0))
        eval_entry["eval_mAP50"] = float(trainer.metrics.get("metrics/mAP50(B)", 0.0))
        eval_entry["eval_mAP50-95"] = float(trainer.metrics.get("metrics/mAP50-95(B)", 0.0))
        
        # Map all custom metrics directly
        for k, v in trainer.metrics.items():
            if k.startswith("metrics/custom_") and isinstance(v, (int, float, np.integer, np.floating)):
                new_key = k.replace("metrics/custom_", "eval_custom_")
                eval_entry[new_key] = float(v)
                
    state["log_history"].append(eval_entry)
    
    # 4. Update best metric and best checkpoint path
    # Prioritize abnormal classification F1 as best metric, fallback to mAP50
    metric_for_best = "eval_custom_cls_f1/abnormal"
    current_best_val = eval_entry.get(metric_for_best, 0.0)
    if current_best_val == 0.0:
        metric_for_best = "eval_mAP50"
        current_best_val = eval_entry.get(metric_for_best, 0.0)
        
    if current_best_val > state.get("best_metric", 0.0):
        state["best_metric"] = current_best_val
        state["best_global_step"] = step
        # Path to best weight checkpoint
        best_ckpt = save_dir / "weights" / "best.pt"
        if best_ckpt.exists():
            try:
                state["best_model_checkpoint"] = str(best_ckpt.relative_to(Path.cwd()))
            except Exception:
                state["best_model_checkpoint"] = str(best_ckpt)
        else:
            state["best_model_checkpoint"] = str(best_ckpt)
            
    # Write to file
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"⚠️ [WARN] Failed to write trainer_state.json: {e}")
