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
        # Bbox IoU and Dice collectors
        self.custom_iou_dice_stats: List[tuple[float, float]] = []
        
        # Image-level abnormal classification collectors
        self.cls_gt_abnormal: List[int] = []
        self.cls_pred_abnormal: List[int] = []
        self.cls_prob_abnormal: List[float] = []

        # Image-level text classification collectors
        self.cls_gt_text: List[int] = []
        self.cls_pred_text: List[int] = []
        self.cls_prob_text: List[float] = []

    def init_metrics(self, model):
        """Initializes/resets metric collectors at the start of each validation epoch."""
        super().init_metrics(model)
        self.custom_iou_dice_stats = []
        self.cls_gt_abnormal = []
        self.cls_pred_abnormal = []
        self.cls_prob_abnormal = []
        self.cls_gt_text = []
        self.cls_pred_text = []
        self.cls_prob_text = []

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Accumulates standard metrics and extracts custom verification metrics."""
        # 1. Update standard Ultralytics metrics
        super().update_metrics(preds, batch)

        # Class names to indices map
        abnormal_idx = None
        text_idx = None
        cfg_abnormal = getattr(self, "abnormal_class_name", "abnormal")
        for idx, name in self.names.items():
            if name in (cfg_abnormal, "abnormal", "abnormal"):
                abnormal_idx = idx
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
            gt_has_abn = int(abnormal_idx in gt_classes) if abnormal_idx is not None else 0
            gt_has_txt = int(text_idx in gt_classes) if text_idx is not None else 0

            # Find predictions above decision threshold (standard 0.25 confidence)
            pred_has_abn = 0
            prob_abn = 0.0
            pred_has_txt = 0
            prob_txt = 0.0

            for c_idx, conf in zip(pred_classes, pred_confs):
                if abnormal_idx is not None and c_idx == abnormal_idx:
                    prob_abn = max(prob_abn, float(conf))
                    if conf >= 0.25:
                        pred_has_abn = 1
                elif text_idx is not None and c_idx == text_idx:
                    prob_txt = max(prob_txt, float(conf))
                    if conf >= 0.25:
                        pred_has_txt = 1

            if abnormal_idx is not None:
                self.cls_gt_abnormal.append(gt_has_abn)
                self.cls_pred_abnormal.append(pred_has_abn)
                self.cls_prob_abnormal.append(prob_abn)

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

        # Gather custom validator metrics across all DDP ranks
        cls_gt_abnormal = ddp_gather_list(self.cls_gt_abnormal)
        cls_pred_abnormal = ddp_gather_list(self.cls_pred_abnormal)
        cls_prob_abnormal = ddp_gather_list(self.cls_prob_abnormal)
        
        cls_gt_text = ddp_gather_list(self.cls_gt_text)
        cls_pred_text = ddp_gather_list(self.cls_pred_text)
        cls_prob_text = ddp_gather_list(self.cls_prob_text)
        
        custom_iou_dice_stats = ddp_gather_list(self.custom_iou_dice_stats)

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
            if custom_iou_dice_stats:
                ious = [x[0] for x in custom_iou_dice_stats]
                dices = [x[1] for x in custom_iou_dice_stats]
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
                ("abnormal", cls_gt_abnormal, cls_pred_abnormal, cls_prob_abnormal),
                ("text", cls_gt_text, cls_pred_text, cls_prob_text)
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

                    # Abnormal and Text classification confusion matrix counts
                    cm = confusion_matrix(gt_arr, pred_arr, labels=[0, 1])
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
        cfg_abnormal = getattr(self, "abnormal_class_name", "abnormal")
        printed_classes = []
        for cls_name in ["abnormal", cfg_abnormal, "text"]:
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
        self.normal_class_name = kwargs.pop("normal_class_name", "normal")
        self.abnormal_class_name = kwargs.pop("abnormal_class_name", "abnormal")
        super().__init__(*args, **kwargs)
        # Register the trainer_state.json writing callback
        self.add_callback("on_fit_epoch_end", save_yolo_trainer_state_callback)

    def get_validator(self) -> CustomDetectionValidator:
        """Returns the custom validator instance."""
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
    train_loss = 0.0
    if hasattr(trainer, "tloss") and trainer.tloss is not None:
        if isinstance(trainer.tloss, (list, tuple, np.ndarray)):
            train_loss = float(sum(trainer.tloss))
        elif torch.is_tensor(trainer.tloss):
            train_loss = float(trainer.tloss.sum().item())
            
    # Get current learning rate
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
