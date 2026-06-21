import os
import glob
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Using Context7 for Streamlit and Plotly API usage and layout setup.

import re
from typing import Any, Optional, Tuple

def parse_fold_and_base_name(display_name: str, short_cfg_name: str, config_fold: Any = None) -> Tuple[Optional[str], str]:
    """
    Extract fold index and base config name by stripping fold patterns.
    """
    # If fold is explicitly provided in config, use it
    if config_fold is not None:
        fold = str(config_fold)
        base_cfg = re.sub(rf'[-_]fold[-_]?{fold}\b', '', short_cfg_name, flags=re.IGNORECASE)
        base_cfg = re.sub(r'__+', '_', base_cfg).strip('-_')
        return fold, base_cfg

    # Fallback: parse from short_cfg_name or display_name
    for name in [short_cfg_name, display_name]:
        match = re.search(r'[-_]fold[-_]?(\d+)\b', name, re.IGNORECASE)
        if match:
            fold = match.group(1)
            base_cfg = re.sub(rf'[-_]fold[-_]?{fold}\b', '', short_cfg_name, flags=re.IGNORECASE)
            base_cfg = re.sub(r'__+', '_', base_cfg).strip('-_')
            return fold, base_cfg
            
    return None, short_cfg_name


def find_latest_checkpoint_state(run_dir):
    """
    Locate the latest checkpoint folder inside a run directory and return
    the path to its trainer_state.json if it exists.
    """
    checkpoints = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort checkpoints numerically by step number
    def get_step(path):
        try:
            return int(path.split("-")[-1])
        except ValueError:
            return -1
            
    checkpoints.sort(key=get_step, reverse=True)
    for ckpt in checkpoints:
        state_path = os.path.join(ckpt, "trainer_state.json")
        if os.path.exists(state_path):
            return state_path
    return None

def estimate_model_params(model_name, task, peft_type, peft_config, is_yolo_dino=False, yolo_variant=None):
    vit_s_params = 22050816
    vit_b_params = 85955328
    
    yolo_params = {
        "yolo11n": 2600000, "yolo11s": 9400000, "yolo11m": 20100000, "yolo11l": 25300000, "yolo11x": 56900000,
        "yolo26n": 5500000, "yolo26s": 19300000, "yolo26m": 40700000, "yolo26l": 87800000, "yolo26x": 136900000,
        "yolov8n": 3200000, "yolov8s": 11200000, "yolov8m": 25900000, "yolov8l": 43700000, "yolov8x": 68200000
    }
    
    is_vit_b = "vitb16" in model_name.lower() or "vit-base" in model_name.lower()
    d = 768 if is_vit_b else 384
    num_layers = 12
    backbone_params = vit_b_params if is_vit_b else vit_s_params
    
    if task == "Classification":
        head_params = 1538 if is_vit_b else 770
        if peft_type == "none" or not peft_type:
            return {"total": backbone_params + head_params, "trainable": head_params}
            
        peft_params = 0
        target_blocks = peft_config.get("lora_target_blocks") or peft_config.get("adapter_target_blocks") or peft_config.get("vpt_target_blocks")
        targeted_layers = len(target_blocks) if target_blocks else num_layers
        
        if peft_type == "lora":
            r = peft_config.get("lora_r", 8)
            peft_params = targeted_layers * (4 * r * d)
        elif peft_type == "adapter":
            bottleneck_dim = peft_config.get("adapter_bottleneck_dim", 64)
            peft_params = targeted_layers * (2 * d * bottleneck_dim + d + bottleneck_dim)
        elif peft_type == "visual_prompt":
            num_tokens = peft_config.get("vpt_num_tokens", 10)
            deep = peft_config.get("vpt_deep", False)
            if deep:
                peft_params = targeted_layers * num_tokens * d
            else:
                peft_params = num_tokens * d
                
        return {
            "total": backbone_params + head_params + peft_params,
            "trainable": head_params + peft_params
        }
    else:
        if not is_yolo_dino:
            name_key = Path(yolo_variant or model_name).stem.replace(".pt", "").lower()
            tot = yolo_params.get(name_key, 5500000)
            return {"total": tot, "trainable": tot}
        else:
            yolo_head_params = {
                "yolo26n": 2800000, "yolo26s": 9900000, "yolo26m": 20700000, "yolo26l": 44800000, "yolo26x": 69900000,
                "yolo11n": 1400000, "yolo11s": 4800000, "yolo11m": 10100000, "yolo11l": 21800000, "yolo11x": 34100000
            }
            variant = Path(yolo_variant).stem.lower()
            head_params = yolo_head_params.get(variant, 2800000)
            
            if peft_type == "none" or not peft_type:
                return {"total": backbone_params + head_params, "trainable": head_params}
                
            peft_params = 0
            target_blocks = peft_config.get("lora_target_blocks") or peft_config.get("adapter_target_blocks") or peft_config.get("vpt_target_blocks")
            targeted_layers = len(target_blocks) if target_blocks else num_layers
            
            if peft_type == "lora":
                r = peft_config.get("lora_r", 8)
                peft_params = targeted_layers * (4 * r * d)
            elif peft_type == "adapter":
                bottleneck_dim = peft_config.get("adapter_bottleneck_dim", 64)
                peft_params = targeted_layers * (2 * d * bottleneck_dim + d + bottleneck_dim)
            elif peft_type == "visual_prompt":
                num_tokens = peft_config.get("vpt_num_tokens", 10)
                deep = peft_config.get("vpt_deep", False)
                if deep:
                    peft_params = targeted_layers * num_tokens * d
                else:
                    peft_params = num_tokens * d
                    
            return {
                "total": backbone_params + head_params + peft_params,
                "trainable": head_params + peft_params
            }

def load_results(base_path="outputs"):
    """
    Recursively scans base_path to load config.yaml and trainer_state.json files.
    """
    runs_data = []
    base_path_obj = Path(base_path)
    
    if not base_path_obj.exists():
        return pd.DataFrame()

    # Load parameter cache if available
    param_cache = {}
    cache_path = base_path_obj / "parameter_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                param_cache = json.load(f)
        except Exception:
            pass

    # Locate directories containing config.yaml
    for root, dirs, files in os.walk(base_path):
        # Skip the standard 'log' or 'tb' log folders
        if "log" in root or "runs" in root:
            continue
            
        if "config.yaml" in files:
            run_dir = Path(root)
            config_path = run_dir / "config.yaml"
            
            # 1. Parse config.yaml
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
            except Exception as e:
                st.warning(f"⚠️ Error parsing config at {config_path}: {e}")
                continue
                
            if not cfg:
                continue

            # Extract config values (with fallbacks)
            model_name = cfg.get("model_name", "unknown")
            peft_cfg = cfg.get("peft", {})
            peft_type = peft_cfg.get("type", "none")
            
            imb_cfg = cfg.get("imbalance", {})
            imbalance_strategy = imb_cfg.get("strategy") or imb_cfg.get("oversampling_method") or "none"
            loss_type = imb_cfg.get("loss_type") or "cross_entropy"
            
            # Resolve specific PEFT details
            peft_detail = "none"
            if peft_type == "lora":
                peft_detail = f"r={peft_cfg.get('lora_r', 8)}"
            elif peft_type == "adapter":
                peft_detail = f"d={peft_cfg.get('adapter_bottleneck_dim', 64)}"
            elif peft_type == "visual_prompt":
                peft_detail = f"t={peft_cfg.get('vpt_num_tokens', 10)}"
            
            # Determine task dynamically
            is_det = "yolo_model_config" in cfg
            task = "Detection" if is_det else "Classification"
            custom_param = "default"
            abnormal_class_name = cfg.get("data", {}).get("abnormal_class_name", "abnormal")

            # Resolve dataset
            if is_det:
                yolo_yaml = cfg.get("yolo_data_yaml", "")
                dataset = Path(yolo_yaml).stem if yolo_yaml else "unknown"
            else:
                data_dir = cfg.get("data", {}).get("data_dir", "") or cfg.get("data_dir", "")
                dataset = Path(data_dir).name if data_dir else "cls_v1.0"

            # Resolve parameter counts
            rel_run_path = str(run_dir.relative_to(base_path_obj))
            if rel_run_path in param_cache:
                total_params = param_cache[rel_run_path]["total"]
                trainable_params = param_cache[rel_run_path]["trainable"]
            else:
                yolo_variant = cfg.get("yolo_model_config", "")
                is_yolo_dino = is_det and ("dino" in yolo_variant.lower() or "dino" in model_name.lower())
                est = estimate_model_params(
                    model_name=model_name,
                    task=task,
                    peft_type=peft_type,
                    peft_config=peft_cfg,
                    is_yolo_dino=is_yolo_dino,
                    yolo_variant=yolo_variant
                )
                total_params = est["total"]
                trainable_params = est["trainable"]
            
            pct_trainable = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
            
            # 2. Parse trainer_state.json (look in root, fallback to latest checkpoint)
            state_path = run_dir / "trainer_state.json"
            if not state_path.exists():
                checkpoint_state = find_latest_checkpoint_state(run_dir)
                if checkpoint_state:
                    state_path = Path(checkpoint_state)
            
            history = []
            best_eval_f1 = 0.0
            best_eval_loss = float('inf')
            final_train_loss = None
            best_epoch_metrics = {}
            completed = "DONE" in files
            
            if state_path.exists():
                try:
                    with open(state_path, "r") as f:
                        state = json.load(f)
                    
                    history = state.get("log_history", [])
                    
                    # Extract metrics from evaluation steps in history
                    if is_det:
                        eval_steps = [item for item in history if "eval_mAP50" in item or "eval_custom_cls_f1/abnormal" in item or "eval_custom_cls_f1/abnormal" in item]
                    else:
                        eval_steps = [item for item in history if "eval_f1" in item]
                        
                    train_losses = [item.get("loss") for item in history if "loss" in item]
                    
                    if eval_steps:
                        if is_det:
                            # Prioritize abnormal classification conversion F1, fallback to mAP50
                            best_step = max(
                                eval_steps,
                                key=lambda x: (
                                    x.get("eval_custom_cls_f1/abnormal", 0.0) or x.get("eval_custom_cls_f1/abnormal", 0.0) or x.get("eval_mAP50", 0.0),
                                    -x.get("eval_loss", float('inf'))
                                )
                             )
                            best_eval_f1 = best_step.get("eval_custom_cls_f1/abnormal", 0.0) or best_step.get("eval_custom_cls_f1/abnormal", 0.0)
                        else:
                            # Find step with max eval_f1. If tie, select lowest eval_loss
                            best_step = max(eval_steps, key=lambda x: (x.get("eval_f1", 0.0), -x.get("eval_loss", float('inf'))))
                            best_eval_f1 = best_step.get("eval_f1", 0.0)
                            
                        best_eval_loss = best_step.get("eval_loss", float('inf'))
                        best_epoch_metrics = best_step
                    
                    if train_losses:
                        final_train_loss = train_losses[-1]
                        
                except Exception as e:
                    st.warning(f"⚠️ Error parsing state at {state_path}: {e}")
            
            # Unified image-level classification metrics
            img_f1 = 0.0
            img_auroc = 0.5
            if best_epoch_metrics:
                if is_det:
                    img_f1 = best_epoch_metrics.get("eval_custom_cls_f1/abnormal", 0.0) or best_epoch_metrics.get("eval_custom_cls_f1/abnormal", 0.0)
                    img_auroc = best_epoch_metrics.get("eval_custom_cls_auroc/abnormal", 0.5) or best_epoch_metrics.get("eval_custom_cls_auroc/abnormal", 0.5)
                else:
                    img_f1 = best_epoch_metrics.get("eval_f1", 0.0)
                    img_auroc = best_epoch_metrics.get("eval_auroc", 0.5)

            # Calculate short directory name for display
            display_name = run_dir.parent.name if run_dir.parent.name != base_path_obj.name else run_dir.name
            # If the path looks like task__model__cfg_stem, extract the cfg_stem for easier reading
            parts = display_name.split("__")
            short_cfg_name = parts[-1] if len(parts) > 1 else display_name
            
            # Parse fold and base configuration name
            config_fold = cfg.get("fold", None)
            parsed_fold, base_cfg_name = parse_fold_and_base_name(display_name, short_cfg_name, config_fold)
            
            runs_data.append({
                "dir": str(run_dir.relative_to(base_path_obj)),
                "display_name": display_name,
                "short_cfg_name": short_cfg_name,
                "base_cfg_name": base_cfg_name,
                "fold": parsed_fold,
                "task": task,
                "model": model_name,
                "peft_type": peft_type,
                "peft_detail": peft_detail,
                "imbalance_strategy": imbalance_strategy,
                "loss_type": loss_type,
                "lr": cfg.get("learning_rate", 0.0),
                "epochs_configured": cfg.get("num_epochs", 0),
                "custom_param": custom_param,
                "best_eval_f1": best_eval_f1,
                "best_eval_loss": best_eval_loss if best_eval_loss != float('inf') else None,
                "final_train_loss": final_train_loss,
                "completed": completed,
                "best_metrics": best_epoch_metrics,
                "history": history,
                "img_abnormal_f1": img_f1,
                "img_abnormal_auroc": img_auroc,
                "abnormal_class_name": abnormal_class_name,
                "dataset": dataset,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "pct_trainable": pct_trainable
            })
            
    return pd.DataFrame(runs_data)


def get_best_epoch_metrics(history: list, benchmark_metric: str, mode: str = "max") -> dict:
    if not history:
        return {}
    
    # Filter history to items containing the target metric
    valid_steps = [item for item in history if benchmark_metric in item and "epoch" in item]
    if not valid_steps:
        # Fallback to any eval steps if the selected benchmark metric isn't present
        valid_steps = [item for item in history if any(k.startswith("eval_") for k in item.keys()) and "epoch" in item]
        if not valid_steps:
            return {}
        # Try to find a fallback metric that exists in the step
        for fallback in ["eval_f1", "eval_custom_cls_f1/abnormal", "eval_loss", "loss"]:
            if any(fallback in item for item in valid_steps):
                benchmark_metric = fallback
                mode = "min" if "loss" in fallback else "max"
                valid_steps = [item for item in valid_steps if benchmark_metric in item]
                break

    if not valid_steps:
        return {}

    # Find the step matching the benchmark objective
    try:
        if mode == "max":
            # Sort key prioritizes higher value, then lower eval_loss if present as a tie-breaker
            best_step = max(
                valid_steps,
                key=lambda x: (
                    float(x.get(benchmark_metric, 0.0) or 0.0),
                    -float(x.get("eval_loss", float('inf')) or float('inf'))
                )
            )
        else:
            # Sort key prioritizes lower value
            best_step = min(
                valid_steps,
                key=lambda x: (
                    float(x.get(benchmark_metric, float('inf')) or float('inf'))
                )
            )
        return best_step
    except Exception:
        # Fail-safe fallback: return the last eval step
        return valid_steps[-1]

def update_best_metrics_inplace(df: pd.DataFrame, benchmark_metric: str, mode: str, selected_label: str = "abnormal", use_custom: bool = True):
    if df.empty:
        return
    
    # We will update: best_eval_f1, best_eval_loss, best_metrics, img_abnormal_f1, img_abnormal_auroc
    for idx, row in df.iterrows():
        history = row["history"]
        is_det = row["task"] == "Detection"
        
        # Check if the run has custom metrics or normal metrics in its history
        run_has_custom = False
        run_has_normal = False
        run_labels = []
        for entry in history:
            for key in entry.keys():
                if key.startswith("eval_custom_cls_f1/"):
                    run_has_custom = True
                    lbl = key.split("/")[-1]
                    if lbl not in run_labels:
                        run_labels.append(lbl)
                if key in ["eval_f1", "eval_accuracy"]:
                    run_has_normal = True
        
        # Determine actual mode to use for this specific run
        actual_use_custom = use_custom
        if use_custom and not run_has_custom and run_has_normal:
            actual_use_custom = False
        elif not use_custom and not run_has_normal and run_has_custom:
            actual_use_custom = True
            
        actual_label = selected_label
        if actual_use_custom:
            if run_labels:
                if selected_label not in run_labels:
                    actual_label = run_labels[0]
            else:
                actual_label = "abnormal"
        
        # Resolve dynamic benchmark key
        target_metric = benchmark_metric
        target_mode = mode
        
        if benchmark_metric == "default":
            target_metric = f"eval_custom_cls_f1/{actual_label}" if actual_use_custom else "eval_f1"
            target_mode = "max"
            
        best_step = get_best_epoch_metrics(history, target_metric, target_mode)
        
        # Write updated metrics to the row
        df.at[idx, "best_metrics"] = best_step
        if best_step:
            if actual_use_custom:
                df.at[idx, "best_eval_f1"] = best_step.get(f"eval_custom_cls_f1/{actual_label}", 0.0)
                df.at[idx, "img_abnormal_f1"] = best_step.get(f"eval_custom_cls_f1/{actual_label}", 0.0)
                df.at[idx, "img_abnormal_auroc"] = best_step.get(f"eval_custom_cls_auroc/{actual_label}", 0.5)
                
                df.at[idx, "image_cls_f1"] = best_step.get(f"eval_custom_cls_f1/{actual_label}", None)
                df.at[idx, "image_cls_auroc"] = best_step.get(f"eval_custom_cls_auroc/{actual_label}", None)
                df.at[idx, "image_cls_precision"] = best_step.get(f"eval_custom_cls_precision/{actual_label}", None)
                df.at[idx, "image_cls_recall"] = best_step.get(f"eval_custom_cls_recall/{actual_label}", None)
            else:
                df.at[idx, "best_eval_f1"] = best_step.get("eval_f1", 0.0)
                df.at[idx, "img_abnormal_f1"] = best_step.get("eval_f1", 0.0)
                df.at[idx, "img_abnormal_auroc"] = best_step.get("eval_auroc", 0.5)
                
                df.at[idx, "image_cls_f1"] = best_step.get("eval_f1", None)
                df.at[idx, "image_cls_auroc"] = best_step.get("eval_auroc", None)
                df.at[idx, "image_cls_precision"] = best_step.get("eval_precision", None)
                df.at[idx, "image_cls_recall"] = best_step.get("eval_recall", None)
            
            # Handle float conversions safely
            val_loss = best_step.get("eval_loss", None)
            df.at[idx, "best_eval_loss"] = float(val_loss) if val_loss is not None else None
            
            # Extract individual columns for leaderboard selection
            df.at[idx, "eval_f1"] = best_step.get("eval_f1", None)
            df.at[idx, "eval_accuracy"] = best_step.get("eval_accuracy", None)
            df.at[idx, "eval_auroc"] = best_step.get("eval_auroc", None)
            df.at[idx, "eval_loss"] = best_step.get("eval_loss", None)
            df.at[idx, "eval_mAP50"] = best_step.get("eval_mAP50", None)
            df.at[idx, "eval_mAP50-95"] = best_step.get("eval_mAP50-95", None)
            df.at[idx, "eval_precision"] = best_step.get("eval_precision", None)
            df.at[idx, "eval_recall"] = best_step.get("eval_recall", None)
            df.at[idx, "eval_custom_mean_bbox_IoU"] = best_step.get("eval_custom_mean_bbox_IoU", None)
            df.at[idx, "eval_custom_mean_bbox_Dice"] = best_step.get("eval_custom_mean_bbox_Dice", None)
        else:
            df.at[idx, "best_eval_f1"] = 0.0
            df.at[idx, "best_eval_loss"] = None
            df.at[idx, "img_abnormal_f1"] = 0.0
            df.at[idx, "img_abnormal_auroc"] = 0.5
            
            for col in ["eval_f1", "eval_accuracy", "eval_auroc", "eval_loss", "eval_mAP50", "eval_mAP50-95", 
                        "eval_precision", "eval_recall", "eval_custom_mean_bbox_IoU", "eval_custom_mean_bbox_Dice",
                        "image_cls_f1", "image_cls_auroc", "image_cls_precision", "image_cls_recall"]:
                df.at[idx, col] = None

def group_results_by_fold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups dataframe of runs by common parameters, averaging numeric columns over folds.
    """
    if df.empty:
        return df
        
    from collections import defaultdict
        
    # Group columns
    group_cols = [
        "task", "model", "peft_type", "peft_detail", "imbalance_strategy", 
        "loss_type", "lr", "epochs_configured", "dataset", "base_cfg_name"
    ]
    
    # Check which group columns actually exist in df
    group_cols = [c for c in group_cols if c in df.columns]
    
    # We'll group by these columns and aggregate
    aggregated_rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        # Create dict of keys
        row = dict(zip(group_cols, keys))
        
        # Fold counts
        total_folds = len(group)
        completed_folds = group["completed"].sum()
        
        row["completed_folds_count"] = completed_folds
        row["total_folds_count"] = total_folds
        row["completed"] = (completed_folds == total_folds)
        
        # Add status string
        row["status"] = f"✅ All Folds Completed ({completed_folds}/{total_folds})" if completed_folds == total_folds else f"⏳ Folds: {completed_folds}/{total_folds} Completed"
        
        # Use first values for non-numeric/metadata columns
        row["short_cfg_name"] = row["base_cfg_name"]
        row["display_name"] = row["base_cfg_name"]
        row["dir"] = group.iloc[0]["dir"] if "dir" in group.columns else ""
        row["abnormal_class_name"] = group.iloc[0]["abnormal_class_name"] if "abnormal_class_name" in group.columns else "abnormal"
        
        # Average parameter counts
        row["total_params"] = group["total_params"].mean() if "total_params" in group.columns else None
        row["trainable_params"] = group["trainable_params"].mean() if "trainable_params" in group.columns else None
        row["pct_trainable"] = group["pct_trainable"].mean() if "pct_trainable" in group.columns else None
        
        # Average target metric columns
        numeric_cols = [
            "best_eval_f1", "best_eval_loss", "final_train_loss", 
            "img_abnormal_f1", "img_abnormal_auroc", "image_cls_f1", 
            "image_cls_auroc", "image_cls_precision", "image_cls_recall",
            "eval_f1", "eval_accuracy", "eval_auroc", "eval_loss", 
            "eval_mAP50", "eval_mAP50-95", "eval_precision", "eval_recall", 
            "eval_custom_mean_bbox_IoU", "eval_custom_mean_bbox_Dice"
        ]
        for col in numeric_cols:
            if col in group.columns:
                # Use mean, skipping NaNs
                vals = group[col].dropna()
                row[col] = vals.mean() if not vals.empty else None
                
        # Keep track of individual runs in the group for single-run inspection
        row["fold_runs"] = group.to_dict(orient="records")
        
        # Group history logs by epoch and average the metric values
        history_by_epoch = defaultdict(list)
        if "history" in group.columns:
            for _, item in group.iterrows():
                history = item["history"]
                if isinstance(history, list):
                    for entry in history:
                        if isinstance(entry, dict) and "epoch" in entry:
                            history_by_epoch[entry["epoch"]].append(entry)
                        
        merged_history = []
        for epoch, entries in sorted(history_by_epoch.items()):
            merged_entry = {"epoch": epoch}
            # Find all keys across entries
            all_keys = set()
            for entry in entries:
                all_keys.update(entry.keys())
            all_keys.discard("epoch")
            all_keys.discard("step")
            
            for k in all_keys:
                vals = [entry[k] for entry in entries if k in entry and entry[k] is not None]
                if vals:
                    merged_entry[k] = sum(vals) / len(vals)
            
            # Average step number
            steps = [entry["step"] for entry in entries if "step" in entry]
            if steps:
                merged_entry["step"] = int(sum(steps) / len(steps))
                
            merged_history.append(merged_entry)
            
        row["history"] = merged_history
        row["best_metrics"] = get_best_epoch_metrics(merged_history, "eval_loss", "min") if merged_history else {}

        
        aggregated_rows.append(row)
        
    return pd.DataFrame(aggregated_rows)

def main():

    st.set_page_config(
        page_title="🔋 Anomaly Detection - Ablation Results Visualizer",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling for rich aesthetics
    st.markdown("""
        <style>
            .main-header {
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.2rem;
            }
            .sub-header {
                font-size: 1.1rem;
                color: #a1a1aa;
                margin-bottom: 1.5rem;
            }
            .kpi-card {
                background-color: #1e1e2f;
                padding: 1.2rem;
                border-radius: 0.8rem;
                border: 1px solid #2e2e4f;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .kpi-val {
                font-size: 2rem;
                font-weight: bold;
                color: #00f2fe;
            }
            .kpi-lbl {
                font-size: 0.85rem;
                color: #a1a1aa;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            /* Styling for confusion matrix */
            .matrix-table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }
            .matrix-cell {
                border: 2px solid #2e2e4f;
                text-align: center;
                padding: 15px;
                font-size: 1.2rem;
                font-weight: bold;
            }
            .matrix-label-row {
                font-weight: bold;
                color: #00f2fe;
                background-color: #1e1e2f;
            }
            .matrix-label-col {
                font-weight: bold;
                color: #00f2fe;
                background-color: #1e1e2f;
                width: 120px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🔋 Battery Anomaly Detection Study Visualizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive results dashboard for DINOv3 + PEFT classification ablation sweeps</div>', unsafe_allow_html=True)

    # Sidebar parameters & data loader
    st.sidebar.markdown("### 📂 Data Settings")
    outputs_dir = st.sidebar.text_input("Outputs Directory", value="outputs")

    # Load results
    with st.spinner("Scanning outputs directory..."):
        df_results = load_results(outputs_dir)

    if df_results.empty:
        st.warning(f"No results found in '{outputs_dir}' directory. Please ensure runs containing `config.yaml` exist under outputs.")
        
        # Display dummy placeholder dashboard if no data is found to demonstrate layout
        st.info("💡 Showing layout placeholder since no output files were found.")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔮 Placeholders & Filters")
        st.sidebar.selectbox("Select Task", ["Classification (Ablation Grid)", "Segmentation (Future)", "Object Detection (Future)"])
        st.sidebar.multiselect("Backbone Models", ["DINOv3-ViT-S/16", "DINOv3-ViT-B/16", "ViT-L/16 (Future)"], default=["DINOv3-ViT-S/16", "DINOv3-ViT-B/16"])
        st.sidebar.multiselect("PEFT Methods", ["LoRA", "Bottleneck Adapters", "VPT", "Prefix Tuning (Future)"], default=["LoRA", "Bottleneck Adapters", "VPT"])
        return

    # Extract unique classes/labels for monitoring
    def get_unique_classes(df):
        classes = set()
        for history in df["history"]:
            for entry in history:
                for key in entry.keys():
                    if key.startswith("eval_custom_cls_f1/"):
                        classes.add(key.split("/")[-1])
                    elif key.startswith("eval_custom_F1/"):
                        classes.add(key.split("/")[-1])
                    elif key.startswith("eval_custom_TP/"):
                        classes.add(key.split("/")[-1])
        return sorted(list(classes))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Classification Metric Type & Class")

    # Check if any run in df_results has custom metrics or normal metrics
    has_custom = False
    has_normal = False
    for history in df_results["history"]:
        for entry in history:
            for key in entry.keys():
                if key.startswith("eval_custom_cls_f1/"):
                    has_custom = True
                if key in ["eval_f1", "eval_accuracy"]:
                    has_normal = True

    metric_modes = []
    if has_custom:
        metric_modes.append("Custom Classification Metrics (by Class)")
    if has_normal:
        metric_modes.append("Normal/Standard Classification Metrics")
    if not metric_modes:
        metric_modes = ["Custom Classification Metrics (by Class)", "Normal/Standard Classification Metrics"]

    selected_mode = st.sidebar.selectbox(
        "Metric Source Mode",
        options=metric_modes,
        index=0
    )

    metric_source_is_custom = "Custom" in selected_mode

    unique_classes = get_unique_classes(df_results)
    if not unique_classes:
        unique_classes = ["abnormal", "cell", "text"]

    if metric_source_is_custom:
        selected_label = st.sidebar.selectbox(
            "Target Class Label",
            options=unique_classes,
            index=unique_classes.index("abnormal") if "abnormal" in unique_classes else 0
        )
    else:
        selected_label = "normal"

    # Benchmark Metric selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Best Epoch Benchmark Selection")
    
    if metric_source_is_custom:
        f1_label = f"F1 Score / Converted {selected_label.capitalize()} F1 (Max)"
    else:
        f1_label = "F1 Score / Standard F1 (Max)"
        
    benchmark_option = st.sidebar.selectbox(
        "Benchmark Metric",
        options=[
            f1_label,
            "Validation Loss (Min)",
            "Validation mAP50 (Max)",
            "Validation Mean Bbox IoU (Max)",
            "Training Loss (Min)"
        ],
        index=0
    )
    
    # Map selection to key and mode
    if benchmark_option.endswith("F1 (Max)"):
        benchmark_metric = "default"
        benchmark_mode = "max"
    elif benchmark_option.startswith("Validation Loss"):
        benchmark_metric = "eval_loss"
        benchmark_mode = "min"
    elif benchmark_option.startswith("Validation mAP50"):
        benchmark_metric = "eval_mAP50"
        benchmark_mode = "max"
    elif benchmark_option.startswith("Validation Mean Bbox IoU"):
        benchmark_metric = "eval_custom_mean_bbox_IoU"
        benchmark_mode = "max"
    elif benchmark_option.startswith("Training Loss"):
        benchmark_metric = "loss"
        benchmark_mode = "min"
        
    # Recalculate metrics in-place for df_results
    update_best_metrics_inplace(df_results, benchmark_metric, benchmark_mode, selected_label=selected_label, use_custom=metric_source_is_custom)

    # Task Profile filter
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filter Experiments")

    task_options = ["All Tasks", "Classification", "Detection"]
    selected_task = st.sidebar.selectbox("Task Profile", task_options, index=0)

    # Extract unique values from data
    unique_models = df_results["model"].unique().tolist() if not df_results.empty else []
    unique_pefts = df_results["peft_type"].unique().tolist() if not df_results.empty else []
    unique_lrs = sorted(df_results["lr"].unique().tolist()) if not df_results.empty else []
    unique_imbs = df_results["imbalance_strategy"].unique().tolist() if not df_results.empty else []
    unique_datasets = sorted(df_results["dataset"].unique().tolist()) if not df_results.empty else []

    # Sidebar Filter Controls (with placeholder options)
    dataset_filter = st.sidebar.multiselect(
        "Datasets",
        options=unique_datasets,
        default=unique_datasets
    )

    model_filter = st.sidebar.multiselect(
        "Backbone Models", 
        options=unique_models + ["facebook/dinov3-vitl16-pretrain (Future)"],
        default=unique_models
    )

    peft_filter = st.sidebar.multiselect(
        "PEFT Methods",
        options=unique_pefts + ["prefix_tuning (Future)", "full_finetune (Future)"],
        default=unique_pefts
    )

    lr_filter = st.sidebar.multiselect(
        "Learning Rates",
        options=unique_lrs + [0.001, 0.005],
        default=unique_lrs
    )

    imb_filter = st.sidebar.multiselect(
        "Imbalance Strategies",
        options=unique_imbs + ["smote_oversampling (Future)", "class_balanced_loss (Future)"],
        default=unique_imbs
    )

    # Run Status selector
    status_filter = st.sidebar.radio("Run Status", ["All Runs", "Completed Only (DONE)", "Incomplete/Active Only"])

    # K-Fold Options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔀 K-Fold Settings")
    group_folds = st.sidebar.checkbox("Average Metrics Over Folds", value=True)

    # Placeholder for future hyperparameter selectors
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Future Hyperparameters (Placeholder)")
    st.sidebar.selectbox("Future Optimizer Type", ["AdamW (Active)", "SGD (Future)", "AdamW-ScheduleFree (Future)"])
    st.sidebar.select_slider("Future Weight Decay", options=["0.01 (Active)", "0.05 (Future)", "0.10 (Future)"])


    # Apply filters to DataFrame
    df_filtered = df_results.copy()
    
    # Filter by dataset
    if dataset_filter:
        df_filtered = df_filtered[df_filtered["dataset"].isin(dataset_filter)]
    
    # Filter by task
    if selected_task == "Classification":
        df_filtered = df_filtered[df_filtered["task"] == "Classification"]
    elif selected_task == "Detection":
        df_filtered = df_filtered[df_filtered["task"] == "Detection"]

    # Filter by model (ignore future placeholders if selected)
    active_models = [m for m in model_filter if m in unique_models]
    if active_models:
        df_filtered = df_filtered[df_filtered["model"].isin(active_models)]
    else:
        df_filtered = df_filtered.iloc[0:0] # empty

    # Filter by PEFT
    active_pefts = [p for p in peft_filter if p in unique_pefts]
    if active_pefts:
        df_filtered = df_filtered[df_filtered["peft_type"].isin(active_pefts)]
    else:
        df_filtered = df_filtered.iloc[0:0]

    # Filter by LR
    active_lrs = [lr for lr in lr_filter if lr in unique_lrs]
    if active_lrs:
        df_filtered = df_filtered[df_filtered["lr"].isin(active_lrs)]
    else:
        df_filtered = df_filtered.iloc[0:0]

    # Filter by Imbalance
    active_imbs = [imb for imb in imb_filter if imb in unique_imbs]
    if active_imbs:
        df_filtered = df_filtered[df_filtered["imbalance_strategy"].isin(active_imbs)]
    else:
        df_filtered = df_filtered.iloc[0:0]

    # Filter by status
    if status_filter == "Completed Only (DONE)":
        df_filtered = df_filtered[df_filtered["completed"] == True]
    elif status_filter == "Incomplete/Active Only":
        df_filtered = df_filtered[df_filtered["completed"] == False]

    # Average folds if selected
    if group_folds:
        df_filtered = group_results_by_fold(df_filtered)

    # Sort filtered runs by best F1 score descending
    if not df_filtered.empty and "img_abnormal_f1" in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by="img_abnormal_f1", ascending=False)


    # ── Render top metrics dashboard ──────────────────────────────────────────
    total_scanned = len(df_results)
    completed_scanned = df_results["completed"].sum()
    active_scanned = total_scanned - completed_scanned
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val">{total_scanned}</div>
                <div class="kpi-lbl">Total Runs Scanned</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val" style="color: #22c55e;">{completed_scanned}</div>
                <div class="kpi-lbl">Completed Runs</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val" style="color: #eab308;">{active_scanned}</div>
                <div class="kpi-lbl">Incomplete / Active</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        best_f1_overall = df_results["img_abnormal_f1"].max() if not df_results.empty else 0.0
        best_run_overall = df_results.loc[df_results["img_abnormal_f1"].idxmax()]["short_cfg_name"] if not df_results.empty and best_f1_overall > 0 else "N/A"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val" style="color: #f43f5e;">{best_f1_overall:.4f}</div>
                <div class="kpi-lbl">Best Image Abn F1 ({best_run_overall[:10]})</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ── Tabs Setup ────────────────────────────────────────────────────────────
    tab_leaderboard, tab_curves, tab_inspector, tab_peft_analysis, tab_comparison = st.tabs([
        "🏆 Leaderboard", 
        "📈 Trajectory Curves", 
        "🔬 Single Run Inspector", 
        "📊 PEFT & Hyperparameter Analysis",
        "⚖️ Classification vs. Detection Comparison"
    ])

    # ── Tab 1: Leaderboard ─────────────────────────────────────────────────────
    with tab_leaderboard:
        st.subheader("🏆 Ablation Experiment Leaderboard")
        st.write("Showing all configurations matching filters. Select which validation metrics to display in the table.")
        
        if df_filtered.empty:
            st.info("No runs match the current filters. Please adjust the sidebar settings.")
        else:
            # Prepare leaderboard DataFrame for clean display
            display_df = df_filtered.copy()
            
            # Map completed boolean to emojis for rich design
            display_df["status"] = display_df["completed"].apply(lambda x: "✅ Completed" if x else "⏳ Active/Interrupted")
            
            # Available metrics choices
            available_metrics_map = {
                "eval_loss": "Validation Loss",
                "eval_f1": "Validation F1 (Classification)",
                "eval_accuracy": "Validation Accuracy (Classification)",
                "eval_auroc": "Validation AUROC (Classification)",
                "eval_mAP50": "Bbox mAP50 (Detection)",
                "eval_mAP50-95": "Bbox mAP50-95 (Detection)",
                "eval_precision": "Bbox/Cls Precision",
                "eval_recall": "Bbox/Cls Recall",
                "eval_custom_mean_bbox_IoU": "Bbox Mean IoU (Detection)",
                "eval_custom_mean_bbox_Dice": "Bbox Mean Dice (Detection)",
                "image_cls_f1": f"Image {selected_label.capitalize()} F1 (Custom)" if metric_source_is_custom else "Image F1 (Standard)",
                "image_cls_auroc": f"Image {selected_label.capitalize()} AUROC (Custom)" if metric_source_is_custom else "Image AUROC (Standard)",
                "image_cls_precision": f"Image {selected_label.capitalize()} Precision (Custom)" if metric_source_is_custom else "Image Precision (Standard)",
                "image_cls_recall": f"Image {selected_label.capitalize()} Recall (Custom)" if metric_source_is_custom else "Image Recall (Standard)"
            }
            
            default_metrics = ["eval_loss", "image_cls_f1", "image_cls_auroc"]
            if selected_task == "Detection" or any(df_filtered["task"] == "Detection"):
                default_metrics.extend(["eval_mAP50", "eval_custom_mean_bbox_IoU", "eval_custom_mean_bbox_Dice"])
            elif selected_task == "Classification" or any(df_filtered["task"] == "Classification"):
                default_metrics.append("eval_f1")
            
            # Keep only metrics that exist in map
            default_metrics = [m for m in default_metrics if m in available_metrics_map]
            
            selected_metrics_keys = st.multiselect(
                "Leaderboard Metrics to Display",
                options=list(available_metrics_map.keys()),
                default=default_metrics,
                format_func=lambda x: available_metrics_map[x]
            )
            
            # Sort controls
            col_sort_1, col_sort_2 = st.columns(2)
            with col_sort_1:
                sort_options = ["Configuration", "Dataset", "Total Params", "Trainable Params", "% Trainable"] + [available_metrics_map[m] for m in selected_metrics_keys]
                default_sort_val = available_metrics_map["image_cls_f1"] if "image_cls_f1" in selected_metrics_keys else sort_options[0]
                default_sort_idx = sort_options.index(default_sort_val) if default_sort_val in sort_options else 0
                sort_by_col = st.selectbox(
                    "Sort Leaderboard By",
                    options=sort_options,
                    index=default_sort_idx
                )
            with col_sort_2:
                sort_direction = st.radio(
                    "Sort Order",
                    options=["Descending", "Ascending"],
                    index=0,
                    horizontal=True
                )
            
            # Map clean name back to raw key for proper numerical sorting
            reverse_rename_map = {
                "Configuration": "short_cfg_name",
                "Dataset": "dataset",
                "Total Params": "total_params",
                "Trainable Params": "trainable_params",
                "% Trainable": "pct_trainable"
            }
            reverse_metrics_map = {v: k for k, v in available_metrics_map.items()}
            
            raw_sort_key = None
            if sort_by_col in reverse_rename_map:
                raw_sort_key = reverse_rename_map[sort_by_col]
            elif sort_by_col in reverse_metrics_map:
                raw_sort_key = reverse_metrics_map[sort_by_col]
                
            if raw_sort_key and raw_sort_key in display_df.columns:
                display_df = display_df.sort_values(by=raw_sort_key, ascending=(sort_direction == "Ascending"))
                
            # Parameter formatting
            display_df["Total Params Formatted"] = display_df["total_params"].map(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
            display_df["Trainable Params Formatted"] = display_df["trainable_params"].map(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
            display_df["% Trainable Formatted"] = display_df["pct_trainable"].map(lambda x: f"{x:.4f}%" if pd.notna(x) else "N/A")
            display_df["LR"] = display_df["lr"].map(lambda x: f"{x:.5f}")
            
            # Base columns (always present)
            leaderboard_cols = [
                "short_cfg_name", "task", "dataset", "model", "peft_type", "peft_detail", 
                "imbalance_strategy", "LR", "Total Params Formatted", "Trainable Params Formatted", "% Trainable Formatted"
            ]
            
            rename_map = {
                "short_cfg_name": "Configuration",
                "task": "Task",
                "dataset": "Dataset",
                "model": "Backbone",
                "peft_type": "PEFT Type",
                "peft_detail": "PEFT Hyperparams",
                "imbalance_strategy": "Imbalance Strategy",
                "Total Params Formatted": "Total Params",
                "Trainable Params Formatted": "Trainable Params",
                "% Trainable Formatted": "% Trainable",
                "status": "Status"
            }
            
            # Map selected metrics into display_df and leaderboard_cols
            for metric_key in selected_metrics_keys:
                clean_name = available_metrics_map[metric_key]
                display_df[clean_name] = display_df[metric_key].map(lambda x: f"{x:.5f}" if pd.notna(x) else "N/A")
                leaderboard_cols.append(clean_name)
                
            # Add final train loss and status at the end
            display_df["Final Train Loss"] = display_df["final_train_loss"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "N/A")
            leaderboard_cols.extend(["Final Train Loss", "status"])
            
            renamed_df = display_df[leaderboard_cols].rename(columns=rename_map)
            
            # Highlight max values for F1/mAP/accuracy columns dynamically
            def highlight_max_metric(s):
                is_metric = any(term in s.name for term in ["F1", "mAP", "IoU", "Dice", "Accuracy", "AUROC", "Recall", "Precision"]) and "Loss" not in s.name
                if is_metric:
                    numeric_vals = pd.to_numeric(s, errors='coerce')
                    is_max = numeric_vals == numeric_vals.max()
                    return ['background-color: rgba(79, 172, 254, 0.25)' if v else '' for v in is_max]
                return [''] * len(s)
            
            st.dataframe(
                renamed_df.style.apply(highlight_max_metric),
                use_container_width=True
            )

    # ── Tab 2: Trajectory Curves ──────────────────────────────────────────────
    with tab_curves:
        st.subheader("📈 Multi-Run Training Trajectories")
        st.write("Select runs from the leaderboard to plot and compare their metric trajectories side-by-side.")
        
        if df_filtered.empty:
            st.info("No runs available to plot.")
        else:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Select multiple runs to compare
                run_mapping = {row["short_cfg_name"]: idx for idx, row in df_filtered.iterrows()}
                selected_run_names = st.multiselect(
                    "Compare Runs", 
                    options=list(run_mapping.keys()),
                    default=list(run_mapping.keys())[:min(3, len(run_mapping))]
                )
                
                selected_indices = [run_mapping[name] for name in selected_run_names]
                
                # Gather all metric keys available in the selected runs' history
                available_metrics = set()
                for idx in selected_indices:
                    run = df_results.loc[idx]
                    for entry in run["history"]:
                        for k in entry.keys():
                            if k not in ("epoch", "step", "learning_rate"):
                                available_metrics.add(k)
                                
                metric_options = sorted(list(available_metrics))
                # Put common metrics first if they exist
                preferred_order = ["eval_f1", f"eval_custom_cls_f1/{selected_label}", "eval_mAP50", "eval_loss", "loss", "eval_custom_mean_bbox_IoU"]
                metric_options = [m for m in preferred_order if m in metric_options] + [m for m in metric_options if m not in preferred_order]
                
                # Dynamic format dictionary
                format_dict = {
                    "eval_f1": "Validation F1 Score (Cls)",
                    "eval_mAP50": "Validation mAP50 (Det Box)",
                    "eval_mAP50-95": "Validation mAP50-95 (Det Box)",
                    "eval_custom_mean_bbox_IoU": "Mean Bbox IoU (Det Box)",
                    "eval_custom_mean_bbox_Dice": "Mean Bbox Dice (Det Box)",
                    "eval_loss": "Validation Loss",
                    "loss": "Training Loss",
                    "eval_accuracy": "Validation Accuracy (Cls)",
                    "eval_auroc": "Validation AUROC (Cls)",
                    "eval_precision": "Validation Precision (Cls)",
                    "eval_recall": "Validation Recall (Cls)"
                }
                # Add label specific metrics dynamically
                for l in unique_classes:
                    format_dict[f"eval_custom_cls_f1/{l}"] = f"Converted Image-Level {l.capitalize()} F1 (Det)"
                    format_dict[f"eval_custom_cls_auroc/{l}"] = f"Converted Image-Level {l.capitalize()} AUROC (Det)"
                    format_dict[f"eval_custom_cls_precision/{l}"] = f"Converted Image-Level {l.capitalize()} Precision (Det)"
                    format_dict[f"eval_custom_cls_recall/{l}"] = f"Converted Image-Level {l.capitalize()} Recall (Det)"

                plot_metric = st.selectbox(
                    "Select Metric to Compare",
                    options=metric_options,
                    format_func=lambda x: format_dict.get(x, x)
                )
                
                # Checkbox to isolate metric up to the best epoch
                truncate_at_best = st.checkbox("Truncate curves at Best Epoch", value=False)
            
            with col2:
                if not selected_indices:
                    st.warning("Please select at least one run from the sidebar list.")
                else:
                    fig = go.Figure()
                    
                    for idx in selected_indices:
                        run = df_results.loc[idx]
                        history = run["history"]
                        
                        if not history:
                            continue
                            
                        epochs = []
                        values = []
                        
                        # Find best epoch if truncation is requested
                        best_ep = float('inf')
                        if truncate_at_best and run["best_metrics"]:
                            best_ep = run["best_metrics"].get("epoch", float('inf'))
                            
                        for log_entry in history:
                            if plot_metric in log_entry and "epoch" in log_entry:
                                ep = log_entry["epoch"]
                                if ep <= best_ep:
                                    epochs.append(ep)
                                    values.append(log_entry[plot_metric])
                                    
                        if epochs:
                            # Sort by epoch to guarantee line continuity
                            sort_idx = np.argsort(epochs)
                            epochs = np.array(epochs)[sort_idx]
                            values = np.array(values)[sort_idx]
                            
                            label = f"{run['peft_type']} ({run['peft_detail']}) | ds={run['dataset']} | lr={run['lr']} | {run['short_cfg_name']}"
                            fig.add_trace(go.Scatter(
                                x=epochs,
                                y=values,
                                mode="lines+markers",
                                name=label,
                                line=dict(width=2),
                                marker=dict(size=4)
                            ))
                            
                    fig.update_layout(
                        title=f"{plot_metric.replace('eval_', 'Validation ').capitalize()} curves over training epochs",
                        xaxis_title="Epoch",
                        yaxis_title=plot_metric,
                        template="plotly_dark",
                        hovermode="x unified",
                        height=500,
                        legend=dict(yanchor="top", y=-0.2, xanchor="left", x=0.0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Single Run Inspector ───────────────────────────────────────────
    with tab_inspector:
        st.subheader("🔬 Single Run Detailed Diagnostics")
        st.write("Inspect hyperparameters, classification confusion matrices, and detailed metrics for a single selected run.")
        
        if df_filtered.empty:
            st.info("No runs available to inspect.")
        else:
            # Select run to inspect
            selected_run_name = st.selectbox(
                "Run to Inspect", 
                options=df_filtered["short_cfg_name"].tolist()
            )
            
            run_idx = df_filtered[df_filtered["short_cfg_name"] == selected_run_name].index[0]
            run_data = df_results.loc[run_idx]
            
            # Grid layout for detail panels
            col_info, col_metrics = st.columns([1, 1])
            
            with col_info:
                st.markdown("#### ⚙️ Configuration & Hyperparameters")
                st.markdown(f"**Run Path**: `{run_data['dir']}`")
                
                # Show config parameters in a clean table
                st.table(pd.DataFrame({
                    "Parameter": [
                        "Dataset", "Backbone Model", "PEFT Method", "PEFT Details", 
                        "Learning Rate", "Imbalance Strategy", "Loss Type", "Epochs Configured",
                        "Total Parameters", "Trainable Parameters", "% Trainable"
                    ],
                    "Value": [
                        run_data["dataset"],
                        run_data["model"],
                        run_data["peft_type"],
                        run_data["peft_detail"],
                        f"{run_data['lr']:.5f}",
                        run_data["imbalance_strategy"],
                        run_data["loss_type"],
                        str(run_data["epochs_configured"]),
                        f"{run_data['total_params']:,}" if pd.notna(run_data['total_params']) else "N/A",
                        f"{run_data['trainable_params']:,}" if pd.notna(run_data['trainable_params']) else "N/A",
                        f"{run_data['pct_trainable']:.4f}%" if pd.notna(run_data['pct_trainable']) else "N/A"
                    ]
                }))
                
                # Status flag
                if run_data["completed"]:
                    st.success("✅ Training completed successfully (DONE file verified).")
                else:
                    st.warning("⏳ Training in progress or interrupted (latest checkpoint parsed).")
            
            with col_metrics:
                best_metrics = run_data["best_metrics"]
                if not best_metrics:
                    st.info("No evaluation logs found for this run yet.")
                else:
                    if run_data["task"] == "Detection":
                        st.markdown("#### 📊 Best Bbox Detection Metrics (IoU=0.50:0.95)")
                        subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                        with subcol1:
                            st.metric("mAP50", f"{best_metrics.get('eval_mAP50', 0.0):.4f}")
                        with subcol2:
                            st.metric("mAP50-95", f"{best_metrics.get('eval_mAP50-95', 0.0):.4f}")
                        with subcol3:
                            st.metric("Box Precision", f"{best_metrics.get('eval_precision', 0.0):.4f}")
                        with subcol4:
                            st.metric("Box Recall", f"{best_metrics.get('eval_recall', 0.0):.4f}")

                        st.markdown("#### 📐 Custom Box Matching Metrics")
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.metric("Mean Bbox IoU", f"{best_metrics.get('eval_custom_mean_bbox_IoU', 0.0):.4f}")
                        with subcol2:
                            st.metric("Mean Bbox Dice", f"{best_metrics.get('eval_custom_mean_bbox_Dice', 0.0):.4f}")
                        with subcol3:
                            st.metric("Val Loss", f"{best_metrics.get('eval_loss', 0.0):.4f}")

                        run_custom_classes = []
                        for key in best_metrics.keys():
                            if key.startswith("eval_custom_cls_f1/"):
                                run_custom_classes.append(key.split("/")[-1])
                        
                        if run_custom_classes:
                            st.markdown("#### 🖥️ Converted Image-Level Classification")
                            cls_data = []
                            for c in run_custom_classes:
                                cls_data.append({
                                    "Class Label": c,
                                    "Image-Level F1": f"{best_metrics.get(f'eval_custom_cls_f1/{c}', 0.0):.4f}",
                                    "Image-Level AUROC": f"{best_metrics.get(f'eval_custom_cls_auroc/{c}', 0.0):.4f}",
                                    "Image-Level Accuracy": f"{best_metrics.get(f'eval_custom_cls_accuracy/{c}', 0.0):.4f}",
                                    "Image-Level Precision": f"{best_metrics.get(f'eval_custom_cls_precision/{c}', 0.0):.4f}",
                                    "Image-Level Recall": f"{best_metrics.get(f'eval_custom_cls_recall/{c}', 0.0):.4f}",
                                })
                            st.dataframe(pd.DataFrame(cls_data), use_container_width=True)
                            
                            # Per-class box metrics table
                            st.markdown("#### 📦 Bbox Metrics Per-Class")
                            class_rows = []
                            abnormal_name = run_data.get("abnormal_class_name", "abnormal")
                            classes_to_check = []
                            for c in ["abnormal", abnormal_name, "cell", "text"]:
                                if c not in classes_to_check:
                                    classes_to_check.append(c)
                            for c_name in classes_to_check:
                                tp_key = f"eval_custom_TP/{c_name}"
                                if tp_key in best_metrics:
                                    class_rows.append({
                                        "Class": c_name,
                                        "TP": int(best_metrics.get(f"eval_custom_TP/{c_name}", 0)),
                                        "FP": int(best_metrics.get(f"eval_custom_FP/{c_name}", 0)),
                                        "FN": int(best_metrics.get(f"eval_custom_FN/{c_name}", 0)),
                                        "Precision": f"{best_metrics.get(f'eval_custom_P/{c_name}', 0.0):.4f}",
                                        "Recall": f"{best_metrics.get(f'eval_custom_R/{c_name}', 0.0):.4f}",
                                        "F1": f"{best_metrics.get(f'eval_custom_F1/{c_name}', 0.0):.4f}",
                                        "mAP50": f"{best_metrics.get(f'eval_custom_mAP50/{c_name}', 0.0):.4f}",
                                        "mAP50-95": f"{best_metrics.get(f'eval_custom_mAP50-95/{c_name}', 0.0):.4f}",
                                    })
                            if class_rows:
                                st.dataframe(pd.DataFrame(class_rows), use_container_width=True)
                            
                            # Let them choose which class to show the Confusion Matrix for
                            selected_cm_class = st.selectbox(
                                "Select Class Label for Confusion Matrix",
                                options=run_custom_classes,
                                index=run_custom_classes.index(selected_label) if selected_label in run_custom_classes else 0,
                                key="cm_class_selector"
                            )
                            tp = best_metrics.get(f"eval_custom_cls_tp/{selected_cm_class}")
                            fp = best_metrics.get(f"eval_custom_cls_fp/{selected_cm_class}")
                            tn = best_metrics.get(f"eval_custom_cls_tn/{selected_cm_class}")
                            fn = best_metrics.get(f"eval_custom_cls_fn/{selected_cm_class}")
                            cm_title = f"Converted {selected_cm_class.capitalize()} Confusion Matrix"
                        else:
                            tp = fp = tn = fn = None
                            cm_title = ""
                    else:
                        st.markdown("#### 📊 Best Validation Metrics")
                        st.write(f"The following metrics were achieved at the best epoch (**Epoch {best_metrics.get('epoch', 'N/A')}**):")
                        
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.metric("Eval F1 Score", f"{best_metrics.get('eval_f1', 0.0):.4f}")
                            st.metric("Eval Precision", f"{best_metrics.get('eval_precision', 0.0):.4f}")
                        with subcol2:
                            st.metric("Eval Loss", f"{best_metrics.get('eval_loss', 0.0):.4f}")
                            st.metric("Eval Recall", f"{best_metrics.get('eval_recall', 0.0):.4f}")
                        with subcol3:
                            st.metric("Eval Accuracy", f"{best_metrics.get('eval_accuracy', 0.0):.4f}")
                            st.metric("Eval AUROC", f"{best_metrics.get('eval_auroc', 0.0):.4f}")
                            
                        # Confusion Matrix calculation
                        tp = best_metrics.get("eval_tp")
                        fp = best_metrics.get("eval_fp")
                        tn = best_metrics.get("eval_tn")
                        fn = best_metrics.get("eval_fn")
                        cm_title = "Confusion Matrix"
                    
                    if all(v is not None for v in [tp, fp, tn, fn]):
                        st.markdown(f"#### 🧮 {cm_title} (Best Epoch)")
                        
                        # Create interactive Plotly Heatmap for confusion matrix
                        z = [[tn, fp], [fn, tp]]
                        x = ["Predicted Normal", "Predicted Abnormal"]
                        y = ["Actual Normal", "Actual Abnormal"]
                        
                        fig_cm = px.imshow(
                            z, x=x, y=y,
                            color_continuous_scale="Blues",
                            aspect="auto",
                            text_auto=True,
                            title=cm_title
                        )
                        fig_cm.update_layout(
                            coloraxis_showscale=False,
                            width=380,
                            height=250,
                            margin=dict(l=10, r=10, t=40, b=10),
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_cm, use_container_width=False)
                        
                        # Alternative HTML layout in case Plotly fails
                        with st.expander("Show Matrix Details (Raw numbers)"):
                            st.write(f"**True Positives (TP)**: {tp} | **True Negatives (TN)**: {tn}")
                            st.write(f"**False Positives (FP)**: {fp} | **False Negatives (FN)**: {fn}")
                            
            # Render a table of history logs
            if run_data["history"]:
                st.markdown("#### 📜 Full Epoch Trajectory History")
                hist_df = pd.DataFrame(run_data["history"])
                
                # Filter down to display columns
                cols_to_display = [
                    "epoch", "step", "loss", "eval_loss", "eval_f1", "eval_accuracy", 
                    "eval_auroc", "eval_precision", "eval_recall", "eval_mAP50", 
                    "eval_custom_mean_bbox_IoU", "eval_custom_mean_bbox_Dice"
                ]
                cols_present = [c for c in cols_to_display if c in hist_df.columns]
                
                # Clean history to only show validation epochs
                val_cols = [c for c in cols_present if c.startswith("eval_")]
                if val_cols:
                    hist_df_clean = hist_df.dropna(subset=val_cols, how="all")
                else:
                    hist_df_clean = hist_df
                
                st.dataframe(
                    hist_df_clean[cols_present].sort_values(by="epoch"),
                    use_container_width=True
                )

    # ── Tab 4: PEFT & Hyperparameter Analysis ──────────────────────────────────
    with tab_peft_analysis:
        st.subheader("📊 PEFT Methods & Hyperparameter Sweeps Comparison")
        st.write("Compare performance aggregated across backbones, PEFT configurations, and training parameters.")
        
        if df_results.empty:
            st.info("No runs available for aggregated comparisons.")
        else:
            # Metric selector for hyperparameter sweeps
            available_sweep_metrics = {
                "image_cls_f1": f"Image {selected_label.capitalize()} F1 (Custom)" if metric_source_is_custom else "Image F1 (Standard)",
                "image_cls_auroc": f"Image {selected_label.capitalize()} AUROC (Custom)" if metric_source_is_custom else "Image AUROC (Standard)",
                "eval_loss": "Validation Loss",
                "eval_mAP50": "Bbox mAP50 (Detection)",
                "eval_custom_mean_bbox_IoU": "Bbox Mean IoU (Detection)",
                "eval_custom_mean_bbox_Dice": "Bbox Mean Dice (Detection)",
                "eval_accuracy": "Validation Accuracy (Classification)",
                "eval_precision": "Bbox/Cls Precision",
                "eval_recall": "Bbox/Cls Recall"
            }
            
            # Filter options to only include columns that exist and have non-null values
            valid_sweep_metrics = {}
            for k, name in available_sweep_metrics.items():
                if k in df_filtered.columns and df_filtered[k].notna().any():
                    valid_sweep_metrics[k] = name
                    
            if not valid_sweep_metrics:
                valid_sweep_metrics = {"image_cls_f1": "Image F1"}
                
            selected_sweep_metric_key = st.selectbox(
                "Select Metric for Hyperparameter Sweep Analysis",
                options=list(valid_sweep_metrics.keys()),
                format_func=lambda x: valid_sweep_metrics[x],
                index=0,
                key="sweep_metric_selector"
            )
            
            sweep_metric_name = valid_sweep_metrics[selected_sweep_metric_key]
            is_loss = "loss" in selected_sweep_metric_key.lower()
            agg_fn = "min" if is_loss else "max"
            
            col_peft, col_lr, col_ds = st.columns(3)
            
            with col_peft:
                st.markdown(f"#### ⚙️ {agg_fn.capitalize()} {sweep_metric_name} by PEFT Type")
                peft_summary = df_filtered.groupby("peft_type")[selected_sweep_metric_key].agg(agg_fn).reset_index()
                
                fig_peft = px.bar(
                    peft_summary, 
                    x="peft_type", 
                    y=selected_sweep_metric_key,
                    color="peft_type",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    labels={"peft_type": "PEFT Method", selected_sweep_metric_key: sweep_metric_name},
                    title=f"PEFT Performance ({sweep_metric_name})"
                )
                fig_peft.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_peft, use_container_width=True)
                
            with col_lr:
                st.markdown(f"#### 📈 {agg_fn.capitalize()} {sweep_metric_name} by Learning Rate")
                lr_summary = df_filtered.groupby("lr")[selected_sweep_metric_key].agg(agg_fn).reset_index()
                lr_summary["lr"] = lr_summary["lr"].astype(str)
                
                fig_lr = px.bar(
                    lr_summary, 
                    x="lr", 
                    y=selected_sweep_metric_key,
                    color="lr",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    labels={"lr": "Learning Rate", selected_sweep_metric_key: sweep_metric_name},
                    title=f"Learning Rate Performance ({sweep_metric_name})"
                )
                fig_lr.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_lr, use_container_width=True)

            with col_ds:
                st.markdown(f"#### 📁 {agg_fn.capitalize()} {sweep_metric_name} by Dataset")
                dataset_summary = df_filtered.groupby("dataset")[selected_sweep_metric_key].agg(agg_fn).reset_index()
                
                fig_dataset = px.bar(
                    dataset_summary,
                    x="dataset",
                    y=selected_sweep_metric_key,
                    color="dataset",
                    color_discrete_sequence=px.colors.qualitative.Pastel1,
                    labels={"dataset": "Dataset", selected_sweep_metric_key: sweep_metric_name},
                    title=f"Dataset Performance ({sweep_metric_name})"
                )
                fig_dataset.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_dataset, use_container_width=True)
                
            st.divider()
            
            # Parallel Coordinates Plot for Numerical Hyperparameters
            st.markdown("#### 🕸️ Parallel Hyperparameter Trajectory")
            st.write(f"Visualize how combinations of numerical parameters (learning rate, rank/bottleneck size, parameter counts, validation {sweep_metric_name}) stack together.")
            
            # Map PEFT sizes into numerical column
            coord_df = df_filtered.copy()
            coord_df = coord_df.dropna(subset=[selected_sweep_metric_key])
            
            def get_peft_size_num(row):
                detail = row["peft_detail"]
                if row["peft_type"] == "lora":
                    try: return float(detail.split("r=")[-1])
                    except: return 0.0
                elif row["peft_type"] == "adapter":
                    try: return float(detail.split("d=")[-1])
                    except: return 0.0
                elif row["peft_type"] == "visual_prompt":
                    try: return float(detail.split("t=")[-1])
                    except: return 0.0
                return 0.0
                
            coord_df["peft_hyperparam_size"] = coord_df.apply(get_peft_size_num, axis=1)
            
            # Numeric column filter
            numeric_cols = ["lr", "peft_hyperparam_size", "total_params", "trainable_params", selected_sweep_metric_key]
            if "epochs_configured" in coord_df.columns:
                numeric_cols.insert(-1, "epochs_configured")
                
            fig_par = px.parallel_coordinates(
                coord_df,
                dimensions=numeric_cols,
                color=selected_sweep_metric_key,
                color_continuous_scale=px.colors.diverging.Tealrose,
                labels={
                    "lr": "Learning Rate",
                    "peft_hyperparam_size": "PEFT Size (Rank/Dim/Token)",
                    "total_params": "Total Params",
                    "trainable_params": "Trainable Params",
                    selected_sweep_metric_key: sweep_metric_name,
                    "epochs_configured": "Epochs"
                }
            )
            fig_par.update_layout(template="plotly_dark")
            st.plotly_chart(fig_par, use_container_width=True)

            st.divider()
            st.markdown("#### ⚖️ Parameter-Performance Trade-off Analysis")
            st.write(f"Examine how model parameter counts affect final validation performance. Efficient models should achieve high {sweep_metric_name} with fewer trainable parameters.")
            
            fig_scatter = px.scatter(
                df_filtered,
                x="trainable_params",
                y=selected_sweep_metric_key,
                color="peft_type",
                size="total_params",
                hover_name="short_cfg_name",
                hover_data=["model", "lr", "dataset"],
                labels={
                    "trainable_params": "Trainable Parameters",
                    selected_sweep_metric_key: sweep_metric_name,
                    "peft_type": "PEFT Type",
                    "total_params": "Total Parameters"
                },
                title=f"{sweep_metric_name} vs. Trainable Parameters Trade-off (Size corresponds to Total Parameters)"
            )
            fig_scatter.update_layout(template="plotly_dark", xaxis_type="log")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab 5: Classification vs. Detection Comparison ──────────────────────────
    with tab_comparison:
        st.subheader("⚖️ Classification vs. Detection Model Comparison")
        
        comp_label = selected_label if selected_label in ["abnormal", "text"] else "abnormal"
        st.write(f"Compare the classification models directly with the detection models on the target task: **image-level {comp_label} classification**.")
        
        if df_results.empty:
            st.info("No runs available for comparison.")
        else:
            # Filter to get classification and detection runs
            cls_runs = df_results[df_results["task"] == "Classification"]
            det_runs = df_results[df_results["task"] == "Detection"]
            
            if cls_runs.empty or det_runs.empty:
                st.info("To see comparative charts, make sure you have at least one completed run for both Classification and Detection tasks.")
            else:
                best_cls = cls_runs.loc[cls_runs["img_abnormal_f1"].idxmax()]
                best_det = det_runs.loc[det_runs["img_abnormal_f1"].idxmax()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🏆 Best Classification Model")
                    st.write(f"**Configuration**: {best_cls['short_cfg_name']}")
                    st.write(f"**Model**: {best_cls['model']}")
                    st.write(f"**PEFT Type**: {best_cls['peft_type']} ({best_cls['peft_detail']})")
                    
                    st.metric("Image Abnormal F1", f"{best_cls['img_abnormal_f1']:.4f}")
                    st.metric("Image Abnormal AUROC", f"{best_cls['img_abnormal_auroc']:.4f}")
                    
                    # Confusion Matrix
                    bm_cls = best_cls["best_metrics"]
                    tp_c, fp_c, tn_c, fn_c = bm_cls.get("eval_tp"), bm_cls.get("eval_fp"), bm_cls.get("eval_tn"), bm_cls.get("eval_fn")
                    if all(v is not None for v in [tp_c, fp_c, tn_c, fn_c]):
                        fig_cm_cls = px.imshow(
                            [[tn_c, fp_c], [fn_c, tp_c]], x=["Predicted Normal", "Predicted Abnormal"], y=["Actual Normal", "Actual Abnormal"],
                            color_continuous_scale="Greens", aspect="auto", text_auto=True, title="Best Classification Confusion Matrix"
                        )
                        fig_cm_cls.update_layout(coloraxis_showscale=False, width=350, height=220, template="plotly_dark")
                        st.plotly_chart(fig_cm_cls, use_container_width=False)
                        
                with col2:
                    st.markdown(f"### 🔍 Best Detection Model (Image-Level {comp_label.capitalize()} Conversion)")
                    st.write(f"**Configuration**: {best_det['short_cfg_name']}")
                    st.write(f"**Model**: {best_det['model']}")
                    st.write(f"**PEFT Type**: {best_det['peft_type']} ({best_det['peft_detail']})")
                    
                    st.metric(f"Image {comp_label.capitalize()} F1", f"{best_det['img_abnormal_f1']:.4f}")
                    st.metric(f"Image {comp_label.capitalize()} AUROC", f"{best_det['img_abnormal_auroc']:.4f}")
                    
                    # Confusion Matrix
                    bm_det = best_det["best_metrics"]
                    tp_d = bm_det.get(f"eval_custom_cls_tp/{comp_label}") or bm_det.get("eval_custom_cls_tp/abnormal")
                    fp_d = bm_det.get(f"eval_custom_cls_fp/{comp_label}") or bm_det.get("eval_custom_cls_fp/abnormal")
                    tn_d = bm_det.get(f"eval_custom_cls_tn/{comp_label}") or bm_det.get("eval_custom_cls_tn/abnormal")
                    fn_d = bm_det.get(f"eval_custom_cls_fn/{comp_label}") or bm_det.get("eval_custom_cls_fn/abnormal")
                    if all(v is not None for v in [tp_d, fp_d, tn_d, fn_d]):
                        fig_cm_det = px.imshow(
                            [[tn_d, fp_d], [fn_d, tp_d]], x=["Predicted Normal", "Predicted Abnormal"], y=["Actual Normal", "Actual Abnormal"],
                            color_continuous_scale="Oranges", aspect="auto", text_auto=True, title=f"Best Converted Detection {comp_label.capitalize()} Confusion Matrix"
                        )
                        fig_cm_det.update_layout(coloraxis_showscale=False, width=350, height=220, template="plotly_dark")
                        st.plotly_chart(fig_cm_det, use_container_width=False)
                        
                # Summary bar plot
                st.divider()
                st.markdown("### 📊 Performance Metrics Comparison")
                comp_data = pd.DataFrame({
                    "Task": ["Classification", "Classification", "Detection", "Detection"],
                    "Metric": ["F1 Score", "AUROC", "F1 Score", "AUROC"],
                    "Value": [
                        float(best_cls['img_abnormal_f1']), 
                        float(best_cls['img_abnormal_auroc']), 
                        float(best_det['img_abnormal_f1']), 
                        float(best_det['img_abnormal_auroc'])
                    ]
                })
                fig_comp = px.bar(
                    comp_data, x="Metric", y="Value", color="Task", barmode="group",
                    color_discrete_sequence=["#22c55e", "#ff7f0e"], title=f"Classification vs. Converted Detection {comp_label.capitalize()} Performance",
                    text_auto=".4f"
                )
                fig_comp.update_layout(template="plotly_dark", yaxis_range=[0.0, 1.05])
                st.plotly_chart(fig_comp, use_container_width=True)

if __name__ == "__main__":
    main()
