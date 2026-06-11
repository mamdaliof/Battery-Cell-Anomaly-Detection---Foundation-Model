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

def load_results(base_path="outputs"):
    """
    Recursively scans base_path to load config.yaml and trainer_state.json files.
    """
    runs_data = []
    base_path_obj = Path(base_path)
    
    if not base_path_obj.exists():
        return pd.DataFrame()

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
                        eval_steps = [item for item in history if "eval_mAP50" in item or "eval_custom_cls_f1/abnormality" in item]
                    else:
                        eval_steps = [item for item in history if "eval_f1" in item]
                        
                    train_losses = [item.get("loss") for item in history if "loss" in item]
                    
                    if eval_steps:
                        if is_det:
                            # Prioritize abnormality classification conversion F1, fallback to mAP50
                            best_step = max(
                                eval_steps,
                                key=lambda x: (
                                    x.get("eval_custom_cls_f1/abnormality", 0.0) or x.get("eval_mAP50", 0.0),
                                    -x.get("eval_loss", float('inf'))
                                )
                            )
                            best_eval_f1 = best_step.get("eval_custom_cls_f1/abnormality", 0.0)
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
                    img_f1 = best_epoch_metrics.get("eval_custom_cls_f1/abnormality", 0.0)
                    img_auroc = best_epoch_metrics.get("eval_custom_cls_auroc/abnormality", 0.5)
                else:
                    img_f1 = best_epoch_metrics.get("eval_f1", 0.0)
                    img_auroc = best_epoch_metrics.get("eval_auroc", 0.5)

            # Calculate short directory name for display
            display_name = run_dir.parent.name if run_dir.parent.name != base_path_obj.name else run_dir.name
            # If the path looks like task__model__cfg_stem, extract the cfg_stem for easier reading
            parts = display_name.split("__")
            short_cfg_name = parts[-1] if len(parts) > 1 else display_name
            
            runs_data.append({
                "dir": str(run_dir.relative_to(base_path_obj)),
                "display_name": display_name,
                "short_cfg_name": short_cfg_name,
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
                "img_abnormality_f1": img_f1,
                "img_abnormality_auroc": img_auroc,
                "abnormal_class_name": abnormal_class_name
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
        for fallback in ["eval_f1", "eval_custom_cls_f1/abnormality", "eval_loss", "loss"]:
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

def update_best_metrics_inplace(df: pd.DataFrame, benchmark_metric: str, mode: str):
    if df.empty:
        return
    
    # We will update: best_eval_f1, best_eval_loss, best_metrics, img_abnormality_f1, img_abnormality_auroc
    for idx, row in df.iterrows():
        history = row["history"]
        is_det = row["task"] == "Detection"
        
        # Resolve dynamic benchmark key
        target_metric = benchmark_metric
        target_mode = mode
        
        if benchmark_metric == "default":
            target_metric = "eval_custom_cls_f1/abnormality" if is_det else "eval_f1"
            target_mode = "max"
            
        best_step = get_best_epoch_metrics(history, target_metric, target_mode)
        
        # Write updated metrics to the row
        df.at[idx, "best_metrics"] = best_step
        if best_step:
            df.at[idx, "best_eval_f1"] = best_step.get("eval_f1", 0.0) if not is_det else best_step.get("eval_custom_cls_f1/abnormality", 0.0)
            
            # Handle float conversions safely
            val_loss = best_step.get("eval_loss", None)
            df.at[idx, "best_eval_loss"] = float(val_loss) if val_loss is not None else None
            
            df.at[idx, "img_abnormality_f1"] = best_step.get("eval_custom_cls_f1/abnormality", 0.0) if is_det else best_step.get("eval_f1", 0.0)
            df.at[idx, "img_abnormality_auroc"] = best_step.get("eval_custom_cls_auroc/abnormality", 0.5) if is_det else best_step.get("eval_auroc", 0.5)
        else:
            df.at[idx, "best_eval_f1"] = 0.0
            df.at[idx, "best_eval_loss"] = None
            df.at[idx, "img_abnormality_f1"] = 0.0
            df.at[idx, "img_abnormality_auroc"] = 0.5

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

    # Benchmark Metric selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Best Epoch Benchmark Selection")
    benchmark_option = st.sidebar.selectbox(
        "Benchmark Metric",
        options=[
            "F1 Score / Converted Abnormality F1 (Max)",
            "Validation Loss (Min)",
            "Validation mAP50 (Max)",
            "Validation Mean Bbox IoU (Max)",
            "Training Loss (Min)"
        ],
        index=0
    )
    
    # Map selection to key and mode
    if benchmark_option.startswith("F1 Score"):
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
    update_best_metrics_inplace(df_results, benchmark_metric, benchmark_mode)

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

    # Sidebar Filter Controls (with placeholder options)
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

    # Placeholder for future hyperparameter selectors
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Future Hyperparameters (Placeholder)")
    st.sidebar.selectbox("Future Optimizer Type", ["AdamW (Active)", "SGD (Future)", "AdamW-ScheduleFree (Future)"])
    st.sidebar.select_slider("Future Weight Decay", options=["0.01 (Active)", "0.05 (Future)", "0.10 (Future)"])

    # Apply filters to DataFrame
    df_filtered = df_results.copy()
    
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

    # Sort filtered runs by best F1 score descending
    df_filtered = df_filtered.sort_values(by="img_abnormality_f1", ascending=False)

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
        best_f1_overall = df_results["img_abnormality_f1"].max() if not df_results.empty else 0.0
        best_run_overall = df_results.loc[df_results["img_abnormality_f1"].idxmax()]["short_cfg_name"] if not df_results.empty and best_f1_overall > 0 else "N/A"
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
        st.write("Showing all configurations matching filters. Sort by any column, prioritized by **Image-Level Abnormality F1 Score**.")
        
        if df_filtered.empty:
            st.info("No runs match the current filters. Please adjust the sidebar settings.")
        else:
            # Prepare leaderboard DataFrame for clean display
            display_df = df_filtered.copy()
            
            # Map completed boolean to emojis for rich design
            display_df["status"] = display_df["completed"].apply(lambda x: "✅ Completed" if x else "⏳ Active/Interrupted")
            
            # Formatted column values for presentation
            display_df["Task Metric (F1/mAP50)"] = display_df.apply(
                lambda r: f"{r['best_metrics'].get('eval_mAP50', 0.0):.5f} (mAP50)" if r["task"] == "Detection"
                else f"{r['best_eval_f1']:.5f} (F1)",
                axis=1
            )
            display_df["Image Abnormality F1"] = display_df["img_abnormality_f1"].map(lambda x: f"{x:.5f}")
            display_df["Image Abnormality AUROC"] = display_df["img_abnormality_auroc"].map(lambda x: f"{x:.5f}")
            display_df["Best Val Loss"] = display_df["best_eval_loss"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "N/A")
            display_df["Final Train Loss"] = display_df["final_train_loss"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "N/A")
            display_df["LR"] = display_df["lr"].map(lambda x: f"{x:.5f}")
            
            leaderboard_cols = [
                "short_cfg_name", "task", "model", "peft_type", "peft_detail", 
                "imbalance_strategy", "LR", "Task Metric (F1/mAP50)", "Image Abnormality F1", "Image Abnormality AUROC", "Best Val Loss", "Final Train Loss", "status"
            ]
            
            # Rename columns for presentation
            renamed_df = display_df[leaderboard_cols].rename(columns={
                "short_cfg_name": "Configuration",
                "task": "Task",
                "model": "Backbone",
                "peft_type": "PEFT Type",
                "peft_detail": "PEFT Hyperparams",
                "imbalance_strategy": "Imbalance Strategy",
                "status": "Status"
            })
            
            # Highlight max F1 score row
            def highlight_max_f1(s):
                is_max = s == s.max() if s.name == "Image Abnormality F1" else [False] * len(s)
                return ['background-color: rgba(79, 172, 254, 0.25)' if v else '' for v in is_max]
            
            st.dataframe(
                renamed_df.style.apply(highlight_max_f1, subset=["Image Abnormality F1"]),
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
                preferred_order = ["eval_f1", "eval_custom_cls_f1/abnormality", "eval_mAP50", "eval_loss", "loss", "eval_custom_mean_bbox_IoU"]
                metric_options = [m for m in preferred_order if m in metric_options] + [m for m in metric_options if m not in preferred_order]
                
                plot_metric = st.selectbox(
                    "Select Metric to Compare",
                    options=metric_options,
                    format_func=lambda x: {
                        "eval_f1": "Validation F1 Score (Cls)",
                        "eval_custom_cls_f1/abnormality": "Converted Image-Level Abnormality F1 (Det)",
                        "eval_custom_cls_auroc/abnormality": "Converted Image-Level Abnormality AUROC (Det)",
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
                    }.get(x, x)
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
                            
                            label = f"{run['peft_type']} ({run['peft_detail']}) | lr={run['lr']} | {run['short_cfg_name']}"
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
                        "Backbone Model", "PEFT Method", "PEFT Details", 
                        "Learning Rate", "Imbalance Strategy", "Loss Type", "Epochs Configured"
                    ],
                    "Value": [
                        run_data["model"],
                        run_data["peft_type"],
                        run_data["peft_detail"],
                        f"{run_data['lr']:.5f}",
                        run_data["imbalance_strategy"],
                        run_data["loss_type"],
                        str(run_data["epochs_configured"])
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

                        st.markdown("#### 🖥️ Converted Image-Level Classification")
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.metric("Image Abnormality F1", f"{best_metrics.get('eval_custom_cls_f1/abnormality', 0.0):.4f}")
                            st.metric("Image Text F1", f"{best_metrics.get('eval_custom_cls_f1/text', 0.0):.4f}")
                        with subcol2:
                            st.metric("Image Abnormality AUROC", f"{best_metrics.get('eval_custom_cls_auroc/abnormality', 0.0):.4f}")
                            st.metric("Image Text AUROC", f"{best_metrics.get('eval_custom_cls_auroc/text', 0.0):.4f}")
                        with subcol3:
                            st.metric("Image Abnormality Acc", f"{best_metrics.get('eval_custom_cls_accuracy/abnormality', 0.0):.4f}")
                            st.metric("Image Text Acc", f"{best_metrics.get('eval_custom_cls_accuracy/text', 0.0):.4f}")

                        # Per-class box metrics table
                        st.markdown("#### 📦 Bbox Metrics Per-Class")
                        class_rows = []
                        abnormal_name = run_data.get("abnormal_class_name", "abnormal")
                        classes_to_check = []
                        for c in ["abnormality", abnormal_name, "cell", "text"]:
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

                        # Confusion Matrix for converted image-level abnormality classification
                        tp = best_metrics.get(f"eval_custom_cls_tp/{abnormal_name}") or best_metrics.get("eval_custom_cls_tp/abnormality")
                        fp = best_metrics.get(f"eval_custom_cls_fp/{abnormal_name}") or best_metrics.get("eval_custom_cls_fp/abnormality")
                        tn = best_metrics.get(f"eval_custom_cls_tn/{abnormal_name}") or best_metrics.get("eval_custom_cls_tn/abnormality")
                        fn = best_metrics.get(f"eval_custom_cls_fn/{abnormal_name}") or best_metrics.get("eval_custom_cls_fn/abnormality")
                        cm_title = "Converted Abnormality Confusion Matrix"
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
            col_peft, col_lr = st.columns(2)
            
            with col_peft:
                st.markdown("#### ⚙️ Best F1 Score by PEFT Type")
                # Group by PEFT type to get max F1
                peft_summary = df_filtered.groupby("peft_type")["best_eval_f1"].max().reset_index()
                
                fig_peft = px.bar(
                    peft_summary, 
                    x="peft_type", 
                    y="best_eval_f1",
                    color="peft_type",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    labels={"peft_type": "PEFT Method", "best_eval_f1": "Max F1 Score"},
                    title="PEFT Performance Comparison"
                )
                fig_peft.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_peft, use_container_width=True)
                
            with col_lr:
                st.markdown("#### 📈 Max F1 Score by Learning Rate")
                # Group by LR to get max F1
                lr_summary = df_filtered.groupby("lr")["best_eval_f1"].max().reset_index()
                lr_summary["lr"] = lr_summary["lr"].astype(str) # category format
                
                fig_lr = px.bar(
                    lr_summary, 
                    x="lr", 
                    y="best_eval_f1",
                    color="lr",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    labels={"lr": "Learning Rate", "best_eval_f1": "Max F1 Score"},
                    title="Learning Rate Performance Comparison"
                )
                fig_lr.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_lr, use_container_width=True)
                
            st.divider()
            
            # Parallel Coordinates Plot for Numerical Hyperparameters
            st.markdown("#### 🕸️ Parallel Hyperparameter Trajectory")
            st.write("Visualize how combinations of numerical parameters (learning rate, rank/bottleneck size, validation F1) stack together.")
            
            # Map PEFT sizes into numerical column
            coord_df = df_filtered.copy()
            
            def get_peft_size_num(row):
                detail = row["peft_detail"]
                if row["peft_type"] == "lora":
                    # extract rank
                    try: return float(detail.split("r=")[-1])
                    except: return 0.0
                elif row["peft_type"] == "adapter":
                    # extract dim
                    try: return float(detail.split("d=")[-1])
                    except: return 0.0
                elif row["peft_type"] == "visual_prompt":
                    # extract token count
                    try: return float(detail.split("t=")[-1])
                    except: return 0.0
                return 0.0
                
            coord_df["peft_hyperparam_size"] = coord_df.apply(get_peft_size_num, axis=1)
            
            # Numeric column filter
            numeric_cols = ["lr", "peft_hyperparam_size", "best_eval_f1"]
            if "epochs_configured" in coord_df.columns:
                numeric_cols.append("epochs_configured")
                
            fig_par = px.parallel_coordinates(
                coord_df,
                dimensions=numeric_cols,
                color="best_eval_f1",
                color_continuous_scale=px.colors.diverging.Tealrose,
                labels={
                    "lr": "Learning Rate",
                    "peft_hyperparam_size": "PEFT Size (Rank/Dim/Token)",
                    "best_eval_f1": "Best F1 Score",
                    "epochs_configured": "Epochs"
                }
            )
            fig_par.update_layout(template="plotly_dark")
            st.plotly_chart(fig_par, use_container_width=True)

    # ── Tab 5: Classification vs. Detection Comparison ──────────────────────────
    with tab_comparison:
        st.subheader("⚖️ Classification vs. Detection Model Comparison")
        st.write("Compare the classification models directly with the detection models on the target task: **image-level abnormality classification**.")
        
        if df_results.empty:
            st.info("No runs available for comparison.")
        else:
            # Filter to get classification and detection runs
            cls_runs = df_results[df_results["task"] == "Classification"]
            det_runs = df_results[df_results["task"] == "Detection"]
            
            if cls_runs.empty or det_runs.empty:
                st.info("To see comparative charts, make sure you have at least one completed run for both Classification and Detection tasks.")
            else:
                best_cls = cls_runs.loc[cls_runs["img_abnormality_f1"].idxmax()]
                best_det = det_runs.loc[det_runs["img_abnormality_f1"].idxmax()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🏆 Best Classification Model")
                    st.write(f"**Configuration**: {best_cls['short_cfg_name']}")
                    st.write(f"**Model**: {best_cls['model']}")
                    st.write(f"**PEFT Type**: {best_cls['peft_type']} ({best_cls['peft_detail']})")
                    
                    st.metric("Image Abnormality F1", f"{best_cls['img_abnormality_f1']:.4f}")
                    st.metric("Image Abnormality AUROC", f"{best_cls['img_abnormality_auroc']:.4f}")
                    
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
                    st.markdown("### 🔍 Best Detection Model (Image-Level Conversion)")
                    st.write(f"**Configuration**: {best_det['short_cfg_name']}")
                    st.write(f"**Model**: {best_det['model']}")
                    st.write(f"**PEFT Type**: {best_det['peft_type']} ({best_det['peft_detail']})")
                    
                    st.metric("Image Abnormality F1", f"{best_det['img_abnormality_f1']:.4f}")
                    st.metric("Image Abnormality AUROC", f"{best_det['img_abnormality_auroc']:.4f}")
                    
                    # Confusion Matrix
                    bm_det = best_det["best_metrics"]
                    abnormal_name_d = best_det.get("abnormal_class_name", "abnormal")
                    tp_d = bm_det.get(f"eval_custom_cls_tp/{abnormal_name_d}") or bm_det.get("eval_custom_cls_tp/abnormality")
                    fp_d = bm_det.get(f"eval_custom_cls_fp/{abnormal_name_d}") or bm_det.get("eval_custom_cls_fp/abnormality")
                    tn_d = bm_det.get(f"eval_custom_cls_tn/{abnormal_name_d}") or bm_det.get("eval_custom_cls_tn/abnormality")
                    fn_d = bm_det.get(f"eval_custom_cls_fn/{abnormal_name_d}") or bm_det.get("eval_custom_cls_fn/abnormality")
                    if all(v is not None for v in [tp_d, fp_d, tn_d, fn_d]):
                        fig_cm_det = px.imshow(
                            [[tn_d, fp_d], [fn_d, tp_d]], x=["Predicted Normal", "Predicted Abnormal"], y=["Actual Normal", "Actual Abnormal"],
                            color_continuous_scale="Oranges", aspect="auto", text_auto=True, title="Best Converted Detection Confusion Matrix"
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
                        float(best_cls['img_abnormality_f1']), 
                        float(best_cls['img_abnormality_auroc']), 
                        float(best_det['img_abnormality_f1']), 
                        float(best_det['img_abnormality_auroc'])
                    ]
                })
                fig_comp = px.bar(
                    comp_data, x="Metric", y="Value", color="Task", barmode="group",
                    color_discrete_sequence=["#22c55e", "#ff7f0e"], title="Classification vs. Converted Detection Abnormality Performance",
                    text_auto=".4f"
                )
                fig_comp.update_layout(template="plotly_dark", yaxis_range=[0.0, 1.05])
                st.plotly_chart(fig_comp, use_container_width=True)

if __name__ == "__main__":
    main()
