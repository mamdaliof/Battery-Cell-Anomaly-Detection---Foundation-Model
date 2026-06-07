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
            
            # Default task is classification
            task = "Classification"
            custom_param = "default"
            
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
                    eval_steps = [item for item in history if "eval_f1" in item]
                    train_losses = [item.get("loss") for item in history if "loss" in item]
                    
                    if eval_steps:
                        # Find step with max eval_f1. If tie, select lowest eval_loss
                        best_step = max(eval_steps, key=lambda x: (x.get("eval_f1", 0.0), -x.get("eval_loss", float('inf'))))
                        best_eval_f1 = best_step.get("eval_f1", 0.0)
                        best_eval_loss = best_step.get("eval_loss", float('inf'))
                        best_epoch_metrics = best_step
                    
                    if train_losses:
                        final_train_loss = train_losses[-1]
                        
                except Exception as e:
                    st.warning(f"⚠️ Error parsing state at {state_path}: {e}")
            
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
                "history": history
            })
            
    return pd.DataFrame(runs_data)

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

    # Placeholders for future features in filters (Tasks and custom hyperparameters)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filter Experiments")

    # 1. Task filter (Placeholder for future tasks)
    task_options = ["Classification"] + ["Segmentation (Future)", "Object Detection (Future)", "Anomaly Localization (Future)"]
    selected_task = st.sidebar.selectbox("Task Profile", task_options, index=0)

    if selected_task != "Classification":
        st.info(f"No active runs found for '{selected_task}'. Showing future placeholder mockups.")
        
        # Render clean placeholder layout
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Future Model Architectures")
            st.write("Planned backbones: DINOv3-ViT-L/16, Segment Anything (SAM-2), and YOLOv11.")
            fig = px.bar(
                x=["ViT-S/16", "ViT-B/16", "ViT-L/16 (Future)", "SAM-2 (Future)"],
                y=[0.89, 0.92, 0.94, 0.95],
                labels={"x": "Architecture", "y": "Target F1 Score (Est.)"},
                title="Target Performance Projections"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Hyperparameter Space Expansion")
            st.write("Planned additions: Cosine schedule restarts, AdamW ScheduleFree optimizer, and batch sizes up to 256.")
            st.table(pd.DataFrame({
                "Hyperparameter": ["Optimizer", "Batch Size", "LR Schedule", "Augmentations"],
                "Current Ablation": ["AdamW", "64", "Cosine Annealing", "Gaussian + Color Jitter"],
                "Proposed Future": ["AdamW / ScheduleFree", "32, 64, 128, 256", "Cosine with Restarts", "CutMix / MixUp / Synthetics"]
            }))
        return

    # Extract unique values from data
    unique_models = df_results["model"].unique().tolist()
    unique_pefts = df_results["peft_type"].unique().tolist()
    unique_lrs = sorted(df_results["lr"].unique().tolist())
    unique_imbs = df_results["imbalance_strategy"].unique().tolist()

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
    df_filtered = df_filtered.sort_values(by="best_eval_f1", ascending=False)

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
        best_f1_overall = df_results["best_eval_f1"].max() if not df_results.empty else 0.0
        best_run_overall = df_results.loc[df_results["best_eval_f1"].idxmax()]["short_cfg_name"] if not df_results.empty and best_f1_overall > 0 else "N/A"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val" style="color: #f43f5e;">{best_f1_overall:.4f}</div>
                <div class="kpi-lbl">Best F1 ({best_run_overall[:15]})</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ── Tabs Setup ────────────────────────────────────────────────────────────
    tab_leaderboard, tab_curves, tab_inspector, tab_peft_analysis = st.tabs([
        "🏆 Leaderboard", 
        "📈 Trajectory Curves", 
        "🔬 Single Run Inspector", 
        "📊 PEFT & Hyperparameter Analysis"
    ])

    # ── Tab 1: Leaderboard ─────────────────────────────────────────────────────
    with tab_leaderboard:
        st.subheader("🏆 Ablation Experiment Leaderboard")
        st.write("Showing all configurations matching filters. Sort by any column, prioritized by **Validation F1 Score**.")
        
        if df_filtered.empty:
            st.info("No runs match the current filters. Please adjust the sidebar settings.")
        else:
            # Prepare leaderboard DataFrame for clean display
            display_df = df_filtered.copy()
            
            # Map completed boolean to emojis for rich design
            display_df["status"] = display_df["completed"].apply(lambda x: "✅ Completed" if x else "⏳ Active/Interrupted")
            
            # Formatted column values for presentation
            display_df["Best F1 Score"] = display_df["best_eval_f1"].map(lambda x: f"{x:.5f}")
            display_df["Best Val Loss"] = display_df["best_eval_loss"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "N/A")
            display_df["Final Train Loss"] = display_df["final_train_loss"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "N/A")
            display_df["LR"] = display_df["lr"].map(lambda x: f"{x:.5f}")
            
            leaderboard_cols = [
                "short_cfg_name", "task", "model", "peft_type", "peft_detail", 
                "imbalance_strategy", "LR", "Best F1 Score", "Best Val Loss", "Final Train Loss", "status"
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
                is_max = s == s.max() if s.name == "Best F1 Score" else [False] * len(s)
                return ['background-color: rgba(79, 172, 254, 0.25)' if v else '' for v in is_max]
            
            st.dataframe(
                renamed_df.style.apply(highlight_max_f1, subset=["Best F1 Score"]),
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
                
                # Metric to plot selection
                plot_metric = st.selectbox(
                    "Select Metric to Compare",
                    options=[
                        "eval_f1", 
                        "eval_loss", 
                        "loss", # train loss
                        "eval_accuracy", 
                        "eval_auroc",
                        "eval_precision",
                        "eval_recall"
                    ],
                    format_func=lambda x: {
                        "eval_f1": "Validation F1 Score (Priority)",
                        "eval_loss": "Validation Loss",
                        "loss": "Training Loss",
                        "eval_accuracy": "Validation Accuracy",
                        "eval_auroc": "Validation AUROC",
                        "eval_precision": "Validation Precision",
                        "eval_recall": "Validation Recall"
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
                        run_data["epochs_configured"]
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
                    
                    if all(v is not None for v in [tp, fp, tn, fn]):
                        st.markdown("#### 🧮 Confusion Matrix (Best Epoch)")
                        
                        # Create interactive Plotly Heatmap for confusion matrix
                        z = [[tn, fp], [fn, tp]]
                        x = ["Predicted Normal", "Predicted Abnormal"]
                        y = ["Actual Normal", "Actual Abnormal"]
                        
                        fig_cm = px.imshow(
                            z, x=x, y=y,
                            color_continuous_scale="Blues",
                            aspect="auto",
                            text_auto=True,
                            title="Confusion Matrix"
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
                cols_to_display = ["epoch", "step", "loss", "eval_loss", "eval_f1", "eval_accuracy", "eval_auroc", "eval_precision", "eval_recall"]
                cols_present = [c for c in cols_to_display if c in hist_df.columns]
                
                st.dataframe(
                    hist_df[cols_present].sort_values(by="epoch").dropna(subset=["eval_f1"]),
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

if __name__ == "__main__":
    main()
