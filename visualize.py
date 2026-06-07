import os
import glob
import pandas as pd
import streamlit as st
import yaml
import plotly.express as px

def get_best_epoch(df, task):
    """Finds the epoch with the lowest validation loss."""
    if task == 'Classification':
        loss_cols = [c for c in df.columns if 'val/loss' in c]
        if loss_cols and not df[loss_cols[0]].isna().all():
            return df[loss_cols[0]].idxmin()
    elif task == 'Detection':
        # Sum the validation losses to find the best epoch
        loss_cols = [c for c in df.columns if 'val/' in c and 'loss' in c]
        if loss_cols:
            total_loss = df[loss_cols].sum(axis=1)
            if not total_loss.isna().all():
                return total_loss.idxmin()
    return df.index.max() # Fallback to last epoch if not found

def parse_experiment_name(exp_name):
    """Extracts features from experiment directory name."""
    features = {}
    name_lower = exp_name.lower()
    
    if 'no_aug' in name_lower:
        features['Augmentation'] = 'No'
    else:
        features['Augmentation'] = 'Yes' # Assume Yes if not explicitly 'no_aug'
        
    if 'vr3' in name_lower:
        features['Dataset_VR'] = 'VR3'
    elif 'vr4' in name_lower:
        features['Dataset_VR'] = 'VR4'
    elif 'vr5' in name_lower:
        features['Dataset_VR'] = 'VR5'
        
    if 'high' in name_lower:
        features['Density'] = 'High'
    elif 'medium' in name_lower:
        features['Density'] = 'Medium'
    elif 'low' in name_lower:
        features['Density'] = 'Low'
        
    if 'nano' in name_lower:
        features['Model_Size'] = 'Nano'
    elif 'small' in name_lower:
        features['Model_Size'] = 'Small'
        
    if '_a_' in name_lower or name_lower.endswith('_a'):
        features['Variant'] = 'A'
    elif '_b_' in name_lower or name_lower.endswith('_b'):
        features['Variant'] = 'B'
    elif '_c_' in name_lower or name_lower.endswith('_c'):
        features['Variant'] = 'C'
    elif 'granular' in name_lower:
        features['Variant'] = 'Granular'
        
    return features

def load_experiments(base_path):
    experiments = {}
    # Use recursive glob to find all results.csv
    csv_files = glob.glob(os.path.join(base_path, '**', 'results.csv'), recursive=True)
    
    for file_path in csv_files:
        exp_dir = os.path.dirname(file_path)
        # Use the relative path from base_path as the experiment name to handle subfolders correctly
        exp_name = os.path.relpath(exp_dir, base_path)
        
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            if 'epoch' in df.columns:
                df = df.set_index('epoch')
            
            # Determine task based on columns
            if 'metrics/accuracy_top1' in df.columns:
                task = 'Classification'
            elif 'metrics/mAP50(B)' in df.columns or 'metrics/mAP_0.5' in df.columns:
                task = 'Detection'
            else:
                task = 'Unknown'
                
            # Load args.yaml if it exists
            args_path = os.path.join(exp_dir, 'args.yaml')
            args = {}
            if os.path.exists(args_path):
                with open(args_path, 'r') as f:
                    args = yaml.safe_load(f)
                    
            best_epoch = get_best_epoch(df, task)
            
            # Extract key stats for the leaderboard
            best_stats = {"Best Epoch": best_epoch}
            if task == 'Classification' and 'metrics/accuracy_top1' in df.columns:
                best_stats["Best Acc Top1"] = df.loc[best_epoch, 'metrics/accuracy_top1'] if best_epoch in df.index else df['metrics/accuracy_top1'].max()
            elif task == 'Detection':
                # Prefer mAP50 as requested
                if 'metrics/mAP50(B)' in df.columns:
                    best_stats["Best mAP50"] = df.loc[best_epoch, 'metrics/mAP50(B)'] if best_epoch in df.index else df['metrics/mAP50(B)'].max()
                elif 'metrics/mAP_0.5' in df.columns:
                    best_stats["Best mAP50"] = df.loc[best_epoch, 'metrics/mAP_0.5'] if best_epoch in df.index else df['metrics/mAP_0.5'].max()
                    
                if 'metrics/mAP50-95(B)' in df.columns:
                    best_stats["Best mAP50-95"] = df.loc[best_epoch, 'metrics/mAP50-95(B)'] if best_epoch in df.index else df['metrics/mAP50-95(B)'].max()
                
            if task not in experiments:
                experiments[task] = {}
                
            experiments[task][exp_name] = {
                'df': df,
                'args': args,
                'best_epoch': best_epoch,
                'best_stats': best_stats
            }
        except Exception as e:
            st.warning(f"Error loading {file_path}: {e}")
            
    return experiments

def main():
    st.set_page_config(page_title="YOLO Results Visualizer", layout="wide")
    st.title("YOLO Training Results Visualizer")
    
    base_dir = st.text_input("Trains Directory", value="trains")
    
    if not os.path.isdir(base_dir):
        st.error(f"Directory '{base_dir}' does not exist.")
        return
        
    experiments = load_experiments(base_dir)
    
    if not experiments:
        st.warning("No results.csv found in the specified directory.")
        return
        
    # Task selection
    tasks = list(experiments.keys())
    selected_task = st.sidebar.selectbox("Select Task", tasks)
    
    task_exps = experiments[selected_task]
    exp_names = list(task_exps.keys())
    
    # Global Experiment selection
    selected_exps = st.sidebar.multiselect("Select Experiments", exp_names, default=exp_names)
    
    if not selected_exps:
        st.info("Please select at least one experiment.")
        return
        
    # Pre-compute leaderboard DataFrame for multiple tabs
    table_data = []
    for exp in selected_exps:
        row = {"Experiment": exp}
        
        # Parse features from name
        parsed_features = parse_experiment_name(exp)
        row.update(parsed_features)
        
        row.update(task_exps[exp]['best_stats'])
        
        args = task_exps[exp]['args']
        keys_to_extract = ['model', 'epochs', 'batch', 'imgsz', 'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay']
        for k in keys_to_extract:
            if k in args:
                row[k] = args[k]
                
        table_data.append(row)
        
    leaderboard_df = pd.DataFrame(table_data) if table_data else pd.DataFrame()
        
    # Define target_metric here so it can be used across tabs
    target_metric = 'Best Acc Top1' if selected_task == 'Classification' else 'Best mAP50'
    
    # App Tabs
    tab_lines, tab_leaderboard, tab_hyperparams = st.tabs(["Line Charts (Epochs)", "Leaderboard", "Hyperparameters Analysis"])
    
    with tab_lines:
        st.header("Performance over Epochs")
        stop_at_best = st.checkbox("Stop plot at Best Epoch (lowest validation loss)", value=False)
        
        available_metrics = set()
        for exp in selected_exps:
            available_metrics.update(task_exps[exp]['df'].columns.tolist())
            
        available_metrics = sorted([m for m in available_metrics if m not in ['time']])
        selected_metrics = st.multiselect("Select Metrics to Plot", available_metrics, default=available_metrics[:2] if available_metrics else None)
        
        for metric in selected_metrics:
            st.subheader(f"{metric}")
            plot_data = pd.DataFrame()
            
            for exp in selected_exps:
                df = task_exps[exp]['df']
                if metric in df.columns:
                    if stop_at_best:
                        best_ep = task_exps[exp]['best_epoch']
                        df_plot = df.loc[:best_ep]
                    else:
                        df_plot = df
                        
                    plot_data[exp] = df_plot[metric]
                    
            if not plot_data.empty:
                st.line_chart(plot_data)
            else:
                st.write(f"Metric '{metric}' not found.")

    with tab_leaderboard:
        st.header("Aggregated Leaderboard")
        st.write("Compare best metrics and hyperparameters extracted from `args.yaml`.")
        if not leaderboard_df.empty:
            # Enable sorting
            sort_cols = st.multiselect("Sort by", leaderboard_df.columns.tolist(), default=[target_metric if target_metric in leaderboard_df.columns else leaderboard_df.columns[1]])
            sort_asc = st.checkbox("Ascending order", value=False)
            
            if sort_cols:
                display_df = leaderboard_df.sort_values(by=sort_cols, ascending=sort_asc)
            else:
                display_df = leaderboard_df
                
            st.dataframe(display_df, use_container_width=True)

    with tab_hyperparams:
        st.header("Hyperparameter Analysis")
        st.write("Explore relationships between settings and the final performance metric.")
        
        if not leaderboard_df.empty:
            cols = leaderboard_df.columns.tolist()
            
            # --- Scatter Plot ---
            st.subheader("Scatter Plot")
            col1, col2, col3 = st.columns(3)
            
            default_x = 'lr0' if 'lr0' in cols else cols[0]
            default_y = target_metric if target_metric in cols else cols[0]
            
            x_axis = col1.selectbox("X-Axis", cols, index=cols.index(default_x))
            y_axis = col2.selectbox("Y-Axis", cols, index=cols.index(default_y))
            color_col = col3.selectbox("Color By", ["None"] + cols, index=0)
            
            color_arg = None if color_col == "None" else color_col
            fig_scatter = px.scatter(
                leaderboard_df, x=x_axis, y=y_axis, color=color_arg, 
                hover_data=["Experiment"], title=f"{y_axis} vs {x_axis}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.divider()
            
            # --- Parallel Coordinates ---
            st.subheader("Parallel Coordinates Plot")
            st.write("This plot is great for visualizing how combinations of hyperparameters perform. It only uses numerical data.")
            
            # Filter to numeric columns for Parallel Coordinates
            numeric_cols = leaderboard_df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 1:
                selected_par_cols = st.multiselect("Select Columns (Numeric)", numeric_cols, default=numeric_cols)
                
                if len(selected_par_cols) > 1:
                    color_par = target_metric if target_metric in selected_par_cols else selected_par_cols[-1]
                    fig_par = px.parallel_coordinates(
                        leaderboard_df, 
                        dimensions=selected_par_cols, 
                        color=color_par,
                        color_continuous_scale=px.colors.diverging.Tealrose
                    )
                    st.plotly_chart(fig_par, use_container_width=True)
                else:
                    st.warning("Please select at least 2 columns to generate the plot.")
            else:
                st.info("Not enough numeric columns available for Parallel Coordinates.")

if __name__ == "__main__":
    main()
