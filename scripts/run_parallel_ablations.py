#!/usr/bin/env python3
import os
import subprocess
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Set

# VRAM threshold in MiB to consider a GPU "free"
MIN_FREE_VRAM_MIB = 12000

def get_free_gpus() -> List[int]:
    """Query nvidia-smi for GPUs with enough free memory."""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True, text=True)
        free_gpus = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            idx_str, free_mem_str = line.split(",")
            idx = int(idx_str.strip())
            free_mem = int(free_mem_str.strip())
            if free_mem >= MIN_FREE_VRAM_MIB:
                free_gpus.append(idx)
        return free_gpus
    except Exception as e:
        print(f"⚠️ Warning: Failed to query nvidia-smi ({e}). Defaulting to CPU or assuming GPU 0 is free.")
        # If nvidia-smi is not available, we assume only GPU 0 is free (or CPU)
        return [0]

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def is_config_equivalent(cfg1: Dict[str, Any], cfg2: Dict[str, Any]) -> bool:
    """Compare two configs ignoring output directory and random seeds if needed."""
    # We compare the critical fields: model_name, data config, head config, peft config, hyperparams, imbalance config
    keys = ["model_name", "data", "head", "peft", "learning_rate", "num_epochs", "imbalance"]
    for k in keys:
        if cfg1.get(k) != cfg2.get(k):
            return False
    return True

def scan_completed_configs(outputs_dir: Path) -> List[Dict[str, Any]]:
    """Scan all output subdirectories containing a DONE file and load their configurations."""
    completed = []
    if not outputs_dir.exists():
        return completed

    # Walk outputs/ looking for 'DONE' files
    for done_path in outputs_dir.glob("**/DONE"):
        run_dir = done_path.parent
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            try:
                cfg = load_yaml(config_path)
                completed.append(cfg)
            except Exception:
                pass
    return completed

def main():
    config_dir = Path("configs/ablations")
    outputs_dir = Path("outputs")
    
    if not config_dir.exists():
        print(f"❌ Error: Ablations config directory {config_dir} does not exist. Run scripts/generate_ablation_grid.py first.")
        return

    # Find all yaml files in config_dir
    config_files = sorted(list(config_dir.glob("*.yaml")))
    if not config_files:
        print(f"❌ Error: No YAML config files found in {config_dir}")
        return

    print(f"🔍 Found {len(config_files)} total config files.")

    # Scan already completed runs
    print("🧹 Scanning outputs to find completed runs...")
    completed_configs = scan_completed_configs(outputs_dir)
    print(f"✓ Found {len(completed_configs)} completed runs with a DONE file.")

    # Filter configs to run
    configs_to_run = []
    for cfg_path in config_files:
        try:
            cfg = load_yaml(cfg_path)
            # Check if this config has already run successfully
            already_done = False
            for comp_cfg in completed_configs:
                if is_config_equivalent(cfg, comp_cfg):
                    already_done = True
                    break
            
            if not already_done:
                configs_to_run.append((cfg_path, cfg))
        except Exception as e:
            print(f"⚠️ Error reading config {cfg_path}: {e}")

    print(f"📋 Configurations remaining to train: {len(configs_to_run)}")
    if not configs_to_run:
        print("🎉 All configurations are already completed! Nothing to do.")
        return

    # Track running processes
    # Maps gpu_idx -> (subprocess.Popen, config_path_str)
    running_jobs: Dict[int, tuple[subprocess.Popen, str]] = {}
    
    # Copy of configs to run
    pending = list(configs_to_run)

    print("\n🚀 Starting parallel ablation runs. Max 8 concurrent GPUs.")
    print("Press Ctrl+C to terminate all runs safely.\n")

    try:
        while pending or running_jobs:
            # 1. Clean up completed processes
            finished_gpus = []
            for gpu_idx, (proc, cfg_name, log_file) in running_jobs.items():
                ret = proc.poll()
                if ret is not None:
                    # Process completed
                    finished_gpus.append(gpu_idx)
                    log_file.close()  # Close the file descriptor
                    if ret == 0:
                        print(f"✅ [GPU {gpu_idx}] Completed: {cfg_name}")
                    else:
                        print(f"❌ [GPU {gpu_idx}] Failed with exit code {ret}: {cfg_name}")

            for gpu_idx in finished_gpus:
                running_jobs.pop(gpu_idx)

            # 2. Query free GPUs
            free_gpus = get_free_gpus()
            # Filter down to GPUs that are not currently running a job in this runner
            available_gpus = [g for g in free_gpus if g not in running_jobs]
            # Max 8 concurrent jobs limit (and keep 1 job per GPU)
            available_gpus = available_gpus[:8 - len(running_jobs)]

            # 3. Spawn new jobs if possible
            while pending and available_gpus:
                cfg_path, cfg_dict = pending.pop(0)
                gpu_idx = available_gpus.pop(0)
                
                cfg_name = cfg_path.name
                print(f"🚀 [GPU {gpu_idx}] Launching training: {cfg_name}")

                # Configure environment with specific GPU
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                env["PYTHONPATH"] = f"{os.getcwd()}/src:" + env.get("PYTHONPATH", "")

                # Create log file for this config
                log_dir = Path("outputs/logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file_path = log_dir / f"{cfg_path.stem}.log"
                log_file = open(log_file_path, "w")

                # Run process
                cmd = ["python3", "scripts/train.py", "--config", str(cfg_path)]
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
                running_jobs[gpu_idx] = (proc, cfg_name, log_file)

            # 4. Wait a bit before checking again
            if pending or running_jobs:
                time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received! Terminating all running training processes...")
        for gpu_idx, (proc, cfg_name, log_file) in running_jobs.items():
            print(f"💀 Killing training on GPU {gpu_idx}: {cfg_name}")
            proc.terminate()
            log_file.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("👋 Terminated.")

if __name__ == "__main__":
    main()
