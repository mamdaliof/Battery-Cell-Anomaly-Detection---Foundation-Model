#!/usr/bin/env python3
import os
import subprocess
import sys
import threading
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List

# VRAM threshold in MiB to consider a GPU "free"
MIN_FREE_VRAM_MIB = 12000

# ANSI colour codes — one per GPU slot (cycles if >10 GPUs)
GPU_COLORS = [
    "\033[36m",   # 0 cyan
    "\033[32m",   # 1 green
    "\033[33m",   # 2 yellow
    "\033[35m",   # 3 magenta
    "\033[34m",   # 4 blue
    "\033[91m",   # 5 bright red
    "\033[92m",   # 6 bright green
    "\033[93m",   # 7 bright yellow
    "\033[94m",   # 8 bright blue
    "\033[95m",   # 9 bright magenta
]
RESET = "\033[0m"
BOLD  = "\033[1m"

# Global print lock to prevent interleaved output across threads
_print_lock = threading.Lock()

def gpu_color(gpu_idx: int) -> str:
    return GPU_COLORS[gpu_idx % len(GPU_COLORS)]

def prefixed_print(gpu_idx: int, cfg_stem: str, line: str) -> None:
    """Print a single line prefixed with [GPUx | cfg_stem], thread-safe."""
    color = gpu_color(gpu_idx)
    tag = f"{color}{BOLD}[GPU{gpu_idx}|{cfg_stem}]{RESET} "
    with _print_lock:
        sys.stdout.write(tag + line + "\n")
        sys.stdout.flush()


def stream_output(proc: subprocess.Popen, gpu_idx: int, cfg_stem: str, log_file) -> None:
    """
    Read lines from proc.stdout in a background thread.
    Each line is printed to the terminal (prefixed) and written to the log file.
    Handles both \\n and \\r (tqdm carriage-return updates) gracefully.
    """
    buf = ""
    while True:
        chunk = proc.stdout.read(1)
        if not chunk:
            # EOF — flush any remaining buffer
            if buf.strip():
                prefixed_print(gpu_idx, cfg_stem, buf.rstrip())
                log_file.write(buf + "\n")
                log_file.flush()
            break

        char = chunk  # already a str because text=True
        if char == "\n":
            prefixed_print(gpu_idx, cfg_stem, buf.rstrip())
            log_file.write(buf + "\n")
            log_file.flush()
            buf = ""
        elif char == "\r":
            # tqdm uses \r to overwrite the same line — treat as a new line for us
            if buf.strip():
                prefixed_print(gpu_idx, cfg_stem, buf.rstrip())
                log_file.write(buf + "\n")
                log_file.flush()
            buf = ""
        else:
            buf += char


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
        print(f"⚠️  Warning: Failed to query nvidia-smi ({e}). Assuming GPU 0 is free.")
        return [0]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_config_equivalent(cfg1: Dict[str, Any], cfg2: Dict[str, Any]) -> bool:
    keys = ["model_name", "data", "head", "peft", "learning_rate", "num_epochs", "imbalance"]
    for k in keys:
        if cfg1.get(k) != cfg2.get(k):
            return False
    return True


def scan_completed_configs(outputs_dir: Path) -> List[Dict[str, Any]]:
    """Scan all output subdirectories containing a DONE file."""
    completed = []
    if not outputs_dir.exists():
        return completed
    for done_path in outputs_dir.glob("**/DONE"):
        config_path = done_path.parent / "config.yaml"
        if config_path.exists():
            try:
                completed.append(load_yaml(config_path))
            except Exception:
                pass
    return completed


def main():
    config_dir  = Path("configs/ablations")
    outputs_dir = Path("outputs")

    if not config_dir.exists():
        print(f"❌  Config directory {config_dir} not found. Run generate_ablation_grid.py first.")
        return

    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        print(f"❌  No YAML configs in {config_dir}")
        return

    print(f"🔍 Found {len(config_files)} total config files.")
    print("🧹 Scanning outputs for already-completed runs...")
    completed_configs = scan_completed_configs(outputs_dir)
    print(f"✓  Found {len(completed_configs)} completed runs (DONE file present).\n")

    configs_to_run = []
    for cfg_path in config_files:
        try:
            cfg = load_yaml(cfg_path)
            already_done = any(is_config_equivalent(cfg, c) for c in completed_configs)
            if not already_done:
                configs_to_run.append((cfg_path, cfg))
            else:
                print(f"⏭️  Skipping (already done): {cfg_path.name}")
        except Exception as e:
            print(f"⚠️  Error reading {cfg_path}: {e}")

    print(f"\n📋 Remaining to train: {len(configs_to_run)}")
    if not configs_to_run:
        print("🎉 All runs are already complete!")
        return

    # Maps gpu_idx -> (proc, cfg_name_stem, log_file, reader_thread)
    running_jobs: Dict[int, tuple] = {}
    pending = list(configs_to_run)

    print(f"\n{'='*70}")
    print("🚀 Starting parallel ablation runs  |  Max 8 GPUs  |  Ctrl+C to stop")
    print(f"{'='*70}\n")

    try:
        while pending or running_jobs:
            # 1. Reap finished processes
            finished_gpus = []
            for gpu_idx, (proc, cfg_stem, log_file, reader_thread) in running_jobs.items():
                if proc.poll() is not None:
                    reader_thread.join(timeout=2)   # wait for last output to flush
                    log_file.close()
                    finished_gpus.append(gpu_idx)
                    symbol = "✅" if proc.returncode == 0 else "❌"
                    code   = "" if proc.returncode == 0 else f" (exit {proc.returncode})"
                    with _print_lock:
                        color = gpu_color(gpu_idx)
                        print(f"{color}{BOLD}[GPU{gpu_idx}|{cfg_stem}]{RESET} "
                              f"{symbol} Finished{code}\n")

            for g in finished_gpus:
                running_jobs.pop(g)

            # 2. Find available GPUs
            free_gpus     = get_free_gpus()
            available     = [g for g in free_gpus if g not in running_jobs]
            available     = available[:8 - len(running_jobs)]

            # 3. Spawn new jobs
            while pending and available:
                cfg_path, cfg_dict = pending.pop(0)
                gpu_idx  = available.pop(0)
                cfg_stem = cfg_path.stem   # e.g. "03_lora_vits16_r8_all_lr0.0003"

                color = gpu_color(gpu_idx)
                with _print_lock:
                    print(f"{color}{BOLD}[GPU{gpu_idx}]{RESET} "
                          f"🚀 Launching: {cfg_stem}\n")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                env["PYTHONPATH"] = f"{os.getcwd()}/src:" + env.get("PYTHONPATH", "")
                # Force tqdm to use plain lines (no ANSI cursor moves)
                env["TQDM_NCOLS"]     = "120"
                env["TQDM_MININTERVAL"] = "5"   # update at most every 5 s to reduce noise

                log_dir  = Path("outputs/logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = open(log_dir / f"{cfg_stem}.log", "w")

                proc = subprocess.Popen(
                    ["python3", "scripts/train.py", "--config", str(cfg_path)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,          # line-buffered
                )

                # Start a daemon reader thread for this process
                reader = threading.Thread(
                    target=stream_output,
                    args=(proc, gpu_idx, cfg_stem, log_file),
                    daemon=True,
                )
                reader.start()

                running_jobs[gpu_idx] = (proc, cfg_stem, log_file, reader)

            # 4. Brief sleep before next poll cycle
            if pending or running_jobs:
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n🛑 Interrupt received — terminating all jobs...")
        for gpu_idx, (proc, cfg_stem, log_file, reader) in running_jobs.items():
            print(f"💀 Killing GPU {gpu_idx}: {cfg_stem}")
            proc.terminate()
            reader.join(timeout=2)
            log_file.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("👋 Done.")


if __name__ == "__main__":
    main()
