#!/usr/bin/env python3
"""
validate_ablation_configs.py

Runs check_model_init.py for every config in configs/cls/ablations/ in parallel.
Uses available free GPUs if found (>= 12GB free VRAM). Otherwise, falls back to
parallel CPU execution to avoid deadlocks.

Usage:
    python3 scripts/validate_ablation_configs.py
"""
import os
import subprocess
import sys
import threading
import time
import yaml
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Automatically redirect Hugging Face cache to local workspace directory
if "HF_HOME" not in os.environ:
    workspace_root = Path(__file__).resolve().parents[1]
    hf_cache_dir = workspace_root / "models" / "hf_cache"
    os.environ["HF_HOME"] = str(hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # If the model is cached in the default home directory, copy it to the local workspace cache
    default_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if default_cache.exists():
        for p in default_cache.glob("models--facebook--dinov3*"):
            if p.is_dir():
                target_hub_dir = hf_cache_dir / "hub"
                target_dir = target_hub_dir / p.name
                if not target_dir.exists():
                    try:
                        import shutil
                        target_hub_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(p, target_dir, symlinks=True)
                    except Exception:
                        pass


# ── Settings ──────────────────────────────────────────────────────────────────
MIN_FREE_VRAM_MIB = 12000
MAX_PARALLEL_JOBS  = 8
REPORT_PATH        = Path("outputs/cls_all/validation_report.txt")

# ── ANSI colours ──────────────────────────────────────────────────────────────
GPU_COLORS = [
    "\033[36m", "\033[32m", "\033[33m", "\033[35m",
    "\033[34m", "\033[91m", "\033[92m", "\033[93m",
]
GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
BOLD  = "\033[1m"

_print_lock = threading.Lock()


def slot_color(slot_idx: int) -> str:
    return GPU_COLORS[slot_idx % len(GPU_COLORS)]


def tprint(slot_idx: int, label: str, cfg_stem: str, line: str) -> None:
    color = slot_color(slot_idx)
    tag   = f"{color}{BOLD}[{label}{slot_idx}|{cfg_stem}]{RESET} "
    with _print_lock:
        sys.stdout.write(tag + line + "\n")
        sys.stdout.flush()


def get_free_gpus() -> List[int]:
    try:
        cmd    = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True, text=True)
        gpus   = []
        for line in output.strip().splitlines():
            if not line:
                continue
            idx_str, mem_str = line.split(",")
            if int(mem_str.strip()) >= MIN_FREE_VRAM_MIB:
                gpus.append(int(idx_str.strip()))
        return gpus
    except Exception:
        return []


def stream_and_capture(
    proc:      subprocess.Popen,
    slot_idx:  int,
    label:     str,
    cfg_stem:  str,
    lines_out: List[str],
) -> None:
    """Read proc stdout line by line, print prefixed, accumulate into lines_out."""
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        if line:
            tprint(slot_idx, label, cfg_stem, line)
            lines_out.append(line)


def run_check(slot_idx: int, cfg_path: Path, use_gpu: bool, gpu_id: int = 0) -> Tuple[bool, str]:
    """
    Spawn check_model_init.py for one config on a specific slot (GPU or CPU).
    Returns (passed: bool, full_output: str).
    """
    cfg_stem = cfg_path.stem
    label = "GPU" if use_gpu else "CPU"
    env = os.environ.copy()
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"]           = f"{os.getcwd()}/src:" + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, "scripts/check_model_init.py", "--config", str(cfg_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured: List[str] = []
    reader = threading.Thread(
        target=stream_and_capture,
        args=(proc, slot_idx, label, cfg_stem, captured),
        daemon=True,
    )
    reader.start()
    proc.wait()
    reader.join(timeout=5)

    passed = proc.returncode == 0
    return passed, "\n".join(captured)


def main():
    config_dir = Path("configs/cls/ablations_all_label")
    if not config_dir.exists():
        print("❌  configs/cls/ablations_all_label/ not found. Run generate_ablation_grid.py first.")
        return

    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        print("❌  No YAML files in configs/cls/ablations_all_label/")
        return

    print(f"🔍 Found {len(config_files)} configs to validate.\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Detect environment capability ─────────────────────────────────────────
    free_gpus = get_free_gpus()
    if free_gpus:
        use_gpu = True
        slots = free_gpus
        label = "GPU"
        concurrency = len(free_gpus)
        print(f"✅ Found free GPUs: {free_gpus}")
    else:
        use_gpu = False
        # Limit CPU concurrency to avoid overloading the system
        cpu_count = multiprocessing.cpu_count() or 2
        concurrency = min(MAX_PARALLEL_JOBS, max(1, cpu_count // 2))
        slots = list(range(concurrency))
        label = "CPU"
        print(f"⚠️  No free GPUs >= {MIN_FREE_VRAM_MIB} MiB found. Falling back to CPU validation.")

    # Results: list of (cfg_stem, passed, output_text)
    results: List[Tuple[str, bool, str]] = []
    results_lock = threading.Lock()

    # pending queue and slot allocation
    pending = list(config_files)
    # Maps slot_idx -> (thread, config_path)
    running_jobs: Dict[int, Tuple[threading.Thread, Path]] = {}

    print(f"{'='*70}")
    print(f"🧪 Validating configs in parallel  |  Mode: {label}  |  Concurrency: {concurrency}")
    print(f"{'='*70}\n")

    def job_worker(slot_idx: int, gpu_id: int, cfg_path: Path):
        passed, output = run_check(slot_idx, cfg_path, use_gpu, gpu_id)
        symbol = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        with _print_lock:
            print(f"\n{slot_color(slot_idx)}{BOLD}[{label}{slot_idx}]{RESET} "
                  f"{symbol} → {cfg_path.name}\n")
        with results_lock:
            results.append((cfg_path.stem, passed, output))

    try:
        while pending or running_jobs:
            # Reap finished threads
            done_slots = [s for s, (t, _) in running_jobs.items() if not t.is_alive()]
            for s in done_slots:
                running_jobs.pop(s)

            # Find empty slots
            empty_slots = [s for s in slots if s not in running_jobs]

            # Spawn new jobs in empty slots
            while pending and empty_slots:
                cfg_path = pending.pop(0)
                slot_idx = empty_slots.pop(0)
                
                gpu_id = slot_idx if use_gpu else 0
                tprint(slot_idx, label, cfg_path.stem, f"🔬 Validating model init…")
                t = threading.Thread(
                    target=job_worker,
                    args=(slot_idx, gpu_id, cfg_path),
                    daemon=True,
                )
                t.start()
                running_jobs[slot_idx] = (t, cfg_path)

            if pending or running_jobs:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")

    # ── Write consolidated report ──────────────────────────────────────────────
    passed_list = [(s, o) for s, p, o in results if p]
    failed_list = [(s, o) for s, p, o in results if not p]

    with open(REPORT_PATH, "w") as f:
        f.write(f"Ablation Config Validation Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total: {len(results)}  |  PASS: {len(passed_list)}  |  FAIL: {len(failed_list)}\n")
        f.write("=" * 70 + "\n\n")

        f.write("── PASSED ──────────────────────────────────────────────────────────\n\n")
        for stem, output in sorted(passed_list):
            f.write(f"[PASS] {stem}\n")
            for line in output.splitlines():
                f.write(f"       {line}\n")
            f.write("\n")

        f.write("── FAILED ──────────────────────────────────────────────────────────\n\n")
        for stem, output in sorted(failed_list):
            f.write(f"[FAIL] {stem}\n")
            for line in output.splitlines():
                f.write(f"       {line}\n")
            f.write("\n")

    # ── Terminal summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"  Total: {len(results)}  |  {GREEN}PASS: {len(passed_list)}{RESET}  |  {RED}FAIL: {len(failed_list)}{RESET}")
    print(f"  Report written to: {REPORT_PATH}")
    print(f"{'='*70}\n")

    if failed_list:
        print(f"{RED}Failed configs:{RESET}")
        for stem, _ in sorted(failed_list):
            print(f"  ❌ {stem}")
        print()


if __name__ == "__main__":
    main()
