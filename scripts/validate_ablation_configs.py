#!/usr/bin/env python3
"""
validate_ablation_configs.py

Runs check_model_init.py for every config in configs/ablations/ in parallel
(one GPU per job, up to 8 concurrent), collects PASS/FAIL results, and writes
a consolidated report to outputs/validation_report.txt.

Usage:
    python3 scripts/validate_ablation_configs.py
"""
import os
import subprocess
import sys
import threading
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Settings ──────────────────────────────────────────────────────────────────
MIN_FREE_VRAM_MIB = 12000
MAX_PARALLEL_JOBS  = 8
REPORT_PATH        = Path("outputs/validation_report.txt")

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


def gpu_color(gpu_idx: int) -> str:
    return GPU_COLORS[gpu_idx % len(GPU_COLORS)]


def tprint(gpu_idx: int, cfg_stem: str, line: str) -> None:
    color = gpu_color(gpu_idx)
    tag   = f"{color}{BOLD}[GPU{gpu_idx}|{cfg_stem}]{RESET} "
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
    except Exception as e:
        with _print_lock:
            print(f"⚠️  nvidia-smi failed ({e}) — assuming GPU 0 is free.")
        return [0]


def stream_and_capture(
    proc:      subprocess.Popen,
    gpu_idx:   int,
    cfg_stem:  str,
    lines_out: List[str],
) -> None:
    """Read proc stdout line by line, print prefixed, accumulate into lines_out."""
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        if line:
            tprint(gpu_idx, cfg_stem, line)
            lines_out.append(line)


def run_check(gpu_idx: int, cfg_path: Path) -> Tuple[bool, str]:
    """
    Spawn check_model_init.py for one config on a specific GPU.
    Returns (passed: bool, full_output: str).
    """
    cfg_stem = cfg_path.stem
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    env["PYTHONPATH"]           = f"{os.getcwd()}/src:" + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        ["python3", "scripts/check_model_init.py", "--config", str(cfg_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured: List[str] = []
    reader = threading.Thread(
        target=stream_and_capture,
        args=(proc, gpu_idx, cfg_stem, captured),
        daemon=True,
    )
    reader.start()
    proc.wait()
    reader.join(timeout=5)

    passed = proc.returncode == 0
    return passed, "\n".join(captured)


def main():
    config_dir = Path("configs/ablations")
    if not config_dir.exists():
        print("❌  configs/ablations/ not found. Run generate_ablation_grid.py first.")
        return

    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        print("❌  No YAML files in configs/ablations/")
        return

    print(f"🔍 Found {len(config_files)} configs to validate.\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Results: list of (cfg_stem, passed, output_text)
    results: List[Tuple[str, bool, str]] = []
    results_lock = threading.Lock()

    # pending queue and job map
    pending      = list(config_files)
    running_jobs: Dict[int, Tuple[threading.Thread, Path]] = {}

    print(f"{'='*70}")
    print(f"🧪 Validating all configs in parallel  |  Max {MAX_PARALLEL_JOBS} GPUs")
    print(f"{'='*70}\n")

    def job_worker(gpu_idx: int, cfg_path: Path):
        passed, output = run_check(gpu_idx, cfg_path)
        symbol = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        with _print_lock:
            print(f"\n{gpu_color(gpu_idx)}{BOLD}[GPU{gpu_idx}]{RESET} "
                  f"{symbol} → {cfg_path.name}\n")
        with results_lock:
            results.append((cfg_path.stem, passed, output))

    try:
        while pending or running_jobs:
            # Reap finished threads
            done = [g for g, (t, _) in running_jobs.items() if not t.is_alive()]
            for g in done:
                running_jobs.pop(g)

            # Find free GPUs not already occupied
            free_gpus  = get_free_gpus()
            available  = [g for g in free_gpus if g not in running_jobs]
            available  = available[:MAX_PARALLEL_JOBS - len(running_jobs)]

            # Spawn new jobs
            while pending and available:
                cfg_path = pending.pop(0)
                gpu_idx  = available.pop(0)
                tprint(gpu_idx, cfg_path.stem, f"🔬 Validating model init…")
                t = threading.Thread(
                    target=job_worker,
                    args=(gpu_idx, cfg_path),
                    daemon=True,
                )
                t.start()
                running_jobs[gpu_idx] = (t, cfg_path)

            if pending or running_jobs:
                time.sleep(3)

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
