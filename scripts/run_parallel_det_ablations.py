#!/usr/bin/env python3
"""Parallel detection ablation runner — compact in-place dashboard.

One persistent terminal line per GPU slot (0-7), refreshed every 0.5s.
All subprocess output is silently parsed for progress, epoch, loss,
and mAP50 — nothing else is printed to the terminal.
Full raw output is still written to outputs/det/logs/<config>.log for debugging.
"""
import os
import re
import subprocess
import sys
import threading
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

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

MIN_FREE_VRAM_MIB = 12_000
MAX_PARALLEL_JOBS = 8
REFRESH_SECS      = 0.5   # dashboard redraw interval

# ── ANSI ──────────────────────────────────────────────────────────────────────
_COLORS = ["\033[36m", "\033[32m", "\033[33m", "\033[35m",
           "\033[34m", "\033[91m", "\033[92m", "\033[93m"]
RST   = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[32m"
RED   = "\033[31m"

def _col(idx: int) -> str:
    return _COLORS[idx % len(_COLORS)]


# ── Per-GPU mutable status ────────────────────────────────────────────────────
class _Slot:
    __slots__ = ("state", "name", "pct", "step", "total",
                 "speed", "epoch", "max_epoch", "loss", "map50", "rc")

    def __init__(self):
        self.state:     str            = "idle"
        self.name:      str            = ""
        self.pct:       int            = 0
        self.step:      int            = 0
        self.total:     int            = 0
        self.speed:     str            = ""
        self.epoch:     float          = 0.0
        self.max_epoch: int            = 300
        self.loss:      Optional[float] = None
        self.map50:     Optional[float] = None
        self.rc:        Optional[int]  = None

    def reset(self, name: str, max_epoch: int) -> None:
        self.state     = "loading"
        self.name      = name
        self.pct       = 0
        self.step      = 0
        self.total     = 0
        self.speed     = ""
        self.epoch     = 0.0
        self.max_epoch = max_epoch
        self.loss      = None
        self.map50     = None
        self.rc        = None


_slots: Dict[int, _Slot] = {i: _Slot() for i in range(MAX_PARALLEL_JOBS)}
_lock  = threading.Lock()

# ── Regex patterns ─────────────────────────────────────────────────────────────
# Standard training progress tqdm
_RE_TQDM = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s+\[[^\]]+,\s*([\d.]+\s*(?:it/s|s/it))"
)

# YOLO Epoch progress line
#   "      1/300      3.95G      1.196     0.9856      1.228         15        640"
_RE_YOLO_PROGRESS = re.compile(
    r"^\s*(\d+)/(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

# YOLO validation summary line:
#   "      all         81        105      0.548      0.485      0.512      0.325"
_RE_VAL_ALL = re.compile(
    r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def _parse(line: str, s: _Slot) -> None:
    """Update slot in-place from one line of subprocess stdout."""
    # 1. tqdm progress
    m = _RE_TQDM.search(line)
    if m:
        s.pct   = int(m.group(1))
        s.step  = int(m.group(2))
        s.total = int(m.group(3))
        s.speed = m.group(4).strip()
        s.state = "training"
        return

    # 2. YOLO Epoch progress (updates loss to sum of box_loss + cls_loss)
    m = _RE_YOLO_PROGRESS.search(line)
    if m:
        s.epoch     = float(m.group(1))
        s.max_epoch = int(m.group(2))
        box_loss    = float(m.group(3))
        cls_loss    = float(m.group(4))
        s.loss      = box_loss + cls_loss
        return

    # 3. Validation mAP50
    m = _RE_VAL_ALL.search(line)
    if m:
        s.map50 = float(m.group(5))  # mAP50
        return


def _reader(proc: subprocess.Popen, gpu_idx: int, log_fh) -> None:
    """Read subprocess stdout line-by-line; parse metrics, write to log file."""
    s   = _slots[gpu_idx]
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in ("\n", "\r"):
            stripped = buf.strip()
            if stripped:
                log_fh.write(buf + "\n")
                log_fh.flush()
                with _lock:
                    _parse(stripped, s)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        log_fh.write(buf + "\n")
        log_fh.flush()
        with _lock:
            _parse(buf.strip(), s)


# ── Dashboard renderer ─────────────────────────────────────────────────────────
_N = MAX_PARALLEL_JOBS    # always 8 rows — fixed dashboard height

def _bar(pct: int, w: int = 14) -> str:
    n = max(0, min(w, int(w * pct / 100)))
    return "█" * n + "░" * (w - n)

def _fmt(i: int) -> str:
    """Format one slot line (always fits one terminal line)."""
    with _lock:
        s = _slots[i]
        state, name  = s.state, s.name
        pct, step, total, speed = s.pct, s.step, s.total, s.speed
        epoch, max_ep = s.epoch, s.max_epoch
        loss, map50, rc = s.loss, s.map50, s.rc

    c   = _col(i)
    tag = f"{c}{BOLD}[GPU{i}]{RST}"

    if state == "idle":
        return f"{tag} {DIM}idle{RST}"

    name_s = (name[:26] + "…") if len(name) > 27 else name
    tag    = f"{c}{BOLD}[GPU{i}|{name_s}]{RST}"

    if state == "loading":
        return f"{tag}  ⏳ loading…"

    if state == "done":
        parts = []
        if map50 is not None: parts.append(f"mAP50={map50:.4f}")
        if loss  is not None: parts.append(f"loss={loss:.4f}")
        return f"{tag}  {GREEN}✅ done{RST}  " + "  ".join(parts)

    if state == "failed":
        return f"{tag}  {RED}❌ failed (exit {rc}){RST}"

    # training
    ep_s   = f"Ep {int(epoch):>3}/{max_ep}"
    bar_s  = _bar(pct)
    pct_s  = f"{pct:>3}%"
    step_s = f"{step}/{total}"
    parts  = [ep_s, bar_s, pct_s, step_s]
    if speed: parts.append(speed)
    if loss is not None: parts.append(f"loss={loss:.4f}")
    if map50 is not None: parts.append(f"mAP50={map50:.4f}")
    return f"{tag}  {'  '.join(parts)}"


_first_draw = True

def _draw() -> None:
    global _first_draw
    lines = [_fmt(i) for i in range(_N)]
    out   = []
    if _first_draw:
        _first_draw = False
    else:
        out.append(f"\033[{_N}A")          # cursor up N lines
    for line in lines:
        out.append(f"\r\033[2K{line}\n")   # clear line, print, newline
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def _dashboard_loop(stop_evt: threading.Event) -> None:
    sys.stdout.write("\n" * _N)   # reserve N lines
    sys.stdout.flush()
    while not stop_evt.is_set():
        _draw()
        time.sleep(REFRESH_SECS)
    _draw()   # final render


# ── YAML / completion helpers ──────────────────────────────────────────────────
def _load(p: Path) -> Dict[str, Any]:
    with open(p) as f:
        return yaml.safe_load(f)

def _equiv(a: Dict, b: Dict) -> bool:
    for k in ("model_name", "data", "peft", "learning_rate", "num_epochs",
              "yolo_model_config", "yolo_data_yaml"):
        if a.get(k) != b.get(k):
            return False
    return True

def _verify_weights(parent_dir: Path) -> bool:
    weights_dir = parent_dir / "weights"
    return any(
        (weights_dir / name).exists() and (weights_dir / name).stat().st_size > 0
        for name in ("best.pt", "last.pt")
    )

def _done_cfgs(out_dir: Path) -> List[Dict]:
    result = []
    for p in out_dir.glob("**/DONE"):
        if not _verify_weights(p.parent):
            continue
        cfg_p = p.parent / "config.yaml"
        if cfg_p.exists():
            try: result.append(_load(cfg_p))
            except: pass
    return result

def _free_gpus() -> List[int]:
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True, text=True)
        gpus = []
        for line in out.strip().splitlines():
            if not line: continue
            idx_s, mem_s = line.split(",")
            if int(mem_s.strip()) >= MIN_FREE_VRAM_MIB:
                gpus.append(int(idx_s.strip()))
        return gpus
    except:
        return [0]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg_dir = Path("configs/det/ablations")
    out_dir = Path("outputs/det")
    log_dir = out_dir / "logs"

    if not cfg_dir.exists():
        sys.exit("❌  configs/det/ablations/ not found. Run generate_det_ablation_grid.py first.")

    cfg_files = sorted(cfg_dir.glob("*.yaml"))
    if not cfg_files:
        sys.exit("❌  No YAML configs in configs/det/ablations/")

    print(f"🔍  Found {len(cfg_files)} configs.")
    print("🧹  Scanning for completed runs…")
    done = _done_cfgs(out_dir)
    print(f"✓   {len(done)} already completed.\n")

    pending = []
    skipped = 0
    for p in cfg_files:
        try:
            cfg = _load(p)
            if any(_equiv(cfg, d) for d in done):
                skipped += 1
            else:
                pending.append((p, cfg))
        except Exception as e:
            print(f"⚠️   {p.name}: {e}")

    if skipped:
        print(f"⏭️   Skipped {skipped} already-completed configs.")
    print(f"📋  {len(pending)} runs queued.\n")

    if not pending:
        print("🎉  All runs already complete!")
        return

    print(f"{'='*70}")
    print("🚀  Parallel detection ablation training  |  Ctrl+C to stop")
    print(f"{'='*70}")

    log_dir.mkdir(parents=True, exist_ok=True)

    stop_evt = threading.Event()
    dash_t   = threading.Thread(target=_dashboard_loop, args=(stop_evt,), daemon=True)
    dash_t.start()

    # Maps gpu_idx -> (proc, reader_thread, log_file_handle)
    running: Dict[int, tuple] = {}

    try:
        while pending or running:
            # Reap finished jobs
            for gpu_idx in list(running):
                proc, reader_t, log_fh = running[gpu_idx]
                if proc.poll() is not None:
                    reader_t.join(timeout=2)
                    log_fh.close()
                    with _lock:
                        s = _slots[gpu_idx]
                        s.state = "done" if proc.returncode == 0 else "failed"
                        s.rc    = proc.returncode
                    running.pop(gpu_idx)

            # Launch new jobs on free GPUs
            free = [g for g in _free_gpus() if g not in running]
            free = free[:MAX_PARALLEL_JOBS - len(running)]

            while pending and free:
                cfg_path, cfg_dict = pending.pop(0)
                gpu_idx  = free.pop(0)
                cfg_stem = cfg_path.stem
                max_ep   = int(cfg_dict.get("num_epochs", 300))

                with _lock:
                    _slots[gpu_idx].reset(cfg_stem, max_ep)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                env["PYTHONPATH"]           = f"{os.getcwd()}/src:" + env.get("PYTHONPATH", "")
                env["TQDM_NCOLS"]           = "80"
                env["TQDM_MININTERVAL"]     = "10"

                log_fh = open(log_dir / f"{cfg_stem}.log", "w")
                proc   = subprocess.Popen(
                    ["python3", "scripts/train_detection.py", "--config", str(cfg_path)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                reader_t = threading.Thread(
                    target=_reader, args=(proc, gpu_idx, log_fh), daemon=True
                )
                reader_t.start()
                running[gpu_idx] = (proc, reader_t, log_fh)

            time.sleep(5)

    except KeyboardInterrupt:
        stop_evt.set()
        sys.stdout.write(f"\n\033[{_N+1}B")   # move below dashboard
        print("\n🛑  Stopping all jobs…")
        for gpu_idx, (proc, reader_t, log_fh) in running.items():
            proc.terminate()
            reader_t.join(timeout=2)
            log_fh.close()
            try: proc.wait(timeout=5)
            except: proc.kill()
        return

    stop_evt.set()
    dash_t.join(timeout=2)
    sys.stdout.write(f"\n\033[{_N+1}B\n")
    print("🎉  All ablation runs complete!\n")


if __name__ == "__main__":
    main()
