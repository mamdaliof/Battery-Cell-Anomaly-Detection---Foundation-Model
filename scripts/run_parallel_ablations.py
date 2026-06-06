#!/usr/bin/env python3
"""
Parallel ablation runner — compact in-place dashboard.

One persistent terminal line per GPU slot (0-7), refreshed every 0.5s.
All subprocess output is silently parsed for tqdm progress, epoch, loss,
and F1 — nothing else is printed to the terminal.
Full raw output is still written to outputs/logs/<config>.log for debugging.
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


# ── Per-GPU mutable status (written by reader threads, read by dashboard) ──────
class _Slot:
    __slots__ = ("state", "name", "pct", "step", "total",
                 "speed", "epoch", "max_epoch", "loss", "f1", "rc")

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
        self.f1:        Optional[float] = None
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
        self.f1        = None
        self.rc        = None


_slots: Dict[int, _Slot] = {i: _Slot() for i in range(MAX_PARALLEL_JOBS)}
_lock  = threading.Lock()

# ── Regex patterns ─────────────────────────────────────────────────────────────
# Main training tqdm:  " 42%|████▍     | 504/1200 [04:12<05:49,  2.00s/it]"
# We filter on total > 500 to skip the "Loading weights" tqdm (total ≈ 211)
_RE_TQDM = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s+\[[^\]]+,\s*([\d.]+\s*(?:it/s|s/it))"
)
# BeautifulLoggingCallback train line:
#   "📈 [Epoch 3.00 | Step 12] Loss: 0.2134 | LR: 2.00e-04"
_RE_TRAIN = re.compile(r"Epoch\s+([\d.]+).*?Loss:\s*([\d.]+)")
# Eval F1 from BeautifulLoggingCallback:
#   "  🔹 F1                     : 0.8421"
_RE_F1 = re.compile(r"\bF\s*1\s*:\s*([\d.]+)", re.I)
# Eval loss line (only the 🔹 prefixed one to avoid train loss false-positives)
_RE_EVAL_LOSS = re.compile(r"🔹\s*Loss\s*:\s*([\d.]+)", re.I)


def _parse(line: str, s: _Slot) -> None:
    """Update slot in-place from one line of subprocess stdout."""
    # tqdm progress (only the training bar, not weight loading)
    m = _RE_TQDM.search(line)
    if m:
        total = int(m.group(3))
        if total > 500:           # training steps, not the 211-shard weight loader
            s.pct   = int(m.group(1))
            s.step  = int(m.group(2))
            s.total = total
            s.speed = m.group(4).strip()
            s.state = "training"
        return

    # Epoch / train loss from BeautifulLoggingCallback
    m = _RE_TRAIN.search(line)
    if m:
        s.epoch = float(m.group(1))
        s.loss  = float(m.group(2))
        return

    # Eval F1
    m = _RE_F1.search(line)
    if m:
        s.f1 = float(m.group(1))
        return

    # Eval loss (overwrite with the eval-phase value)
    m = _RE_EVAL_LOSS.search(line)
    if m:
        s.loss = float(m.group(1))


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
        loss, f1, rc  = s.loss, s.f1, s.rc

    c   = _col(i)
    tag = f"{c}{BOLD}[GPU{i}]{RST}"

    if state == "idle":
        return f"{tag} {DIM}idle{RST}"

    # Trim long names to keep line width reasonable
    name_s = (name[:26] + "…") if len(name) > 27 else name
    tag    = f"{c}{BOLD}[GPU{i}|{name_s}]{RST}"

    if state == "loading":
        return f"{tag}  ⏳ loading…"

    if state == "done":
        parts = []
        if f1   is not None: parts.append(f"f1={f1:.4f}")
        if loss is not None: parts.append(f"loss={loss:.4f}")
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
    if f1   is not None: parts.append(f"f1={f1:.4f}")
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
    for k in ("model_name", "data", "head", "peft",
              "learning_rate", "num_epochs", "imbalance"):
        if a.get(k) != b.get(k):
            return False
    return True

def _done_cfgs(out_dir: Path) -> List[Dict]:
    result = []
    for p in out_dir.glob("**/DONE"):
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
    cfg_dir = Path("configs/ablations")
    out_dir = Path("outputs")
    log_dir = out_dir / "logs"

    if not cfg_dir.exists():
        sys.exit("❌  configs/ablations/ not found. Run generate_ablation_grid.py first.")

    cfg_files = sorted(cfg_dir.glob("*.yaml"))
    if not cfg_files:
        sys.exit("❌  No YAML configs in configs/ablations/")

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
    print("🚀  Parallel ablation training  |  Ctrl+C to stop")
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
                env["TQDM_MININTERVAL"]     = "10"    # tqdm line every 10s max

                log_fh = open(log_dir / f"{cfg_stem}.log", "w")
                proc   = subprocess.Popen(
                    ["python3", "scripts/train.py", "--config", str(cfg_path)],
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
