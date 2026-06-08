#!/usr/bin/env python

import argparse
import time
import torch
import sys

def main():
    parser = argparse.ArgumentParser(description="Allocate 4 GB of memory on visible GPUs.")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds to hold memory")
    parser.add_argument("--gpus", type=str, default="all", help="Comma-separated local GPU indices (e.g. '0,1') or 'all'")
    args = parser.parse_args()

    print("==================================================")
    print("🔋 GPU MEMORY ALLOCATION TEST")
    print("==================================================")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this machine!")
        return

    num_visible = torch.cuda.device_count()
    print(f"Visible GPUs count: {num_visible}")
    for idx in range(num_visible):
        print(f"  - Device {idx}: {torch.cuda.get_device_name(idx)}")

    # Determine which devices to allocate on
    if args.gpus == "all":
        devices = list(range(num_visible))
    else:
        devices = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    tensors = []
    print("\n🔹 Allocating ~4 GB on target devices...")
    for dev_idx in devices:
        if dev_idx >= num_visible:
            print(f"⚠️ Device index {dev_idx} is out of visible range (0-{num_visible-1})!")
            continue

        device = torch.device(f"cuda:{dev_idx}")
        print(f"👉 Allocating on {device} ({torch.cuda.get_device_name(dev_idx)})...")

        # 1024 * 1024 * 1024 float32 elements = 4 GB
        try:
            t = torch.zeros((1024, 1024, 1024), dtype=torch.float32, device=device)
            tensors.append(t)
            allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
            print(f"   ✅ Allocated: {allocated_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB")
        except Exception as e:
            print(f"   ❌ Allocation failed on device {dev_idx}: {e}")

    print(f"\n⏳ Holding allocation for {args.duration} seconds. Check 'nvidia-smi' now...")
    try:
        for remaining in range(args.duration, 0, -1):
            sys.stdout.write(f"\rTime remaining: {remaining}s ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user.")
    
    print("\n🧹 Releasing memory...")
    del tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Done!")

if __name__ == "__main__":
    main()
