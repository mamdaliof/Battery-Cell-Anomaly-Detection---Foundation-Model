#!/usr/bin/env python

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="DDP Dummy allocation test.")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds to hold memory")
    args = parser.parse_args()

    # Verify launched via distributed launcher
    if "LOCAL_RANK" not in os.environ:
        print("❌ This script must be launched via torchrun (e.g., torchrun --nproc_per_node=2 ...)")
        sys.exit(1)

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"🚀 Rank {rank}/{world_size} (Local Rank {local_rank}) initialized on {torch.cuda.get_device_name(local_rank)}")

    # Allocate 4 GB (1024^3 float32 = 4 GB)
    try:
        t = torch.zeros((1024, 1024, 1024), dtype=torch.float32, device=device)
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"   ✅ Rank {rank} allocated {allocated_gb:.2f} GB on {device}")
    except Exception as e:
        print(f"   ❌ Rank {rank} allocation failed on {device}: {e}")

    # Hold allocation
    if rank == 0:
        print(f"\n⏳ Holding allocation for {args.duration} seconds. Check 'nvidia-smi' now...")
    
    try:
        for remaining in range(args.duration, 0, -1):
            if rank == 0:
                sys.stdout.write(f"\rTime remaining: {remaining}s ")
                sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # Clean up
    dist.destroy_process_group()
    if rank == 0:
        print("\n✅ Done!")

if __name__ == "__main__":
    main()
