#!/usr/bin/env python3
import os
import glob
import sys
sys.path.append(os.path.abspath("scripts"))
from check_ablation_status import check_outputs

def main():
    # Get completed runs from the existing status check function
    completed_runs = check_outputs()
    
    # Convert to list of dicts if it is a pandas DataFrame
    if hasattr(completed_runs, "to_dict"):
        completed_runs_list = completed_runs.to_dict(orient="records")
    else:
        completed_runs_list = completed_runs

    # Extract the matched configs
    completed_configs = set()
    for run in completed_runs_list:
        if run.get("matched_cfg"):
            completed_configs.add(os.path.abspath(run["matched_cfg"]))

    print("\n" + "=" * 80)
    print("🔎 CHECKING FOR MISSING / UNSTARTED ABLATION RUNS")
    print("=" * 80)

    # 1. Check CLS configurations
    cls_configs = glob.glob("configs/cls/ablations/*.yaml")
    missing_cls = []
    for cfg in cls_configs:
        abs_cfg = os.path.abspath(cfg)
        if abs_cfg not in completed_configs:
            missing_cls.append(cfg)

    # 2. Check DET configurations
    det_configs = glob.glob("configs/det/ablations/*.yaml")
    missing_det = []
    for cfg in det_configs:
        abs_cfg = os.path.abspath(cfg)
        if abs_cfg not in completed_configs:
            missing_det.append(cfg)

    print(f"\n📝 Classification (CLS) Ablations:")
    print(f"  Total Configs: {len(cls_configs)}")
    print(f"  Completed:     {len(cls_configs) - len(missing_cls)}")
    print(f"  Not Started:   {len(missing_cls)}")
    if missing_cls:
        print("\n  ❌ Missing CLS Configs:")
        for cfg in sorted(missing_cls):
            print(f"    - {cfg}")

    print(f"\n📝 Detection (DET) Ablations:")
    print(f"  Total Configs: {len(det_configs)}")
    print(f"  Completed:     {len(det_configs) - len(missing_det)}")
    print(f"  Not Started:   {len(missing_det)}")
    if missing_det:
        print("\n  ❌ Missing DET Configs:")
        for cfg in sorted(missing_det):
            print(f"    - {cfg}")

if __name__ == "__main__":
    main()
