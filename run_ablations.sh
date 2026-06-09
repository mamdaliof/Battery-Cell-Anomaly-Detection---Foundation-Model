#!/bin/bash

# Exit immediately on failure
set -e

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo "🚀 Starting Ablation Study execution sequence (58 runs)..."

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/01_baseline_vits16.yaml"
python scripts/train.py --config configs/cls/ablations/01_baseline_vits16.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/02_baseline_vitb16.yaml"
python scripts/train.py --config configs/cls/ablations/02_baseline_vitb16.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/03_lora_vits16_r8_all_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/03_lora_vits16_r8_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/04_lora_vits16_r8_all_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/04_lora_vits16_r8_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/05_lora_vits16_r8_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/05_lora_vits16_r8_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/06_lora_vits16_r8_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/06_lora_vits16_r8_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/07_lora_vits16_r8_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/07_lora_vits16_r8_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/08_lora_vits16_r8_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/08_lora_vits16_r8_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/09_lora_vits16_r16_all_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/09_lora_vits16_r16_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/10_lora_vits16_r16_all_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/10_lora_vits16_r16_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/11_lora_vits16_r16_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/11_lora_vits16_r16_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/12_lora_vits16_r16_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/12_lora_vits16_r16_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/13_lora_vits16_r16_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/13_lora_vits16_r16_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/14_lora_vits16_r16_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/14_lora_vits16_r16_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/15_lora_vitb16_r8_all_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/15_lora_vitb16_r8_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/16_lora_vitb16_r8_all_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/16_lora_vitb16_r8_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/17_lora_vitb16_r8_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/17_lora_vitb16_r8_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/18_lora_vitb16_r8_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/18_lora_vitb16_r8_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/19_lora_vitb16_r8_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/19_lora_vitb16_r8_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/20_lora_vitb16_r8_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/20_lora_vitb16_r8_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/21_lora_vitb16_r16_all_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/21_lora_vitb16_r16_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/22_lora_vitb16_r16_all_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/22_lora_vitb16_r16_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/23_lora_vitb16_r16_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/23_lora_vitb16_r16_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/24_lora_vitb16_r16_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/24_lora_vitb16_r16_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/25_lora_vitb16_r16_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/25_lora_vitb16_r16_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/26_lora_vitb16_r16_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/26_lora_vitb16_r16_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/27_adapter_vits16_d32_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/27_adapter_vits16_d32_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/28_adapter_vits16_d32_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/28_adapter_vits16_d32_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/29_adapter_vits16_d32_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/29_adapter_vits16_d32_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/30_adapter_vits16_d32_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/30_adapter_vits16_d32_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/31_adapter_vits16_d64_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/31_adapter_vits16_d64_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/32_adapter_vits16_d64_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/32_adapter_vits16_d64_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/33_adapter_vits16_d64_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/33_adapter_vits16_d64_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/34_adapter_vits16_d64_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/34_adapter_vits16_d64_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/35_adapter_vitb16_d32_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/35_adapter_vitb16_d32_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/36_adapter_vitb16_d32_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/36_adapter_vitb16_d32_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/37_adapter_vitb16_d32_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/37_adapter_vitb16_d32_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/38_adapter_vitb16_d32_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/38_adapter_vitb16_d32_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/39_adapter_vitb16_d64_last4_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/39_adapter_vitb16_d64_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/40_adapter_vitb16_d64_last4_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/40_adapter_vitb16_d64_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/41_adapter_vitb16_d64_last2_lr0.0003.yaml"
python scripts/train.py --config configs/cls/ablations/41_adapter_vitb16_d64_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/42_adapter_vitb16_d64_last2_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/42_adapter_vitb16_d64_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/43_vpt_vits16_shallow_t10_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/43_vpt_vits16_shallow_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/44_vpt_vits16_shallow_t10_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/44_vpt_vits16_shallow_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/45_vpt_vits16_shallow_t20_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/45_vpt_vits16_shallow_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/46_vpt_vits16_shallow_t20_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/46_vpt_vits16_shallow_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/47_vpt_vits16_deep_last4_t10_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/47_vpt_vits16_deep_last4_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/48_vpt_vits16_deep_last4_t10_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/48_vpt_vits16_deep_last4_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/49_vpt_vits16_deep_last4_t20_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/49_vpt_vits16_deep_last4_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/50_vpt_vits16_deep_last4_t20_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/50_vpt_vits16_deep_last4_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/51_vpt_vitb16_shallow_t10_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/51_vpt_vitb16_shallow_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/52_vpt_vitb16_shallow_t10_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/52_vpt_vitb16_shallow_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/53_vpt_vitb16_shallow_t20_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/53_vpt_vitb16_shallow_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/54_vpt_vitb16_shallow_t20_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/54_vpt_vitb16_shallow_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/55_vpt_vitb16_deep_last4_t10_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/55_vpt_vitb16_deep_last4_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/56_vpt_vitb16_deep_last4_t10_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/56_vpt_vitb16_deep_last4_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/57_vpt_vitb16_deep_last4_t20_lr0.0005.yaml"
python scripts/train.py --config configs/cls/ablations/57_vpt_vitb16_deep_last4_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/cls/ablations/58_vpt_vitb16_deep_last4_t20_lr0.001.yaml"
python scripts/train.py --config configs/cls/ablations/58_vpt_vitb16_deep_last4_t20_lr0.001.yaml

echo "🎉 All 58 ablation runs completed successfully!"
