#!/bin/bash

# Exit immediately on failure
set -e

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo "🚀 Starting Ablation Study execution sequence (58 runs)..."

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/01_baseline_vits14.yaml"
python scripts/train.py --config configs/ablations/01_baseline_vits14.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/02_baseline_vitb14.yaml"
python scripts/train.py --config configs/ablations/02_baseline_vitb14.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/03_lora_vits14_r8_all_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/03_lora_vits14_r8_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/04_lora_vits14_r8_all_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/04_lora_vits14_r8_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/05_lora_vits14_r8_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/05_lora_vits14_r8_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/06_lora_vits14_r8_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/06_lora_vits14_r8_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/07_lora_vits14_r8_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/07_lora_vits14_r8_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/08_lora_vits14_r8_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/08_lora_vits14_r8_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/09_lora_vits14_r16_all_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/09_lora_vits14_r16_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/10_lora_vits14_r16_all_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/10_lora_vits14_r16_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/11_lora_vits14_r16_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/11_lora_vits14_r16_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/12_lora_vits14_r16_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/12_lora_vits14_r16_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/13_lora_vits14_r16_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/13_lora_vits14_r16_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/14_lora_vits14_r16_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/14_lora_vits14_r16_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/15_lora_vitb14_r8_all_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/15_lora_vitb14_r8_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/16_lora_vitb14_r8_all_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/16_lora_vitb14_r8_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/17_lora_vitb14_r8_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/17_lora_vitb14_r8_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/18_lora_vitb14_r8_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/18_lora_vitb14_r8_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/19_lora_vitb14_r8_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/19_lora_vitb14_r8_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/20_lora_vitb14_r8_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/20_lora_vitb14_r8_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/21_lora_vitb14_r16_all_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/21_lora_vitb14_r16_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/22_lora_vitb14_r16_all_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/22_lora_vitb14_r16_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/23_lora_vitb14_r16_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/23_lora_vitb14_r16_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/24_lora_vitb14_r16_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/24_lora_vitb14_r16_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/25_lora_vitb14_r16_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/25_lora_vitb14_r16_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/26_lora_vitb14_r16_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/26_lora_vitb14_r16_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/27_adapter_vits14_d32_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/27_adapter_vits14_d32_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/28_adapter_vits14_d32_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/28_adapter_vits14_d32_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/29_adapter_vits14_d32_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/29_adapter_vits14_d32_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/30_adapter_vits14_d32_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/30_adapter_vits14_d32_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/31_adapter_vits14_d64_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/31_adapter_vits14_d64_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/32_adapter_vits14_d64_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/32_adapter_vits14_d64_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/33_adapter_vits14_d64_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/33_adapter_vits14_d64_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/34_adapter_vits14_d64_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/34_adapter_vits14_d64_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/35_adapter_vitb14_d32_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/35_adapter_vitb14_d32_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/36_adapter_vitb14_d32_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/36_adapter_vitb14_d32_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/37_adapter_vitb14_d32_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/37_adapter_vitb14_d32_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/38_adapter_vitb14_d32_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/38_adapter_vitb14_d32_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/39_adapter_vitb14_d64_last4_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/39_adapter_vitb14_d64_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/40_adapter_vitb14_d64_last4_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/40_adapter_vitb14_d64_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/41_adapter_vitb14_d64_last2_lr0.0003.yaml"
python scripts/train.py --config configs/ablations/41_adapter_vitb14_d64_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/42_adapter_vitb14_d64_last2_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/42_adapter_vitb14_d64_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/43_vpt_vits14_shallow_t10_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/43_vpt_vits14_shallow_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/44_vpt_vits14_shallow_t10_lr0.001.yaml"
python scripts/train.py --config configs/ablations/44_vpt_vits14_shallow_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/45_vpt_vits14_shallow_t20_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/45_vpt_vits14_shallow_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/46_vpt_vits14_shallow_t20_lr0.001.yaml"
python scripts/train.py --config configs/ablations/46_vpt_vits14_shallow_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/47_vpt_vits14_deep_last4_t10_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/47_vpt_vits14_deep_last4_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/48_vpt_vits14_deep_last4_t10_lr0.001.yaml"
python scripts/train.py --config configs/ablations/48_vpt_vits14_deep_last4_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/49_vpt_vits14_deep_last4_t20_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/49_vpt_vits14_deep_last4_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/50_vpt_vits14_deep_last4_t20_lr0.001.yaml"
python scripts/train.py --config configs/ablations/50_vpt_vits14_deep_last4_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/51_vpt_vitb14_shallow_t10_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/51_vpt_vitb14_shallow_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/52_vpt_vitb14_shallow_t10_lr0.001.yaml"
python scripts/train.py --config configs/ablations/52_vpt_vitb14_shallow_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/53_vpt_vitb14_shallow_t20_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/53_vpt_vitb14_shallow_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/54_vpt_vitb14_shallow_t20_lr0.001.yaml"
python scripts/train.py --config configs/ablations/54_vpt_vitb14_shallow_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/55_vpt_vitb14_deep_last4_t10_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/55_vpt_vitb14_deep_last4_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/56_vpt_vitb14_deep_last4_t10_lr0.001.yaml"
python scripts/train.py --config configs/ablations/56_vpt_vitb14_deep_last4_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/57_vpt_vitb14_deep_last4_t20_lr0.0005.yaml"
python scripts/train.py --config configs/ablations/57_vpt_vitb14_deep_last4_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/ablations/58_vpt_vitb14_deep_last4_t20_lr0.001.yaml"
python scripts/train.py --config configs/ablations/58_vpt_vitb14_deep_last4_t20_lr0.001.yaml

echo "🎉 All 58 ablation runs completed successfully!"
