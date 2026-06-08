#!/bin/bash

# Exit immediately on failure
set -e

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo "🚀 Starting YOLO Detection Ablation Study execution sequence (58 runs)..."

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/01_baseline_vits16.yaml"
python scripts/train_detection.py --config configs/det/ablations/01_baseline_vits16.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/02_baseline_vitb16.yaml"
python scripts/train_detection.py --config configs/det/ablations/02_baseline_vitb16.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/03_lora_vits16_r8_all_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/03_lora_vits16_r8_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/04_lora_vits16_r8_all_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/04_lora_vits16_r8_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/05_lora_vits16_r8_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/05_lora_vits16_r8_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/06_lora_vits16_r8_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/06_lora_vits16_r8_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/07_lora_vits16_r8_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/07_lora_vits16_r8_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/08_lora_vits16_r8_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/08_lora_vits16_r8_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/09_lora_vits16_r16_all_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/09_lora_vits16_r16_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/10_lora_vits16_r16_all_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/10_lora_vits16_r16_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/11_lora_vits16_r16_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/11_lora_vits16_r16_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/12_lora_vits16_r16_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/12_lora_vits16_r16_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/13_lora_vits16_r16_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/13_lora_vits16_r16_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/14_lora_vits16_r16_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/14_lora_vits16_r16_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/15_lora_vitb16_r8_all_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/15_lora_vitb16_r8_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/16_lora_vitb16_r8_all_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/16_lora_vitb16_r8_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/17_lora_vitb16_r8_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/17_lora_vitb16_r8_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/18_lora_vitb16_r8_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/18_lora_vitb16_r8_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/19_lora_vitb16_r8_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/19_lora_vitb16_r8_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/20_lora_vitb16_r8_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/20_lora_vitb16_r8_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/21_lora_vitb16_r16_all_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/21_lora_vitb16_r16_all_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/22_lora_vitb16_r16_all_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/22_lora_vitb16_r16_all_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/23_lora_vitb16_r16_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/23_lora_vitb16_r16_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/24_lora_vitb16_r16_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/24_lora_vitb16_r16_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/25_lora_vitb16_r16_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/25_lora_vitb16_r16_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/26_lora_vitb16_r16_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/26_lora_vitb16_r16_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/27_adapter_vits16_d32_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/27_adapter_vits16_d32_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/28_adapter_vits16_d32_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/28_adapter_vits16_d32_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/29_adapter_vits16_d32_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/29_adapter_vits16_d32_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/30_adapter_vits16_d32_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/30_adapter_vits16_d32_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/31_adapter_vits16_d64_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/31_adapter_vits16_d64_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/32_adapter_vits16_d64_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/32_adapter_vits16_d64_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/33_adapter_vits16_d64_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/33_adapter_vits16_d64_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/34_adapter_vits16_d64_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/34_adapter_vits16_d64_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/35_adapter_vitb16_d32_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/35_adapter_vitb16_d32_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/36_adapter_vitb16_d32_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/36_adapter_vitb16_d32_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/37_adapter_vitb16_d32_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/37_adapter_vitb16_d32_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/38_adapter_vitb16_d32_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/38_adapter_vitb16_d32_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/39_adapter_vitb16_d64_last4_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/39_adapter_vitb16_d64_last4_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/40_adapter_vitb16_d64_last4_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/40_adapter_vitb16_d64_last4_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/41_adapter_vitb16_d64_last2_lr0.0003.yaml"
python scripts/train_detection.py --config configs/det/ablations/41_adapter_vitb16_d64_last2_lr0.0003.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/42_adapter_vitb16_d64_last2_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/42_adapter_vitb16_d64_last2_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/43_vpt_vits16_shallow_t10_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/43_vpt_vits16_shallow_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/44_vpt_vits16_shallow_t10_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/44_vpt_vits16_shallow_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/45_vpt_vits16_shallow_t20_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/45_vpt_vits16_shallow_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/46_vpt_vits16_shallow_t20_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/46_vpt_vits16_shallow_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/47_vpt_vits16_deep_last4_t10_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/47_vpt_vits16_deep_last4_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/48_vpt_vits16_deep_last4_t10_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/48_vpt_vits16_deep_last4_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/49_vpt_vits16_deep_last4_t20_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/49_vpt_vits16_deep_last4_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/50_vpt_vits16_deep_last4_t20_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/50_vpt_vits16_deep_last4_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/51_vpt_vitb16_shallow_t10_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/51_vpt_vitb16_shallow_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/52_vpt_vitb16_shallow_t10_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/52_vpt_vitb16_shallow_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/53_vpt_vitb16_shallow_t20_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/53_vpt_vitb16_shallow_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/54_vpt_vitb16_shallow_t20_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/54_vpt_vitb16_shallow_t20_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/55_vpt_vitb16_deep_last4_t10_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/55_vpt_vitb16_deep_last4_t10_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/56_vpt_vitb16_deep_last4_t10_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/56_vpt_vitb16_deep_last4_t10_lr0.001.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/57_vpt_vitb16_deep_last4_t20_lr0.0005.yaml"
python scripts/train_detection.py --config configs/det/ablations/57_vpt_vitb16_deep_last4_t20_lr0.0005.yaml

echo "----------------------------------------------------------------"
echo "🏃 Running config: configs/det/ablations/58_vpt_vitb16_deep_last4_t20_lr0.001.yaml"
python scripts/train_detection.py --config configs/det/ablations/58_vpt_vitb16_deep_last4_t20_lr0.001.yaml

echo "🎉 All 58 detection ablation runs completed successfully!"
