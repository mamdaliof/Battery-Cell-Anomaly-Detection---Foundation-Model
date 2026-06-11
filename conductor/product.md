# Product Definition - Battery Cell Anomaly Detection - Foundation Model

## Vision
A computer vision project utilizing frozen DINOv3 backbones and parameter-efficient fine-tuning (PEFT) for detection and classification and perform ablation study.

## Problem Statement
Identifying anomalous battery cells from images under severe class imbalance (normal vs abnormal) to ensure quality control, while maximizing F1-score or mAP metric. It also provides a basis for scaling the project to other tasks and other datasets that might not face imbalance problems. Also, it compares the hyperparameter effects.

## Target Users
Machine learning researchers and engineers studying or building parameter-efficient fine-tuning (PEFT) pipelines for using foundation models and ViT based models like DINO-vit to train models for detection and multi-label classification.

## Core Pipelines

### 1. Classification Pipeline (`DinoV3Classifier`)
- **Backbone**: Frozen DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) or standard ViT.
- **Head**: Configurable Multi-Layer Perceptron (MLP) classification head, trainable.
- **Integration**: Hugging Face `peft` wraps ONLY the backbone, keeping the classification head fully trainable.
- **Goal**: Binary classification (normal vs. abnormal cells) on skewed distributions.

### 2. Object Detection Pipeline (YOLO26 + DINOv3 SFP)
- **Backbone**: Frozen DINOv3 backbone.
- **Neck**: Simple Feature Pyramid (SFP) neck generating multi-scale representations at stride 8 (P3), stride 16 (P4), and stride 32 (P5).
- **Head**: Standard Ultralytics YOLO26 detection head and native losses.
- **Integration**: Dynamic custom module registration hooks inside standard Ultralytics task parser.

## Key Metrics

### Classification Metrics
- **Primary**: F1-score (with emphasis on the anomalous class).
- **Secondary**: Accuracy, Precision, Recall, AUROC.
- **Diagnostics**: Confusion matrix absolute counts (TN, FP, FN, TP).

### Object Detection Metrics
- **Primary**: mAP50, mAP50-95.
- **Custom**: Bounding box overlap IoU, Dice Coefficient.
- **Image-level indicators**: Image-level classification accuracy, precision, recall, F1, and AUROC computed based on max box confidence.

## Key Goals
1. Implement parameter-efficient training using foundation models like DINOv3 for multi-label classification and object detection.
2. Analyze the impact of different hyperparameter choices (such as ranks, bottleneck dimensions, and loss functions) on performance.
3. Establish an ablation study framework to compare PEFT methods (LoRA, Adapters, VPT) vs frozen baselines.
4. Design a modular codebase structure that scales to other vision tasks, models, and datasets.
5. Maximize F1-score on classification and mAP on detection tasks for the anomalous class under high class imbalance.
