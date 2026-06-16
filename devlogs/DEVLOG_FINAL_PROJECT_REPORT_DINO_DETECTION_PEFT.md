# Dev log: Final Project Report - DINO-Based Anomaly Detection and PEFT

Date: 2026-06-12

This devlog captures the final project report, analyzing findings from DINO-based classification and detection models compared with YOLO baselines, evaluating parameter-efficient fine-tuning (PEFT) methods, and examining the effects of different label schemes on a small, highly imbalanced battery-cell anomaly detection dataset.

---

## 1. Problem and Data

The project operates under the following constraints:
*   **Data Scarcity & Imbalance**: The dataset of battery cells is small and highly imbalanced, with very few anomaly examples.
*   **Environment**: The inspection setup provides a strict, structured environment with minimal background variation.
*   **Labeling Schemes**: Experiments were run across three distinct label configurations:
    1.  **Full Labels (`all_label`)**: Cell bounds + defect types (classes: abnormal, cell, text).
    2.  **No-Cell Labels (`no_cell`)**: Bypasses general cell structures (classes: abnormal, text).
    3.  **Abnormal-Only (`abnormal_only`)**: Collapses all anomalies into a single class without background/structural labels (class: abnormal).

**Goal**: Compare DINO-based classification and detection to standard YOLO baselines, and analyze how parameter-efficient fine-tuning (PEFT) configurations and label schemes impact detection performance.

---

## 2. Methodology

The implementation consists of two main pipelines:
*   **Anomaly Classification**:
    *   Uses a DINO-based classifier to separate normal from abnormal cells.
    *   Demonstrated strong, robust transfer learning performance, indicating that self-supervised DINO features capture global anomalies well.
*   **Object Detection Architecture**:
    *   **Backbone**: Frozen DINO Vision Transformer (ViT) backbone.
    *   **Neck**: Simple Feature Pyramid (SFP) neck based on Detectron2's ViTDet design. The SFP neck projects flat, single-scale stride-16 patch tokens into multi-scale feature maps (P3, P4, P5). The neck layers are initialized randomly and trained from scratch.
    *   **Head**: YOLO-style neck and head modules (standard Ultralytics layers). The head layers are initialized with pretrained weights and then fine-tuned.
    *   **Backbone Adaptation**: The frozen backbone is adapted using Parameter-Efficient Fine-Tuning (PEFT) techniques, including Low-Rank Adaptation (LoRA), Pfeiffer Bottleneck Adapters, and Visual Prompt Tuning (VPT).
*   **Baseline Comparisons**:
    *   Baseline models utilize standard YOLO-nano and YOLO-small architectures.
    *   **Resolution Differences**: YOLO models are trained at 640px input resolution; the DINO-based models use 256px resolution.
    *   **Training & Parameters**: Schedulers are aligned, though augmentation pipelines differ slightly. Trainable parameter counts are not perfectly matched, so evaluations focus primarily on parameter efficiency.

---

## 3. Key Results & Findings

### 3.1. Classification vs. Detection Performance
*   The DINO classifier effectively isolates abnormal cells on a global scale.
*   For localizing defects, the DINO + FPN + YOLO head combination achieves comparable detection performance to standard YOLO-nano and YOLO-small baselines.
*   Importantly, the DINO pipeline achieves this competitive performance at a lower input resolution (256px vs. 640px) and with significantly fewer trainable parameters. This highlights that PEFT is highly effective for adaptation in low-data, highly structured environments.

### 3.2. Impact of Label Schemes
*   **Full Labels (`all_label`)**: When full labels (including structural cell boundaries) are provided, PEFT adaptation yields a clearer positive impact on detection metrics.
*   **Abnormal-Only (`abnormal_only`)**: In this configuration, **top-tuning** (fine-tuning the top layers of the backbone) performs best among the PEFT variations.
*   **Task Difficulty**: Reducing the available label categories makes localization harder. Auxiliary structural labels (like cell boundaries) provide crucial spatial context and supervision that the PEFT layers can exploit.

### 3.3. Training Fragility
*   Training a Feature Pyramid Network (FPN) from scratch on a small, imbalanced industrial dataset is inherently fragile.
*   Resolution mismatches and augmentation differences between DINO and YOLO baselines mean that these results serve as a study of parameter-efficient adaptation rather than a definitive architectural verdict.

---

## 4. Deep Analysis: Data Limits vs. Model Roles

*   The highly structured nature and limited scale of the battery dataset suggest that performance is currently **data-bound** rather than model-bound.
*   The strong performance of the frozen DINO backbone indicates that its self-supervised features are highly representative of this industrial environment. The core bottlenecks remain data scarcity, class imbalance, and training instability.
*   Auxiliary labels help guide the model's focus, functioning as an implicit spatial attention mechanism. Without them, identifying small anomalies against a cell background becomes significantly harder, decreasing the benefits of PEFT.

---

## 5. Future Directions

*   **Richer Pretraining & Datasets**: Benchmark the models on larger, more diverse industrial anomaly datasets (e.g., MVTec-AD, VisA) to evaluate if DINO-based detectors pull ahead of YOLO as dataset complexity grows.
*   **Aligned Comparison Metrics**: Standardize input resolutions and augmentation strategies between YOLO and DINO detectors, and run comparisons using matched trainable parameter counts.
*   **SFP Neck Pretraining**: Pretrain the SFP neck and detection head on general object detection datasets (e.g., COCO or industrial defect datasets) prior to battery fine-tuning to mitigate initialization fragility.
*   **CNN vs. ViT Backbones**: Implement CNN alternatives (e.g., ConvNeXt) under the same detection head and neck structure to compare spatial inductive biases in highly constrained settings.
*   **Ablation Extensions**:
    *   Conduct deeper parameter sweeps for PEFT (specific layer configurations, rank values, and prompt token counts).
    *   Include a fully fine-tuned DINO baseline on a larger dataset to construct a complete performance-to-trainable-parameters curve.

---

## 6. Limitations and Scope

*   This evaluation is focused strictly on a single, structured battery cell production environment.
*   Parameters analyzed concern training efficiency and localization accuracy. Practical deployment considerations—such as inference latency, runtime memory footprints, and hardware acceleration constraints—remain out of scope for the current phase.
