# DINO-Based Object Detection Options for Battery Cell Anomaly Detection

This report evaluates and reviews the integration of DINOv2/DINOv3 backbones and DINO-based object detection frameworks for anomaly detection (such as defect localization on battery cells). It covers the three user-proposed approaches, lists key technical challenges, and introduces alternative modern paradigms.

---

## Approach 1: Fine-Tuning Grounding DINO for Anomaly Detection

[Grounding DINO](https://arxiv.org/abs/2303.05499) is an open-vocabulary (zero-shot) object detector that merges the transformer-based [DINO detector](https://arxiv.org/abs/2203.03605) with text grounding. By passing text prompts like `"burnt area"` or `"crack"`, the model localizes the target objects.

*   **Architecture & Workflow**:
    1.  **Data Format**: Requires bounding boxes labeled in standard COCO format.
    2.  **Prompt Selection**: Fine-tuning matches domain-specific visual representations to semantic tags (e.g. text phrases).
    3.  **PEFT Integration**: Because the model is complex, applying **LoRA (Low-Rank Adaptation)** is recommended to avoid overfitting on small datasets of anomalies.
*   **Key References**:
    *   **Official Repository**: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
    *   **Updated API (v1.5)**: [IDEA-Research/Grounding-DINO-1.5-API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)
    *   **Fine-Tuning Example**: [techwolf/Grounding-Dino-FineTuning](https://github.com/techwolf/Grounding-Dino-FineTuning) provides custom dataset loaders and training scripts.

---

## Approach 2: YOLO26 (or YOLOv8) with DINOv2/v3 Backbone/Neck

This approach couples the self-supervised dense representations of DINOv2/DINOv3 with YOLO's real-time necks (like C3k2, SPPF, C2PSA) and head.

*   **About YOLO26**: Ultralytics YOLO26 is a real-time computer vision family emphasizing end-to-end NMS-free inference and DFL-free regression to simplify hardware export (ONNX/TensorRT).
*   **Architecture & Workflow**:
    1.  **Backbone Swap**: Swapping YOLO's default CNN (CSP-Darknet) for a DINO vision transformer (ViT). This requires writing a custom YAML config mapping ViT block outputs to YOLO's Neck.
    2.  **Hybrid Injection**: Injecting DINO global features as an auxiliary context layer into the neck rather than replacing the backbone entirely.
*   **Technical Challenges**:
    *   **Multi-Scale Neck Alignment**: YOLO's neck expects multi-scale feature maps. Since ViT outputs are single-scale (flat patch embeddings), a projection layer (like a Feature Pyramid Network) is necessary.
    *   **Latency Overhead**: DINO backbones are significantly slower than CSP-Darknet, compromising YOLO's real-time capabilities without TensorRT optimization.
*   **Key References**:
    *   **DINO-YOLO Integration**: [itsprakhar/Yolo-DinoV2](https://github.com/itsprakhar/Yolo-DinoV2)
    *   **YOLO26 Framework**: [Ultralytics YOLO26 Documentation](https://github.com/ultralytics/ultralytics)

### Concrete Implementation & Ablation Summary (Approach 2)

We have fully implemented, trained, and evaluated this framework:
1.  **Model Structure**: Integrates DINOv3 backbones (ViT-S/16 and ViT-B/16) with a **Simple Feature Pyramid (SFP)** neck to feed multi-scale representation maps directly to YOLO26's detection head.
2.  **Unified Metric Conversion**: Formulated a box-to-image conversion layer during validation. Bounding box detections are evaluated at a decision threshold (0.25 confidence) to yield image-level abnormal classification indicators. This computes Accuracy, Precision, Recall, F1, and AUROC side-by-side with classification runs.
3.  **Ablation Sweep (58 Runs)**: Executed a grid sweep across:
    - **Backbones**: DINOv3 ViT-S/16 and ViT-B/16.
    - **PEFT Methods**: LoRA (ranks 8/16, targeting attention projections), Bottleneck Adapters (dimensions 32/64), and Visual Prompt Tuning (VPT; shallow/deep, 10/20 tokens).
    - **Hyperparameters**: Learning rates (3e-4, 5e-4, 1e-3).
4.  **Dashboard Integration**: Completed runs are saved in a unified `trainer_state.json` schema inside `outputs/det/`, enabling direct leaderboard rankings and trajectory plots comparing classification vs. detection performance.

---

## Approach 3: DINOv2/v3 Backbone with Faster/Mask R-CNN Head

This approach treats DINOv2/v3 as a pure feature extractor and attaches standard convolutional detection heads (Faster R-CNN or Mask R-CNN).

*   **Architecture & Workflow**:
    1.  **Feature Map Adapter**: A standard ViT backbone does not output hierarchical resolution layers (like ResNet's $C_2, C_3, C_4, C_5$).
    2.  **Simple Feature Pyramid (SFP)**: Leveraging the [ViTDet framework](https://arxiv.org/abs/2203.16527), which constructs multi-scale pyramids from a single, flat ViT feature map by using upsampling/downsampling convolutions.
*   **Technical Challenges**:
    *   **Convergence**: Standard Faster R-CNN heads struggle to converge with direct ViT features unless properly adapted via FPN/SFP.
*   **Key References**:
    *   **ViTDet Paper**: [ViTDet: Exploring Plain Vision Transformer Backbones for Object Detection (Li et al., 2022)](https://arxiv.org/abs/2203.16527)
    *   **Community Discussions**: [facebookresearch/dinov2 GitHub Issue #350](https://github.com/facebookresearch/dinov2/issues/350) covers the practical mechanics of wrapping DINOv2 for Faster R-CNN/Mask R-CNN.

---

## Alternative Paradigms

### 1. RT-DETR with DINOv2/v3 Backbone
[RT-DETR (Real-Time DEtection TRansformer)](https://arxiv.org/abs/2304.08069) from Baidu is an end-to-end detector that replaces the YOLO family in many applications due to its superior accuracy and training speed.
*   **RF-DETR**: Integrates DINOv2 directly with RT-DETR. Refer to [roboflow/rf-detr](https://github.com/roboflow/rf-detr).
*   **DEIMv2**: A state-of-the-art detection framework using DINOv3 backbone features with RT-DETR. Refer to [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2).

### 2. Knowledge Distillation (DINO -> YOLO)
Instead of running a heavy Vision Transformer backbone at inference time, researchers use DINOv2/v3 as a **teacher** model to distill rich, dense features into a standard, fast YOLOv8/v26 **student** backbone during training. This retains YOLO's speed while transferring DINO's anomaly detection capabilities.
*   **Reference Framework**: [lightly-ai/lightly-train](https://github.com/lightly-ai/lightly-train) supports training student architectures via vision model distillation.
