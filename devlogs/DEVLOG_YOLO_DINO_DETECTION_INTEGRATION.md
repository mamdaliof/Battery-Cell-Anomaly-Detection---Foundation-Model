# 🛠️ Dev log: YOLO26 + DINOv3 SFP Object Detection Pipeline Integration

Date: 2026-06-07

This log documents the design, implementation, and verification of the frozen DINOv3 vision backbone and Simple Feature Pyramid (SFP) neck integrated with the standard Ultralytics YOLO26 object detection model.

---

## 1. 📐 Multi-Scale Feature Alignment (SFP Neck)

- **Problem**: Vision Transformers output a single sequence of flat patch tokens at a fixed stride 16. The downstream YOLO26 neck and detection head expect multi-scale features at stride 8 (P3), stride 16 (P4), and stride 32 (P5) to perform multi-scale detection.
- **Solution**: Implemented a ViTDet-style Simple Feature Pyramid (SFP) neck in `src/bcadfm/models/yolo_dino.py`:
  - **`DinoV3SFP_P3` (Stride 8)**: Projects the stride 16 token grid, upsamples it 2x using `nn.ConvTranspose2d` with kernel size 2 and stride 2, and applies standard SiLU, BatchNorm2d, and 3x3 spatial smoothing.
  - **`DinoV3SFP_P4` (Stride 16)**: Direct projection and 3x3 spatial smoothing of the stride 16 features.
  - **`DinoV3SFP_P5` (Stride 32)**: Projects features, downsamples them 2x using `nn.MaxPool2d` with kernel size 2 and stride 2, and applies standard SiLU, BatchNorm2d, and 3x3 spatial smoothing.

---

## 2. 🔌 Dynamic Registration & Attribute Preservation

- **Problem**: Ultralytics parses layers from a YAML configuration file natively. Custom modules are not registered in the `base_modules` of Ultralytics' `parse_model`. Additionally, when intercepting parsing to swap in custom modules, the model crashed with:
  `AttributeError: 'DinoV3Backbone' object has no attribute 'f'`
- **Cause**: The Ultralytics parser attaches vital runtime attributes (`i` for module index, `f` for input source, `type` for layer type, and `np` for number of parameters) to placeholder modules. When swapping out these placeholders for custom modules, these attributes were lost.
- **Fix**: Implemented a global patching script `src/bcadfm/utils/yolo_utils.py` that overrides `ultralytics.nn.tasks.parse_model`. During module reconstruction, the patcher copies all metadata attributes (`i`, `f`, `type`, `np`) from the placeholder module to the constructed custom `actual_layer` before inserting it into the model layout.

---

## 3. 🖼️ Dynamic Input Resolution Support (Positional Embedding Interpolation)

- **Problem**: During compilation, the Ultralytics framework automatically runs dummy forward passes with size $256 \times 256$ to calculate layer strides. However, standard vision transformer backbones (like ViT-Base used in test) raise a `ValueError` if the input image size does not match their pre-trained size ($224 \times 224$).
- **Solution**: Modified the frozen backbone's forward pass in `DinoV3Backbone` to invoke the transformer model with `interpolate_pos_encoding=True`. This allows Hugging Face ViT/DINOv2 architectures to dynamically interpolate positional embeddings, supporting any arbitrary image size during training and inference.

---

## 4. 🧪 Shapes Verification Test Suite

- **Implementation**: Created `tests/test_yolo_shapes.py` to verify compilation, attribute preservation, and shape propagation:
  - Dynamically builds a temporary configuration swapping the gated DINOv3 backbone for the open-access `google/vit-base-patch16-224` model, allowing tests to run locally without Hugging Face login tokens.
  - Verifies that custom layer classes (`DinoV3Backbone`, `DinoV3SFP_P3`, `DinoV3SFP_P4`, `DinoV3SFP_P5`) exist at the correct indices in the compiled sequential model.
  - Verifies that a forward pass with a dummy input image of size $640 \times 640$ completes without errors and returns the expected prediction tensor shape.
- **Result**: The test ran and passed successfully in the `pytorch` environment, yielding a correct post-processed prediction shape of `[1, 300, 6]`.

---

## 5. 📈 Status

- All custom layers, register helpers, and test scripts are committed to git and verified.
- The pipeline utilizes **standard YOLO losses and training loops** to ensure backbone performance comparison is completely isolated.
