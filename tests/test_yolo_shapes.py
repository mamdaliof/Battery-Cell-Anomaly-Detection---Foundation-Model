import sys
import unittest
import torch
import yaml
import os
from pathlib import Path

# Add project src/ directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

# Register custom layers before importing YOLO
from bcadfm.utils.yolo_utils import register_yolo_dino
register_yolo_dino()

from ultralytics import YOLO

class TestYoloShapes(unittest.TestCase):
    """
    Unit tests to verify the feature map resolutions, dynamic position embedding
    interpolation, registration hooks, and final prediction shapes of the
    integrated YOLO26 + DINOv3 SFP object detection model.

    Why We Have It:
    This test ensures that the custom DINOv3 backbone is successfully registered
    within Ultralytics, can build without stride/shape resolution errors, and
    correctly propagates the feature map grids down the PANet FPN hierarchy.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up a temporary test configuration referencing a small DINOv3 model from Hugging Face
        to test structural mapping and token grid slicing.
        """
        cls.config_path = str(project_root / "configs" / "det" / "yolo26_dino.yaml")
        cls.temp_config_path = str(project_root / "configs" / "det" / "yolo26_dino_temp.yaml")
        
        with open(cls.config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        # Swap model name to small open-access facebook/dinov3-vits16-pretrain-lvd1689m
        # (hidden size 384, patch size 16) to speed up execution
        cfg["backbone"][0][3] = [384, "facebook/dinov3-vits16-pretrain-lvd1689m"]
        
        with open(cls.temp_config_path, "w") as f:
            yaml.safe_dump(cfg, f)
            
        print(f"\n🔬 Instantiating YOLO model from test config: {cls.temp_config_path}")
        cls.model = YOLO(cls.temp_config_path)
        
    @classmethod
    def tearDownClass(cls):
        """
        Clean up the temporary YOLO config files.
        """
        if hasattr(cls, 'temp_config_path') and os.path.exists(cls.temp_config_path):
            try:
                os.remove(cls.temp_config_path)
                print(f"🧹 Cleaned up temporary test config: {cls.temp_config_path}")
            except Exception as e:
                print(f"⚠️ Failed to remove temp config {cls.temp_config_path}: {e}")
        
    def test_backbone_and_sfp_shapes(self):
        """
        Verify that intermediate layers in the model exist, have the correct attributes,
        and preserve their metadata properties (i, f, type, np).

        How It Should Behave:
        The custom layers must be instantiated at indices 0 to 3 in the PyTorch model list,
        and should possess the required attributes and expected hidden dimensions.
        """
        py_model = self.model.model
        
        # Verify DINOv3 backbone exists at layer index 0
        backbone = py_model.model[0]
        from bcadfm.models.yolo_dino import DinoV3Backbone
        self.assertTrue(isinstance(backbone, DinoV3Backbone))
        self.assertEqual(backbone.hidden_size, 384) # ViT-S has hidden dim 384
        
        # Verify SFP neck layers exist at indices 1, 2, 3
        from bcadfm.models.yolo_dino import DinoV3SFP_P3, DinoV3SFP_P4, DinoV3SFP_P5
        self.assertTrue(isinstance(py_model.model[1], DinoV3SFP_P3))
        self.assertTrue(isinstance(py_model.model[2], DinoV3SFP_P4))
        self.assertTrue(isinstance(py_model.model[3], DinoV3SFP_P5))

        # Verify Metadata Preservation (H10 Fix checking)
        # Verify that index 'i', input 'f', type 'type', and parameter count 'np' attributes are preserved
        for idx in range(4):
            module = py_model.model[idx]
            self.assertTrue(hasattr(module, "i"), f"Module {idx} missing 'i' attribute")
            self.assertTrue(hasattr(module, "f"), f"Module {idx} missing 'f' attribute")
            self.assertTrue(hasattr(module, "type"), f"Module {idx} missing 'type' attribute")
            self.assertTrue(hasattr(module, "np"), f"Module {idx} missing 'np' attribute")

    def test_model_forward_pass_shapes(self):
        """
        Verify that a forward pass with a dummy input image of size 640x640
        runs without errors and yields the expected prediction tensor shapes.

        How It Should Behave:
        The model forward pass should compile dynamically using positional embedding
        interpolation, slice the token grid after register tokens, and output
        the bounding boxes prediction tensor of shape [batch, 4 + nc, num_anchors].
        """
        # input image of size 640x640
        x = torch.randn(1, 3, 640, 640)
        
        # Run forward pass (in evaluation mode)
        self.model.model.eval()
        with torch.no_grad():
            output = self.model.model(x)
            
        if isinstance(output, tuple):
            pred_tensor = output[0]
        else:
            pred_tensor = output
            
        print(f"  - Inference output shape: {pred_tensor.shape}")
        
        # Check batch size
        self.assertEqual(pred_tensor.shape[0], 1)
        
        # Check shape (either raw anchor outputs or post-processed predictions)
        if pred_tensor.shape[1] == 5:
            # Raw predictions: [batch, 4 + nc, num_anchors]
            # num_anchors = 80*80 (P3) + 40*40 (P4) + 20*20 (P5) = 8400
            self.assertEqual(pred_tensor.shape[2], 8400)
        else:
            # Post-processed detections: [batch, max_det, 6]
            self.assertEqual(pred_tensor.shape[1], 300)
            self.assertEqual(pred_tensor.shape[2], 6)

if __name__ == "__main__":
    unittest.main()
