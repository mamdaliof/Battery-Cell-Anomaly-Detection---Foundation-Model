import sys
import unittest
import torch
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
    Unit tests to verify the feature map resolutions and final prediction shapes
    of the YOLO26 + DINOv3 SFP object detection model.
    """
    
    @classmethod
    def setUpClass(cls):
        # Build a temporary test config that uses a non-gated model to avoid HF 401 authentication errors locally
        import yaml
        cls.config_path = str(project_root / "configs" / "yolo26_dino.yaml")
        cls.temp_config_path = str(project_root / "configs" / "yolo26_dino_temp.yaml")
        
        with open(cls.config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        # Swap model name to open-access google/vit-base-patch16-224 (hidden size 768, patch size 16)
        cfg["backbone"][0][3] = [768, "google/vit-base-patch16-224"]
        
        with open(cls.temp_config_path, "w") as f:
            yaml.safe_dump(cfg, f)
            
        print(f"\n🔬 Instantiating YOLO model from test config: {cls.temp_config_path}")
        cls.model = YOLO(cls.temp_config_path)
        
    @classmethod
    def tearDownClass(cls):
        import os
        if hasattr(cls, 'temp_config_path') and os.path.exists(cls.temp_config_path):
            try:
                os.remove(cls.temp_config_path)
                print(f"🧹 Cleaned up temporary test config: {cls.temp_config_path}")
            except Exception as e:
                print(f"⚠️ Failed to remove temp config {cls.temp_config_path}: {e}")
        
    def test_backbone_and_sfp_shapes(self):
        """
        Verify that intermediate layers in the model exist and have the correct attributes.
        """
        py_model = self.model.model
        
        # Verify DINOv3 backbone exists at layer index 0
        backbone = py_model.model[0]
        from bcadfm.models.yolo_dino import DinoV3Backbone
        self.assertTrue(isinstance(backbone, DinoV3Backbone))
        self.assertEqual(backbone.hidden_size, 768) # ViT-B/16 has hidden dim 768
        
        # Verify SFP neck layers exist at indices 1, 2, 3
        from bcadfm.models.yolo_dino import DinoV3SFP_P3, DinoV3SFP_P4, DinoV3SFP_P5
        self.assertTrue(isinstance(py_model.model[1], DinoV3SFP_P3))
        self.assertTrue(isinstance(py_model.model[2], DinoV3SFP_P4))
        self.assertTrue(isinstance(py_model.model[3], DinoV3SFP_P5))

    def test_model_forward_pass_shapes(self):
        """
        Verify that a forward pass with a dummy input image of size 640x640
        runs without errors and yields the expected prediction tensor shapes.
        """
        # input image of size 640x640
        x = torch.randn(1, 3, 640, 640)
        
        # Run forward pass (in evaluation mode)
        self.model.model.eval()
        with torch.no_grad():
            output = self.model.model(x)
            
        # Standard YOLO detection head output in inference mode is a tuple or a tensor.
        # In eval/inference mode, it returns a list of prediction outputs, where the first element
        # is the concatenated prediction tensor of shape (B, 4 + nc, num_anchors)
        # where num_anchors = 80*80 (P3) + 40*40 (P4) + 20*20 (P5) = 6400 + 1600 + 400 = 8400.
        # nc is 1 (abnormal class). Or it can be a post-processed detection of shape (B, max_det, 6).
        
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
            self.assertEqual(pred_tensor.shape[2], 8400)
        else:
            # Post-processed detections: [batch, max_det, 6]
            self.assertEqual(pred_tensor.shape[1], 300)
            self.assertEqual(pred_tensor.shape[2], 6)

if __name__ == "__main__":
    unittest.main()
