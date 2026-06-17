import sys
import os
import yaml
import tempfile
from pathlib import Path

# Add src/ directory to Python path so we can import bcadfm modules
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))

from bcadfm.utils.yolo_utils import register_yolo_dino
from ultralytics import YOLO

def debug_load_model(config_path="configs/det/yolo26_dino.yaml", model_name="facebook/dinov3-vits16-pretrain-lvd1689m"):
    """
    Dummy function to register custom layers, load the YOLO-DINOv3 configuration,
    and instantiate the model. 
    
    You can set a breakpoint (checkpoint) inside:
      - `custom_parse_model` in `src/bcadfm/utils/yolo_utils.py`
      - `DinoV3Backbone.__init__` in `src/bcadfm/models/yolo_dino.py`
    to inspect variables and verify parser behavior.
    """
    print("Step 1: Registering custom YOLO DINOv3 layers...")
    register_yolo_dino()
    
    config_abs_path = project_root / config_path
    print(f"Step 2: Loading config from {config_abs_path}...")
    if not config_abs_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_abs_path}")
        
    with open(config_abs_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Override the model name dynamically to a small or custom model if desired
    if "backbone" in cfg and len(cfg["backbone"]) > 0:
        original_args = cfg["backbone"][0][3]
        out_channels = original_args[0]
        print(f"Overriding backbone model from '{original_args[1]}' to '{model_name}'")
        cfg["backbone"][0][3] = [out_channels, model_name]
        
    # Write updated config to a temp file so we can pass it to Ultralytics YOLO loader
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as temp_file:
        yaml.safe_dump(cfg, temp_file)
        temp_config_path = temp_file.name
        
    try:
        print(f"Step 3: Instantiating YOLO model using {temp_config_path}...")
        # --- PLACE VS CODE BREAKPOINT HERE ---
        # Step into the line below to debug the model initialization!
        model = YOLO(temp_config_path)
        
        print("\n=== Model successfully loaded ===")
        print(f"Model backbone type: {type(model.model.model[0])}")
        print(f"Output layers count: {len(model.model.model)}")
        print("=================================\n")
        
        return model
    finally:
        # Clean up temporary configuration file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print("Cleaned up temporary configuration file.")

if __name__ == "__main__":
    # Run the dummy function. 
    # By default, uses the smaller dinov3-vits16 model to save memory/loading time.
    try:
        debug_load_model()
    except Exception as e:
        print(f"\nError encountered during model loading: {e}", file=sys.stderr)
        print("\nEnsure you have access/credentials to Hugging Face or the model weights are cached.", file=sys.stderr)
