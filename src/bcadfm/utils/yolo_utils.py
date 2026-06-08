import copy
import sys
import logging
import warnings
import torch.nn as nn
import ultralytics.nn.tasks
from ultralytics.nn.tasks import parse_model as original_parse_model
from ultralytics.utils import LOGGER

# Using Context7 for Python import hijacking, monkey-patching, and model parser wrappers.

# Suppress PyTorch user warning about non-deterministic attention
warnings.filterwarnings("ignore", category=UserWarning, message=".*Memory Efficient attention.*")

# Suppress Ultralytics requires_grad=True logs for frozen layers
class FrozenLayerWarningFilter(logging.Filter):
    def filter(self, record):
        if "setting 'requires_grad=True' for frozen layer" in record.getMessage():
            return False
        return True

LOGGER.addFilter(FrozenLayerWarningFilter())

# We will store custom layers in this utility to avoid circular imports.
from bcadfm.models.yolo_dino import DinoV3Backbone, DinoV3SFP_P3, DinoV3SFP_P4, DinoV3SFP_P5

_ACTIVE_PEFT_CONFIG = None

def set_active_peft_config(peft_cfg):
    global _ACTIVE_PEFT_CONFIG
    _ACTIVE_PEFT_CONFIG = peft_cfg

def get_active_peft_config():
    global _ACTIVE_PEFT_CONFIG
    return _ACTIVE_PEFT_CONFIG

def custom_parse_model(d, ch, verbose=True):
    """
    Patched wrapper for parse_model.
    Intercepts the configuration dictionary, replaces custom layers with placeholder Conv layers
    so the Ultralytics model parser can natively calculate width and depth scaling, and then
    reconstructs and injects the actual custom DinoV3Backbone and SFP layers with the correct
    channel dimensions.
    """
    # 1. Clone the dictionary to prevent modifying the caller's object
    d_placeholder = copy.deepcopy(d)
    
    custom_layers = {}
    
    # 2. Identify custom layers in backbone and head, replacing them with placeholder Conv blocks
    # backbone
    for idx, layer in enumerate(d_placeholder.get("backbone", [])):
        f, n, m, args = layer
        if m in ("DinoV3Backbone", "DinoV3SFP_P3", "DinoV3SFP_P4", "DinoV3SFP_P5"):
            custom_layers[idx] = (m, args)
            # Replace with a standard Conv layer [out_channels, kernel_size, stride]
            # args[0] in our yaml is the target out_channels
            out_channels = args[0]
            layer[2] = "Conv"
            layer[3] = [out_channels, 1, 1]
            
    # head
    backbone_len = len(d_placeholder.get("backbone", []))
    for idx, layer in enumerate(d_placeholder.get("head", [])):
        f, n, m, args = layer
        global_idx = backbone_len + idx
        if m in ("DinoV3Backbone", "DinoV3SFP_P3", "DinoV3SFP_P4", "DinoV3SFP_P5"):
            custom_layers[global_idx] = (m, args)
            out_channels = args[0]
            layer[2] = "Conv"
            layer[3] = [out_channels, 1, 1]

    # 3. Let the original Ultralytics parser run natively on standard Conv layers
    model, save = original_parse_model(d_placeholder, ch, verbose=verbose)
    
    # 4. Replace the placeholders with actual DinoV3 / SFP layers
    backbone_hidden_size = 768  # fallback default
    
    for i, (module_name, original_args) in custom_layers.items():
        placeholder = model[i]
        
        # Get in/out channels from placeholder layer
        # If repeated (n > 1), parse_model wraps the modules inside a nn.Sequential
        if isinstance(placeholder, nn.Sequential):
            c1 = placeholder[0].conv.in_channels
            c2 = placeholder[-1].conv.out_channels
        else:
            c1 = placeholder.conv.in_channels
            c2 = placeholder.conv.out_channels
            
        # Reconstruct actual module using the correct PyTorch shapes
        if module_name == "DinoV3Backbone":
            # DinoV3Backbone(c1, c2, model_name)
            model_name = original_args[1] if len(original_args) > 1 else "facebook/dinov3-vits16-pretrain-lvd1689m"
            actual_layer = DinoV3Backbone(c1=c1, c2=c2, model_name=model_name, peft_config=get_active_peft_config())
            backbone_hidden_size = actual_layer.hidden_size
        elif module_name == "DinoV3SFP_P3":
            # Map from raw DINO hidden size to scaled output neck channels
            actual_layer = DinoV3SFP_P3(in_channels=backbone_hidden_size, out_channels=c2)
        elif module_name == "DinoV3SFP_P4":
            actual_layer = DinoV3SFP_P4(in_channels=backbone_hidden_size, out_channels=c2)
        elif module_name == "DinoV3SFP_P5":
            actual_layer = DinoV3SFP_P5(in_channels=backbone_hidden_size, out_channels=c2)
            
        # Copy Ultralytics metadata attributes from placeholder
        for attr in ("i", "f", "type", "np"):
            if hasattr(placeholder, attr):
                setattr(actual_layer, attr, getattr(placeholder, attr))
            
        # Swap placeholder out for the real module
        model[i] = actual_layer
        
    return model, save


def register_yolo_dino():
    """
    Registers the custom layers and wraps parse_model globally within
    the ultralytics package to support custom DINOv3 Vision Transformer pipelines.
    """
    # Dynamic class registration into ultralytics global scope
    setattr(ultralytics.nn.tasks, "DinoV3Backbone", DinoV3Backbone)
    setattr(ultralytics.nn.tasks, "DinoV3SFP_P3", DinoV3SFP_P3)
    setattr(ultralytics.nn.tasks, "DinoV3SFP_P4", DinoV3SFP_P4)
    setattr(ultralytics.nn.tasks, "DinoV3SFP_P5", DinoV3SFP_P5)
    
    # Override parse_model function in tasks module
    ultralytics.nn.tasks.parse_model = custom_parse_model
    
    # Also register in sys.modules to ensure pickle compatibility during model saving/reloading
    sys.modules["ultralytics.nn.tasks"].DinoV3Backbone = DinoV3Backbone
    sys.modules["ultralytics.nn.tasks"].DinoV3SFP_P3 = DinoV3SFP_P3
    sys.modules["ultralytics.nn.tasks"].DinoV3SFP_P4 = DinoV3SFP_P4
    sys.modules["ultralytics.nn.tasks"].DinoV3SFP_P5 = DinoV3SFP_P5

    # Log exactly one warning explaining the suppressed requires_grad logs
    LOGGER.warning(
        "⚠️ [BCADFM] Parameter-Efficient Fine-Tuning (PEFT) is active on the DINOv3 backbone. "
        "Trainable adapter parameters (LoRA, Adapters, or VPT) are nested inside the frozen model. "
        "Detailed warnings regarding 'requires_grad=True for frozen layers' have been suppressed."
    )
