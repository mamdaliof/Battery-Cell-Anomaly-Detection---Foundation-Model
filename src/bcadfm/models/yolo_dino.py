import torch
import torch.nn as nn
from transformers import AutoModel

# Using Context7 for PyTorch module design, transformers model loading, and Tensor reshaping operations.

class DinoV3Backbone(nn.Module):
    """
    DINOv3 backbone wrapper for YOLO.
    Loads a frozen DINOv3 model, applies ImageNet normalization, extracts
    spatial patch features (skipping CLS and register tokens), and outputs
    a 2D feature grid of shape (B, D, H_patch, W_patch) at stride 16.
    """
    def __init__(self, c1=3, c2=768, model_name="facebook/dinov3-vits16-pretrain-lvd1689m"):
        """
        Args:
            c1 (int): Input channels (normally 3 for RGB images, automatically passed by YOLO parser).
            c2 (int): Target channels (matches the backbone output dimension in the YAML, e.g. 768 or 384).
            model_name (str): Hugging Face model repository path.
        """
        super().__init__()
        # Load backbone model
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # Ensure c2 is set correctly to match the backbone hidden size
        self.out_channels = self.hidden_size
        
        # Freeze all DINOv3 weights
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Register ImageNet stats as buffers for device and float precision matching
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Check DINOv3 specific configurations
        self.num_registers = getattr(self.model.config, "num_register_tokens", 4 if "dinov3" in model_name.lower() else 0)
        self.patch_size = self.model.config.patch_size

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
        Returns:
            torch.Tensor: Feature map grid of shape (B, hidden_size, H_patch, W_patch).
        """
        # YOLO inputs are pre-processed to float32. We scale to [0, 1] if not already scaled.
        if x.max() > 1.0:
            x = x / 255.0
            
        # Apply standardization (ImageNet normalization)
        x_norm = (x - self.mean) / self.std
        
        # Execute frozen backbone
        with torch.no_grad():
            outputs = self.model(x_norm, interpolate_pos_encoding=True)
            
        # outputs.last_hidden_state has shape (B, seq_len, D)
        # Sequence layout: [CLS] + [Register Tokens (e.g. 4)] + [Patch Tokens]
        H, W = x.shape[2], x.shape[3]
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        
        num_patches = H_patch * W_patch
        start_idx = 1 + self.num_registers
        
        # Extract patch tokens only
        patch_tokens = outputs.last_hidden_state[:, start_idx : start_idx + num_patches, :]
        
        # Reshape sequence back to 2D grid: (B, D, H_patch, W_patch)
        B, N, D = patch_tokens.shape
        grid = patch_tokens.transpose(1, 2).view(B, D, H_patch, W_patch)
        return grid


class DinoV3SFP_P3(nn.Module):
    """
    Simple Feature Pyramid projection for Stride 8 (P3).
    Upsamples the backbone output by 2x using ConvTranspose2d, projects channels,
    and applies spatial smoothing.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample 2x: e.g. from stride 16 (40x40) to stride 8 (80x80)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(in_channels // 2)
        self.act1 = nn.SiLU()
        
        self.project = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.SiLU()

    def forward(self, x):
        x = self.act1(self.norm1(self.upsample(x)))
        x = self.act2(self.norm2(self.project(x)))
        x = self.act3(self.norm3(self.smooth(x)))
        return x


class DinoV3SFP_P4(nn.Module):
    """
    Simple Feature Pyramid projection for Stride 16 (P4).
    Projects the backbone output to target neck dimension directly, and
    applies spatial smoothing.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()
        
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.act1(self.norm1(self.project(x)))
        x = self.act2(self.norm2(self.smooth(x)))
        return x


class DinoV3SFP_P5(nn.Module):
    """
    Simple Feature Pyramid projection for Stride 32 (P5).
    Downsamples the backbone output by 2x using MaxPool2d, projects channels,
    and applies spatial smoothing.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Downsample 2x: e.g. from stride 16 (40x40) to stride 32 (20x20)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()
        
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.downsample(x)
        x = self.act1(self.norm1(self.project(x)))
        x = self.act2(self.norm2(self.smooth(x)))
        return x
