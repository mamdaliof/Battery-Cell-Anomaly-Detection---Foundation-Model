import torch
import torch.nn as nn
from transformers import AutoModel


def _get_transformer_blocks_from_model(model):
    """
    Locate the transformer block list inside a (possibly PEFT-wrapped) backbone.
    Probes common attribute paths in order.
    Returns a list/ModuleList of blocks, or an empty list if not found.
    """
    # Unwrap PEFT wrapper if present
    base = getattr(model, "base_model", model)
    candidates = [
        lambda m: m.encoder.layer,
        lambda m: m.model.encoder.layer,
        lambda m: m.model.layer,
        lambda m: m.encoder.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
        lambda m: m.layer,
    ]
    for getter in candidates:
        try:
            blocks = getter(base)
            if blocks is not None and len(blocks) > 0:
                return list(blocks)
        except AttributeError:
            continue
    # Fallback: walk named modules for a ModuleList whose children have attention
    for _, mod in base.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if hasattr(list(mod)[0], "attention"):
                return list(mod)
    return []


class DinoV3Backbone(nn.Module):
    """
    DINOv3 backbone wrapper for YOLO.
    Loads a frozen DINOv3 model, applies ImageNet normalization, extracts
    spatial patch features (skipping CLS and register tokens), and outputs
    a 2D feature grid of shape (B, D, H_patch, W_patch) at stride 16.
    """
    def __init__(self, c1=3, c2=768, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", peft_config=None):
        """
        Args:
            c1 (int): Input channels (normally 3 for RGB images, automatically passed by YOLO parser).
            c2 (int): Target channels (matches the backbone output dimension in the YAML, e.g. 768 or 384).
            model_name (str): Hugging Face model repository path.
            peft_config (PeftConfigSchema, optional): PEFT configuration.
        """
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            # Fallback to local files only if gated repository / offline error
            try:
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            except Exception:
                raise e
        self.hidden_size = self.model.config.hidden_size

        # Ensure c2 is set correctly to match the backbone hidden size
        self.out_channels = self.hidden_size

        # Freeze all DINOv3 weights initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Register ImageNet stats as buffers for device and float precision matching
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Check DINOv3 specific configurations
        self.num_registers = getattr(self.model.config, "num_register_tokens", 4 if "dinov3" in model_name.lower() else 0)
        self.patch_size = self.model.config.patch_size

        # Apply requested PEFT wrapping on backbone
        self.peft_type = "none"
        if peft_config is not None:
            if hasattr(peft_config, "type"):
                self.peft_type = peft_config.type
            elif isinstance(peft_config, dict):
                self.peft_type = peft_config.get("type", "none")

        if self.peft_type == "lora":
            from peft import LoraConfig, get_peft_model

            if hasattr(peft_config, "lora_r"):
                r = peft_config.lora_r
                alpha = peft_config.lora_alpha
                dropout = peft_config.lora_dropout
                target_modules = peft_config.lora_target_modules
                target_blocks = peft_config.lora_target_blocks
            else:
                r = peft_config.get("lora_r", 8)
                alpha = peft_config.get("lora_alpha", 16)
                dropout = peft_config.get("lora_dropout", 0.0)
                target_modules = peft_config.get("lora_target_modules", None)
                target_blocks = peft_config.get("lora_target_blocks", None)

            if target_modules is None:
                target_modules = ["q_proj", "v_proj"]

            # Apply LoRA to ALL blocks — no layers_to_transform / layers_pattern
            # (PEFT path resolution is unreliable for DINOv3 architecture).
            peft_lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_lora_config)

            # Post-wrap: freeze lora_* weights in non-target blocks
            if target_blocks is not None and len(target_blocks) > 0:
                target_set = set(target_blocks)
                blocks = _get_transformer_blocks_from_model(self.model)
                for idx, block in enumerate(blocks):
                    if idx not in target_set:
                        for name, param in block.named_parameters():
                            if "lora_" in name:
                                param.requires_grad = False

        elif self.peft_type == "adapter":
            from bcadfm.models.dinov3_classifier import apply_adapters
            if hasattr(peft_config, "adapter_bottleneck_dim"):
                bottleneck_dim = peft_config.adapter_bottleneck_dim
                dropout = peft_config.adapter_dropout
                target_blocks = peft_config.adapter_target_blocks
            else:
                bottleneck_dim = peft_config.get("adapter_bottleneck_dim", 64)
                dropout = peft_config.get("adapter_dropout", 0.0)
                target_blocks = peft_config.get("adapter_target_blocks", None)

            apply_adapters(self.model, bottleneck_dim, dropout, target_blocks)

        elif self.peft_type == "visual_prompt":
            from bcadfm.models.dinov3_classifier import VptWrappedBackbone
            if hasattr(peft_config, "vpt_num_tokens"):
                num_tokens = peft_config.vpt_num_tokens
                deep = peft_config.vpt_deep
                target_blocks = peft_config.vpt_target_blocks
            else:
                num_tokens = peft_config.get("vpt_num_tokens", 10)
                deep = peft_config.get("vpt_deep", False)
                target_blocks = peft_config.get("vpt_target_blocks", None)

            self.model = VptWrappedBackbone(
                self.model,
                num_tokens=num_tokens,
                deep=deep,
                target_blocks=target_blocks,
            )

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

        is_training = self.training

        if self.peft_type != "none" and is_training:
            outputs = self.model(x_norm)
        else:
            with torch.no_grad():
                outputs = self.model(x_norm)

        # outputs.last_hidden_state has shape (B, seq_len, D)
        # Sequence layout: [CLS] + [Prompts] + [Register Tokens (e.g. 4)] + [Patch Tokens]
        H, W = x.shape[2], x.shape[3]
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        num_patches = H_patch * W_patch

        # VPT prompt length (if visual prompt is used, self.model is a VptWrappedBackbone or wraps one)
        num_prompts = 0
        if self.peft_type == "visual_prompt":
            num_prompts = getattr(self.model, "num_tokens", 0)

        start_idx = 1 + num_prompts + self.num_registers

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
