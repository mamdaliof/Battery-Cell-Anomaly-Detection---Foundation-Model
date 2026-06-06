from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any, Union

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class HeadConfig:
    """Configuration of the classification head.

    depth: number of linear layers in the head (>= 1).
    hidden_dim: size of hidden layers when depth > 1. Can be:
                - int: absolute number of neurons.
                - float: multiplier of input embedding dimension (e.g. 0.5).
                - str: multiplier of input embedding dimension with 'X' suffix (e.g. "0.5X").
    dropout: dropout probability between layers.
    """

    num_labels: int = 2
    depth: int = 1
    hidden_dim: Optional[Union[int, float, str]] = None
    dropout: float = 0.0


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter module (Pfeiffer-style) inserted after FFN/MLP."""

    def __init__(self, input_dim: int, bottleneck_dim: int, dropout: float = 0.0):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights: down-proj with Kaiming, up-proj with zeros to act as identity
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_proj(x)
        out = self.non_linear(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        return x + out


class AdapterWrappedMLP(nn.Module):
    """Wraps the original MLP of a transformer block with a bottleneck adapter."""

    def __init__(self, original_mlp: nn.Module, bottleneck_dim: int, dropout: float = 0.0):
        super().__init__()
        self.original_mlp = original_mlp

        # Determine input/output dimension from original MLP's final linear layer
        if hasattr(original_mlp, "fc2"):
            input_dim = original_mlp.fc2.out_features
        elif hasattr(original_mlp, "dense"):
            input_dim = original_mlp.dense.out_features
        else:
            # Fallback to look at parameters
            fc_layers = [m for m in original_mlp.modules() if isinstance(m, nn.Linear)]
            if len(fc_layers) > 0:
                input_dim = fc_layers[-1].out_features
            else:
                raise ValueError("Could not determine embedding dimension of MLP.")

        self.adapter = BottleneckAdapter(input_dim, bottleneck_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_out = self.original_mlp(x)
        return self.adapter(mlp_out)


def apply_adapters(
    backbone: nn.Module,
    bottleneck_dim: int = 64,
    dropout: float = 0.0,
    target_blocks: Optional[List[int]] = None,
) -> None:
    """Recursively search for MLP blocks and wrap them with Pfeiffer adapters."""
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        layers = backbone.encoder.layer
    elif hasattr(backbone, "model") and hasattr(backbone.model, "layer"):
        layers = backbone.model.layer
    elif hasattr(backbone, "layer"):
        layers = backbone.layer
    elif hasattr(backbone, "layers"):
        layers = backbone.layers
    else:
        raise ValueError("Backbone model must have a layer attribute (encoder.layer, model.layer, layer, or layers) to apply adapters.")

    num_layers = len(layers)
    blocks_to_wrap = target_blocks if target_blocks is not None else list(range(num_layers))

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Apply adapters to target blocks
    for idx in blocks_to_wrap:
        if idx < 0 or idx >= num_layers:
            continue
        layer = layers[idx]
        if hasattr(layer, "mlp"):
            layer.mlp = AdapterWrappedMLP(layer.mlp, bottleneck_dim, dropout)
            # Make adapter parameters trainable
            for param in layer.mlp.adapter.parameters():
                param.requires_grad = True


class VptLayerWrapper(nn.Module):
    """Wraps a transformer block to inject new visual prompts for Deep VPT."""

    def __init__(self, original_layer: nn.Module, num_tokens: int, prompt_parameter: nn.Parameter):
        super().__init__()
        self.original_layer = original_layer
        self.num_tokens = num_tokens
        self.prompt = prompt_parameter

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # Discard the prompt tokens from the previous layer (indices 1 to num_tokens+1)
        cls_token = hidden_states[:, :1, :]
        patch_tokens = hidden_states[:, 1 + self.num_tokens :, :]
        batch_size = hidden_states.shape[0]

        # Prepend new prompt tokens
        new_prompts = self.prompt.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, new_prompts, patch_tokens], dim=1)

        return self.original_layer(x, *args, **kwargs)


class VptWrappedBackbone(nn.Module):
    """Wraps a DINOv3 backbone to support Shallow and Deep Visual Prompt Tuning."""

    def __init__(
        self,
        original_backbone: nn.Module,
        num_tokens: int = 10,
        deep: bool = False,
        target_blocks: Optional[List[int]] = None,
    ):
        super().__init__()
        self.original_backbone = original_backbone
        self.num_tokens = num_tokens
        self.deep = deep
        self.target_blocks = target_blocks
        self.hidden_size = original_backbone.config.hidden_size
        self.config = original_backbone.config

        # Freeze original backbone
        for param in self.original_backbone.parameters():
            param.requires_grad = False

        if hasattr(original_backbone, "encoder") and hasattr(original_backbone.encoder, "layer"):
            self.layers = original_backbone.encoder.layer
        elif hasattr(original_backbone, "model") and hasattr(original_backbone.model, "layer"):
            self.layers = original_backbone.model.layer
        elif hasattr(original_backbone, "layer"):
            self.layers = original_backbone.layer
        elif hasattr(original_backbone, "layers"):
            self.layers = original_backbone.layers
        else:
            raise ValueError("Backbone model must have a layer attribute (encoder.layer, model.layer, layer, or layers) to apply VPT.")

        num_layers = len(self.layers)
        self.target_layers = target_blocks if target_blocks is not None else list(range(num_layers))

        # Shallow/input prompts
        self.prompt = nn.Parameter(torch.zeros(1, num_tokens, self.hidden_size))
        nn.init.xavier_uniform_(self.prompt)
        self.prompt.requires_grad = True

        if self.deep:
            self.deep_prompts = nn.ParameterDict()
            for idx in self.target_layers:
                # Prompt for layer 0 is already handled by self.prompt
                if idx == 0:
                    continue
                p = nn.Parameter(torch.zeros(1, num_tokens, self.hidden_size))
                nn.init.xavier_uniform_(p)
                p.requires_grad = True
                self.deep_prompts[f"layer_{idx}"] = p

                # Wrap layer l with prompt replacement
                layer = self.layers[idx]
                self.layers[idx] = VptLayerWrapper(
                    layer, num_tokens, self.deep_prompts[f"layer_{idx}"]
                )

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 1. Base patch embeddings (with pos encoding added)
        if hasattr(self.original_backbone, "embeddings"):
            embeddings_module = self.original_backbone.embeddings
        elif hasattr(self.original_backbone, "model") and hasattr(self.original_backbone.model, "embeddings"):
            embeddings_module = self.original_backbone.model.embeddings
        else:
            raise ValueError("Could not find embeddings module in backbone.")

        x = embeddings_module(pixel_values, bool_masked_pos=bool_masked_pos)

        # 2. Prepend prompt tokens after CLS token (at index 1)
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        batch_size = x.shape[0]

        prompts = self.prompt.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, prompts, patch_tokens], dim=1)

        # 3. Pass through encoder
        if hasattr(self.original_backbone, "encoder"):
            encoder_module = self.original_backbone.encoder
        elif hasattr(self.original_backbone, "model") and hasattr(self.original_backbone.model, "encoder"):
            encoder_module = self.original_backbone.model.encoder
        else:
            raise ValueError("Could not find encoder module in backbone.")

        encoder_outputs = encoder_module(
            x,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_outputs, tuple):
            return encoder_outputs

        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DinoV3Classifier(nn.Module):
    """Frozen DINOv3 backbone + configurable classification head, with PEFT support.

    - Backbone: loaded from Hugging Face `transformers` using `AutoModel`.
      PEFT integration wraps the backbone with LoRA, Adapters, or VPT.
    - Head: a configurable MLP ending in `num_labels` logits.

    The model expects `pixel_values` as input, as produced by the
    DINOv3 image processor.
    """

    def __init__(
        self,
        model_name_or_path: str,
        head_config: Optional[HeadConfig] = None,
        peft_config: Optional[Any] = None,
        freeze_backbone: bool = True,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name_or_path)

        # Determine embedding dimension from backbone config
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Could not infer hidden_size from backbone config. "
                "Check the DINOv3 model and adjust DinoV3Classifier accordingly."
            )

        if head_config is None:
            head_config = HeadConfig(num_labels=2, depth=1, hidden_dim=None, dropout=0.0)
        self.head_config = head_config

        # 1. Parse PEFT configurations
        peft_type = "none"
        if peft_config is not None:
            if hasattr(peft_config, "type"):
                peft_type = peft_config.type
            elif isinstance(peft_config, dict):
                peft_type = peft_config.get("type", "none")

        # 2. Freeze backbone by default (needed if freeze_backbone is True or PEFT is enabled)
        if freeze_backbone or peft_type != "none":
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Apply requested PEFT wrapping on backbone
        if peft_type == "lora":
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
                target_modules = ["query", "value"]

            lora_kwargs = {
                "r": r,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
                "target_modules": target_modules,
                "bias": "none",
            }
            if target_blocks is not None and len(target_blocks) > 0:
                lora_kwargs["layers_to_transform"] = target_blocks
                # Auto-detect the layer pattern from the backbone structure
                # DINOv3 / standard ViT: encoder.layer
                # Some ViTs: model.encoder.layer or just layer
                backbone = self.backbone
                if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
                    lora_kwargs["layers_pattern"] = "encoder.layer"
                elif hasattr(backbone, "model") and hasattr(backbone.model, "encoder") and hasattr(backbone.model.encoder, "layer"):
                    lora_kwargs["layers_pattern"] = "model.encoder.layer"
                elif hasattr(backbone, "model") and hasattr(backbone.model, "layer"):
                    lora_kwargs["layers_pattern"] = "model.layer"
                elif hasattr(backbone, "layer"):
                    lora_kwargs["layers_pattern"] = "layer"
                else:
                    # Fallback: don't set layers_pattern — all layers will be adapted
                    pass

            peft_lora_config = LoraConfig(**lora_kwargs)
            self.backbone = get_peft_model(self.backbone, peft_lora_config)

        elif peft_type == "adapter":
            if hasattr(peft_config, "adapter_bottleneck_dim"):
                bottleneck_dim = peft_config.adapter_bottleneck_dim
                dropout = peft_config.adapter_dropout
                target_blocks = peft_config.adapter_target_blocks
            else:
                bottleneck_dim = peft_config.get("adapter_bottleneck_dim", 64)
                dropout = peft_config.get("adapter_dropout", 0.0)
                target_blocks = peft_config.get("adapter_target_blocks", None)

            apply_adapters(self.backbone, bottleneck_dim, dropout, target_blocks)

        elif peft_type == "visual_prompt":
            if hasattr(peft_config, "vpt_num_tokens"):
                num_tokens = peft_config.vpt_num_tokens
                deep = peft_config.vpt_deep
                target_blocks = peft_config.vpt_target_blocks
            else:
                num_tokens = peft_config.get("vpt_num_tokens", 10)
                deep = peft_config.get("vpt_deep", False)
                target_blocks = peft_config.get("vpt_target_blocks", None)

            self.backbone = VptWrappedBackbone(
                self.backbone,
                num_tokens=num_tokens,
                deep=deep,
                target_blocks=target_blocks,
            )

        # Build classification head (remains trainable)
        self.classifier = self._build_head(input_dim=hidden_size, cfg=head_config)

        # Ensure head parameters require grad (always trainable)
        for param in self.classifier.parameters():
            param.requires_grad = True

        # Store label mappings for convenience
        if id2label is None:
            id2label = {0: "class_0", 1: "class_1"}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        self.id2label = id2label
        self.label2id = label2id

    @staticmethod
    def _build_head(input_dim: int, cfg: HeadConfig) -> nn.Module:
        """Create a classification head (linear or MLP)."""
        layers: List[nn.Module] = []

        if cfg.depth <= 1:
            layers.append(nn.Linear(input_dim, cfg.num_labels))
        else:
            if cfg.hidden_dim is None:
                raise ValueError("hidden_dim must be set when depth > 1")

            # Resolve hidden_dim if float or string
            hidden_dim_resolved = cfg.hidden_dim
            if isinstance(hidden_dim_resolved, str):
                val_str = hidden_dim_resolved.lower().strip()
                if val_str.endswith("x"):
                    val_str = val_str[:-1]
                try:
                    factor = float(val_str)
                    hidden_dim_resolved = int(factor * input_dim)
                except ValueError:
                    raise ValueError(f"Could not parse hidden_dim string: {cfg.hidden_dim}")
            elif isinstance(hidden_dim_resolved, float):
                hidden_dim_resolved = int(hidden_dim_resolved * input_dim)
            elif isinstance(hidden_dim_resolved, int):
                pass
            else:
                raise ValueError(f"Invalid type for hidden_dim: {type(cfg.hidden_dim)}")

            in_dim = input_dim
            for _ in range(cfg.depth - 1):
                layers.append(nn.Linear(in_dim, hidden_dim_resolved))
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                layers.append(nn.GELU())
                in_dim = hidden_dim_resolved

            layers.append(nn.Linear(in_dim, cfg.num_labels))

        return nn.Sequential(*layers)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass through backbone + head."""
        outputs = self.backbone(pixel_values=pixel_values)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]

        logits = self.classifier(features)

        result: dict = {"logits": logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result
