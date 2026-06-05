from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class HeadConfig:
    """Configuration of the classification head.

    depth: number of linear layers in the head (>= 1).
    hidden_dim: size of hidden layers when depth > 1.
    dropout: dropout probability between layers.
    """

    num_labels: int = 2
    depth: int = 1
    hidden_dim: Optional[int] = None
    dropout: float = 0.0


class DinoV3Classifier(nn.Module):
    """Frozen DINOv3 backbone + configurable classification head.

    - Backbone: loaded from Hugging Face `transformers` using `AutoModel`.
      It is frozen by default (no gradient updates).
    - Head: a configurable MLP ending in `num_labels` logits.

    The model expects `pixel_values` as input, as produced by the
    DINOv3 image processor.
    """

    def __init__(
        self,
        model_name_or_path: str,
        head_config: Optional[HeadConfig] = None,
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

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Build classification head
        self.classifier = self._build_head(input_dim=hidden_size, cfg=head_config)

        # Store label mappings for convenience (used by Trainer / saving config later)
        if id2label is None:
            id2label = {0: "class_0", 1: "class_1"}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        self.id2label = id2label
        self.label2id = label2id

    @staticmethod
    def _build_head(input_dim: int, cfg: HeadConfig) -> nn.Module:
        """Create a (possibly multi-layer) classification head.

        If cfg.depth == 1, this is just a single Linear layer.
        If cfg.depth > 1, it builds: Linear -> (Dropout) -> GELU -> ... -> Linear.
        """

        layers: List[nn.Module] = []

        if cfg.depth <= 1:
            layers.append(nn.Linear(input_dim, cfg.num_labels))
        else:
            if cfg.hidden_dim is None:
                raise ValueError("hidden_dim must be set when depth > 1")

            in_dim = input_dim
            for _ in range(cfg.depth - 1):
                layers.append(nn.Linear(in_dim, cfg.hidden_dim))
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                layers.append(nn.GELU())
                in_dim = cfg.hidden_dim

            # Final classification layer
            layers.append(nn.Linear(in_dim, cfg.num_labels))

        return nn.Sequential(*layers)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Returns a dict with at least `logits`. If `labels` are provided,
        also returns `loss` (cross-entropy by default).
        """

        # Backbone forward: DINOv3 outputs last_hidden_state and optionally pooled output
        outputs = self.backbone(pixel_values=pixel_values)

        # Prefer pooled output if available; otherwise use CLS token from last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            # Assume CLS token is at position 0
            features = outputs.last_hidden_state[:, 0]

        logits = self.classifier(features)

        result: dict = {"logits": logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result
