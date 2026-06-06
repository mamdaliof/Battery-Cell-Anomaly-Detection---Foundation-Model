import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from bcadfm.data.config import DataConfig
from bcadfm.models.dinov3_classifier import HeadConfig


@dataclass
class PeftConfigSchema:
    """Configuration schema for parameter-efficient fine-tuning (PEFT)."""

    type: str = "none"  # none, lora, adapter, visual_prompt

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None
    lora_target_blocks: Optional[List[int]] = None

    # Adapter settings
    adapter_bottleneck_dim: int = 64
    adapter_dropout: float = 0.0
    adapter_target_blocks: Optional[List[int]] = None

    # Visual Prompt Tuning (VPT) settings
    vpt_num_tokens: int = 10
    vpt_deep: bool = False
    vpt_target_blocks: Optional[List[int]] = None


@dataclass
class AmpConfig:
    """Automatic mixed precision configuration."""

    fp16: bool = False  # use 16-bit floating point precision (fp16) if supported
    bf16: bool = False  # use bfloat16 precision if supported (e.g., on recent GPUs)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    lr_scheduler_type: str = "linear"  # scheduler type: linear, cosine, cosine_with_restarts, constant, etc.
    warmup_ratio: float = 0.0  # fraction of total steps used for LR warmup


@dataclass
class ImbalanceConfig:
    """Configuration for class imbalance handling strategies."""

    loss_type: str = "cross_entropy"  # "cross_entropy", "focal"
    class_weights: str = "none"       # "none", "balanced", "inverse"
    focal_gamma: float = 2.0          # gamma parameter for Focal Loss
    focal_alpha: Optional[float] = None  # balancing factor for Focal Loss (e.g. 0.25)
    oversampling_method: str = "none"  # "none", "weighted_sampler", "data_level"


@dataclass
class TrainingConfig:
    """Top-level training configuration loaded from YAML.

    This is a minimal schema; it can be extended as needed.
    """

    model_name: str
    output_dir: str

    # Data
    data: DataConfig

    # Model head
    head: HeadConfig

    # PEFT settings
    peft: PeftConfigSchema

    # Imbalance handling configuration
    imbalance: ImbalanceConfig

    # Training hyperparameters
    num_epochs: int
    batch_size: int
    learning_rate: float

    # Early stopping / best model
    early_stopping_patience: int
    metric_for_best: str
    greater_is_better: bool

    # Scheduler and AMP
    scheduler: SchedulerConfig
    amp: AmpConfig


def load_yaml_config(path: str | Path) -> TrainingConfig:
    """Load a YAML config file into a TrainingConfig instance."""

    path = Path(path)
    with path.open("r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Nested dataclasses
    data_cfg = DataConfig(**raw["data"])
    head_cfg = HeadConfig(**raw["head"])

    scheduler_raw = raw.get("scheduler", {})
    amp_raw = raw.get("amp", {})
    peft_raw = raw.get("peft", {})
    imbalance_raw = raw.get("imbalance", {})

    scheduler_cfg = SchedulerConfig(**scheduler_raw)
    amp_cfg = AmpConfig(**amp_raw)
    peft_cfg = PeftConfigSchema(**peft_raw)
    imbalance_cfg = ImbalanceConfig(**imbalance_raw)

    return TrainingConfig(
        model_name=raw["model_name"],
        output_dir=raw["output_dir"],
        data=data_cfg,
        head=head_cfg,
        peft=peft_cfg,
        imbalance=imbalance_cfg,
        num_epochs=raw["num_epochs"],
        batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"],
        early_stopping_patience=raw.get("early_stopping_patience", 3),
        metric_for_best=raw.get("metric_for_best", "eval_loss"),
        greater_is_better=raw.get("greater_is_better", False),
        scheduler=scheduler_cfg,
        amp=amp_cfg,
    )
