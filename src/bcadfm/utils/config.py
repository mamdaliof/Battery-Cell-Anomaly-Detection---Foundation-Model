import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from bcadfm.data.config import DataConfig
from bcadfm.models.dinov3_classifier import HeadConfig


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

    scheduler_cfg = SchedulerConfig(**scheduler_raw)
    amp_cfg = AmpConfig(**amp_raw)

    return TrainingConfig(
        model_name=raw["model_name"],
        output_dir=raw["output_dir"],
        data=data_cfg,
        head=head_cfg,
        num_epochs=raw["num_epochs"],
        batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"],
        early_stopping_patience=raw.get("early_stopping_patience", 3),
        metric_for_best=raw.get("metric_for_best", "eval_loss"),
        greater_is_better=raw.get("greater_is_better", False),
        scheduler=scheduler_cfg,
        amp=amp_cfg,
    )
