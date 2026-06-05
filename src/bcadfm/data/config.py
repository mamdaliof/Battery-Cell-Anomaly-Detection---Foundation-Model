from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for local image datasets.

    All paths are local, data is not uploaded to any external service.
    """

    data_dir: str = "data"  # root directory with train/ and val/
    train_subdir: str = "train"
    val_subdir: str = "val"
    normal_class_name: str = "normal"
    abnormal_class_name: str = "abnormal"

    # If None, use DINOv3 image processor defaults
    image_size: Optional[int] = None

    # Augmentations (applied only on train split)
    augmentations_enabled: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 10.0
    random_resized_crop_scale: tuple[float, float] = (0.9, 1.0)
    random_resized_crop_ratio: tuple[float, float] = (0.9, 1.1)
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.02
    gaussian_noise_std: float = 0.01

    def train_dir(self) -> Path:
        return Path(self.data_dir) / self.train_subdir

    def val_dir(self) -> Path:
        return Path(self.data_dir) / self.val_subdir
