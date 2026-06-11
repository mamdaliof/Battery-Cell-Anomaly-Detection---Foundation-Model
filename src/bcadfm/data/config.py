from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DataConfig:
    """Configuration for local image datasets.

    All paths are local, data is not uploaded to any external service.
    """

    data_dir: str = "data"  # root directory with train/ and val/
    train_subdir: str = "train"  # subdirectory name for training split
    val_subdir: str = "val"  # subdirectory name for validation split
    normal_class_name: str = "normal"  # folder name for the normal class
    abnormal_class_name: str = "abnormal"  # folder name for the abnormal class

    # If None, use DINOv3 image processor defaults for input size
    image_size: Optional[int] = None

    # Global augmentation controls (applied only on train split)
    augmentations_enabled: bool = True  # master switch to enable/disable all augmentations
    aug_global_prob: float = 1.0  # probability that any augmentation is applied to an image
    aug_max_transforms: int = 3  # maximum number of different transforms applied to one image

    # Individual augmentation parameters and probabilities
    horizontal_flip_prob: float = 0.5  # probability of selecting horizontal flip when augmenting
    rotation_degrees: float = 10.0  # maximum rotation in degrees (applied if rotation selected)
    rotation_prob: float = 0.3  # probability of selecting rotation when augmenting

    random_resized_crop_scale: Tuple[float, float] = (0.9, 1.0)  # scale range for random resized crop
    random_resized_crop_ratio: Tuple[float, float] = (0.9, 1.1)  # aspect ratio range for random resized crop
    random_resized_crop_prob: float = 0.7  # probability of selecting random resized crop when augmenting

    color_jitter_brightness: float = 0.1  # brightness jitter factor for color jitter
    color_jitter_contrast: float = 0.1  # contrast jitter factor for color jitter
    color_jitter_saturation: float = 0.1  # saturation jitter factor for color jitter
    color_jitter_hue: float = 0.02  # hue jitter factor for color jitter
    color_jitter_prob: float = 0.9  # probability of selecting color jitter when augmenting

    gaussian_noise_std: float = 0.01  # standard deviation of Gaussian noise added to image
    gaussian_noise_prob: float = 0.4  # probability of selecting Gaussian noise when augmenting

    # YOLO-specific overrides (optional dict)
    yolo_augmentations: Optional[dict] = None

    def train_dir(self) -> Path:
        return Path(self.data_dir) / self.train_subdir

    def val_dir(self) -> Path:
        return Path(self.data_dir) / self.val_subdir
