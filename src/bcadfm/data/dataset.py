from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from .config import DataConfig


@dataclass
class ImageSample:
    image_path: Path
    label: int


class BatteryCellDataset(Dataset):
    """Local image dataset for battery cell anomaly detection.

    Expects a directory layout like:

        data_dir/
          train/
            normal/
            abnormal/
          val/
            normal/
            abnormal/

    Labels are encoded as 0 = normal, 1 = abnormal.
    """

    def __init__(
        self,
        split: str,
        data_config: DataConfig,
        model_name_or_path: str,
        transform: Optional[Callable] = None,
        image_size_override: Optional[int] = None,
    ) -> None:
        assert split in {"train", "val"}, f"Unsupported split: {split}"
        self.split = split
        self.config = data_config

        # Resolve directory for this split
        if split == "train":
            base_dir = self.config.train_dir()
        else:
            base_dir = self.config.val_dir()

        self.normal_dir = base_dir / self.config.normal_class_name
        self.abnormal_dir = base_dir / self.config.abnormal_class_name

        self.samples: List[ImageSample] = []
        self._collect_samples()

        # DINOv3 image processor handles resize/normalization/RGB
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)

        # Optional override of image size
        if image_size_override is not None:
            # Most HF processors expose a "size" dict with "height"/"width" keys
            if isinstance(self.processor.size, dict):
                self.processor.size["height"] = image_size_override
                self.processor.size["width"] = image_size_override
            else:
                self.processor.size = image_size_override

        self.transform = transform

        # Class mapping
        self.label2id: Dict[str, int] = {
            self.config.normal_class_name: 0,
            self.config.abnormal_class_name: 1,
        }
        self.id2label: Dict[int, str] = {v: k for k, v in self.label2id.items()}

    def _collect_samples(self) -> None:
        def list_images(folder: Path) -> List[Path]:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

        normal_images = list_images(self.normal_dir)
        abnormal_images = list_images(self.abnormal_dir)

        for p in normal_images:
            self.samples.append(ImageSample(image_path=p, label=0))
        for p in abnormal_images:
            self.samples.append(ImageSample(image_path=p, label=1))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")

        # Apply custom augmentations (if any) BEFORE processor
        if self.transform is not None:
            image = self.transform(image)

        # Processor returns a dict with "pixel_values"
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"][0]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(sample.label, dtype=torch.long),
        }


def build_augmentation_pipeline(config: DataConfig, split: str) -> Optional[Callable]:
    """Builds a torchvision-style augmentation pipeline from config.

    Augmentations are only applied to the training split.
    """

    if split != "train" or not config.augmentations_enabled:
        return None

    from torchvision import transforms as T

    t_list: List[Callable] = []

    # Random resized crop (slight)
    t_list.append(
        T.RandomResizedCrop(
            size=config.image_size or 224,
            scale=config.random_resized_crop_scale,
            ratio=config.random_resized_crop_ratio,
        )
    )

    # Horizontal flip
    if config.horizontal_flip_prob > 0:
        t_list.append(T.RandomHorizontalFlip(p=config.horizontal_flip_prob))

    # Small rotation
    if config.rotation_degrees > 0:
        t_list.append(T.RandomRotation(degrees=config.rotation_degrees))

    # Color jitter (can approximate HSV changes)
    if any(
        x > 0
        for x in [
            config.color_jitter_brightness,
            config.color_jitter_contrast,
            config.color_jitter_saturation,
            config.color_jitter_hue,
        ]
    ):
        t_list.append(
            T.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
            )
        )

    # Gaussian noise: implement as a simple transform on tensors
    class AddGaussianNoise:
        def __init__(self, std: float) -> None:
            self.std = std

        def __call__(self, img: Image.Image) -> Image.Image:
            # Convert to tensor, add noise, convert back to PIL
            t = T.ToTensor()(img)
            noise = torch.randn_like(t) * self.std
            t = torch.clamp(t + noise, 0.0, 1.0)
            return T.ToPILImage()(t)

    if config.gaussian_noise_std > 0:
        t_list.append(AddGaussianNoise(config.gaussian_noise_std))

    return T.Compose(t_list)
