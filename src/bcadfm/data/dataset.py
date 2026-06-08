from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import random
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
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
        oversample: bool = False,
        seed: int = 42,
    ) -> None:
        assert split in {"train", "val"}, f"Unsupported split: {split}"
        self.split = split
        self.config = data_config
        self.seed = seed

        # Resolve directory for this split
        if split == "train":
            base_dir = self.config.train_dir()
        else:
            base_dir = self.config.val_dir()

        self.normal_dir = base_dir / self.config.normal_class_name
        self.abnormal_dir = base_dir / self.config.abnormal_class_name

        self.samples: List[ImageSample] = []
        self._collect_samples()

        if split == "train" and oversample:
            self.oversample_dataset()

        # Try to use the Hugging Face image processor first. For some models
        # (e.g. certain DINOv3 checkpoints), this can fail because the
        # repository does not ship a usable preprocessor_config.json or
        # image_processor_type, which is a known transformers/DINOv3 issue.
        # In that case, fall back to a manual torchvision-based transform.
        # Try to use the Hugging Face image processor first (H12 Fix)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        except Exception as e:
            # Try to load it from local cache files only (C7 Fix)
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name_or_path, local_files_only=True)
            except Exception:
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(f"⚠️ Warning: Failed to load HF processor for {model_name_or_path}: {e}")
                    print("Attempting fallback to default google/vit-base-patch16-224 processor...")
                try:
                    self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                except Exception:
                    self.processor = None

        self.image_size_override = image_size_override

        # Optional override of image size when using the HF processor
        if self.processor is not None and image_size_override is not None:
            if isinstance(self.processor.size, dict):
                if "shortest_edge" in self.processor.size:
                    self.processor.size["shortest_edge"] = image_size_override
                else:
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

        normal_images = sorted(list_images(self.normal_dir))
        abnormal_images = sorted(list_images(self.abnormal_dir))

        for p in normal_images:
            self.samples.append(ImageSample(image_path=p, label=0))
        for p in abnormal_images:
            self.samples.append(ImageSample(image_path=p, label=1))

    def oversample_dataset(self) -> None:
        """Perform data-level minority class oversampling."""
        class_counts = {}
        for sample in self.samples:
            class_counts[sample.label] = class_counts.get(sample.label, 0) + 1

        if len(class_counts) <= 1:
            return

        max_size = max(class_counts.values())

        grouped: Dict[int, List[ImageSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.label, []).append(sample)

        # Seed python's random state for reproducible DDP oversampling (C5 Fix)
        state = random.getstate()
        random.seed(self.seed)
        try:
            oversampled_samples: List[ImageSample] = []
            for label, samples_list in grouped.items():
                if len(samples_list) < max_size:
                    # Replicate minority class samples
                    replicated = random.choices(samples_list, k=max_size)
                    oversampled_samples.extend(replicated)
                else:
                    oversampled_samples.extend(samples_list)

            # Shuffle to mix classes
            random.shuffle(oversampled_samples)
        finally:
            random.setstate(state)

        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"🔄 Data-level oversampling applied. Sample counts changed from {class_counts} to:")
            new_counts = {}
            for s in oversampled_samples:
                new_counts[s.label] = new_counts.get(s.label, 0) + 1
            print(f"   => {new_counts}")

        self.samples = oversampled_samples

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")

        # Apply custom augmentations (if any) BEFORE processor/manual transform
        if self.transform is not None:
            image = self.transform(image)

        # HF processor path (ViT and backbones that ship a proper
        # preprocessor_config.json / image_processor_type) (H12 Fix)
        if self.processor is not None:
            encoded = self.processor(images=image, return_tensors="pt")
            pixel_values = encoded["pixel_values"][0]
        else:
            # Manual fallback using ImageNet normalization and target size
            from torchvision.transforms import functional as F_tv
            size = self.image_size_override if self.image_size_override is not None else 224
            img_resized = F_tv.resize(image, [size, size])
            tensor = F_tv.to_tensor(img_resized)  # scales to [0.0, 1.0]
            pixel_values = F_tv.normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

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

    import numpy as np

    # Base transforms (individual ops)
    # ── Fix: build transform objects ONCE here, not inside the closure ──────
    ops: List[tuple[str, float, Callable[[Image.Image], Image.Image]]] = []

    # Random resized crop
    if config.random_resized_crop_prob > 0:
        _crop = T.RandomResizedCrop(
            size=config.image_size or 224,
            scale=config.random_resized_crop_scale,
            ratio=config.random_resized_crop_ratio,
        )
        def rnd_resized_crop_op(img: Image.Image, _t=_crop) -> Image.Image:
            return _t(img)
        ops.append(("random_resized_crop", config.random_resized_crop_prob, rnd_resized_crop_op))

    # Horizontal flip
    if config.horizontal_flip_prob > 0:
        def hflip_op(img: Image.Image) -> Image.Image:
            return T.functional.hflip(img)
        ops.append(("horizontal_flip", config.horizontal_flip_prob, hflip_op))

    # Rotation
    if config.rotation_prob > 0 and config.rotation_degrees > 0:
        _rotate = T.RandomRotation(degrees=config.rotation_degrees)
        def rotate_op(img: Image.Image, _t=_rotate) -> Image.Image:
            return _t(img)
        ops.append(("rotation", config.rotation_prob, rotate_op))

    # Color jitter
    if config.color_jitter_prob > 0 and any(
        x > 0
        for x in [
            config.color_jitter_brightness,
            config.color_jitter_contrast,
            config.color_jitter_saturation,
            config.color_jitter_hue,
        ]
    ):
        _cj = T.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue,
        )
        def color_jitter_op(img: Image.Image, _t=_cj) -> Image.Image:
            return _t(img)
        ops.append(("color_jitter", config.color_jitter_prob, color_jitter_op))

    # Gaussian noise
    # ── Fix: use NumPy to avoid PIL→Tensor→PIL round-trip ───────────────────
    if config.gaussian_noise_prob > 0 and config.gaussian_noise_std > 0:
        _noise_std = float(config.gaussian_noise_std)
        def noise_op(img: Image.Image, _std=_noise_std) -> Image.Image:
            arr = np.array(img, dtype=np.float32) / 255.0
            arr += np.random.randn(*arr.shape).astype(np.float32) * _std
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr)
        ops.append(("gaussian_noise", config.gaussian_noise_prob, noise_op))

    if not ops:
        return None

    class RandomAugmentationCombo:
        """Apply up to N transforms per image, sampled by per-op probabilities."""

        def __init__(self, ops: List[tuple[str, float, Callable]], global_prob: float, max_transforms: int) -> None:
            self.ops = ops
            self.global_prob = global_prob
            self.max_transforms = max_transforms

        def __call__(self, img: Image.Image) -> Image.Image:
            # Decide whether to apply any augmentation at all
            if random.random() > self.global_prob:
                return img

            # Sample which operations to apply, up to max_transforms, without replacement
            # First, build a list of candidate indices weighted by probability
            candidates = list(range(len(self.ops)))
            chosen_indices: List[int] = []

            probs = [p for (_, p, _) in self.ops]
            if sum(probs) == 0:
                return img

            # We sample iteratively to avoid duplicates (H3 Fix)
            # We use the raw operator probabilities as weights directly.
            for _ in range(min(self.max_transforms, len(candidates))):
                current_weights = [probs[i] for i in candidates]
                idx = random.choices(candidates, weights=current_weights, k=1)[0]
                chosen_indices.append(idx)
                candidates.remove(idx)
                if not candidates:
                    break

            # Apply chosen transforms in a fixed, logical order (order of definition in self.ops)
            for idx in sorted(chosen_indices):
                _, _, op = self.ops[idx]
                img = op(img)

            return img

    return RandomAugmentationCombo(
        ops=ops,
        global_prob=config.aug_global_prob,
        max_transforms=config.aug_max_transforms,
    )
