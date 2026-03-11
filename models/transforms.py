"""
models/transforms.py
---------------------
Image preprocessing pipelines for TrueLens AI.

Keeps transforms in one place so training and inference
always use identical preprocessing — no subtle bugs from
mismatched normalisation.
"""

from __future__ import annotations

from torchvision import transforms

from core.config import settings

# ── ImageNet normalisation constants ──────────────────────────────
# EfficientNet was pretrained with these values
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_inference_transforms() -> transforms.Compose:
    """
    Minimal, deterministic transforms for inference.
    No augmentation — just resize, convert, normalise.
    """
    return transforms.Compose([
        transforms.Resize((settings.image_size, settings.image_size)),
        transforms.ToTensor(),                      # [0, 255] → [0.0, 1.0]
        transforms.Normalize(
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
        ),
    ])


def get_training_transforms() -> transforms.Compose:
    """
    Augmented transforms for the training split.
    Augmentation increases generalisation and reduces overfitting.
    """
    return transforms.Compose([
        transforms.Resize((settings.image_size + 32, settings.image_size + 32)),
        transforms.RandomCrop(settings.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
        ),
    ])


def get_validation_transforms() -> transforms.Compose:
    """
    Deterministic transforms for the validation split.
    Identical to inference — no augmentation.
    """
    return get_inference_transforms()


def denormalise(tensor):
    """
    Reverse ImageNet normalisation for visualisation (e.g. Grad-CAM overlay).

    Args:
        tensor: Float tensor (C, H, W) or (B, C, H, W).

    Returns:
        Tensor with values roughly in [0, 1].
    """
    import torch
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)

    return tensor * std + mean


