"""
training/dataset.py
--------------------
TrueLens AI — Dataset Loader

Expects the following folder structure:

    data/
    ├── train/
    │   ├── real/          ← real photographs
    │   └── ai_generated/  ← AI-generated images
    └── val/
        ├── real/
        └── ai_generated/

Uses torchvision ImageFolder for clean, scalable loading.
Supports automatic train/val split if only a single root is provided.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

from core.config import settings
from models.transforms import get_training_transforms, get_validation_transforms
from utils.logger import get_logger

log = get_logger(__name__)


class TrueLensDataset:
    """
    Wrapper around ImageFolder that handles both:
      A) Pre-split data  (train/ and val/ folders exist separately)
      B) Single root     (one folder — auto split by train_val_split ratio)

    Usage::

        ds = TrueLensDataset(root=Path("data"))
        train_loader, val_loader = ds.get_loaders()
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Check dataset root exists and has expected subfolders."""
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}\n"
                "Create the folder structure:\n"
                "  data/train/real/\n"
                "  data/train/ai_generated/\n"
                "  data/val/real/\n"
                "  data/val/ai_generated/"
            )

    def _count_samples(self, dataset: Dataset) -> dict[str, int]:
        """Return per-class sample counts for logging."""
        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "dataset"):
            # Subset — get targets via indices
            indices = dataset.indices
            targets = [dataset.dataset.targets[i] for i in indices]
        else:
            return {}

        counts: dict[int, int] = {}
        for t in targets:
            counts[t] = counts.get(t, 0) + 1

        classes = getattr(
            dataset,
            "classes",
            getattr(getattr(dataset, "dataset", None), "classes", []),
        )
        return {
            classes[k] if k < len(classes) else str(k): v
            for k, v in sorted(counts.items())
        }

    def get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Build and return (train_loader, val_loader).

        Automatically detects whether data is pre-split or needs splitting.
        """
        train_dir = self.root / "train"
        val_dir   = self.root / "val"

        if train_dir.exists() and val_dir.exists():
            # ── Pre-split dataset ─────────────────────────────────
            log.info("dataset_mode", mode="pre_split")
            train_dataset = ImageFolder(
                root=str(train_dir),
                transform=get_training_transforms(),
            )
            val_dataset = ImageFolder(
                root=str(val_dir),
                transform=get_validation_transforms(),
            )
        else:
            # ── Auto-split from single root ───────────────────────
            log.info("dataset_mode", mode="auto_split")
            full_dataset = ImageFolder(
                root=str(self.root),
                transform=get_training_transforms(),
            )
            n_total = len(full_dataset)
            n_val   = int(n_total * settings.train_val_split)
            n_train = n_total - n_val

            train_dataset, val_dataset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            # Apply correct transforms to val subset
            val_dataset.dataset.transform = get_validation_transforms()

        # ── Log dataset stats ─────────────────────────────────────
        train_counts = self._count_samples(train_dataset)
        val_counts   = self._count_samples(val_dataset)

        log.info(
            "dataset_loaded",
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            train_class_counts=train_counts,
            val_class_counts=val_counts,
        )

        # ── Build DataLoaders ─────────────────────────────────────
        num_workers = min(4, os.cpu_count() or 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=settings.device == "cuda",
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=settings.val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=settings.device == "cuda",
        )

        return train_loader, val_loader


def get_class_names(root: Path) -> list[str]:
    """Return sorted class names from the dataset folder."""
    train_dir = root / "train" if (root / "train").exists() else root
    dataset = ImageFolder(root=str(train_dir))
    return dataset.classes


