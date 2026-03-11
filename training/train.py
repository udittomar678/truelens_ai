"""
train.py
--------
TrueLens AI — Training Entry Point

Run this script to train the model:

    python train.py --data data/

The script expects this folder structure:

    data/
    ├── train/
    │   ├── real/          ← real photographs
    │   └── ai_generated/  ← AI-generated images
    └── val/
        ├── real/
        └── ai_generated/

Or a single folder (auto split):

    data/
    ├── real/
    └── ai_generated/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the TrueLens AI dual-branch CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset root folder",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Apply CLI overrides to settings ──────────────────────────
    from core.config import settings

    if args.epochs:
        settings.num_epochs = args.epochs
    if args.batch_size:
        settings.train_batch_size = args.batch_size
    if args.lr:
        settings.learning_rate = args.lr

    # ── Validate data path ────────────────────────────────────────
    if not args.data.exists():
        print(f"\n❌ Data folder not found: {args.data}")
        print("\nCreate the folder structure:")
        print("  data/train/real/")
        print("  data/train/ai_generated/")
        print("  data/val/real/")
        print("  data/val/ai_generated/")
        sys.exit(1)

    print(f"\n  Device:     {settings.device.upper()}")
    print(f"  Data root:  {args.data}")
    print(f"  Epochs:     {settings.num_epochs}")
    print(f"  Batch size: {settings.train_batch_size}")
    print(f"  LR:         {settings.learning_rate}")

    # ── Run training ──────────────────────────────────────────────
    from training.trainer import Trainer

    trainer = Trainer(data_root=args.data)
    trainer.train()


if __name__ == "__main__":
    main()


