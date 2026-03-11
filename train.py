"""
train.py
--------
TrueLens AI — Training Entry Point
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
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from core.config import settings

    if args.epochs:
        settings.num_epochs = args.epochs
    if args.batch_size:
        settings.train_batch_size = args.batch_size
    if args.lr:
        settings.learning_rate = args.lr

    if not args.data.exists():
        print(f"\nData folder not found: {args.data}")
        sys.exit(1)

    print(f"\n  Device:     {settings.device.upper()}")
    print(f"  Data root:  {args.data}")
    print(f"  Epochs:     {settings.num_epochs}")
    print(f"  Batch size: {settings.train_batch_size}")
    print(f"  LR:         {settings.learning_rate}")

    from training.trainer import Trainer
    trainer = Trainer(data_root=args.data)
    trainer.train()


if __name__ == "__main__":
    main()
