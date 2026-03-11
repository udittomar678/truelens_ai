"""
training/trainer.py
--------------------
TrueLens AI — Training Pipeline

Full training loop with:
  - Mixed precision support (where available)
  - Early stopping
  - Best model checkpointing
  - Per-epoch metric logging
  - Clean progress display
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from core.config import settings
from models.dual_branch_cnn import TrueLensModel
from models.model_loader import build_model, get_device, save_checkpoint
from training.dataset import TrueLensDataset
from training.early_stopping import EarlyStopping
from training.metrics import MetricsTracker, TrainingHistory
from utils.logger import get_logger

log = get_logger(__name__)


class Trainer:
    """
    Manages the full training lifecycle for TrueLensModel.

    Usage::

        trainer = Trainer(data_root=Path("data"))
        trainer.train()
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root  = data_root
        self.device     = get_device()
        self.history    = TrainingHistory()

    def _build_optimizer(
        self, model: TrueLensModel
    ) -> tuple[AdamW, CosineAnnealingLR]:
        """
        AdamW with weight decay + cosine annealing LR schedule.
        Lower LR for the pretrained backbone, higher for new layers.
        """
        backbone_params = list(model.spatial_branch.backbone.parameters())
        new_params = (
            list(model.spatial_branch.projection.parameters()) +
            list(model.frequency_branch.parameters()) +
            list(model.fusion_classifier.parameters())
        )

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": settings.learning_rate * 0.1},
                {"params": new_params,      "lr": settings.learning_rate},
            ],
            weight_decay=settings.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=settings.num_epochs,
            eta_min=1e-6,
        )

        return optimizer, scheduler

    def _run_epoch(
        self,
        model: TrueLensModel,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: AdamW | None,
        phase: str,
        epoch: int,
    ):
        """
        Run one full epoch (train or val).

        Args:
            model:     The model to train/evaluate.
            loader:    DataLoader for this phase.
            criterion: Loss function.
            optimizer: Optimizer (None during validation).
            phase:     "train" or "val".
            epoch:     Current epoch number.
        """
        is_train = phase == "train"
        model.train() if is_train else model.eval()

        tracker = MetricsTracker()
        n_batches = len(loader)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(is_train):
                logits = model(images)                      # (B, 2)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # ── Collect predictions ───────────────────────────────
            probs = torch.softmax(logits.detach(), dim=1)
            preds = torch.argmax(probs, dim=1)

            tracker.update(
                predictions=preds.cpu().tolist(),
                labels=labels.cpu().tolist(),
                probs=probs[:, 1].cpu().tolist(),  # P(ai_generated)
                loss=loss.item(),
            )

            # ── Progress print every 10 batches ──────────────────
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                print(
                    f"  [{phase.upper()}] Epoch {epoch:03d} "
                    f"Batch {batch_idx+1:04d}/{n_batches:04d} "
                    f"Loss: {loss.item():.4f}",
                    end="\r",
                )

        print()  # newline after \r progress
        return tracker.finalize(epoch=epoch, phase=phase)

    def train(self) -> TrainingHistory:
        """
        Run the full training pipeline.

        Returns:
            TrainingHistory with all epoch metrics.
        """
        log.info("training_started", data_root=str(self.data_root))
        t_start = time.time()

        # ── Dataset & Loaders ─────────────────────────────────────
        dataset = TrueLensDataset(root=self.data_root)
        train_loader, val_loader = dataset.get_loaders()

        # ── Model ─────────────────────────────────────────────────
        model = build_model(pretrained=True)

        # ── Loss — weighted cross-entropy handles class imbalance ─
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # ── Optimiser & Scheduler ─────────────────────────────────
        optimizer, scheduler = self._build_optimizer(model)

        # ── Early stopping ────────────────────────────────────────
        stopper = EarlyStopping(
            patience=settings.early_stopping_patience,
            monitor="f1",
        )

        best_val_f1   = 0.0
        best_val_acc  = 0.0

        print(f"\n{'='*60}")
        print(f"  TrueLens AI — Training on {str(self.device).upper()}")
        print(f"  Epochs: {settings.num_epochs}  |  "
              f"Batch: {settings.train_batch_size}  |  "
              f"LR: {settings.learning_rate}")
        print(f"{'='*60}\n")

        for epoch in range(1, settings.num_epochs + 1):
            epoch_start = time.time()

            # ── Train ─────────────────────────────────────────────
            train_metrics = self._run_epoch(
                model, train_loader, criterion, optimizer, "train", epoch
            )
            self.history.add(train_metrics)

            # ── Validate ──────────────────────────────────────────
            val_metrics = self._run_epoch(
                model, val_loader, criterion, None, "val", epoch
            )
            self.history.add(val_metrics)

            scheduler.step()

            epoch_time = time.time() - epoch_start

            # ── Print epoch summary ───────────────────────────────
            print(f"\n{str(train_metrics)}")
            print(f"{str(val_metrics)}")
            print(f"  Epoch time: {epoch_time:.1f}s  |  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  {'-'*56}")

            # ── Save best model ───────────────────────────────────
            if val_metrics.f1 > best_val_f1:
                best_val_f1  = val_metrics.f1
                best_val_acc = val_metrics.accuracy
                save_checkpoint(
                    model=model,
                    epoch=epoch,
                    val_accuracy=val_metrics.accuracy,
                    val_loss=val_metrics.loss,
                    extra={"val_f1": val_metrics.f1},
                )
                print(f"  ✓ New best model saved  "
                      f"(F1: {best_val_f1:.4f}  Acc: {best_val_acc:.4f})")

            # ── Early stopping check ──────────────────────────────
            if stopper(val_metrics.f1, epoch):
                print(f"\n  Early stopping at epoch {epoch}.")
                print(f"  Best epoch: {stopper.best_epoch}  |  "
                      f"Best F1: {stopper._best_score:.4f}")
                break

        # ── Save training history ─────────────────────────────────
        history_path = settings.log_dir / "training_history.json"
        with open(history_path, "w") as fh:
            json.dump(self.history.to_dict(), fh, indent=2)

        total_time = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"  Training complete in {total_time/60:.1f} minutes")
        print(f"  Best Val F1:       {best_val_f1:.4f}")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        print(f"  Weights saved to:  {settings.model_dir}")
        print(f"  History saved to:  {history_path}")
        print(f"{'='*60}\n")

        log.info(
            "training_complete",
            best_val_f1=round(best_val_f1, 4),
            best_val_accuracy=round(best_val_acc, 4),
            total_minutes=round(total_time / 60, 1),
        )

        return self.history


