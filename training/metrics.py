"""
training/metrics.py
--------------------
TrueLens AI — Training Metrics

Computes and tracks:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

Keeps a running history so training curves can be plotted
or logged after each epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    phase: str = "train"       # "train" or "val"

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "phase": self.phase,
            "loss": round(self.loss, 6),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
        }

    def __str__(self) -> str:
        return (
            f"[{self.phase.upper():5s}] Epoch {self.epoch:03d} | "
            f"Loss: {self.loss:.4f} | "
            f"Acc: {self.accuracy:.4f} | "
            f"F1: {self.f1:.4f} | "
            f"AUC: {self.roc_auc:.4f}"
        )


class MetricsTracker:
    """
    Accumulates predictions and labels during an epoch,
    then computes all metrics at once when finalize() is called.

    Usage::

        tracker = MetricsTracker()
        for batch in loader:
            ...
            tracker.update(predictions, labels, loss)
        metrics = tracker.finalize(epoch=1, phase="val")
    """

    def __init__(self) -> None:
        self._all_preds:  list[int]   = []
        self._all_labels: list[int]   = []
        self._all_probs:  list[float] = []   # P(ai_generated)
        self._losses:     list[float] = []

    def update(
        self,
        predictions: list[int],
        labels: list[int],
        probs: list[float],
        loss: float,
    ) -> None:
        """
        Add a batch of results.

        Args:
            predictions: Predicted class indices (0=real, 1=ai).
            labels:      Ground truth class indices.
            probs:       Predicted probability of class 1 (ai_generated).
            loss:        Scalar batch loss.
        """
        self._all_preds.extend(predictions)
        self._all_labels.extend(labels)
        self._all_probs.extend(probs)
        self._losses.append(loss)

    def finalize(self, epoch: int, phase: str = "train") -> EpochMetrics:
        """
        Compute all metrics from accumulated data.

        Args:
            epoch: Current epoch number.
            phase: "train" or "val".

        Returns:
            EpochMetrics dataclass.
        """
        y_true = np.array(self._all_labels)
        y_pred = np.array(self._all_preds)
        y_prob = np.array(self._all_probs)

        avg_loss = float(np.mean(self._losses))

        accuracy  = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(
            y_true, y_pred, zero_division=0, average="binary"
        ))
        recall = float(recall_score(
            y_true, y_pred, zero_division=0, average="binary"
        ))
        f1 = float(f1_score(
            y_true, y_pred, zero_division=0, average="binary"
        ))

        # ROC-AUC needs probability scores
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # Only one class present in batch — happens in small datasets
            roc_auc = 0.5

        metrics = EpochMetrics(
            epoch=epoch,
            loss=avg_loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            phase=phase,
        )

        log.info("epoch_metrics", **metrics.to_dict())
        return metrics

    def reset(self) -> None:
        """Clear all accumulated data — call at start of each epoch."""
        self._all_preds.clear()
        self._all_labels.clear()
        self._all_probs.clear()
        self._losses.clear()


class TrainingHistory:
    """
    Stores the full history of train and val metrics across all epochs.
    Used for early stopping and saving the best model.
    """

    def __init__(self) -> None:
        self.train_history: list[EpochMetrics] = []
        self.val_history:   list[EpochMetrics] = []

    def add(self, metrics: EpochMetrics) -> None:
        if metrics.phase == "train":
            self.train_history.append(metrics)
        else:
            self.val_history.append(metrics)

    @property
    def best_val_f1(self) -> float:
        if not self.val_history:
            return 0.0
        return max(m.f1 for m in self.val_history)

    @property
    def best_val_accuracy(self) -> float:
        if not self.val_history:
            return 0.0
        return max(m.accuracy for m in self.val_history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": [m.to_dict() for m in self.train_history],
            "val":   [m.to_dict() for m in self.val_history],
        }


