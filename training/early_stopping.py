"""
training/early_stopping.py
---------------------------
TrueLens AI — Early Stopping

Stops training when validation performance stops improving,
preventing overfitting and saving compute time.

Monitors validation F1 score by default (better than accuracy
for imbalanced datasets).
"""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger(__name__)


class EarlyStopping:
    """
    Stops training if the monitored metric does not improve
    for `patience` consecutive epochs.

    Usage::

        stopper = EarlyStopping(patience=5)
        for epoch in range(num_epochs):
            val_metrics = ...
            if stopper(val_metrics.f1):
                print("Early stopping triggered")
                break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        monitor: str = "f1",
    ) -> None:
        """
        Args:
            patience:  Number of epochs to wait without improvement.
            min_delta: Minimum change to qualify as improvement.
            monitor:   Metric name being monitored (for logging only).
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.monitor    = monitor

        self._best_score: float | None = None
        self._counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Call after each validation epoch.

        Args:
            score: Current epoch's monitored metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        if self._best_score is None:
            # First epoch — set baseline
            self._best_score = score
            self.best_epoch  = epoch
            return False

        improvement = score - self._best_score

        if improvement > self.min_delta:
            # Improved — reset counter
            log.info(
                "early_stopping_improvement",
                monitor=self.monitor,
                prev=round(self._best_score, 4),
                current=round(score, 4),
                delta=round(improvement, 4),
            )
            self._best_score = score
            self._counter    = 0
            self.best_epoch  = epoch
        else:
            # No improvement
            self._counter += 1
            log.info(
                "early_stopping_no_improvement",
                monitor=self.monitor,
                best=round(self._best_score, 4),
                current=round(score, 4),
                patience_used=f"{self._counter}/{self.patience}",
            )

            if self._counter >= self.patience:
                self.should_stop = True
                log.info(
                    "early_stopping_triggered",
                    best_epoch=self.best_epoch,
                    best_score=round(self._best_score, 4),
                )
                return True

        return False

    def reset(self) -> None:
        """Reset state — useful for fine-tuning runs."""
        self._best_score = None
        self._counter    = 0
        self.should_stop = False
        self.best_epoch  = 0


