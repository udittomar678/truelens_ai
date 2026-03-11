"""
models/model_loader.py
-----------------------
Handles all model lifecycle operations:
  - Loading weights safely from disk
  - Device placement (CUDA / MPS / CPU)
  - Model versioning
  - Checkpoint saving

This is the single point of contact for anything related to
persisting or restoring the TrueLensModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.config import settings
from models.dual_branch_cnn import TrueLensModel
from utils.logger import get_logger

log = get_logger(__name__)


def get_device() -> torch.device:
    """
    Return the best available torch.device based on settings.

    Priority: CUDA > MPS > CPU
    The setting is already resolved at startup by the config validator,
    so we just honour it here.
    """
    device_str = settings.device
    device = torch.device(device_str)
    log.info("device_selected", device=str(device))
    return device


def build_model(pretrained: bool = True) -> TrueLensModel:
    """
    Instantiate a fresh TrueLensModel and move it to the configured device.

    Args:
        pretrained: Whether to initialise the EfficientNet backbone
                    with ImageNet weights.  Set False for unit tests.

    Returns:
        TrueLensModel on the correct device, in eval mode.
    """
    device = get_device()
    model = TrueLensModel(pretrained=pretrained)
    model = model.to(device)
    model.eval()

    info = model.get_model_info()
    log.info(
        "model_built",
        backbone=info["backbone"],
        total_params=f"{info['total_parameters']:,}",
        trainable_params=f"{info['trainable_parameters']:,}",
        device=str(device),
    )
    return model


def load_weights(
    model: TrueLensModel,
    weights_path: Path | None = None,
) -> TrueLensModel:
    """
    Load saved weights into an existing model instance.

    Args:
        model:        An instantiated TrueLensModel (already on device).
        weights_path: Path to a `.pth` checkpoint file.  Defaults to
                      `settings.model_dir / settings.model_weights_file`.

    Returns:
        The same model with weights loaded, in eval mode.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        RuntimeError:      If the checkpoint is incompatible.
    """
    if weights_path is None:
        weights_path = settings.model_dir / settings.model_weights_file

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Run the training pipeline first, or provide a valid path."
        )

    device = next(model.parameters()).device
    checkpoint: dict[str, Any] = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,          # safe loading — no arbitrary code exec
    )

    # Support both raw state_dict and full checkpoint dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        val_acc = checkpoint.get("val_accuracy", "unknown")
        log.info(
            "checkpoint_loaded",
            path=str(weights_path),
            epoch=epoch,
            val_accuracy=val_acc,
        )
    else:
        state_dict = checkpoint
        log.info("weights_loaded", path=str(weights_path))

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def save_checkpoint(
    model: TrueLensModel,
    epoch: int,
    val_accuracy: float,
    val_loss: float,
    extra: dict[str, Any] | None = None,
    filename: str | None = None,
) -> Path:
    """
    Save a full training checkpoint to disk.

    Args:
        model:        The model to save.
        epoch:        Current epoch number.
        val_accuracy: Validation accuracy at this epoch.
        val_loss:     Validation loss at this epoch.
        extra:        Any additional metadata to store in the checkpoint.
        filename:     Override the default filename.

    Returns:
        Path where the checkpoint was saved.
    """
    save_path = settings.model_dir / (filename or settings.model_weights_file)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "model_config": model.get_model_info(),
        **(extra or {}),
    }

    torch.save(checkpoint, save_path)
    log.info(
        "checkpoint_saved",
        path=str(save_path),
        epoch=epoch,
        val_accuracy=round(val_accuracy, 4),
    )
    return save_path


def load_model_for_inference(
    weights_path: Path | None = None,
) -> TrueLensModel:
    """
    Convenience function: build model + load weights in one call.
    This is what the InferenceService calls at startup.

    Args:
        weights_path: Optional override for the weights file path.

    Returns:
        Ready-to-use TrueLensModel on the correct device.
    """
    model = build_model(pretrained=False)   # don't download ImageNet weights
    model = load_weights(model, weights_path)
    return model


