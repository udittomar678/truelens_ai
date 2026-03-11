"""
services/inference_service.py
------------------------------
TrueLens AI — Inference Service

Responsible for:
  - Loading the trained model once at startup
  - Preprocessing images from disk or PIL
  - Running forward pass (single + batch)
  - Returning clean probability outputs
  - Proper error handling for corrupt / invalid files
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError

from core.config import settings
from core.schemas import RiskLevel
from models.dual_branch_cnn import TrueLensModel
from models.model_loader import build_model, get_device, load_weights
from models.transforms import get_inference_transforms
from utils.logger import get_logger

log = get_logger(__name__)


class InferenceResult:
    """Lightweight result container for a single image inference."""

    __slots__ = (
        "ai_probability",
        "real_probability",
        "confidence",
        "risk_level",
        "model_score",
    )

    def __init__(
        self,
        ai_probability: float,
        real_probability: float,
        confidence: float,
        risk_level: RiskLevel,
    ) -> None:
        self.ai_probability = ai_probability
        self.real_probability = real_probability
        self.confidence = confidence
        self.risk_level = risk_level
        self.model_score = ai_probability  # alias used by fusion service

    def __repr__(self) -> str:
        return (
            f"InferenceResult("
            f"ai={self.ai_probability:.3f}, "
            f"real={self.real_probability:.3f}, "
            f"risk={self.risk_level})"
        )


class InferenceService:
    """
    Singleton-style service that holds the loaded model and runs inference.

    Lifecycle:
      1. Instantiated once during FastAPI lifespan startup.
      2. Stored on app.state.inference_service.
      3. Injected into route handlers via request.app.state.

    Thread safety:
      PyTorch inference with torch.no_grad() is safe for concurrent
      reads on a single model instance.
    """

    MODEL_VERSION = "1.0.0"

    def __init__(self) -> None:
        self._device: torch.device = get_device()
        self._model: TrueLensModel | None = None
        self._transforms = get_inference_transforms()
        self._loaded: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Load model weights and run a dummy forward pass to initialise
        all CUDA/MPS kernels.  Called once at server startup.
        """
        weights_path = settings.model_dir / settings.model_weights_file

        if weights_path.exists():
            log.info("loading_model_weights", path=str(weights_path))
            self._model = build_model(pretrained=False)
            self._model = load_weights(self._model, weights_path)
        else:
            # No weights yet — build with random weights for structure
            # This allows the API to start; inference will be unreliable
            # until training completes and weights are saved.
            log.warning(
                "model_weights_not_found",
                path=str(weights_path),
                note="Using random weights — train the model first.",
            )
            self._model = build_model(pretrained=True)

        self._model.eval()

        # Warm up: single dummy forward pass
        dummy = torch.zeros(
            1, 3, settings.image_size, settings.image_size,
            device=self._device,
        )
        with torch.no_grad():
            _ = self._model(dummy)

        self._loaded = True
        log.info("inference_service_warmed_up", device=str(self._device))

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    # ── Image loading ─────────────────────────────────────────────

    def _load_image(self, image_path: Path) -> Image.Image:
        """
        Load and validate an image from disk.

        Raises:
            ValueError: If the file is not a valid image.
        """
        try:
            img = Image.open(image_path)
            img.verify()                            # catches truncated files
            img = Image.open(image_path)            # re-open after verify
            img = img.convert("RGB")                # ensure 3-channel RGB
            return img
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file: {image_path.name}")
        except Exception as exc:
            raise ValueError(f"Failed to load image {image_path.name}: {exc}")

    def _preprocess(
        self,
        image: Union[Image.Image, Path],
    ) -> torch.Tensor:
        """
        Apply inference transforms and move tensor to device.

        Args:
            image: PIL Image or path to image file.

        Returns:
            Float tensor (1, 3, H, W) on the configured device.
        """
        if isinstance(image, Path):
            image = self._load_image(image)

        tensor: torch.Tensor = self._transforms(image)  # (3, H, W)
        tensor = tensor.unsqueeze(0)                     # (1, 3, H, W)
        tensor = tensor.to(self._device)
        return tensor

    # ── Inference ─────────────────────────────────────────────────

    def _compute_risk_level(self, ai_probability: float) -> RiskLevel:
        """Map AI probability to a risk level."""
        if ai_probability >= settings.confidence_threshold_high:
            return RiskLevel.HIGH
        elif ai_probability >= settings.confidence_threshold_low:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def predict_single(
        self,
        image: Union[Image.Image, Path],
    ) -> InferenceResult:
        """
        Run inference on a single image.

        Args:
            image: PIL Image or path to image file.

        Returns:
            InferenceResult with probabilities and risk level.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError:   If the image cannot be loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Call warmup() before running inference."
            )

        tensor = self._preprocess(image)

        with torch.no_grad():
            probs = self._model.predict_proba(tensor)  # (1, 2)

        probs_np: np.ndarray = probs.cpu().numpy()[0]  # [p_real, p_ai]

        real_prob = float(probs_np[TrueLensModel.LABEL_REAL])
        ai_prob   = float(probs_np[TrueLensModel.LABEL_AI])

        # Confidence = how far from 0.5 the prediction is
        confidence = float(abs(ai_prob - 0.5) * 2)    # maps [0.5,1.0] → [0,1]
        risk_level = self._compute_risk_level(ai_prob)

        log.debug(
            "single_inference_complete",
            ai_probability=round(ai_prob, 4),
            confidence=round(confidence, 4),
            risk_level=risk_level,
        )

        return InferenceResult(
            ai_probability=ai_prob,
            real_probability=real_prob,
            confidence=confidence,
            risk_level=risk_level,
        )

    def predict_batch(
        self,
        images: list[Union[Image.Image, Path]],
    ) -> list[InferenceResult]:
        """
        Run inference on a batch of images efficiently.

        Args:
            images: List of PIL Images or file paths.

        Returns:
            List of InferenceResult, one per input image.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call warmup() first.")

        if not images:
            return []

        # Stack all tensors into a single batch
        tensors = [self._preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)              # (B, 3, H, W)

        with torch.no_grad():
            probs = self._model.predict_proba(batch)   # (B, 2)

        probs_np: np.ndarray = probs.cpu().numpy()

        results: list[InferenceResult] = []
        for row in probs_np:
            real_prob = float(row[TrueLensModel.LABEL_REAL])
            ai_prob   = float(row[TrueLensModel.LABEL_AI])
            confidence = float(abs(ai_prob - 0.5) * 2)
            risk_level = self._compute_risk_level(ai_prob)
            results.append(InferenceResult(
                ai_probability=ai_prob,
                real_probability=real_prob,
                confidence=confidence,
                risk_level=risk_level,
            ))

        log.debug("batch_inference_complete", batch_size=len(images))
        return results


