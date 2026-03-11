"""
services/forensic_fusion_service.py
-------------------------------------
TrueLens AI — Forensic Fusion Engine

Combines signals from three sources into a single verdict:
  1. CNN model score      (spatial + frequency branches)
  2. EXIF anomaly score   (metadata forensics)
  3. FFT anomaly score    (standalone frequency analysis)

Fusion weights are tuned so the CNN dominates but metadata
provides meaningful signal — especially for images with
obvious AI software tags or completely missing EXIF.
"""

from __future__ import annotations

from pathlib import Path

from core.config import settings
from core.schemas import (
    ExifAnalysis,
    FrequencyFeatures,
    ImageAnalysisResponse,
    MediaType,
    RiskLevel,
)
from services.exif_service import ExifAnalysisService
from services.frequency_service import FrequencyAnalysisService
from services.inference_service import InferenceService
from utils.logger import get_logger

log = get_logger(__name__)

# ── Fusion weights — must sum to 1.0 ─────────────────────────────
_W_MODEL = 0.65      # CNN model score
_W_EXIF  = 0.20      # EXIF anomaly score
_W_FFT   = 0.15      # Standalone FFT score


class ForensicFusionService:
    """
    Orchestrates the full forensic analysis pipeline for a single image.

    Flow:
      image_path
        → InferenceService.predict_single()    → model_score
        → ExifAnalysisService.analyse()        → exif_score
        → FrequencyAnalysisService.analyse()   → fft_score
        → weighted fusion                      → fused_score
        → risk classification                  → ImageAnalysisResponse
    """

    def __init__(self, inference_service: InferenceService) -> None:
        self._inference  = inference_service
        self._exif_svc   = ExifAnalysisService()
        self._freq_svc   = FrequencyAnalysisService()

    def _classify_risk(self, fused_score: float) -> RiskLevel:
        """Map fused score to a risk level."""
        if fused_score >= settings.confidence_threshold_high:
            return RiskLevel.HIGH
        elif fused_score >= settings.confidence_threshold_low:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _fuse_scores(
        self,
        model_score: float,
        exif_score: float,
        fft_score: float,
    ) -> float:
        """Weighted linear fusion of all three signals."""
        fused = (
            _W_MODEL * model_score +
            _W_EXIF  * exif_score  +
            _W_FFT   * fft_score
        )
        return round(float(fused), 4)

    async def analyse_image(
        self,
        image_path: Path,
        filename: str,
        include_explainability: bool = False,
    ) -> ImageAnalysisResponse:
        """
        Full forensic analysis pipeline for one image.

        Args:
            image_path:             Path to the temp-saved image file.
            filename:               Original filename for the response.
            include_explainability: If True, generate Grad-CAM heatmap.

        Returns:
            Complete ImageAnalysisResponse ready to return from the API.
        """
        # ── 1. CNN inference ──────────────────────────────────────
        inference_result = self._inference.predict_single(image_path)
        model_score = inference_result.ai_probability

        # ── 2. EXIF analysis ──────────────────────────────────────
        exif_analysis: ExifAnalysis = self._exif_svc.analyse(image_path)
        exif_score = exif_analysis.exif_anomaly_score

        # ── 3. FFT frequency analysis ─────────────────────────────
        freq_features: FrequencyFeatures = self._freq_svc.analyse(image_path)
        fft_score = freq_features.fft_anomaly_score

        # ── 4. Fuse all scores ────────────────────────────────────
        fused_score = self._fuse_scores(model_score, exif_score, fft_score)
        risk_level  = self._classify_risk(fused_score)

        # ── 5. Recalculate confidence from fused score ────────────
        confidence = round(abs(fused_score - 0.5) * 2, 4)

        log.info(
            "fusion_complete",
            filename=filename,
            model_score=round(model_score, 4),
            exif_score=round(exif_score, 4),
            fft_score=round(fft_score, 4),
            fused_score=fused_score,
            risk_level=risk_level,
        )

        # ── 6. Optional Grad-CAM explainability ───────────────────
        explainability = None
        if include_explainability:
            try:
                from explainability.gradcam import GradCAMService
                gradcam_svc = GradCAMService(self._inference._model)
                explainability = gradcam_svc.generate(
                    image_path=image_path,
                    device=self._inference._device,
                )
            except Exception as exc:
                log.warning("gradcam_failed", error=str(exc))

        # ── 7. Build response ─────────────────────────────────────
        return ImageAnalysisResponse(
            filename=filename,
            media_type=MediaType.IMAGE,
            ai_probability=fused_score,
            real_probability=round(1.0 - fused_score, 4),
            confidence=confidence,
            risk_level=risk_level,
            frequency_features=freq_features,
            exif_analysis=exif_analysis,
            model_score=round(model_score, 4),
            metadata_score=round(exif_score, 4),
            fused_score=fused_score,
            explainability=explainability,
            processing_time_ms=0.0,         # updated by route handler
            model_version=InferenceService.MODEL_VERSION,
        )


