"""
services/video_service.py
--------------------------
TrueLens AI — Video Analysis Service

Extracts evenly-spaced frames from a video using OpenCV,
runs per-frame CNN inference via InferenceService,
and returns an aggregated forensic verdict.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import settings
from core.schemas import MediaType, RiskLevel, VideoAnalysisResponse
from services.inference_service import InferenceService
from utils.logger import get_logger

log = get_logger(__name__)


class VideoAnalysisService:
    """
    Analyses a video file by:
      1. Extracting up to `max_video_frames` evenly-spaced frames.
      2. Running CNN inference on each frame.
      3. Aggregating frame scores into a final verdict.
    """

    def __init__(self, inference_service: InferenceService) -> None:
        self._inference = inference_service

    def _extract_frames(self, video_path: Path) -> list[Image.Image]:
        """
        Extract evenly-spaced frames from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of PIL Images (RGB).

        Raises:
            ValueError: If the video cannot be opened or has no frames.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path.name}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path.name}")

        log.info(
            "video_opened",
            filename=video_path.name,
            total_frames=total_frames,
            fps=round(fps, 2),
        )

        # Evenly sample up to max_video_frames indices
        max_frames = settings.max_video_frames
        n_frames   = min(max_frames, total_frames)
        indices    = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            # OpenCV reads BGR — convert to RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)

        cap.release()

        if not frames:
            raise ValueError("Could not extract any frames from video.")

        log.info("frames_extracted", count=len(frames))
        return frames

    def _aggregate_scores(self, scores: list[float]) -> dict:
        """
        Aggregate per-frame AI probability scores into a verdict.

        Uses mean as primary signal, with std dev as a consistency flag.
        High std dev means the video has mixed real and AI frames —
        which itself is suspicious (deepfake splice indicator).
        """
        arr = np.array(scores, dtype=np.float32)
        mean_score = float(np.mean(arr))
        std_score  = float(np.std(arr))

        # Boost score slightly if std dev is high (inconsistency = suspicious)
        consistency_penalty = min(std_score * 0.3, 0.1)
        fused = float(np.clip(mean_score + consistency_penalty, 0.0, 1.0))

        return {
            "fused_score": round(fused, 4),
            "std_dev": round(std_score, 4),
        }

    def _classify_risk(self, score: float) -> RiskLevel:
        if score >= settings.confidence_threshold_high:
            return RiskLevel.HIGH
        elif score >= settings.confidence_threshold_low:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    async def analyse_video(
        self,
        video_path: Path,
        filename: str,
    ) -> VideoAnalysisResponse:
        """
        Full video forensic analysis pipeline.

        Args:
            video_path: Path to the temp-saved video file.
            filename:   Original filename for the response.

        Returns:
            VideoAnalysisResponse with per-frame scores and verdict.
        """
        # ── Extract frames ────────────────────────────────────────
        frames = self._extract_frames(video_path)

        # ── Run batch inference ───────────────────────────────────
        results = self._inference.predict_batch(frames)
        frame_scores = [r.ai_probability for r in results]

        # ── Aggregate ─────────────────────────────────────────────
        aggregated   = self._aggregate_scores(frame_scores)
        fused_score  = aggregated["fused_score"]
        std_dev      = aggregated["std_dev"]
        risk_level   = self._classify_risk(fused_score)
        confidence   = round(abs(fused_score - 0.5) * 2, 4)

        log.info(
            "video_analysis_complete",
            filename=filename,
            frames_analysed=len(frame_scores),
            mean_ai_probability=fused_score,
            std_dev=std_dev,
            risk_level=risk_level,
        )

        return VideoAnalysisResponse(
            filename=filename,
            media_type=MediaType.VIDEO,
            ai_probability=fused_score,
            real_probability=round(1.0 - fused_score, 4),
            confidence=confidence,
            risk_level=risk_level,
            frames_analysed=len(frame_scores),
            frame_scores=frame_scores,
            score_std_dev=std_dev,
            processing_time_ms=0.0,
            model_version=InferenceService.MODEL_VERSION,
        )


