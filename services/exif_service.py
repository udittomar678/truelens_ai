"""
services/exif_service.py
-------------------------
TrueLens AI — EXIF Metadata Forensic Analyser

Real photographs almost always contain EXIF metadata — camera make,
model, GPS, shutter speed, ISO, etc.  AI-generated images typically
have no EXIF, stripped EXIF, or suspicious software tags.

This service analyses EXIF data and returns an anomaly score.
"""

from __future__ import annotations

from pathlib import Path

import exifread

from core.schemas import ExifAnalysis
from utils.logger import get_logger

log = get_logger(__name__)

# ── Known AI-generation software signatures ───────────────────────
_AI_SOFTWARE_SIGNATURES = {
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "firefly", "imagen", "kandinsky", "novelai", "automatic1111",
    "comfyui", "invokeai", "adobe firefly", "generative fill",
}

# ── Suspicious / missing field patterns ──────────────────────────
_REAL_CAMERA_MAKES = {
    "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
    "leica", "hasselblad", "pentax", "ricoh", "samsung", "apple",
    "google", "huawei", "xiaomi", "oneplus",
}


class ExifAnalysisService:
    """
    Extracts and analyses EXIF metadata from an image file.

    Scoring logic:
      - No EXIF at all            → high suspicion (+0.4)
      - AI software tag found     → very high suspicion (+0.5)
      - Unknown camera make       → moderate suspicion (+0.2)
      - Missing typical fields    → light suspicion (+0.1 each)
      - GPS present               → slight real signal (-0.05)
      - Score clamped to [0, 1]
    """

    def analyse(self, image_path: Path) -> ExifAnalysis:
        """
        Analyse EXIF metadata of an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            ExifAnalysis schema with flags and anomaly score.
        """
        try:
            with open(image_path, "rb") as fh:
                tags = exifread.process_file(fh, details=False, strict=False)
        except Exception as exc:
            log.warning("exif_read_failed", path=str(image_path), error=str(exc))
            tags = {}

        has_exif = bool(tags)

        # ── Extract key fields ────────────────────────────────────
        camera_make  = self._get_tag(tags, "Image Make")
        camera_model = self._get_tag(tags, "Image Model")
        software     = self._get_tag(tags, "Image Software")
        timestamp    = self._get_tag(tags, "EXIF DateTimeOriginal") or \
                       self._get_tag(tags, "Image DateTime")
        gps_present  = any("GPS" in str(k) for k in tags)

        # ── Build suspicious flags ────────────────────────────────
        suspicious_flags: list[str] = []
        score: float = 0.0

        if not has_exif:
            suspicious_flags.append("NO_EXIF_DATA")
            score += 0.4

        if software:
            sw_lower = software.lower()
            for sig in _AI_SOFTWARE_SIGNATURES:
                if sig in sw_lower:
                    suspicious_flags.append(f"AI_SOFTWARE_DETECTED:{software}")
                    score += 0.5
                    break

        if has_exif and not camera_make:
            suspicious_flags.append("MISSING_CAMERA_MAKE")
            score += 0.2

        if camera_make:
            make_lower = camera_make.lower()
            if not any(brand in make_lower for brand in _REAL_CAMERA_MAKES):
                suspicious_flags.append(f"UNKNOWN_CAMERA_MAKE:{camera_make}")
                score += 0.15

        if has_exif and not timestamp:
            suspicious_flags.append("MISSING_TIMESTAMP")
            score += 0.1

        if has_exif and not camera_model:
            suspicious_flags.append("MISSING_CAMERA_MODEL")
            score += 0.1

        # GPS is a mild real-image signal
        if gps_present:
            score = max(0.0, score - 0.05)

        # Clamp to [0, 1]
        score = min(1.0, max(0.0, score))

        return ExifAnalysis(
            has_exif=has_exif,
            camera_make=camera_make,
            camera_model=camera_model,
            software=software,
            gps_present=gps_present,
            timestamp=timestamp,
            suspicious_flags=suspicious_flags,
            exif_anomaly_score=round(score, 4),
        )

    @staticmethod
    def _get_tag(tags: dict, key: str) -> str | None:
        """Safely extract a string value from an exifread tags dict."""
        value = tags.get(key)
        if value is None:
            return None
        result = str(value).strip()
        return result if result else None


