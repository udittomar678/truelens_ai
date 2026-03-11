"""
core/schemas.py
---------------
Pydantic v2 request / response schemas for TrueLens AI.
A single source of truth for all data contracts — avoids drift between
API layer and service layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


# ── Enumerations ──────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


# ── Sub-models ───────────────────────────────────────────────────

class FrequencyFeatures(BaseModel):
    """Spectral analysis sub-result."""
    model_config = ConfigDict(frozen=True)

    fft_anomaly_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="0 = natural spectrum, 1 = highly anomalous"
    )
    dominant_frequency_ratio: float = Field(
        ..., description="Ratio of high-frequency energy to total energy"
    )
    spectral_flatness: float = Field(
        ..., description="Wiener entropy of the frequency spectrum"
    )


class ExifAnalysis(BaseModel):
    """EXIF metadata analysis sub-result."""
    model_config = ConfigDict(frozen=True)

    has_exif: bool
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None
    gps_present: bool = False
    timestamp: str | None = None
    suspicious_flags: list[str] = Field(default_factory=list)
    exif_anomaly_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="0 = normal EXIF, 1 = suspicious / missing"
    )


class ExplainabilityData(BaseModel):
    """Optional Grad-CAM / heatmap payload."""
    model_config = ConfigDict(frozen=True)

    heatmap_base64: str = Field(
        ..., description="Base64-encoded PNG of the Grad-CAM overlay"
    )
    top_regions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Bounding boxes of highest-activation regions"
    )


# ── Primary response schemas ──────────────────────────────────────

class ImageAnalysisResponse(BaseModel):
    """Full response for POST /analyze-image."""
    model_config = ConfigDict(frozen=True)

    filename: str
    media_type: MediaType = MediaType.IMAGE

    # Core verdict
    ai_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability that the image is AI-generated"
    )
    real_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability that the image is real (1 - ai_probability)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence in its prediction"
    )
    risk_level: RiskLevel

    # Sub-analyses
    frequency_features: FrequencyFeatures
    exif_analysis: ExifAnalysis

    # Fusion metadata
    model_score: float = Field(..., ge=0.0, le=1.0)
    metadata_score: float = Field(..., ge=0.0, le=1.0)
    fused_score: float = Field(..., ge=0.0, le=1.0)

    # Optional explainability
    explainability: ExplainabilityData | None = None

    # Audit
    processing_time_ms: float
    model_version: str


class VideoAnalysisResponse(BaseModel):
    """Full response for POST /analyze-video."""
    model_config = ConfigDict(frozen=True)

    filename: str
    media_type: MediaType = MediaType.VIDEO

    # Aggregated verdict
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel

    # Frame-level breakdown
    frames_analysed: int
    frame_scores: list[float] = Field(
        ..., description="Per-frame AI probability scores"
    )
    score_std_dev: float = Field(
        ..., description="Std deviation across frames — high = inconsistent"
    )

    # Audit
    processing_time_ms: float
    model_version: str


# ── Error schema ─────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    code: str
