"""
core/config.py
--------------
Centralised, environment-aware configuration for TrueLens AI.
All tuneable parameters live here; nothing is hard-coded elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings.  Override any value via environment variable
    (e.g.  TRUELENS_ENV=production) or a `.env` file at the project root."""

    model_config = SettingsConfigDict(
        env_prefix="TRUELENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────
    app_name: str = "TrueLens AI"
    app_version: str = "1.0.0"
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ── Server ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # ── Paths ────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parents[1]
    model_dir: Path = Field(default=None)          # resolved below
    log_dir: Path = Field(default=None)
    upload_tmp_dir: Path = Field(default=None)

    # ── Model ────────────────────────────────────────────────────
    image_backbone: str = "efficientnet_b3"        # timm model name
    image_size: int = 224                          # input resolution
    num_classes: int = 2                           # real / ai
    model_weights_file: str = "truelens_image.pth"

    # ── Inference ────────────────────────────────────────────────
    confidence_threshold_low: float = 0.55         # below → LOW risk
    confidence_threshold_high: float = 0.80        # above → HIGH risk
    max_image_mb: float = 20.0
    max_video_mb: float = 200.0
    max_video_frames: int = 64                     # frames sampled per video

    # ── Training ─────────────────────────────────────────────────
    train_batch_size: int = 32
    val_batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    train_val_split: float = 0.2                   # fraction used for validation

    # ── Device ───────────────────────────────────────────────────
    device: str = "auto"                           # auto | cpu | cuda | mps

    # ── Explainability ───────────────────────────────────────────
    gradcam_layer: str = "spatial_branch.backbone.blocks[-1]"
    heatmap_alpha: float = 0.5

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = "INFO"
    log_to_file: bool = True
    structured_log_file: str = "analysis_log.jsonl"

    # ── Validators ───────────────────────────────────────────────
    @field_validator("model_dir", "log_dir", "upload_tmp_dir", mode="before")
    @classmethod
    def _set_default_paths(cls, v: Path | None, info) -> Path:
        if v is not None:
            return Path(v)
        defaults = {
            "model_dir": "weights",
            "log_dir": "logs",
            "upload_tmp_dir": "tmp_uploads",
        }
        # info.field_name available in pydantic v2
        field = info.field_name
        base = Path(__file__).resolve().parents[1]
        return base / defaults.get(field, field)

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, v: str) -> str:
        if v != "auto":
            return v
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def ensure_directories(self) -> None:
        """Create required directories if they don't already exist."""
        for d in (self.model_dir, self.log_dir, self.upload_tmp_dir):
            d.mkdir(parents=True, exist_ok=True)


# ── Singleton ─────────────────────────────────────────────────────
settings = Settings()
settings.ensure_directories()
