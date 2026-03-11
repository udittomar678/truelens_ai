"""
services/frequency_service.py
------------------------------
TrueLens AI — Frequency Domain Forensic Analyser

AI-generated images (especially GANs and diffusion models) leave
distinctive signatures in the frequency domain:
  - GANs produce a characteristic grid pattern in FFT space
  - Diffusion models show unusual high-frequency energy distribution
  - Real photos have smooth, natural spectral roll-off

This service computes interpretable frequency features and an
anomaly score — separate from the CNN's internal FFT branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from core.schemas import FrequencyFeatures
from utils.logger import get_logger

log = get_logger(__name__)


class FrequencyAnalysisService:
    """
    Computes FFT-based forensic features from an image.

    Features computed:
      1. fft_anomaly_score        — overall spectral anomaly [0, 1]
      2. dominant_frequency_ratio — ratio of high-freq to total energy
      3. spectral_flatness        — Wiener entropy (natural vs synthetic)
    """

    # Frequency band split: pixels above this fraction of Nyquist
    # are considered "high frequency"
    HIGH_FREQ_THRESHOLD: float = 0.3

    def analyse(self, image_path: Path) -> FrequencyFeatures:
        """
        Compute frequency-domain forensic features.

        Args:
            image_path: Path to the image file.

        Returns:
            FrequencyFeatures schema with scores.
        """
        try:
            img = Image.open(image_path).convert("L")  # grayscale for FFT
            arr = np.array(img, dtype=np.float32) / 255.0
        except Exception as exc:
            log.warning(
                "frequency_analysis_failed",
                path=str(image_path),
                error=str(exc),
            )
            # Return neutral scores on failure
            return FrequencyFeatures(
                fft_anomaly_score=0.5,
                dominant_frequency_ratio=0.5,
                spectral_flatness=0.5,
            )

        # ── 2D FFT ───────────────────────────────────────────────
        fft        = np.fft.fft2(arr)
        fft_shift  = np.fft.fftshift(fft)
        magnitude  = np.abs(fft_shift)
        log_mag    = np.log1p(magnitude)

        # ── Feature 1: High-frequency energy ratio ───────────────
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Radius of "low frequency" region
        low_r = int(min(h, w) * self.HIGH_FREQ_THRESHOLD)

        # Create circular mask for low-frequency region
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_centre = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        low_freq_mask  = dist_from_centre <= low_r
        high_freq_mask = ~low_freq_mask

        total_energy    = float(np.sum(magnitude ** 2)) + 1e-8
        high_freq_energy = float(np.sum((magnitude * high_freq_mask) ** 2))
        dominant_freq_ratio = high_freq_energy / total_energy

        # ── Feature 2: Spectral flatness (Wiener entropy) ────────
        # Ratio of geometric mean to arithmetic mean of spectrum
        # Flat spectrum → high value (more synthetic-looking)
        # Peaked spectrum → low value (more natural)
        flat_mag = log_mag.flatten() + 1e-8
        geo_mean = float(np.exp(np.mean(np.log(flat_mag))))
        ari_mean = float(np.mean(flat_mag))
        spectral_flatness = geo_mean / (ari_mean + 1e-8)
        spectral_flatness = float(np.clip(spectral_flatness, 0.0, 1.0))

        # ── Feature 3: GAN grid artefact detection ───────────────
        # GANs often produce periodic spikes in the FFT.
        # We detect this by looking for abnormal peaks in the spectrum.
        mean_mag   = float(np.mean(log_mag))
        std_mag    = float(np.std(log_mag))
        peak_ratio = float(np.max(log_mag)) / (mean_mag + 1e-8)

        # Normalise peak ratio: values > 10 are very suspicious
        normalised_peak = float(np.clip((peak_ratio - 1.0) / 20.0, 0.0, 1.0))

        # ── Composite anomaly score ───────────────────────────────
        # Weighted combination of all signals
        fft_anomaly = (
            0.4 * float(np.clip(dominant_freq_ratio * 3.0, 0.0, 1.0)) +
            0.3 * spectral_flatness +
            0.3 * normalised_peak
        )
        fft_anomaly = float(np.clip(fft_anomaly, 0.0, 1.0))

        log.debug(
            "frequency_analysis_complete",
            fft_anomaly=round(fft_anomaly, 4),
            dominant_freq_ratio=round(dominant_freq_ratio, 4),
            spectral_flatness=round(spectral_flatness, 4),
        )

        return FrequencyFeatures(
            fft_anomaly_score=round(fft_anomaly, 4),
            dominant_frequency_ratio=round(dominant_freq_ratio, 4),
            spectral_flatness=round(spectral_flatness, 4),
        )


