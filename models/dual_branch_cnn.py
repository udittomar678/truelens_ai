"""
models/dual_branch_cnn.py
--------------------------
TrueLens AI — Dual-Branch CNN Architecture

Architecture overview:
  ┌─────────────────┐     ┌──────────────────────┐
  │  Spatial Branch │     │  Frequency Branch    │
  │  (EfficientNet) │     │  (FFT → CNN)         │
  │  RGB image      │     │  Magnitude spectrum  │
  └────────┬────────┘     └──────────┬───────────┘
           │                         │
           └──────────┬──────────────┘
                      │
              ┌───────▼────────┐
              │  Fusion Layer  │
              │  (concat + MLP)│
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Classifier    │
              │  (real vs AI)  │
              └────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from core.config import settings


# ── Frequency Branch ──────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Converts an RGB image to its 2D FFT magnitude spectrum and
    extracts discriminative frequency-domain features via a small CNN.

    AI-generated images often show unnatural periodicity and spectral
    artefacts invisible to the human eye but detectable in FFT space.
    """

    def __init__(self, out_features: int = 256) -> None:
        super().__init__()

        # Small CNN operating on the 3-channel FFT magnitude map
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),            # 4x4 regardless of input
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                           # 256 * 4 * 4 = 4096
            nn.Linear(256 * 4 * 4, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    @staticmethod
    def _to_frequency_map(x: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of RGB images to their log-magnitude FFT spectrum.

        Steps:
          1. Compute 2D FFT per channel.
          2. Shift zero-frequency to centre.
          3. Take log of magnitude (compresses dynamic range).
          4. Normalise to [0, 1] per sample.

        Args:
            x: Float tensor of shape (B, 3, H, W) in range [0, 1].

        Returns:
            Float tensor of shape (B, 3, H, W).
        """
        # FFT over spatial dims — output is complex
        fft = torch.fft.fft2(x, norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Log magnitude spectrum
        magnitude = torch.abs(fft)
        log_magnitude = torch.log1p(magnitude)     # log(1 + |F|)

        # Per-sample min-max normalisation
        b = log_magnitude.shape[0]
        flat = log_magnitude.view(b, -1)
        min_v = flat.min(dim=1).values.view(b, 1, 1, 1)
        max_v = flat.max(dim=1).values.view(b, 1, 1, 1)
        normalised = (log_magnitude - min_v) / (max_v - min_v + 1e-8)

        return normalised

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_map = self._to_frequency_map(x)
        features = self.cnn(freq_map)
        return self.fc(features)


# ── Spatial Branch ────────────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    EfficientNet-B3 backbone (pretrained on ImageNet) with the
    classifier head removed.  Outputs a spatial feature vector.

    EfficientNet-B3 is chosen for its strong accuracy/efficiency
    trade-off on M1/M2 MPS devices.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        out_features: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Load backbone — drop classifier head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # removes the final FC layer
            global_pool="avg",      # global average pooling
        )

        backbone_out_dim: int = self.backbone.num_features

        # Projection head: compress backbone features
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                # (B, backbone_out_dim)
        return self.projection(features)           # (B, out_features)


# ── Fusion + Classifier ───────────────────────────────────────────

class FusionClassifier(nn.Module):
    """
    Combines spatial and frequency features via concatenation,
    then passes through an MLP to produce a binary classification.
    """

    def __init__(
        self,
        spatial_dim: int = 512,
        freq_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        fused_dim = spatial_dim + freq_dim         # 768

        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([spatial_feat, freq_feat], dim=1)  # (B, 768)
        return self.mlp(fused)                               # (B, num_classes)


# ── Full Model ────────────────────────────────────────────────────

class TrueLensModel(nn.Module):
    """
    Full dual-branch forensic CNN.

    Input : RGB image tensor — (B, 3, H, W), values in [0, 1]
    Output: Logits tensor   — (B, 2)  [real_logit, ai_logit]

    Use `predict_proba()` for softmax probabilities.

    Example::

        model = TrueLensModel()
        model.to(device)
        logits = model(images)                     # (B, 2) logits
        probs  = model.predict_proba(images)       # (B, 2) probabilities
    """

    # Class label mapping
    LABEL_REAL: int = 0
    LABEL_AI: int = 1

    def __init__(
        self,
        backbone_name: str = settings.image_backbone,
        spatial_out: int = 512,
        freq_out: int = 256,
        num_classes: int = settings.num_classes,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_branch = SpatialBranch(
            backbone_name=backbone_name,
            out_features=spatial_out,
            pretrained=pretrained,
        )
        self.frequency_branch = FrequencyBranch(out_features=freq_out)
        self.fusion_classifier = FusionClassifier(
            spatial_dim=spatial_out,
            freq_dim=freq_out,
            num_classes=num_classes,
        )

        # Store config for serialisation
        self.backbone_name = backbone_name
        self.spatial_out = spatial_out
        self.freq_out = freq_out
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor (B, 3, H, W) normalised to [0, 1].

        Returns:
            Raw logits (B, num_classes).
        """
        spatial_feat = self.spatial_branch(x)
        freq_feat = self.frequency_branch(x)
        return self.fusion_classifier(spatial_feat, freq_feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities.

        Returns:
            Float tensor (B, 2) — columns: [P(real), P(ai_generated)]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def get_model_info(self) -> dict:
        """Returns a summary dict — useful for logging and health checks."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "backbone": self.backbone_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "spatial_out_dim": self.spatial_out,
            "freq_out_dim": self.freq_out,
            "num_classes": self.num_classes,
        }


