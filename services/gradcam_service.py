"""
services/gradcam_service.py
---------------------------
TrueLens AI — Grad-CAM Explainability Service

Generates visual heatmaps showing which regions
of an image triggered the AI detection signal.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class GradCAMService:
    """
    Grad-CAM implementation for TrueLens EfficientNet-B3 backbone.

    Hooks into the last convolutional block of EfficientNet-B3
    and computes gradient-weighted class activation maps.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model   = model
        self.device  = torch.device(settings.device)
        self._hooks  = []
        self._grads  = None
        self._acts   = None

        # Target: last conv block of EfficientNet-B3
        self._target_layer = self._find_target_layer()
        log.info("gradcam_service_ready", target_layer=str(self._target_layer.__class__.__name__))

    # ── Layer discovery ──────────────────────────────────────────

    def _find_target_layer(self) -> torch.nn.Module:
        """Find the last convolutional block in EfficientNet backbone."""
        backbone = self.model.spatial_branch.backbone
        # EfficientNet stores blocks as a flat Sequential
        # We want the last MBConv block
        blocks = list(backbone.blocks.children())
        return blocks[-1]

    # ── Hook management ──────────────────────────────────────────

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self._acts = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._grads = grad_output[0].detach()

        self._hooks.append(
            self._target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self._target_layer.register_full_backward_hook(backward_hook)
        )

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ── Core Grad-CAM ────────────────────────────────────────────

    def generate(
        self,
        image: Image.Image,
        class_idx: int = 1,          # 1 = FAKE class
    ) -> dict:
        """
        Generate Grad-CAM heatmap for a given PIL image.

        Returns dict with:
          - heatmap_base64   : raw heatmap as base64 PNG
          - overlay_base64   : heatmap overlaid on original image
          - top_score        : max activation score 0-1
          - explanation      : human readable explanation
        """
        try:
            self._register_hooks()
            self.model.eval()

            # Preprocess
            tensor = self._preprocess(image).to(self.device)
            tensor.requires_grad_(True)

            # Forward pass
            logits = self.model(tensor)
            score  = F.softmax(logits, dim=1)[0, class_idx]

            # Backward pass
            self.model.zero_grad()
            score.backward()

            # Compute Grad-CAM
            grads = self._grads          # (1, C, H, W)
            acts  = self._acts           # (1, C, H, W)

            # Global average pool gradients
            weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam     = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam     = F.relu(cam)

            # Normalize to 0-1
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)

            # Resize to original image size
            orig_w, orig_h = image.size
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (orig_w, orig_h), Image.BILINEAR
                )
            ) / 255.0

            # Generate outputs
            heatmap_img  = self._cam_to_heatmap(cam_resized)
            overlay_img  = self._overlay_heatmap(image, cam_resized)
            top_score    = float(cam_resized.max())
            explanation  = self._generate_explanation(
                cam_resized, float(score.item())
            )

            return {
                "heatmap_base64": self._img_to_base64(heatmap_img),
                "overlay_base64": self._img_to_base64(overlay_img),
                "top_score":      round(top_score, 3),
                "ai_score":       round(float(score.item()), 3),
                "explanation":    explanation,
                "success":        True,
            }

        except Exception as exc:
            log.warning("gradcam_failed", error=str(exc))
            return {
                "heatmap_base64": None,
                "overlay_base64": None,
                "top_score":      0.0,
                "ai_score":       0.0,
                "explanation":    "Heatmap unavailable",
                "success":        False,
            }

        finally:
            self._remove_hooks()
            self._grads = None
            self._acts  = None

    # ── Image processing ─────────────────────────────────────────

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return transform(image.convert("RGB")).unsqueeze(0)

    def _cam_to_heatmap(self, cam: np.ndarray) -> Image.Image:
        """Convert grayscale CAM to colourmap heatmap (jet colormap)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colormap  = cm.get_cmap("jet")
        heatmap   = colormap(cam)                     # RGBA
        heatmap   = (heatmap[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(heatmap)

    def _overlay_heatmap(
        self,
        original: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """Blend original image with heatmap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        orig_rgb  = original.convert("RGB").resize(
            (original.width, original.height)
        )
        colormap  = cm.get_cmap("jet")
        heatmap   = colormap(cam)[:, :, :3]
        heatmap   = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap).resize(
            orig_rgb.size, Image.BILINEAR
        )

        orig_arr  = np.array(orig_rgb).astype(float)
        heat_arr  = np.array(heatmap_img).astype(float)
        blended   = (orig_arr * (1 - alpha) + heat_arr * alpha).astype(np.uint8)
        return Image.fromarray(blended)

    def _img_to_base64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")

    def _generate_explanation(
        self,
        cam: np.ndarray,
        ai_score: float,
    ) -> str:
        """Generate human readable explanation based on heatmap."""
        high_activation = (cam > 0.7).sum() / cam.size
        med_activation  = (cam > 0.4).sum() / cam.size

        if ai_score > 0.80:
            intensity = "Strong"
        elif ai_score > 0.55:
            intensity = "Moderate"
        else:
            intensity = "Weak"

        if high_activation > 0.3:
            spread = "across large areas of the image"
        elif high_activation > 0.1:
            spread = "in several key regions"
        else:
            spread = "in isolated regions"

        return (
            f"{intensity} AI indicators detected {spread}. "
            f"Red zones highlight where the model found "
            f"synthetic patterns most strongly."
        )




