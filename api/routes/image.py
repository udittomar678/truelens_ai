"""
api/routes/image.py  (updated with Grad-CAM)
--------------------------------------------
TrueLens AI — Image Analysis Routes
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from core.config import settings
from core.schemas import ImageAnalysisResponse
from services.forensic_fusion_service import ForensicFusionService
from utils.logger import get_logger

log    = get_logger(__name__)
router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def get_fusion_service(request: Request) -> ForensicFusionService:
    inf = getattr(request.app.state, "inference_service", None)
    if inf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not ready.",
        )
    return ForensicFusionService(inference_service=inf)


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    fusion: ForensicFusionService = Depends(get_fusion_service),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    tmp_dir  = Path(settings.upload_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}_{file.filename}"

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        t0       = time.perf_counter()
        result   = await fusion.analyse_image(image_path=tmp_path, filename=file.filename)
        elapsed  = (time.perf_counter() - t0) * 1000
        result.processing_time_ms = elapsed

        log.info(
            "image_analyzed",
            filename=file.filename,
            ai_probability=result.ai_probability,
            risk_level=result.risk_level,
            processing_time_ms=round(elapsed, 2),
        )
        return result

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/analyze-image/gradcam")
async def analyze_image_gradcam(
    request: Request,
    file: UploadFile = File(...),
    fusion: ForensicFusionService = Depends(get_fusion_service),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    tmp_dir  = Path(settings.upload_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}_{file.filename}"

    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        t0     = time.perf_counter()
        result = await fusion.analyse_image(image_path=tmp_path, filename=file.filename)

        gradcam_data = {"success": False, "explanation": "Grad-CAM unavailable"}
        try:
            from PIL import Image as PILImage
            from services.gradcam_service import GradCAMService

            inf_service = request.app.state.inference_service
            if inf_service and inf_service._model is not None:
                gcam   = GradCAMService(model=inf_service._model)
                image  = PILImage.open(tmp_path).convert("RGB")
                gradcam_data = gcam.generate(image=image)
        except Exception as exc:
            log.warning("gradcam_skipped", reason=str(exc))

        elapsed = (time.perf_counter() - t0) * 1000

        response = result.model_dump()
        response["processing_time_ms"] = round(elapsed, 2)
        response["gradcam"] = gradcam_data

        log.info(
            "gradcam_analyzed",
            filename=file.filename,
            ai_probability=result.ai_probability,
            gradcam_success=gradcam_data.get("success", False),
        )
        return JSONResponse(content=response)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()