"""
api/routes/video.py
-------------------
POST /api/v1/analyze-video

Accepts a multipart video upload, extracts frames via OpenCV, runs
per-frame CNN inference, and returns an aggregated forensic verdict.
"""

from __future__ import annotations

import time
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from core.config import settings
from core.schemas import ErrorResponse, VideoAnalysisResponse
from utils.logger import analysis_logger, get_logger

log = get_logger(__name__)
router = APIRouter()

_ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/webm", "video/x-matroska",
}


# ── Helpers ───────────────────────────────────────────────────────

async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload").suffix or ".mp4"
    tmp_path = settings.upload_tmp_dir / f"vid_{int(time.time() * 1000)}{suffix}"
    async with aiofiles.open(tmp_path, "wb") as fh:
        while chunk := await upload.read(4 * 1024 * 1024):  # 4 MB chunks
            await fh.write(chunk)
    return tmp_path


def _validate_upload(upload: UploadFile) -> None:
    if upload.content_type not in _ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported video type: {upload.content_type}. "
                   f"Allowed: {sorted(_ALLOWED_VIDEO_TYPES)}",
        )


# ── Endpoint ──────────────────────────────────────────────────────

@router.post(
    "/analyze-video",
    response_model=VideoAnalysisResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Unsupported file / validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Detect whether a video is AI-generated or real",
)
async def analyze_video(
    request: Request,
    file: UploadFile = File(..., description="Video file to analyse"),
) -> VideoAnalysisResponse:
    """
    Extracts up to `max_video_frames` evenly-spaced frames from the
    video, runs per-frame forensic CNN inference, and returns an
    aggregated verdict with per-frame score breakdown.
    """
    t_start = time.perf_counter()

    _validate_upload(file)

    svc = request.app.state.inference_service
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service unavailable — model weights not loaded.",
        )

    tmp_path: Path | None = None
    try:
        tmp_path = await _save_upload(file)

        from services.video_service import VideoAnalysisService
        video_svc = VideoAnalysisService(inference_service=svc)
        result: VideoAnalysisResponse = await video_svc.analyse_video(
            video_path=tmp_path,
            filename=file.filename or tmp_path.name,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        log.info(
            "video_analysed",
            filename=file.filename,
            ai_probability=result.ai_probability,
            risk_level=result.risk_level,
            frames=result.frames_analysed,
            elapsed_ms=round(elapsed_ms, 2),
        )
        analysis_logger.record({
            "event": "video_analysis",
            "filename": file.filename,
            "ai_probability": result.ai_probability,
            "risk_level": result.risk_level,
            "frames_analysed": result.frames_analysed,
            "processing_time_ms": round(elapsed_ms, 2),
        })

        return result.model_copy(update={"processing_time_ms": round(elapsed_ms, 2)})

    except HTTPException:
        raise
    except Exception as exc:
        log.error("video_analysis_failed", filename=file.filename, error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {exc}",
        ) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
