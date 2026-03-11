"""
main.py
-------
TrueLens AI — FastAPI application entry point.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.schemas import ErrorResponse
from utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    log.info(
        "truelens_starting",
        app=settings.app_name,
        version=settings.app_version,
        env=settings.env,
        device=settings.device,
    )
    try:
        from services.inference_service import InferenceService
        app.state.inference_service = InferenceService()
        app.state.inference_service.warmup()
        log.info("inference_service_ready")
    except Exception as exc:
        log.warning("inference_service_warmup_skipped", reason=str(exc))
        app.state.inference_service = None

    yield

    log.info("truelens_shutting_down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Forensic AI platform for detecting AI-generated images and videos.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=str(request.url), error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(detail="An unexpected error occurred.", code="INTERNAL_SERVER_ERROR").model_dump(),
        )

    from api.routes.image import router as image_router
    from api.routes.video import router as video_router
    from api.routes.health import router as health_router

    app.include_router(health_router, tags=["Health"])
    app.include_router(image_router, prefix="/api/v1", tags=["Image Analysis"])
    app.include_router(video_router, prefix="/api/v1", tags=["Video Analysis"])

    # ── Serve Frontend ────────────────────────────────────────
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)


