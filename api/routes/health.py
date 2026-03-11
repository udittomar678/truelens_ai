from fastapi import APIRouter, Request
from pydantic import BaseModel
from core.config import settings

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    env: str
    device: str
    inference_service_ready: bool

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        env=settings.env,
        device=settings.device,
        inference_service_ready=request.app.state.inference_service is not None,
    )

@router.get("/ready")
async def readiness_check(request: Request):
    from fastapi import status
    from fastapi.responses import JSONResponse
    if request.app.state.inference_service is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}
