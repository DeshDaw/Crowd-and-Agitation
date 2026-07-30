"""
FastAPI router for health checks.
"""

import sys
from pathlib import Path

from fastapi import APIRouter

from ..models import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

# Check CUDA availability
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE_AVAILABLE = "cuda" if CUDA_AVAILABLE else "cpu"
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE_AVAILABLE = "cpu"

# Add crowd_project to check Detectron2
CROWD_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent / "crowd_project"
if str(CROWD_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CROWD_PROJECT_ROOT))

DETECTRON2_AVAILABLE = False
try:
    import detectron2

    DETECTRON2_AVAILABLE = True
except ImportError:
    pass

ULTRALYTICS_AVAILABLE = False
try:
    import ultralytics  # noqa: F401

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    backends = []
    if DETECTRON2_AVAILABLE:
        backends.append("detectron2")
    if ULTRALYTICS_AVAILABLE:
        backends.append("yolo")
    return HealthResponse(
        status="healthy",
        device_available=DEVICE_AVAILABLE,
        cuda_available=CUDA_AVAILABLE,
        backends_available=backends,
    )


@router.get("/detailed")
async def detailed_health() -> dict:
    """Detailed health check with component status."""
    return {
        "status": "healthy",
        "components": {
            "api": "ok",
            "torch": DEVICE_AVAILABLE,
            "cuda_available": CUDA_AVAILABLE,
            "detectron2": "ok" if DETECTRON2_AVAILABLE else "not_available",
            "yolo": "ok" if ULTRALYTICS_AVAILABLE else "not_available",
        },
        "version": "1.0.0",
    }
