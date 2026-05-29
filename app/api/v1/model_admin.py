"""
Model administration API — versioning and rollback (P6-2).
Internal auth only — never expose to Graph Engine or Investigators.

Endpoints:
  GET  /api/v1/internal/model/versions  — list all available model versions
  POST /api/v1/internal/model/activate  — activate a specific version (rollback)
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import require_internal_key

logger = structlog.get_logger()
router = APIRouter(
    prefix="/api/v1/internal/model",
    dependencies=[Depends(require_internal_key)],
)


class ActivateRequest(BaseModel):
    model_name: str


@router.get("/versions")
def list_versions():
    """List all available model versions with integrity status."""
    try:
        from app.utils.model_integrity import list_model_versions
        versions = list_model_versions()
        return {"data": versions, "error": None}
    except Exception as exc:
        logger.error("model_versions_list_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to list model versions")


@router.post("/activate")
def activate_version(req: ActivateRequest):
    """
    Activate a specific model version (rollback P6-2).
    Verifies SHA-256 integrity before activation.
    Resets in-memory cache — next scoring request loads the activated version.
    """
    try:
        from app.utils.model_integrity import activate_model_version
        result = activate_model_version(req.model_name)
        logger.warning("model_rollback_activated", model=req.model_name)
        return {"data": result, "error": None}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("model_activation_failed", model=req.model_name, error=str(exc))
        raise HTTPException(status_code=500, detail="Activation failed")
