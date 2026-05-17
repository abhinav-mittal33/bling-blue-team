from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.utils.redis_client import get_redis
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "version": settings.model_version}


@router.get("/ready")
async def ready(db: Session = Depends(get_db)):
    """Returns 200 only when PostgreSQL + Redis are reachable."""
    errors = []

    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        errors.append(f"postgres: {exc}")

    try:
        get_redis().ping()
    except Exception as exc:
        errors.append(f"redis: {exc}")

    if errors:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail={"errors": errors})

    return {"status": "ready"}
