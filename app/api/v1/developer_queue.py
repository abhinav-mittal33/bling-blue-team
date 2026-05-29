"""
GET/POST /api/v1/developer-queue/prototype-candidates
Auth: INTERNAL_KEY ONLY. Returns 403 for INVESTIGATOR_KEY or GRAPH_ENGINE_KEY.

Routes:
  GET  /api/v1/developer-queue/prototype-candidates
       → list PENDING_REVIEW candidates (paginated)
  POST /api/v1/developer-queue/prototype-candidates/{id}/inject
       → call prototype_vault.inject_prototype(), mark INJECTED, audit
  POST /api/v1/developer-queue/prototype-candidates/{id}/reject
       → mark REJECTED, save developer_notes, audit

Prototype injection is one-way: once INJECTED, cannot be removed via this API.
Use prototype_vault.inject_prototype() removal is intentionally not exposed.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.core.config import settings
from app.core.exceptions import AuditWriteError

logger = structlog.get_logger()
router = APIRouter()


def _require_internal_key(request: Request) -> None:
    """Strict INTERNAL_KEY check — 403 for any other key including INVESTIGATOR."""
    api_key = request.headers.get("x-api-key", "")
    if not settings.internal_api_key or api_key != settings.internal_api_key:
        raise HTTPException(status_code=403, detail="Internal developer endpoint — access denied")


class InjectRequest(BaseModel):
    developer_notes: str = Field(default="", max_length=2000)


class RejectRequest(BaseModel):
    developer_notes: str = Field(..., max_length=2000)


@router.get("/api/v1/developer-queue/prototype-candidates")
async def list_prototype_candidates(
    request: Request,
    db: Session = Depends(get_db),
    status: str = "PENDING_REVIEW",
    page: int = 1,
    per_page: int = 20,
):
    _require_internal_key(request)

    if per_page > 100:
        per_page = 100

    offset = (page - 1) * per_page

    rows = db.execute(
        text("""
            SELECT id, transaction_id, alert_id, fraud_type,
                   investigator_notes, status, submitted_at, reviewed_at
            FROM prototype_injection_candidates
            WHERE status = :status
            ORDER BY submitted_at ASC
            LIMIT :limit OFFSET :offset
        """),
        {"status": status, "limit": per_page, "offset": offset},
    ).fetchall()

    total = db.execute(
        text("SELECT COUNT(*) FROM prototype_injection_candidates WHERE status = :status"),
        {"status": status},
    ).scalar() or 0

    return {
        "data": [dict(r._mapping) for r in rows],
        "error": None,
        "meta": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "has_more": (offset + per_page) < total,
        },
    }


@router.post("/api/v1/developer-queue/prototype-candidates/{candidate_id}/inject")
async def inject_prototype_candidate(
    candidate_id: int,
    body: InjectRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    _require_internal_key(request)

    row = db.execute(
        text("""
            SELECT id, transaction_id, fraud_type, feature_vector, status
            FROM prototype_injection_candidates
            WHERE id = :id
        """),
        {"id": candidate_id},
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if row.status != "PENDING_REVIEW":
        raise HTTPException(
            status_code=422,
            detail=f"Candidate is already '{row.status}' — can only inject PENDING_REVIEW candidates",
        )

    # Inject into prototype vault
    import numpy as np
    from app.detection.tier3.prototype_vault import prototype_vault
    from ml.feature_registry import FEATURE_NAMES

    if not prototype_vault.loaded:
        raise HTTPException(
            status_code=503,
            detail="Prototype vault not loaded. Run: python ml/scripts/build_initial_prototypes.py",
        )

    fv_dict = row.feature_vector or {}
    vec = np.array(
        [float(fv_dict.get(k, float("nan"))) for k in FEATURE_NAMES],
        dtype=np.float32,
    )

    success = prototype_vault.inject_prototype(
        feature_vector=vec,
        label=1,
        fraud_type=row.fraud_type or "unknown",
        source_transaction_id=row.transaction_id or "",
    )

    if not success:
        raise HTTPException(status_code=422, detail="Injection failed — vector may be invalid or degenerate")

    # Mark as INJECTED
    db.execute(
        text("""
            UPDATE prototype_injection_candidates
            SET status = 'INJECTED', developer_notes = :notes, reviewed_at = now()
            WHERE id = :id
        """),
        {"notes": body.developer_notes, "id": candidate_id},
    )

    # Audit (INSERT-only)
    _write_inject_audit(db, row.transaction_id or "", candidate_id, row.fraud_type)

    db.commit()
    logger.info("prototype_injected", candidate_id=candidate_id, fraud_type=row.fraud_type)
    return {"data": {"injected": True, "vault_size": prototype_vault.size}, "error": None, "meta": {}}


@router.post("/api/v1/developer-queue/prototype-candidates/{candidate_id}/reject")
async def reject_prototype_candidate(
    candidate_id: int,
    body: RejectRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    _require_internal_key(request)

    row = db.execute(
        text("SELECT id, transaction_id, status FROM prototype_injection_candidates WHERE id = :id"),
        {"id": candidate_id},
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if row.status != "PENDING_REVIEW":
        raise HTTPException(
            status_code=422,
            detail=f"Candidate is already '{row.status}'",
        )

    db.execute(
        text("""
            UPDATE prototype_injection_candidates
            SET status = 'REJECTED', developer_notes = :notes, reviewed_at = now()
            WHERE id = :id
        """),
        {"notes": body.developer_notes, "id": candidate_id},
    )

    _write_inject_audit(db, row.transaction_id or "", candidate_id, "rejected", action="REJECT")
    db.commit()
    logger.info("prototype_rejected", candidate_id=candidate_id)
    return {"data": {"rejected": True}, "error": None, "meta": {}}


def _write_inject_audit(
    db: Session,
    transaction_id: str,
    candidate_id: int,
    fraud_type: str | None,
    action: str = "INJECT",
) -> None:
    """Best-effort audit write — never raises (prototype inject is dev-only)."""
    try:
        from app.utils.audit_logger import log_feedback_routing_event
        log_feedback_routing_event(
            db=db,
            alert_id="developer_queue",
            transaction_id=transaction_id,
            route=f"PROTOTYPE_{action}",
            event_data={"candidate_id": candidate_id, "fraud_type": fraud_type},
        )
    except AuditWriteError as exc:
        logger.warning("prototype_audit_write_failed", candidate_id=candidate_id, error=str(exc))
