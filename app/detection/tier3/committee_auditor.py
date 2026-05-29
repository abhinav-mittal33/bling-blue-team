from __future__ import annotations
"""
Audit logger for committee scoring decisions.
CRITICAL: Same contract as audit_logger.py — raises AuditWriteError on failure.
The caller propagates this to a 500 response. No silent skips.
RBI PMLA Section 12 compliance.
"""
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.database import ModelAudit
from app.core.config import settings
from app.core.exceptions import AuditWriteError
from app.detection.tier3.committee_types import CommitteeResult


def log_committee_score(
    db: Session,
    transaction_id: str,
    result: CommitteeResult,
    shadow_mode: bool,
) -> None:
    """
    INSERT to model_audit for every committee scoring decision.
    Raises AuditWriteError if the write fails — caller propagates to 500.
    """
    try:
        event_data = {
            "committee_version": settings.committee_model_version,
            "shadow_mode": shadow_mode,
            "final_score": round(result.final_score, 4),
            "specialist_override": result.specialist_override,
            "meta_score": round(result.meta_score, 4) if result.meta_score is not None else None,
            "scorer_breakdown": result.as_breakdown_dict()["scorers"],
        }
        audit_row = ModelAudit(
            event_type="COMMITTEE_SCORE",
            transaction_id=transaction_id,
            model_version=settings.committee_model_version,
            event_data=event_data,
            event_timestamp=datetime.now(timezone.utc),
        )
        db.add(audit_row)
        db.flush()
    except Exception as exc:
        raise AuditWriteError(
            f"Committee audit write failed for txn {transaction_id}: {exc}"
        ) from exc


def log_meta_learner_train(
    db: Session,
    version: str,
    sample_count: int,
    pr_auc: float,
) -> None:
    """Records meta-learner training event in audit trail. Best-effort — does not raise."""
    try:
        audit_row = ModelAudit(
            event_type="META_TRAIN",
            transaction_id=None,
            model_version=version,
            event_data={
                "sample_count": sample_count,
                "pr_auc": round(pr_auc, 4),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            event_timestamp=datetime.now(timezone.utc),
        )
        db.add(audit_row)
        db.commit()
    except Exception:
        pass


def log_feedback_routing_event(
    db: Session,
    alert_id: str,
    transaction_id: str,
    route: str,
    event_data: dict,
) -> None:
    """
    Records feedback routing decision (FALSE_POSITIVE or CONFIRMED_FRAUD).
    Called from feedback_router.py after FTRL removal.
    Raises AuditWriteError on failure.
    """
    try:
        audit_row = ModelAudit(
            event_type="FEEDBACK_ROUTING",
            transaction_id=transaction_id,
            model_version=settings.committee_model_version,
            event_data={"alert_id": alert_id, "route": route, **event_data},
            event_timestamp=datetime.now(timezone.utc),
        )
        db.add(audit_row)
        db.flush()
    except Exception as exc:
        raise AuditWriteError(
            f"Feedback routing audit write failed for alert {alert_id}: {exc}"
        ) from exc
