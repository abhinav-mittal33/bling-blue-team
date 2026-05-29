
"""
POST /api/v1/feedback
Investigator submits verdict → structured feedback routing + blockchain seal + red team notification.
Auth: INVESTIGATOR_API_KEY only.

River FTRL online learning removed (Phase 3). Feedback now routes to:
  - confirmed_fraud  → prototype_injection_candidates + curated_dataset_queue (label=1)
  - false_positive   → curated_dataset_queue (label=0) + reviewed_novelty_registry
"""
import json
import math

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.security import require_investigator_key, pseudonymize
from app.core.exceptions import AuditWriteError
from app.models.schemas import InvestigatorFeedbackRequest, FeedbackResponse
from app.models.database import Alert, FraudScore, FeedbackLog
from app.utils.audit_logger import log_feedback_event
from app.utils.metrics import feedback_received_total

logger = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_investigator_key)])


@router.post("/api/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    payload: InvestigatorFeedbackRequest,
    db: Session = Depends(get_db),
):
    alert = db.query(Alert).filter(Alert.id == payload.alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    score_row = db.query(FraudScore).filter(
        FraudScore.transaction_id == alert.transaction_id
    ).first()

    investigator_pseudo = pseudonymize(payload.investigator_id or "unknown")

    # 1. Structured feedback routing (replaces FTRL — Phase 3)
    model_updated = False
    if score_row:
        try:
            from app.detection.feedback.feedback_router import (
                route_false_positive, route_confirmed_fraud
            )
            fv = {}
            if score_row.feature_vector:
                fv = (
                    json.loads(score_row.feature_vector)
                    if isinstance(score_row.feature_vector, str)
                    else score_row.feature_vector
                ) or {}

            if payload.confirmed_fraud:
                route_confirmed_fraud(
                    alert_id=payload.alert_id,
                    transaction_id=alert.transaction_id,
                    investigator_id_hash=investigator_pseudo,
                    feature_vector=fv,
                    fraud_type=payload.fraud_type,
                    notes=payload.notes,
                    db=db,
                )
            else:
                route_false_positive(
                    alert_id=payload.alert_id,
                    transaction_id=alert.transaction_id,
                    investigator_id_hash=investigator_pseudo,
                    feature_vector=fv,
                    db=db,
                )
            model_updated = True   # feedback routed successfully (replaces FTRL update flag)
        except AuditWriteError:
            raise   # audit failures propagate → 500
        except Exception as exc:
            logger.error("feedback_routing_failed", alert_id=payload.alert_id, error=str(exc))

    # 2. Blockchain seal (async, best-effort)
    blockchain_sealed = False
    try:
        from app.integrations.blockchain_client import seal_evidence
        blockchain_sealed = seal_evidence(
            alert_id=payload.alert_id,
            transaction_id=alert.transaction_id,
            confirmed_fraud=payload.confirmed_fraud,
        )
    except Exception as exc:
        logger.warning("Blockchain seal failed", alert_id=payload.alert_id, error=str(exc))

    # 3. Red team notification (async, best-effort)
    red_team_notified = False
    if payload.confirmed_fraud:
        try:
            from app.integrations.red_team_client import notify_confirmed_fraud
            red_team_notified = notify_confirmed_fraud(
                alert_id=payload.alert_id,
                fraud_type=payload.fraud_type,
                transaction_id=alert.transaction_id,
            )
        except Exception as exc:
            logger.warning("Red team notification failed", alert_id=payload.alert_id, error=str(exc))

    # 4. Update alert status
    alert.status = "CONFIRMED_FRAUD" if payload.confirmed_fraud else "FALSE_POSITIVE"
    db.add(FeedbackLog(
        alert_id=payload.alert_id,
        transaction_id=alert.transaction_id,
        label=1 if payload.confirmed_fraud else 0,
    ))

    # 5. Audit — must succeed before returning
    log_feedback_event(db, payload.alert_id, alert.transaction_id, {
        "confirmed_fraud": payload.confirmed_fraud,
        "fraud_type": payload.fraud_type,
        "model_updated": model_updated,
        "blockchain_sealed": blockchain_sealed,
        "red_team_notified": red_team_notified,
    })

    db.commit()

    feedback_received_total.labels(
        confirmed=str(payload.confirmed_fraud)
    ).inc()

    logger.info("Feedback processed",
                alert_id=payload.alert_id,
                confirmed=payload.confirmed_fraud,
                model_updated=model_updated,
                blockchain_sealed=blockchain_sealed,
                red_team=red_team_notified)

    return FeedbackResponse(
        alert_id=payload.alert_id,
        model_updated=model_updated,
        blockchain_sealed=blockchain_sealed,
        red_team_notified=red_team_notified,
    )
