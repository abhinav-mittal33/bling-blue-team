"""
app/detection/feedback/feedback_router.py

Structured feedback routing — replaces River FTRL online learning.

Two paths:
  route_false_positive → curated_dataset_queue (label=0) + reviewed_novelty_registry
  route_confirmed_fraud → prototype_injection_candidates (status=PENDING_REVIEW)

Both paths write to audit log before committing (log_feedback_routing_event raises
AuditWriteError on failure, which propagates to 500 in feedback.py — feedback cannot
be silently dropped).

Prototype injection_candidates go through developer review before entering the vault.
Investigators cannot inject directly — the endpoint is INTERNAL_KEY only.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Optional

import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.utils.audit_logger import log_feedback_routing_event
from app.core.exceptions import AuditWriteError

log = structlog.get_logger()


def route_false_positive(
    alert_id: str,
    transaction_id: str,
    investigator_id_hash: str,
    feature_vector: dict,
    db: Session,
) -> None:
    """
    Route investigator false-positive verdict:
      1. Insert to curated_dataset_queue with label=0 (for retraining)
      2. Insert fingerprint to reviewed_novelty_registry (label=0, prevents re-flagging)
      3. Write FEEDBACK_ROUTING audit log entry

    Raises AuditWriteError if audit write fails — caller must not silently suppress it.
    """
    fingerprint = _compute_centroid_fingerprint(feature_vector)

    db.execute(
        text("""
            INSERT INTO curated_dataset_queue
              (transaction_id, alert_id, label, label_source, feature_vector,
               investigator_id_hash, batch_exported)
            VALUES
              (:txn_id, :alert_id, 0, 'investigator_false_positive',
               CAST(:fv AS jsonb), :inv_hash, false)
            ON CONFLICT DO NOTHING
        """),
        {
            "txn_id": transaction_id,
            "alert_id": alert_id,
            "fv": json.dumps(_sanitize_fv(feature_vector)),
            "inv_hash": investigator_id_hash,
        },
    )

    # Register as a reviewed-benign pattern so discovery pipeline doesn't re-flag it
    db.execute(
        text("""
            INSERT INTO reviewed_novelty_registry
              (fingerprint, centroid_features, label, source_transaction_id, registered_by)
            VALUES
              (:fingerprint, CAST(:centroid AS jsonb), 0, :txn_id, :inv_hash)
            ON CONFLICT (fingerprint) DO UPDATE SET label = 0
        """),
        {
            "fingerprint": fingerprint,
            "centroid": json.dumps(_sanitize_fv(feature_vector)),
            "txn_id": transaction_id,
            "inv_hash": investigator_id_hash,
        },
    )

    log_feedback_routing_event(
        db=db,
        alert_id=alert_id,
        transaction_id=transaction_id,
        route="FALSE_POSITIVE",
        event_data={
            "label_source": "investigator_false_positive",
            "fingerprint": fingerprint[:16],
            "investigator_id_hash": investigator_id_hash,
        },
    )

    log.info(
        "feedback_routed_false_positive",
        alert_id=alert_id,
        fingerprint=fingerprint[:16],
    )


def route_confirmed_fraud(
    alert_id: str,
    transaction_id: str,
    investigator_id_hash: str,
    feature_vector: dict,
    fraud_type: Optional[str],
    notes: Optional[str],
    db: Session,
) -> None:
    """
    Route investigator confirmed-fraud verdict:
      1. Insert to prototype_injection_candidates (status=PENDING_REVIEW)
         → developer reviews before injecting into vault
      2. Insert to curated_dataset_queue with label=1 (for retraining)
      3. Write FEEDBACK_ROUTING audit log entry

    Raises AuditWriteError if audit write fails.
    """
    db.execute(
        text("""
            INSERT INTO prototype_injection_candidates
              (transaction_id, alert_id, fraud_type, feature_vector,
               investigator_id_hash, investigator_notes, status)
            VALUES
              (:txn_id, :alert_id, :fraud_type, CAST(:fv AS jsonb),
               :inv_hash, :notes, 'PENDING_REVIEW')
            ON CONFLICT DO NOTHING
        """),
        {
            "txn_id": transaction_id,
            "alert_id": alert_id,
            "fraud_type": fraud_type or "unknown",
            "fv": json.dumps(_sanitize_fv(feature_vector)),
            "inv_hash": investigator_id_hash,
            "notes": notes or "",
        },
    )

    db.execute(
        text("""
            INSERT INTO curated_dataset_queue
              (transaction_id, alert_id, label, label_source, feature_vector,
               investigator_id_hash, batch_exported)
            VALUES
              (:txn_id, :alert_id, 1, 'confirmed_fraud',
               CAST(:fv AS jsonb), :inv_hash, false)
            ON CONFLICT DO NOTHING
        """),
        {
            "txn_id": transaction_id,
            "alert_id": alert_id,
            "fv": json.dumps(_sanitize_fv(feature_vector)),
            "inv_hash": investigator_id_hash,
        },
    )

    log_feedback_routing_event(
        db=db,
        alert_id=alert_id,
        transaction_id=transaction_id,
        route="CONFIRMED_FRAUD",
        event_data={
            "fraud_type": fraud_type,
            "label_source": "confirmed_fraud",
            "investigator_id_hash": investigator_id_hash,
            "pending_prototype_review": True,
        },
    )

    log.info(
        "feedback_routed_confirmed_fraud",
        alert_id=alert_id,
        fraud_type=fraud_type,
    )


def compute_centroid_fingerprint(feature_vector: dict) -> str:
    """Public: SHA-256 of top-10 features by magnitude. Used by developer_queue.py."""
    return _compute_centroid_fingerprint(feature_vector)


def _compute_centroid_fingerprint(feature_vector: dict) -> str:
    top = sorted(
        feature_vector.items(),
        key=lambda x: abs(float(x[1] or 0)),
        reverse=True,
    )[:10]
    fp_input = "|".join(f"{k}:{round(float(v or 0), 3)}" for k, v in top)
    return hashlib.sha256(fp_input.encode()).hexdigest()


def _sanitize_fv(fv: dict) -> dict:
    """Replace NaN/Inf with None so JSONB accepts the value."""
    result = {}
    for k, v in fv.items():
        if v is None:
            result[k] = None
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            result[k] = None
        else:
            result[k] = v
    return result
