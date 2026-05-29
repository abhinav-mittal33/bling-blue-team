"""
app/detection/novelty/discovery_router.py

Routes structurally novel transactions from the PASS stream to the developer
novelty queue. Replaces novelty_router.py with two critical improvements:

1. Deduplication: checks reviewed_novelty_registry before inserting. Patterns
   already reviewed and labelled benign (label=0) are silently skipped.
2. Explicit precondition: exits immediately if fraud_score >= LOG threshold,
   ensuring the discovery pipeline only examines cleared transactions.

All exceptions caught — routing failure must NEVER affect the scoring response.
NEVER creates investigator alerts. NEVER modifies fraud_score.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import text

log = structlog.get_logger()

ESCALATION_THRESHOLD = 10   # occurrences in 7d before Red Team escalation


def route_discovery(
    transaction_id: str,
    account_id: str,
    anomaly_score: float,
    fraud_score: float,
    fraud_action: str,
    gate_fired: Optional[str],
    graph_features: dict,
) -> None:
    """
    Route a novel transaction to the developer review queue.

    Precondition: fraud_score < LOG threshold (action == "PASS").
    Exits silently if this is not satisfied — belt-and-suspenders guard.

    Creates its own SQLAlchemy session so it's safe to call from a thread pool
    after the request session closes.
    """
    from app.core.config import settings

    # Precondition guard — must only process PASS transactions
    if fraud_score >= settings.threshold_log:
        log.debug(
            "discovery_skipped_non_pass",
            txn_id=transaction_id,
            fraud_score=fraud_score,
            action=fraud_action,
        )
        return

    try:
        fingerprint = _compute_fingerprint(graph_features)
        _route(
            transaction_id=transaction_id,
            account_id=account_id,
            anomaly_score=anomaly_score,
            fraud_score=fraud_score,
            fraud_action=fraud_action,
            gate_fired=gate_fired,
            graph_features=graph_features,
            fingerprint=fingerprint,
        )
    except Exception as exc:
        log.error("discovery_routing_failed", txn_id=transaction_id, error=str(exc))


def _compute_fingerprint(graph_features: dict) -> str:
    """SHA-256 of top-10 features by absolute magnitude."""
    top = sorted(
        graph_features.items(),
        key=lambda x: abs(float(x[1] or 0)),
        reverse=True,
    )[:10]
    fp_input = "|".join(f"{k}:{round(float(v or 0), 2)}" for k, v in top)
    return hashlib.sha256(fp_input.encode()).hexdigest()


def _route(
    transaction_id: str,
    account_id: str,
    anomaly_score: float,
    fraud_score: float,
    fraud_action: str,
    gate_fired: Optional[str],
    graph_features: dict,
    fingerprint: str,
) -> None:
    from app.utils.postgres_client import SessionLocal
    from app.utils.redis_client import get_redis

    r = get_redis()
    fp_key = f"novelty:fp:{fingerprint[:16]}"
    current_count = int(r.incr(fp_key))
    r.expire(fp_key, 86400 * 7)
    requires_escalation = current_count >= ESCALATION_THRESHOLD

    with SessionLocal() as db:
        # Dedup check: skip if this fingerprint was already reviewed as benign
        existing = db.execute(
            text("""
                SELECT label FROM reviewed_novelty_registry
                WHERE fingerprint = :fingerprint
                LIMIT 1
            """),
            {"fingerprint": fingerprint},
        ).fetchone()

        if existing is not None and existing.label == 0:
            log.debug("discovery_dedup_benign_skip", fingerprint=fingerprint[:16])
            return

        db.execute(
            text("""
                INSERT INTO novelty_queue (
                    transaction_id, account_id, anomaly_score, fraud_score,
                    fraud_action, gate_fired, novelty_fingerprint,
                    fingerprint_occurrences, graph_features_snapshot,
                    requires_escalation, status, created_at
                ) VALUES (
                    :txn_id, :acct_id, :anomaly_score, :fraud_score,
                    :fraud_action, :gate_fired, :fingerprint,
                    :fp_count, CAST(:features_json AS jsonb),
                    :escalate, 'PENDING_REVIEW', :created_at
                )
                ON CONFLICT (transaction_id) DO NOTHING
            """),
            {
                "txn_id": transaction_id,
                "acct_id": account_id,
                "anomaly_score": anomaly_score,
                "fraud_score": fraud_score,
                "fraud_action": fraud_action,
                "gate_fired": gate_fired,
                "fingerprint": fingerprint,
                "fp_count": current_count,
                "features_json": json.dumps(_sanitize(graph_features)),
                "escalate": requires_escalation,
                "created_at": datetime.now(timezone.utc),
            },
        )
        db.commit()

    try:
        from app.utils.metrics import novelty_flags_total
        novelty_flags_total.inc()
    except Exception:
        pass

    log.info(
        "discovery_flagged",
        txn_id=transaction_id,
        anomaly_score=round(anomaly_score, 4),
        fraud_score=round(fraud_score, 4),
        fingerprint=fingerprint[:16],
        occurrences=current_count,
        escalation=requires_escalation,
    )

    if requires_escalation:
        _escalate_to_red_team(transaction_id, anomaly_score, fraud_score, fingerprint, current_count, graph_features)


def _escalate_to_red_team(
    transaction_id: str,
    anomaly_score: float,
    fraud_score: float,
    fingerprint: str,
    fp_count: int,
    graph_features: dict,
) -> None:
    try:
        from app.integrations.red_team_client import notify_novelty_pattern

        novelty_dna = {
            "pattern_type": "structural_novelty",
            "novelty_fingerprint": fingerprint[:16],
            "occurrences_in_7d": fp_count,
            "sample_transaction_id": transaction_id,
            "anomaly_score": round(anomaly_score, 4),
            "fraud_score_at_detection": round(fraud_score, 4),
            "evaded_detection": True,   # Always True — this only fires on PASS transactions
            "source": "discovery_ensemble",
            "requires_developer_review": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        notify_novelty_pattern(novelty_dna)

        try:
            from app.utils.metrics import novelty_escalations_total
            novelty_escalations_total.inc()
        except Exception:
            pass

        log.info("discovery_escalated_to_red_team", fingerprint=fingerprint[:16], occurrences=fp_count)
    except Exception as exc:
        log.error("discovery_red_team_escalation_failed", error=str(exc))


def _sanitize(features: dict) -> dict:
    result = {}
    for k, v in features.items():
        if v is None:
            result[k] = None
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            result[k] = None
        else:
            result[k] = v
    return result
