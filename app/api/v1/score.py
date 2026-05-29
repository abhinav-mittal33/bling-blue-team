
"""
POST /api/v1/score
Called by Graph Engine teammate after each transaction settles.
Rate limit: 1000/minute.
Auth: GRAPH_ENGINE_API_KEY only.
"""
import asyncio
import functools
import time
import uuid
import structlog
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db
from app.core.security import require_graph_engine_key, pseudonymize, limiter
from app.core.exceptions import AuditWriteError
from app.models.schemas import TransactionScoreRequest, ScoreResponse
from app.models.database import FraudScore, Alert
from app.detection.pipeline import run_pipeline
from app.utils.audit_logger import log_score_event
from app.utils.redis_client import increment_velocity
from app.utils.sanitize import sanitize_for_log
from app.utils.metrics import scoring_requests_total, scoring_latency_ms, alerts_created_total

# Constant-time response target (P1-4 — timing oracle prevention).
# All /score responses padded to this value so Tier 2 path (20ms) and
# Tier 3 path (47ms) are indistinguishable to external observers.
_TARGET_RESPONSE_MS = 55

logger = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_graph_engine_key)])


@router.post("/api/v1/score", response_model=ScoreResponse)
@limiter.limit("1000/minute")
async def score_transaction(
    request: Request,
    txn: TransactionScoreRequest,
    db: Session = Depends(get_db),
):
    # Timing-oracle prevention (P1-4): start wall clock before any logic.
    # We pad the response to _TARGET_RESPONSE_MS at the end.
    _wall_start = time.monotonic()

    # Log injection sanitization (P1-5): sanitize attacker-controlled fields
    # BEFORE they reach any structlog call.
    _safe_payee = sanitize_for_log(txn.payee_vpa or "")
    _safe_txn_id = sanitize_for_log(txn.transaction_id)

    acct_pseudo = pseudonymize(txn.account_id)

    # Ensure transaction exists in DB (Graph Engine calls us after it settles the txn)
    _upsert_transaction(txn, db)

    # Fetch account context from PostgreSQL (needed by pipeline)
    account_ctx = _fetch_account_context(txn.account_id, txn.payee_account_id, db)

    # Run detection pipeline
    result = run_pipeline(
        txn=txn,
        db=db,
        account_age_days=account_ctx["account_age_days"],
        avg_amount_30d=account_ctx["avg_amount_30d"],
        payee_in_known_contacts=account_ctx["payee_in_known_contacts"],
        payee_account_age_days=account_ctx["payee_account_age_days"],
        payee_vpa_age_days=_compute_vpa_age(txn),
        kyc_occupation=account_ctx["kyc_occupation"],
        kyc_age=account_ctx["kyc_age"],
        account_type=account_ctx["account_type"],
        daily_txn_count=account_ctx["daily_txn_count"],
        has_festival_gifting_history=account_ctx["has_festival_gifting_history"],
    )

    action = result["action"]
    score = result["score"]
    gate_fired = result.get("gate_fired")

    # Persist fraud score row
    fraud_score_row = FraudScore(
        transaction_id=txn.transaction_id,
        score=score,
        gate_fired=gate_fired,
        action=action,
        tier1_flags=result.get("tier1_flags"),
        tier2_gate=result.get("tier2_gate"),
        tier3_score=result.get("tier3_score_raw"),
        feature_vector=_sanitize_for_json(result.get("feature_vector")),
        shap_values=_sanitize_for_json(result.get("shap_explanation")),
        indian_context_applied=result.get("indian_context_applied"),
        model_version=_model_version(),
        processing_ms=result["processing_ms"],
    )
    db.add(fraud_score_row)

    # Create alert if action warrants it
    alert_id = None
    if action in ("REVIEW", "HIGH_RISK"):
        alert_id = str(uuid.uuid4())
        alert_row = Alert(
            id=alert_id,
            transaction_id=txn.transaction_id,
            score=score,
            gate=gate_fired,
            action=action,
            status="OPEN",
            trail_status="PENDING",
        )
        db.add(alert_row)
        alerts_created_total.labels(action=action).inc()

        # Queue async fund trail reconstruction
        try:
            from app.evidence.trail_builder import reconstruct_fund_trail
            reconstruct_fund_trail.delay(txn.transaction_id, alert_id)
        except Exception as exc:
            logger.error("Failed to queue trail reconstruction", alert_id=alert_id, error=str(exc))

    # Audit INSERT — must succeed before returning. Fail request if it fails.
    log_score_event(db, txn.transaction_id, {
        "score": score,
        "action": action,
        "gate_fired": gate_fired,
        "alert_id": alert_id,
        "account_pseudo": acct_pseudo,
        "processing_ms": result["processing_ms"],
    })

    db.commit()
    increment_velocity(txn.account_id, float(txn.amount))

    # Dispatch async SHAP computation (P1-6) — runs off the hot path
    # Only for Tier 3 transactions that produced a real ML score
    if result.get("tier3_score_raw") is not None and fraud_score_row.id:
        try:
            from app.detection.tier3.shap_explainer import compute_shap
            feature_vec = result.get("feature_vector") or {}
            compute_shap.delay(fraud_score_row.id, feature_vec)
        except Exception as exc:
            logger.warning("shap_task_dispatch_failed", error=str(exc))

    scoring_requests_total.labels(action=action, gate_fired=gate_fired or "none").inc()
    scoring_latency_ms.observe(result["processing_ms"])

    # ── DISCOVERY PIPELINE (PASS stream only — silent background sensor) ─────
    # Runs ONLY when action == "PASS". NEVER modifies score or action.
    # Phase 3: replaced IsoForest-only with discovery_ensemble (IsoForest + ECOD).
    # Uses run_in_executor so DB I/O doesn't block the event loop.
    if action == "PASS":
        try:
            from app.detection.novelty.discovery_ensemble import discovery_ensemble
            from app.detection.novelty.discovery_router import route_discovery

            if discovery_ensemble.available:
                feature_vector = result.get("feature_vector") or {}
                anomaly_score = discovery_ensemble.score(feature_vector)
                if discovery_ensemble.is_novel(anomaly_score):
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(
                        None,
                        functools.partial(
                            route_discovery,
                            transaction_id=txn.transaction_id,
                            account_id=txn.account_id,
                            anomaly_score=anomaly_score,
                            fraud_score=score,
                            fraud_action=action,
                            gate_fired=gate_fired,
                            graph_features=feature_vector,
                        ),
                    )
        except Exception as _disc_exc:
            # Discovery errors must NEVER affect the scoring response
            logger.error("discovery_pipeline_error", error=str(_disc_exc))
    # ── END DISCOVERY PIPELINE ────────────────────────────────────────────────

    logger.info("Scored transaction",
                txn_id=_safe_txn_id,
                account=acct_pseudo,
                payee=_safe_payee,
                score=score,
                action=action,
                gate=gate_fired,
                ms=result["processing_ms"])

    # Timing-oracle prevention (P1-4): pad response to constant _TARGET_RESPONSE_MS.
    # Applied AFTER all processing so it doesn't affect the pipeline latency metrics.
    _elapsed_ms = (time.monotonic() - _wall_start) * 1000
    if _elapsed_ms < _TARGET_RESPONSE_MS:
        await asyncio.sleep((_TARGET_RESPONSE_MS - _elapsed_ms) / 1000)

    return ScoreResponse(
        transaction_id=txn.transaction_id,
        score=score,
        action=action,
        gate_fired=gate_fired,
        alert_id=alert_id,
        processing_ms=result["processing_ms"],
    )


def _fetch_account_context(account_id: str, payee_account_id: str | None, db: Session) -> dict:
    """Fetch account fields needed by pipeline. Returns safe defaults if account not found."""
    row = db.execute(
        text("""
            SELECT account_age_days, kyc_occupation, kyc_age, account_type,
                   kyc_completeness_score
            FROM accounts WHERE id = :id
        """),
        {"id": account_id},
    ).fetchone()

    avg_amount = db.execute(
        text("""
            SELECT COALESCE(AVG(amount), 0) FROM transactions
            WHERE account_id = :id AND timestamp > NOW() - INTERVAL '30 days'
        """),
        {"id": account_id},
    ).scalar()

    daily_count = db.execute(
        text("""
            SELECT COUNT(*) FROM transactions
            WHERE account_id = :id AND timestamp > NOW() - INTERVAL '24 hours'
        """),
        {"id": account_id},
    ).scalar()

    payee_age = None
    if payee_account_id:
        payee_row = db.execute(
            text("SELECT account_age_days FROM accounts WHERE id = :id"),
            {"id": payee_account_id},
        ).fetchone()
        if payee_row:
            payee_age = payee_row.account_age_days

    # Festival gifting history: had >3 small transactions in Oct/Nov in prior year
    festival_history = db.execute(
        text("""
            SELECT COUNT(*) FROM transactions
            WHERE account_id = :id
              AND amount < 5000
              AND EXTRACT(MONTH FROM timestamp) IN (10, 11)
              AND timestamp < NOW() - INTERVAL '1 year'
        """),
        {"id": account_id},
    ).scalar()

    # Known contact: any prior outbound txn from this account to same payee (indexed query)
    known_contact = False
    if payee_account_id:
        known_contact = db.execute(
            text("""
                SELECT 1 FROM transactions
                WHERE account_id = :acct AND payee_account_id = :payee
                LIMIT 1
            """),
            {"acct": account_id, "payee": payee_account_id},
        ).fetchone() is not None

    if row:
        return {
            "account_age_days": row.account_age_days or 0,
            "kyc_occupation": row.kyc_occupation,
            "kyc_age": row.kyc_age,
            "account_type": row.account_type,
            "avg_amount_30d": float(avg_amount or 0),
            "payee_in_known_contacts": known_contact,
            "payee_account_age_days": payee_age,
            "daily_txn_count": int(daily_count or 0),
            "has_festival_gifting_history": (festival_history or 0) > 3,
        }

    return {
        "account_age_days": 0,
        "kyc_occupation": None,
        "kyc_age": None,
        "account_type": "SAVINGS",
        "avg_amount_30d": 0.0,
        "payee_in_known_contacts": known_contact,
        "payee_account_age_days": payee_age,
        "daily_txn_count": 0,
        "has_festival_gifting_history": False,
    }


def _compute_vpa_age(txn: TransactionScoreRequest) -> int | None:
    if not txn.payee_vpa_created_at:
        return None
    ts = txn.timestamp
    vpa_ts = txn.payee_vpa_created_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if vpa_ts.tzinfo is None:
        vpa_ts = vpa_ts.replace(tzinfo=timezone.utc)
    return max(0, (ts - vpa_ts).days)


def _upsert_transaction(txn: TransactionScoreRequest, db: Session) -> None:
    """Insert transaction if it doesn't exist. Idempotent — Graph Engine may retry."""
    from app.models.database import Transaction, Account
    from datetime import timezone

    # Ensure account row exists (safe default — real data comes from core banking)
    now = datetime.now(timezone.utc)
    for aid in filter(None, [txn.account_id, getattr(txn, "payee_account_id", None)]):
        exists = db.execute(text("SELECT 1 FROM accounts WHERE id = :id"), {"id": aid}).scalar()
        if not exists:
            db.execute(
                text("INSERT INTO accounts (id, account_type, kyc_completeness_score, account_age_days, created_at, updated_at) VALUES (:id, 'SAVINGS', 0.5, 0, :now, :now) ON CONFLICT (id) DO NOTHING"),
                {"id": aid, "now": now},
            )

    exists = db.execute(text("SELECT 1 FROM transactions WHERE id = :id"), {"id": txn.transaction_id}).scalar()
    if not exists:
        ts = txn.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        db.execute(
            text("""
                INSERT INTO transactions (id, account_id, payee_account_id, amount, channel, timestamp, created_at)
                VALUES (:id, :account_id, :payee_id, :amount, :channel, :ts, :now)
                ON CONFLICT (id) DO NOTHING
            """),
            {
                "id": txn.transaction_id,
                "account_id": txn.account_id,
                "payee_id": getattr(txn, "payee_account_id", None),
                "amount": float(txn.amount),
                "channel": txn.channel,
                "ts": ts,
                "now": now,
            },
        )


def _model_version() -> str:
    from app.core.config import settings
    return settings.model_version


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None so PostgreSQL JSONB accepts the value."""
    import math
    if obj is None:
        return None
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj
