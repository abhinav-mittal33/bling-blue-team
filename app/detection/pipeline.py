from __future__ import annotations

"""
Main detection pipeline orchestrator.
Tier 1 → Tier 2 → Tier 3 → action + evidence.
Delegates all logic to tier modules. No business logic here.
"""
import time
import structlog
from sqlalchemy.orm import Session

from app.models.schemas import TransactionScoreRequest
from app.detection.tier1.heuristics import tier1_classify
from app.detection.tier2.gates import run_all_gates
from app.detection.tier3.feature_builder import build_features
from app.detection.tier3.committee_scorer import tier3_committee_score
from app.detection.context.indian_adjuster import apply_indian_context
from app.detection.scoring.thresholds import score_to_action, is_canary_account, score_canary
from app.core.security import pseudonymize

logger = structlog.get_logger()


def run_pipeline(
    txn: TransactionScoreRequest,
    db: Session,
    account_age_days: int = 0,
    avg_amount_30d: float = 0.0,
    payee_in_known_contacts: bool = False,
    payee_account_age_days: int | None = None,
    payee_vpa_age_days: int | None = None,
    kyc_occupation: str | None = None,
    kyc_age: int | None = None,
    account_type: str | None = None,
    daily_txn_count: int = 0,
    has_festival_gifting_history: bool = False,
) -> dict:
    """
    Run the full 3-tier detection pipeline.

    Returns a result dict with: score, action, gate_fired, shap_explanation,
    tier1_flags, tier3_score_raw, indian_context_applied, processing_ms.
    """
    t_start = time.monotonic()
    acct_pseudo = pseudonymize(txn.account_id)

    # ── Canary account check (P6-1) ────────────────────────────────────────────
    if is_canary_account(txn.account_id):
        canary_score, canary_action = score_canary(txn.account_id)
        logger.debug("canary_account_scored", account=acct_pseudo, score=canary_score)
        return _result(
            score=canary_score, action=canary_action, gate_fired=None,
            tier1_flags=["canary"], tier2_gate=None,
            tier3_score_raw=canary_score, shap_explanation=[],
            indian_context_applied={},
            processing_ms=_ms(t_start),
        )

    # ── Tier 1: Fast heuristic classification ─────────────────────────────────
    tier1_result, tier1_flags = tier1_classify(
        txn=txn,
        account_age_days=account_age_days,
        avg_amount_30d=avg_amount_30d,
        payee_in_known_contacts=payee_in_known_contacts,
        payee_account_age_days=payee_account_age_days,
        payee_vpa_age_days=payee_vpa_age_days,
        kyc_occupation=kyc_occupation,
    )

    logger.info("Tier1 complete", account=acct_pseudo, result=tier1_result, flags=tier1_flags)

    if tier1_result == "FAST_CLEAN":
        return _result(
            score=0.05, action="PASS", gate_fired=None,
            tier1_flags=tier1_flags, tier2_gate=None,
            tier3_score_raw=None, shap_explanation=None,
            indian_context_applied={},
            processing_ms=_ms(t_start),
        )

    # ── Tier 2: Hard graph gates ───────────────────────────────────────────────
    gate_result = run_all_gates(
        account_id=txn.account_id,
        terminal_id=txn.merchant_terminal_id,
        db=db,
    )

    if gate_result.get("fired"):
        gate_name = gate_result["gate"]
        logger.warning("Tier2 gate fired", account=acct_pseudo, gate=gate_name)
        return _result(
            score=1.0, action="REVIEW", gate_fired=gate_name,
            tier1_flags=tier1_flags, tier2_gate=gate_name,
            tier3_score_raw=None, shap_explanation=None,
            indian_context_applied={},
            processing_ms=_ms(t_start),
            gate_evidence=gate_result.get("evidence"),
        )

    # UNCERTAIN that cleared all gates exits as LOG — no Tier 3
    if tier1_result == "UNCERTAIN":
        logger.info("UNCERTAIN cleared all gates", account=acct_pseudo)
        return _result(
            score=0.10, action="LOG", gate_fired=None,
            tier1_flags=tier1_flags, tier2_gate=None,
            tier3_score_raw=None, shap_explanation=None,
            indian_context_applied={},
            processing_ms=_ms(t_start),
        )

    # ── Tier 3: ML ensemble (SUSPICIOUS only, gates cleared) ──────────────────
    features = build_features(txn, db)
    # SHAP removed from hot path (P1-6) — computed async by evidence.compute_shap task
    _ts = txn.timestamp
    context_features = {
        "account_type": account_type,
        "kyc_age": float(kyc_age or 0),
        "is_festival": bool(_ts.month == 10 or (_ts.month == 11 and _ts.day <= 15)),
        "is_night": bool(_ts.hour >= 23 or _ts.hour < 5),
        "daily_txn_count": float(daily_txn_count),
    }
    raw_score = tier3_committee_score(features, txn, db, context_features)

    adjusted_score, context_adjustments = apply_indian_context(
        raw_score=raw_score,
        txn_amount=float(txn.amount),
        txn_timestamp=txn.timestamp,
        txn_channel=txn.channel,
        payee_vpa_age_days=payee_vpa_age_days,
        account_type=account_type,
        kyc_age=kyc_age,
        kyc_occupation=kyc_occupation,
        has_festival_gifting_history=has_festival_gifting_history,
        daily_txn_count=daily_txn_count,
        graph_staleness_hours=features.get("graph_staleness_hours"),
    )

    # score_to_action returns (action, jittered_score) — both derived from the same
    # jitter draw so the stored score and action decision are always consistent.
    action, final_score = score_to_action(adjusted_score)
    logger.info("Tier3 complete", account=acct_pseudo,
                raw=round(raw_score, 3), adjusted=round(adjusted_score, 3), action=action)

    return _result(
        score=final_score, action=action, gate_fired=None,
        tier1_flags=tier1_flags, tier2_gate=None,
        tier3_score_raw=raw_score, shap_explanation=[],
        indian_context_applied=context_adjustments,
        processing_ms=_ms(t_start),
        feature_vector=features,
    )


def _ms(t_start: float) -> int:
    return int((time.monotonic() - t_start) * 1000)


def _result(
    score: float,
    action: str,
    gate_fired: str | None,
    tier1_flags: list,
    tier2_gate: str | None,
    tier3_score_raw: float | None,
    shap_explanation: list | None,
    indian_context_applied: dict,
    processing_ms: int,
    gate_evidence: dict | None = None,
    feature_vector: dict | None = None,
) -> dict:
    return {
        "score": round(score, 4),
        "action": action,
        "gate_fired": gate_fired,
        "tier1_flags": tier1_flags,
        "tier2_gate": tier2_gate,
        "tier3_score_raw": tier3_score_raw,
        "shap_explanation": shap_explanation,
        "indian_context_applied": indian_context_applied,
        "gate_evidence": gate_evidence,
        "feature_vector": feature_vector,
        "processing_ms": processing_ms,
    }
