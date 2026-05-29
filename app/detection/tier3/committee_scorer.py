"""
Tier 3 Committee — Main Orchestrator

Drop-in replacement for ensemble.score() in pipeline.py.

  SHADOW mode (default): calls existing tier3_score(features) first, fires all 5
    scorers in a daemon thread, writes to shadow_score_committee, returns the live
    score unchanged. Zero latency impact on POST /api/v1/score.

  LIVE mode (Phase 5 go-live): committee score replaces single XGBoost output.
    Track B specialist override: any scorer exceeding its high-precision threshold
    escalates the final score — single expert signal is not buried by the ensemble.

Both modes are mutually exclusive (enforced by Settings validator in config.py).

Pipeline change contract (CLAUDE.md):
  pipeline.py receives exactly two changes total across the whole rebuild.
  Change 1/2 (Phase 1): replace `tier3_score(features)` with
    `tier3_committee_score(features, txn, db)`.
  Change 2/2 (Phase 2): add context_features dict parameter.
"""
from __future__ import annotations

import threading
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from app.models.schemas import TransactionScoreRequest
from app.detection.tier3.committee_types import (
    CommitteeResult,
    FALLBACK_WEIGHTS,
    SCORER_IDS,
    ScorerOutput,
)
from app.detection.tier3.ensemble import score as tier3_score
from app.detection.tier3 import scorer_a, scorer_b, scorer_c, scorer_d, scorer_f
from app.detection.tier3.shadow_writer import write_shadow_row
from app.core.config import settings

logger = structlog.get_logger()


def tier3_committee_score(
    features: dict[str, float],
    txn: TransactionScoreRequest,
    db: Session,
    context_features: Optional[dict] = None,
) -> float:
    """
    Score transaction through the committee engine.

    Phase 1: shadow mode only — returns existing ensemble score unchanged.
    Phase 5: live mode — returns committee final score.

    context_features: populated by pipeline.py in Phase 2 (Change 2/2).
      Keys: account_type, kyc_age, is_festival, is_night, daily_txn_count.
      None in Phase 1 (meta-learner not yet trained).
    """
    if settings.committee_live_mode:
        return _run_live_committee(features, txn, db, context_features or {})

    if settings.committee_shadow_mode:
        live_score = tier3_score(features)
        _submit_shadow_task(features, txn, db, live_score)
        return live_score

    # Neither mode configured — fall back to existing scorer
    return tier3_score(features)


# ── Shadow mode ────────────────────────────────────────────────────────────────

def _submit_shadow_task(
    features: dict[str, float],
    txn: TransactionScoreRequest,
    db: Session,
    live_score: float,
) -> None:
    """
    Fire-and-forget: launch daemon thread to run committee + write shadow row.

    The live response is returned before this thread does anything.
    All failures inside the thread are absorbed silently — never raise.
    DB session is created fresh inside the thread (sessions are not thread-safe).
    """
    # Snapshot everything the thread needs — txn and features are safe to read
    account_id = txn.account_id
    transaction_id = txn.transaction_id
    # action will be computed post-committee; pass live_score as stand-in for now
    live_action = _score_to_action_str(live_score)

    t = threading.Thread(
        target=_run_shadow_scorers,
        args=(features, account_id, transaction_id, live_score, live_action),
        daemon=True,
        name=f"committee-shadow-{transaction_id[:8]}",
    )
    t.start()


def _run_shadow_scorers(
    features: dict[str, float],
    account_id: str,
    transaction_id: str,
    live_score: float,
    live_action: str,
) -> None:
    """
    Runs inside daemon thread. Creates own DB session.
    Any exception is caught and logged — never propagates.
    """
    try:
        from app.utils.postgres_client import SessionLocal
        shadow_db = SessionLocal()
        try:
            outputs = _run_all_scorers(features, account_id, shadow_db)
            committee_score = _compute_fallback_aggregate(outputs)

            write_shadow_row(
                db=shadow_db,
                transaction_id=transaction_id,
                scorer_outputs=outputs,
                live_score=live_score,
                live_action=live_action,
                meta_score=None,
                specialist_override=False,
                final_committee_score=committee_score,
            )
        finally:
            shadow_db.close()
    except Exception as exc:
        logger.warning(
            "committee_shadow_thread_failed",
            transaction_id=transaction_id,
            error=str(exc),
        )


# ── Live mode (Phase 5) ────────────────────────────────────────────────────────

def _run_live_committee(
    features: dict[str, float],
    txn: TransactionScoreRequest,
    db: Session,
    context_features: dict,
) -> float:
    """
    Committee scoring for live decision-making (Phase 5).

    Track B override: if any scorer surpasses its high-precision threshold,
    the final score escalates regardless of meta-learner output.
    Audit written here — failures propagate (AuditWriteError → 500).
    """
    from app.detection.tier3.committee_auditor import log_committee_score
    from app.core.exceptions import AuditWriteError

    outputs = _run_all_scorers(features, txn.account_id, db)
    override_score = _apply_track_b_override(outputs)
    specialist_override = override_score is not None

    # Try meta-learner if available
    meta_score: Optional[float] = None
    try:
        from app.detection.tier3 import meta_learner
        if meta_learner.is_loaded():
            meta_score, _ = meta_learner.predict(outputs, context_features)
    except Exception as exc:
        logger.warning("committee_meta_learner_unavailable", error=str(exc))

    if specialist_override:
        final_score = float(override_score)
    elif meta_score is not None:
        final_score = float(meta_score)
    else:
        final_score = _compute_fallback_aggregate(outputs)

    # MAPIE conformal interval — for ranking display only, never used for gating
    mapie_lower: Optional[float] = None
    mapie_upper: Optional[float] = None
    try:
        from app.detection.tier3 import conformal_calibrator
        if conformal_calibrator.is_fitted():
            mapie_lower, mapie_upper = conformal_calibrator.get_prediction_interval(final_score)
    except Exception:
        pass

    result = CommitteeResult(
        scorer_outputs=outputs,
        final_score=final_score,
        specialist_override=specialist_override,
        meta_score=meta_score,
        mapie_lower=mapie_lower,
        mapie_upper=mapie_upper,
    )

    log_committee_score(db, txn.transaction_id, result, shadow_mode=False)
    return final_score


# ── Shared scorer execution ────────────────────────────────────────────────────

def _run_all_scorers(
    features: dict[str, float],
    account_id: str,
    db: Session,
) -> list[ScorerOutput]:
    """
    Run all 5 scorers. Each failure degrades to ScorerOutput.unavailable — never raises.
    features dict is also used as cached_graph_features for Scorer B
    (graph features already fetched by feature_builder.py are present in it).
    """
    outputs: list[ScorerOutput] = []

    # Scorer A — Upgraded GBM
    try:
        outputs.append(scorer_a.score(features))
    except Exception as exc:
        logger.warning("scorer_a_unexpected_exception", error=str(exc))
        outputs.append(ScorerOutput.unavailable("A"))

    # Scorer B — Graph embedding + structural context
    try:
        outputs.append(scorer_b.score(account_id, cached_graph_features=features))
    except Exception as exc:
        logger.warning("scorer_b_unexpected_exception", error=str(exc))
        outputs.append(ScorerOutput.unavailable("B"))

    # Scorer C — Prototype vault (FAISS)
    try:
        outputs.append(scorer_c.score(features))
    except Exception as exc:
        logger.warning("scorer_c_unexpected_exception", error=str(exc))
        outputs.append(ScorerOutput.unavailable("C"))

    # Scorer D — Sequence / set-based
    try:
        outputs.append(scorer_d.score(account_id, db))
    except Exception as exc:
        logger.warning("scorer_d_unexpected_exception", error=str(exc))
        outputs.append(ScorerOutput.unavailable("D"))

    # Scorer F — Multilingual remark screener
    try:
        # txn_remark not in current schema — None triggers missing_flag=True gracefully
        remark: Optional[str] = features.get("_txn_remark")   # type: ignore[assignment]
        outputs.append(scorer_f.score(remark))
    except Exception as exc:
        logger.warning("scorer_f_unexpected_exception", error=str(exc))
        outputs.append(ScorerOutput.unavailable("F"))

    return outputs


def _apply_track_b_override(outputs: list[ScorerOutput]) -> Optional[float]:
    """
    Track B: single high-confidence specialist fires → escalate to 1.0.
    Returns 1.0 if any scorer exceeds its threshold; None otherwise.
    """
    thresholds = {
        "A": settings.specialist_override_threshold_a,
        "B": settings.specialist_override_threshold_b,
        "C": settings.specialist_override_threshold_c,
        "D": settings.specialist_override_threshold_d,
        "F": settings.specialist_override_threshold_f,
    }
    for out in outputs:
        if not out.missing_flag and out.score >= thresholds.get(out.scorer_id, 1.0):
            logger.info(
                "committee_track_b_override",
                scorer=out.scorer_id,
                score=round(out.score, 4),
                threshold=thresholds[out.scorer_id],
            )
            return 1.0
    return None


def _compute_fallback_aggregate(outputs: list[ScorerOutput]) -> float:
    """
    Weighted average using FALLBACK_WEIGHTS when meta-learner not yet trained.
    Scorers with missing_flag redistribute their weight proportionally to available scorers.
    """
    available = [o for o in outputs if not o.missing_flag]
    if not available:
        return 0.5   # no signal → neutral

    total_weight = sum(FALLBACK_WEIGHTS.get(o.scorer_id, 0.0) for o in available)
    if total_weight == 0:
        return float(sum(o.score for o in available) / len(available))

    weighted_sum = sum(
        o.score * FALLBACK_WEIGHTS.get(o.scorer_id, 0.0) for o in available
    )
    return float(weighted_sum / total_weight)


def _score_to_action_str(score: float) -> str:
    """Map score to action string for shadow row annotation."""
    if score >= settings.threshold_high_risk:
        return "HIGH_RISK"
    if score >= settings.threshold_review:
        return "REVIEW"
    if score >= settings.threshold_log:
        return "LOG"
    return "PASS"
