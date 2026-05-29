from __future__ import annotations
"""
Shadow table writer for committee scoring.
CRITICAL: Failures in this module must NEVER propagate to the live scoring path.
Every public function absorbs exceptions silently and returns — the caller keeps running.
"""
import structlog
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.detection.tier3.committee_types import ScorerOutput

logger = structlog.get_logger(__name__)


def write_shadow_row(
    db: Session,
    transaction_id: str,
    scorer_outputs: list[ScorerOutput],
    live_score: float,
    live_action: str,
    meta_score: Optional[float] = None,
    specialist_override: bool = False,
    final_committee_score: Optional[float] = None,
    mapie_lower: Optional[float] = None,
    mapie_upper: Optional[float] = None,
) -> None:
    """Parameterized INSERT to shadow_score_committee. Silent on any failure."""
    try:
        scores: dict[str, Optional[float]] = {o.scorer_id: o.score for o in scorer_outputs}
        confs: dict[str, Optional[float]] = {o.scorer_id: o.confidence for o in scorer_outputs}
        flags: dict[str, bool] = {o.scorer_id: o.missing_flag for o in scorer_outputs}

        db.execute(
            text("""
                INSERT INTO shadow_score_committee (
                    transaction_id, scored_at,
                    scorer_a_score, scorer_a_confidence, scorer_a_missing_flag,
                    scorer_b_score, scorer_b_confidence, scorer_b_missing_flag,
                    scorer_c_score, scorer_c_confidence, scorer_c_missing_flag,
                    scorer_d_score, scorer_d_confidence, scorer_d_missing_flag,
                    scorer_f_score, scorer_f_confidence, scorer_f_missing_flag,
                    meta_score, specialist_override, final_committee_score,
                    live_score, live_action, mapie_lower, mapie_upper
                ) VALUES (
                    :txn_id, :scored_at,
                    :a_score, :a_conf, :a_flag,
                    :b_score, :b_conf, :b_flag,
                    :c_score, :c_conf, :c_flag,
                    :d_score, :d_conf, :d_flag,
                    :f_score, :f_conf, :f_flag,
                    :meta_score, :override, :final_score,
                    :live_score, :live_action, :mapie_lower, :mapie_upper
                )
            """),
            {
                "txn_id": transaction_id,
                "scored_at": datetime.now(timezone.utc),
                "a_score": scores.get("A"), "a_conf": confs.get("A"), "a_flag": flags.get("A", True),
                "b_score": scores.get("B"), "b_conf": confs.get("B"), "b_flag": flags.get("B", True),
                "c_score": scores.get("C"), "c_conf": confs.get("C"), "c_flag": flags.get("C", True),
                "d_score": scores.get("D"), "d_conf": confs.get("D"), "d_flag": flags.get("D", True),
                "f_score": scores.get("F"), "f_conf": confs.get("F"), "f_flag": flags.get("F", True),
                "meta_score": meta_score,
                "override": specialist_override,
                "final_score": final_committee_score,
                "live_score": live_score,
                "live_action": live_action,
                "mapie_lower": mapie_lower,
                "mapie_upper": mapie_upper,
            },
        )
        db.commit()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("shadow_write_failed", transaction_id=transaction_id, error=str(exc))


def get_shadow_count(db: Session) -> int:
    """Returns row count from shadow table. Used by meta-learner training gate check."""
    try:
        result = db.execute(text("SELECT COUNT(*) FROM shadow_score_committee"))
        row = result.fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:
        logger.warning("shadow_count_failed", error=str(exc))
        return 0


def get_shadow_training_batch(db: Session, limit: int = 50_000) -> list[dict]:
    """
    Returns rows for meta-learner training.
    Each row has all scorer columns + live_score + live_action.
    Ordered by scored_at DESC to get most recent data.
    """
    try:
        result = db.execute(
            text("""
                SELECT
                    scorer_a_score, scorer_a_confidence, scorer_a_missing_flag,
                    scorer_b_score, scorer_b_confidence, scorer_b_missing_flag,
                    scorer_c_score, scorer_c_confidence, scorer_c_missing_flag,
                    scorer_d_score, scorer_d_confidence, scorer_d_missing_flag,
                    scorer_f_score, scorer_f_confidence, scorer_f_missing_flag,
                    meta_score, specialist_override, live_score, live_action
                FROM shadow_score_committee
                ORDER BY scored_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        )
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]
    except Exception as exc:
        logger.error("shadow_batch_fetch_failed", error=str(exc))
        return []
