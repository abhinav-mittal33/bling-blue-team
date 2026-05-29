"""
Tier 3 Committee — Scorer D: Sequence / Set-Based Scorer

Two modes controlled by settings.mamba_limited_mode:

  True  (default): Set-based features from last-90d PostgreSQL history → RF classifier.
         Usable immediately without any labeled sequence data.

  False (future):  Full Mamba state-space sequence model. Returns unavailable stub
         until training infrastructure and labeled session sequences exist.

missing_flag=True when the account has fewer than 5 historical transactions
(insufficient history for reliable set-based features).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import structlog
from sqlalchemy.orm import Session

from app.detection.tier3.committee_types import ScorerOutput
from app.core.config import settings

logger = structlog.get_logger()

_scorer_d_model: Optional[Any] = None
_load_attempted: bool = False

_SCORER_D_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "models", "scorer_d_v1.joblib")
)

# Min transactions required to trust set-based features
_MIN_TXN_HISTORY = 5


def _load_scorer_d() -> None:
    """Lazy-load RF classifier. Thread-safe via GIL."""
    global _scorer_d_model, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    if not settings.mamba_limited_mode:
        return   # Mamba mode — no RF to load

    if os.path.exists(_SCORER_D_PATH):
        try:
            import joblib
            _scorer_d_model = joblib.load(_SCORER_D_PATH)
            logger.info("scorer_d_rf_loaded", path=_SCORER_D_PATH)
        except Exception as exc:
            logger.warning("scorer_d_rf_load_failed", path=_SCORER_D_PATH, error=str(exc))
    else:
        logger.warning("scorer_d_no_rf_model", path=_SCORER_D_PATH)


def _build_set_features(account_id: str, db: Session) -> tuple[dict, bool]:
    """
    Query 90-day behavioral history to produce 7 set-based features.

    Returns (feature_dict, missing_flag).
    missing_flag=True when fewer than MIN_TXN_HISTORY transactions found.
    Uses parameterized query — no f-string interpolation.
    """
    try:
        from sqlalchemy import text
        result = db.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE EXTRACT(HOUR FROM created_at) < 6
                                     OR EXTRACT(HOUR FROM created_at) >= 22)        AS night_count,
                    COUNT(*) FILTER (WHERE payee_vpa_created_at IS NOT NULL
                                     AND EXTRACT(EPOCH FROM (created_at - payee_vpa_created_at))
                                         / 86400.0 < 7)                             AS new_vpa_count,
                    COUNT(*) FILTER (WHERE amount > 100000)                          AS high_amount_count,
                    COUNT(DISTINCT channel)                                           AS distinct_channels,
                    COUNT(*) FILTER (WHERE amount < 1)                               AS micro_test_count,
                    COUNT(*) FILTER (WHERE MOD(CAST(amount AS BIGINT), 1000) = 0
                                     AND amount > 10000)                             AS round_burst_count,
                    COUNT(DISTINCT CASE
                        WHEN EXTRACT(HOUR FROM created_at) < 6
                             OR EXTRACT(HOUR FROM created_at) >= 22 THEN 'night'
                        WHEN amount > 100000 THEN 'high'
                        WHEN payee_vpa_created_at IS NOT NULL
                             AND EXTRACT(EPOCH FROM (created_at - payee_vpa_created_at))
                                 / 86400.0 < 7 THEN 'new_vpa'
                        ELSE NULL
                    END)                                                             AS action_type_count,
                    COUNT(*)                                                          AS total_count
                FROM transactions
                WHERE account_id = :account_id
                  AND created_at > NOW() - INTERVAL '90 days'
            """),
            {"account_id": account_id},
        ).fetchone()

        if result is None or (result.total_count or 0) < _MIN_TXN_HISTORY:
            return {}, True

        total = max(result.total_count, 1)
        feats = {
            "count_of_night_txns": float(result.night_count or 0) / total,
            "count_of_new_vpa_txns": float(result.new_vpa_count or 0) / total,
            "count_of_high_amount_txns": float(result.high_amount_count or 0) / total,
            "count_of_channel_switches": float(min(result.distinct_channels or 1, 5) - 1) / 4.0,
            "has_any_micro_test_payment": float(1 if (result.micro_test_count or 0) > 0 else 0),
            "has_any_round_amount_burst": float(1 if (result.round_burst_count or 0) > 0 else 0),
            "distinct_fraud_proximate_action_types": float(min(result.action_type_count or 0, 3)) / 3.0,
        }
        return feats, False

    except Exception as exc:
        logger.warning("scorer_d_set_features_failed", error=str(exc))
        return {}, True


# Canonical order for the RF feature matrix — must match train_scorer_d.py
_SET_FEATURE_ORDER = [
    "count_of_night_txns",
    "count_of_new_vpa_txns",
    "count_of_high_amount_txns",
    "count_of_channel_switches",
    "has_any_micro_test_payment",
    "has_any_round_amount_burst",
    "distinct_fraud_proximate_action_types",
]


def score(account_id: str, db: Session) -> ScorerOutput:
    """
    Score account using behavioral set features.

    In mamba_limited_mode=True: RF on 7 set features from 90d history.
    In mamba_limited_mode=False: stub (unavailable) until Mamba is trained.
    """
    if not settings.mamba_limited_mode:
        return ScorerOutput.unavailable("D")   # Mamba mode not yet implemented

    _load_scorer_d()

    feat_dict, missing_flag = _build_set_features(account_id, db)

    if _scorer_d_model is None or missing_flag:
        return ScorerOutput.unavailable("D")

    try:
        vec = np.array(
            [feat_dict.get(f, 0.0) for f in _SET_FEATURE_ORDER],
            dtype=np.float32,
        ).reshape(1, -1)

        prob = float(_scorer_d_model.predict_proba(vec)[0, 1])
        confidence = abs(prob - 0.5) * 2.0
        return ScorerOutput(
            score=prob,
            confidence=confidence,
            missing_flag=False,
            scorer_id="D",
        )
    except Exception as exc:
        logger.warning("scorer_d_score_failed", error=str(exc))
        return ScorerOutput.unavailable("D")
