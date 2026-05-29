"""
Tier 3 Committee — Stacking Meta-Learner

Predicts fraud probability from the 5 scorer outputs + 5 context features.
Input vector: 15 scorer cols (5 scores + 5 confidences + 5 missing_flags)
            + 5 context cols (account_type_encoded, kyc_age_norm, is_festival,
                              is_night, daily_txn_count_norm)
            = 20-dim total.

Falls back to _compute_fallback_aggregate() when model not loaded.
Train with: python ml/train_meta_learner.py (requires ≥10k shadow rows).

Loaded lazily — is_loaded() is safe to call before predict().
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import structlog

from app.detection.tier3.committee_types import FALLBACK_WEIGHTS, ScorerOutput
from app.core.config import settings

logger = structlog.get_logger()

_meta_model: Optional[Any] = None
_load_attempted: bool = False

_META_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", settings.meta_learner_model_path)
)

# Canonical scorer order — must match training order in train_meta_learner.py
_SCORER_ORDER = ["A", "B", "C", "D", "F"]

# Context feature encoding
_ACCOUNT_TYPE_ENCODING = {
    "SAVINGS": 0.0,
    "CURRENT": 0.33,
    "SALARY": 0.67,
    "NRI": 1.0,
}
_KYC_AGE_CAP = 80.0       # cap for normalization
_DAILY_TXN_CAP = 50.0     # cap for normalization


def _load_meta_learner() -> None:
    """Lazy-load once. Thread-safe via GIL."""
    global _meta_model, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    if os.path.exists(_META_MODEL_PATH):
        try:
            import joblib
            _meta_model = joblib.load(_META_MODEL_PATH)
            logger.info("meta_learner_loaded", path=_META_MODEL_PATH)
        except Exception as exc:
            logger.warning("meta_learner_load_failed", path=_META_MODEL_PATH, error=str(exc))
    else:
        logger.info(
            "meta_learner_not_trained_yet",
            path=_META_MODEL_PATH,
            detail="Using fallback weighted aggregate until >=10k shadow rows collected",
        )


def is_loaded() -> bool:
    """True if a trained meta-learner model is in memory."""
    _load_meta_learner()
    return _meta_model is not None


def predict(
    scorer_outputs: list[ScorerOutput],
    context_features: dict,
) -> tuple[float, bool]:
    """
    Predict fraud probability using the stacking meta-learner.

    Returns (meta_score, specialist_override). specialist_override is always
    False from this function — Track B override is computed separately in
    committee_scorer._apply_track_b_override().

    Falls back to weighted aggregate when model not loaded.
    Never raises — returns (0.5, False) on any exception.
    """
    _load_meta_learner()

    if _meta_model is None:
        return _compute_fallback_aggregate(scorer_outputs), False

    try:
        vec = _build_meta_feature_vector(scorer_outputs, context_features)
        prob = float(_meta_model.predict_proba(vec.reshape(1, -1))[0, 1])
        prob = max(0.0, min(1.0, prob))
        return prob, False
    except Exception as exc:
        logger.warning("meta_learner_predict_failed", error=str(exc))
        return _compute_fallback_aggregate(scorer_outputs), False


def _build_meta_feature_vector(
    scorer_outputs: list[ScorerOutput],
    context_features: dict,
) -> np.ndarray:
    """
    Build the 20-dim meta-feature vector in the canonical training order.

    First 15: [score_A, conf_A, miss_A, score_B, conf_B, miss_B, ..., score_F, conf_F, miss_F]
    Last 5:   [account_type_enc, kyc_age_norm, is_festival, is_night, daily_txn_count_norm]
    """
    output_map = {o.scorer_id: o for o in scorer_outputs}

    scorer_features: list[float] = []
    for sid in _SCORER_ORDER:
        out = output_map.get(sid, ScorerOutput.unavailable(sid))
        scorer_features.extend([
            float(out.score),
            float(out.confidence),
            float(out.missing_flag),
        ])

    account_type = str(context_features.get("account_type", "SAVINGS") or "SAVINGS").upper()
    kyc_age = float(context_features.get("kyc_age") or 0.0)
    is_festival = float(bool(context_features.get("is_festival", False)))
    is_night = float(bool(context_features.get("is_night", False)))
    daily_txn_count = float(context_features.get("daily_txn_count", 0) or 0.0)

    context_vec: list[float] = [
        _ACCOUNT_TYPE_ENCODING.get(account_type, 0.0),
        min(kyc_age / _KYC_AGE_CAP, 1.0),
        is_festival,
        is_night,
        min(daily_txn_count / _DAILY_TXN_CAP, 1.0),
    ]

    return np.array(scorer_features + context_vec, dtype=np.float32)


def _compute_fallback_aggregate(scorer_outputs: list[ScorerOutput]) -> float:
    """Weighted average of available scorers. Used when meta-learner not trained."""
    available = [o for o in scorer_outputs if not o.missing_flag]
    if not available:
        return 0.5
    total_weight = sum(FALLBACK_WEIGHTS.get(o.scorer_id, 0.0) for o in available)
    if total_weight == 0:
        return float(sum(o.score for o in available) / len(available))
    return float(
        sum(o.score * FALLBACK_WEIGHTS.get(o.scorer_id, 0.0) for o in available) / total_weight
    )
