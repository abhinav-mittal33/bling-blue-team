"""
Tier 3 Committee — Scorer A: Upgraded GBM

Loads scorer_a_v1.joblib (trained on V5 feature set including UPI session features).
Falls back to xgboost_calibrated_v2.joblib (existing calibrated model) when Scorer A
has not yet been trained — Phase 1 committee runs without requiring a new training run.

SHAP invariant (preserved from ensemble.py):
  get_base_model_for_shap() always returns uncalibrated base estimator.
  Never returns CalibratedClassifierCV — TreeExplainer breaks on the wrapper.
"""
from __future__ import annotations

import os
import math
from typing import Any, Optional

import numpy as np
import structlog

from app.detection.tier3.committee_types import ScorerOutput
from app.core.config import settings

logger = structlog.get_logger()

_scorer_a_model: Optional[Any] = None   # CalibratedClassifierCV or XGBClassifier
_scorer_a_base: Optional[Any] = None    # Uncalibrated estimator for SHAP
_load_attempted: bool = False

# 8 new UPI session features — may all be NaN on existing transaction schema
# Appended as UPI_SESSION_FEATURES in feature_registry.py (V5)
_UPI_SESSION_FEATURES = frozenset([
    "upi_collect_request",
    "upi_intent_flag",
    "payee_vpa_verified",
    "upi_app_type",
    "upi_deregistration_flag",
    "upi_pin_attempts_session",
    "upi_session_id_hash",
    "session_amount_ratio",
])

_SCORER_A_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", settings.scorer_a_model_path)
)
_FALLBACK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "models", "xgboost_calibrated_v2.joblib")
)


def _load_scorer_a() -> None:
    """Lazy-load once. GIL makes this effectively thread-safe for CPython."""
    global _scorer_a_model, _scorer_a_base, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    # Prefer dedicated Scorer A model (trained on V5 features)
    if os.path.exists(_SCORER_A_PATH):
        try:
            import joblib
            _scorer_a_model = joblib.load(_SCORER_A_PATH)
            _scorer_a_base = _extract_base(_scorer_a_model)
            logger.info("scorer_a_loaded", path=_SCORER_A_PATH)
            return
        except Exception as exc:
            logger.warning("scorer_a_load_failed", path=_SCORER_A_PATH, error=str(exc))

    # Fallback: existing calibrated XGBoost (same feature set, V2 only)
    if os.path.exists(_FALLBACK_PATH):
        try:
            import joblib
            _scorer_a_model = joblib.load(_FALLBACK_PATH)
            _scorer_a_base = _extract_base(_scorer_a_model)
            logger.info("scorer_a_using_fallback", path=_FALLBACK_PATH)
            return
        except Exception as exc:
            logger.warning("scorer_a_fallback_load_failed", path=_FALLBACK_PATH, error=str(exc))

    logger.warning("scorer_a_no_model", detail="will return ScorerOutput.unavailable on every call")


def _extract_base(model: Any) -> Optional[Any]:
    """Extract uncalibrated base from CalibratedClassifierCV if wrapped; else return as-is."""
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(model, CalibratedClassifierCV):
            return model.calibrated_classifiers_[0].estimator
    except Exception:
        pass
    return model


def _count_upi_nan(features: dict[str, float]) -> int:
    """Count how many UPI session features are NaN (absent from schema → NaN)."""
    count = 0
    for feat in _UPI_SESSION_FEATURES:
        val = features.get(feat, float("nan"))
        if math.isnan(val):
            count += 1
    return count


def score(features: dict[str, float]) -> ScorerOutput:
    """
    Score transaction with Scorer A (GBM).

    missing_flag=True when >20% of the 8 UPI session features are NaN — signals
    that the enriched feature set is not yet available for this transaction.
    confidence is the margin from the decision boundary: |p - 0.5| * 2.
    """
    _load_scorer_a()

    if _scorer_a_model is None:
        return ScorerOutput.unavailable("A")

    nan_count = _count_upi_nan(features)
    missing_flag = nan_count > int(len(_UPI_SESSION_FEATURES) * 0.20)   # >20% threshold = >1.6 → >1

    try:
        feature_names = sorted(features.keys())
        feature_values = np.array(
            [features.get(k, float("nan")) for k in feature_names],
            dtype=np.float32,
        ).reshape(1, -1)

        prob = _predict_proba(feature_names, feature_values)
        confidence = abs(prob - 0.5) * 2.0
        return ScorerOutput(
            score=float(prob),
            confidence=float(confidence),
            missing_flag=missing_flag,
            scorer_id="A",
        )
    except Exception as exc:
        logger.warning("scorer_a_score_failed", error=str(exc))
        return ScorerOutput.unavailable("A")


def _predict_proba(feature_names: list[str], feature_values: np.ndarray) -> float:
    """Route prediction to calibrated wrapper or raw Booster."""
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(_scorer_a_model, CalibratedClassifierCV):
        return float(_scorer_a_model.predict_proba(feature_values)[0, 1])

    try:
        # XGBClassifier
        return float(_scorer_a_model.predict_proba(feature_values)[0, 1])
    except AttributeError:
        pass

    # xgb.Booster
    import xgboost as xgb
    dmatrix = xgb.DMatrix(feature_values, feature_names=feature_names, missing=float("nan"))
    return float(_scorer_a_model.predict(dmatrix)[0])


def get_base_model_for_shap() -> Optional[Any]:
    """
    Return uncalibrated base estimator for SHAP TreeExplainer.
    Called by shap_explainer.py when committee_live_mode=True.
    Never returns the CalibratedClassifierCV wrapper.
    """
    _load_scorer_a()
    return _scorer_a_base
