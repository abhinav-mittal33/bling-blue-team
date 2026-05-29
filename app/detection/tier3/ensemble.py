"""
LEGACY — will be removed after 2-week observation window post committee go-live.
Do not add new functionality here. Use scorer_a.py for Scorer A upgrades.

Tier 3: XGBoost ensemble scorer — Phase 4 update (P4-4 calibration).

Model loading priority:
  1. ml/models/xgboost_calibrated_v2.joblib — Platt-calibrated (Phase 4+)
  2. ml/models/xgboost_v1.json              — raw Booster (Phase 3 and earlier)

SHAP invariant (P1-6):
  get_base_model() returns the base (uncalibrated) estimator.
  SHAP must NEVER receive the CalibratedClassifierCV wrapper.
  After Phase 4 training: base = calibrated.calibrated_classifiers_[0].estimator.get_booster()

scale_pos_weight: recomputed from actual distribution per training run.
eval_metric='aucpr': NOT 'auc' — PR-AUC for imbalanced data.
"""
from __future__ import annotations
import os
import threading
import numpy as np
import structlog

logger = structlog.get_logger()

_load_lock = threading.Lock()

_BASE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "ml", "models", "xgboost_base_v2.json"
)
_CAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "ml", "models", "xgboost_calibrated_v2.joblib"
)
_LEGACY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "ml", "models", "xgboost_v1.json"
)

_calibrated_model = None   # For scoring (P4-4 Platt wrapper or None)
_base_model = None         # For SHAP (uncalibrated — SHAP invariant)
_legacy_model = None       # Pre-Phase-4 fallback (xgb.Booster)


def _load_models():
    """Load calibrated + base models on first call. Thread-safe via lock."""
    global _calibrated_model, _base_model, _legacy_model

    if _calibrated_model is not None or _base_model is not None:
        return

    with _load_lock:
        # Re-check after acquiring lock — another thread may have loaded between first check and lock
        if _calibrated_model is not None or _base_model is not None:
            return

        # Try Phase 4 calibrated model first
        cal_path = os.path.abspath(_CAL_MODEL_PATH)
        if os.path.exists(cal_path):
            try:
                import joblib
                _calibrated_model = joblib.load(cal_path)
                # Extract base XGB for SHAP — calibrated_classifiers_[0].estimator
                _base_model = _calibrated_model.calibrated_classifiers_[0].estimator
                logger.info("calibrated_model_loaded", path=cal_path)
                return
            except Exception as exc:
                logger.warning("calibrated_model_load_failed", path=cal_path, error=str(exc))

        # Try Phase 4 base model (without calibration)
        base_path = os.path.abspath(_BASE_MODEL_PATH)
        if os.path.exists(base_path):
            try:
                import xgboost as xgb
                _base_model = xgb.Booster()
                _base_model.load_model(base_path)
                logger.info("base_model_loaded", path=base_path)
                return
            except Exception as exc:
                logger.warning("base_model_load_failed", error=str(exc))

        # Fallback: legacy Booster (Phase 3 and earlier)
        legacy_path = os.path.abspath(_LEGACY_MODEL_PATH)
        if os.path.exists(legacy_path):
            try:
                import xgboost as xgb
                _legacy_model = xgb.Booster()
                _legacy_model.load_model(legacy_path)
                _base_model = _legacy_model
                logger.info("legacy_model_loaded", path=legacy_path)
            except Exception as exc:
                logger.warning("legacy_model_load_failed", error=str(exc))


def get_base_model():
    """
    Return base (uncalibrated) model for SHAP computation (P1-6 SHAP invariant).
    After Phase 4: XGBClassifier extracted from calibrated wrapper.
    Before Phase 4: xgb.Booster loaded from legacy JSON.
    NEVER return calibrated_model here — it breaks TreeExplainer.
    """
    _load_models()
    return _base_model


def score(features: dict[str, float]) -> float:
    """
    Score transaction. Returns calibrated probability 0.0-1.0 (or raw if no calibration).
    SHAP computed async by evidence.compute_shap task (P1-6).
    """
    _load_models()

    feature_names = sorted(features.keys())
    feature_values = np.array(
        [features.get(k, float("nan")) for k in feature_names],
        dtype=np.float32,
    ).reshape(1, -1)

    # Prefer calibrated model (Phase 4)
    if _calibrated_model is not None:
        try:
            return float(_calibrated_model.predict_proba(feature_values)[0, 1])
        except Exception as exc:
            logger.error("calibrated_scoring_failed", error=str(exc))

    # Fall back to base model (pre-Phase 4)
    if _base_model is not None:
        try:
            import xgboost as xgb
            if isinstance(_base_model, xgb.Booster):
                dmatrix = xgb.DMatrix(feature_values, feature_names=feature_names, missing=float("nan"))
                return float(_base_model.predict(dmatrix)[0])
            else:
                # XGBClassifier
                return float(_base_model.predict_proba(feature_values)[0, 1])
        except Exception as exc:
            logger.error("base_scoring_failed", error=str(exc))

    return _heuristic_fallback(features)


def _heuristic_fallback(features: dict[str, float]) -> float:
    """Simple heuristic when no model is available."""
    s = 0.0
    if not np.isnan(features.get("payee_vpa_age_days", float("nan"))):
        if features["payee_vpa_age_days"] < 7:
            s += 0.25
    if features.get("is_night", 0) == 1.0:
        s += 0.15
    if not np.isnan(features.get("txn_count_last_1h", float("nan"))):
        if features["txn_count_last_1h"] > 5:
            s += 0.20
    if not np.isnan(features.get("pagerank_fraud_seeded", float("nan"))):
        s += features["pagerank_fraud_seeded"] * 0.30
    if not np.isnan(features.get("cycle_membership", float("nan"))):
        if features["cycle_membership"] == 1.0:
            s += 0.40
    return min(s, 1.0)
