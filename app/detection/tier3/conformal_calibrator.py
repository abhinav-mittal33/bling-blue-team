"""
Tier 3 Committee — MAPIE Conformal Calibrator

Provides 90% conformal prediction intervals for the meta-learner output.

CRITICAL: These intervals are for RANKING and DISPLAY only.
  - score_to_action() always uses the meta-learner point estimate.
  - Track B override always uses scorer point estimates.
  - MAPIE output NEVER determines gating thresholds.

Used in:
  - committee_scorer.py → CommitteeResult.mapie_lower / mapie_upper
  - alert.py → committee_breakdown dict (display-only)

Train / calibrate with: ml/train_meta_learner.py (saves MapieClassifier state).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger()

_mapie: Optional[object] = None   # MapieClassifier
_is_fitted: bool = False


def load(mapie_model: object) -> None:
    """
    Load a pre-fitted MapieClassifier. Called from train_meta_learner.py after
    calibration, and from main.py startup if a serialized MAPIE state exists.
    """
    global _mapie, _is_fitted
    _mapie = mapie_model
    _is_fitted = True
    logger.info("conformal_calibrator_loaded")


def is_fitted() -> bool:
    return _is_fitted


def get_prediction_interval(score: float, alpha: float = 0.10) -> Tuple[float, float]:
    """
    Return 90% conformal prediction interval for a given meta-learner score.

    score: point estimate from meta_learner.predict()
    alpha: error rate (0.10 = 90% coverage)
    Returns: (lower_bound, upper_bound) both in [0.0, 1.0]

    Falls back to symmetric heuristic (score ± 0.05) when MAPIE not fitted.
    This fallback is deliberately narrow — it signals "no calibration data yet"
    rather than a spuriously wide interval.
    """
    if not _is_fitted or _mapie is None:
        # Narrow fallback: small symmetric interval indicating uncalibrated state
        lower = max(0.0, score - 0.05)
        upper = min(1.0, score + 0.05)
        return float(lower), float(upper)

    try:
        # MapieClassifier expects 2D input and returns (y_pred, y_pis)
        # y_pis shape: (n_samples, 2, n_classes) for multi-class or (n_samples, 2) for binary
        X = np.array([[score]], dtype=np.float32)
        _, y_pis = _mapie.predict(X, alpha=alpha)

        # Binary classification: y_pis[:, 0] = lower, y_pis[:, 1] = upper for fraud class
        if y_pis.ndim == 3:
            lower = float(y_pis[0, 0, 1])   # fraud class lower
            upper = float(y_pis[0, 1, 1])   # fraud class upper
        else:
            lower = float(y_pis[0, 0])
            upper = float(y_pis[0, 1])

        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))
        return lower, upper

    except Exception as exc:
        logger.warning("conformal_calibrator_interval_failed", error=str(exc))
        lower = max(0.0, score - 0.05)
        upper = min(1.0, score + 0.05)
        return float(lower), float(upper)


def calibrate(meta_scores: np.ndarray, y_labels: np.ndarray) -> None:
    """
    Fit MapieClassifier on calibration set.

    Called from ml/train_meta_learner.py after meta-learner validation split.
    meta_scores: 1D array of meta-learner output probabilities
    y_labels: 1D array of ground-truth labels (0/1)
    """
    global _mapie, _is_fitted
    try:
        from mapie.classification import MapieClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer

        # Wrap a pre-scored array so MAPIE treats each score as a "model output"
        passthrough = FunctionTransformer()   # identity transform
        mapie = MapieClassifier(estimator=None, cv="prefit")

        # MAPIE needs a fitted estimator that returns predict_proba
        # Use scores directly via a thin wrapper
        _mapie = _FittedScoreWrapper(meta_scores)
        mapie = MapieClassifier(estimator=_mapie, cv="prefit")
        X_cal = meta_scores.reshape(-1, 1)
        mapie.fit(X_cal, y_labels)
        _mapie = mapie
        _is_fitted = True
        logger.info("conformal_calibrator_fitted", n_samples=len(y_labels))
    except Exception as exc:
        logger.error("conformal_calibrator_fit_failed", error=str(exc))


class _FittedScoreWrapper:
    """
    Minimal sklearn estimator wrapper so MapieClassifier (cv='prefit') can use
    pre-computed probability scores directly.
    """
    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.clip(X[:, 0], 0.0, 1.0)
        return np.column_stack([1 - probs, probs])
