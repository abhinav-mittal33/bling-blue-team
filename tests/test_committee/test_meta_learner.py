"""
Tests for meta_learner.py and conformal_calibrator.py.

Key invariants:
  - predict() never raises (returns fallback on any exception)
  - is_loaded() = False when model file absent (does not crash)
  - _build_meta_feature_vector produces exactly 20 features in correct order
  - conformal interval is for display only: lower ≤ score ≤ upper (not always true
    for MAPIE, but the fallback heuristic always satisfies it)
  - is_fitted() = False when calibrate() never called
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.detection.tier3.committee_types import ScorerOutput, SCORER_IDS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_outputs(scores: dict) -> list[ScorerOutput]:
    defaults = {"A": 0.3, "B": 0.4, "C": 0.5, "D": 0.2, "F": 0.1}
    defaults.update(scores)
    return [
        ScorerOutput(score=defaults[sid], confidence=0.8, missing_flag=False, scorer_id=sid)
        for sid in SCORER_IDS
    ]


def _all_missing() -> list[ScorerOutput]:
    return [ScorerOutput.unavailable(sid) for sid in SCORER_IDS]


CONTEXT = {
    "account_type": "SAVINGS",
    "kyc_age": 35,
    "is_festival": False,
    "is_night": False,
    "daily_txn_count": 3,
}


# ── Meta-learner ───────────────────────────────────────────────────────────────

class TestMetaLearner:
    def _fresh(self):
        import app.detection.tier3.meta_learner as mod
        mod._meta_model = None
        mod._load_attempted = False
        return mod

    def test_is_loaded_false_when_no_model(self, tmp_path):
        mod = self._fresh()
        with patch.object(mod, "_META_MODEL_PATH", str(tmp_path / "none.joblib")):
            result = mod.is_loaded()
        assert result is False

    def test_predict_fallback_when_no_model(self, tmp_path):
        mod = self._fresh()
        outputs = _make_outputs({"A": 0.8})
        with patch.object(mod, "_META_MODEL_PATH", str(tmp_path / "none.joblib")):
            score, override = mod.predict(outputs, CONTEXT)
        assert 0.0 <= score <= 1.0
        assert override is False

    def test_predict_all_missing_returns_neutral(self, tmp_path):
        mod = self._fresh()
        with patch.object(mod, "_META_MODEL_PATH", str(tmp_path / "none.joblib")):
            score, override = mod.predict(_all_missing(), CONTEXT)
        assert score == 0.5
        assert override is False

    def test_predict_with_mock_model(self, tmp_path):
        mod = self._fresh()
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
        mod._meta_model = mock_model
        outputs = _make_outputs({})
        score, override = mod.predict(outputs, CONTEXT)
        assert abs(score - 0.75) < 0.001
        assert override is False

    def test_predict_model_exception_falls_back(self, tmp_path):
        mod = self._fresh()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("model crashed")
        mod._meta_model = mock_model
        outputs = _make_outputs({})
        # Must not raise
        score, override = mod.predict(outputs, CONTEXT)
        assert 0.0 <= score <= 1.0
        assert override is False

    def test_predict_clamps_score_to_0_1(self, tmp_path):
        mod = self._fresh()
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.0, 1.5]])
        mod._meta_model = mock_model
        score, _ = mod.predict(_make_outputs({}), CONTEXT)
        assert score <= 1.0

    def test_build_meta_feature_vector_has_20_dims(self):
        from app.detection.tier3.meta_learner import _build_meta_feature_vector
        outputs = _make_outputs({})
        vec = _build_meta_feature_vector(outputs, CONTEXT)
        assert vec.shape == (20,)

    def test_build_meta_feature_vector_scorer_order(self):
        """Verify scorer features are interleaved in SCORER_ORDER."""
        from app.detection.tier3.meta_learner import _build_meta_feature_vector, _SCORER_ORDER
        outputs = [
            ScorerOutput(score=0.1 * (i + 1), confidence=0.5, missing_flag=False, scorer_id=sid)
            for i, sid in enumerate(_SCORER_ORDER)
        ]
        vec = _build_meta_feature_vector(outputs, CONTEXT)
        # A is first: score at index 0
        assert abs(vec[0] - 0.1) < 0.001   # Scorer A score
        assert abs(vec[3] - 0.2) < 0.001   # Scorer B score

    def test_build_meta_feature_vector_missing_scorer(self):
        """Missing scorer output should be replaced with neutral values."""
        from app.detection.tier3.meta_learner import _build_meta_feature_vector, _SCORER_ORDER
        # Only provide A — B/C/D/F absent
        outputs = [ScorerOutput(score=0.9, confidence=0.8, missing_flag=False, scorer_id="A")]
        vec = _build_meta_feature_vector(outputs, CONTEXT)
        assert vec.shape == (20,)
        # B should be neutral (missing_flag=True → 0.5, 0.0, 1.0 pattern)
        # B starts at index 3
        assert vec[3] == 0.5   # score
        assert vec[4] == 0.0   # confidence
        assert vec[5] == 1.0   # missing_flag

    def test_account_type_encoding(self):
        from app.detection.tier3.meta_learner import _build_meta_feature_vector
        ctx_current = {**CONTEXT, "account_type": "CURRENT"}
        vec = _build_meta_feature_vector(_all_missing(), ctx_current)
        assert abs(vec[15] - 0.33) < 0.01   # CURRENT encoding at index 15

    def test_none_context_values_safe(self):
        from app.detection.tier3.meta_learner import _build_meta_feature_vector
        ctx = {"account_type": None, "kyc_age": None, "is_festival": None}
        vec = _build_meta_feature_vector(_all_missing(), ctx)
        assert vec.shape == (20,)

    def test_fallback_aggregate_only_a_available(self):
        from app.detection.tier3.meta_learner import _compute_fallback_aggregate
        outputs = [
            ScorerOutput(score=0.6, confidence=0.7, missing_flag=False, scorer_id="A"),
        ] + [ScorerOutput.unavailable(sid) for sid in ["B", "C", "D", "F"]]
        result = _compute_fallback_aggregate(outputs)
        assert abs(result - 0.6) < 0.001

    def test_fallback_aggregate_equal_weight_when_all_missing(self):
        from app.detection.tier3.meta_learner import _compute_fallback_aggregate
        result = _compute_fallback_aggregate(_all_missing())
        assert result == 0.5


# ── Conformal calibrator ───────────────────────────────────────────────────────

class TestConformalCalibrator:
    def _fresh(self):
        import app.detection.tier3.conformal_calibrator as mod
        mod._mapie = None
        mod._is_fitted = False
        return mod

    def test_is_fitted_false_initially(self):
        mod = self._fresh()
        assert mod.is_fitted() is False

    def test_unfitted_returns_narrow_heuristic(self):
        mod = self._fresh()
        lower, upper = mod.get_prediction_interval(0.70)
        assert lower == pytest.approx(0.65, abs=0.001)
        assert upper == pytest.approx(0.75, abs=0.001)

    def test_unfitted_clamps_at_0(self):
        mod = self._fresh()
        lower, upper = mod.get_prediction_interval(0.02)
        assert lower >= 0.0

    def test_unfitted_clamps_at_1(self):
        mod = self._fresh()
        lower, upper = mod.get_prediction_interval(0.98)
        assert upper <= 1.0

    def test_load_sets_is_fitted(self):
        mod = self._fresh()
        mock_mapie = MagicMock()
        mod.load(mock_mapie)
        assert mod.is_fitted() is True

    def test_predict_interval_with_mock_mapie(self):
        mod = self._fresh()
        mock_mapie = MagicMock()
        # Return a 3D array [n_samples=1, bounds=2, classes=2]
        mock_mapie.predict.return_value = (
            None,
            np.array([[[0.2, 0.6], [0.3, 0.7]]])   # shape (1, 2, 2)
        )
        mod._mapie = mock_mapie
        mod._is_fitted = True
        lower, upper = mod.get_prediction_interval(0.65)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_predict_interval_exception_falls_back(self):
        mod = self._fresh()
        mock_mapie = MagicMock()
        mock_mapie.predict.side_effect = RuntimeError("mapie failed")
        mod._mapie = mock_mapie
        mod._is_fitted = True
        # Must not raise
        lower, upper = mod.get_prediction_interval(0.60)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
