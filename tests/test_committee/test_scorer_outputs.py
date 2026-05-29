"""
Tests for all 5 scorers — graceful degradation is the primary invariant.

Every scorer must:
  1. Return ScorerOutput (never raise) under any error condition
  2. Return ScorerOutput.unavailable when model file absent
  3. Score all-NaN feature dict without raising
  4. Score empty feature dict without raising
  5. Return score in [0.0, 1.0] and confidence in [0.0, 1.0]
"""
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.detection.tier3.committee_types import ScorerOutput, SCORER_IDS


# ── Helpers ────────────────────────────────────────────────────────────────────

ALL_NAN_FEATURES: dict = {f: float("nan") for f in [
    "txn_amount", "is_night", "hour_of_day", "payee_vpa_age_days",
    "velocity_ratio", "burst_score", "pagerank_fraud_seeded",
]}

EMPTY_FEATURES: dict = {}

MINIMAL_FEATURES: dict = {
    "txn_amount": 50000.0,
    "is_night": 0.0,
    "hour_of_day": 14.0,
    "payee_vpa_age_days": 365.0,
    "velocity_ratio": 0.1,
    "burst_score": 0.05,
    "pagerank_fraud_seeded": 0.02,
    "sink_score": 0.01,
    "bipartite_score": 0.0,
    "betweenness_centrality": 0.001,
    "clustering_coefficient": 0.3,
    "temporal_acceleration": 0.0,
    "community_fraud_ratio": 0.05,
    "txn_count_last_1h": 1.0,
}


def _assert_valid_output(out: ScorerOutput, expected_id: str) -> None:
    assert isinstance(out, ScorerOutput)
    assert out.scorer_id == expected_id
    assert 0.0 <= out.score <= 1.0, f"score {out.score} out of [0,1]"
    assert 0.0 <= out.confidence <= 1.0, f"confidence {out.confidence} out of [0,1]"
    assert isinstance(out.missing_flag, bool)


# ── ScorerOutput dataclass ────────────────────────────────────────────────────

class TestScorerOutput:
    def test_score_clamped_above_1(self):
        out = ScorerOutput(score=1.5, confidence=0.5, missing_flag=False, scorer_id="A")
        assert out.score == 1.0

    def test_score_clamped_below_0(self):
        out = ScorerOutput(score=-0.3, confidence=0.5, missing_flag=False, scorer_id="A")
        assert out.score == 0.0

    def test_confidence_clamped(self):
        out = ScorerOutput(score=0.5, confidence=2.0, missing_flag=False, scorer_id="B")
        assert out.confidence == 1.0

    def test_unavailable_contract(self):
        for sid in SCORER_IDS:
            out = ScorerOutput.unavailable(sid)
            assert out.score == 0.5
            assert out.confidence == 0.0
            assert out.missing_flag is True
            assert out.scorer_id == sid


# ── Scorer A ──────────────────────────────────────────────────────────────────

class TestScorerA:
    def _fresh_scorer_a(self):
        """Import with a fresh module state — reset lazy-load flag."""
        import importlib
        import app.detection.tier3.scorer_a as mod
        mod._load_attempted = False
        mod._scorer_a_model = None
        mod._scorer_a_base = None
        return mod

    def test_no_model_returns_unavailable(self, tmp_path):
        mod = self._fresh_scorer_a()
        with patch.object(mod, "_SCORER_A_PATH", str(tmp_path / "nonexistent.joblib")), \
             patch.object(mod, "_FALLBACK_PATH", str(tmp_path / "nonexistent2.joblib")):
            out = mod.score(MINIMAL_FEATURES)
        _assert_valid_output(out, "A")
        assert out.missing_flag is True

    def test_all_nan_input_no_raise(self, tmp_path):
        mod = self._fresh_scorer_a()
        with patch.object(mod, "_SCORER_A_PATH", str(tmp_path / "none.joblib")), \
             patch.object(mod, "_FALLBACK_PATH", str(tmp_path / "none2.joblib")):
            out = mod.score(ALL_NAN_FEATURES)
        assert isinstance(out, ScorerOutput)

    def test_missing_flag_when_upi_features_all_nan(self, tmp_path):
        mod = self._fresh_scorer_a()
        with patch.object(mod, "_SCORER_A_PATH", str(tmp_path / "none.joblib")), \
             patch.object(mod, "_FALLBACK_PATH", str(tmp_path / "none2.joblib")):
            out = mod.score(MINIMAL_FEATURES)   # no UPI features → all NaN → missing_flag
        assert out.missing_flag is True   # unavailable (no model) or missing_flag from UPI NaN

    def test_get_base_model_for_shap_returns_none_when_unloaded(self, tmp_path):
        mod = self._fresh_scorer_a()
        with patch.object(mod, "_SCORER_A_PATH", str(tmp_path / "none.joblib")), \
             patch.object(mod, "_FALLBACK_PATH", str(tmp_path / "none2.joblib")):
            base = mod.get_base_model_for_shap()
        assert base is None   # no model loaded — never raises

    def test_mock_model_returns_valid_output(self):
        mod = self._fresh_scorer_a()
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mod._scorer_a_model = mock_model
        mod._scorer_a_base = MagicMock()
        out = mod.score(MINIMAL_FEATURES)
        _assert_valid_output(out, "A")
        assert abs(out.score - 0.7) < 0.001
        assert abs(out.confidence - 0.4) < 0.01   # |0.7 - 0.5| * 2

    def test_model_predict_proba_exception_returns_unavailable(self):
        mod = self._fresh_scorer_a()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("model exploded")
        mod._scorer_a_model = mock_model
        out = mod.score(MINIMAL_FEATURES)
        _assert_valid_output(out, "A")
        assert out.missing_flag is True


# ── Scorer B ──────────────────────────────────────────────────────────────────

class TestScorerB:
    def _fresh_scorer_b(self):
        import app.detection.tier3.scorer_b as mod
        mod._load_attempted = False
        mod._scorer_b_model = None
        return mod

    def test_no_embedding_returns_unavailable(self):
        mod = self._fresh_scorer_b()
        with patch("app.detection.tier3.scorer_b.get_redis") as mock_redis:
            r = MagicMock()
            r.get.return_value = None   # embedding absent
            mock_redis.return_value = r
            out = mod.score("acc_001", MINIMAL_FEATURES)
        _assert_valid_output(out, "B")
        assert out.missing_flag is True

    def test_no_model_returns_unavailable(self, tmp_path):
        mod = self._fresh_scorer_b()
        with patch.object(mod, "_SCORER_B_PATH", str(tmp_path / "none.joblib")):
            out = mod.score("acc_001", MINIMAL_FEATURES)
        _assert_valid_output(out, "B")
        assert out.missing_flag is True

    def test_redis_exception_returns_unavailable(self):
        mod = self._fresh_scorer_b()
        with patch("app.detection.tier3.scorer_b.get_redis") as mock_redis:
            mock_redis.side_effect = Exception("redis down")
            out = mod.score("acc_001", MINIMAL_FEATURES)
        _assert_valid_output(out, "B")
        assert out.missing_flag is True

    def test_wrong_embedding_dim_returns_unavailable(self):
        import json
        mod = self._fresh_scorer_b()
        # node2vec_runner stores as json.dumps(vec.tolist()) — 16-dim is wrong dim
        wrong_dim_json = json.dumps(np.zeros(16, dtype=np.float32).tolist())
        with patch("app.detection.tier3.scorer_b.get_redis") as mock_redis:
            r = MagicMock()
            r.get.return_value = wrong_dim_json
            mock_redis.return_value = r
            out = mod.score("acc_001", MINIMAL_FEATURES)
        _assert_valid_output(out, "B")
        assert out.missing_flag is True

    def test_mock_model_valid_embedding_returns_score(self):
        import json
        mod = self._fresh_scorer_b()
        # 32-dim embedding stored as JSON list (node2vec_runner format)
        emb = np.random.rand(32).astype(np.float32)
        emb_json = json.dumps(emb.tolist())

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        mod._scorer_b_model = mock_model

        with patch("app.detection.tier3.scorer_b.get_redis") as mock_redis:
            r = MagicMock()
            r.get.return_value = emb_json
            mock_redis.return_value = r
            out = mod.score("acc_001", MINIMAL_FEATURES)

        _assert_valid_output(out, "B")
        assert abs(out.score - 0.6) < 0.001
        assert out.missing_flag is False


# ── Prototype Vault (Scorer C backend) ────────────────────────────────────────

class TestPrototypeVault:
    def test_unloaded_vault_returns_unavailable(self):
        from app.detection.tier3.prototype_vault import PrototypeVault
        vault = PrototypeVault()
        vec = np.zeros(69, dtype=np.float32)
        out = vault.score(vec)
        _assert_valid_output(out, "C")
        assert out.missing_flag is True

    def test_validate_vector_rejects_all_nan(self):
        from app.detection.tier3.prototype_vault import PrototypeVault
        vault = PrototypeVault()
        vault._loaded = True
        vault._dim = 10
        vec = np.full(10, float("nan"), dtype=np.float32)
        assert vault._validate_vector(vec) is False

    def test_validate_vector_rejects_all_zeros(self):
        from app.detection.tier3.prototype_vault import PrototypeVault
        vault = PrototypeVault()
        vault._loaded = True
        vault._dim = 10
        vec = np.zeros(10, dtype=np.float32)
        assert vault._validate_vector(vec) is False

    def test_validate_vector_rejects_wrong_dim(self):
        from app.detection.tier3.prototype_vault import PrototypeVault
        vault = PrototypeVault()
        vault._loaded = True
        vault._dim = 69
        vec = np.ones(32, dtype=np.float32)
        assert vault._validate_vector(vec) is False

    def test_validate_vector_rejects_mostly_nan(self):
        from app.detection.tier3.prototype_vault import PrototypeVault
        vault = PrototypeVault()
        vault._loaded = True
        vault._dim = 10
        vec = np.array([1.0, float("nan")] * 5, dtype=np.float32)   # 50% NaN > 30% limit
        assert vault._validate_vector(vec) is False

    def test_validate_vector_accepts_valid(self):
        from app.detection.tier3.prototype_vault import PrototypeVault
        vault = PrototypeVault()
        vault._loaded = True
        vault._dim = 10
        vec = np.array([0.5] * 8 + [float("nan")] * 2, dtype=np.float32)   # 20% NaN ≤ 30%
        assert vault._validate_vector(vec) is True

    def test_score_returns_unavailable_on_invalid_vector(self):
        from app.detection.tier3.prototype_vault import PrototypeVault
        vault = PrototypeVault()
        vault._loaded = True
        vault._dim = 69
        # Wrong dim
        vec = np.ones(10, dtype=np.float32)
        out = vault.score(vec)
        assert out.missing_flag is True


# ── Scorer C ──────────────────────────────────────────────────────────────────

class TestScorerC:
    def test_unloaded_vault_returns_unavailable(self):
        from app.detection.tier3 import scorer_c
        from app.detection.tier3.prototype_vault import prototype_vault
        # Ensure vault is not loaded in this test
        original = prototype_vault._loaded
        prototype_vault._loaded = False
        try:
            out = scorer_c.score(MINIMAL_FEATURES)
            _assert_valid_output(out, "C")
            assert out.missing_flag is True
        finally:
            prototype_vault._loaded = original

    def test_score_dict_uses_feature_names_order(self):
        from app.detection.tier3 import scorer_c
        from app.detection.tier3.prototype_vault import prototype_vault
        from ml.feature_registry import FEATURE_NAMES

        mock_vault = MagicMock()
        mock_vault.loaded = True
        mock_vault.score.return_value = ScorerOutput(
            score=0.8, confidence=0.6, missing_flag=False, scorer_id="C"
        )

        with patch("app.detection.tier3.scorer_c.prototype_vault", mock_vault):
            out = scorer_c.score(MINIMAL_FEATURES)

        # Verify vault.score was called with array of len(FEATURE_NAMES)
        mock_vault.score.assert_called_once()
        vec_arg = mock_vault.score.call_args[0][0]
        assert vec_arg.shape == (len(FEATURE_NAMES),)
        _assert_valid_output(out, "C")


# ── Scorer D ──────────────────────────────────────────────────────────────────

class TestScorerD:
    def test_mamba_mode_false_returns_unavailable(self):
        from app.detection.tier3 import scorer_d
        from app.core.config import settings
        mock_db = MagicMock()
        with patch.object(settings, "mamba_limited_mode", False):
            out = scorer_d.score("acc_001", mock_db)
        _assert_valid_output(out, "D")
        assert out.missing_flag is True

    def test_insufficient_history_returns_unavailable(self):
        from app.detection.tier3 import scorer_d
        mock_db = MagicMock()
        # fetchone returns row with total_count < MIN_TXN_HISTORY
        row = MagicMock()
        row.total_count = 2
        mock_db.execute.return_value.fetchone.return_value = row

        with patch.object(scorer_d, "_scorer_d_model", MagicMock()), \
             patch("app.detection.tier3.scorer_d.settings") as mock_settings:
            mock_settings.mamba_limited_mode = True
            out = scorer_d.score("acc_001", mock_db)
        _assert_valid_output(out, "D")
        assert out.missing_flag is True

    def test_db_exception_returns_unavailable(self):
        from app.detection.tier3 import scorer_d
        mock_db = MagicMock()
        mock_db.execute.side_effect = Exception("db error")
        with patch("app.detection.tier3.scorer_d.settings") as mock_settings:
            mock_settings.mamba_limited_mode = True
            out = scorer_d.score("acc_001", mock_db)
        _assert_valid_output(out, "D")
        assert out.missing_flag is True

    def test_no_model_returns_unavailable(self, tmp_path):
        from app.detection.tier3 import scorer_d
        scorer_d._load_attempted = False
        scorer_d._scorer_d_model = None
        row = MagicMock()
        row.total_count = 20
        row.night_count = 5
        row.new_vpa_count = 1
        row.high_amount_count = 2
        row.distinct_channels = 2
        row.micro_test_count = 0
        row.round_burst_count = 0
        row.action_type_count = 1
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchone.return_value = row

        with patch.object(scorer_d, "_SCORER_D_PATH", str(tmp_path / "none.joblib")), \
             patch("app.detection.tier3.scorer_d.settings") as mock_settings:
            mock_settings.mamba_limited_mode = True
            out = scorer_d.score("acc_001", mock_db)
        _assert_valid_output(out, "D")
        assert out.missing_flag is True


# ── Scorer F ──────────────────────────────────────────────────────────────────

class TestScorerF:
    def _fresh_scorer_f(self):
        import app.detection.tier3.scorer_f as mod
        mod._load_attempted = False
        mod._model = None
        mod._cluster_centroids = None
        return mod

    def test_none_remark_returns_neutral(self):
        from app.detection.tier3 import scorer_f
        out = scorer_f.score(None)
        _assert_valid_output(out, "F")
        assert out.score == 0.5
        assert out.missing_flag is True

    def test_empty_remark_returns_neutral(self):
        from app.detection.tier3 import scorer_f
        out = scorer_f.score("   ")
        _assert_valid_output(out, "F")
        assert out.missing_flag is True

    def test_no_phrase_dict_returns_unavailable(self, tmp_path):
        mod = self._fresh_scorer_f()
        with patch("app.detection.tier3.scorer_f.settings") as mock_settings:
            mock_settings.scorer_f_phrase_dict_path = str(tmp_path / "nonexistent.json")
            out = mod.score("send money for customs")
        _assert_valid_output(out, "F")
        assert out.missing_flag is True

    def test_mock_model_high_similarity_gives_high_score(self):
        mod = self._fresh_scorer_f()
        import numpy as np

        # Pre-computed centroid for single cluster
        centroid = np.array([0.9, 0.0, 0.0, 0.3] + [0.0] * 380, dtype=np.float32)
        centroid = centroid / np.linalg.norm(centroid)
        mod._cluster_centroids = {"test_cluster": centroid}

        # Mock model — encode returns same vector as centroid (similarity=1.0)
        mock_model = MagicMock()
        mock_model.encode.return_value = centroid.reshape(1, -1)
        mod._model = mock_model

        out = mod.score("digital arrest payment")
        _assert_valid_output(out, "F")
        # cosine similarity=1.0 → fraud_score=(1+1)/2=1.0
        assert out.score > 0.8
        assert out.missing_flag is False

    def test_mock_model_low_similarity_gives_low_score(self):
        mod = self._fresh_scorer_f()
        import numpy as np

        centroid = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        mod._cluster_centroids = {"test_cluster": centroid}

        # Orthogonal to centroid → cosine=0
        encoding = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = encoding.reshape(1, -1)
        mod._model = mock_model

        out = mod.score("salary payment march")
        _assert_valid_output(out, "F")
        assert out.score < 0.6   # near neutral (cosine=0 → (0+1)/2=0.5)
        assert out.missing_flag is False


# ── Committee Scorer ──────────────────────────────────────────────────────────

class TestCommitteeScorer:
    def test_shadow_mode_returns_live_score_unchanged(self):
        """Shadow mode must return existing ensemble score, never committee score."""
        from app.detection.tier3 import committee_scorer
        from app.core.config import settings

        mock_txn = MagicMock()
        mock_txn.account_id = "acc_test"
        mock_txn.transaction_id = "txn_test_001"
        mock_db = MagicMock()

        with patch.object(settings, "committee_shadow_mode", True), \
             patch.object(settings, "committee_live_mode", False), \
             patch("app.detection.tier3.committee_scorer.tier3_score", return_value=0.72), \
             patch("app.detection.tier3.committee_scorer._submit_shadow_task"):
            score = committee_scorer.tier3_committee_score(
                MINIMAL_FEATURES, mock_txn, mock_db
            )

        assert score == 0.72   # exact live score unchanged

    def test_shadow_mode_submits_shadow_task(self):
        """Shadow task must be launched (even if it runs async in thread)."""
        from app.detection.tier3 import committee_scorer
        from app.core.config import settings

        mock_txn = MagicMock()
        mock_txn.account_id = "acc_test"
        mock_txn.transaction_id = "txn_test_002"
        mock_db = MagicMock()

        with patch.object(settings, "committee_shadow_mode", True), \
             patch.object(settings, "committee_live_mode", False), \
             patch("app.detection.tier3.committee_scorer.tier3_score", return_value=0.50), \
             patch("app.detection.tier3.committee_scorer._submit_shadow_task") as mock_submit:
            committee_scorer.tier3_committee_score(MINIMAL_FEATURES, mock_txn, mock_db)

        mock_submit.assert_called_once()

    def test_shadow_task_absorbs_db_failure(self):
        """Shadow thread must never raise when DB write fails."""
        from app.detection.tier3 import committee_scorer

        with patch("app.detection.tier3.committee_scorer.write_shadow_row") as mock_write, \
             patch("app.detection.tier3.committee_scorer._run_all_scorers", return_value=[
                 ScorerOutput.unavailable("A"),
                 ScorerOutput.unavailable("B"),
                 ScorerOutput.unavailable("C"),
                 ScorerOutput.unavailable("D"),
                 ScorerOutput.unavailable("F"),
             ]):
            mock_write.side_effect = Exception("DB is down")
            mock_db = MagicMock()

            # Must not raise even when shadow write fails
            committee_scorer._run_shadow_scorers(
                MINIMAL_FEATURES, "acc_001", "txn_001", 0.65, "REVIEW"
            )

    def test_fallback_aggregate_all_missing(self):
        from app.detection.tier3.committee_scorer import _compute_fallback_aggregate
        outputs = [ScorerOutput.unavailable(sid) for sid in SCORER_IDS]
        result = _compute_fallback_aggregate(outputs)
        assert result == 0.5   # neutral when all scorers unavailable

    def test_fallback_aggregate_weighted_correctly(self):
        from app.detection.tier3.committee_scorer import _compute_fallback_aggregate
        # Only Scorer A available (weight 0.40)
        outputs = [
            ScorerOutput(score=0.8, confidence=0.6, missing_flag=False, scorer_id="A"),
            ScorerOutput.unavailable("B"),
            ScorerOutput.unavailable("C"),
            ScorerOutput.unavailable("D"),
            ScorerOutput.unavailable("F"),
        ]
        result = _compute_fallback_aggregate(outputs)
        assert abs(result - 0.8) < 0.001   # weight redistributes to A only

    def test_track_b_override_fires_on_threshold(self):
        from app.detection.tier3.committee_scorer import _apply_track_b_override
        from app.core.config import settings

        outputs = [
            ScorerOutput(score=0.95, confidence=0.9, missing_flag=False, scorer_id="A"),  # > 0.92
            ScorerOutput.unavailable("B"),
            ScorerOutput.unavailable("C"),
            ScorerOutput.unavailable("D"),
            ScorerOutput.unavailable("F"),
        ]
        override = _apply_track_b_override(outputs)
        assert override == 1.0

    def test_track_b_override_does_not_fire_below_threshold(self):
        from app.detection.tier3.committee_scorer import _apply_track_b_override

        outputs = [
            ScorerOutput(score=0.80, confidence=0.7, missing_flag=False, scorer_id="A"),  # < 0.92
            ScorerOutput(score=0.75, confidence=0.6, missing_flag=False, scorer_id="B"),  # < 0.90
        ]
        override = _apply_track_b_override(outputs)
        assert override is None

    def test_track_b_missing_flag_does_not_fire(self):
        from app.detection.tier3.committee_scorer import _apply_track_b_override

        # Score above threshold but missing_flag=True — must not override
        outputs = [
            ScorerOutput(score=0.99, confidence=0.0, missing_flag=True, scorer_id="A"),
        ]
        override = _apply_track_b_override(outputs)
        assert override is None

    def test_score_to_action_str(self):
        from app.detection.tier3.committee_scorer import _score_to_action_str
        assert _score_to_action_str(0.10) == "PASS"
        assert _score_to_action_str(0.38) == "LOG"
        assert _score_to_action_str(0.62) == "REVIEW"
        assert _score_to_action_str(0.83) == "HIGH_RISK"

    def test_neither_mode_falls_back_to_legacy_scorer(self):
        from app.detection.tier3 import committee_scorer
        from app.core.config import settings

        mock_txn = MagicMock()
        mock_txn.account_id = "acc_001"
        mock_txn.transaction_id = "txn_fallback_001"
        mock_db = MagicMock()

        with patch.object(settings, "committee_shadow_mode", False), \
             patch.object(settings, "committee_live_mode", False), \
             patch("app.detection.tier3.committee_scorer.tier3_score", return_value=0.45) as mock_legacy:
            score = committee_scorer.tier3_committee_score(MINIMAL_FEATURES, mock_txn, mock_db)

        mock_legacy.assert_called_once()
        assert score == 0.45


# ── Feature registry ──────────────────────────────────────────────────────────

class TestFeatureRegistry:
    def test_upi_session_features_appended_not_reordered(self):
        from ml.feature_registry import (
            FEATURE_NAMES_V4, FEATURE_NAMES_V5, UPI_SESSION_FEATURES
        )
        assert FEATURE_NAMES_V5 == FEATURE_NAMES_V4 + UPI_SESSION_FEATURES

    def test_upi_session_features_count(self):
        from ml.feature_registry import UPI_SESSION_FEATURES
        assert len(UPI_SESSION_FEATURES) == 8

    def test_feature_names_default_unchanged(self):
        from ml.feature_registry import FEATURE_NAMES, FEATURE_NAMES_V2
        assert FEATURE_NAMES is FEATURE_NAMES_V2   # default must stay V2 until scorer_a retrains

    def test_no_duplicates_in_v5(self):
        from ml.feature_registry import FEATURE_NAMES_V5
        assert len(FEATURE_NAMES_V5) == len(set(FEATURE_NAMES_V5)), "Duplicate feature names found"
