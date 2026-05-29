"""
Phase 3 tests: discovery pipeline + feedback routing.

Key invariants to verify:
  1. discovery_router exits immediately when fraud_score >= LOG threshold
  2. FTRL references are zero in the codebase (import check)
  3. feedback_router.route_false_positive writes to curated_dataset_queue + novelty_registry
  4. feedback_router.route_confirmed_fraud writes to prototype_injection_candidates + curated_dataset_queue
  5. audit_logger has log_feedback_routing_event (not log_ftrl_update)
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call


# ── FTRL removal gate ─────────────────────────────────────────────────────────

class TestFTRLRemovalGate:
    def test_online_learning_file_deleted(self):
        base = '/Users/abhinavmittal/manage /blue team - union bank '
        path = os.path.join(base, 'app/detection/tier3/online_learning.py')
        assert not os.path.exists(path), "online_learning.py must be deleted (Phase 3)"

    def test_novelty_router_file_deleted(self):
        base = '/Users/abhinavmittal/manage /blue team - union bank '
        path = os.path.join(base, 'app/detection/novelty/novelty_router.py')
        assert not os.path.exists(path), "novelty_router.py must be deleted (Phase 3)"

    def test_audit_logger_has_feedback_routing_not_ftrl(self):
        from app.utils import audit_logger
        assert hasattr(audit_logger, 'log_feedback_routing_event'), \
            "log_feedback_routing_event must exist"
        assert not hasattr(audit_logger, 'log_ftrl_update'), \
            "log_ftrl_update must be removed"

    def test_config_has_no_ftrl_cap(self):
        from app.core.config import Settings
        assert not hasattr(Settings, 'ftrl_cap_per_investigator') or \
               'ftrl_cap_per_investigator' not in Settings.model_fields, \
               "ftrl_cap_per_investigator must be removed from config"

    def test_config_has_no_online_model_path(self):
        from app.core.config import Settings
        assert not hasattr(Settings, 'online_model_path') or \
               'online_model_path' not in Settings.model_fields, \
               "online_model_path must be removed from config"


# ── Discovery router ──────────────────────────────────────────────────────────

class TestDiscoveryRouter:
    def test_exits_immediately_on_high_score(self):
        """Discovery must not run on REVIEW/HIGH_RISK transactions."""
        from app.detection.novelty import discovery_router
        from app.core.config import settings

        with patch.object(settings, 'threshold_log', 0.38):
            with patch('app.detection.novelty.discovery_router._route') as mock_route:
                discovery_router.route_discovery(
                    transaction_id='txn_001',
                    account_id='acc_001',
                    anomaly_score=-0.35,
                    fraud_score=0.75,   # above LOG threshold
                    fraud_action='REVIEW',
                    gate_fired=None,
                    graph_features={},
                )
            mock_route.assert_not_called()

    def test_exits_immediately_at_log_threshold(self):
        from app.detection.novelty import discovery_router
        from app.core.config import settings

        with patch.object(settings, 'threshold_log', 0.38):
            with patch('app.detection.novelty.discovery_router._route') as mock_route:
                discovery_router.route_discovery(
                    transaction_id='txn_002',
                    account_id='acc_002',
                    anomaly_score=-0.35,
                    fraud_score=0.38,   # exactly at LOG threshold — must skip
                    fraud_action='LOG',
                    gate_fired=None,
                    graph_features={},
                )
            mock_route.assert_not_called()

    def test_routes_when_fraud_score_below_log(self):
        from app.detection.novelty import discovery_router
        from app.core.config import settings

        with patch.object(settings, 'threshold_log', 0.38):
            with patch('app.detection.novelty.discovery_router._route') as mock_route:
                discovery_router.route_discovery(
                    transaction_id='txn_003',
                    account_id='acc_003',
                    anomaly_score=-0.35,
                    fraud_score=0.10,   # PASS — below LOG
                    fraud_action='PASS',
                    gate_fired=None,
                    graph_features={'sink_score': 0.5},
                )
            mock_route.assert_called_once()

    def test_exception_does_not_propagate(self):
        from app.detection.novelty import discovery_router
        from app.core.config import settings

        with patch.object(settings, 'threshold_log', 0.38):
            with patch('app.detection.novelty.discovery_router._route', side_effect=Exception("DB down")):
                # Must not raise
                discovery_router.route_discovery(
                    transaction_id='txn_004',
                    account_id='acc_004',
                    anomaly_score=-0.35,
                    fraud_score=0.05,
                    fraud_action='PASS',
                    gate_fired=None,
                    graph_features={},
                )

    def test_compute_fingerprint_consistent(self):
        from app.detection.novelty.discovery_router import _compute_fingerprint
        features = {'sink_score': 0.8, 'burst_score': 0.5, 'pagerank_fraud_seeded': 0.3}
        fp1 = _compute_fingerprint(features)
        fp2 = _compute_fingerprint(features)
        assert fp1 == fp2
        assert len(fp1) == 64   # SHA-256 hex


# ── Discovery ensemble ────────────────────────────────────────────────────────

class TestDiscoveryEnsemble:
    def test_unavailable_when_not_loaded(self):
        from app.detection.novelty.discovery_ensemble import DiscoveryEnsemble
        ensemble = DiscoveryEnsemble()
        assert ensemble.available is False

    def test_score_returns_none_when_unavailable(self):
        from app.detection.novelty.discovery_ensemble import DiscoveryEnsemble
        ensemble = DiscoveryEnsemble()
        result = ensemble.score({'sink_score': 0.5})
        assert result is None

    def test_is_novel_false_on_none(self):
        from app.detection.novelty.discovery_ensemble import DiscoveryEnsemble
        ensemble = DiscoveryEnsemble()
        assert ensemble.is_novel(None) is False

    def test_is_novel_true_below_threshold(self):
        from app.detection.novelty.discovery_ensemble import DiscoveryEnsemble, NOVELTY_THRESHOLD
        ensemble = DiscoveryEnsemble()
        assert ensemble.is_novel(NOVELTY_THRESHOLD - 0.01) is True

    def test_is_novel_false_above_threshold(self):
        from app.detection.novelty.discovery_ensemble import DiscoveryEnsemble, NOVELTY_THRESHOLD
        ensemble = DiscoveryEnsemble()
        assert ensemble.is_novel(NOVELTY_THRESHOLD + 0.01) is False


# ── Feedback router ───────────────────────────────────────────────────────────

class TestFeedbackRouter:
    def _make_mock_db(self):
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = None
        return db

    def test_route_false_positive_writes_to_curated_queue(self):
        from app.detection.feedback import feedback_router
        mock_db = self._make_mock_db()

        with patch.object(feedback_router, 'log_feedback_routing_event') as mock_audit:
            feedback_router.route_false_positive(
                alert_id='alert_001',
                transaction_id='txn_001',
                investigator_id_hash='hash_001',
                feature_vector={'sink_score': 0.5, 'burst_score': 0.3},
                db=mock_db,
            )

        # DB execute called at least twice: curated_dataset_queue + novelty_registry
        assert mock_db.execute.call_count >= 2
        # Audit must be called
        mock_audit.assert_called_once()

    def test_route_confirmed_fraud_writes_to_prototype_candidates(self):
        from app.detection.feedback import feedback_router
        mock_db = self._make_mock_db()

        with patch.object(feedback_router, 'log_feedback_routing_event') as mock_audit:
            feedback_router.route_confirmed_fraud(
                alert_id='alert_002',
                transaction_id='txn_002',
                investigator_id_hash='hash_002',
                feature_vector={'sink_score': 0.9},
                fraud_type='rapid_layering',
                notes='Confirmed layering pattern',
                db=mock_db,
            )

        # Called: prototype_injection_candidates + curated_dataset_queue = 2+ executes
        assert mock_db.execute.call_count >= 2
        mock_audit.assert_called_once()

    def test_audit_write_error_propagates_from_false_positive(self):
        from app.detection.feedback import feedback_router
        from app.core.exceptions import AuditWriteError
        mock_db = self._make_mock_db()

        with patch.object(feedback_router, 'log_feedback_routing_event',
                          side_effect=AuditWriteError("audit failed")):
            with pytest.raises(AuditWriteError):
                feedback_router.route_false_positive(
                    alert_id='alert_003',
                    transaction_id='txn_003',
                    investigator_id_hash='hash_003',
                    feature_vector={},
                    db=mock_db,
                )

    def test_audit_write_error_propagates_from_confirmed_fraud(self):
        from app.detection.feedback import feedback_router
        from app.core.exceptions import AuditWriteError
        mock_db = self._make_mock_db()

        with patch.object(feedback_router, 'log_feedback_routing_event',
                          side_effect=AuditWriteError("audit failed")):
            with pytest.raises(AuditWriteError):
                feedback_router.route_confirmed_fraud(
                    alert_id='alert_004',
                    transaction_id='txn_004',
                    investigator_id_hash='hash_004',
                    feature_vector={},
                    fraud_type='hawala',
                    notes=None,
                    db=mock_db,
                )

    def test_fingerprint_deterministic(self):
        from app.detection.feedback.feedback_router import compute_centroid_fingerprint
        fv = {'sink_score': 0.9, 'burst_score': 0.7, 'pagerank_fraud_seeded': 0.5}
        fp1 = compute_centroid_fingerprint(fv)
        fp2 = compute_centroid_fingerprint(fv)
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_sanitize_nan_in_feature_vector(self):
        from app.detection.feedback import feedback_router
        from app.detection.feedback.feedback_router import _sanitize_fv
        import math
        fv = {'valid': 0.5, 'nan_val': float('nan'), 'inf_val': float('inf'), 'none_val': None}
        sanitized = _sanitize_fv(fv)
        assert sanitized['valid'] == 0.5
        assert sanitized['nan_val'] is None
        assert sanitized['inf_val'] is None
        assert sanitized['none_val'] is None


# ── Audit logger ──────────────────────────────────────────────────────────────

class TestAuditLoggerPhase3:
    def test_log_feedback_routing_event_exists(self):
        from app.utils.audit_logger import log_feedback_routing_event
        assert callable(log_feedback_routing_event)

    def test_log_feedback_routing_event_signature(self):
        import inspect
        from app.utils.audit_logger import log_feedback_routing_event
        sig = inspect.signature(log_feedback_routing_event)
        params = list(sig.parameters.keys())
        assert 'db' in params
        assert 'alert_id' in params
        assert 'route' in params
        assert 'event_data' in params

    def test_log_feedback_routing_raises_audit_write_error_on_failure(self):
        from app.utils.audit_logger import log_feedback_routing_event
        from app.core.exceptions import AuditWriteError

        mock_db = MagicMock()
        mock_db.add.side_effect = Exception("DB is down")

        with pytest.raises(AuditWriteError):
            log_feedback_routing_event(
                db=mock_db,
                alert_id='alert_test',
                transaction_id='txn_test',
                route='FALSE_POSITIVE',
                event_data={'key': 'value'},
            )
