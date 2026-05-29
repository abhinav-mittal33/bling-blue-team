"""
Phase 4 tests: developer_queue.py auth + prototype injection security.

Key invariants:
  1. INTERNAL_KEY required — investigator key returns 403
  2. inject endpoint calls prototype_vault.inject_prototype()
  3. Non-PENDING_REVIEW candidates cannot be re-injected
  4. committee_breakdown field is optional in AlertResponse (backward compat)
"""
import pytest
from unittest.mock import MagicMock, patch


# ── Auth enforcement ───────────────────────────────────────────────────────────

class TestDeveloperQueueAuth:
    def _make_request(self, api_key: str) -> MagicMock:
        req = MagicMock()
        req.headers = {"x-api-key": api_key}
        return req

    def test_internal_key_passes(self):
        from app.api.v1.developer_queue import _require_internal_key
        from app.core.config import settings
        from fastapi import HTTPException

        with patch.object(settings, 'internal_api_key', 'test-internal-key'):
            req = self._make_request('test-internal-key')
            # Must not raise
            _require_internal_key(req)

    def test_investigator_key_returns_403(self):
        from app.api.v1.developer_queue import _require_internal_key
        from app.core.config import settings
        from fastapi import HTTPException

        with patch.object(settings, 'internal_api_key', 'test-internal-key'), \
             patch.object(settings, 'investigator_api_key', 'test-investigator-key'):
            req = self._make_request('test-investigator-key')
            with pytest.raises(HTTPException) as exc_info:
                _require_internal_key(req)
            assert exc_info.value.status_code == 403

    def test_graph_engine_key_returns_403(self):
        from app.api.v1.developer_queue import _require_internal_key
        from app.core.config import settings
        from fastapi import HTTPException

        with patch.object(settings, 'internal_api_key', 'test-internal-key'), \
             patch.object(settings, 'graph_engine_api_key', 'test-graph-key'):
            req = self._make_request('test-graph-key')
            with pytest.raises(HTTPException) as exc_info:
                _require_internal_key(req)
            assert exc_info.value.status_code == 403

    def test_empty_key_returns_403(self):
        from app.api.v1.developer_queue import _require_internal_key
        from app.core.config import settings
        from fastapi import HTTPException

        with patch.object(settings, 'internal_api_key', 'test-internal-key'):
            req = self._make_request('')
            with pytest.raises(HTTPException) as exc_info:
                _require_internal_key(req)
            assert exc_info.value.status_code == 403

    def test_no_internal_key_configured_returns_403(self):
        from app.api.v1.developer_queue import _require_internal_key
        from app.core.config import settings
        from fastapi import HTTPException

        with patch.object(settings, 'internal_api_key', ''):
            req = self._make_request('')
            with pytest.raises(HTTPException) as exc_info:
                _require_internal_key(req)
            assert exc_info.value.status_code == 403


# ── Prototype injection security ───────────────────────────────────────────────

class TestPrototypeInjectionSecurity:
    """
    Verifies the core security contract: prototype vectors are never returned
    from any public endpoint, and injection only works from the developer queue.
    """

    def test_prototype_vault_score_returns_scorer_output_not_vector(self):
        """score() must return ScorerOutput, never the raw vector."""
        pytest.importorskip("faiss", reason="faiss-cpu not installed in this env")
        import faiss
        from app.detection.tier3.prototype_vault import PrototypeVault
        from app.detection.tier3.committee_types import ScorerOutput
        import numpy as np

        vault = PrototypeVault()
        dim = 10
        index = faiss.IndexFlatL2(dim)
        vec = np.ones((1, dim), dtype=np.float32)
        index.add(vec)
        vault._index = index
        vault._labels = np.array([1], dtype=np.int32)
        vault._dim = dim
        vault._loaded = True

        query = np.ones(dim, dtype=np.float32)
        result = vault.score(query)
        assert isinstance(result, ScorerOutput), "score() must return ScorerOutput"
        assert not hasattr(result, 'feature_vector'), "ScorerOutput must not expose feature vectors"

    def test_inject_prototype_not_callable_from_scoring_path(self):
        """inject_prototype must not be imported by any scorer (scorer_c.py)."""
        import ast
        import os
        base = '/Users/abhinavmittal/manage /blue team - union bank '
        scorer_c_path = os.path.join(base, 'app/detection/tier3/scorer_c.py')
        with open(scorer_c_path) as f:
            source = f.read()
        assert 'inject_prototype' not in source, \
            "scorer_c.py must not reference inject_prototype"

    def test_committee_scorer_does_not_reference_inject(self):
        """committee_scorer.py must not reference inject_prototype."""
        import os
        base = '/Users/abhinavmittal/manage /blue team - union bank '
        path = os.path.join(base, 'app/detection/tier3/committee_scorer.py')
        with open(path) as f:
            source = f.read()
        assert 'inject_prototype' not in source, \
            "committee_scorer.py must not reference inject_prototype"


# ── Alert schema backward compat ──────────────────────────────────────────────

class TestAlertResponseSchema:
    def test_committee_breakdown_is_optional(self):
        from app.models.schemas import AlertResponse
        import inspect

        # committee_breakdown must have a default (None) — backward compatible
        fields = AlertResponse.model_fields
        assert 'committee_breakdown' in fields, "committee_breakdown field missing from AlertResponse"
        field = fields['committee_breakdown']
        assert field.default is None, "committee_breakdown must default to None"

    def test_alert_response_without_committee_breakdown(self):
        from app.models.schemas import AlertResponse
        # Must be constructable without committee_breakdown
        response = AlertResponse(
            alert_id="alert_001",
            transaction_id="txn_001",
            score=0.75,
            action="REVIEW",
            status="OPEN",
            trail_status="PENDING",
            created_at="2026-05-29T00:00:00Z",
        )
        assert response.committee_breakdown is None

    def test_alert_response_with_committee_breakdown(self):
        from app.models.schemas import AlertResponse
        breakdown = {
            "scorers": {"A": {"score": 0.8, "confidence": 0.6, "missing": False}},
            "meta_score": None,
            "specialist_override": False,
        }
        response = AlertResponse(
            alert_id="alert_002",
            transaction_id="txn_002",
            score=0.80,
            action="HIGH_RISK",
            status="OPEN",
            trail_status="PENDING",
            committee_breakdown=breakdown,
            created_at="2026-05-29T00:00:00Z",
        )
        assert response.committee_breakdown == breakdown


# ── Audit write in developer queue ────────────────────────────────────────────

class TestDeveloperQueueAudit:
    def test_write_inject_audit_best_effort_on_failure(self):
        """Audit failure in developer queue should NOT raise (developer-only endpoint)."""
        from app.api.v1.developer_queue import _write_inject_audit
        from app.core.exceptions import AuditWriteError

        mock_db = MagicMock()
        with patch(
            'app.utils.audit_logger.log_feedback_routing_event',
            side_effect=AuditWriteError("audit down")
        ):
            # Must not raise — audit is best-effort for dev endpoint
            _write_inject_audit(mock_db, 'txn_001', 1, 'rapid_layering')
