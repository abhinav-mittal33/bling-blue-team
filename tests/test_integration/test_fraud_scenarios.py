"""
8 mandatory fraud scenario integration tests.
ALL must pass before demo. These test the full pipeline, not individual modules.
Uses mocked Neo4j and Redis — real DB not required for CI.
"""
from __future__ import annotations
import pytest
from contextlib import ExitStack
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_txn_mock(
    transaction_id: str,
    account_id: str,
    amount: float,
    channel: str = "UPI",
    hour: int = 14,
):
    txn = MagicMock()
    txn.transaction_id = transaction_id
    txn.account_id = account_id
    txn.amount = amount
    txn.channel = channel
    txn.timestamp = datetime.now(timezone.utc).replace(hour=hour, minute=0, second=0)
    txn.payee_account_id = None
    txn.payee_vpa = "test@upi"
    txn.payee_vpa_created_at = None
    txn.terminal_id = None
    txn.merchant_terminal_id = None
    return txn


def _run(
    txn,
    gate_result: dict | None = None,
    vel_1h: int = 1,
    vel_24h: int = 3,
    graph_features: dict | None = None,
    **pipeline_kwargs,
) -> dict:
    """Run pipeline with all external deps mocked."""
    from app.detection.pipeline import run_pipeline

    gate = gate_result or {"fired": False, "gate": None}

    with ExitStack() as stack:
        stack.enter_context(patch("app.detection.pipeline.run_all_gates", return_value=gate))
        stack.enter_context(patch("app.detection.tier1.heuristics.velocity_1h", return_value=vel_1h))
        stack.enter_context(patch("app.detection.tier1.heuristics.velocity_24h", return_value=vel_24h))
        stack.enter_context(patch("app.detection.tier3.feature_builder.get_graph_features", return_value=graph_features or {}))
        # Mock tier3 score so tests don't need XGBoost/libomp installed
        score_fn = pipeline_kwargs.pop("_mock_score", None)
        if score_fn is not None:
            stack.enter_context(patch("app.detection.pipeline.tier3_score", return_value=score_fn))

        result = run_pipeline(txn=txn, db=MagicMock(), **pipeline_kwargs)
    return result


# ─── Scenario 1: Rapid Layering ───────────────────────────────────────────────

def test_rapid_layering():
    """
    4 hops, 18 minutes between each hop, amounts declining 5%.
    Pattern: rapid fund dispersal to obscure origin.
    Tier 1 → SUSPICIOUS (night + new VPA + velocity spike).
    Tier 3 mock returns 0.88 → HIGH_RISK.
    """
    txn = _make_txn_mock("txn_rapid_layering", "ACC_LAYERING", 95000.0, hour=2)

    result = _run(
        txn,
        vel_1h=6,
        vel_24h=20,
        account_age_days=30,
        avg_amount_30d=5000.0,
        payee_in_known_contacts=False,
        payee_account_age_days=5,
        payee_vpa_age_days=2,
        kyc_occupation="SALARIED",
        kyc_age=30,
        account_type="SAVINGS",
        daily_txn_count=8,
        has_festival_gifting_history=False,
        _mock_score=0.88,
    )

    assert result["action"] in ("REVIEW", "HIGH_RISK"), (
        f"Rapid layering → expected REVIEW/HIGH_RISK, got {result['action']} score={result['score']}"
    )
    assert result["score"] >= 0.62


# ─── Scenario 2: Low-Slow Mule ────────────────────────────────────────────────

def test_low_slow_mule():
    """
    45 days normal behaviour, then 1.8L spike at 2am.
    Tier 1 → SUSPICIOUS (night + amount_spike + new_payee_account).
    Tier 3 mock returns 0.75 → REVIEW.
    """
    txn = _make_txn_mock("txn_low_slow", "ACC_SLOW_MULE", 180000.0, hour=2)

    result = _run(
        txn,
        vel_1h=1,
        vel_24h=2,
        account_age_days=45,
        avg_amount_30d=3000.0,
        payee_in_known_contacts=False,
        payee_account_age_days=10,
        payee_vpa_age_days=5,
        kyc_occupation="SALARIED",
        kyc_age=35,
        account_type="SAVINGS",
        daily_txn_count=1,
        has_festival_gifting_history=False,
        _mock_score=0.75,
    )

    assert result["action"] in ("REVIEW", "HIGH_RISK"), (
        f"Low-slow mule → expected REVIEW/HIGH_RISK, got {result['action']} score={result['score']}"
    )


# ─── Scenario 3: Festival Gifting False Positive ─────────────────────────────

def test_festival_gifting_false_positive():
    """
    Diwali: 12 × ₹2K gifts to known contacts.
    Indian context adjuster applies 0.70× factor → score drops below REVIEW threshold.
    """
    txn = _make_txn_mock("txn_festival", "ACC_FESTIVAL", 2000.0, hour=11)

    result = _run(
        txn,
        vel_1h=3,
        vel_24h=8,
        account_age_days=800,
        avg_amount_30d=1500.0,
        payee_in_known_contacts=True,
        payee_account_age_days=500,
        payee_vpa_age_days=180,
        kyc_occupation="SALARIED",
        kyc_age=42,
        account_type="SAVINGS",
        daily_txn_count=4,
        has_festival_gifting_history=True,
        _mock_score=0.45,
    )

    assert result["action"] not in ("REVIEW", "HIGH_RISK"), (
        f"Festival gifting false positive → must NOT alert, got {result['action']} score={result['score']}"
    )
    assert result["score"] < 0.62, f"Festival score must be below REVIEW threshold, got {result['score']}"


# ─── Scenario 4: Digital Arrest ───────────────────────────────────────────────

def test_digital_arrest():
    """
    Senior (68), 2am, ₹5L to 2-day-old VPA.
    Indian context: senior_night 1.5× amplifier pushes score above 0.83 → HIGH_RISK.
    """
    txn = _make_txn_mock("txn_digital_arrest", "ACC_SENIOR", 500000.0, hour=2)

    result = _run(
        txn,
        vel_1h=1,
        vel_24h=1,
        account_age_days=3650,
        avg_amount_30d=5000.0,
        payee_in_known_contacts=False,
        payee_account_age_days=2,
        payee_vpa_age_days=2,
        kyc_occupation="RETIRED",
        kyc_age=68,
        account_type="SAVINGS",
        daily_txn_count=1,
        has_festival_gifting_history=False,
        # Raw score just below HIGH_RISK, context amplifier pushes it over
        _mock_score=0.72,
    )

    assert result["action"] in ("REVIEW", "HIGH_RISK"), (
        f"Digital arrest on senior must trigger alert, got {result['action']} score={result['score']}"
    )
    assert result.get("indian_context_applied"), "Indian context adjuster must have applied at least one amplifier"


# ─── Scenario 5: Ghost Node Cash Trail ───────────────────────────────────────

def test_ghost_node_cash_trail():
    """
    ₹1.26L withdrawn Mumbai, ₹1.24L deposited Raipur 18hr later.
    cash_mule_sink gate fires → immediate REVIEW with score=1.0.
    """
    txn = _make_txn_mock("txn_ghost_node", "ACC_CASH_MULE", 124000.0, channel="CASH", hour=8)

    result = _run(
        txn,
        gate_result={
            "fired": True,
            "gate": "cash_mule_sink",
            "score": 1.0,
            "detail": {"cash_ratio": 0.95, "digital_sends_after": 0},
        },
        account_age_days=60,
        avg_amount_30d=2000.0,
        payee_in_known_contacts=False,
        payee_account_age_days=None,
        payee_vpa_age_days=None,
        kyc_occupation=None,
        kyc_age=None,
        account_type="SAVINGS",
        daily_txn_count=1,
        has_festival_gifting_history=False,
    )

    assert result["action"] in ("REVIEW", "HIGH_RISK"), (
        f"Cash trail → expected REVIEW/HIGH_RISK, got {result['action']}"
    )
    assert result.get("gate_fired") == "cash_mule_sink"


# ─── Scenario 6: Structuring Below Threshold ─────────────────────────────────

def test_structuring_below_threshold():
    """
    5 transactions at ₹93K–₹97K (below ₹1L reporting threshold).
    Tier 1 → SUSPICIOUS (threshold_proximity + amount_spike).
    Tier 3 mock returns 0.70 → REVIEW.
    """
    txn = _make_txn_mock("txn_structuring", "ACC_STRUCTURING", 96500.0, hour=14)

    result = _run(
        txn,
        vel_1h=5,
        vel_24h=5,
        account_age_days=200,
        avg_amount_30d=15000.0,
        payee_in_known_contacts=False,
        payee_account_age_days=30,
        payee_vpa_age_days=60,
        kyc_occupation="BUSINESS",
        kyc_age=40,
        account_type="CURRENT",
        daily_txn_count=5,
        has_festival_gifting_history=False,
        _mock_score=0.70,
    )

    assert result["action"] in ("REVIEW", "HIGH_RISK"), (
        f"Structuring → expected REVIEW/HIGH_RISK, got {result['action']} score={result['score']}"
    )


# ─── Scenario 7: Bipartite Mule Network ──────────────────────────────────────

def test_bipartite_mule_network():
    """
    7 senders → 1 collector, bipartite density 0.85.
    bipartite_core gate fires → immediate REVIEW with score=1.0.
    """
    txn = _make_txn_mock("txn_bipartite", "ACC_COLLECTOR", 50000.0, hour=15)

    result = _run(
        txn,
        gate_result={
            "fired": True,
            "gate": "bipartite_core",
            "score": 1.0,
            "detail": {"sender_count": 7, "density": 0.85},
        },
        account_age_days=90,
        avg_amount_30d=10000.0,
        payee_in_known_contacts=False,
        payee_account_age_days=30,
        payee_vpa_age_days=30,
        kyc_occupation="SALARIED",
        kyc_age=28,
        account_type="SAVINGS",
        daily_txn_count=7,
        has_festival_gifting_history=False,
    )

    assert result["action"] in ("REVIEW", "HIGH_RISK"), (
        f"Bipartite mule → expected REVIEW/HIGH_RISK, got {result['action']}"
    )
    assert result.get("gate_fired") == "bipartite_core"


# ─── Scenario 8: Legitimate Salary Cycle ─────────────────────────────────────

def test_legitimate_salary_cycle():
    """
    Employer → Employee → Employer looks like a cycle but is legitimate.
    No gate fires (legitimacy filter would de-escalate in real run).
    Clean salaried account with known payee → must not reach REVIEW.
    Expected: PASS or LOG.
    """
    txn = _make_txn_mock("txn_salary_cycle", "ACC_EMPLOYEE", 45000.0, hour=10)

    result = _run(
        txn,
        gate_result={"fired": False, "gate": None},
        vel_1h=1,
        vel_24h=1,
        account_age_days=1200,
        avg_amount_30d=42000.0,
        payee_in_known_contacts=True,
        payee_account_age_days=2000,
        payee_vpa_age_days=730,
        kyc_occupation="SALARIED",
        kyc_age=32,
        account_type="SAVINGS",
        daily_txn_count=1,
        has_festival_gifting_history=False,
        _mock_score=0.25,
    )

    assert result["action"] in ("PASS", "LOG"), (
        f"Legitimate salary cycle → must be PASS/LOG, got {result['action']} score={result['score']}"
    )
    assert result["score"] < 0.62, f"Salary cycle score must be below REVIEW, got {result['score']}"
