from __future__ import annotations
"""
Gate 2: Abandoned Sink Gate
D-00: Account received burst of funds, retained >80%, then went dormant 30+ days.
D-01: Two-path abandoned sink — funds arrived via two distinct paths, both paths
      went silent simultaneously (P3-1, requires P2-9 days_since_last_receive).

Classic mule account signature: receive → forward some → sit → go quiet.
Uses pre-computed nightly Neo4j attributes + Redis feat:{account} — no traversal at query time.
"""
import structlog
from app.graph.queries.sink_queries import check_abandoned_sink
from app.utils.redis_client import get_graph_features
from app.core.security import pseudonymize

logger = structlog.get_logger()

# Account types / occupations where cash retention is normal (legit)
_LEGIT_RETENTION_OCCUPATIONS = frozenset({"merchant", "retailer", "shopkeeper"})
_LEGIT_RETENTION_TYPES = frozenset({"JAN_DHAN"})


def run(account_id: str) -> dict:
    """Check D-01 (two-path, P3-1) then D-00 (original abandoned sink)."""
    # D-01: two-path abandoned sink check (P3-1) — uses days_since_last_receive (P2-9)
    d01 = _check_d01_two_path_sink(account_id)
    if d01.get("fired"):
        return d01

    # D-00: original abandoned sink pattern
    return _run_d00(account_id)


def _check_d01_two_path_sink(account_id: str) -> dict:
    """
    D-01: Two distinct inflow paths have both gone silent simultaneously.
    Both days_since_last_send AND days_since_last_receive > 30 with high sink_score.
    """
    try:
        cached = get_graph_features(account_id)
        # days_since_last_send written by nightly_batch.py (P2-9).
        # If absent from Redis (account not yet batched), do NOT fire — missing data
        # means we cannot confirm dormancy. account_age_days is NOT a substitute.
        raw_send = cached.get("days_since_last_send")
        raw_receive = cached.get("days_since_last_receive")
        if raw_send is None or raw_receive is None:
            return {"fired": False}

        days_since_send = float(raw_send)
        days_since_receive = float(raw_receive)
        sink_score = float(cached.get("sink_score") or 0)
        retention = float(cached.get("retention_ratio") or 0)

        if (days_since_receive > 30
                and days_since_send > 30
                and sink_score > 0.5
                and retention > 0.60):
            logger.warning(
                "sink_gate_d01_fired",
                account=pseudonymize(account_id),
                days_since_receive=days_since_receive,
                sink_score=sink_score,
                retention=retention,
            )
            return {
                "fired": True,
                "gate": "abandoned_sink_d01",
                "evidence": {
                    "pattern": "two_path_abandoned_sink",
                    "days_since_last_receive": days_since_receive,
                    "sink_score": sink_score,
                    "retention_ratio": retention,
                },
            }
    except Exception as exc:
        logger.debug("d01_check_error", error=str(exc))
    return {"fired": False}


def _run_d00(account_id: str) -> dict:
    """
    Returns:
        {'fired': False} if no sink pattern
        {'fired': True, 'gate': 'abandoned_sink', 'evidence': {...}} if sink detected
    """
    sink_data = check_abandoned_sink(account_id)
    if not sink_data:
        return {"fired": False}

    # Legitimacy check: Jan Dhan accounts and cash-heavy businesses retain legitimately
    account_type = sink_data.get("account_type", "")
    kyc_occupation = sink_data.get("kyc_occupation", "")

    if (account_type in _LEGIT_RETENTION_TYPES
            or kyc_occupation in _LEGIT_RETENTION_OCCUPATIONS):
        logger.info("Sink gate: legitimacy filter explained", account=pseudonymize(account_id),
                    reason="legitimate_cash_retention", account_type=account_type)
        return {"fired": False}

    logger.warning(
        "Sink gate fired",
        account=pseudonymize(account_id),
        inflow=sink_data.get("inflow_last_30d"),
        retention=sink_data.get("retention_ratio"),
        dormant_days=sink_data.get("days_since_last_send"),
    )
    return {
        "fired": True,
        "gate": "abandoned_sink",
        "evidence": {
            "inflow_last_30d": sink_data.get("inflow_last_30d"),
            "retention_ratio": sink_data.get("retention_ratio"),
            "days_since_last_send": sink_data.get("days_since_last_send"),
            "account_age_days": sink_data.get("account_age_days"),
        },
    }
