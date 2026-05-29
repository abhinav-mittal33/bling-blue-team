"""
Gate 0: Rapid Relay Detector (P3-2 — LOG-ONLY pilot).
Detects accounts that receive funds and immediately forward 80%+ to 3+ recipients.
Classic layering / hawala relay pattern.

INVARIANT: LOG-ONLY until GATE0_LIVE=true (env var set after 2-week pilot review).
           When GATE0_LIVE=false: logs detection, returns {'fired': False} to pipeline.
           When GATE0_LIVE=true: returns {'fired': True} → escalates to REVIEW.

Conservation formula: total_outflow / total_inflow
NEVER use amounts[-1] — that is only the most recent outflow, not total.
"""
from __future__ import annotations

import structlog

from app.utils.redis_client import get_graph_features, velocity_1h
from app.core.security import pseudonymize

logger = structlog.get_logger()

_MIN_INFLOW_RS = 5_000.0       # Minimum ₹5K inflow before relay check is meaningful
_CONSERVATION_THRESHOLD = 0.80  # 80% of inflow must leave as outflow
_MIN_OUTGOING_TXN = 3           # At least 3 distinct outgoing transactions in 1h


def _neo4j_relay_check(account_id: str) -> tuple[float, float, int]:
    """
    Real-time Neo4j fallback for relay detection when Redis cache is cold.
    Returns (inflow_6h, outflow_6h, sender_count) via Cypher.
    """
    try:
        from app.graph.neo4j_client import run_query
        rows = run_query(
            """
            MATCH (sender:Account)-[r:SENT]->(a:Account {id: $id})
            WHERE r.timestamp > datetime() - duration({hours: 6})
            WITH sum(r.amount) AS inflow, count(distinct sender) AS n_senders
            OPTIONAL MATCH (a:Account {id: $id})-[r2:SENT]->(recv:Account)
            WHERE r2.timestamp > datetime() - duration({hours: 6})
            RETURN inflow, sum(r2.amount) AS outflow, n_senders
            """,
            {"id": account_id},
        )
        if rows:
            r = rows[0]
            return (
                float(r.get("inflow") or 0),
                float(r.get("outflow") or 0),
                int(r.get("n_senders") or 0),
            )
    except Exception:
        pass
    return 0.0, 0.0, 0


def run(account_id: str) -> dict:
    """
    Check for rapid relay pattern using:
      1. Redis feat:{account} inflow_1h/outflow_1h (fast, 5-min micro-batch fresh)
      2. Neo4j real-time Cypher fallback when Redis cache is cold (6h window)

    Conservation: total_outflow / total_inflow — NEVER amounts[-1].
    """
    from app.core.config import settings

    try:
        cached = get_graph_features(account_id)
        inflow_1h = float(cached.get("inflow_1h") or 0)
        outflow_1h = float(cached.get("outflow_1h") or 0)

        # If Redis cache is cold (no inflow data), fall back to Neo4j real-time
        if inflow_1h < _MIN_INFLOW_RS:
            inflow_1h, outflow_1h, _ = _neo4j_relay_check(account_id)

        # Not enough inflow to be meaningful
        if inflow_1h < _MIN_INFLOW_RS:
            return {"fired": False}

        # Conservation formula — the invariant from CLAUDE.md
        conservation = outflow_1h / inflow_1h

        if conservation < _CONSERVATION_THRESHOLD:
            return {"fired": False}

        # Must have sent to multiple recipients (proxy: outgoing txn count)
        outgoing_count = velocity_1h(account_id)
        if outgoing_count < _MIN_OUTGOING_TXN:
            return {"fired": False}

        evidence = {
            "inflow_1h": round(inflow_1h, 2),
            "outflow_1h": round(outflow_1h, 2),
            "conservation_ratio": round(conservation, 4),
            "outgoing_txn_count_1h": outgoing_count,
        }

        if not settings.gate0_live:
            # LOG-ONLY pilot: record detection but do not escalate
            logger.warning(
                "rapid_relay_detected_log_only",
                account=pseudonymize(account_id),
                conservation=round(conservation, 4),
                outgoing_count=outgoing_count,
                note="GATE0_LIVE=false — not escalating. Review after 2 weeks.",
                **evidence,
            )
            return {"fired": False}

        # Full live mode
        logger.warning(
            "rapid_relay_gate_fired",
            account=pseudonymize(account_id),
            **evidence,
        )
        return {
            "fired": True,
            "gate": "rapid_relay",
            "score": 1.0,
            "evidence": evidence,
        }

    except Exception as exc:
        logger.error("rapid_relay_gate_error", account=pseudonymize(account_id), error=str(exc))
        return {"fired": False}
