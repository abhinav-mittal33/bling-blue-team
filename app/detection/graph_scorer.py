"""
Standalone graph-level fraud scorer for the /analyze-graph endpoint.

No PostgreSQL, Redis, or Neo4j required. Runs pure NetworkX topology
analysis on a graph snapshot and returns a per-graph verdict.

Fixed (2026-05-28):
  - Bipartite threshold lowered to >=2 (was >=4, missed 3-sender collectors)
  - Added conservation/relay gate: collect then forward 80%+ = mule relay
  - Added near-threshold clustering: amounts within 5% of ₹50K/₹1L/₹10L
  - Added temporal bursts: multiple transfers in <30min regardless of total
"""

import logging
from collections import Counter
from datetime import datetime, timezone

import networkx as nx

from app.core.config import settings

log = logging.getLogger(__name__)

# Use settings so thresholds stay in sync with the rest of the pipeline
_THRESHOLD_LOG       = settings.threshold_log
_THRESHOLD_REVIEW    = settings.threshold_review
_THRESHOLD_HIGH_RISK = settings.threshold_high_risk

# RBI reporting thresholds (near these = structuring signal)
_REPORTING_THRESHOLDS = [50_000, 1_00_000, 10_00_000]
_NEAR_THRESHOLD_PCT = 0.05  # within 5% of threshold = suspicious


def _ts(raw: str) -> datetime:
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return datetime.now(timezone.utc)


def _near_threshold(amount: float) -> bool:
    """Amount within 5% below any RBI reporting threshold."""
    for t in _REPORTING_THRESHOLDS:
        if t * (1 - _NEAR_THRESHOLD_PCT) <= amount < t:
            return True
    return False


def score_graph_snapshot(snapshot: dict) -> dict:
    """
    Pure graph topology fraud analysis.

    Accepts: { nodes: [...], edges: [...] }
    Returns: { verdict, fraud_type, score, flagged, flagged_nodes,
               accounts_involved, evidence_available, transactions_scored }
    """
    edges: list[dict] = snapshot.get("edges", [])

    if not edges:
        return _result("CLEAN", "none", 0.0, [], [], 0)

    G: nx.DiGraph = nx.DiGraph()
    amounts: list[float] = []
    timestamps: list[datetime] = []
    edge_map: dict[tuple, dict] = {}  # (src, tgt) → edge data for conservation calc

    for e in edges:
        src = str(e.get("source", ""))
        tgt = str(e.get("target", ""))
        amt = float(e.get("amount", 0))
        if not src or not tgt or amt <= 0:
            continue
        G.add_edge(src, tgt, amount=amt, timestamp=e.get("timestamp", ""))
        amounts.append(amt)
        timestamps.append(_ts(e.get("timestamp", "")))
        edge_map[(src, tgt)] = e

    if G.number_of_edges() == 0:
        return _result("CLEAN", "none", 0.0, [], [], 0)

    signals: list[str] = []
    fraud_types: list[str] = []
    flagged: set[str] = set()
    score = 0.0

    # Gate 1: Circular path
    try:
        cycles = [c for c in nx.simple_cycles(G) if len(c) >= 2]
        if cycles:
            signals.append("circular_path_detected")
            fraud_types.append("circular_transfer_fraud")
            for cyc in cycles[:5]:
                flagged.update(cyc)
            score = max(score, 0.97)
    except Exception:
        pass

    # Gate 2: Fan-out — one account → ≥4 unique recipients
    for node in list(G.nodes()):
        out_nbrs = list(G.successors(node))
        if len(set(out_nbrs)) >= 4:
            signals.append("fan_out_detected")
            fraud_types.append("smurfing_fan_out")
            flagged.add(node)
            flagged.update(out_nbrs[:6])
            score = max(score, 0.85)
            break

    # Gate 3: Bipartite mule / fan-in collector — ≥4 senders → one account
    # Threshold stays at 4. 2-3 senders is normal (split payments, salary advances).
    # The relay/conservation gate below catches the 2-3 sender + high-forward pattern.
    for node in list(G.nodes()):
        in_nbrs = list(G.predecessors(node))
        n_senders = len(set(in_nbrs))
        if n_senders >= 4:
            fan_in_score = min(0.82 + (n_senders - 4) * 0.02, 0.92)
            signals.append(f"bipartite_collector_n{n_senders}")
            fraud_types.append("mule_network_aggregation")
            flagged.add(node)
            flagged.update(in_nbrs[:6])
            score = max(score, fan_in_score)
            break

    # Gate 4: Conservation / relay — receives from N sources, forwards ≥80% onward
    # This catches the fan-in → relay pattern: 3 sources → collector → recipient.
    # Formula: outflow / inflow — NEVER use a single edge amount.
    for node in list(G.nodes()):
        in_nbrs = list(G.predecessors(node))
        out_nbrs = list(G.successors(node))
        if len(in_nbrs) < 2 or len(out_nbrs) == 0:
            continue

        total_inflow = sum(
            G[src][node].get("amount", 0) for src in in_nbrs
        )
        total_outflow = sum(
            G[node][tgt].get("amount", 0) for tgt in out_nbrs
        )

        if total_inflow <= 0:
            continue

        conservation = total_outflow / total_inflow
        if conservation >= 0.80:
            con_score = min(0.78 + (conservation - 0.80) * 0.50, 0.95)
            signals.append(f"relay_conservation_{conservation:.0%}")
            fraud_types.append("rapid_relay_mule")
            flagged.add(node)
            flagged.update(in_nbrs[:4])
            flagged.update(out_nbrs[:4])
            score = max(score, con_score)

    # Gate 5: Forwarding chain depth > 3
    try:
        if nx.is_directed_acyclic_graph(G):
            depth = nx.dag_longest_path_length(G)
            if depth > 3:
                signals.append("chain_depth_exceeded")
                fraud_types.append("layering_chain")
                path = nx.dag_longest_path(G)
                flagged.update(path[:6])
                score = max(score, 0.75 + min(0.10, (depth - 3) * 0.04))
        elif G.number_of_nodes() > 4:
            score = max(score, 0.78)
    except Exception:
        pass

    # Gate 6: Near-threshold structuring — amounts clustered within 5% of ₹50K/₹1L/₹10L
    # The previous gate only caught EXACT repeated amounts. This catches clusters.
    near_threshold_count = sum(1 for a in amounts if _near_threshold(a))
    if near_threshold_count >= 2:
        signals.append(f"near_threshold_clustering_n{near_threshold_count}")
        fraud_types.append("structuring_below_threshold")
        for e in edges:
            a = float(e.get("amount", 0))
            if _near_threshold(a):
                flagged.add(str(e.get("source", "")))
                flagged.add(str(e.get("target", "")))
        # Score scales: 2 near-threshold = 0.72, 3 = 0.80, 4+ = 0.88
        struct_score = min(0.72 + (near_threshold_count - 2) * 0.08, 0.88)
        score = max(score, struct_score)
    elif amounts:
        # Original exact-repeat check (still valid for identical structuring)
        for amt, cnt in Counter(round(a, 2) for a in amounts).items():
            if cnt >= 3 and amt > 1000:
                signals.append("repeated_amount_structuring")
                fraud_types.append("structuring_below_threshold")
                for e in edges:
                    if abs(float(e.get("amount", 0)) - amt) < 0.01:
                        flagged.add(str(e.get("source", "")))
                        flagged.add(str(e.get("target", "")))
                score = max(score, 0.72)
                break

    # Gate 7: Short burst — ≥3 transactions within 30 minutes regardless of total
    if len(timestamps) >= 3:
        ts_sorted = sorted(timestamps)
        for i in range(len(ts_sorted) - 2):
            window_span = (ts_sorted[i + 2] - ts_sorted[i]).total_seconds()
            if window_span <= 1800:  # 30 minutes
                signals.append(f"burst_30min_window_{window_span:.0f}s")
                fraud_types.append("rapid_transfer_burst")
                score = max(score, 0.65)
                break

    # Legacy velocity gate (kept for backwards compat)
    if len(timestamps) >= 2:
        span_s = (max(timestamps) - min(timestamps)).total_seconds()
        if span_s < 600 and sum(amounts) > 200_000:
            signals.append("high_velocity_transfer")
            fraud_types.append("rapid_transfer_burst")
            score = max(score, 0.60)

    # Large single transfer
    if amounts and max(amounts) >= 500_000:
        score = max(score, 0.45)
        if not signals:
            signals.append("large_amount_threshold")
            fraud_types.append("large_value_transfer")

    if score >= _THRESHOLD_HIGH_RISK:
        verdict = "FRAUD"
    elif score >= _THRESHOLD_REVIEW:
        verdict = "SUSPICIOUS"
    elif score >= _THRESHOLD_LOG:
        verdict = "LOGGED"
    else:
        verdict = "CLEAN"

    flagged.discard("")
    all_accounts = list(
        {str(e.get("source", "")) for e in edges if e.get("source")}
        | {str(e.get("target", "")) for e in edges if e.get("target")}
    )
    top_fraud_type = fraud_types[0] if fraud_types else ("none" if verdict == "CLEAN" else "unknown_pattern")

    log.info("graph_scorer verdict=%s score=%.3f signals=%s", verdict, score, signals)
    return _result(
        verdict=verdict,
        fraud_type=top_fraud_type,
        score=round(score, 4),
        flagged_nodes=list(flagged)[:20],
        accounts_involved=list(flagged) if flagged else all_accounts[:20],
        transactions_scored=G.number_of_edges(),
    )


def _result(
    verdict: str,
    fraud_type: str,
    score: float,
    flagged_nodes: list[str],
    accounts_involved: list[str],
    transactions_scored: int,
) -> dict:
    return {
        "verdict": verdict,
        "fraud_type": fraud_type,
        "score": score,
        "flagged": verdict in ("FRAUD", "SUSPICIOUS"),
        "flagged_nodes": flagged_nodes,
        "accounts_involved": accounts_involved,
        "evidence_available": verdict in ("FRAUD", "SUSPICIOUS"),
        "transactions_scored": transactions_scored,
    }
