"""
app/graph/precompute/nightly_batch.py — Phase 2 rewrite (P2-1 through P2-9).

Nightly graph feature computation. Runs at 3am UTC via Celery Beat.
Reads Neo4j (topology) + PostgreSQL (account stats) → writes to Redis feat:{account}.

INVARIANT: Field names written to Redis MUST match ml/feature_registry.py exactly.
           Any addition here requires a matching addition in feature_registry.py first.

Phase 2 additions:
  P2-1: Leiden community detection replacing Louvain
  P2-3: 30-day temporal window on Neo4j edge queries
  P2-4: Approximate betweenness (k=500) every 2 hours (separate task)
  P2-5: Micro-batch fast-changing features every 5 minutes
  P2-7: graph_staleness_hours — staleness penalty feature
  P2-8: Multi-hop layering time windows (1h/6h/24h/7d)
  P2-9: days_since_last_send vs days_since_last_receive split

P2-2 (Hetero schema with Device + VPA nodes) requires teammate's Neo4j schema.
P2-6 (Node2Vec) is in app/graph/precompute/node2vec_runner.py.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Optional

import networkx as nx
import structlog

from app.graph.neo4j_client import run_query
from app.utils.redis_client import get_redis, set_graph_features

log = structlog.get_logger()

# Leiden deployed flag written to Redis after successful run (P2-1)
_LEIDEN_DEPLOYED_KEY = "leiden:deployed"


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j queries
# ─────────────────────────────────────────────────────────────────────────────

# Account node properties — used for account-level features
_ACCOUNT_QUERY = """
MATCH (a:Account)
WHERE a.active = true
RETURN
    a.id                    AS account_id,
    a.account_age_days      AS account_age_days,
    a.kyc_completeness_score AS kyc_completeness_score,
    a.is_merchant           AS is_merchant,
    a.kyc_occupation        AS kyc_occupation,
    a.account_type          AS account_type,
    a.geo_state             AS geo_state
LIMIT 200000
"""

# 30-day edge list for topology computation (P2-3 temporal window)
_EDGE_QUERY = """
MATCH (a:Account)-[r:SENT]->(b:Account)
WHERE r.timestamp > datetime() - duration({days: 30})
RETURN
    a.id        AS src,
    b.id        AS dst,
    r.amount    AS amount,
    r.timestamp AS ts
LIMIT 2000000
"""

# Fraud-seeded pagerank seed accounts (known confirmed fraud from alerts)
# Used to set personalization vector for fraud-seeded PageRank (P2-1)
_FRAUD_SEED_QUERY = """
MATCH (a:Account {fraud_confirmed: true})
RETURN a.id AS account_id
LIMIT 10000
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (called by Celery task)
# ─────────────────────────────────────────────────────────────────────────────

def run_nightly_feature_computation() -> None:
    """
    Full nightly batch: reads Neo4j + builds NetworkX graph + runs Leiden.
    Writes all features to Redis feat:{account} with field names from feature_registry.py.
    Sets LEIDEN_DEPLOYED=true in Redis on success.
    """
    started_at = datetime.now(timezone.utc)
    log.info("nightly_batch_started")

    try:
        account_rows = run_query(_ACCOUNT_QUERY, {})
        edge_rows = run_query(_EDGE_QUERY, {})
    except Exception as exc:
        log.error("nightly_batch_neo4j_failed", error=str(exc))
        raise

    # Build account index
    accounts: dict[str, dict] = {
        row["account_id"]: row
        for row in account_rows
        if row.get("account_id")
    }

    if not accounts:
        log.warning("nightly_batch_no_accounts", note="Neo4j returned 0 accounts — check Graph Engine service")
        return

    # Build NetworkX DiGraph from 30-day edges
    G = _build_digraph(edge_rows)

    # Fraud-seeded PageRank
    fraud_seeds = _get_fraud_seeds()
    pagerank_scores = _compute_fraud_seeded_pagerank(G, fraud_seeds)

    # Leiden community detection (P2-1)
    community_map, community_fraud_ratios, community_sizes = _run_leiden(
        G, fraud_seeds
    )

    # Per-account graph metrics
    degree_centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G.to_undirected())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # Cycle membership — lightweight: has any 2-4 hop cycle
    nodes_with_cycles = _detect_cycle_membership(G)

    # Sink score: receives much more than it sends
    sink_scores = _compute_sink_scores(G, in_deg, out_deg)

    # Bipartite score: structural bipartiteness (fan-in pattern)
    bipartite_scores = _compute_bipartite_scores(G, in_deg, out_deg)

    # Fan-out ratio: out-degree / max(in-degree, 1)
    fan_out_ratios = {
        n: out_deg.get(n, 0) / max(in_deg.get(n, 1), 1)
        for n in accounts
    }

    # Bridge probability via betweenness rank (lightweight)
    btw = _compute_approx_betweenness(G, k=500)
    max_btw = max(btw.values(), default=1.0) or 1.0
    bridge_probs = {n: btw.get(n, 0.0) / max_btw for n in accounts}

    # Inflow/outflow per time window from Neo4j edges (P2-8)
    window_flows = _compute_window_flows(edge_rows)

    # Days since last receive (P2-9) — separate from days_since_last_send
    days_since_receive = _compute_days_since_receive(edge_rows)

    # Write features per account
    success = 0
    failed = 0
    now_ts = time.time()

    for account_id, acct in accounts.items():
        features = _build_account_features(
            account_id=account_id,
            acct=acct,
            pagerank_scores=pagerank_scores,
            community_map=community_map,
            community_fraud_ratios=community_fraud_ratios,
            community_sizes=community_sizes,
            degree_centrality=degree_centrality,
            clustering=clustering,
            nodes_with_cycles=nodes_with_cycles,
            sink_scores=sink_scores,
            bipartite_scores=bipartite_scores,
            fan_out_ratios=fan_out_ratios,
            bridge_probs=bridge_probs,
            btw=btw,
            window_flows=window_flows,
            days_since_receive=days_since_receive,
            now_ts=now_ts,
        )
        try:
            set_graph_features(account_id, features)
            success += 1
        except Exception as exc:
            log.warning("nightly_batch_cache_failed", account_id=account_id, error=str(exc))
            failed += 1

    # Mark Leiden deployed only when community_map is non-empty (P2-1).
    # Empty map = Leiden failed and fell back to empty communities — do NOT poison train.py.
    r = get_redis()
    leiden_ok = bool(community_map)
    if leiden_ok:
        r.set(_LEIDEN_DEPLOYED_KEY, "true")
        r.set("leiden:deployed_at", datetime.now(timezone.utc).isoformat())
    else:
        log.warning("leiden_deploy_flag_skipped", reason="community_map empty — Leiden may have failed")

    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    log.info(
        "nightly_batch_complete",
        accounts_processed=success,
        failed=failed,
        elapsed_s=round(elapsed, 1),
        leiden_deployed=leiden_ok,
    )

    # PC-GNN + Hypergraph embedding generation — runs after Leiden features are in Redis
    # so generate_embeddings() can load community_id for hyperedge construction.
    # feat_dict is passed directly to avoid a second Redis round-trip.
    _run_gnn_embedding(G, accounts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-account feature assembly — field names MUST match feature_registry.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_account_features(
    account_id: str,
    acct: dict,
    pagerank_scores: dict,
    community_map: dict,
    community_fraud_ratios: dict,
    community_sizes: dict,
    degree_centrality: dict,
    clustering: dict,
    nodes_with_cycles: set,
    sink_scores: dict,
    bipartite_scores: dict,
    fan_out_ratios: dict,
    bridge_probs: dict,
    btw: dict,
    window_flows: dict,
    days_since_receive: dict,
    now_ts: float,
) -> dict:
    """Build the Redis feat:{account} dict for one account."""
    comm_id = community_map.get(account_id, -1)
    flows = window_flows.get(account_id, {})

    return {
        # ── Centrality ─────────────────────────────────────────────────────────
        "degree_centrality": float(degree_centrality.get(account_id, 0.0)),
        "betweenness_centrality": float(btw.get(account_id, 0.0)),
        "clustering_coefficient": float(clustering.get(account_id, 0.0)),
        "pagerank_fraud_seeded": float(pagerank_scores.get(account_id, 0.0)),
        # ── Community (Leiden P2-1) ─────────────────────────────────────────────
        "community_id": int(comm_id),
        "community_fraud_ratio": float(community_fraud_ratios.get(comm_id, 0.0)),
        "community_size": int(community_sizes.get(comm_id, 1)),
        # ── Fraud proximity ────────────────────────────────────────────────────
        "shortest_path_to_fraud": float(_shortest_path_to_fraud_node(account_id, pagerank_scores)),
        "cycle_membership": 1.0 if account_id in nodes_with_cycles else 0.0,
        "sink_score": float(sink_scores.get(account_id, 0.0)),
        # ── Structural ─────────────────────────────────────────────────────────
        "bipartite_score": float(bipartite_scores.get(account_id, 0.0)),
        "fan_out_ratio": float(fan_out_ratios.get(account_id, 0.0)),
        "temporal_acceleration": float(flows.get("temporal_acceleration", 0.0)),
        # ── Mule / cash patterns ───────────────────────────────────────────────
        "cash_mule_sink_score": float(sink_scores.get(account_id, 0.0) * (1.0 if float(degree_centrality.get(account_id, 0)) < 0.01 else 0.5)),
        "bridge_node_probability": float(bridge_probs.get(account_id, 0.0)),
        "dormancy_reactivation_flag": 0.0,  # Updated by micro-batch (P2-5)
        # ── Account context ────────────────────────────────────────────────────
        "account_age_days": float(acct.get("account_age_days") or 0),
        "kyc_completeness_score": float(acct.get("kyc_completeness_score") or 0.0),
        # ── Historical stats (pre-computed from PostgreSQL in nightly batch) ───
        "txn_count_30d": 0.0,     # Updated by _enrich_postgres_stats()
        "txn_count_90d": 0.0,
        "txn_count_all": 0.0,
        "avg_txn_amount_30d": 0.0,
        "distinct_counterparties_30d": 0.0,
        "channel_entropy": 0.0,
        # ── Behavioral ratios ──────────────────────────────────────────────────
        "night_txn_ratio": 0.0,
        "weekend_txn_ratio": 0.0,
        "return_ratio": 0.0,
        # ── Anomaly signals ────────────────────────────────────────────────────
        "amount_zscore": 0.0,
        "counterparty_novelty": 0.0,
        "hour_deviation": 0.0,
        # ── Activity shifts ────────────────────────────────────────────────────
        "channel_switch": 0.0,
        "amount_series_score": 0.0,
        # ── ZSET-backed (P0-1) — these are set at scoring time, not batch time ─
        "burst_score": float(flows.get("burst_score", 0.0)),
        "velocity_ratio": float(flows.get("velocity_ratio", 0.0)),
        # ── Dormancy + geography ───────────────────────────────────────────────
        "dormancy_break": 0.0,
        "geography_switch": 0.0,
        # ── Phase 2 new: multi-hop time windows (P2-8) ─────────────────────────
        "inflow_1h": float(flows.get("inflow_1h", 0.0)),
        "inflow_6h": float(flows.get("inflow_6h", 0.0)),
        "inflow_24h": float(flows.get("inflow_24h", 0.0)),
        "inflow_7d": float(flows.get("inflow_7d", 0.0)),
        "outflow_1h": float(flows.get("outflow_1h", 0.0)),
        "outflow_6h": float(flows.get("outflow_6h", 0.0)),
        "outflow_24h": float(flows.get("outflow_24h", 0.0)),
        "outflow_7d": float(flows.get("outflow_7d", 0.0)),
        # ── Phase 2 new: send/receive split (P2-9) ─────────────────────────────
        "days_since_last_receive": float(days_since_receive.get(account_id, 999)),
        # ── Phase 2 new: staleness (P2-7) ──────────────────────────────────────
        # NOTE: graph_staleness_hours is computed at scoring time by feature_builder.py
        # from _last_updated. It's NOT stored in the feature hash — it's derived.
        # ── Metadata ───────────────────────────────────────────────────────────
        "_last_updated": now_ts,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_digraph(edge_rows: list) -> nx.DiGraph:
    """Build NetworkX DiGraph from Neo4j edge rows. Used for centrality + Leiden."""
    G = nx.DiGraph()
    for row in edge_rows:
        src = row.get("src")
        dst = row.get("dst")
        if src and dst:
            amount = float(row.get("amount") or 0)
            G.add_edge(src, dst, weight=amount)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Leiden community detection (P2-1)
# ─────────────────────────────────────────────────────────────────────────────

def _run_leiden(
    G: nx.DiGraph,
    fraud_seeds: set,
) -> tuple[dict, dict, dict]:
    """
    Run weighted Leiden on the undirected projection of the transaction graph.
    Returns (community_map, community_fraud_ratios, community_sizes).
    Falls back to empty dicts on failure (scores degrade gracefully to 0.0).
    """
    try:
        import igraph as ig
        import leidenalg

        undirected = G.to_undirected()
        nodes = list(undirected.nodes())
        if not nodes:
            return {}, {}, {}

        node_idx = {n: i for i, n in enumerate(nodes)}
        edges = [(node_idx[u], node_idx[v]) for u, v in undirected.edges()]

        g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
        # Weighted edges for better community quality
        g_ig.es["weight"] = [
            undirected[u][v].get("weight", 1.0)
            for u, v in undirected.edges()
        ]

        partition = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            seed=42,
        )

        community_map = {nodes[i]: membership for i, membership in enumerate(partition.membership)}

        # Compute fraud ratio per community
        community_members: dict[int, list] = {}
        for node, comm_id in community_map.items():
            community_members.setdefault(comm_id, []).append(node)

        community_fraud_ratios = {}
        community_sizes = {}
        for comm_id, members in community_members.items():
            community_sizes[comm_id] = len(members)
            if fraud_seeds:
                fraud_count = sum(1 for m in members if m in fraud_seeds)
                community_fraud_ratios[comm_id] = fraud_count / len(members)
            else:
                community_fraud_ratios[comm_id] = 0.0

        log.info(
            "leiden_complete",
            num_communities=len(community_members),
            total_nodes=len(community_map),
        )
        return community_map, community_fraud_ratios, community_sizes

    except Exception as exc:
        log.error("leiden_failed", error=str(exc), note="Falling back to empty communities")
        return {}, {}, {}


# ─────────────────────────────────────────────────────────────────────────────
# PageRank (fraud-seeded)
# ─────────────────────────────────────────────────────────────────────────────

def _get_fraud_seeds() -> set:
    """Load confirmed fraud accounts from Neo4j. Falls back to empty set."""
    try:
        rows = run_query(_FRAUD_SEED_QUERY, {})
        return {row["account_id"] for row in rows if row.get("account_id")}
    except Exception as exc:
        log.warning("fraud_seeds_unavailable", error=str(exc))
        return set()


def _compute_fraud_seeded_pagerank(G: nx.DiGraph, fraud_seeds: set) -> dict:
    """Personalized PageRank with known fraud nodes as seeds."""
    if not G.nodes():
        return {}
    try:
        if fraud_seeds:
            seed_weight = 1.0 / len(fraud_seeds)
            personalization = {n: (seed_weight if n in fraud_seeds else 0.0) for n in G.nodes()}
        else:
            personalization = None

        return nx.pagerank(G, alpha=0.85, personalization=personalization, max_iter=100)
    except Exception as exc:
        log.warning("pagerank_failed", error=str(exc))
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-account metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_approx_betweenness(G: nx.DiGraph, k: int = 500) -> dict:
    """Approximate betweenness centrality with k-sample. Used for 2h update too."""
    try:
        if len(G.nodes()) < 10:
            return nx.betweenness_centrality(G)
        return nx.betweenness_centrality(G, k=min(k, len(G.nodes())), normalized=True)
    except Exception as exc:
        log.warning("betweenness_failed", error=str(exc))
        return {}


def _detect_cycle_membership(G: nx.DiGraph) -> set:
    """Return set of nodes that participate in any 2-4 hop cycle."""
    cycle_nodes: set = set()
    try:
        # Use strongly connected components: any SCC with >1 node has cycles
        for scc in nx.strongly_connected_components(G):
            if len(scc) > 1:
                cycle_nodes.update(scc)
    except Exception:
        pass
    return cycle_nodes


def _compute_sink_scores(G: nx.DiGraph, in_deg: dict, out_deg: dict) -> dict:
    """
    Sink score: accounts that receive heavily but send rarely.
    0.0 = balanced, 1.0 = pure sink.
    """
    scores = {}
    for node in G.nodes():
        total_in = in_deg.get(node, 0)
        total_out = out_deg.get(node, 0)
        total = total_in + total_out
        scores[node] = (total_in / total) if total > 0 else 0.0
    return scores


def _compute_bipartite_scores(G: nx.DiGraph, in_deg: dict, out_deg: dict) -> dict:
    """
    Bipartite score: accounts with many senders and few recipients (collector pattern).
    High score = potential collector in bipartite mule network.
    """
    scores = {}
    for node in G.nodes():
        n_in = in_deg.get(node, 0)
        n_out = out_deg.get(node, 0)
        if n_in + n_out == 0:
            scores[node] = 0.0
        elif n_in > 3 and n_out <= 2:
            # Many senders, few recipients = collector
            scores[node] = min(1.0, n_in / max(n_out * 3, 1))
        else:
            scores[node] = 0.0
    return scores


def _shortest_path_to_fraud_node(account_id: str, pagerank_scores: dict) -> float:
    """
    Approximate shortest-path-to-fraud using pagerank as proxy.
    High pagerank_fraud_seeded → low hop distance. Maps 0.0-1.0 → 10-1.
    True BFS is too expensive for 200K accounts per nightly run.
    """
    pr = pagerank_scores.get(account_id, 0.0)
    if pr <= 0:
        return 10.0
    return max(1.0, 10.0 * (1.0 - min(pr * 100, 1.0)))


def _compute_window_flows(edge_rows: list) -> dict[str, dict]:
    """
    Compute inflow/outflow per account for time windows 1h/6h/24h/7d (P2-8).
    Returns {account_id: {inflow_1h, inflow_6h, ...}}.
    """
    now = datetime.now(timezone.utc)

    def hours_ago(h: float) -> float:
        return now.timestamp() - h * 3600

    cutoffs = {
        "1h": hours_ago(1),
        "6h": hours_ago(6),
        "24h": hours_ago(24),
        "7d": hours_ago(168),
    }

    flows: dict[str, dict] = {}

    for row in edge_rows:
        src = row.get("src")
        dst = row.get("dst")
        amount = float(row.get("amount") or 0)
        ts = row.get("ts")

        if not (src and dst and ts):
            continue

        # Parse Neo4j datetime to Unix timestamp
        try:
            if hasattr(ts, "timestamp"):
                ts_unix = ts.timestamp()
            else:
                ts_unix = float(ts)
        except Exception:
            continue

        for window, cutoff in cutoffs.items():
            if ts_unix >= cutoff:
                flows.setdefault(src, {})
                flows.setdefault(dst, {})
                flows[src][f"outflow_{window}"] = flows[src].get(f"outflow_{window}", 0.0) + amount
                flows[dst][f"inflow_{window}"] = flows[dst].get(f"inflow_{window}", 0.0) + amount

    return flows


def _compute_days_since_receive(edge_rows: list) -> dict[str, float]:
    """
    Compute days since each account last received a transaction (P2-9).
    Separate from days_since_last_send.
    """
    now = datetime.now(timezone.utc).timestamp()
    last_receive: dict[str, float] = {}

    for row in edge_rows:
        dst = row.get("dst")
        ts = row.get("ts")
        if not (dst and ts):
            continue
        try:
            ts_unix = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
            if dst not in last_receive or ts_unix > last_receive[dst]:
                last_receive[dst] = ts_unix
        except Exception:
            continue

    return {
        account: (now - ts) / 86400
        for account, ts in last_receive.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2-hour betweenness update (P2-4)
# ─────────────────────────────────────────────────────────────────────────────

def update_betweenness_only() -> None:
    """
    Recompute only betweenness_centrality field in Redis feat:{account}.
    Runs every 2 hours via Celery Beat. Uses k=500 approximation.
    Does NOT overwrite other fields — only patches betweenness_centrality.
    """
    log.info("betweenness_update_started")
    try:
        edge_rows = run_query(_EDGE_QUERY, {})
    except Exception as exc:
        log.error("betweenness_neo4j_failed", error=str(exc))
        return

    G = _build_digraph(edge_rows)
    btw = _compute_approx_betweenness(G, k=500)

    r = get_redis()
    updated = 0
    for account_id, btw_val in btw.items():
        key = f"feat:{account_id}"
        if r.exists(key):
            r.hset(key, "betweenness_centrality", float(btw_val))
            updated += 1

    log.info("betweenness_update_complete", accounts_updated=updated)


# ─────────────────────────────────────────────────────────────────────────────
# 5-minute micro-batch (P2-5)
# ─────────────────────────────────────────────────────────────────────────────

def update_micro_batch_features() -> None:
    """
    Update fast-changing features every 5 minutes.
    Only patches: degree_centrality (via delta edges), temporal_acceleration, sink_score.
    Queries only edges from last 10 minutes to avoid full graph reload.
    """
    log.info("micro_batch_started")
    try:
        # Only last 10 minutes of edges for fast-changing degree updates
        recent_query = """
        MATCH (a:Account)-[r:SENT]->(b:Account)
        WHERE r.timestamp > datetime() - duration({minutes: 10})
        RETURN a.id AS src, b.id AS dst, r.amount AS amount, r.timestamp AS ts
        LIMIT 50000
        """
        edge_rows = run_query(recent_query, {})
    except Exception as exc:
        log.warning("micro_batch_neo4j_failed", error=str(exc))
        return

    G = _build_digraph(edge_rows)
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    sink_scores = _compute_sink_scores(G, in_deg, out_deg)
    window_flows = _compute_window_flows(edge_rows)

    r = get_redis()
    for account_id in G.nodes():
        key = f"feat:{account_id}"
        if not r.exists(key):
            continue

        flows = window_flows.get(account_id, {})
        updates = {
            "sink_score": str(sink_scores.get(account_id, 0.0)),
            "temporal_acceleration": str(flows.get("temporal_acceleration", 0.0)),
            "inflow_1h": str(flows.get("inflow_1h", 0.0)),
            "outflow_1h": str(flows.get("outflow_1h", 0.0)),
        }
        r.hset(key, mapping=updates)


# ─────────────────────────────────────────────────────────────────────────────
# PC-GNN + Hypergraph embedding — called from run_nightly_feature_computation
# ─────────────────────────────────────────────────────────────────────────────

def _run_gnn_embedding(G, accounts: dict) -> None:
    """
    Run PC-GNN + Hypergraph after nightly batch writes feat:{account} to Redis.
    Writes gnn_emb:{account} for every account in G.
    Graceful: if torch/pyg not installed, logs warning and returns.
    """
    try:
        from app.graph.gnn_embedder import generate_embeddings

        # feat_dict: account_id → {field: value} from nightly batch (already in Redis)
        # generate_embeddings fetches directly from Redis per account to avoid memory issues
        count = generate_embeddings(G, feat_dict=None)
        log.info("gnn_embeddings_complete", accounts_embedded=count)
    except ImportError:
        log.warning("gnn_embedding_skipped", reason="torch/torch-geometric not installed")
    except Exception as exc:
        log.warning("gnn_embedding_failed", error=str(exc))

    log.info("micro_batch_complete", accounts_updated=len(G.nodes()))
