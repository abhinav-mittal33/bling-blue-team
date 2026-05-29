"""
Node2Vec 32-dimensional account embeddings (P2-6).
Runs nightly after the main Leiden batch completes.
Writes emb:{account} → 32-dim float array to Redis.

Embeddings are stored separately from feat:{account} to allow independent
staleness checks and to avoid bloating the main feature hash.
"""
from __future__ import annotations
import json
import structlog

log = structlog.get_logger()

_EMB_DIMENSIONS = 32
_WALK_LENGTH = 30
_NUM_WALKS = 10
_WORKERS = 4


def run_node2vec_embedding(G=None) -> int:
    """
    Train Node2Vec on the 30-day transaction graph and write embeddings to Redis.
    If G is None, loads fresh edges from Neo4j.
    Returns number of accounts embedded.
    """
    try:
        import networkx as nx
        from node2vec import Node2Vec
        from app.utils.redis_client import get_redis
        from app.graph.neo4j_client import run_query
        from app.graph.precompute.nightly_batch import _build_digraph, _EDGE_QUERY

        if G is None:
            edge_rows = run_query(_EDGE_QUERY, {})
            G = _build_digraph(edge_rows)

        if len(G.nodes()) < 10:
            log.warning("node2vec_skipped", reason="too_few_nodes", count=len(G.nodes()))
            return 0

        node2vec = Node2Vec(
            G.to_undirected(),
            dimensions=_EMB_DIMENSIONS,
            walk_length=_WALK_LENGTH,
            num_walks=_NUM_WALKS,
            workers=_WORKERS,
            quiet=True,
        )
        model = node2vec.fit(window=5, min_count=1, batch_words=4)

        r = get_redis()
        count = 0
        for node in G.nodes():
            if node in model.wv:
                vec = model.wv[node].tolist()
                r.set(f"emb:{node}", json.dumps(vec), ex=90000)  # ~25h TTL
                count += 1

        log.info("node2vec_complete", embeddings_written=count, dimensions=_EMB_DIMENSIONS)
        return count

    except Exception as exc:
        log.error("node2vec_failed", error=str(exc))
        return 0
