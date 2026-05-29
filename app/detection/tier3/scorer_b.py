"""
Tier 3 Committee — Scorer B: Graph Embedding Scorer

Combines Node2Vec account embeddings (emb:{account} in Redis, 32-dim) with
8 structural context features already fetched by feature_builder.py.

Key design: cached_graph_features is passed in from committee_scorer.py so
no extra Redis round-trip is needed — feature_builder already fetched feat:{account}.

missing_flag=True when Node2Vec embedding is absent from Redis (nightly job not
yet run for this account, or account is brand-new).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import structlog

from app.detection.tier3.committee_types import ScorerOutput
from app.core.config import settings
from app.utils.redis_client import get_redis

logger = structlog.get_logger()

_scorer_b_model: Optional[Any] = None
_load_attempted: bool = False

# 8 structural features pulled from cached feat:{account} hash.
# Must match field names written by nightly_batch.py (feature_registry.py is authoritative).
_STRUCTURAL_FEATURES = [
    "pagerank_fraud_seeded",
    "community_fraud_ratio",
    "sink_score",
    "bipartite_score",
    "betweenness_centrality",
    "burst_score",
    "clustering_coefficient",
    "temporal_acceleration",
]

_SCORER_B_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", settings.scorer_b_model_path)
)


def _load_scorer_b() -> None:
    """Lazy-load once. Thread-safe via GIL."""
    global _scorer_b_model, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    if os.path.exists(_SCORER_B_PATH):
        try:
            import joblib
            _scorer_b_model = joblib.load(_SCORER_B_PATH)
            logger.info("scorer_b_loaded", path=_SCORER_B_PATH)
        except Exception as exc:
            logger.warning("scorer_b_load_failed", path=_SCORER_B_PATH, error=str(exc))
    else:
        logger.warning("scorer_b_no_model", path=_SCORER_B_PATH)


def _fetch_embedding(account_id: str) -> tuple[Optional[np.ndarray], bool]:
    """
    Fetch account embedding from Redis.
    Priority: gnn_emb:{account} (PC-GNN) → emb:{account} (Node2Vec) → missing_flag=True.

    PC-GNN embeddings (gnn_emb:) are richer — camouflage-resistant + hypergraph enriched.
    Node2Vec (emb:) is the warm fallback when GNN hasn't run for this account yet.
    Both stored as JSON strings by their respective runners.
    """
    try:
        import json
        r = get_redis()

        # Prefer PC-GNN embedding (gnn_emb:{account}) — written by gnn_embedder.py
        raw = r.get(f"gnn_emb:{account_id}")
        embedding_source = "pcgnn"
        if raw is None:
            # Fall back to Node2Vec embedding (emb:{account}) — written by node2vec_runner.py
            raw = r.get(f"emb:{account_id}")
            embedding_source = "node2vec"

        if raw is None:
            return None, True

        vec_list = json.loads(raw)
        emb = np.array(vec_list, dtype=np.float32)
        if emb.shape[0] != settings.scorer_b_embedding_dim:
            logger.warning(
                "scorer_b_embedding_dim_mismatch",
                expected=settings.scorer_b_embedding_dim,
                got=emb.shape[0],
                source=embedding_source,
            )
            return None, True
        return emb, False
    except Exception as exc:
        logger.warning("scorer_b_embedding_fetch_failed", error=str(exc))
        return None, True


def _build_structural_context(cached_features: dict) -> np.ndarray:
    """
    Extract 8 structural features from the feat:{account} hash already in memory.
    NaN for any absent field — the model must handle missing gracefully.
    """
    return np.array(
        [float(cached_features.get(f, float("nan"))) for f in _STRUCTURAL_FEATURES],
        dtype=np.float32,
    )


def score(account_id: str, cached_graph_features: dict) -> ScorerOutput:
    """
    Score using Node2Vec embedding + structural context.

    account_id: raw account identifier — embedding fetch only; never logged.
    cached_graph_features: dict already fetched by feature_builder (no extra Redis call).
    """
    _load_scorer_b()

    emb, missing_flag = _fetch_embedding(account_id)
    structural = _build_structural_context(cached_graph_features)

    if _scorer_b_model is None or emb is None:
        return ScorerOutput.unavailable("B")

    try:
        # Input: [32-dim embedding || 8 structural] = 40-dim
        feature_vec = np.concatenate([emb, structural]).reshape(1, -1)
        prob = float(_scorer_b_model.predict_proba(feature_vec)[0, 1])
        confidence = abs(prob - 0.5) * 2.0
        return ScorerOutput(
            score=prob,
            confidence=confidence,
            missing_flag=missing_flag,
            scorer_id="B",
        )
    except Exception as exc:
        logger.warning("scorer_b_score_failed", error=str(exc))
        return ScorerOutput.unavailable("B")
