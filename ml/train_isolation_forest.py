"""
ml/train_isolation_forest.py

Train Isolation Forest on LEGITIMATE transactions only.
Fraud examples are explicitly excluded — the model learns what "normal" looks like
so that structurally novel transactions (potential new fraud patterns) score as anomalous.

Run: python ml/train_isolation_forest.py
Output: ml/models/isolation_forest_v1.joblib

Requires: POSTGRES_URL env var pointing to a populated graph_features_cache table.
If the DB is not available, falls back to synthetic legitimate data for demo purposes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import structlog

log = structlog.get_logger()

# 17 structural graph features only.
# Time and amount features deliberately excluded — they vary legitimately
# (night transactions, large amounts) and would cause false positives.
# These 17 describe HOW an account is positioned in the graph, not WHAT it does.
ISOLATION_FOREST_FEATURES = [
    "pagerank_fraud_seeded",
    "betweenness_centrality",
    "clustering_coefficient",
    "degree_centrality",
    "fan_out_ratio",
    "sink_score",
    "bipartite_score",
    "community_fraud_ratio",
    "shortest_path_to_fraud",
    "return_ratio",
    "burst_score",
    "velocity_ratio",
    "channel_entropy",
    "counterparty_novelty",
    "bridge_node_probability",
    "temporal_acceleration",
    "dormancy_reactivation_flag",
]

# Threshold used at inference time.
# decision_function returns more negative = more anomalous.
# At -0.20 with contamination=0.001: expect ~1-3 flags per 1000 transactions.
# Never lower than -0.15 (floods developer queue).
# Never higher than -0.25 (misses genuine novel patterns).
NOVELTY_THRESHOLD = -0.20

MODEL_PATH = Path("ml/models/isolation_forest_v1.joblib")


def _synthetic_legit_data(n: int = 2000) -> np.ndarray:
    """
    Generate synthetic legitimate account feature vectors for demo/fallback.
    Legitimate accounts have low fraud proximity scores and normal velocity.
    """
    rng = np.random.default_rng(42)
    X = np.zeros((n, len(ISOLATION_FOREST_FEATURES)), dtype=np.float32)

    feat_idx = {f: i for i, f in enumerate(ISOLATION_FOREST_FEATURES)}

    # Legit: low fraud proximity, moderate activity, stable structure
    X[:, feat_idx["pagerank_fraud_seeded"]] = rng.beta(1, 9, n)          # mostly low
    X[:, feat_idx["betweenness_centrality"]] = rng.beta(1, 10, n)
    X[:, feat_idx["clustering_coefficient"]] = rng.beta(5, 2, n)         # high clustering
    X[:, feat_idx["degree_centrality"]] = rng.beta(2, 5, n)
    X[:, feat_idx["fan_out_ratio"]] = rng.beta(5, 2, n)                  # legit sends money out
    X[:, feat_idx["sink_score"]] = rng.beta(1, 8, n)                     # low sink
    X[:, feat_idx["bipartite_score"]] = rng.beta(1, 8, n)                # low fan-in density
    X[:, feat_idx["community_fraud_ratio"]] = rng.beta(1, 9, n)          # clean community
    X[:, feat_idx["shortest_path_to_fraud"]] = rng.uniform(3, 5, n)      # far from fraud
    X[:, feat_idx["return_ratio"]] = rng.beta(2, 8, n)                   # low forwarding
    X[:, feat_idx["burst_score"]] = rng.beta(1, 6, n)                    # no velocity burst
    X[:, feat_idx["velocity_ratio"]] = rng.uniform(0, 3, n)              # normal velocity
    X[:, feat_idx["channel_entropy"]] = rng.beta(3, 3, n)                # moderate diversity
    X[:, feat_idx["counterparty_novelty"]] = rng.beta(2, 8, n)           # known payees
    X[:, feat_idx["bridge_node_probability"]] = rng.beta(1, 10, n)       # not a bridge
    X[:, feat_idx["temporal_acceleration"]] = rng.uniform(0, 2, n)
    X[:, feat_idx["dormancy_reactivation_flag"]] = rng.choice([0.0, 1.0], n, p=[0.97, 0.03])

    return X.astype(np.float32)


def _load_from_db(db_url: str) -> np.ndarray | None:
    """Load legitimate account graph features from PostgreSQL."""
    try:
        import sqlalchemy as sa
        import pandas as pd

        engine = sa.create_engine(db_url)
        query = """
            SELECT gfc.*
            FROM graph_features_cache gfc
            WHERE gfc.account_id NOT IN (
                SELECT DISTINCT transaction_account_id
                FROM alerts
                WHERE status = 'CONFIRMED_FRAUD'
            )
            AND gfc.computed_at > NOW() - INTERVAL '48 hours'
        """
        df = pd.read_sql(query, engine)
        if len(df) < 50:
            log.warning("insufficient_db_data", count=len(df))
            return None

        # Extract only the 17 features, fill missing with 0
        available = [f for f in ISOLATION_FOREST_FEATURES if f in df.columns]
        missing = [f for f in ISOLATION_FOREST_FEATURES if f not in df.columns]
        if missing:
            log.info("features_not_in_db", missing=missing, action="filling_with_zero")

        X = np.zeros((len(df), len(ISOLATION_FOREST_FEATURES)), dtype=np.float32)
        for i, feat in enumerate(ISOLATION_FOREST_FEATURES):
            if feat in df.columns:
                X[:, i] = df[feat].fillna(0).values.astype(np.float32)

        log.info("loaded_legit_data_from_db", rows=len(df))
        return X

    except Exception as e:
        log.warning("db_load_failed", error=str(e), action="using_synthetic_fallback")
        return None


def train(db_url: str | None = None) -> bool:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # Prefer real DB data; fall back to synthetic for demo
    X = None
    if db_url:
        X = _load_from_db(db_url)

    if X is None:
        log.info("using_synthetic_legit_data", n=2000)
        X = _synthetic_legit_data(2000)

    # Scale — prevents high-magnitude graph features from dominating
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination=0.001: expect 0.1% of training data to be anomalous
    # n_estimators=200: stable scores across runs
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.001,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_scaled)

    # Sanity check on training data
    scores = model.decision_function(X_scaled)
    flagged = int((scores < NOVELTY_THRESHOLD).sum())
    flagged_pct = flagged / len(scores) * 100

    log.info(
        "model_trained",
        training_samples=len(X),
        n_features=len(ISOLATION_FOREST_FEATURES),
        flagged_at_threshold=flagged,
        flagged_pct=f"{flagged_pct:.3f}%",
    )

    if flagged_pct > 1.0:
        log.warning(
            "threshold_may_be_too_aggressive",
            flagged_pct=flagged_pct,
            advice="Consider tightening NOVELTY_THRESHOLD from -0.20 to -0.25",
        )

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": ISOLATION_FOREST_FEATURES,
        "threshold": NOVELTY_THRESHOLD,
        "trained_on_samples": len(X),
        "contamination": 0.001,
        "version": "v1",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    log.info("model_saved", path=str(MODEL_PATH), size_kb=MODEL_PATH.stat().st_size // 1024)
    return True


if __name__ == "__main__":
    db_url = os.environ.get("POSTGRES_URL")
    if not db_url:
        print("No POSTGRES_URL set — using synthetic legitimate data (demo mode)")

    success = train(db_url)
    if success:
        size = MODEL_PATH.stat().st_size
        print(f"✓ Isolation Forest trained → {MODEL_PATH}  ({size:,} bytes)")
        print(f"  Features: {len(ISOLATION_FOREST_FEATURES)}")
        print(f"  Threshold: {NOVELTY_THRESHOLD}")
    else:
        print("✗ Training failed — check logs above")
        sys.exit(1)
