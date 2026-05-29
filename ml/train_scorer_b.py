"""
ml/train_scorer_b.py -- Train Scorer B (MLP) for the Committee Engine.

Scorer B is an MLPClassifier trained on [32-dim GNN embedding || 8 structural
graph features] = 40-dim input per account.

GNN embedding source priority:
  1. Redis key gnn_emb:{account}  -- preferred (post-GNN training)
  2. Redis key emb:{account}      -- Node2Vec fallback
  3. Synthetic mode               -- random 32-dim vectors if Redis is empty (WARNING logged)

Structural features (8) are read from Redis feat:{account}.
In synthetic mode: fraud embeddings cluster near [0.8]*32, legit near [0.2]*32.

Output:
  ml/models/scorer_b_v1.joblib  -- trained MLPClassifier

Run: python ml/train_scorer_b.py
"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

import structlog

log = structlog.get_logger()

random.seed(42)

# Structural features Scorer B uses from Redis feat:{account}
_STRUCTURAL = [
    "pagerank_fraud_seeded",
    "community_fraud_ratio",
    "sink_score",
    "bipartite_score",
    "betweenness_centrality",
    "burst_score",
    "clustering_coefficient",
    "temporal_acceleration",
]

_EMB_DIM = 32
_STRUCT_DIM = len(_STRUCTURAL)      # 8
_INPUT_DIM = _EMB_DIM + _STRUCT_DIM  # 40


# -- Redis embedding loader (best-effort; falls back to synthetic) -------------

def _load_from_redis(redis_url: str):
    """
    Try to load (embedding, structural) pairs from Redis.
    Returns (X list, y list) or (None, None) if Redis is unavailable/empty.
    Each embedding is 32-dim; structural is 8-dim.
    Labels derived from community_fraud_ratio > 0.5 as a proxy.
    """
    try:
        import redis as redis_lib
        r = redis_lib.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2)

        # Scan for gnn_emb:* first, fall back to emb:*
        gnn_keys = list(r.scan_iter("gnn_emb:*", count=5000))
        emb_prefix = "gnn_emb:"
        if not gnn_keys:
            gnn_keys = list(r.scan_iter("emb:*", count=5000))
            emb_prefix = "emb:"

        if not gnn_keys:
            log.warning("scorer_b.redis_no_embeddings",
                        note="No gnn_emb:* or emb:* keys found -- using synthetic mode")
            return None, None

        log.info("scorer_b.redis_found_embeddings",
                 prefix=emb_prefix, count=len(gnn_keys))

        X, y = [], []
        for key in gnn_keys[:50_000]:  # cap at 50K to bound memory
            account_id = key.replace(emb_prefix, "", 1)
            raw_emb = r.get(key)
            if raw_emb is None:
                continue

            try:
                emb = json.loads(raw_emb)
                if len(emb) != _EMB_DIM:
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            feat_raw = r.hgetall(f"feat:{account_id}")
            struct_vec = []
            for fname in _STRUCTURAL:
                try:
                    struct_vec.append(float(feat_raw.get(fname, 0.0) or 0.0))
                except (ValueError, TypeError):
                    struct_vec.append(0.0)

            X.append(emb + struct_vec)

            # Label proxy: high community_fraud_ratio + high sink_score = likely fraud
            fraud_ratio = float(feat_raw.get("community_fraud_ratio", 0.0) or 0.0)
            sink = float(feat_raw.get("sink_score", 0.0) or 0.0)
            label = 1 if (fraud_ratio > 0.5 and sink > 0.4) else 0
            y.append(label)

        if len(X) < 100:
            log.warning("scorer_b.redis_too_few_samples", count=len(X),
                        note="Falling back to synthetic mode")
            return None, None

        return X, y

    except Exception as exc:
        log.warning("scorer_b.redis_unavailable", error=str(exc),
                    note="Using synthetic mode")
        return None, None


def _generate_synthetic(n_samples: int = 5000):
    """
    Synthetic training when Redis is empty (GNN not trained yet).

    Fraud embeddings: cluster near [0.8]*32 + Gaussian noise sigma=0.15.
    Legit embeddings: cluster near [0.2]*32 + Gaussian noise sigma=0.15.
    Structural features: fraud has higher sink_score, pagerank_fraud_seeded, burst_score.
    """
    import numpy as np

    n_fraud = n_samples // 4       # 25% fraud for committee balance
    n_legit = n_samples - n_fraud

    def _gauss_clamp(base, sigma=0.15):
        v = base + float(np.random.normal(0, sigma))
        return max(0.0, min(1.0, v))

    def _make_fraud():
        emb = [_gauss_clamp(0.8, 0.15) for _ in range(_EMB_DIM)]
        struct = [
            _gauss_clamp(0.6, 0.12),   # pagerank_fraud_seeded
            _gauss_clamp(0.65, 0.12),  # community_fraud_ratio
            _gauss_clamp(0.55, 0.10),  # sink_score
            _gauss_clamp(0.5, 0.10),   # bipartite_score
            _gauss_clamp(0.4, 0.10),   # betweenness_centrality
            _gauss_clamp(0.6, 0.10),   # burst_score
            _gauss_clamp(0.3, 0.08),   # clustering_coefficient
            float(np.clip(np.random.normal(4.0, 1.5), 0, 20)),  # temporal_acceleration
        ]
        return emb + struct

    def _make_legit():
        emb = [_gauss_clamp(0.2, 0.15) for _ in range(_EMB_DIM)]
        struct = [
            _gauss_clamp(0.05, 0.04),  # pagerank_fraud_seeded
            _gauss_clamp(0.04, 0.03),  # community_fraud_ratio
            _gauss_clamp(0.08, 0.05),  # sink_score
            _gauss_clamp(0.08, 0.05),  # bipartite_score
            _gauss_clamp(0.01, 0.008), # betweenness_centrality
            _gauss_clamp(0.07, 0.05),  # burst_score
            _gauss_clamp(0.45, 0.12),  # clustering_coefficient
            float(np.clip(np.random.normal(1.1, 0.3), 0, 20)),   # temporal_acceleration
        ]
        return emb + struct

    X = [_make_fraud() for _ in range(n_fraud)] + [_make_legit() for _ in range(n_legit)]
    y = [1] * n_fraud + [0] * n_legit
    return X, y


def main() -> None:
    try:
        import numpy as np
        import joblib
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import average_precision_score, accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        log.error("scorer_b.import_error", error=str(exc))
        sys.exit(1)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    mode = "redis"

    X_raw, y_raw = _load_from_redis(redis_url)

    if X_raw is None:
        log.warning("scorer_b.synthetic_mode",
                    reason="Redis empty or unavailable",
                    note="Results will improve once GNN embeddings are trained")
        print("WARNING: Scorer B running in SYNTHETIC mode. "
              "Train GNN embeddings and re-run for production-quality model.")
        mode = "synthetic"
        X_raw, y_raw = _generate_synthetic(n_samples=5000)

    X = np.array(X_raw, dtype=np.float32)
    y = np.array(y_raw, dtype=np.int32)

    assert X.shape[1] == _INPUT_DIM, (
        f"Expected {_INPUT_DIM} input dims, got {X.shape[1]}"
    )

    log.info("scorer_b.dataset_ready",
             mode=mode, shape=list(X.shape),
             fraud_count=int(y.sum()), legit_count=int((y == 0).sum()))

    # Standard-scale before MLP (MLP is sensitive to feature scale)
    scaler = StandardScaler()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # -- MLP -------------------------------------------------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,        # holds out 10% of train for val stopping
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )

    log.info("scorer_b.training_mlp", input_dim=_INPUT_DIM, layers=(64, 32))
    mlp.fit(X_train_sc, y_train)

    # -- Save model + scaler together in a pipeline ----------------------------
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("scaler", scaler), ("mlp", mlp)])

    out_path = MODELS_DIR / "scorer_b_v1.joblib"
    joblib.dump(pipe, str(out_path))
    log.info("scorer_b.saved", path=str(out_path))

    # -- Evaluation ------------------------------------------------------------
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    pr_auc = average_precision_score(y_val, y_proba)

    fraud_mask = y_val == 1
    legit_mask = y_val == 0

    print(f"\n{chr(8212)*60}")
    print("Scorer B -- Training Summary")
    print(f"{chr(8212)*60}")
    print(f"Mode:                    {mode.upper()}")
    print(f"Input dims:              {_INPUT_DIM} (emb={_EMB_DIM} + struct={_STRUCT_DIM})")
    print(f"Train samples:           {len(X_train)} ({int(y_train.sum())} fraud)")
    print(f"Val samples:             {len(X_val)}")
    print(f"Accuracy (val):          {acc:.4f}")
    print(f"PR-AUC (val):            {pr_auc:.4f}")
    if fraud_mask.sum() > 0:
        print(f"Fraud mean score:        {y_proba[fraud_mask].mean():.3f}")
    if legit_mask.sum() > 0:
        print(f"Legit mean score:        {y_proba[legit_mask].mean():.3f}")
    print(f"Model saved:             {out_path}")
    print(f"{chr(8212)*60}")
    if mode == "synthetic":
        print("RERUN after GNN embeddings are populated in Redis for production quality.")


if __name__ == "__main__":
    main()
