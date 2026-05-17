"""
Pre-populate Redis with dummy graph feature cache for demo accounts.
Lets Tier 3 feature builder find cached values without running the full nightly batch.

Keys match EXACTLY what feature_builder.py reads (graph_feature_names list).
Run: python scripts/seed_redis.py
Requires: REDIS_URL set in .env
"""
from __future__ import annotations
import json
import sys
import random
from pathlib import Path

PROD_KEYWORDS = ("prod", "production", "bling_prod", "live")

random.seed(42)


def _check_not_prod(url: str) -> None:
    if any(kw in url.lower() for kw in PROD_KEYWORDS):
        print("ERROR: Refusing to seed — REDIS_URL looks like a production instance.")
        sys.exit(1)


def _fraud_features() -> dict:
    """High-risk graph features — match feature_builder.py graph_feature_names exactly."""
    return {
        # Graph topology — fraud-indicator values
        "degree_centrality": round(random.uniform(0.02, 0.08), 4),
        "betweenness_centrality": round(random.uniform(0.001, 0.01), 4),
        "clustering_coefficient": round(random.uniform(0.0, 0.05), 4),
        "pagerank_fraud_seeded": round(random.uniform(0.5, 0.9), 4),
        "community_id": float(random.randint(1, 10)),
        "community_fraud_ratio": round(random.uniform(0.6, 0.95), 4),
        "shortest_path_to_fraud": float(random.randint(0, 1)),
        "cycle_membership": 1.0,
        "sink_score": round(random.uniform(0.65, 0.95), 4),
        "bipartite_score": round(random.uniform(0.55, 0.90), 4),
        "fan_out_ratio": round(random.uniform(0.05, 0.2), 4),
        "temporal_acceleration": round(random.uniform(3.0, 10.0), 4),
        "cash_mule_sink_score": round(random.uniform(0.6, 0.95), 4),
        "bridge_node_probability": round(random.uniform(0.5, 0.85), 4),
        "dormancy_reactivation_flag": 1.0,
        # Account features
        "account_age_days": float(random.randint(30, 120)),
        "kyc_completeness_score": round(random.uniform(0.3, 0.6), 4),
        # Transaction history (pattern-based)
        "txn_count_30d": float(random.randint(30, 100)),
        "txn_count_90d": float(random.randint(50, 200)),
        "txn_count_all": float(random.randint(80, 300)),
        "avg_txn_amount_30d": round(random.uniform(80000, 300000), 2),
        "distinct_counterparties_30d": float(random.randint(1, 5)),
        "channel_entropy": round(random.uniform(0.0, 0.3), 4),
        # Behavioral ratios
        "night_txn_ratio": round(random.uniform(0.6, 0.95), 4),
        "weekend_txn_ratio": round(random.uniform(0.4, 0.8), 4),
        "return_ratio": round(random.uniform(0.75, 0.98), 4),
        "amount_zscore": round(random.uniform(3.0, 8.0), 4),
        "counterparty_novelty": round(random.uniform(0.7, 1.0), 4),
        "hour_deviation": round(random.uniform(4.0, 10.0), 4),
        "channel_switch": round(random.uniform(0.5, 0.9), 4),
        "amount_series_score": round(random.uniform(0.6, 0.95), 4),
        "burst_score": round(random.uniform(0.6, 0.95), 4),
        "velocity_ratio": round(random.uniform(8.0, 30.0), 4),
        "dormancy_break": 1.0,
        "geography_switch": round(random.uniform(0.5, 1.0), 4),
    }


def _legit_features() -> dict:
    """Low-risk graph features — match feature_builder.py graph_feature_names exactly."""
    return {
        # Graph topology — normal values
        "degree_centrality": round(random.uniform(0.1, 0.5), 4),
        "betweenness_centrality": round(random.uniform(0.001, 0.02), 4),
        "clustering_coefficient": round(random.uniform(0.2, 0.6), 4),
        "pagerank_fraud_seeded": round(random.uniform(0.02, 0.1), 4),
        "community_id": float(random.randint(1, 100)),
        "community_fraud_ratio": round(random.uniform(0.0, 0.05), 4),
        "shortest_path_to_fraud": float(random.randint(3, 6)),
        "cycle_membership": 0.0,
        "sink_score": round(random.uniform(0.0, 0.15), 4),
        "bipartite_score": round(random.uniform(0.0, 0.15), 4),
        "fan_out_ratio": round(random.uniform(0.4, 0.9), 4),
        "temporal_acceleration": round(random.uniform(0.8, 1.5), 4),
        "cash_mule_sink_score": round(random.uniform(0.0, 0.1), 4),
        "bridge_node_probability": round(random.uniform(0.0, 0.1), 4),
        "dormancy_reactivation_flag": 0.0,
        # Account features
        "account_age_days": float(random.randint(365, 3650)),
        "kyc_completeness_score": round(random.uniform(0.7, 1.0), 4),
        # Transaction history
        "txn_count_30d": float(random.randint(5, 30)),
        "txn_count_90d": float(random.randint(15, 90)),
        "txn_count_all": float(random.randint(50, 500)),
        "avg_txn_amount_30d": round(random.uniform(1000, 40000), 2),
        "distinct_counterparties_30d": float(random.randint(3, 20)),
        "channel_entropy": round(random.uniform(0.5, 1.5), 4),
        # Behavioral ratios
        "night_txn_ratio": round(random.uniform(0.0, 0.15), 4),
        "weekend_txn_ratio": round(random.uniform(0.2, 0.4), 4),
        "return_ratio": round(random.uniform(0.05, 0.3), 4),
        "amount_zscore": round(random.uniform(0.0, 1.5), 4),
        "counterparty_novelty": round(random.uniform(0.1, 0.4), 4),
        "hour_deviation": round(random.uniform(0.5, 2.0), 4),
        "channel_switch": round(random.uniform(0.0, 0.15), 4),
        "amount_series_score": round(random.uniform(0.0, 0.2), 4),
        "burst_score": round(random.uniform(0.0, 0.15), 4),
        "velocity_ratio": round(random.uniform(0.8, 2.0), 4),
        "dormancy_break": 0.0,
        "geography_switch": round(random.uniform(0.0, 0.1), 4),
    }


def main() -> None:
    from app.core.config import settings
    _check_not_prod(settings.redis_url)

    import redis
    r = redis.from_url(settings.redis_url, decode_responses=True)
    try:
        r.ping()
    except Exception as exc:
        print(f"ERROR: Cannot connect to Redis at {settings.redis_url}: {exc}")
        sys.exit(1)

    account_ids: list[str] = []
    fraud_account_ids: list[str] = []

    data_file = Path("test_data.json")
    if data_file.exists():
        data = json.loads(data_file.read_text())
        seen: set[str] = set()
        for t in data["transactions"]:
            aid = t["account_id"]
            if aid not in seen:
                seen.add(aid)
                if t["label"] == 1:
                    fraud_account_ids.append(aid)
                else:
                    account_ids.append(aid)
    else:
        account_ids = [f"ACC_{i:06d}" for i in range(1, 2001)]
        fraud_account_ids = [f"ACC_FRAUD_{i:03d}" for i in range(1, 100)]
        print("test_data.json not found — seeding with synthetic account IDs")

    pipe = r.pipeline(transaction=False)
    count = 0

    for aid in account_ids:
        features = _legit_features()
        key = f"feat:{aid}"
        pipe.hset(key, mapping={k: str(v) for k, v in features.items()})
        pipe.expire(key, 26 * 3600)
        count += 1
        if count % 200 == 0:
            pipe.execute()
            print(f"  Seeded {count} legit accounts...")
            pipe = r.pipeline(transaction=False)

    for aid in fraud_account_ids:
        features = _fraud_features()
        key = f"feat:{aid}"
        pipe.hset(key, mapping={k: str(v) for k, v in features.items()})
        pipe.expire(key, 26 * 3600)
        count += 1

    pipe.execute()
    print(f"Redis seeded: {len(account_ids)} legit + {len(fraud_account_ids)} fraud accounts = {count} total")


if __name__ == "__main__":
    main()
