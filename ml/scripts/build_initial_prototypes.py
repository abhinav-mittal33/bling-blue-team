"""
Offline script: Build initial FAISS prototype index for Scorer C.

Seeds the prototype vault with synthetic archetype feature vectors from the
training distribution in ml/train.py. These are not real transactions — they are
the centroid feature vectors for each named fraud archetype, hand-tuned to be
near the cluster centres discovered during Phase 4 XGBoost training.

Archetypes seeded (label=1 / fraud):
  rapid_layering, low_slow_mule, digital_arrest, ghost_node_cash, structuring,
  bipartite_mule_network, hawala, crypto_onramp, benami

Control samples seeded (label=0 / benign):
  salary_payment, routine_utility, festival_gifting, peer_transfer

Output:
  ml/models/prototype_faiss.index
  ml/models/prototype_meta.joblib

Run:
  python ml/scripts/build_initial_prototypes.py
  python ml/scripts/build_initial_prototypes.py --dim 69 --output-dir ml/models/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.feature_registry import FEATURE_NAMES   # noqa: E402 — must be after sys.path insert

# Feature index lookup for constructing archetype vectors
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_DIM = len(FEATURE_NAMES)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _vec(**overrides: float) -> np.ndarray:
    """Build a prototype vector. NaN for all unspecified features."""
    v = np.full(_DIM, float("nan"), dtype=np.float32)
    for feat, val in overrides.items():
        if feat in _IDX:
            v[_IDX[feat]] = val
        else:
            print(f"  WARNING: feature '{feat}' not in FEATURE_NAMES — ignored")
    return v


# ── Fraud archetypes (label=1) ─────────────────────────────────────────────────

FRAUD_ARCHETYPES: dict[str, np.ndarray] = {
    "rapid_layering": _vec(
        velocity_ratio=0.95, burst_score=0.90, txn_count_last_1h=12,
        txn_count_last_24h=28, sink_score=0.85, bipartite_score=0.80,
        pagerank_fraud_seeded=0.72, temporal_acceleration=0.88,
        payee_vpa_age_days=3, is_night=0, channel_upi=1,
        txn_amount_log=9.5, amount_zscore=3.2,
    ),
    "low_slow_mule": _vec(
        dormancy_reactivation_flag=1, dormancy_break=1, account_age_days=1200,
        txn_count_30d=1, txn_count_90d=2, velocity_ratio=0.05,
        txn_amount_log=12.0, is_night=1, hour_of_day=2,
        pagerank_fraud_seeded=0.45, sink_score=0.60,
        amount_zscore=5.8, payee_vpa_age_days=2,
    ),
    "digital_arrest": _vec(
        is_night=1, hour_of_day=3, txn_amount_log=12.9,
        payee_vpa_age_days=1, account_age_days=3650,
        amount_vs_threshold_1000000=0.95, velocity_ratio=0.10,
        kyc_completeness_score=0.95, amount_zscore=6.0,
        counterparty_novelty=0.95, channel_upi=1,
    ),
    "ghost_node_cash": _vec(
        cash_mule_sink_score=0.92, bridge_node_probability=0.88,
        sink_score=0.85, channel_imps=1, is_night=0,
        txn_amount_log=11.7, amount_zscore=4.5,
        geography_switch=1, pagerank_fraud_seeded=0.65,
        counterparty_novelty=0.80,
    ),
    "structuring": _vec(
        amount_vs_threshold_50000=0.93, amount_vs_threshold_100000=0.95,
        benford_deviation=0.75, txn_count_last_24h=5,
        txn_count_last_7d=12, velocity_ratio=0.70,
        amount_series_score=0.80, txn_amount_rounded=1,
        pagerank_fraud_seeded=0.35,
    ),
    "bipartite_mule_network": _vec(
        bipartite_score=0.92, fan_out_ratio=0.10, fan_in_sender_zscore=3.8,
        sink_score=0.88, community_fraud_ratio=0.75,
        distinct_counterparties_30d=8, txn_count_last_24h=14,
        pagerank_fraud_seeded=0.60, payee_shared_alert_count=5,
    ),
    "hawala": _vec(
        geography_switch=1, channel_switch=1, amount_zscore=3.5,
        return_ratio=0.85, txn_count_30d=40,
        pagerank_fraud_seeded=0.55, community_fraud_ratio=0.60,
        sink_score=0.70, temporal_acceleration=0.75,
        distinct_counterparties_30d=15,
    ),
    "crypto_onramp": _vec(
        txn_amount_log=12.5, amount_vs_threshold_1000000=0.90,
        payee_vpa_age_days=5, counterparty_novelty=0.90,
        velocity_ratio=0.80, txn_count_last_24h=6,
        channel_upi=1, is_night=0.5, amount_zscore=4.0,
        micro_test_payment=1,
    ),
    "benami": _vec(
        cycle_membership=1, community_fraud_ratio=0.70,
        shortest_path_to_fraud=2, bipartite_score=0.65,
        return_ratio=0.70, txn_count_30d=25,
        pagerank_fraud_seeded=0.58, sink_score=0.55,
    ),
}

# ── Benign archetypes (label=0) ────────────────────────────────────────────────

BENIGN_ARCHETYPES: dict[str, np.ndarray] = {
    "salary_payment": _vec(
        account_age_days=1800, kyc_completeness_score=0.98,
        txn_count_30d=5, txn_count_90d=15, velocity_ratio=0.05,
        payee_vpa_age_days=730, txn_amount_log=10.8,
        amount_zscore=0.3, is_night=0, is_weekend=0,
        day_of_week=2, return_ratio=0.02,
    ),
    "routine_utility": _vec(
        account_age_days=900, kyc_completeness_score=0.90,
        txn_count_30d=12, velocity_ratio=0.10,
        payee_vpa_age_days=365, txn_amount_log=8.5,
        amount_zscore=0.1, is_night=0, txn_amount_rounded=1,
    ),
    "festival_gifting": _vec(
        is_festival_period=1, txn_count_last_24h=12,
        velocity_ratio=0.50, txn_amount_log=7.6,
        payee_vpa_age_days=200, amount_zscore=1.2,
        kyc_completeness_score=0.88, account_age_days=1200,
    ),
    "peer_transfer": _vec(
        account_age_days=600, txn_count_30d=8,
        payee_vpa_age_days=150, txn_amount_log=9.0,
        amount_zscore=0.5, velocity_ratio=0.08,
        counterparty_novelty=0.15, is_night=0,
    ),
}


def build_index(output_dir: str) -> None:
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    try:
        import joblib
    except ImportError:
        print("ERROR: joblib not installed. Run: pip install joblib")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []

    for name, vec in FRAUD_ARCHETYPES.items():
        vectors.append(vec)
        labels.append(1)
        names.append(name)

    for name, vec in BENIGN_ARCHETYPES.items():
        vectors.append(vec)
        labels.append(0)
        names.append(name)

    # Replace NaN with 0 before FAISS indexing
    matrix = np.array(vectors, dtype=np.float32)
    matrix = np.where(np.isnan(matrix), 0.0, matrix)

    index = faiss.IndexFlatL2(_DIM)
    index.add(matrix)

    faiss_path = os.path.join(output_dir, "prototype_faiss.index")
    meta_path = os.path.join(output_dir, "prototype_meta.joblib")

    faiss.write_index(index, faiss_path)
    joblib.dump({"labels": labels, "names": names}, meta_path)

    print(f"Prototype vault built:")
    print(f"  {len(FRAUD_ARCHETYPES)} fraud archetypes + {len(BENIGN_ARCHETYPES)} benign")
    print(f"  Dimension: {_DIM}")
    print(f"  FAISS index: {faiss_path}")
    print(f"  Metadata:   {meta_path}")
    print("\nVerify Scorer C loads:")
    print("  python -c \"from app.detection.tier3.prototype_vault import prototype_vault; "
          "print(prototype_vault.load('ml/models/prototype_faiss.index', "
          "'ml/models/prototype_meta.joblib'))\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build initial FAISS prototype index for Scorer C")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    build_index(args.output_dir)
