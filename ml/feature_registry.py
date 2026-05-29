"""
ml/feature_registry.py — Single source of truth for feature names and order (P2-*).

INVARIANT: Both train.py and feature_builder.py MUST import FEATURE_NAMES from here.
           Never hardcode a feature list in any other file.
           Run tests/test_ml/test_feature_registry.py after every change.

Feature sources:
  - Graph (Redis feat:{account}): 35 pre-computed, updated nightly by Leiden batch
  - Real-time tabular (PostgreSQL): 24 computed per-request in feature_builder.py
  - New Phase 2 graph: 10 additional features from Leiden + temporal graph
  - Node2Vec embeddings (Phase 2): 32 dimensions from emb:{account}
  - Phase 3 new real-time: ~6 features (added in P3-* tasks)

Phase 2 complete = 35 + 24 + 10 = 69 core features (+ 32 embeddings = 101 if Node2Vec deployed).
Phase 4 target after all additions: ~107 features.

NEVER reorder existing features — XGBoost assigns by position.
Only APPEND new features at the end of each section.
"""

# ── Section 1: Pre-computed graph features (Redis feat:{account}) ─────────────
# Aligned to nightly_batch.py field names after P2-1 Leiden rewrite.
# Field names here MUST match exactly what nightly_batch.py writes.
GRAPH_FEATURES = [
    # Centrality
    "degree_centrality",
    "betweenness_centrality",
    "clustering_coefficient",
    "pagerank_fraud_seeded",
    # Community (Leiden — P2-1)
    "community_id",
    "community_fraud_ratio",
    "community_size",
    # Fraud proximity
    "shortest_path_to_fraud",
    "cycle_membership",
    "sink_score",
    # Structural
    "bipartite_score",
    "fan_out_ratio",
    "temporal_acceleration",
    # Mule / cash patterns
    "cash_mule_sink_score",
    "bridge_node_probability",
    "dormancy_reactivation_flag",
    # Account context (from accounts table, cached at batch time)
    "account_age_days",
    "kyc_completeness_score",
    # Historical transaction stats (from PostgreSQL, cached at batch time)
    "txn_count_30d",
    "txn_count_90d",
    "txn_count_all",
    "avg_txn_amount_30d",
    "distinct_counterparties_30d",
    "channel_entropy",
    # Behavioral ratios
    "night_txn_ratio",
    "weekend_txn_ratio",
    "return_ratio",
    # Anomaly signals
    "amount_zscore",
    "counterparty_novelty",
    "hour_deviation",
    # Activity shifts
    "channel_switch",
    "amount_series_score",
    # ZSET-backed velocity (P0-1) — updated via increment_velocity()
    "burst_score",
    "velocity_ratio",
    # Dormancy + geography
    "dormancy_break",
    "geography_switch",
]

# ── Section 2: Real-time tabular features (computed per request) ──────────────
REALTIME_FEATURES = [
    # Transaction core
    "txn_amount",
    "txn_amount_log",
    "txn_amount_rounded",
    # Channel encoding
    "channel_upi",
    "channel_imps",
    "channel_rtgs",
    "channel_neft",
    # Temporal
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "is_festival_period",
    # Threshold proximity
    "amount_vs_threshold_50000",
    "amount_vs_threshold_100000",
    "amount_vs_threshold_1000000",
    # Payee signals
    "payee_vpa_age_days",
    # Recent velocity (PostgreSQL windows)
    "txn_count_last_1h",
    "txn_count_last_24h",
    "txn_count_last_7d",
    "txn_volume_last_1h",
    "txn_volume_last_24h",
    "distinct_payees_24h",
    # Payee risk signals
    "payee_in_alert_log",
    "payee_shared_alert_count",
]

# ── Section 3: Phase 2 new graph features (added by Leiden batch rewrite) ─────
PHASE2_GRAPH_FEATURES = [
    # Temporal multi-hop windows (P2-8)
    "inflow_1h",
    "inflow_6h",
    "inflow_24h",
    "inflow_7d",
    "outflow_1h",
    "outflow_6h",
    "outflow_24h",
    "outflow_7d",
    # Send/receive split (P2-9)
    "days_since_last_receive",
    # Staleness penalty (P2-7)
    "graph_staleness_hours",
]

# ── Section 4: Node2Vec embeddings (Phase 2, P2-6) ────────────────────────────
# 32 dimensions — deployed after Node2Vec training run completes
NODE2VEC_FEATURES = [f"emb_{i}" for i in range(32)]

# ── Section 5: Phase 3 new real-time features (added in P3-* tasks) ──────────
PHASE3_FEATURES = [
    "micro_test_payment",       # P3-5: suspicious sub-₹1 test payment flag
    "benford_deviation",        # P3-6: leading-digit Benford deviation
    "fan_in_sender_zscore",     # P3-9: z-score of incoming sender count
]


# ── Assembled feature sets ─────────────────────────────────────────────────────
# FEATURE_NAMES_V1: original 59-feature baseline (train.py v1.0)
FEATURE_NAMES_V1 = GRAPH_FEATURES[:35] + REALTIME_FEATURES

# FEATURE_NAMES_V2: Phase 2 complete (pre-Node2Vec)
FEATURE_NAMES_V2 = GRAPH_FEATURES + REALTIME_FEATURES + PHASE2_GRAPH_FEATURES

# FEATURE_NAMES_V3: Phase 2 + Node2Vec embeddings
FEATURE_NAMES_V3 = FEATURE_NAMES_V2 + NODE2VEC_FEATURES

# FEATURE_NAMES_V4: Phase 3 additions
FEATURE_NAMES_V4 = FEATURE_NAMES_V3 + PHASE3_FEATURES

# ── Section 6: UPI session features (Phase 1 committee — Scorer A) ───────────
# All 8 may be NaN initially: existing transaction schema may not capture them.
# Scorer A sets missing_flag=True when >20% are NaN (>1 of 8).
# Train scorer_a_v1 on FEATURE_NAMES_V5 once UPI enrichment is wired.
# NEVER change this list — Scorer A model position assignments depend on order.
UPI_SESSION_FEATURES = [
    "upi_collect_request",       # 1 if pull/collect payment (payer initiated)
    "upi_intent_flag",           # 1 if QR/intent vs VPA direct entry
    "payee_vpa_verified",        # 1 if NPCI name lookup passed
    "upi_app_type",              # encoded [0,1]: GPay=0, PhonePe=0.25, Paytm=0.5, BharatPe=0.75, other=1
    "upi_deregistration_flag",   # 1 if device UPI registration changed last 7d
    "upi_pin_attempts_session",  # PIN attempts in session, capped at 5 (NaN if unavailable)
    "upi_session_id_hash",       # 1 if same device session has another recent transaction
    "session_amount_ratio",      # this txn amount / total session amount
]

# FEATURE_NAMES_V5: adds UPI session features for Scorer A training
FEATURE_NAMES_V5 = FEATURE_NAMES_V4 + UPI_SESSION_FEATURES

# Default: what feature_builder.py and train.py use right now
# Update this after each Phase that adds features AND retrains the model.
# Do NOT set FEATURE_NAMES = FEATURE_NAMES_V5 until scorer_a_v1 is trained on V5.
FEATURE_NAMES = FEATURE_NAMES_V2
