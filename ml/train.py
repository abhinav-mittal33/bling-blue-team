"""
Train XGBoost fraud detection model.
Feature names match EXACTLY what feature_builder.py produces:
  - 35 graph features (from Redis cache / nightly batch)
  - 24 real-time tabular features (59 unique total)

Training data: 100K synthetic examples, 2700 fraud across 16 archetypes.
Gaussian noise added to all features for realistic distribution overlap.

Key constraints (from spec):
  - scale_pos_weight=37: 2700 fraud / 97300 legit ≈ 1:36 ratio
  - eval_metric='aucpr': PR-AUC not ROC-AUC for imbalanced data
  - Output: ml/models/xgboost_v1.json

Run: python ml/train.py
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

random.seed(42)

FRAUD_ARCHETYPES = [
    # Original 8 Indian banking archetypes
    "rapid_layering",
    "low_slow_mule",
    "digital_arrest",
    "ghost_node_cash",
    "structuring",
    "bipartite_mule",
    "cycle_round_trip",
    "merchant_terminal",
    # Additional real-world archetypes
    "romance_scam",          # Long trust-building then large transfer to new VPA
    "pig_butchering",        # Small inflows then sudden large outflow to crypto/new VPA
    "sim_swap",              # Account takeover: geography switch + new channel + spike
    "otp_fraud",             # Rapid high-value transfers at 3am, multiple payees
    "investment_fraud",      # Multiple victims → single collector → rapid cash-out
    "salary_mule",           # Receives salary-like amounts, immediately relays out
    "cash_in_mule",          # Cash deposit → immediate UPI transfer → empty account
    "account_takeover",      # Dormant account suddenly active: new geography, high amount
]

# Must match feature_builder.py graph_feature_names list EXACTLY
GRAPH_FEATURE_NAMES = [
    "degree_centrality", "betweenness_centrality", "clustering_coefficient",
    "pagerank_fraud_seeded", "community_id", "community_fraud_ratio",
    "shortest_path_to_fraud", "cycle_membership", "sink_score",
    "bipartite_score", "fan_out_ratio", "temporal_acceleration",
    "cash_mule_sink_score", "bridge_node_probability", "dormancy_reactivation_flag",
    "account_age_days", "kyc_completeness_score",
    "txn_count_30d", "txn_count_90d", "txn_count_all",
    "avg_txn_amount_30d", "distinct_counterparties_30d", "channel_entropy",
    "night_txn_ratio", "weekend_txn_ratio", "return_ratio",
    "amount_zscore", "counterparty_novelty", "hour_deviation",
    "channel_switch", "amount_series_score", "burst_score",
    "velocity_ratio", "dormancy_break", "geography_switch",
]

# Must match real-time features feature_builder.py computes
REALTIME_FEATURE_NAMES = [
    "txn_amount", "txn_amount_log", "txn_amount_rounded",
    "channel_upi", "channel_imps", "channel_rtgs", "channel_neft",
    "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_festival_period",
    "amount_vs_threshold_50000", "amount_vs_threshold_100000", "amount_vs_threshold_1000000",
    "payee_vpa_age_days",
    "txn_count_last_1h", "txn_count_last_24h", "txn_count_last_7d",
    "txn_volume_last_1h", "txn_volume_last_24h",
    "distinct_payees_24h",
    "payee_in_alert_log", "payee_shared_alert_count",
]

ALL_FEATURE_NAMES = sorted(set(GRAPH_FEATURE_NAMES + REALTIME_FEATURE_NAMES))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _noise(v: float, sigma: float) -> float:
    """Add Gaussian noise for realistic distribution overlap."""
    return v + random.gauss(0, sigma * abs(v + 1e-6))


def _legit_graph() -> dict:
    return {
        "degree_centrality": random.uniform(0.05, 0.5),
        "betweenness_centrality": random.uniform(0.0005, 0.025),
        "clustering_coefficient": random.uniform(0.15, 0.7),
        "pagerank_fraud_seeded": random.uniform(0.01, 0.12),
        "community_id": float(random.randint(1, 200)),
        "community_fraud_ratio": random.uniform(0.0, 0.08),
        "shortest_path_to_fraud": float(random.randint(3, 7)),
        "cycle_membership": 0.0,
        "sink_score": random.uniform(0.0, 0.2),
        "bipartite_score": random.uniform(0.0, 0.18),
        "fan_out_ratio": random.uniform(0.35, 0.95),
        "temporal_acceleration": random.uniform(0.7, 1.6),
        "cash_mule_sink_score": random.uniform(0.0, 0.12),
        "bridge_node_probability": random.uniform(0.0, 0.12),
        "dormancy_reactivation_flag": 0.0,
        "account_age_days": float(random.randint(180, 4000)),
        "kyc_completeness_score": random.uniform(0.65, 1.0),
        "txn_count_30d": float(random.randint(3, 40)),
        "txn_count_90d": float(random.randint(10, 120)),
        "txn_count_all": float(random.randint(30, 600)),
        "avg_txn_amount_30d": random.uniform(500, 50000),
        "distinct_counterparties_30d": float(random.randint(2, 25)),
        "channel_entropy": random.uniform(0.4, 1.8),
        "night_txn_ratio": random.uniform(0.0, 0.18),
        "weekend_txn_ratio": random.uniform(0.15, 0.45),
        "return_ratio": random.uniform(0.02, 0.35),
        "amount_zscore": random.uniform(0.0, 2.0),
        "counterparty_novelty": random.uniform(0.05, 0.45),
        "hour_deviation": random.uniform(0.3, 2.5),
        "channel_switch": random.uniform(0.0, 0.18),
        "amount_series_score": random.uniform(0.0, 0.25),
        "burst_score": random.uniform(0.0, 0.18),
        "velocity_ratio": random.uniform(0.5, 2.5),
        "dormancy_break": 0.0,
        "geography_switch": random.uniform(0.0, 0.12),
    }


def _legit_realtime() -> dict:
    amount = random.uniform(200, 80000)
    hour = random.randint(7, 22)
    dow = random.randint(0, 6)
    return {
        "txn_amount": amount,
        "txn_amount_log": math.log1p(amount),
        "txn_amount_rounded": 1.0 if amount == round(amount, -3) else 0.0,
        "channel_upi": 1.0, "channel_imps": 0.0, "channel_rtgs": 0.0, "channel_neft": 0.0,
        "hour_of_day": float(hour),
        "day_of_week": float(dow),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
        "is_night": 0.0,
        "is_festival_period": 0.0,
        "amount_vs_threshold_50000": amount / 50000,
        "amount_vs_threshold_100000": amount / 100000,
        "amount_vs_threshold_1000000": amount / 1000000,
        "payee_vpa_age_days": float(random.randint(60, 1000)),
        "txn_count_last_1h": float(random.randint(0, 3)),
        "txn_count_last_24h": float(random.randint(1, 8)),
        "txn_count_last_7d": float(random.randint(2, 25)),
        "txn_volume_last_1h": random.uniform(0, 8000),
        "txn_volume_last_24h": random.uniform(200, 80000),
        "distinct_payees_24h": float(random.randint(1, 6)),
        "payee_in_alert_log": 0.0,
        "payee_shared_alert_count": 0.0,
    }


def _legit_features() -> dict:
    f = _legit_graph()
    f.update(_legit_realtime())
    return f


def _fraud_base_overrides() -> dict:
    """Base graph overrides applied to ALL fraud archetypes before archetype-specific tuning."""
    return {
        "pagerank_fraud_seeded": random.uniform(0.35, 0.9),
        "community_fraud_ratio": random.uniform(0.45, 0.95),
        "shortest_path_to_fraud": float(random.randint(0, 2)),
        "burst_score": random.uniform(0.45, 0.95),
        "velocity_ratio": random.uniform(4.0, 25.0),
        "amount_zscore": random.uniform(2.5, 10.0),
        "counterparty_novelty": random.uniform(0.55, 1.0),
        "payee_in_alert_log": random.choice([0.0, 1.0]),
        "payee_shared_alert_count": float(random.randint(0, 5)),
    }


def _fraud_features(archetype: str) -> dict:
    f = _legit_features()
    f.update(_fraud_base_overrides())

    if archetype in ("rapid_layering", "cycle_round_trip"):
        f.update({
            "cycle_membership": 1.0,
            "temporal_acceleration": random.uniform(5.0, 18.0),
            "night_txn_ratio": random.uniform(0.65, 1.0),
            "hour_deviation": random.uniform(5.0, 12.0),
            "is_night": 1.0,
            "hour_of_day": float(random.randint(0, 4)),
            "txn_count_last_1h": float(random.randint(5, 20)),
            "velocity_ratio": random.uniform(10.0, 40.0),
            "payee_vpa_age_days": float(random.randint(1, 6)),
            "txn_amount": random.uniform(75000, 99500),
            "account_age_days": float(random.randint(25, 100)),
            "return_ratio": random.uniform(0.7, 0.98),
        })

    elif archetype == "low_slow_mule":
        f.update({
            "sink_score": random.uniform(0.65, 0.95),
            "cash_mule_sink_score": random.uniform(0.6, 0.92),
            "return_ratio": random.uniform(0.78, 0.99),
            "dormancy_reactivation_flag": 1.0,
            "dormancy_break": 1.0,
            "is_night": 1.0,
            "night_txn_ratio": random.uniform(0.65, 0.95),
            "txn_amount": random.uniform(130000, 220000),
            "account_age_days": float(random.randint(25, 70)),
            "amount_zscore": random.uniform(5.0, 12.0),
            "avg_txn_amount_30d": random.uniform(100000, 300000),
        })

    elif archetype == "digital_arrest":
        f.update({
            "bridge_node_probability": random.uniform(0.45, 0.88),
            "dormancy_reactivation_flag": 1.0,
            "dormancy_break": 1.0,
            "is_night": 1.0,
            "night_txn_ratio": random.uniform(0.55, 0.9),
            "payee_vpa_age_days": float(random.randint(1, 4)),
            "txn_amount": random.uniform(150000, 600000),
            "geography_switch": random.uniform(0.65, 1.0),
            "channel_switch": random.uniform(0.55, 0.92),
            "amount_zscore": random.uniform(5.0, 15.0),
            "account_age_days": float(random.randint(365, 4000)),
            "kyc_completeness_score": random.uniform(0.25, 0.6),
        })

    elif archetype == "ghost_node_cash":
        f.update({
            "sink_score": random.uniform(0.62, 0.96),
            "cash_mule_sink_score": random.uniform(0.65, 0.96),
            "dormancy_reactivation_flag": 1.0,
            "dormancy_break": 1.0,
            "geography_switch": random.uniform(0.65, 1.0),
            "return_ratio": random.uniform(0.82, 0.99),
            "txn_amount": random.uniform(90000, 140000),
            "distinct_counterparties_30d": float(random.randint(1, 4)),
            "channel_entropy": random.uniform(0.0, 0.2),
        })

    elif archetype == "structuring":
        amount = random.uniform(88000, 99000)
        f.update({
            "amount_series_score": random.uniform(0.65, 0.96),
            "bridge_node_probability": random.uniform(0.35, 0.72),
            "txn_amount": amount,
            "txn_count_last_24h": float(random.randint(4, 10)),
            "distinct_payees_24h": float(random.randint(3, 8)),
            "burst_score": random.uniform(0.55, 0.9),
            "velocity_ratio": random.uniform(5.0, 15.0),
        })

    elif archetype == "bipartite_mule":
        f.update({
            "bipartite_score": random.uniform(0.62, 0.96),
            "sink_score": random.uniform(0.5, 0.88),
            "cash_mule_sink_score": random.uniform(0.45, 0.82),
            "degree_centrality": random.uniform(0.18, 0.55),
            "community_fraud_ratio": random.uniform(0.55, 0.92),
            "distinct_counterparties_30d": float(random.randint(8, 35)),
            "account_age_days": float(random.randint(25, 130)),
            "fan_out_ratio": random.uniform(0.04, 0.18),
        })

    elif archetype == "merchant_terminal":
        f.update({
            "burst_score": random.uniform(0.75, 1.0),
            "velocity_ratio": random.uniform(18.0, 70.0),
            "txn_count_last_1h": float(random.randint(25, 120)),
            "distinct_payees_24h": float(random.randint(15, 90)),
            "txn_amount": random.uniform(500, 6000),
            "temporal_acceleration": random.uniform(8.0, 35.0),
            "txn_count_30d": float(random.randint(200, 1000)),
        })

    elif archetype == "romance_scam":
        # Victim sends large amount to new VPA; relationship built over weeks
        f.update({
            "payee_vpa_age_days": float(random.randint(1, 14)),
            "counterparty_novelty": random.uniform(0.85, 1.0),
            "geography_switch": random.uniform(0.6, 1.0),
            "txn_amount": random.uniform(50000, 500000),
            "amount_zscore": random.uniform(4.0, 12.0),
            "is_night": 1.0 if random.random() < 0.5 else 0.0,
            "bridge_node_probability": random.uniform(0.4, 0.8),
            "distinct_payees_24h": float(random.randint(1, 2)),
            "account_age_days": float(random.randint(365, 5000)),
        })

    elif archetype == "pig_butchering":
        # Small inflows from many → single large outflow to new crypto/VPA
        f.update({
            "sink_score": random.uniform(0.55, 0.88),
            "return_ratio": random.uniform(0.0, 0.1),
            "txn_amount": random.uniform(200000, 1000000),
            "payee_vpa_age_days": float(random.randint(1, 7)),
            "geography_switch": random.uniform(0.7, 1.0),
            "amount_zscore": random.uniform(6.0, 15.0),
            "bipartite_score": random.uniform(0.45, 0.85),
            "community_fraud_ratio": random.uniform(0.5, 0.9),
            "txn_count_last_24h": float(random.randint(1, 3)),
        })

    elif archetype == "sim_swap":
        # Account taken over: new geography, new channel, sudden spike
        f.update({
            "dormancy_reactivation_flag": 1.0,
            "dormancy_break": 1.0,
            "geography_switch": random.uniform(0.8, 1.0),
            "channel_switch": random.uniform(0.7, 1.0),
            "counterparty_novelty": random.uniform(0.8, 1.0),
            "burst_score": random.uniform(0.7, 1.0),
            "txn_amount": random.uniform(80000, 300000),
            "payee_vpa_age_days": float(random.randint(1, 5)),
            "is_night": 1.0,
            "hour_of_day": float(random.randint(0, 4)),
            "txn_count_last_1h": float(random.randint(3, 8)),
        })

    elif archetype == "otp_fraud":
        # Victim tricked into sharing OTP; rapid transfers 3am
        f.update({
            "is_night": 1.0,
            "hour_of_day": float(random.randint(0, 4)),
            "night_txn_ratio": random.uniform(0.6, 1.0),
            "burst_score": random.uniform(0.65, 0.95),
            "velocity_ratio": random.uniform(8.0, 25.0),
            "txn_count_last_1h": float(random.randint(3, 10)),
            "distinct_payees_24h": float(random.randint(3, 10)),
            "txn_amount": random.uniform(30000, 200000),
            "payee_vpa_age_days": float(random.randint(1, 10)),
            "counterparty_novelty": random.uniform(0.7, 1.0),
        })

    elif archetype == "investment_fraud":
        # Many victims send money to single collector over days
        f.update({
            "bipartite_score": random.uniform(0.5, 0.88),
            "sink_score": random.uniform(0.55, 0.9),
            "txn_count_30d": float(random.randint(40, 150)),
            "distinct_counterparties_30d": float(random.randint(20, 80)),
            "txn_amount": random.uniform(5000, 50000),
            "community_fraud_ratio": random.uniform(0.5, 0.9),
            "pagerank_fraud_seeded": random.uniform(0.4, 0.85),
            "return_ratio": random.uniform(0.0, 0.08),
            "cash_mule_sink_score": random.uniform(0.5, 0.85),
        })

    elif archetype == "salary_mule":
        # Receives regular salary-size amounts, immediately relays everything out
        f.update({
            "return_ratio": random.uniform(0.85, 0.99),
            "sink_score": random.uniform(0.4, 0.75),
            "account_age_days": float(random.randint(45, 180)),
            "txn_amount": random.uniform(20000, 80000),
            "avg_txn_amount_30d": random.uniform(15000, 70000),
            "burst_score": random.uniform(0.5, 0.88),
            "velocity_ratio": random.uniform(3.0, 10.0),
            "txn_count_last_24h": float(random.randint(3, 8)),
            "kyc_completeness_score": random.uniform(0.3, 0.65),
        })

    elif archetype == "cash_in_mule":
        # Cash deposit → immediate UPI → empty account
        f.update({
            "dormancy_reactivation_flag": 1.0,
            "dormancy_break": 1.0,
            "return_ratio": random.uniform(0.88, 0.99),
            "channel_switch": random.uniform(0.6, 1.0),  # cash→UPI switch
            "txn_amount": random.uniform(50000, 150000),
            "burst_score": random.uniform(0.6, 0.9),
            "txn_count_last_24h": float(random.randint(2, 5)),
            "account_age_days": float(random.randint(30, 180)),
            "cash_mule_sink_score": random.uniform(0.5, 0.85),
        })

    elif archetype == "account_takeover":
        # Old dormant account suddenly used with new patterns
        f.update({
            "dormancy_reactivation_flag": 1.0,
            "dormancy_break": 1.0,
            "geography_switch": random.uniform(0.75, 1.0),
            "channel_switch": random.uniform(0.6, 1.0),
            "counterparty_novelty": random.uniform(0.75, 1.0),
            "burst_score": random.uniform(0.6, 0.95),
            "txn_amount": random.uniform(80000, 400000),
            "payee_vpa_age_days": float(random.randint(1, 7)),
            "amount_zscore": random.uniform(5.0, 15.0),
            "account_age_days": float(random.randint(730, 5000)),
            "txn_count_30d": float(random.randint(20, 60)),
        })

    # Keep amount-derived features consistent
    amt = f["txn_amount"]
    f["txn_amount_log"] = math.log1p(amt)
    f["txn_amount_rounded"] = 1.0 if amt == round(amt, -3) else 0.0
    f["amount_vs_threshold_50000"] = amt / 50000
    f["amount_vs_threshold_100000"] = amt / 100000
    f["amount_vs_threshold_1000000"] = amt / 1000000

    # Add 5% Gaussian noise to all float features for realistic overlap
    for k, v in f.items():
        if isinstance(v, float) and k not in (
            "cycle_membership", "dormancy_reactivation_flag", "dormancy_break",
            "is_night", "is_weekend", "is_festival_period",
            "channel_upi", "channel_imps", "channel_rtgs", "channel_neft",
            "payee_in_alert_log",
        ):
            f[k] = _noise(v, 0.05)

    # Clamp probabilities / ratios to [0,1]
    ratio_features = {
        "cycle_membership", "sink_score", "bipartite_score", "fan_out_ratio",
        "clustering_coefficient", "cash_mule_sink_score", "bridge_node_probability",
        "dormancy_reactivation_flag", "kyc_completeness_score", "night_txn_ratio",
        "weekend_txn_ratio", "return_ratio", "counterparty_novelty", "channel_switch",
        "amount_series_score", "burst_score", "dormancy_break", "geography_switch",
        "community_fraud_ratio", "pagerank_fraud_seeded",
    }
    for k in ratio_features:
        if k in f:
            f[k] = _clamp(f[k], 0.0, 1.0)

    return f


def _to_vector(features: dict) -> list:
    return [features.get(name, 0.0) or 0.0 for name in ALL_FEATURE_NAMES]


def main() -> None:
    try:
        import xgboost as xgb
        import numpy as np
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    N_TOTAL = 100_000
    N_FRAUD = 2_700  # 2.7% — heavier weight than real but gives model enough signal

    rows_X, rows_y = [], []

    per_arch = N_FRAUD // len(FRAUD_ARCHETYPES)
    remainder = N_FRAUD - per_arch * len(FRAUD_ARCHETYPES)

    for i, arch in enumerate(FRAUD_ARCHETYPES):
        extra = 1 if i < remainder else 0
        for _ in range(per_arch + extra):
            rows_X.append(_to_vector(_fraud_features(arch)))
            rows_y.append(1)

    for _ in range(N_TOTAL - N_FRAUD):
        rows_X.append(_to_vector(_legit_features()))
        rows_y.append(0)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)

    # Auto-merge real-world labeled data from ml/data/ if available
    data_dir = Path("ml/data")
    for prefix in ("baf", "kaggle"):
        x_file = data_dir / f"{prefix}_X.npy"
        y_file = data_dir / f"{prefix}_y.npy"
        if x_file.exists() and y_file.exists():
            X_ext = np.load(x_file).astype(np.float32)
            y_ext = np.load(y_file).astype(np.float32)
            if X_ext.shape[1] == X.shape[1]:
                X = np.vstack([X, X_ext])
                y = np.concatenate([y, y_ext])
                ext_fraud = int(y_ext.sum())
                print(f"Merged {prefix}: +{len(X_ext)} rows ({ext_fraud} fraud)")
            else:
                print(f"SKIP {prefix}: feature count mismatch ({X_ext.shape[1]} vs {X.shape[1]})")

    total_fraud = int(y.sum())
    total_legit = len(y) - total_fraud
    print(f"\nFinal dataset: {len(y)} rows ({total_fraud} fraud, {total_legit} legit)")

    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # scale_pos_weight computed from actual merged distribution
    train_fraud = int(y_train.sum())
    train_legit = len(y_train) - train_fraud
    pos_weight = max(1, train_legit // train_fraud)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ALL_FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=ALL_FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": pos_weight,
        "max_depth": 7,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "tree_method": "hist",
        "seed": 42,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=400,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=30, verbose_eval=40,
    )

    out = Path("ml/models/xgboost_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))

    preds = model.predict(dtest)
    fraud_mask = y_test == 1
    legit_mask = y_test == 0

    print(f"\nModel saved → {out}")
    print(f"Feature count:     {len(ALL_FEATURE_NAMES)}")
    print(f"Training examples: {N_TOTAL} ({N_FRAUD} fraud, {N_TOTAL - N_FRAUD} legit)")
    print(f"Fraud archetypes:  {len(FRAUD_ARCHETYPES)}")
    print(f"scale_pos_weight:  {pos_weight}")
    print(f"Fraud mean score:  {preds[fraud_mask].mean():.3f}")
    print(f"Clean mean score:  {preds[legit_mask].mean():.3f}")
    print(f"Best aucpr:        {model.best_score:.4f}")

    # Per-archetype breakdown on test set
    archetype_labels = []
    per_arch_test = int(per_arch * 0.2)
    for arch in FRAUD_ARCHETYPES:
        archetype_labels.extend([arch] * per_arch_test)
    test_fraud_preds = preds[fraud_mask]
    if len(test_fraud_preds) >= len(archetype_labels):
        print("\nPer-archetype mean score (test):")
        start = 0
        for arch in FRAUD_ARCHETYPES:
            chunk = preds[fraud_mask][start:start + per_arch_test]
            if len(chunk) > 0:
                print(f"  {arch:<25} {chunk.mean():.3f}")
            start += per_arch_test


if __name__ == "__main__":
    main()
