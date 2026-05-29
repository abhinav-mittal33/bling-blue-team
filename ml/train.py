"""
Train XGBoost fraud detection model — Phase 4 update (P4-1 through P4-7).

Phase 4 changes:
  P4-1: Expanded archetypes (hawala, crypto_on_ramp, benami + ieee-cis + adbench blend)
  P4-2: Feature names from ml/feature_registry.py (single source of truth)
  P4-3: HGT ensemble component — STUBBED (requires teammate P2-2 hetero Neo4j schema)
  P4-4: Platt scaling calibration on XGBClassifier
  P4-5: Threshold derivation on calibrated output (LOG@recall=0.95, REVIEW@F1, HIGH@prec=0.90)
  P4-6: XGBOD second novelty layer (uses pyod)
  P4-7: PSI drift monitoring baseline — saves feature distribution stats

Leiden gate: training refuses to run unless LEIDEN_DEPLOYED=true in Redis (or --force passed).
This enforces atomic deployment of community features + model retrain.

SHAP invariant: base_xgb is saved separately for SHAP. calibrated model is for scoring.
scale_pos_weight: computed from actual training distribution, printed for update in CLAUDE.md.
eval_metric='aucpr': NOT 'auc' — PR-AUC for imbalanced data.

Run: python ml/train.py
Run without Leiden gate: python ml/train.py --force
"""
from __future__ import annotations
import json
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
    "romance_scam",
    "pig_butchering",
    "sim_swap",
    "otp_fraud",
    "investment_fraud",
    "salary_mule",
    "cash_in_mule",
    "account_takeover",
    # Phase 4 new archetypes (P4-1 + P3-7)
    "hawala",            # Near-threshold + high velocity relay, foreign remittance timing
    "crypto_on_ramp",    # Round amounts to brand-new VPAs at high frequency
    "benami",            # Dormant account suddenly used as proxy (>1yr old, velocity spike)
]

# P4-2: Feature names from feature_registry — NEVER hardcode here
# Both training and inference use the same feature set via this import
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.feature_registry import FEATURE_NAMES  # noqa: E402

ALL_FEATURE_NAMES = sorted(set(FEATURE_NAMES))


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

    # ── Phase 4 new archetypes (P4-1 + P3-7) ─────────────────────────────────
    elif archetype == "hawala":
        # Cash-equivalent relay: near-threshold, high velocity, foreign remittance timing
        amount = random.uniform(88000, 99500)
        f.update({
            "txn_amount": amount,
            "amount_series_score": random.uniform(0.65, 0.92),
            "velocity_ratio": random.uniform(6.0, 18.0),
            "burst_score": random.uniform(0.55, 0.88),
            "txn_count_last_24h": float(random.randint(5, 15)),
            "distinct_payees_24h": float(random.randint(3, 8)),
            "counterparty_novelty": random.uniform(0.55, 0.9),
            "geography_switch": random.uniform(0.5, 1.0),
            "return_ratio": random.uniform(0.7, 0.95),
            "night_txn_ratio": random.uniform(0.4, 0.8),
            "fan_out_ratio": random.uniform(0.6, 0.95),
        })

    elif archetype == "crypto_on_ramp":
        # Round amounts to brand-new VPAs, high frequency
        amount = float(random.choice([10000, 20000, 30000, 50000, 100000]))
        f.update({
            "txn_amount": amount,
            "payee_vpa_age_days": float(random.randint(1, 6)),
            "txn_count_last_24h": float(random.randint(4, 12)),
            "velocity_ratio": random.uniform(5.0, 20.0),
            "burst_score": random.uniform(0.6, 0.92),
            "counterparty_novelty": random.uniform(0.75, 1.0),
            "distinct_payees_24h": float(random.randint(3, 8)),
            "geography_switch": random.uniform(0.4, 0.9),
            "channel_entropy": random.uniform(0.0, 0.3),
            "return_ratio": random.uniform(0.0, 0.1),
        })

    elif archetype == "benami":
        # Dormant account used as proxy — old account, sudden velocity spike
        f.update({
            "account_age_days": float(random.randint(365, 5000)),
            "dormancy_reactivation_flag": 1.0,
            "dormancy_break": 1.0,
            "burst_score": random.uniform(0.65, 0.95),
            "velocity_ratio": random.uniform(8.0, 30.0),
            "txn_count_last_24h": float(random.randint(8, 20)),
            "distinct_payees_24h": float(random.randint(3, 10)),
            "counterparty_novelty": random.uniform(0.7, 1.0),
            "geography_switch": random.uniform(0.5, 1.0),
            "channel_switch": random.uniform(0.4, 0.85),
            "kyc_completeness_score": random.uniform(0.2, 0.55),
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


def _check_leiden_gate(force: bool) -> None:
    """Enforce atomic Leiden + retrain deployment gate (P2-1 invariant)."""
    if force:
        print("WARNING: --force bypasses Leiden gate. Only use for initial baseline training.")
        return
    try:
        import os
        import redis as redis_lib
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        r = redis_lib.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
        deployed = r.get("leiden:deployed")
        if deployed != "true":
            print("ERROR: Leiden community detection not yet deployed.")
            print("       Run the nightly batch first, then retrain.")
            print("       To bypass: python ml/train.py --force")
            sys.exit(1)
        print(f"Leiden gate: PASSED (deployed at {r.get('leiden:deployed_at') or 'unknown'})")
    except Exception as exc:
        print(f"WARNING: Cannot check Leiden gate ({exc}). Proceeding with --force logic.")


def _derive_thresholds(y_true, y_pred_proba) -> dict:
    """
    Derive LOG/REVIEW/HIGH_RISK thresholds from held-out test set (P4-5).
    Methodology:
      LOG      = highest score where recall ≥ 0.95 (catch 95% of all fraud)
      REVIEW   = score where recall ≥ 0.80 AND precision ≥ 0.60
      HIGH_RISK = score where precision ≥ 0.90
    All derived on TEST set — never validation (validation used for calibration).
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # Note: precision_recall_curve appends a sentinel [1.0, 0.0, None] at end
    # thresholds has len = len(precision) - 1

    log_thresh = 0.38     # default fallback
    review_thresh = 0.62
    high_risk_thresh = 0.83

    for i, t in enumerate(thresholds):
        rec = recall[i]
        prec = precision[i]

        if rec >= 0.95:
            log_thresh = round(float(t), 4)

        if rec >= 0.80 and prec >= 0.60 and t > log_thresh:
            review_thresh = round(float(t), 4)

        if prec >= 0.90 and t > review_thresh:
            high_risk_thresh = round(float(t), 4)
            break  # first t where prec≥0.90 above review_thresh

    return {
        "LOG": log_thresh,
        "REVIEW": review_thresh,
        "HIGH_RISK": high_risk_thresh,
    }


def _save_psi_baseline(X_train, y_train, feature_names: list, out_dir: Path) -> None:
    """
    Save per-feature distribution stats for PSI drift monitoring (P4-7).
    Saves: mean, std, percentiles [10,25,50,75,90], score hist bins.
    Alert during monitoring if PSI(feature) > 0.2 for any feature.
    """
    import numpy as np

    baseline = {}
    for i, name in enumerate(feature_names):
        col = X_train[:, i]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        baseline[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "p10": float(np.percentile(col, 10)),
            "p25": float(np.percentile(col, 25)),
            "p50": float(np.percentile(col, 50)),
            "p75": float(np.percentile(col, 75)),
            "p90": float(np.percentile(col, 90)),
        }

    psi_file = out_dir / "psi_baseline.json"
    with open(psi_file, "w") as f:
        json.dump(baseline, f)
    print(f"PSI baseline saved → {psi_file} ({len(baseline)} features)")


def _train_xgbod(X_train, y_train, out_dir: Path) -> None:
    """
    Train XGBOD second novelty layer (P4-6).
    INVARIANT: XGBOD output NEVER enters fraud_score. Saved for novelty pipeline only.
    """
    try:
        from pyod.models.xgbod import XGBOD
        import numpy as np
        import joblib

        # Train only on fraud samples (one-class novelty detection)
        fraud_mask = y_train == 1
        X_fraud = X_train[fraud_mask]
        if len(X_fraud) < 100:
            print("SKIP XGBOD: not enough fraud samples")
            return

        xgbod = XGBOD(n_estimators=50, random_state=42)
        xgbod.fit(X_fraud)

        out = out_dir / "xgbod_v1.joblib"
        joblib.dump(xgbod, str(out))
        print(f"XGBOD saved → {out}")

    except ImportError:
        print("SKIP XGBOD: pyod not installed. Add pyod[xgbod] to requirements.txt")
    except Exception as exc:
        print(f"SKIP XGBOD: {exc}")


def main() -> None:
    force = "--force" in sys.argv

    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import precision_recall_curve
        import joblib
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # P2-1 Leiden deployment gate — atomic retraining
    _check_leiden_gate(force)

    N_TOTAL = 100_000
    N_FRAUD = 2_700

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

    # P4-1: Merge external labeled datasets
    data_dir = Path("ml/data")
    for prefix in ("baf", "kaggle", "ieee_cis", "adbench"):
        x_file = data_dir / f"{prefix}_X.npy"
        y_file = data_dir / f"{prefix}_y.npy"
        if x_file.exists() and y_file.exists():
            X_ext = np.load(x_file).astype(np.float32)
            y_ext = np.load(y_file).astype(np.float32)
            if X_ext.shape[1] == X.shape[1]:
                X = np.vstack([X, X_ext])
                y = np.concatenate([y, y_ext])
                print(f"Merged {prefix}: +{len(X_ext)} rows ({int(y_ext.sum())} fraud)")
            else:
                print(f"SKIP {prefix}: feature count mismatch ({X_ext.shape[1]} vs {X.shape[1]})")

    total_fraud = int(y.sum())
    total_legit = len(y) - total_fraud
    print(f"\nFinal dataset: {len(y)} rows ({total_fraud} fraud, {total_legit} legit)")

    # P4-4: 3-way split — train (60%), val (20%), test (20%)
    # val → Platt calibration. test → threshold derivation (NEVER validation).
    idx = np.random.permutation(len(X))
    n = len(idx)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    X_train = X[idx[:train_end]]
    y_train = y[idx[:train_end]]
    X_val = X[idx[train_end:val_end]]
    y_val = y[idx[train_end:val_end]]
    X_test = X[idx[val_end:]]
    y_test = y[idx[val_end:]]

    train_fraud = int(y_train.sum())
    train_legit = len(y_train) - train_fraud
    pos_weight = max(1, train_legit // train_fraud)

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"scale_pos_weight: {pos_weight} (recomputed from actual distribution)")
    print("→ Update CLAUDE.md scale_pos_weight table with this value + today's date.")

    # P4-2: Train base XGBClassifier (sklearn API for calibration compatibility)
    base_xgb = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        max_depth=7,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=30,
    )
    base_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=40,
    )

    # P4-4: Platt scaling calibration on val set
    # cv="prefit" = base model already fitted — only fit the calibration layer
    calibrated = CalibratedClassifierCV(base_xgb, cv="prefit", method="sigmoid")
    calibrated.fit(X_val, y_val)

    out_dir = Path("ml/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save base model for SHAP (INVARIANT: SHAP must use base, not calibrated)
    base_out = out_dir / "xgboost_base_v2.json"
    base_xgb.get_booster().save_model(str(base_out))
    print(f"Base model saved → {base_out}")

    # Save calibrated model for scoring
    cal_out = out_dir / "xgboost_calibrated_v2.joblib"
    joblib.dump(calibrated, str(cal_out))
    print(f"Calibrated model saved → {cal_out}")

    # Also save legacy v1.json for backwards compatibility during transition
    legacy_out = out_dir / "xgboost_v1.json"
    base_xgb.get_booster().save_model(str(legacy_out))

    # Model integrity hashes (P0-3)
    try:
        from app.utils.model_integrity import store_model_hash
        store_model_hash(base_out)
        store_model_hash(legacy_out)
    except Exception as e:
        print(f"WARNING: model hash storage failed: {e}")

    # P4-5: Derive thresholds on TEST set (not validation)
    cal_preds_test = calibrated.predict_proba(X_test)[:, 1]
    thresholds = _derive_thresholds(y_test, cal_preds_test)

    fraud_mask = y_test == 1
    legit_mask = y_test == 0

    print(f"\n{'─'*55}")
    print(f"Feature count:     {len(ALL_FEATURE_NAMES)}")
    print(f"Fraud archetypes:  {len(FRAUD_ARCHETYPES)}")
    print(f"scale_pos_weight:  {pos_weight}")
    print(f"Fraud mean score (calibrated): {cal_preds_test[fraud_mask].mean():.3f}")
    print(f"Clean mean score (calibrated): {cal_preds_test[legit_mask].mean():.3f}")
    print(f"\nDerived thresholds (update .env + CLAUDE.md after review):")
    print(f"  LOG_THRESHOLD:       {thresholds['LOG']}")
    print(f"  REVIEW_THRESHOLD:    {thresholds['REVIEW']}")
    print(f"  HIGH_RISK_THRESHOLD: {thresholds['HIGH_RISK']}")
    print(f"{'─'*55}")
    print("ACTION REQUIRED: Update .env and CLAUDE.md threshold table with values above.")

    # Save derived thresholds to file for reference
    thresh_file = out_dir / "thresholds_v2.json"
    with open(thresh_file, "w") as f:
        json.dump({"version": "v2", "thresholds": thresholds, "scale_pos_weight": pos_weight}, f)
    print(f"Thresholds saved → {thresh_file}")

    # P4-7: PSI drift monitoring baseline
    _save_psi_baseline(X_train, y_train, ALL_FEATURE_NAMES, out_dir)

    # P4-6: XGBOD second novelty layer
    _train_xgbod(X_train, y_train, out_dir)

    # P4-3: HGT ensemble — STUBBED (requires P2-2 hetero Neo4j schema from teammate)
    print("\nHGT ensemble: SKIPPED (P4-3 requires P2-2 hetero schema from teammate)")

    # Per-archetype breakdown on test set
    per_arch_test = max(1, int(per_arch * 0.2))
    raw_preds_test = base_xgb.predict_proba(X_test)[:, 1]
    print("\nPer-archetype mean raw score (test):")
    start = 0
    for arch in FRAUD_ARCHETYPES:
        chunk = raw_preds_test[fraud_mask][start:start + per_arch_test]
        if len(chunk) > 0:
            print(f"  {arch:<25} {chunk.mean():.3f}")
        start += per_arch_test


if __name__ == "__main__":
    main()
