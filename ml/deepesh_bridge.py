"""
Bridge: deepeshkansotia banking fraud dataset → graph feature schema.
10K rows, 12.5% fraud rate, excellent graph-mappable columns:
  device_risk_score → pagerank_fraud_seeded
  geo_distance_km → geography_switch
  transaction_velocity_score → velocity_ratio, burst_score
  anomaly_score → bridge_node_probability
  suspicious_ip_flag → community_fraud_ratio
  international_transaction_flag → geography_switch
  failed_transactions_last_30d → amount_series_score
  login_attempts → channel_switch
  account_age_days → account_age_days (direct)
  avg_monthly_balance → sink_score (inverse proxy)

Run: python ml/deepesh_bridge.py
Then: python ml/train.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

CSV_PATH = (
    Path.home()
    / ".cache/kagglehub/datasets"
    / "deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset"
    / "versions/1/banking_transactions.csv"
)
OUTPUT_DIR = Path("ml/data")

FEATURE_NAMES = sorted(set([
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
    "txn_amount", "txn_amount_log", "txn_amount_rounded",
    "channel_upi", "channel_imps", "channel_rtgs", "channel_neft",
    "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_festival_period",
    "amount_vs_threshold_50000", "amount_vs_threshold_100000", "amount_vs_threshold_1000000",
    "payee_vpa_age_days",
    "txn_count_last_1h", "txn_count_last_24h", "txn_count_last_7d",
    "txn_volume_last_1h", "txn_volume_last_24h",
    "distinct_payees_24h",
    "payee_in_alert_log", "payee_shared_alert_count",
]))

CHANNEL_MAP = {
    "Mobile App": ("channel_upi", 1.0),
    "Web Banking": ("channel_neft", 1.0),
    "POS Terminal": ("channel_imps", 1.0),
    "ATM": ("channel_imps", 1.0),
}

AUTH_RISK = {
    "Biometric": 0.05,
    "Two-Factor Authentication": 0.1,
    "OTP": 0.3,
    "Password Only": 0.6,
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _n(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((v - lo) / (hi - lo))


def _row_to_features(row) -> dict:
    amount = float(row["transaction_amount"])
    hour = int(row["transaction_time_hour"])
    acct_age = float(row["account_age_days"])
    login = float(row["login_attempts"])
    dev_risk = float(row["device_risk_score"])     # 0-100
    geo_km = float(row["geo_distance_km"])          # 0-15000
    vel_score = float(row["transaction_velocity_score"])  # 0-100
    anomaly = float(row["anomaly_score"])            # 0-1
    failed_30d = float(row["failed_transactions_last_30d"])  # 0-25
    daily_count = float(row["daily_transaction_count"])     # 1-120
    avg_balance = float(row["avg_monthly_balance"])         # varies
    session_min = float(row["session_duration_minutes"])
    intl = int(row["international_transaction_flag"])
    susp_ip = int(row["suspicious_ip_flag"])

    # Graph feature mappings
    pagerank_fraud = _clamp(
        dev_risk / 100.0 * 0.5
        + float(susp_ip) * 0.3
        + anomaly * 0.2
    )
    community_fraud = _clamp(float(susp_ip) * 0.6 + anomaly * 0.4)
    shortest_path = max(0.0, 5.0 - dev_risk / 25.0)  # higher risk → fewer hops

    geo_switch = _clamp(
        float(intl) * 0.6
        + _n(geo_km, 0, 15000) * 0.4
    )
    bipartite_score = _clamp(_n(daily_count, 1, 120) * 0.7 + anomaly * 0.3)

    # sink: if avg_balance high relative to amount, account is accumulating (could be mule)
    sink_score = _clamp(amount / (avg_balance + 1.0))

    burst_score = _clamp(vel_score / 100.0 * 0.6 + anomaly * 0.4)
    velocity_ratio = _clamp(vel_score / 10.0, 0.0, 30.0)

    channel_switch = _clamp(
        (login - 1) / 11.0 * 0.5
        + AUTH_RISK.get(str(row.get("authentication_type", "OTP")), 0.3) * 0.5
    )
    amount_series_score = _clamp(failed_30d / 25.0)
    bridge_prob = _clamp(anomaly * 0.5 + float(susp_ip) * 0.3 + geo_switch * 0.2)
    dormancy_flag = 1.0 if (session_min < 2 and vel_score > 70) else 0.0

    # Channel
    ch_upi = ch_imps = ch_rtgs = ch_neft = 0.0
    ch = str(row.get("payment_channel", "Mobile App"))
    if ch in ("Mobile App",):
        ch_upi = 1.0
    elif ch in ("ATM", "POS Terminal"):
        ch_imps = 1.0
    elif ch in ("Web Banking",):
        ch_neft = 1.0

    is_night = 1.0 if (hour >= 22 or hour < 5) else 0.0
    hour_dev = abs(hour - 14.0) / 10.0

    # KYC: card_present + auth_type proxy
    card_present = int(row.get("card_present_flag", 1))
    auth_risk = AUTH_RISK.get(str(row.get("authentication_type", "OTP")), 0.3)
    kyc_score = _clamp(float(card_present) * 0.4 + (1 - auth_risk) * 0.6)

    cash_mule_score = _clamp(sink_score * 0.6 + burst_score * 0.4)
    counterparty_novelty = _clamp(geo_switch * 0.5 + _n(login, 1, 12) * 0.5)
    temporal_acceleration = velocity_ratio
    degree_centrality = _clamp(_n(daily_count, 1, 120))
    return_ratio = _clamp(1.0 - (avg_balance / (avg_balance + amount + 1.0)))
    txn_count_1h = _clamp(daily_count / 24.0, 0.0, 20.0)
    txn_count_24h = _clamp(daily_count, 0.0, 120.0)
    txn_vol_1h = amount * txn_count_1h
    txn_vol_24h = amount * min(txn_count_24h, 10.0)
    distinct_payees = _clamp(daily_count / 5.0, 0.0, 20.0)

    return {
        "degree_centrality": degree_centrality,
        "betweenness_centrality": _clamp(bridge_prob * 0.1),
        "clustering_coefficient": _clamp(1.0 - counterparty_novelty),
        "pagerank_fraud_seeded": pagerank_fraud,
        "community_id": float(hash(str(row.get("payment_channel", ""))) % 100),
        "community_fraud_ratio": community_fraud,
        "shortest_path_to_fraud": shortest_path,
        "cycle_membership": 0.0,
        "sink_score": sink_score,
        "bipartite_score": bipartite_score,
        "fan_out_ratio": _clamp(1.0 - sink_score),
        "temporal_acceleration": temporal_acceleration,
        "cash_mule_sink_score": cash_mule_score,
        "bridge_node_probability": bridge_prob,
        "dormancy_reactivation_flag": dormancy_flag,
        "account_age_days": acct_age,
        "kyc_completeness_score": kyc_score,
        "txn_count_30d": txn_count_24h * 30,
        "txn_count_90d": txn_count_24h * 90,
        "txn_count_all": txn_count_24h * 365,
        "avg_txn_amount_30d": amount,
        "distinct_counterparties_30d": distinct_payees,
        "channel_entropy": 0.5 + float(intl) * 0.5,
        "night_txn_ratio": is_night * 0.5 + _n(hour, 22, 4) * 0.5,
        "weekend_txn_ratio": 0.3,
        "return_ratio": return_ratio,
        "amount_zscore": _clamp(_n(amount, 6, 25000) * 5.0, 0.0, 10.0),
        "counterparty_novelty": counterparty_novelty,
        "hour_deviation": hour_dev,
        "channel_switch": channel_switch,
        "amount_series_score": amount_series_score,
        "burst_score": burst_score,
        "velocity_ratio": velocity_ratio,
        "dormancy_break": dormancy_flag,
        "geography_switch": geo_switch,
        "txn_amount": amount,
        "txn_amount_log": math.log1p(amount),
        "txn_amount_rounded": 1.0 if amount == round(amount, -2) else 0.0,
        "channel_upi": ch_upi, "channel_imps": ch_imps,
        "channel_rtgs": ch_rtgs, "channel_neft": ch_neft,
        "hour_of_day": float(hour),
        "day_of_week": 1.0,
        "is_weekend": 0.0,
        "is_night": is_night,
        "is_festival_period": 0.0,
        "amount_vs_threshold_50000": amount / 50000,
        "amount_vs_threshold_100000": amount / 100000,
        "amount_vs_threshold_1000000": amount / 1000000,
        "payee_vpa_age_days": max(1.0, float(session_min)),
        "txn_count_last_1h": txn_count_1h,
        "txn_count_last_24h": txn_count_24h,
        "txn_count_last_7d": txn_count_24h * 7,
        "txn_volume_last_1h": txn_vol_1h,
        "txn_volume_last_24h": txn_vol_24h,
        "distinct_payees_24h": distinct_payees,
        "payee_in_alert_log": float(susp_ip),
        "payee_shared_alert_count": float(failed_30d),
    }


def main() -> None:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("ERROR: pip install pandas numpy")
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: Dataset not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows — fraud: {int(df['fraud_flag'].sum())} ({100*df['fraud_flag'].mean():.1f}%)")

    rows_X, rows_y = [], []
    for _, row in df.iterrows():
        feats = _row_to_features(row)
        rows_X.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        rows_y.append(1 if row["fraud_flag"] else 0)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save with "kaggle" prefix so train.py auto-merges it
    np.save(OUTPUT_DIR / "kaggle_X.npy", X)
    np.save(OUTPUT_DIR / "kaggle_y.npy", y)
    print(f"Saved → {OUTPUT_DIR}/kaggle_X.npy  shape={X.shape}")
    print(f"Saved → {OUTPUT_DIR}/kaggle_y.npy  shape={y.shape}")
    print("Re-run: python ml/train.py  (will auto-merge with BAF data)")


if __name__ == "__main__":
    main()
