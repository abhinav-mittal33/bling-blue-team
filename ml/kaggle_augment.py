"""
Feature bridge: maps Kaggle bank transaction dataset to our 59-feature schema.
Generates rule-based pseudo-labels (no ground truth in this dataset).
Output: ml/data/kaggle_augmented.npy (X array) + labels.npy (y array)

Run standalone: python ml/kaggle_augment.py
Then re-run: python ml/train.py  (it auto-detects and merges this file)

Dataset: valakhorasani/bank-transaction-dataset-for-fraud-detection (2512 rows)
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

DATASET_PATH = (
    Path.home()
    / ".cache/kagglehub/datasets"
    / "valakhorasani/bank-transaction-dataset-for-fraud-detection"
    / "versions/4/bank_transactions_data_2.csv"
)

OUTPUT_DIR = Path("ml/data")

# Our full feature schema — must match train.py ALL_FEATURE_NAMES exactly
FEATURE_NAMES = sorted(set([
    # 35 graph features
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
    # 24 real-time features
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

OCCUPATION_KYC_SCORE = {
    "Doctor": 0.95, "Engineer": 0.92, "Student": 0.70, "Retired": 0.85,
}

CHANNEL_MAP = {
    "Online": ("channel_upi", 1.0),
    "ATM": ("channel_imps", 1.0),
    "Branch": ("channel_neft", 1.0),
}


def _entropy(values: list) -> float:
    from collections import Counter
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((c / total) * math.log2(c / total + 1e-9) for c in counts.values())


def _zscore(val: float, mean: float, std: float) -> float:
    if std < 1e-9:
        return 0.0
    return (val - mean) / std


def main() -> None:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("ERROR: pip install pandas numpy")
        sys.exit(1)

    if not DATASET_PATH.exists():
        print("ERROR: Dataset not downloaded. Run:")
        print("  python3 -c \"import kagglehub; kagglehub.dataset_download('valakhorasani/bank-transaction-dataset-for-fraud-detection')\"")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH, parse_dates=["TransactionDate", "PreviousTransactionDate"])
    print(f"Loaded {len(df)} rows, {df['AccountID'].nunique()} unique accounts")

    # ── Per-account aggregates ────────────────────────────────────────────
    df["Hour"] = df["TransactionDate"].dt.hour
    df["DayOfWeek"] = df["TransactionDate"].dt.dayofweek
    df["IsNight"] = ((df["Hour"] >= 22) | (df["Hour"] < 5)).astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    global_mean = df["TransactionAmount"].mean()
    global_std = df["TransactionAmount"].std()

    per_account = df.groupby("AccountID").agg(
        txn_count_all=("TransactionID", "count"),
        avg_amount=("TransactionAmount", "mean"),
        std_amount=("TransactionAmount", "std"),
        min_balance=("AccountBalance", "min"),
        avg_balance=("AccountBalance", "mean"),
        total_out=("TransactionAmount", lambda x: x[df.loc[x.index, "TransactionType"] == "Debit"].sum()),
        total_in=("TransactionAmount", lambda x: x[df.loc[x.index, "TransactionType"] == "Credit"].sum()),
        n_locations=("Location", "nunique"),
        n_devices=("DeviceID", "nunique"),
        n_merchants=("MerchantID", "nunique"),
        night_txns=("IsNight", "sum"),
        weekend_txns=("IsWeekend", "sum"),
        max_login_attempts=("LoginAttempts", "max"),
        channels=("Channel", lambda x: list(x)),
    ).reset_index()

    per_account["night_txn_ratio"] = per_account["night_txns"] / per_account["txn_count_all"].clip(1)
    per_account["weekend_txn_ratio"] = per_account["weekend_txns"] / per_account["txn_count_all"].clip(1)
    per_account["channel_entropy"] = per_account["channels"].apply(_entropy)
    # sink: receives much more than it sends → return_ratio
    per_account["return_ratio"] = (
        per_account["total_in"] / (per_account["total_out"] + per_account["total_in"] + 1e-6)
    )
    # geography_switch: multiple locations
    per_account["geography_switch"] = (per_account["n_locations"] - 1).clip(0, 1).astype(float)
    # counterparty_novelty: many different merchants → novel counterparties
    per_account["counterparty_novelty"] = (
        per_account["n_merchants"] / per_account["txn_count_all"].clip(1)
    )
    # sink_score: low balance relative to inflow (money exits quickly)
    per_account["sink_score"] = (
        1.0 - (per_account["min_balance"] / (per_account["total_in"] + 1.0))
    ).clip(0.0, 1.0)
    # channel_switch: high login attempts suggest channel-switching attacks
    per_account["channel_switch"] = ((per_account["max_login_attempts"] - 1) / 4.0).clip(0.0, 1.0)

    acct_lookup = per_account.set_index("AccountID").to_dict("index")

    # ── Rule-based pseudo-fraud labels ────────────────────────────────────
    # Fraud criteria (no ground truth — using behavioral anomalies):
    #   - LoginAttempts >= 3 (brute force / OTP abuse)
    #   - TransactionAmount > 3 * account avg_amount AND new location
    #   - High night_txn_ratio (>0.6) + high amount
    #   - Multiple locations (geography_switch) + high login attempts

    fraud_masks = []
    for _, row in df.iterrows():
        acct = acct_lookup.get(row["AccountID"], {})
        is_fraud = False

        if row["LoginAttempts"] >= 3:
            is_fraud = True
        elif (acct.get("n_locations", 1) >= 3 and row["LoginAttempts"] >= 2):
            is_fraud = True
        elif (row["TransactionAmount"] > 3 * (acct.get("avg_amount", row["TransactionAmount"]))
              and acct.get("night_txn_ratio", 0) > 0.5):
            is_fraud = True
        elif (acct.get("channel_switch", 0) > 0.5 and acct.get("geography_switch", 0) > 0):
            is_fraud = True

        fraud_masks.append(1 if is_fraud else 0)

    df["label"] = fraud_masks
    n_fraud = sum(fraud_masks)
    print(f"Pseudo-labels: {n_fraud} fraud ({100*n_fraud/len(df):.1f}%), {len(df)-n_fraud} legit")

    # ── Build feature vectors ─────────────────────────────────────────────
    import numpy as np
    hour_mean = df["Hour"].mean()
    hour_std = df["Hour"].std() or 1.0

    rows_X, rows_y = [], []

    for _, row in df.iterrows():
        acct = acct_lookup.get(row["AccountID"], {})
        amount = float(row["TransactionAmount"])
        hour = int(row["Hour"])
        dow = int(row["DayOfWeek"])

        # Channel mapping (ATM→imps, Online→upi, Branch→neft)
        ch_upi = ch_imps = ch_rtgs = ch_neft = 0.0
        ch = str(row.get("Channel", "Online"))
        if ch == "Online":
            ch_upi = 1.0
        elif ch == "ATM":
            ch_imps = 1.0
        elif ch == "Branch":
            ch_neft = 1.0

        # Amount z-score relative to global distribution (Indian: scale up ×100 for INR proxy)
        amt_zscore = _zscore(amount, global_mean, global_std)

        # Graph feature proxies derived from per-account stats
        acct_age = max(1, acct.get("txn_count_all", 1) * 10)  # proxy (no real account age)
        kyc_score = OCCUPATION_KYC_SCORE.get(str(row.get("CustomerOccupation", "Engineer")), 0.8)

        n_locs = float(acct.get("n_locations", 1))
        n_merch = float(acct.get("n_merchants", 1))
        n_total = float(acct.get("txn_count_all", 1))

        sink_score = float(acct.get("sink_score", 0.1))
        return_ratio = float(acct.get("return_ratio", 0.5))
        night_ratio = float(acct.get("night_txn_ratio", 0.1))
        weekend_ratio = float(acct.get("weekend_txn_ratio", 0.3))
        geo_switch = float(acct.get("geography_switch", 0.0))
        ch_entropy = float(acct.get("channel_entropy", 0.5))
        counterparty_novelty = float(acct.get("counterparty_novelty", 0.3))
        ch_switch = float(acct.get("channel_switch", 0.0))

        # Burst score: high login attempts on this transaction
        burst_score = min(1.0, float(row["LoginAttempts"]) / 5.0)
        velocity_ratio = float(row["LoginAttempts"])  # proxy

        feats = {
            # Real-time
            "txn_amount": amount,
            "txn_amount_log": math.log1p(amount),
            "txn_amount_rounded": 1.0 if amount == round(amount, -2) else 0.0,
            "channel_upi": ch_upi, "channel_imps": ch_imps,
            "channel_rtgs": ch_rtgs, "channel_neft": ch_neft,
            "hour_of_day": float(hour),
            "day_of_week": float(dow),
            "is_weekend": 1.0 if dow >= 5 else 0.0,
            "is_night": 1.0 if (hour >= 22 or hour < 5) else 0.0,
            "is_festival_period": 0.0,
            "amount_vs_threshold_50000": amount / 50000,
            "amount_vs_threshold_100000": amount / 100000,
            "amount_vs_threshold_1000000": amount / 1000000,
            "payee_vpa_age_days": float(row.get("TransactionDuration", 30)),  # proxy
            "txn_count_last_1h": float(row["LoginAttempts"]),  # proxy
            "txn_count_last_24h": min(n_total, 10.0),
            "txn_count_last_7d": min(n_total, 30.0),
            "txn_volume_last_1h": amount * float(row["LoginAttempts"]),
            "txn_volume_last_24h": float(acct.get("avg_amount", amount)) * min(n_total, 5),
            "distinct_payees_24h": min(n_merch, 10.0),
            "payee_in_alert_log": 0.0,
            "payee_shared_alert_count": 0.0,
            # Graph features (proxy-engineered)
            "degree_centrality": min(1.0, n_total / 50.0),
            "betweenness_centrality": min(0.1, n_merch / 100.0),
            "clustering_coefficient": max(0.0, 1.0 - geo_switch),
            "pagerank_fraud_seeded": min(1.0, burst_score * 0.8),
            "community_id": float(hash(row["AccountID"]) % 200),
            "community_fraud_ratio": min(1.0, ch_switch * 0.9),
            "shortest_path_to_fraud": max(0.0, 5.0 - float(row["LoginAttempts"])),
            "cycle_membership": 0.0,  # can't infer without graph
            "sink_score": sink_score,
            "bipartite_score": min(1.0, n_merch / 10.0),
            "fan_out_ratio": min(1.0, n_merch / max(1, n_total)),
            "temporal_acceleration": velocity_ratio,
            "cash_mule_sink_score": sink_score * burst_score,
            "bridge_node_probability": min(1.0, geo_switch * ch_switch),
            "dormancy_reactivation_flag": 0.0,
            "account_age_days": float(acct_age),
            "kyc_completeness_score": kyc_score,
            "txn_count_30d": min(n_total, 30.0),
            "txn_count_90d": min(n_total * 3, 90.0),
            "txn_count_all": n_total,
            "avg_txn_amount_30d": float(acct.get("avg_amount", amount)),
            "distinct_counterparties_30d": min(n_merch, 20.0),
            "channel_entropy": ch_entropy,
            "night_txn_ratio": night_ratio,
            "weekend_txn_ratio": weekend_ratio,
            "return_ratio": return_ratio,
            "amount_zscore": abs(amt_zscore),
            "counterparty_novelty": counterparty_novelty,
            "hour_deviation": abs(hour - 14.0) / 10.0,  # deviation from midday
            "channel_switch": ch_switch,
            "amount_series_score": min(1.0, abs(amt_zscore) / 5.0),
            "burst_score": burst_score,
            "velocity_ratio": velocity_ratio,
            "dormancy_break": 0.0,
            "geography_switch": geo_switch,
        }

        rows_X.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        rows_y.append(int(row["label"]))

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "kaggle_X.npy", X)
    np.save(OUTPUT_DIR / "kaggle_y.npy", y)
    print(f"\nSaved → {OUTPUT_DIR}/kaggle_X.npy ({X.shape})")
    print(f"Saved → {OUTPUT_DIR}/kaggle_y.npy ({y.shape})")
    print("Re-run: python ml/train.py  (will auto-merge this data)")


if __name__ == "__main__":
    main()
