"""
BAF (Bank Account Fraud) NeurIPS 2022 → Graph Feature Bridge.

Maps the BAF dataset's 30 behavioral/velocity features to our 59-feature graph schema.
BAF has 1M+ rows with ground-truth fraud_bool labels — a real labeled dataset.

Feature mapping rationale:
  BAF velocity_6h/24h/4w → txn_count_last_1h/24h/30d (transaction velocity)
  BAF device_fraud_count → pagerank_fraud_seeded, shortest_path_to_fraud
  BAF zip_count_4w / bank_branch_count_8w → bipartite_score (many-to-one pattern)
  BAF date_of_birth_distinct_emails_4w → community_fraud_ratio
  BAF device_distinct_emails_8w → counterparty_novelty
  BAF bank_months_count / current_address_months_count → account_age_days
  BAF foreign_request → geography_switch
  BAF credit_risk_score → pagerank_fraud_seeded (inverse)
  BAF email_is_free + phone validity → kyc_completeness_score
  BAF proposed_credit_limit vs income → amount_zscore
  BAF keep_alive_session → dormancy_reactivation_flag

Run: python ml/baf_graph_bridge.py
Then: python ml/train.py (auto-merges output)
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

BAF_DIR = (
    Path.home()
    / ".cache/kagglehub/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
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


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((v - lo) / (hi - lo))


def _find_baf_csv() -> Path:
    """Find the Base.csv file in the downloaded BAF dataset."""
    for version_dir in sorted(BAF_DIR.glob("versions/*"), reverse=True):
        candidates = list(version_dir.glob("*.csv"))
        if candidates:
            # Prefer Base.csv (main BAF file with fraud labels)
            for c in candidates:
                if c.name.lower() in ("base.csv", "baf_base.csv"):
                    return c
            return candidates[0]
    raise FileNotFoundError(
        f"BAF dataset not found at {BAF_DIR}. "
        "Run: python3 -c \"import kagglehub; "
        "kagglehub.dataset_download('sgpjesus/bank-account-fraud-dataset-neurips-2022')\""
    )


def _map_employment(emp: str) -> float:
    """Map employment status to KYC-like completeness score."""
    mapping = {
        "CA": 0.6,   # Casual
        "CB": 0.65,  # Contract
        "CC": 0.75,  # Part-time
        "CD": 0.85,  # Employed
        "CE": 0.90,  # Self-employed
        "CF": 0.80,  # Retired
        "CG": 0.50,  # Unemployed
    }
    return mapping.get(str(emp), 0.7)


def _map_housing(hs: str) -> float:
    """Stability proxy from housing status."""
    mapping = {
        "BA": 0.9,  # Owner with mortgage
        "BB": 0.95, # Owner outright
        "BC": 0.6,  # Renter
        "BD": 0.5,  # Living with parents
        "BE": 0.4,  # Other
    }
    return mapping.get(str(hs), 0.6)


def _row_to_features(row) -> dict:
    """Convert one BAF row to our 59-feature schema."""
    # ── Velocity / graph behavioral signals ──────────────────────────────
    vel_6h = float(row.get("velocity_6h", 0) or 0)
    vel_24h = float(row.get("velocity_24h", 0) or 0)
    vel_4w = float(row.get("velocity_4w", 0) or 0)

    zip_count = float(row.get("zip_count_4w", 1) or 1)
    branch_count = float(row.get("bank_branch_count_8w", 1) or 1)
    dob_emails = float(row.get("date_of_birth_distinct_emails_4w", 0) or 0)
    device_emails = float(row.get("device_distinct_emails_8w", 0) or 0)
    device_fraud = float(row.get("device_fraud_count", 0) or 0)

    credit_score = float(row.get("credit_risk_score", 500) or 500)
    bank_months = float(row.get("bank_months_count", 12) or 12)
    addr_months = float(row.get("current_address_months_count", 12) or 12)
    prev_addr_months = float(row.get("prev_address_months_count", 12) or 12)

    proposed_limit = float(row.get("proposed_credit_limit", 1000) or 1000)
    income = float(row.get("income", 0.5) or 0.5)  # normalized 0-1 in BAF
    session_len = float(row.get("session_length_in_minutes", 10) or 10)
    days_since_req = float(row.get("days_since_request", 0) or 0)

    is_foreign = int(row.get("foreign_request", 0) or 0)
    email_is_free = int(row.get("email_is_free", 0) or 0)
    phone_home_ok = int(row.get("phone_home_valid", 1) or 1)
    phone_mob_ok = int(row.get("phone_mobile_valid", 1) or 1)
    has_other_cards = int(row.get("has_other_cards", 0) or 0)
    keep_alive = int(row.get("keep_alive_session", 0) or 0)

    emp_status = str(row.get("employment_status", "CD"))
    housing_status = str(row.get("housing_status", "BA"))
    source = str(row.get("source", "INTERNET"))

    # ── Graph feature derivations ─────────────────────────────────────────

    # BAF velocities are system-wide rates: vel_6h / 6 = hourly rate in that window
    # vel_4w / (28 * 24) = long-term hourly baseline
    hourly_recent = vel_6h / max(6.0, 1.0)
    hourly_baseline = vel_4w / max(28.0 * 24.0, 1.0)
    burst_ratio = hourly_recent / max(hourly_baseline, 0.01)
    # burst_score: clamp burst_ratio / 50 → [0,1] (50x burst = fully suspicious)
    burst = _clamp(burst_ratio / 50.0)
    velocity_ratio = _clamp(burst_ratio / 10.0, 0.0, 30.0)  # kept in natural units

    # pagerank_fraud_seeded: high if device_fraud_count > 0, low credit score, free email
    # credit_risk_score range: -144 to 389; lower = riskier
    pagerank_fraud = _clamp(
        (device_fraud / max(1.0, device_fraud + 3)) * 0.5
        + (1 - _normalize(credit_score, -150, 400)) * 0.35
        + email_is_free * 0.15
    )

    # community_fraud_ratio: many people from same zip/branch applying recently
    community_fraud = _clamp(
        _normalize(zip_count, 1, 2000) * 0.3
        + _normalize(dob_emails, 0, 20) * 0.4
        + _normalize(device_emails, 0, 10) * 0.3
    )

    # shortest_path_to_fraud: device_fraud_count > 0 → 0 hops; else scale with credit score
    if device_fraud > 0:
        shortest_path = 0.0
    elif community_fraud > 0.6:
        shortest_path = 1.0
    elif pagerank_fraud > 0.5:
        shortest_path = 2.0
    else:
        shortest_path = _clamp(2.0 + _normalize(credit_score, -150, 400) * 4.0, 0.0, 6.0)

    # bipartite_score: many people (zip_count/branch_count) → same destination (mule account)
    bipartite_score = _clamp(
        _normalize(zip_count, 1, 2000) * 0.5
        + _normalize(branch_count, 1, 50) * 0.5
    )

    # sink_score: intended_balcon_amount (transfer-in) vs income signals mule setup
    # intended_balcon_amount can be -1 (no transfer) to large positive
    intended_balcon = float(row.get("intended_balcon_amount", 0) or 0)
    if income > 0 and intended_balcon > 0:
        balcon_income_ratio = intended_balcon / (income * 50000 + 1)  # INR proxy
        sink_score = _clamp(_normalize(balcon_income_ratio, 0, 5))
    elif intended_balcon > 1000:
        sink_score = _clamp(_normalize(intended_balcon, 0, 10000))
    else:
        sink_score = _clamp((1 - _normalize(credit_score, -150, 400)) * 0.4)

    # counterparty_novelty: many distinct emails from same device → many identity attempts
    counterparty_novelty = _clamp(_normalize(device_emails, 0, 20))

    # account_age_days: use bank_months_count (capped at -1 = unknown in BAF)
    account_age_days = float(max(0, bank_months) * 30)
    if account_age_days == 0:
        account_age_days = float(max(0, addr_months) * 30)

    # kyc_completeness_score: phone validity, employment, housing stability
    kyc_score = _clamp(
        _map_employment(emp_status) * 0.4
        + _map_housing(housing_status) * 0.3
        + float(phone_mob_ok) * 0.15
        + float(phone_home_ok) * 0.1
        + (1.0 - email_is_free * 0.5) * 0.05
    )

    # geography_switch: foreign_request (definitive) + address instability
    prev_addr = max(-1, int(row.get("prev_address_months_count", 12) or 12))
    geo_switch = _clamp(
        float(is_foreign) * 0.8
        + (0.2 if prev_addr in (-1, 0, 1, 2) else 0.0)
    )

    # bridge_node_probability: multiple identities + new email + high velocity
    bridge_prob = _clamp(
        _normalize(dob_emails, 0, 10) * 0.4
        + _normalize(device_emails, 0, 10) * 0.3
        + burst * 0.3
    )

    # dormancy_reactivation_flag: very short session + high velocity → scripted/bot
    dormancy_flag = 1.0 if (session_len < 2.0 and vel_6h >= 3) else 0.0
    dormancy_break = dormancy_flag

    # channel_entropy: INTERNET has higher entropy than TELEAPP
    channel_entropy = 1.5 if source == "INTERNET" else 0.5
    channel_switch = _clamp(float(has_other_cards) * 0.5 + (1 - float(phone_home_ok)) * 0.3)

    # amount_zscore: proposed limit vs typical (normalize 0-10000 range)
    amount_zscore = _clamp(_normalize(proposed_limit, 100, 10000) * 5.0, 0.0, 10.0)

    # Temporal features (BAF doesn't have timestamps, use month as proxy)
    month = int(row.get("month", 6) or 6)
    hour = 12.0  # unknown, default midday
    is_night = 0.0
    is_weekend = 0.0
    hour_deviation = 0.0
    is_festival = 1.0 if month in (10, 11) else 0.0
    night_txn_ratio = 0.1 if dormancy_flag == 0.0 else 0.6
    weekend_txn_ratio = 0.3

    # Amount features — use proposed_credit_limit as amount proxy (INR scale: ×10 for lakh)
    amount = proposed_limit * 10.0  # rough INR proxy
    amount_log = math.log1p(amount)
    amount_rounded = 1.0 if amount == round(amount, -3) else 0.0

    # Channel (BAF has no channel info — default to UPI)
    ch_upi, ch_imps, ch_rtgs, ch_neft = 1.0, 0.0, 0.0, 0.0

    # Transaction counts from velocity features
    txn_count_1h = vel_6h / 6.0
    txn_count_24h = vel_24h
    txn_count_7d = vel_4w / 4.0
    txn_volume_1h = amount * txn_count_1h
    txn_volume_24h = amount * txn_count_24h

    # Derived graph features
    degree_centrality = _clamp(_normalize(vel_4w, 0, 100))
    betweenness_centrality = _clamp(bridge_prob * 0.1)
    clustering_coefficient = _clamp(1.0 - counterparty_novelty)
    cycle_membership = 0.0  # BAF is application-level, can't derive cycles
    fan_out_ratio = _clamp(1.0 - sink_score)
    temporal_acceleration = velocity_ratio
    cash_mule_sink_score = _clamp(sink_score * 0.7 + (1 - kyc_score) * 0.3)
    return_ratio = _clamp(sink_score)  # mule holds received money → low return
    distinct_counterparties = _clamp(device_emails + zip_count / 10.0, 0.0, 50.0)
    payee_vpa_age = float(max(1, days_since_req))
    payee_in_alert = 1.0 if device_fraud > 0 else 0.0
    payee_alert_count = device_fraud

    txn_count_30d = vel_4w
    txn_count_90d = vel_4w * 3
    txn_count_all = vel_4w * 12
    avg_txn_amount_30d = amount
    amount_series_score = _clamp(_normalize(dob_emails, 0, 5))

    community_id = float(hash(str(row.get("employment_status", "CD"))) % 200)

    return {
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality,
        "clustering_coefficient": clustering_coefficient,
        "pagerank_fraud_seeded": pagerank_fraud,
        "community_id": community_id,
        "community_fraud_ratio": community_fraud,
        "shortest_path_to_fraud": shortest_path,
        "cycle_membership": cycle_membership,
        "sink_score": sink_score,
        "bipartite_score": bipartite_score,
        "fan_out_ratio": fan_out_ratio,
        "temporal_acceleration": temporal_acceleration,
        "cash_mule_sink_score": cash_mule_sink_score,
        "bridge_node_probability": bridge_prob,
        "dormancy_reactivation_flag": dormancy_flag,
        "account_age_days": account_age_days,
        "kyc_completeness_score": kyc_score,
        "txn_count_30d": txn_count_30d,
        "txn_count_90d": txn_count_90d,
        "txn_count_all": txn_count_all,
        "avg_txn_amount_30d": avg_txn_amount_30d,
        "distinct_counterparties_30d": distinct_counterparties,
        "channel_entropy": channel_entropy,
        "night_txn_ratio": night_txn_ratio,
        "weekend_txn_ratio": weekend_txn_ratio,
        "return_ratio": return_ratio,
        "amount_zscore": amount_zscore,
        "counterparty_novelty": counterparty_novelty,
        "hour_deviation": hour_deviation,
        "channel_switch": channel_switch,
        "amount_series_score": amount_series_score,
        "burst_score": burst,
        "velocity_ratio": velocity_ratio,
        "dormancy_break": dormancy_break,
        "geography_switch": geo_switch,
        "txn_amount": amount,
        "txn_amount_log": amount_log,
        "txn_amount_rounded": amount_rounded,
        "channel_upi": ch_upi, "channel_imps": ch_imps,
        "channel_rtgs": ch_rtgs, "channel_neft": ch_neft,
        "hour_of_day": hour,
        "day_of_week": 1.0,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_festival_period": is_festival,
        "amount_vs_threshold_50000": amount / 50000,
        "amount_vs_threshold_100000": amount / 100000,
        "amount_vs_threshold_1000000": amount / 1000000,
        "payee_vpa_age_days": payee_vpa_age,
        "txn_count_last_1h": txn_count_1h,
        "txn_count_last_24h": txn_count_24h,
        "txn_count_last_7d": txn_count_7d,
        "txn_volume_last_1h": txn_volume_1h,
        "txn_volume_last_24h": txn_volume_24h,
        "distinct_payees_24h": distinct_counterparties,
        "payee_in_alert_log": payee_in_alert,
        "payee_shared_alert_count": payee_alert_count,
    }


def main(max_rows: int = 200_000) -> None:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("ERROR: pip install pandas numpy")
        sys.exit(1)

    try:
        csv_path = _find_baf_csv()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"Loaded {len(df)} rows")

    if "fraud_bool" not in df.columns:
        print("ERROR: fraud_bool column not found. Columns:", list(df.columns))
        sys.exit(1)

    n_fraud = int(df["fraud_bool"].sum())
    n_legit = len(df) - n_fraud
    print(f"Fraud: {n_fraud} ({100*n_fraud/len(df):.2f}%), Legit: {n_legit}")

    rows_X = []
    rows_y = []

    for _, row in df.iterrows():
        feats = _row_to_features(row)
        rows_X.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        rows_y.append(int(row["fraud_bool"]))

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "baf_X.npy", X)
    np.save(OUTPUT_DIR / "baf_y.npy", y)
    print(f"\nSaved → {OUTPUT_DIR}/baf_X.npy  shape={X.shape}")
    print(f"Saved → {OUTPUT_DIR}/baf_y.npy  shape={y.shape}")
    print("Re-run: python ml/train.py  (will auto-merge BAF data)")


if __name__ == "__main__":
    main()
