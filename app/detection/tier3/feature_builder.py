"""
Tier 3 feature builder — Phase 2 update (P2-7, P2-8, P2-9).
Assembles features from Redis (pre-computed) + PostgreSQL (real-time).
Returns dict: feature_name → float. Missing features return float('nan').

Feature order follows ml/feature_registry.py — NEVER hardcode the list here.
XGBoost assigns features by position: feature order must be identical between
training (train.py) and inference (here).
"""
from __future__ import annotations
import math
import time
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.utils.redis_client import get_graph_features
from app.models.schemas import TransactionScoreRequest

# Authoritative feature name list — import here, never hardcode
from ml.feature_registry import GRAPH_FEATURES, REALTIME_FEATURES, PHASE2_GRAPH_FEATURES


def build_features(txn: TransactionScoreRequest, db: Session) -> dict[str, float]:
    features: dict[str, float] = {}

    # ── Pre-computed graph features from Redis ────────────────────────────────
    cached = get_graph_features(txn.account_id)

    for name in GRAPH_FEATURES:
        val = cached.get(name)
        features[name] = float(val) if val is not None else float("nan")

    # ── graph_staleness_hours (P2-7) ───────────────────────────────────────────
    # Derived at scoring time from _last_updated field — not stored as a feature.
    last_updated = cached.get("_last_updated")
    if last_updated:
        staleness_hours = (time.time() - float(last_updated)) / 3600
    else:
        staleness_hours = float("nan")
    features["graph_staleness_hours"] = staleness_hours

    # ── Phase 2 graph features from Redis (P2-7, P2-8, P2-9) ─────────────────
    for name in PHASE2_GRAPH_FEATURES:
        if name == "graph_staleness_hours":
            continue  # Already computed above
        val = cached.get(name)
        features[name] = float(val) if val is not None else float("nan")

    # ── Real-time tabular features from PostgreSQL ────────────────────────────
    amount = float(txn.amount)
    ts = txn.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    features["txn_amount"] = amount
    features["txn_amount_log"] = math.log1p(amount)
    features["txn_amount_rounded"] = 1.0 if amount == round(amount, -3) else 0.0
    features["channel_upi"] = 1.0 if txn.channel == "UPI" else 0.0
    features["channel_imps"] = 1.0 if txn.channel == "IMPS" else 0.0
    features["channel_rtgs"] = 1.0 if txn.channel == "RTGS" else 0.0
    features["channel_neft"] = 1.0 if txn.channel == "NEFT" else 0.0
    features["hour_of_day"] = float(ts.hour)
    features["day_of_week"] = float(ts.weekday())
    features["is_weekend"] = 1.0 if ts.weekday() >= 5 else 0.0
    features["is_night"] = 1.0 if (ts.hour >= 23 or ts.hour < 5) else 0.0

    # Festival period: Oct 1 – Nov 15 (Navratri + Diwali window)
    features["is_festival_period"] = 1.0 if (
        (ts.month == 10) or (ts.month == 11 and ts.day <= 15)
    ) else 0.0

    for threshold in [50_000, 1_00_000, 10_00_000]:
        features[f"amount_vs_threshold_{threshold}"] = amount / threshold

    # Payee VPA age
    payee_vpa_age = float("nan")
    if txn.payee_vpa_created_at:
        vpa_ts = txn.payee_vpa_created_at
        if vpa_ts.tzinfo is None:
            vpa_ts = vpa_ts.replace(tzinfo=timezone.utc)
        payee_vpa_age = max(0.0, (ts - vpa_ts).days)
    features["payee_vpa_age_days"] = payee_vpa_age

    # Velocity windows from PostgreSQL
    try:
        vel_rows = db.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') AS count_1h,
                    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '24 hours') AS count_24h,
                    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '7 days') AS count_7d,
                    SUM(amount) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') AS vol_1h,
                    SUM(amount) FILTER (WHERE timestamp > NOW() - INTERVAL '24 hours') AS vol_24h,
                    COUNT(DISTINCT payee_account_id) FILTER (WHERE timestamp > NOW() - INTERVAL '24 hours') AS distinct_payees_24h
                FROM transactions
                WHERE account_id = :account_id
            """),
            {"account_id": txn.account_id},
        ).fetchone()

        if vel_rows:
            features["txn_count_last_1h"] = float(vel_rows.count_1h or 0)
            features["txn_count_last_24h"] = float(vel_rows.count_24h or 0)
            features["txn_count_last_7d"] = float(vel_rows.count_7d or 0)
            features["txn_volume_last_1h"] = float(vel_rows.vol_1h or 0)
            features["txn_volume_last_24h"] = float(vel_rows.vol_24h or 0)
            features["distinct_payees_24h"] = float(vel_rows.distinct_payees_24h or 0)
    except Exception:
        for k in ["txn_count_last_1h", "txn_count_last_24h", "txn_count_last_7d",
                  "txn_volume_last_1h", "txn_volume_last_24h", "distinct_payees_24h"]:
            features[k] = float("nan")

    # Payee alert history
    try:
        payee_alert_count = db.execute(
            text("""
                SELECT COUNT(*) FROM alerts a
                JOIN transactions t ON t.id = a.transaction_id
                WHERE t.account_id = :payee_id
            """),
            {"payee_id": txn.payee_account_id or ""},
        ).scalar()
        features["payee_in_alert_log"] = 1.0 if (payee_alert_count or 0) > 0 else 0.0
        features["payee_shared_alert_count"] = float(payee_alert_count or 0)
    except Exception:
        features["payee_in_alert_log"] = float("nan")
        features["payee_shared_alert_count"] = float("nan")

    # ── Phase 3 real-time features ────────────────────────────────────────────

    # P3-5: Micro test payment flag — amount <₹2 to new VPA = mule setup signal
    features["micro_test_payment"] = 1.0 if (
        amount < 2.0
        and not math.isnan(payee_vpa_age)
        and payee_vpa_age < 7
    ) else 0.0

    # P3-6: Benford deviation — leading-digit deviation from Benford's law
    features["benford_deviation"] = _compute_benford_deviation(txn.account_id, db)

    # P3-9: Fan-in sender z-score — unusual concentration of incoming senders today
    features["fan_in_sender_zscore"] = _compute_fan_in_zscore(txn.account_id, db)

    return features


def _compute_benford_deviation(account_id: str, db) -> float:
    """
    Compute Benford's law deviation for last 90-day transaction amounts.
    Returns 0.0 (no deviation) to 1.0 (max deviation). Missing → nan.
    """
    import math
    try:
        rows = db.execute(
            text("""
                SELECT amount FROM transactions
                WHERE account_id = :account_id
                AND timestamp > NOW() - INTERVAL '90 days'
                AND amount > 0
                LIMIT 200
            """),
            {"account_id": account_id},
        ).fetchall()

        if len(rows) < 20:
            return float("nan")  # Not enough data for Benford analysis

        # Expected Benford frequencies for digits 1-9
        benford_expected = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

        # Actual leading digit frequencies
        digit_counts = {d: 0 for d in range(1, 10)}
        for row in rows:
            s = str(abs(float(row[0]))).lstrip("0.")
            if not s:
                continue  # zero amount — no valid leading digit
            leading = int(s[0])
            if 1 <= leading <= 9:
                digit_counts[leading] += 1

        n = sum(digit_counts.values())
        if n == 0:
            return float("nan")

        # Chi-squared-like deviation from Benford's distribution
        deviation = sum(
            abs((digit_counts[d] / n) - benford_expected[d])
            for d in range(1, 10)
        )
        # Normalize to [0, 1]: max theoretical deviation is ~2.0
        return min(deviation / 2.0, 1.0)

    except Exception:
        return float("nan")


def _compute_fan_in_zscore(account_id: str, db) -> float:
    """
    Fan-in sender z-score: how many unique accounts sent money TO this account today
    vs 90-day daily historical baseline. High positive = potential collector node.

    Measures INCOMING senders (payee_account_id = this account), not outgoing.
    Uses CTEs to avoid cross-join ambiguity between subqueries.
    """
    try:
        result = db.execute(
            text("""
                WITH incoming_daily AS (
                    SELECT DATE(timestamp) AS day,
                           COUNT(DISTINCT account_id) AS daily_cnt
                    FROM transactions
                    WHERE payee_account_id = :account_id
                      AND timestamp > NOW() - INTERVAL '90 days'
                    GROUP BY DATE(timestamp)
                ),
                today AS (
                    SELECT COUNT(DISTINCT account_id) AS senders_today
                    FROM transactions
                    WHERE payee_account_id = :account_id
                      AND timestamp > NOW() - INTERVAL '24 hours'
                )
                SELECT
                    today.senders_today,
                    STDDEV(incoming_daily.daily_cnt) AS std_daily,
                    AVG(incoming_daily.daily_cnt)    AS avg_daily
                FROM incoming_daily, today
            """),
            {"account_id": account_id},
        ).fetchone()

        if not result or result.std_daily is None or float(result.std_daily or 0) == 0:
            return float("nan")

        zscore = (float(result.senders_today or 0) - float(result.avg_daily or 0)) / float(result.std_daily)
        return round(zscore, 3)

    except Exception:
        return float("nan")
