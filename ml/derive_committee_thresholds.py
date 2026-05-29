"""
Offline script: Derive committee-specific thresholds from shadow data.

Gate: aborts if shadow_score_committee has fewer than 50,000 rows.

Methodology (matching ml/train.py):
  LOG       at recall=0.95 on labeled fraud
  REVIEW    at F1-max (precision*recall balance)
  HIGH_RISK at precision=0.90

Warns if LOG threshold shifts >0.05 from current settings.threshold_log
(manual review required before deploying to production).

Saves: ml/models/committee_thresholds.json
Deploy: set LOG_THRESHOLD_COMMITTEE, REVIEW_THRESHOLD_COMMITTEE,
        HIGH_RISK_THRESHOLD_COMMITTEE in .env, then restart API.

Run: python ml/derive_committee_thresholds.py
     python ml/derive_committee_thresholds.py --min-rows 10000  # for testing
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings   # noqa: E402

MIN_ROWS_DEFAULT = 50_000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "models", "committee_thresholds.json")


def _compute_thresholds(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Derive LOG/REVIEW/HIGH_RISK thresholds on given score+label arrays."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # LOG: recall = 0.95 (highest threshold where recall ≥ 0.95)
    recall_mask = recall >= 0.95
    if recall_mask.any():
        log_threshold = float(thresholds[recall_mask[:-1]][-1]) if len(thresholds) > 0 else 0.38
    else:
        log_threshold = float(thresholds[0]) if len(thresholds) > 0 else 0.38
        print("WARNING: cannot achieve recall=0.95. Using minimum available threshold.")

    # REVIEW: F1-max
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    if len(thresholds) > 0:
        review_threshold = float(thresholds[np.argmax(f1[:-1])])
    else:
        review_threshold = 0.62

    # HIGH_RISK: precision = 0.90 (lowest threshold achieving ≥ 0.90 precision)
    prec_mask = precision >= 0.90
    if prec_mask.any():
        # Find the lowest threshold (rightmost in precision_recall_curve) with prec ≥ 0.90
        prec_indices = np.where(prec_mask[:-1])[0]
        high_risk_threshold = float(thresholds[prec_indices[0]])
    else:
        high_risk_threshold = 0.83
        print("WARNING: cannot achieve precision=0.90. Using default HIGH_RISK threshold.")

    return {
        "log_threshold": round(log_threshold, 4),
        "review_threshold": round(review_threshold, 4),
        "high_risk_threshold": round(high_risk_threshold, 4),
    }


def derive(min_rows: int) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(settings.postgres_url)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        count = db.execute(
            text("SELECT COUNT(*) FROM shadow_score_committee WHERE live_action IS NOT NULL")
        ).scalar() or 0

        if count < min_rows:
            print(f"ABORT: only {count} shadow rows (need {min_rows}).")
            print("Continue accumulating shadow data. Use --min-rows to override for testing.")
            sys.exit(1)

        print(f"Fetching {count} shadow rows...")
        rows = db.execute(
            text("""
                SELECT final_committee_score, live_action
                FROM shadow_score_committee
                WHERE final_committee_score IS NOT NULL
                  AND live_action IS NOT NULL
                ORDER BY scored_at DESC
                LIMIT :limit
            """),
            {"limit": min_rows * 2},
        ).fetchall()
    finally:
        db.close()

    if not rows:
        print("ABORT: no rows with final_committee_score.")
        sys.exit(1)

    # We don't have ground-truth labels in shadow rows (investigators haven't reviewed these yet).
    # Use live_action as a proxy: HIGH_RISK/REVIEW → label=1, PASS/LOG → label=0.
    # This is an approximation — proper labels come from curated_dataset_queue after Phase 3.
    scores = np.array([float(r.final_committee_score) for r in rows], dtype=np.float32)
    proxy_labels = np.array([1 if r.live_action in ("REVIEW", "HIGH_RISK") else 0 for r in rows], dtype=np.int32)

    fraud_count = proxy_labels.sum()
    print(f"Proxy distribution: {fraud_count} fraud-proxy, {len(proxy_labels)-fraud_count} benign-proxy")

    if fraud_count < 50:
        print("WARNING: fewer than 50 fraud-proxy rows. Thresholds may not be reliable.")

    thresholds = _compute_thresholds(scores, proxy_labels)

    # Drift warning
    current_log = settings.threshold_log
    new_log = thresholds["log_threshold"]
    if abs(new_log - current_log) > 0.05:
        print(f"\n⚠  WARNING: LOG threshold shifted {abs(new_log - current_log):.3f} "
              f"(current={current_log}, derived={new_log}).")
        print("   Manual review required before deploying. Check shadow data distribution.")

    print(f"\nDerived thresholds:")
    print(f"  LOG:       {thresholds['log_threshold']} (current: {settings.threshold_log})")
    print(f"  REVIEW:    {thresholds['review_threshold']} (current: {settings.threshold_review})")
    print(f"  HIGH_RISK: {thresholds['high_risk_threshold']} (current: {settings.threshold_high_risk})")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nSaved: {os.path.abspath(OUTPUT_PATH)}")
    print("\nDeploy by setting in .env and restarting API:")
    print(f"  THRESHOLD_LOG_COMMITTEE={thresholds['log_threshold']}")
    print(f"  THRESHOLD_REVIEW_COMMITTEE={thresholds['review_threshold']}")
    print(f"  THRESHOLD_HIGH_RISK_COMMITTEE={thresholds['high_risk_threshold']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive committee thresholds from shadow data")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS_DEFAULT)
    args = parser.parse_args()
    derive(args.min_rows)
