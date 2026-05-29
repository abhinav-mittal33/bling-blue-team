"""
ml/train_xgbod.py -- Train XGBOD second novelty layer (P4-6).

XGBOD (Extreme Gradient Boosting Outlier Detection) is a semi-supervised novelty
detector. It accepts weak labels: 0 for legitimate accounts (50,000 samples) and
1 for novel anomalies (500 synthetic out-of-distribution samples).

INVARIANT (ABSOLUTE): XGBOD result NEVER enters fraud_score.
                      NEVER shown to investigators.
                      Output routes to novelty_queue with source='xgbod'.
                      This is developer-only signal for new pattern discovery.

Outputs:
  ml/models/xgbod_v1.joblib          -- trained XGBOD model
  ml/models/xgbod_threshold.json     -- {xgbod_threshold: float} at 97th percentile

See also: migration 007_novelty_source.py adds source column to novelty_queue.

Run: python ml/train_xgbod.py
"""
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

import structlog

log = structlog.get_logger()

random.seed(42)

# Reuse the same continuous feature set as ECOD for consistency
_BINARY_EXCLUDE = {
    "cycle_membership",
    "dormancy_reactivation_flag",
    "dormancy_break",
    "is_night",
    "is_weekend",
    "is_festival_period",
    "channel_upi",
    "channel_imps",
    "channel_rtgs",
    "channel_neft",
    "payee_in_alert_log",
}

from ml.train import _legit_features  # noqa: E402
from ml.feature_registry import FEATURE_NAMES_V2  # noqa: E402

XGBOD_FEATURE_NAMES = [f for f in FEATURE_NAMES_V2 if f not in _BINARY_EXCLUDE]


def _legit_vector() -> list:
    """Build one continuous feature vector from a synthetic legitimate account."""
    f = _legit_features()
    return [float(f.get(name, 0.0) or 0.0) for name in XGBOD_FEATURE_NAMES]


def _novel_anomaly_vector() -> list:
    """
    Generate one synthetic novel anomaly: values outside the 99th percentile of
    the legitimate distribution.

    Four out-of-distribution patterns, rotated randomly:
      0 -- extreme amount (500K-5M range)
      1 -- ghost account (near-zero history, large single outflow)
      2 -- hyper-velocity relay (temporal_acceleration + velocity extreme)
      3 -- after-hours concentration (all txns in 3-4am window)

    These mirror what is used in train_ecod.py for consistency across
    both novelty detectors.
    """
    f = _legit_features()
    pattern = random.randint(0, 3)

    if pattern == 0:
        f["txn_amount"] = random.uniform(500_000, 5_000_000)
        f["amount_zscore"] = random.uniform(15.0, 50.0)
        f["amount_vs_threshold_1000000"] = f["txn_amount"] / 1_000_000
        f["txn_amount_log"] = math.log1p(f["txn_amount"])

    elif pattern == 1:
        f["pagerank_fraud_seeded"] = random.uniform(0.0, 0.0005)
        f["degree_centrality"] = random.uniform(0.0, 0.001)
        f["txn_count_all"] = random.uniform(0.0, 1.0)
        f["txn_count_30d"] = random.uniform(0.0, 0.5)
        f["txn_amount"] = random.uniform(300_000, 1_000_000)
        f["amount_zscore"] = random.uniform(20.0, 60.0)

    elif pattern == 2:
        f["temporal_acceleration"] = random.uniform(40.0, 100.0)
        f["velocity_ratio"] = random.uniform(50.0, 200.0)
        f["burst_score"] = random.uniform(0.98, 1.0)
        f["txn_count_last_1h"] = float(random.randint(50, 200))
        f["txn_count_last_24h"] = float(random.randint(200, 1000))

    else:
        f["hour_deviation"] = random.uniform(10.0, 20.0)
        f["night_txn_ratio"] = random.uniform(0.98, 1.0)
        f["txn_count_last_1h"] = float(random.randint(20, 80))
        f["payee_vpa_age_days"] = random.uniform(0.0, 0.5)
        f["counterparty_novelty"] = random.uniform(0.98, 1.0)

    return [float(f.get(name, 0.0) or 0.0) for name in XGBOD_FEATURE_NAMES]


def main() -> None:
    try:
        import numpy as np
        import joblib
        from pyod.models.xgbod import XGBOD
    except ImportError as exc:
        log.error("xgbod.import_error", error=str(exc),
                  hint="pip install pyod==2.0.2  (already in requirements.txt)")
        sys.exit(1)

    # Synthetic training: 5K samples (fast on CPU — takes ~2min vs 90min at 50K).
    # Re-run on real data: N_LEGIT=50_000, N_NOVEL=500 (set via env var XGBOD_FULL_TRAIN=1).
    import os as _os
    _full = _os.getenv("XGBOD_FULL_TRAIN") == "1"
    N_LEGIT = 50_000 if _full else 5_000
    N_NOVEL = 500 if _full else 50
    N_TOTAL = N_LEGIT + N_NOVEL

    log.info("xgbod.generating_data",
             n_legit=N_LEGIT, n_novel=N_NOVEL, n_features=len(XGBOD_FEATURE_NAMES))

    rows_X, rows_y = [], []

    # Legitimate samples -- labeled 0
    for _ in range(N_LEGIT):
        rows_X.append(_legit_vector())
        rows_y.append(0)

    # Novel anomaly samples -- labeled 1 (weak supervised signal)
    for _ in range(N_NOVEL):
        rows_X.append(_novel_anomaly_vector())
        rows_y.append(1)

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.int32)

    # Replace NaN / inf produced by edge cases in feature generation
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # n_estimators=50 for production; 5 for synthetic (5K samples fit in ~2min CPU).
    n_est = 50 if _full else 5
    log.info("xgbod.fitting",
             shape=list(X.shape),
             n_labeled=int(y.sum()),
             contamination=0.01,
             n_estimators=n_est)

    # XGBOD is semi-supervised: it uses the weak labels (y) to guide
    # the ensemble of base detectors toward the novel pattern signal.
    # contamination=0.01 sets the decision boundary.
    xgbod_model = XGBOD(n_estimators=n_est, contamination=0.01, random_state=42)
    xgbod_model.fit(X, y)

    # decision_scores_: higher = more anomalous (same convention as ECOD)
    train_scores = xgbod_model.decision_scores_

    # Threshold at 97th percentile of all training scores
    threshold = float(np.percentile(train_scores, 97))
    flagged_frac = float((train_scores > threshold).mean())

    log.info("xgbod.threshold_computed",
             threshold=round(threshold, 4),
             flagged_fraction=round(flagged_frac, 4))

    # -- Save model ------------------------------------------------------------
    model_path = MODELS_DIR / "xgbod_v1.joblib"
    joblib.dump(xgbod_model, str(model_path))
    log.info("xgbod.saved_model", path=str(model_path))

    # -- Save threshold --------------------------------------------------------
    thresh_path = MODELS_DIR / "xgbod_threshold.json"
    with open(thresh_path, "w") as fh:
        json.dump({"xgbod_threshold": threshold}, fh)
    log.info("xgbod.saved_threshold", path=str(thresh_path))

    # -- Sanity: novel anomaly scores should exceed legit scores ---------------
    legit_scores = train_scores[:N_LEGIT]
    novel_scores = train_scores[N_LEGIT:]

    print(f"\n{chr(8212)*60}")
    print("XGBOD -- Training Summary")
    print(f"{chr(8212)*60}")
    print(f"Feature count (continuous): {len(XGBOD_FEATURE_NAMES)}")
    print(f"Total samples:              {N_TOTAL}")
    print(f"  Legit (label=0):          {N_LEGIT}")
    print(f"  Novel anomaly (label=1):  {N_NOVEL}")
    print(f"n_estimators:               50")
    print(f"contamination:              0.01")
    print(f"Threshold (97th pct):       {threshold:.4f}")
    print(f"Flagged fraction:           {flagged_frac:.4f} ({flagged_frac*100:.1f}%)")
    print(f"Legit mean score:           {legit_scores.mean():.4f}")
    print(f"Novel mean score:           {novel_scores.mean():.4f}")
    print(f"Novel > threshold:          "
          f"{(novel_scores > threshold).mean():.2%} of injected anomalies caught")
    print(f"Model saved:                {model_path}")
    print(f"Threshold saved:            {thresh_path}")
    print(f"{chr(8212)*60}")
    print("INVARIANT: XGBOD scores NEVER enter fraud_score.")
    print("           Route to novelty_queue with source='xgbod' only.")
    print("           See migration 007_novelty_source.py for the source column.")

    if novel_scores.mean() <= legit_scores.mean():
        log.warning("xgbod.sanity_check_failed",
                    note="Novel anomaly scores not higher than legit -- "
                         "check feature generation or XGBOD version")


if __name__ == "__main__":
    main()
