"""
ml/train_ecod.py -- Train ECOD novelty detector for the PASS-stream discovery pipeline.

ECOD (Empirical Cumulative distribution-based Outlier Detection) runs on the stream
of low-scoring transactions (fraud_score < LOG_THRESHOLD) to detect structurally
anomalous but XGBoost-unseen patterns.

Training data: 50,000 legitimate-profile feature vectors (continuous features from
FEATURE_NAMES_V2) + 500 synthetic novel anomalies with extreme values.

INVARIANT: ECOD result NEVER enters fraud_score and is NEVER shown to investigators.
           Output goes to novelty_queue (developer review only). Same separation
           as Isolation Forest.

Outputs:
  ml/models/ecod_v1.joblib          -- trained ECOD model
  ml/models/ecod_threshold.json     -- {ecod_threshold: float} at 97th percentile

Run: python ml/train_ecod.py
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

# -- Feature set for ECOD -----------------------------------------------------
# Use continuous features from FEATURE_NAMES_V2 only.
# Exclude binary flags: time/channel/event flags vary legitimately and would
# cause IF-style false positives on night workers, weekend payroll etc.
# See architecture.md: "IF uses 17 structural features, not all 59".
# For ECOD we use more features (continuous tabular), but still exclude binary flags.

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

# Import archetype generators for synthetic legit data
from ml.train import _legit_features, _fraud_base_overrides  # noqa: E402
from ml.feature_registry import FEATURE_NAMES_V2             # noqa: E402

# Build the continuous feature name list at module level
ECOD_FEATURE_NAMES = [f for f in FEATURE_NAMES_V2 if f not in _BINARY_EXCLUDE]


def _legit_vector() -> list:
    """Build one continuous feature vector from a synthetic legitimate account."""
    f = _legit_features()
    return [float(f.get(name, 0.0) or 0.0) for name in ECOD_FEATURE_NAMES]


def _novel_anomaly_vector(legit_stats: dict) -> list:
    """
    Generate one synthetic novel anomaly: values outside 99th percentile of
    the legitimate distribution, simulating evasion patterns not present in
    the training archetypes.

    Novel patterns modelled here:
    - Extreme txn_amount (10x+ legitimate max)
    - Near-zero pagerank + near-zero degree_centrality simultaneously
    - Extreme temporal_acceleration + extreme velocity_ratio
    - Near-zero txn_count_all (ghost account with zero history, large amount)
    """
    import numpy as np

    # Start from a legit base and push key fields to extremes
    f = _legit_features()

    pattern = random.randint(0, 3)

    if pattern == 0:
        # Extreme amount -- 10x to 50x typical maximum
        f["txn_amount"] = random.uniform(500_000, 5_000_000)
        f["amount_zscore"] = random.uniform(15.0, 50.0)
        f["amount_vs_threshold_1000000"] = f["txn_amount"] / 1_000_000
        f["txn_amount_log"] = math.log1p(f["txn_amount"])

    elif pattern == 1:
        # Ghost account: near-zero network footprint but suddenly active
        f["pagerank_fraud_seeded"] = random.uniform(0.0, 0.0005)
        f["degree_centrality"] = random.uniform(0.0, 0.001)
        f["txn_count_all"] = random.uniform(0.0, 1.0)
        f["txn_count_30d"] = random.uniform(0.0, 0.5)
        f["txn_amount"] = random.uniform(300_000, 1_000_000)
        f["amount_zscore"] = random.uniform(20.0, 60.0)

    elif pattern == 2:
        # Hyper-velocity relay: extreme acceleration + volume spike
        f["temporal_acceleration"] = random.uniform(40.0, 100.0)
        f["velocity_ratio"] = random.uniform(50.0, 200.0)
        f["burst_score"] = random.uniform(0.98, 1.0)
        f["txn_count_last_1h"] = float(random.randint(50, 200))
        f["txn_count_last_24h"] = float(random.randint(200, 1000))

    else:
        # Unusual hour pattern: all transactions in 3am-4am window only
        f["hour_deviation"] = random.uniform(10.0, 20.0)
        f["night_txn_ratio"] = random.uniform(0.98, 1.0)
        f["txn_count_last_1h"] = float(random.randint(20, 80))
        f["payee_vpa_age_days"] = random.uniform(0.0, 0.5)   # brand-new VPA
        f["counterparty_novelty"] = random.uniform(0.98, 1.0)

    return [float(f.get(name, 0.0) or 0.0) for name in ECOD_FEATURE_NAMES]


def main() -> None:
    try:
        import numpy as np
        import joblib
        from pyod.models.ecod import ECOD
    except ImportError as exc:
        log.error("ecod.import_error", error=str(exc),
                  hint="pip install pyod==2.0.2  (already in requirements.txt)")
        sys.exit(1)

    N_LEGIT = 50_000
    N_NOVEL = 500
    N_TOTAL = N_LEGIT + N_NOVEL

    log.info("ecod.generating_data",
             n_legit=N_LEGIT, n_novel=N_NOVEL, n_features=len(ECOD_FEATURE_NAMES))

    rows = []
    for _ in range(N_LEGIT):
        rows.append(_legit_vector())
    for _ in range(N_NOVEL):
        rows.append(_novel_anomaly_vector({}))

    X = np.array(rows, dtype=np.float64)

    # Replace any NaN / inf with 0.0 before fitting
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    log.info("ecod.fitting",
             shape=list(X.shape), contamination=0.01)

    # ECOD is fully unsupervised -- no labels used
    # contamination=0.01: we expect ~1% true outliers in the PASS stream
    ecod_model = ECOD(contamination=0.01)
    ecod_model.fit(X)

    # Anomaly scores: higher score = more anomalous (ECOD convention)
    train_scores = ecod_model.decision_scores_

    # Threshold at 97th percentile of training scores
    threshold = float(np.percentile(train_scores, 97))
    flagged_frac = float((train_scores > threshold).mean())

    log.info("ecod.threshold_computed",
             threshold=round(threshold, 4),
             flagged_fraction=round(flagged_frac, 4))

    # -- Save model ------------------------------------------------------------
    model_path = MODELS_DIR / "ecod_v1.joblib"
    joblib.dump(ecod_model, str(model_path))
    log.info("ecod.saved_model", path=str(model_path))

    # -- Save threshold --------------------------------------------------------
    thresh_path = MODELS_DIR / "ecod_threshold.json"
    with open(thresh_path, "w") as fh:
        json.dump({"ecod_threshold": threshold}, fh)
    log.info("ecod.saved_threshold", path=str(thresh_path))

    # -- Sanity: novel anomaly scores should be higher than legit scores --------
    legit_scores = train_scores[:N_LEGIT]
    novel_scores = train_scores[N_LEGIT:]

    print(f"\n{chr(8212)*60}")
    print("ECOD -- Training Summary")
    print(f"{chr(8212)*60}")
    print(f"Feature count (continuous): {len(ECOD_FEATURE_NAMES)}")
    print(f"Total samples:              {N_TOTAL} ({N_LEGIT} legit, {N_NOVEL} novel)")
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
    print("INVARIANT: ECOD scores NEVER enter fraud_score. novelty_queue only.")

    if novel_scores.mean() <= legit_scores.mean():
        log.warning("ecod.sanity_check_failed",
                    note="Novel anomaly scores not higher than legit -- check feature generation")


if __name__ == "__main__":
    main()
