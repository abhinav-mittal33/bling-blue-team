"""
ml/train_scorer_d.py — Train Scorer D in limited mode (Random Forest on 7 set features).

Scorer D operates in two modes:
  - Limited mode (MAMBA_LIMITED_MODE=true, default): RF on 7 behavioral set features
    from 90-day transaction history. Trained here.
  - Mamba mode (MAMBA_LIMITED_MODE=false): S4 state-space sequence model.
    See ml/train_mamba.py — requires sequence data + causal-conv1d install.

Features (must match _SET_FEATURE_ORDER in scorer_d.py exactly):
  count_of_night_txns                  ratio of night transactions (22h-6h)
  count_of_new_vpa_txns                ratio of transactions to new (<7d) VPAs
  count_of_high_amount_txns            ratio of transactions > ₹1L
  count_of_channel_switches            distinct channels (normalized 0-1)
  has_any_micro_test_payment           binary: any amount < ₹1
  has_any_round_amount_burst           binary: any round-number large transaction
  distinct_fraud_proximate_action_types  normalized count of risky action types

Output:
  ml/models/scorer_d_v1.joblib         RandomForestClassifier (Platt-calibrated)

Run: python ml/train_scorer_d.py
     XGBOD_FULL_TRAIN=1 python ml/train_scorer_d.py   (larger synthetic dataset)
"""
import json
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

# Must match _SET_FEATURE_ORDER in app/detection/tier3/scorer_d.py exactly
FEATURE_ORDER = [
    "count_of_night_txns",
    "count_of_new_vpa_txns",
    "count_of_high_amount_txns",
    "count_of_channel_switches",
    "has_any_micro_test_payment",
    "has_any_round_amount_burst",
    "distinct_fraud_proximate_action_types",
]


def _legit_account() -> dict:
    """Typical legitimate account: mostly daytime, known VPAs, low amounts, single channel."""
    return {
        "count_of_night_txns": random.uniform(0.0, 0.1),
        "count_of_new_vpa_txns": random.uniform(0.0, 0.15),
        "count_of_high_amount_txns": random.uniform(0.0, 0.1),
        "count_of_channel_switches": random.uniform(0.0, 0.3),
        "has_any_micro_test_payment": float(random.random() < 0.02),
        "has_any_round_amount_burst": float(random.random() < 0.05),
        "distinct_fraud_proximate_action_types": random.uniform(0.0, 0.15),
    }


def _fraud_account(fraud_type: str) -> dict:
    """Fraud patterns — each archetype has a distinct set-feature signature."""
    if fraud_type == "night_mule":
        return {
            "count_of_night_txns": random.uniform(0.6, 1.0),
            "count_of_new_vpa_txns": random.uniform(0.3, 0.8),
            "count_of_high_amount_txns": random.uniform(0.1, 0.5),
            "count_of_channel_switches": random.uniform(0.0, 0.25),
            "has_any_micro_test_payment": float(random.random() < 0.4),
            "has_any_round_amount_burst": float(random.random() < 0.3),
            "distinct_fraud_proximate_action_types": random.uniform(0.3, 1.0),
        }
    elif fraud_type == "channel_hopper":
        return {
            "count_of_night_txns": random.uniform(0.1, 0.4),
            "count_of_new_vpa_txns": random.uniform(0.4, 0.9),
            "count_of_high_amount_txns": random.uniform(0.2, 0.6),
            "count_of_channel_switches": random.uniform(0.6, 1.0),
            "has_any_micro_test_payment": float(random.random() < 0.6),
            "has_any_round_amount_burst": float(random.random() < 0.5),
            "distinct_fraud_proximate_action_types": random.uniform(0.5, 1.0),
        }
    elif fraud_type == "new_vpa_spike":
        return {
            "count_of_night_txns": random.uniform(0.05, 0.3),
            "count_of_new_vpa_txns": random.uniform(0.7, 1.0),
            "count_of_high_amount_txns": random.uniform(0.3, 0.8),
            "count_of_channel_switches": random.uniform(0.0, 0.2),
            "has_any_micro_test_payment": float(random.random() < 0.7),
            "has_any_round_amount_burst": float(random.random() < 0.4),
            "distinct_fraud_proximate_action_types": random.uniform(0.4, 1.0),
        }
    else:  # generic layering
        return {
            "count_of_night_txns": random.uniform(0.3, 0.7),
            "count_of_new_vpa_txns": random.uniform(0.2, 0.7),
            "count_of_high_amount_txns": random.uniform(0.2, 0.7),
            "count_of_channel_switches": random.uniform(0.1, 0.5),
            "has_any_micro_test_payment": float(random.random() < 0.5),
            "has_any_round_amount_burst": float(random.random() < 0.5),
            "distinct_fraud_proximate_action_types": random.uniform(0.3, 0.8),
        }


def _to_vec(d: dict) -> list:
    return [d[f] for f in FEATURE_ORDER]


def main() -> None:
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score

    _full = os.getenv("XGBOD_FULL_TRAIN") == "1"
    N_LEGIT = 50_000 if _full else 10_000
    N_FRAUD_PER_TYPE = 1_000 if _full else 250

    fraud_types = ["night_mule", "channel_hopper", "new_vpa_spike", "generic_layering"]

    log.info("scorer_d.generating_data",
             n_legit=N_LEGIT, n_fraud=N_FRAUD_PER_TYPE * len(fraud_types))

    rows_X, rows_y = [], []

    for _ in range(N_LEGIT):
        rows_X.append(_to_vec(_legit_account()))
        rows_y.append(0)

    for ftype in fraud_types:
        for _ in range(N_FRAUD_PER_TYPE):
            rows_X.append(_to_vec(_fraud_account(ftype)))
            rows_y.append(1)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    n_fraud = int(y_train.sum())
    n_legit = len(y_train) - n_fraud
    class_weight = {0: 1, 1: n_legit // max(n_fraud, 1)}

    log.info("scorer_d.training_rf",
             n_train=len(X_train), n_val=len(X_val), class_weight=class_weight)

    base_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    base_rf.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(base_rf, cv="prefit", method="sigmoid")
    calibrated.fit(X_val, y_val)

    val_probs = calibrated.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_probs)

    fraud_mean = float(val_probs[y_val == 1].mean())
    legit_mean = float(val_probs[y_val == 0].mean())

    model_path = MODELS_DIR / "scorer_d_v1.joblib"
    joblib.dump(calibrated, str(model_path))
    log.info("scorer_d.saved", path=str(model_path))

    sep = chr(8212) * 60
    print(f"\n{sep}")
    print("Scorer D (Limited Mode — RF) -- Training Summary")
    print(sep)
    print(f"Features:        {len(FEATURE_ORDER)}")
    print(f"Train samples:   {len(X_train)} ({n_fraud} fraud)")
    print(f"Val samples:     {len(X_val)}")
    print(f"PR-AUC (val):    {pr_auc:.4f}")
    print(f"Fraud mean score:{fraud_mean:.3f}")
    print(f"Legit mean score:{legit_mean:.3f}")
    print(f"Model saved:     {model_path}")
    print(sep)
    print("Set MAMBA_LIMITED_MODE=false + provide scorer_d_mamba_v1.pt for full Mamba.")
    print("See ml/train_mamba.py for Mamba training instructions.")


if __name__ == "__main__":
    main()
