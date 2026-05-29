"""
ml/train_scorer_a.py -- Train Scorer A (Upgraded GBM) for the Committee Engine.

Scorer A is the XGBClassifier trained on FEATURE_NAMES_V5 (113 features = V4 features +
8 UPI session features). The 8 UPI session features may all be NaN at training time --
the model learns to route through XGBoost's native missing-value path for them.

Outputs:
  ml/models/scorer_a_base.joblib     -- uncalibrated base estimator (for SHAP only)
  ml/models/scorer_a_v1.joblib       -- Platt-calibrated model (for scoring)

INVARIANT: SHAP must use scorer_a_base.joblib, never scorer_a_v1.joblib.
           CalibratedClassifierCV wrapper breaks TreeExplainer.

scale_pos_weight is computed from the actual training distribution.
eval_metric='aucpr' -- PR-AUC is correct for imbalanced fraud data.

Run: python ml/train_scorer_a.py
"""
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

# Import from registry -- NEVER hardcode feature lists
from ml.feature_registry import FEATURE_NAMES_V5, UPI_SESSION_FEATURES  # noqa: E402

# Import archetype generators from train.py -- single source of truth for synthetic data
from ml.train import (  # noqa: E402
    _legit_features,
    _fraud_features,
    FRAUD_ARCHETYPES,
)

random.seed(42)

# -- Dataset shape -------------------------------------------------------------
# Heavier fraud ratio than production: committee robustness benefits from seeing
# more fraud examples per archetype than 2.7% production prevalence would give.
N_LEGIT = 150_000
N_FRAUD = 50_000

_UPI_SESSION_SET = set(UPI_SESSION_FEATURES)


def _to_vector_v5(features: dict) -> list:
    """
    Assemble a 113-dim feature vector in FEATURE_NAMES_V5 order.

    UPI session features are set to NaN to simulate the expected state
    before UPI enrichment is wired into the transaction schema.
    XGBoost handles NaN natively via its missing-value split path.
    Any other key absent from features dict also becomes NaN.
    """
    vec = []
    for name in FEATURE_NAMES_V5:
        if name in _UPI_SESSION_SET:
            vec.append(float("nan"))
        else:
            val = features.get(name)
            vec.append(float("nan") if val is None else float(val))
    return vec


def _generate_dataset():
    """Build X (list of vectors) and y (list of labels)."""
    rows_X, rows_y = [], []

    # Fraud: distribute evenly across all archetypes
    per_arch = N_FRAUD // len(FRAUD_ARCHETYPES)
    remainder = N_FRAUD - per_arch * len(FRAUD_ARCHETYPES)

    for i, arch in enumerate(FRAUD_ARCHETYPES):
        count = per_arch + (1 if i < remainder else 0)
        for _ in range(count):
            rows_X.append(_to_vector_v5(_fraud_features(arch)))
            rows_y.append(1)

    # Legitimate
    for _ in range(N_LEGIT):
        rows_X.append(_to_vector_v5(_legit_features()))
        rows_y.append(0)

    return rows_X, rows_y


def main() -> None:
    try:
        import numpy as np
        import joblib
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        log.error("scorer_a.import_error", error=str(exc))
        sys.exit(1)

    log.info("scorer_a.generating_dataset",
             n_legit=N_LEGIT, n_fraud=N_FRAUD, n_features=len(FEATURE_NAMES_V5))

    rows_X, rows_y = _generate_dataset()

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)

    # Verify UPI columns are NaN (sanity check)
    upi_start = len(FEATURE_NAMES_V5) - len(UPI_SESSION_FEATURES)
    upi_nan_frac = float(np.isnan(X[:, upi_start:]).mean())
    log.info("scorer_a.dataset_ready",
             shape=list(X.shape),
             fraud_count=int(y.sum()),
             legit_count=int((y == 0).sum()),
             upi_cols_nan_frac=round(upi_nan_frac, 4))

    # 80/20 train / val split, stratified to preserve fraud ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # scale_pos_weight from actual distribution -- printed for CLAUDE.md update
    train_fraud = int(y_train.sum())
    train_legit = int((y_train == 0).sum())
    pos_weight = max(1, train_legit // train_fraud)

    log.info("scorer_a.scale_pos_weight",
             value=pos_weight, train_fraud=train_fraud, train_legit=train_legit)
    print(f"\nscale_pos_weight (Scorer A): {pos_weight}  "
          f"({train_legit} legit / {train_fraud} fraud)")
    print("ACTION: Update CLAUDE.md scale_pos_weight table if this value changes.")

    # -- Base XGBClassifier ----------------------------------------------------
    base_xgb = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",        # PR-AUC -- NOT auc; see gotchas.md
        scale_pos_weight=pos_weight,
        max_depth=6,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        enable_categorical=False,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=25,
    )

    log.info("scorer_a.training_base_model")
    base_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # -- Platt scaling on val set ----------------------------------------------
    # cv="prefit": base_xgb already fitted; only sigmoid calibration layer trains.
    log.info("scorer_a.calibrating_platt")
    calibrated = CalibratedClassifierCV(base_xgb, cv="prefit", method="sigmoid")
    calibrated.fit(X_val, y_val)

    # -- Save models -----------------------------------------------------------
    base_path = MODELS_DIR / "scorer_a_base.joblib"
    cal_path = MODELS_DIR / "scorer_a_v1.joblib"

    joblib.dump(base_xgb, str(base_path))
    log.info("scorer_a.saved_base", path=str(base_path))

    joblib.dump(calibrated, str(cal_path))
    log.info("scorer_a.saved_calibrated", path=str(cal_path))

    # -- Evaluation on val set (calibrated model) ------------------------------
    cal_proba = calibrated.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, cal_proba)

    fraud_mask = y_val == 1
    legit_mask = y_val == 0

    print(f"\n{chr(8212)*60}")
    print("Scorer A -- Training Summary")
    print(f"{chr(8212)*60}")
    print(f"Feature count:              {len(FEATURE_NAMES_V5)}")
    print(f"UPI session features:       {len(UPI_SESSION_FEATURES)} (all NaN this run)")
    print(f"UPI NaN fraction in val:    {upi_nan_frac:.3f} (expected: 1.000)")
    print(f"Train samples:              {len(X_train)} ({train_fraud} fraud)")
    print(f"Val samples:                {len(X_val)}")
    print(f"scale_pos_weight:           {pos_weight}")
    print(f"PR-AUC (val, calibrated):   {pr_auc:.4f}")
    print(f"Fraud mean score (cal):     {cal_proba[fraud_mask].mean():.3f}")
    print(f"Legit mean score (cal):     {cal_proba[legit_mask].mean():.3f}")
    print(f"Base model:                 {base_path}")
    print(f"Calibrated model:           {cal_path}")
    print(f"{chr(8212)*60}")
    print("SHAP: load scorer_a_base.joblib -- NEVER scorer_a_v1.joblib")
    print("      CalibratedClassifierCV wrapper breaks TreeExplainer.")


if __name__ == "__main__":
    main()
