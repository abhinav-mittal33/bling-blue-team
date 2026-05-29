"""
Offline script: Train stacking meta-learner on shadow committee data.

Gate: aborts if shadow_score_committee has fewer than settings.meta_learner_min_samples rows.

Architecture:
  - Input: 20-dim vector [5 scores, 5 confidences, 5 missing_flags, 5 context features]
  - Input-dropout training: randomly zero one scorer (p=0.15) per sample for robustness
  - Compares LogisticRegression(C=1.0) vs XGBClassifier(max_depth=3)
  - Picks higher PR-AUC on held-out validation (80/20 split)
  - Saves model to meta_learner_model_path and writes to meta_learner_versions table

Run:
  python ml/train_meta_learner.py
  python ml/train_meta_learner.py --min-samples 5000  # override gate for testing
"""
from __future__ import annotations

import argparse
import os
import sys
import random

import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings   # noqa: E402


def _build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert shadow rows to (X, y) for meta-learner training.
    Applies input-dropout: randomly zero one scorer's 3 cols with p=0.15.
    """
    from app.detection.tier3.meta_learner import (
        _SCORER_ORDER, _ACCOUNT_TYPE_ENCODING, _KYC_AGE_CAP, _DAILY_TXN_CAP
    )

    X_rows = []
    y_rows = []

    for row in rows:
        # Skip rows without ground-truth label (still in investigation)
        label = row.get("label")
        if label is None:
            continue

        # Scorer features: [score, confidence, missing_flag] × 5
        scorer_feats: list[float] = []
        for sid in _SCORER_ORDER:
            prefix = f"scorer_{sid.lower()}"
            score = float(row.get(f"{prefix}_score") or 0.5)
            conf = float(row.get(f"{prefix}_confidence") or 0.0)
            miss = float(bool(row.get(f"{prefix}_missing_flag", True)))
            scorer_feats.extend([score, conf, miss])

        # Input-dropout: zero one scorer's block with p=0.15
        if random.random() < 0.15:
            drop_idx = random.randint(0, len(_SCORER_ORDER) - 1)
            base = drop_idx * 3
            scorer_feats[base] = 0.5    # score → neutral
            scorer_feats[base + 1] = 0.0  # confidence → 0
            scorer_feats[base + 2] = 1.0  # missing_flag → True

        # Context features (stored in shadow row metadata or reconstructed)
        account_type = str(row.get("account_type", "SAVINGS") or "SAVINGS").upper()
        kyc_age = float(row.get("kyc_age") or 0.0)
        is_festival = float(bool(row.get("is_festival", False)))
        is_night = float(bool(row.get("is_night", False)))
        daily_txn_count = float(row.get("daily_txn_count", 0) or 0.0)

        context_feats = [
            _ACCOUNT_TYPE_ENCODING.get(account_type, 0.0),
            min(kyc_age / _KYC_AGE_CAP, 1.0),
            is_festival,
            is_night,
            min(daily_txn_count / _DAILY_TXN_CAP, 1.0),
        ]

        X_rows.append(scorer_feats + context_feats)
        y_rows.append(int(label))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def _pr_auc(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    proba = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, proba))


def train(min_samples: int) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(settings.postgres_url)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # Gate: require minimum shadow rows with labels
        count_result = db.execute(
            text("SELECT COUNT(*) FROM shadow_score_committee WHERE live_action IS NOT NULL")
        ).scalar()

        if (count_result or 0) < min_samples:
            print(f"ABORT: only {count_result} labeled shadow rows (need {min_samples}).")
            print("Accumulate more shadow data before running meta-learner training.")
            sys.exit(1)

        print(f"Fetching {count_result} shadow rows...")
        from app.detection.tier3.shadow_writer import get_shadow_training_batch
        rows = get_shadow_training_batch(db, limit=min_samples * 2)
    finally:
        db.close()

    X, y = _build_feature_matrix(rows)
    fraud_count = y.sum()
    print(f"Training matrix: {X.shape}, fraud={fraud_count}, benign={len(y)-fraud_count}")

    if fraud_count < 50:
        print("ABORT: too few fraud labels for reliable meta-learner training (need ≥50).")
        sys.exit(1)

    # 80/20 train/val split (stratified via manual index split)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print("Training LogisticRegression...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_prauc = _pr_auc(lr, X_val, y_val)
    print(f"  LR PR-AUC: {lr_prauc:.4f}")

    print("Training XGBClassifier (depth=3)...")
    try:
        from xgboost import XGBClassifier
        scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
        xgb = XGBClassifier(
            max_depth=3,
            n_estimators=100,
            learning_rate=0.05,
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_prauc = _pr_auc(xgb, X_val, y_val)
        print(f"  XGB PR-AUC: {xgb_prauc:.4f}")
    except Exception as exc:
        print(f"  XGBClassifier failed: {exc} — using LR only")
        xgb_prauc = -1.0

    # Pick winner
    if xgb_prauc >= lr_prauc:
        winner = xgb
        algo = "xgb_depth3"
        best_prauc = xgb_prauc
        print(f"Winner: XGBClassifier (PR-AUC={best_prauc:.4f})")
    else:
        winner = lr
        algo = "logistic_regression"
        best_prauc = lr_prauc
        print(f"Winner: LogisticRegression (PR-AUC={best_prauc:.4f})")

    if best_prauc < 0.50:
        print(f"WARNING: PR-AUC {best_prauc:.4f} < 0.50. Check shadow data quality.")

    # Save model
    import joblib
    model_path = os.path.abspath(settings.meta_learner_model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(winner, model_path)
    print(f"Model saved: {model_path}")

    # Conformal calibration on validation set
    from app.detection.tier3 import conformal_calibrator
    val_scores = winner.predict_proba(X_val)[:, 1]
    conformal_calibrator.calibrate(val_scores, y_val)

    # Write to meta_learner_versions table
    version = _next_version()
    engine2 = create_engine(settings.postgres_url)
    Session2 = sessionmaker(bind=engine2)
    db2 = Session2()
    try:
        db2.execute(
            text("""
                UPDATE meta_learner_versions SET is_active = false WHERE is_active = true
            """)
        )
        db2.execute(
            text("""
                INSERT INTO meta_learner_versions
                  (version, algorithm, training_sample_count, pr_auc_validation, model_path, is_active)
                VALUES
                  (:version, :algorithm, :sample_count, :pr_auc, :model_path, true)
            """),
            {
                "version": version,
                "algorithm": algo,
                "sample_count": len(X_train),
                "pr_auc": round(best_prauc, 6),
                "model_path": model_path,
            },
        )
        db2.commit()
        print(f"Registered as version {version} in meta_learner_versions.")
    except Exception as exc:
        print(f"WARNING: could not write to meta_learner_versions: {exc}")
        db2.rollback()
    finally:
        db2.close()

    print("\nDone. Reload the API to activate the new meta-learner:")
    print("  docker-compose restart api")


def _next_version() -> str:
    import time
    return f"meta_{int(time.time())}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stacking meta-learner on shadow data")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=settings.meta_learner_min_samples,
        help="Minimum labeled shadow rows required",
    )
    args = parser.parse_args()
    train(args.min_samples)
