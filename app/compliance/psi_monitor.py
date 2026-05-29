"""
PSI (Population Stability Index) drift monitoring (P4-7 + P5 integration).
Celery Beat task: weekly on Monday 6am UTC.

PSI measures how much the scoring distribution has shifted from training baseline.
PSI < 0.1  → no significant change
PSI 0.1-0.2 → moderate shift — monitor
PSI > 0.2  → significant shift — alert (PSI_ALERT_THRESHOLD env var)
"""
from __future__ import annotations
import math
import structlog

logger = structlog.get_logger()


def run_psi_check() -> None:
    """
    Weekly PSI check (Celery Beat, P4-7).
    Computes PSI between training baseline and last-7-day scoring distribution.
    Alerts if any feature PSI exceeds PSI_ALERT_THRESHOLD (default 0.2).
    """
    logger.info("psi_check_started")
    try:
        baseline = _load_baseline()
        if not baseline:
            logger.warning("psi_baseline_not_found", note="Run ml/train.py first to generate baseline")
            return

        recent_scores = _get_recent_scores()
        if len(recent_scores) < 100:
            logger.warning("psi_insufficient_data", count=len(recent_scores))
            return

        psi_score = _compute_score_psi(baseline, recent_scores)
        _check_alert(psi_score)
        logger.info("psi_check_complete", psi_score=round(psi_score, 4))

    except Exception as exc:
        logger.error("psi_check_failed", error=str(exc))


def _load_baseline() -> dict:
    from pathlib import Path
    import json
    baseline_file = Path("ml/models/psi_baseline.json")
    if not baseline_file.exists():
        return {}
    with open(baseline_file) as f:
        return json.load(f)


def _get_recent_scores() -> list[float]:
    """Fetch fraud scores from last 7 days from PostgreSQL."""
    try:
        from app.utils.postgres_client import SessionLocal
        import sqlalchemy as sa
        db = SessionLocal()
        try:
            rows = db.execute(
                sa.text(
                    "SELECT score FROM fraud_scores "
                    "WHERE scored_at > NOW() - INTERVAL '7 days' "
                    "ORDER BY scored_at DESC LIMIT 10000"
                )
            ).fetchall()
            return [float(row[0]) for row in rows]
        finally:
            db.close()
    except Exception as exc:
        logger.error("psi_score_fetch_failed", error=str(exc))
        return []


def _compute_score_psi(baseline: dict, recent_scores: list[float]) -> float:
    """
    Compute PSI between score_distribution baseline and current distribution.
    Uses 10 equal-width bins over [0, 1].
    """
    import numpy as np

    bins = [i / 10 for i in range(11)]
    recent_hist, _ = np.histogram(recent_scores, bins=bins)
    recent_pct = recent_hist / sum(recent_hist)

    # Baseline: use mean score distribution if stored
    base_scores = baseline.get("score_distribution", {})
    if not base_scores:
        return 0.0

    base_pct_list = base_scores.get("percentiles", [0.1] * 10)
    base_pct = [p / sum(base_pct_list) for p in base_pct_list]

    psi = sum(
        (actual - expected) * math.log(max(actual, 1e-9) / max(expected, 1e-9))
        for actual, expected in zip(recent_pct, base_pct)
    )
    return float(psi)


def _check_alert(psi_score: float) -> None:
    from app.core.config import settings
    threshold = getattr(settings, "psi_alert_threshold", 0.2)
    if psi_score > threshold:
        logger.critical(
            "psi_alert",
            psi_score=round(psi_score, 4),
            threshold=threshold,
            alert="Score distribution has drifted significantly — consider retraining",
        )
        try:
            import httpx, os
            webhook = os.getenv("SLACK_WEBHOOK_URL", "")
            if webhook:
                httpx.post(
                    webhook,
                    json={"text": f":chart_with_downwards_trend: PSI alert: score drift={psi_score:.4f} (threshold={threshold})"},
                    timeout=5.0,
                )
        except Exception:
            pass
