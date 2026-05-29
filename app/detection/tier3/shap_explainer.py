"""
Async SHAP computation task (P1-6).
SHAP is moved out of the /score hot path — computed asynchronously after scoring.
Results written back to fraud_scores.shap_values and optionally to shap_access_log.

INVARIANT: Always use uncalibrated base estimator. CalibratedClassifierCV breaks TreeExplainer.
  - Shadow mode / pre-live: ensemble.get_base_model() (legacy XGBoost)
  - Live mode (committee_live_mode=True): scorer_a.get_base_model_for_shap() (Scorer A base)
  Never pass CalibratedClassifierCV to TreeExplainer.
"""
import json
import structlog

from app.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(
    name="evidence.compute_shap",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
    soft_time_limit=60,
    time_limit=90,
    queue="evidence",
)
def compute_shap(self, fraud_score_id: int, feature_vector: dict) -> list[dict]:
    """
    Compute SHAP values for a scored transaction and persist to fraud_scores.shap_values.
    Called immediately after /score returns — does NOT block the response.
    """
    try:
        shap_values = _run_shap(feature_vector)
        _persist_shap(fraud_score_id, shap_values)
        logger.info("shap_computed", fraud_score_id=fraud_score_id, top_feature=shap_values[0]["feature"] if shap_values else None)
        return shap_values
    except Exception as exc:
        logger.error("shap_computation_failed", fraud_score_id=fraud_score_id, error=str(exc))
        try:
            raise self.retry(exc=exc)
        except Exception:
            return []


def _run_shap(feature_vector: dict) -> list[dict]:
    """
    Run SHAP on the base (uncalibrated) model. Returns top-10 by |contribution|.
    Phase 4+: model is XGBClassifier → use .get_booster().predict(pred_contribs=True).
    Phase 3-: model is xgb.Booster → use .predict(pred_contribs=True) directly.
    NEVER call this on the CalibratedClassifierCV wrapper.
    """
    import numpy as np
    import xgboost as xgb
    from app.core.config import settings

    # SHAP invariant: always use uncalibrated base estimator
    if settings.committee_live_mode:
        from app.detection.tier3.scorer_a import get_base_model_for_shap
        model = get_base_model_for_shap()
    else:
        from app.detection.tier3.ensemble import get_base_model
        model = get_base_model()

    if model is None:
        return []

    feature_names = sorted(feature_vector.keys())
    feature_values = np.array(
        [feature_vector.get(k, float("nan")) for k in feature_names],
        dtype=np.float32,
    ).reshape(1, -1)

    # Extract the underlying Booster regardless of wrapper type
    if isinstance(model, xgb.Booster):
        booster = model
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
    else:
        logger.warning("shap_unknown_model_type", model_type=type(model).__name__)
        return []

    dmatrix = xgb.DMatrix(feature_values, feature_names=feature_names, missing=float("nan"))
    raw = booster.predict(dmatrix, pred_contribs=True)[0]

    contribs = [
        {"feature": k, "contribution": float(raw[i])}
        for i, k in enumerate(feature_names)
    ]
    return sorted(contribs, key=lambda x: abs(x["contribution"]), reverse=True)[:10]


def _persist_shap(fraud_score_id: int, shap_values: list[dict]) -> None:
    """Write SHAP values back to fraud_scores row. Best-effort — does not raise."""
    try:
        from app.utils.postgres_client import SessionLocal
        import sqlalchemy as sa

        db = SessionLocal()
        try:
            db.execute(
                sa.text(
                    "UPDATE fraud_scores SET shap_values = :shap WHERE id = :id"
                ),
                {"shap": json.dumps(shap_values), "id": fraud_score_id},
            )
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.error("shap_persist_failed", fraud_score_id=fraud_score_id, error=str(exc))
