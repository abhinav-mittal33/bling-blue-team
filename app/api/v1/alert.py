"""
GET /api/v1/alerts/{alert_id}
Returns full evidence package for a specific alert.
Auth: INVESTIGATOR_API_KEY or valid RS256 JWT with role=investigator (P1-7).
SHAP values are role-gated — access logged to shap_access_log (P1-6).
"""
import structlog
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.security import require_investigator_key, pseudonymize
from app.evidence.evidence_packager import build_evidence_package

logger = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_investigator_key)])


@router.get("/api/v1/alerts/{alert_id}")
async def get_alert(
    request: Request,
    alert_id: str,
    db: Session = Depends(get_db),
):
    package = build_evidence_package(alert_id, db)
    if package is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Log SHAP access when SHAP values are present (P1-6 role-gate audit)
    if package and package.get("shap_values"):
        caller = request.headers.get("x-api-key", "") or request.headers.get("authorization", "")
        _log_shap_access(alert_id, pseudonymize(caller[:32]), db)

    # Attach committee breakdown from shadow table (display-only — never used for gating)
    package["committee_breakdown"] = _fetch_committee_breakdown(alert_id, db)

    logger.info("evidence_package_served", alert_id=alert_id)
    return {"data": package, "error": None}


def _fetch_committee_breakdown(alert_id: str, db: Session) -> dict | None:
    """
    Fetch most recent committee shadow row for this alert's transaction.
    Returns None if no shadow data exists yet (normal for first days of Phase 1).
    Best-effort — never raises.
    """
    try:
        row = db.execute(
            sa.text("""
                SELECT
                    ssc.scorer_a_score, ssc.scorer_a_confidence, ssc.scorer_a_missing_flag,
                    ssc.scorer_b_score, ssc.scorer_b_confidence, ssc.scorer_b_missing_flag,
                    ssc.scorer_c_score, ssc.scorer_c_confidence, ssc.scorer_c_missing_flag,
                    ssc.scorer_d_score, ssc.scorer_d_confidence, ssc.scorer_d_missing_flag,
                    ssc.scorer_f_score, ssc.scorer_f_confidence, ssc.scorer_f_missing_flag,
                    ssc.meta_score, ssc.specialist_override,
                    ssc.final_committee_score, ssc.live_score,
                    ssc.mapie_lower, ssc.mapie_upper, ssc.scored_at
                FROM shadow_score_committee ssc
                JOIN alerts a ON a.transaction_id = ssc.transaction_id
                WHERE a.id = :alert_id
                ORDER BY ssc.scored_at DESC
                LIMIT 1
            """),
            {"alert_id": alert_id},
        ).fetchone()

        if row is None:
            return None

        def _scorer_dict(score, conf, missing):
            return {
                "score": round(float(score), 4) if score is not None else None,
                "confidence": round(float(conf), 4) if conf is not None else None,
                "missing": bool(missing) if missing is not None else True,
            }

        return {
            "scorers": {
                "A": _scorer_dict(row.scorer_a_score, row.scorer_a_confidence, row.scorer_a_missing_flag),
                "B": _scorer_dict(row.scorer_b_score, row.scorer_b_confidence, row.scorer_b_missing_flag),
                "C": _scorer_dict(row.scorer_c_score, row.scorer_c_confidence, row.scorer_c_missing_flag),
                "D": _scorer_dict(row.scorer_d_score, row.scorer_d_confidence, row.scorer_d_missing_flag),
                "F": _scorer_dict(row.scorer_f_score, row.scorer_f_confidence, row.scorer_f_missing_flag),
            },
            "meta_score": round(float(row.meta_score), 4) if row.meta_score is not None else None,
            "specialist_override": bool(row.specialist_override),
            "final_committee_score": round(float(row.final_committee_score), 4) if row.final_committee_score is not None else None,
            "live_score": round(float(row.live_score), 4) if row.live_score is not None else None,
            "mapie_lower": round(float(row.mapie_lower), 4) if row.mapie_lower is not None else None,
            "mapie_upper": round(float(row.mapie_upper), 4) if row.mapie_upper is not None else None,
            "scored_at": str(row.scored_at),
            "note": "shadow_mode — for display only, not used for scoring",
        }
    except Exception as exc:
        logger.warning("committee_breakdown_fetch_failed", alert_id=alert_id, error=str(exc))
        return None


def _log_shap_access(alert_id: str, investigator_hash: str, db: Session) -> None:
    """INSERT to shap_access_log (INSERT-only table). Best-effort — never raises."""
    try:
        db.execute(
            sa.text(
                "INSERT INTO shap_access_log (alert_id, investigator_id_hash, action) "
                "VALUES (:alert_id, :inv_hash, 'GET_ALERT_SHAP')"
            ),
            {"alert_id": alert_id, "inv_hash": investigator_hash},
        )
        db.commit()
    except Exception as exc:
        logger.error("shap_access_log_failed", alert_id=alert_id, error=str(exc))
