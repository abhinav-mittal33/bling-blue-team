"""
NPCI Pre-Settlement Notification stub (P5-2).
Notifies NPCI of suspicious high-value transactions before settlement.

LIVE MODE: Set NPCI_LIVE=true only with NPCI approval + integration testing.
STUB MODE: Logs the hold request, never touches NPCI systems.

Threshold: transactions ≥ ₹10L (1,000,000) with HIGH_RISK action.
"""
from __future__ import annotations
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()

_NPCI_HOLD_THRESHOLD = 1_000_000.0  # ₹10L


def request_pre_settlement_hold(
    transaction_id: str,
    account_id: str,
    amount: float,
    fraud_score: float,
    alert_id: str,
) -> dict:
    """
    Request NPCI to hold pre-settlement for suspicious high-value UPI transaction.
    Returns hold status dict.
    """
    from app.core.config import settings
    from app.core.security import pseudonymize

    if amount < _NPCI_HOLD_THRESHOLD:
        return {"hold_requested": False, "reason": "below_threshold"}

    if not settings.npci_live:
        logger.warning(
            "npci_hold_stub",
            transaction_id=transaction_id,
            account_hash=pseudonymize(account_id),
            amount=amount,
            fraud_score=fraud_score,
            note="NPCI_LIVE=false — not requesting hold",
        )
        return {
            "hold_requested": False,
            "stub": True,
            "transaction_id": transaction_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        import httpx
        response = httpx.post(
            "https://upi.npci.org.in/api/v1/hold",
            json={
                "transaction_id": transaction_id,
                "hold_reason": "SUSPICIOUS_FRAUD_SCORE",
                "fraud_score": round(fraud_score, 4),
                "alert_id": alert_id,
            },
            headers={"X-NPCI-API-Key": getattr(settings, "npci_api_key", "")},
            timeout=10.0,
        )
        response.raise_for_status()
        logger.warning(
            "npci_hold_requested",
            transaction_id=transaction_id,
            amount=amount,
        )
        return {"hold_requested": True, "response": response.json()}
    except Exception as exc:
        logger.error("npci_hold_failed", transaction_id=transaction_id, error=str(exc))
        return {"hold_requested": False, "error": str(exc)}
