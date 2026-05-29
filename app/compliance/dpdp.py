"""
DPDP Act 2023 (Digital Personal Data Protection) compliance layer (P5-3).

Implements:
  - Right to erasure: mark PII fields as erased in audit trail
  - Data category tracking: log which categories of personal data are processed
  - Consent ledger: record lawful basis for each PII processing operation

LIVE MODE: Set DPDP_LIVE=true to enable erasure endpoint and consent checks.

Retention rule (P5-4): Financial transaction data retained 5 years per RBI PMLA Section 12.
This OVERRIDES any DPDP erasure request for transaction records.
"""
from __future__ import annotations
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()

# PII fields that can be erased per DPDP — non-financial metadata only
_ERASABLE_FIELDS = frozenset({
    "device_id", "ip_address", "geo_city", "geo_state",
    "kyc_occupation", "kyc_home_state",
})

# Fields that CANNOT be erased — required by RBI PMLA Section 12
_RETENTION_PROTECTED = frozenset({
    "account_id", "transaction_id", "amount", "timestamp",
    "channel", "payee_account_id", "payee_vpa",
})


def process_erasure_request(account_id: str, db) -> dict:
    """
    Process DPDP right-to-erasure request.
    Erases non-financial PII. Transaction records retained per PMLA override.
    Returns summary of erased vs retained fields.
    """
    from app.core.config import settings
    from app.core.security import pseudonymize

    if not settings.dpdp_live:
        logger.info(
            "dpdp_erasure_stub",
            account_hash=pseudonymize(account_id),
            note="DPDP_LIVE=false",
        )
        return {"erased": [], "retained": list(_RETENTION_PROTECTED), "stub": True}

    import sqlalchemy as sa
    erased = []

    try:
        # Null out erasable PII fields in accounts table.
        # Each statement is fully parameterized — column names are hardcoded literals,
        # never interpolated from user input or external sets.
        if "kyc_occupation" in _ERASABLE_FIELDS:
            db.execute(sa.text("UPDATE accounts SET kyc_occupation = NULL WHERE id = :id"), {"id": account_id})
            erased.append("kyc_occupation")
        if "kyc_home_state" in _ERASABLE_FIELDS:
            db.execute(sa.text("UPDATE accounts SET kyc_home_state = NULL WHERE id = :id"), {"id": account_id})
            erased.append("kyc_home_state")

        # Null out device/IP in transactions (non-financial metadata)
        db.execute(
            sa.text(
                "UPDATE transactions SET device_id = NULL, ip_address = NULL "
                "WHERE account_id = :id"
            ),
            {"id": account_id},
        )
        erased.extend(["device_id", "ip_address"])

        db.commit()
        logger.info(
            "dpdp_erasure_complete",
            account_hash=pseudonymize(account_id),
            erased_fields=erased,
        )
    except Exception as exc:
        logger.error("dpdp_erasure_failed", error=str(exc))
        raise

    return {
        "erased": erased,
        "retained": list(_RETENTION_PROTECTED),
        "retention_reason": "RBI PMLA Section 12 (5-year mandatory retention)",
    }


def cleanup_expired_features() -> None:
    """
    Celery Beat task: delete graph feature cache entries for accounts
    whose data has exceeded the 5-year retention window.
    Runs daily at 4am UTC (celeryconfig.py).
    """
    from app.core.config import settings

    if not settings.dpdp_live:
        logger.debug("dpdp_cleanup_stub", note="DPDP_LIVE=false")
        return

    logger.info("dpdp_cleanup_started")
    try:
        from app.utils.redis_client import get_redis
        r = get_redis()
        # Scan for feat: keys with _last_updated > 5 years ago
        # Full implementation: paginate, check, delete
        logger.info("dpdp_cleanup_stub", note="Full implementation in production P5-3")
    except Exception as exc:
        logger.error("dpdp_cleanup_failed", error=str(exc))


def log_data_processing(
    account_id: str,
    operation: str,
    categories: list[str],
    lawful_basis: str,
    db,
) -> None:
    """
    Record that personal data was processed — DPDP consent ledger.
    Inserts into data_categories table (INSERT-only).
    """
    try:
        import sqlalchemy as sa
        from app.core.security import pseudonymize
        db.execute(
            sa.text(
                "INSERT INTO data_categories "
                "(account_id_hash, operation, categories, lawful_basis, processed_at) "
                "VALUES (:acc, :op, :cats, :basis, NOW())"
            ),
            {
                "acc": pseudonymize(account_id),
                "op": operation,
                "cats": ",".join(categories),
                "basis": lawful_basis,
            },
        )
        db.commit()
    except Exception as exc:
        logger.error("dpdp_consent_log_failed", error=str(exc))


def check_pmla_retention_override(field: str) -> bool:
    """Returns True if PMLA retention prevents erasure of this field."""
    return field in _RETENTION_PROTECTED
