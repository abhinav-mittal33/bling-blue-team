"""
app/utils/evidence_seal.py — SHA-256 evidence hash registry (P1-11).

Replaces blockchain_client.py with a local tamper-evident SHA-256 hash record.
The evidence bundle JSON is hashed and stored in the evidence_seal table (INSERT-only).
On-chain sealing via blockchain teammate's API can be added later — routing through here
means the rest of the codebase doesn't need to change when that integration is ready.

The evidence_seal table has an INSERT-only DB trigger (migration 004_evidence_seal.py)
so records cannot be tampered with post-creation.
"""
from __future__ import annotations

import hashlib
import json
import structlog
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.orm import Session

log = structlog.get_logger()


def seal_evidence_bundle(
    db: Session,
    alert_id: str,
    evidence: dict,
    investigator_id: str | None = None,
) -> str:
    """
    Hash the evidence bundle and persist the seal record.

    Args:
        db: SQLAlchemy session (caller commits)
        alert_id: Alert UUID the evidence belongs to
        evidence: Full evidence package dict (will be JSON-serialized for hashing)
        investigator_id: Pseudonymized investigator ID for audit attribution

    Returns:
        The SHA-256 hex digest (64 chars).
    """
    # Canonical JSON: sorted keys, no whitespace, ensures deterministic hash
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()

    try:
        db.execute(
            text("""
                INSERT INTO evidence_seal (alert_id, evidence_hash, sealed_at, investigator_id)
                VALUES (:alert_id, :hash, :sealed_at, :investigator_id)
            """),
            {
                "alert_id": alert_id,
                "hash": digest,
                "sealed_at": datetime.now(timezone.utc),
                "investigator_id": investigator_id,
            },
        )
        log.info("evidence_sealed", alert_id=alert_id, sha256=digest[:16] + "...")
    except Exception as exc:
        log.error("evidence_seal_failed", alert_id=alert_id, error=str(exc))
        raise

    return digest


def verify_evidence_integrity(
    db: Session,
    alert_id: str,
    evidence: dict,
) -> bool:
    """
    Verify that an evidence bundle matches its stored seal.
    Returns True if intact, False if mismatch or no seal found.
    """
    row = db.execute(
        text("SELECT evidence_hash FROM evidence_seal WHERE alert_id = :id ORDER BY id DESC LIMIT 1"),
        {"id": alert_id},
    ).fetchone()

    if not row:
        log.warning("evidence_no_seal_found", alert_id=alert_id)
        return False

    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode()).hexdigest()
    match = actual == row.evidence_hash

    if not match:
        log.error("evidence_integrity_mismatch",
                  alert_id=alert_id,
                  stored=row.evidence_hash[:16],
                  actual=actual[:16])
    return match
