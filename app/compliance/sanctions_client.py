"""
OFAC + UN Consolidated Sanctions List integration (P5-6).

Checks account IDs and names against:
  - OFAC SDN (Specially Designated Nationals) list
  - UN Security Council Consolidated List
  - Indian Ministry of Home Affairs designated entities

Sync: Celery Beat daily at 2:30am UTC (celeryconfig.py).
Check: Called at scoring time for REVIEW+ transactions.

List storage: Redis SET `sanctions:{list_type}` — O(1) lookup.
"""
from __future__ import annotations
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()

_OFAC_LIST_KEY = "sanctions:ofac"
_UN_LIST_KEY = "sanctions:un"
_MHA_LIST_KEY = "sanctions:mha_india"


def is_sanctioned(account_id: str, name: str | None = None) -> dict:
    """
    Check if account or name appears on any sanctions list.
    Returns {'sanctioned': bool, 'list': str|None, 'matched': str|None}.
    """
    try:
        from app.utils.redis_client import get_redis
        r = get_redis()

        for list_key, list_name in [
            (_OFAC_LIST_KEY, "OFAC"),
            (_UN_LIST_KEY, "UN"),
            (_MHA_LIST_KEY, "MHA_INDIA"),
        ]:
            if r.sismember(list_key, account_id):
                logger.critical(
                    "sanctions_match_account_id",
                    account_id=account_id[:8] + "...",
                    sanctions_list=list_name,
                )
                return {"sanctioned": True, "list": list_name, "matched": "account_id"}

            if name and r.sismember(list_key, name.upper()):
                return {"sanctioned": True, "list": list_name, "matched": "name"}

    except Exception as exc:
        logger.error("sanctions_check_failed", error=str(exc))

    return {"sanctioned": False, "list": None, "matched": None}


def sync_sanctions() -> None:
    """
    Celery Beat task: sync sanctions lists from OFAC + UN + MHA.
    Runs daily at 2:30am UTC. STUB until live URLs are configured.
    """
    from app.core.config import settings

    synced = 0
    errors = []

    for list_type, url_env, redis_key in [
        ("OFAC", "OFAC_SDN_URL", _OFAC_LIST_KEY),
        ("UN", "UN_SANCTIONS_URL", _UN_LIST_KEY),
        ("MHA", "MHA_SANCTIONS_URL", _MHA_LIST_KEY),
    ]:
        url = getattr(settings, url_env.lower(), "")
        if not url:
            logger.debug("sanctions_sync_skipped", list_type=list_type, reason="no url configured")
            continue

        try:
            entries = _fetch_list(url, list_type)
            _update_redis(redis_key, entries)
            synced += len(entries)
            logger.info("sanctions_list_synced", list_type=list_type, count=len(entries))
        except Exception as exc:
            errors.append(f"{list_type}: {exc}")
            logger.error("sanctions_sync_failed", list_type=list_type, error=str(exc))

    if errors:
        logger.error("sanctions_sync_partial_failure", errors=errors)
    else:
        logger.info("sanctions_sync_complete", total_entries=synced)


def _fetch_list(url: str, list_type: str) -> list[str]:
    """Fetch and parse sanctions list from URL. Returns list of identifiers."""
    import httpx
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    # Parsing logic varies by list type — stub returns empty list
    # Full implementation: parse OFAC XML, UN XML, MHA CSV
    return []


def _update_redis(redis_key: str, entries: list[str]) -> None:
    from app.utils.redis_client import get_redis
    r = get_redis()
    if not entries:
        return
    tmp_key = f"{redis_key}:tmp"
    r.delete(tmp_key)
    r.sadd(tmp_key, *entries)
    # Atomic swap — no gap where sismember returns empty
    r.rename(tmp_key, redis_key)
    r.set(f"{redis_key}:updated_at", datetime.now(timezone.utc).isoformat())
