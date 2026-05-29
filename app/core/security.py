import hashlib
import hmac
from typing import Optional

from fastapi import Header, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.exceptions import unauthorized, forbidden

limiter = Limiter(key_func=get_remote_address)


def _resolve_role(x_api_key: Optional[str], authorization: Optional[str]) -> Optional[str]:
    """
    Dual-mode auth (P1-7): try JWT Bearer first, then X-API-Key.
    Returns role string or None on failure.
    """
    from app.utils.auth import get_caller_role
    return get_caller_role(x_api_key, authorization)


async def require_any_key(
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> str:
    role = _resolve_role(x_api_key, authorization)
    if not role:
        raise unauthorized()
    return role


async def require_graph_engine_key(
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> str:
    role = _resolve_role(x_api_key, authorization)
    if not role:
        raise unauthorized()
    if role not in ("graph_engine", "internal"):
        raise forbidden()
    return role


async def require_investigator_key(
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> str:
    role = _resolve_role(x_api_key, authorization)
    if not role:
        raise unauthorized()
    if role not in ("investigator", "internal"):
        raise forbidden()
    return role


async def require_internal_key(
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> str:
    """INTERNAL_API_KEY only — never exposed to graph_engine or investigator callers."""
    role = _resolve_role(x_api_key, authorization)
    if not role:
        raise unauthorized()
    if role != "internal":
        raise forbidden()
    return role


def pseudonymize(account_id: str) -> str:
    """
    HMAC-SHA256 pseudonymization for PII in logs (P1-8).
    Uses PSEUDONYMIZATION_KEY (separate from SALT) so keys can be rotated independently.
    Falls back to SHA-256+SALT if PSEUDONYMIZATION_KEY not yet configured (migration window).
    Never log raw account IDs — always pass through pseudonymize() first.
    """
    key = settings.pseudonymization_key
    if key:
        return hmac.new(
            key.encode(),
            account_id.encode(),
            hashlib.sha256,
        ).hexdigest()[:12]
    # Fallback: legacy sha256+salt (pre-P1-8 behavior)
    return hashlib.sha256(f"{settings.salt}{account_id}".encode()).hexdigest()[:12]
