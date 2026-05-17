import hashlib
from fastapi import Header, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.exceptions import unauthorized, forbidden

limiter = Limiter(key_func=get_remote_address)


async def require_any_key(x_api_key: str = Header(None)) -> str:
    if not x_api_key or x_api_key not in settings.valid_api_keys:
        raise unauthorized()
    return x_api_key


async def require_graph_engine_key(x_api_key: str = Header(None)) -> str:
    if not x_api_key:
        raise unauthorized()
    if x_api_key not in settings.valid_api_keys:
        raise unauthorized()
    if x_api_key not in settings.graph_engine_keys:
        raise forbidden()
    return x_api_key


async def require_investigator_key(x_api_key: str = Header(None)) -> str:
    if not x_api_key:
        raise unauthorized()
    if x_api_key not in settings.valid_api_keys:
        raise unauthorized()
    if x_api_key not in settings.investigator_keys:
        raise forbidden()
    return x_api_key


def pseudonymize(account_id: str) -> str:
    """Pseudonymize account ID for safe logging. Never log raw account IDs."""
    return hashlib.sha256(f"{settings.salt}{account_id}".encode()).hexdigest()[:12]
