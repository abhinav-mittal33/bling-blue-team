"""
JWT RS256 authentication utilities (P1-7).
Accepts Bearer tokens alongside existing X-API-Key header — dual-mode during transition.

Token claims:
  sub  — caller identifier (service name or investigator ID)
  role — "graph_engine" | "investigator" | "internal"
  exp  — standard JWT expiry

Key config: JWT_PUBLIC_KEY (PEM-encoded RSA public key) in .env
"""
import structlog
from typing import Optional

import jwt
from jwt.exceptions import PyJWTError
from fastapi import HTTPException

from app.core.config import settings

logger = structlog.get_logger()


def verify_jwt_token(token: str) -> dict:
    """
    Decode and verify an RS256 JWT. Returns claims dict on success.
    Raises HTTPException 401 on any verification failure.
    """
    public_key = settings.jwt_public_key
    if not public_key:
        raise HTTPException(status_code=501, detail="JWT authentication not configured")

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"require": ["sub", "exp", "role"]},
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("jwt_expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        logger.warning("jwt_invalid", error=str(exc))
        raise HTTPException(status_code=401, detail="Invalid token")
    except PyJWTError as exc:
        logger.warning("jwt_error", error=str(exc))
        raise HTTPException(status_code=401, detail="Token verification failed")


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Return raw JWT from 'Bearer <token>' header, or None if header absent/malformed."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def get_caller_role(x_api_key: Optional[str], authorization: Optional[str]) -> Optional[str]:
    """
    Determine role from either JWT or API key.
    JWT → role claim. API key → inferred from which key matches.
    Returns None if neither credential is valid.
    """
    # Try JWT first — if Bearer header present but token fails, do NOT fall through
    token = extract_bearer_token(authorization)
    if token is not None:
        try:
            claims = verify_jwt_token(token)
            return claims.get("role")
        except HTTPException:
            logger.warning("jwt_failed_no_api_key_fallback")
            return None

    # Fall back to API-key role inference (only when no Bearer header present)
    if x_api_key:
        if x_api_key in settings.graph_engine_keys:
            return "graph_engine"
        if x_api_key in settings.investigator_keys:
            return "investigator"
        if x_api_key in settings.valid_api_keys:
            return "internal"

    return None
