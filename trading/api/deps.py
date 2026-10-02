"""FastAPI dependencies: bearer auth + shared services."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from trading.application.keys_service import KeysService
from trading.config import settings
from trading.security import SecretError, decode_access_token

__all__ = ["require_auth", "get_keys_service"]

_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Return the authenticated subject or raise 401."""
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        return decode_access_token(settings.secret_key, creds.credentials)
    except SecretError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc


def get_keys_service() -> KeysService:
    return KeysService(settings.secret_key)
