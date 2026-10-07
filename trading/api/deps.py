"""FastAPI dependencies: bearer auth + shared services."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from trading.application.keys_service import KeysService
from trading.config import settings
from trading.security import SecretError, decode_access_token

__all__ = ["require_auth", "get_keys_service", "ws_bearer_token", "validate_ws_jwt"]

_bearer = HTTPBearer(auto_error=False)

#: Subprotocol name the browser streams offer alongside the JWT:
#: ``new WebSocket(url, ["gex.jwt", token])``. The header is not logged by
#: uvicorn by default (unlike a query string), so it is the preferred carrier.
WS_SUBPROTOCOL = "gex.jwt"


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
    return KeysService(settings.encryption_secret)


def ws_bearer_token(ws) -> str | None:
    """Extract the JWT from a WebSocket handshake.

    Preferred carrier is the ``Sec-WebSocket-Protocol`` subprotocol list
    (``["gex.jwt", "<jwt>"]``); ``?token=<jwt>`` is the fallback for clients
    that cannot set subprotocols.
    """
    proto = ws.headers.get("sec-websocket-protocol")
    if proto:
        parts = [p.strip() for p in proto.split(",")]
        if WS_SUBPROTOCOL in parts:
            idx = parts.index(WS_SUBPROTOCOL)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ws.query_params.get("token")


def validate_ws_jwt(ws) -> bool:
    """True when the handshake carries a valid, unexpired JWT."""
    token = ws_bearer_token(ws)
    if not token:
        return False
    try:
        decode_access_token(settings.secret_key, token)
        return True
    except SecretError:
        return False
