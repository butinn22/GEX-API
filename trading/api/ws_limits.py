"""Per-IP concurrent WebSocket connection cap.

``BaseHTTPMiddleware`` never sees the ASGI ``websocket`` scope, so the HTTP
rate-limit middleware cannot bound WebSocket connections. This tiny module
tracks concurrent connections per client and refuses new ones past
``TRADING_WS_MAX_CONNECTIONS`` (default 20) with WS close code 1013
("try again later").
"""
from __future__ import annotations

import threading

from trading.config import settings

__all__ = ["WSConnectionLimiter", "ws_limiter"]


class WSConnectionLimiter:
    """Counts concurrent WebSocket connections per client key."""

    def __init__(self, max_per_ip: int) -> None:
        self._max = max(1, max_per_ip)
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def max_per_ip(self) -> int:
        return self._max

    def _client_key(self, ws) -> str:
        # Mirror the HTTP limiter: only trust X-Forwarded-For behind a proxy.
        if settings.trust_proxy:
            xff = ws.headers.get("x-forwarded-for")
            if xff:
                first = xff.split(",")[0].strip()
                if first:
                    return first
        return ws.client.host if ws.client else "unknown"

    def acquire(self, ws) -> bool:
        """Reserve a slot; ``False`` when the client is already at the cap."""
        key = self._client_key(ws)
        with self._lock:
            current = self._counts.get(key, 0)
            if current >= self._max:
                return False
            self._counts[key] = current + 1
            return True

    def release(self, ws) -> None:
        """Return a slot (idempotent: a missing key is ignored)."""
        key = self._client_key(ws)
        with self._lock:
            current = self._counts.get(key, 0)
            if current <= 1:
                self._counts.pop(key, None)
            else:
                self._counts[key] = current - 1

    def count(self, ws) -> int:
        with self._lock:
            return self._counts.get(self._client_key(ws), 0)


#: Process-wide limiter used by both WS entrypoints.
ws_limiter = WSConnectionLimiter(settings.ws_max_connections_per_ip)
