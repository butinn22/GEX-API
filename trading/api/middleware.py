"""Rate-limit middleware (token bucket per client IP)."""
from __future__ import annotations

import threading

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from trading.adapters.ratelimit import TokenBucket

__all__ = ["RateLimitMiddleware"]


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, requests: int = 300, per_seconds: float = 60.0) -> None:
        super().__init__(app)
        self.capacity = requests
        self.refill_rate = requests / per_seconds
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str) -> TokenBucket:
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(self.capacity, self.refill_rate)
            return self._buckets[key]

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        allowed, retry = self._bucket(client).try_acquire()
        if not allowed:
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(int(retry) + 1)},
            )
        return await call_next(request)
