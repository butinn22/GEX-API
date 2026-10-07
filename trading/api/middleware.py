"""Rate-limit middleware: per-IP token bucket + a dedicated login bucket/lockout.

Design (Round-2 §3)
-------------------
* **Global bucket** — one in-process token bucket per client IP (default
  300/min, ``TRADING_RATE_LIMIT_REQUESTS``). Rejections return ``429`` with a
  concrete ``Retry-After`` header.
* **Login bucket** — ``POST /api/v1/auth/token`` is throttled far more tightly
  (``TRADING_LOGIN_RATE_LIMIT``, default 10/min) on its own ``login:<ip>`` key,
  fully independent of the global bucket.
* **Login lockout** — after ``TRADING_LOGIN_LOCKOUT_FAILURES`` (default 5)
  *failed* (401) logins within the lockout window, that IP is locked for
  ``TRADING_LOGIN_LOCKOUT_SECONDS`` (default 900); further attempts are ``429``
  until it expires. A successful login clears the counter.
* **Bounded memory** — the bucket registry is an LRU ``OrderedDict`` capped at
  ``max_size`` (default 50k), so a spoofed-IP flood cannot grow it forever.
* **Trusted proxy** — ``X-Forwarded-For`` is honoured **only** when
  ``TRADING_TRUST_PROXY`` is set; otherwise the socket peer is used, so a rogue
  header cannot dodge or poison a bucket.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from trading.adapters.ratelimit import TokenBucket
from trading.config import settings
from trading.observability import RATE_LIMIT_HITS

__all__ = ["RateLimitMiddleware", "LOGIN_PATH", "DEFAULT_MAX_BUCKETS"]

#: Path that gets the tighter bucket + the failure lockout.
LOGIN_PATH = "/api/v1/auth/token"
#: Upper bound on the bucket registry (LRU-evicted beyond this).
DEFAULT_MAX_BUCKETS = 50_000


class _LRUBuckets:
    """A bounded registry of identically-configured token buckets.

    One registry per policy (global vs login). Keys are evicted
    least-recently-used once ``max_size`` is exceeded, so the dict cannot grow
    without bound under a spoofed-IP flood.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        max_size: int = DEFAULT_MAX_BUCKETS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.max_size = max_size
        self._clock = clock
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    def acquire(self, key: str, tokens: float = 1.0) -> tuple[bool, float]:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(
                    self.capacity, self.refill_rate, clock=self._clock or time.monotonic
                )
                self._buckets[key] = bucket
                while len(self._buckets) > self.max_size:
                    self._buckets.popitem(last=False)  # evict least-recently-used
            else:
                self._buckets.move_to_end(key)
            return bucket.try_acquire(tokens)

    def __len__(self) -> int:
        return len(self._buckets)


class _LoginLockout:
    """Consecutive-failure counter with a timed lockout, keyed by client IP."""

    def __init__(
        self,
        failures: int,
        lock_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.failures = failures
        self.lock_seconds = lock_seconds
        self._clock = clock or time.monotonic
        self._hits: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def locked_for(self, key: str) -> float:
        """Seconds remaining on an active lockout (``0.0`` when not locked)."""
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            now = self._clock()
            if until <= now:
                self._locked_until.pop(key, None)
                return 0.0
            return until - now

    def record_failure(self, key: str) -> bool:
        """Register a failed login; ``True`` when it trips a lock."""
        if self.failures <= 0:
            return False
        with self._lock:
            now = self._clock()
            recent = [t for t in self._hits.get(key, []) if now - t <= self.lock_seconds]
            recent.append(now)
            if len(recent) >= self.failures:
                self._hits.pop(key, None)
                self._locked_until[key] = now + self.lock_seconds
                return True
            self._hits[key] = recent
            return False

    def record_success(self, key: str) -> None:
        """Clear a key's failure history after a successful login."""
        with self._lock:
            self._hits.pop(key, None)
            self._locked_until.pop(key, None)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        requests: int = 300,
        per_seconds: float = 60.0,
        login_requests: int | None = None,
        login_window: float = 60.0,
        lockout_failures: int | None = None,
        lockout_seconds: float | None = None,
        trust_proxy: bool | None = None,
        max_buckets: int = DEFAULT_MAX_BUCKETS,
    ) -> None:
        super().__init__(app)
        self.capacity = requests
        self.refill_rate = requests / per_seconds
        self.login_capacity = (
            login_requests if login_requests is not None else settings.login_rate_limit
        )
        self.login_refill = self.login_capacity / login_window
        self.trust_proxy = settings.trust_proxy if trust_proxy is None else trust_proxy
        self._global = _LRUBuckets(self.capacity, self.refill_rate, max_buckets)
        self._login = _LRUBuckets(self.login_capacity, self.login_refill, max_buckets)
        self._lockout = _LoginLockout(
            settings.login_lockout_failures if lockout_failures is None else lockout_failures,
            settings.login_lockout_seconds if lockout_seconds is None else lockout_seconds,
        )

    def _client(self, request: Request) -> str:
        if self.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                first = forwarded.split(",")[0].strip()
                if first:
                    return first
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        client = self._client(request)
        is_login = request.method == "POST" and request.url.path == LOGIN_PATH

        if is_login:
            remaining = self._lockout.locked_for(client)
            if remaining > 0:
                RATE_LIMIT_HITS.labels(scope="login_lockout").inc()
                return JSONResponse(
                    {"detail": "too many failed login attempts"},
                    status_code=429,
                    headers={"Retry-After": str(int(remaining) + 1)},
                )

        registry = self._login if is_login else self._global
        key = f"login:{client}" if is_login else client
        allowed, retry = registry.acquire(key)
        if not allowed:
            RATE_LIMIT_HITS.labels(scope="login" if is_login else "global").inc()
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(int(retry) + 1)},
            )

        response = await call_next(request)
        if is_login:
            if response.status_code == 401:
                self._lockout.record_failure(client)
            elif response.status_code < 400:
                self._lockout.record_success(client)
        return response
