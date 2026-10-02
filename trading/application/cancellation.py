"""Cooperative cancellation for long-running runs (backtests, Monte-Carlo).

A 100 000-path Monte-Carlo can take a while, and the user must be able to stop
it. There are two ways a run stops, and this module supports both:

1. **Client-side abort** — the browser aborts the fetch. Starlette raises
   ``asyncio.CancelledError`` inside the handler and the request unwinds.
   Nothing here is involved; it is listed for completeness.

2. **Explicit cancel request** — the client starts a run with a ``run_token`` and
   later ``POST``s ``/backtest/cancel/{token}``. The engine polls that token
   inside its hot loop and stops **between chunks of work**, so the CPU is freed
   promptly instead of finishing a five-minute simulation nobody wants.

Two things make this actually work rather than look good:

* the heavy synchronous work runs in a thread (:func:`asyncio.to_thread`), so
  the event loop stays free to serve the cancel request while the run proceeds;
* the registry is **Redis-backed** when a broker is reachable, so a cancel
  issued to one uvicorn worker (or from a Celery-side API call) also stops a run
  executing in a *different* worker. Without Redis it degrades to a process-local
  set, which is still correct for a single-worker dev server.

The check is deliberately cheap: a local set lookup, plus a *throttled* Redis
``EXISTS`` (at most once per :attr:`RunRegistry.poll_interval`). Polling Redis on
every chunk would cost more than the chunk itself for small chunks; polling it
never would make cross-worker cancel useless.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

__all__ = [
    "RunCancelled",
    "CancelToken",
    "RunRegistry",
    "run_registry",
    "is_valid_token",
]

log = logging.getLogger(__name__)

#: Tokens come from ``uuid4().hex``; anything else is rejected so a caller cannot
#: turn the cancel endpoint into a Redis key-injection primitive.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_token(token: str | None) -> bool:
    """Whether ``token`` is a well-formed run handle."""
    return bool(token) and bool(_TOKEN_RE.match(token))

#: How long a cancel flag lives in Redis. Longer than any sane run, so a cancel
#: can never expire while someone is still waiting to observe it.
_TTL_SECONDS = 6 * 3600


class RunCancelled(RuntimeError):
    """Raised inside a run when the client asked it to stop."""

    def __init__(self, token: str = "") -> None:
        super().__init__(f"run cancelled (token={token})" if token else "run cancelled")
        self.token = token


class CancelToken:
    """A handle a caller passes into an engine so it can be stopped.

    Engines call :meth:`check` between units of work; it is a no-op unless a
    cancel has actually been requested, so it is safe to call in a hot loop.
    """

    __slots__ = ("token", "_registry")

    def __init__(self, token: str, registry: "RunRegistry") -> None:
        self.token = token
        self._registry = registry

    @property
    def cancelled(self) -> bool:
        return self._registry.is_cancelled(self.token)

    def check(self) -> None:
        """Raise :class:`RunCancelled` if the client asked to stop."""
        if self._registry.is_cancelled(self.token):
            raise RunCancelled(self.token)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CancelToken(token={self.token!r}, cancelled={self.cancelled})"


class RunRegistry:
    """Tracks which run tokens have been cancelled.

    Backed by Redis when available (cross-worker), otherwise process-local.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        poll_interval: float = 0.25,
        prefix: str = "trading:run:cancel:",
    ) -> None:
        self._cancelled: set[str] = set()
        self._active: set[str] = set()
        self._seen: set[str] = set()
        self._cache: dict[str, tuple[float, bool]] = {}
        self._poll_interval = poll_interval
        self._prefix = prefix
        self._redis: Any | None = None
        self._redis_tried = False
        self._redis_url = redis_url

    # ── backend ────────────────────────────────────────────────────────

    def _client(self) -> Any | None:
        """Lazily build a Redis client. Never raises: a missing broker just
        means the registry stays process-local."""
        if self._redis_tried:
            return self._redis
        self._redis_tried = True
        url = self._redis_url
        if url is None:
            try:
                from trading.config import settings

                url = settings.redis_url
            except Exception:  # pragma: no cover - config always importable
                url = None
        if not url:
            return None
        try:
            import redis

            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=0.15,
                socket_timeout=0.15,
            )
            client.ping()
        except Exception as exc:
            log.info("run registry: Redis unavailable (%s); using process-local set", exc)
            return None
        self._redis = client
        return client

    @property
    def backend(self) -> str:
        return "redis" if self._client() is not None else "local"

    def _key(self, token: str) -> str:
        return f"{self._prefix}{token}"

    # ── public API ─────────────────────────────────────────────────────

    def new(self, token: str | None = None) -> CancelToken:
        """Create (or adopt) a token for a new run.

        An existing cancel flag is deliberately **preserved**. A queued Celery
        job can be cancelled before a worker ever picks it up, and wiping the
        flag here would silently ignore that Stop. Stale flags are cleaned up by
        :meth:`clear` when a run *finishes*, which is the correct lifetime: a
        unique token is never left dirty between runs.
        """
        if token is None:
            token = uuid.uuid4().hex
        elif not _TOKEN_RE.match(token):
            raise ValueError("invalid run token")
        self._seen.add(token)
        self._active.add(token)
        return CancelToken(token, self)

    def is_active(self, token: str) -> bool:
        return token in self._active

    def active_tokens(self) -> list[str]:
        """Tokens of runs currently executing in this process."""
        return sorted(self._active)

    def mark_finished(self, token: str) -> None:
        """Record that a run stopped (normally, by error, or by cancellation).

        Only clears the active flag; use :meth:`clear` to also drop the
        cancellation state (which is what the API layer does in its ``finally``).
        """
        self._active.discard(token)

    def cancel(self, token: str) -> bool:
        """Flag ``token`` as cancelled. Returns ``True`` if the run was still active."""
        if not _TOKEN_RE.match(token or ""):
            return False
        was_active = token in self._active
        self._cancelled.add(token)
        self._cache[token] = (time.monotonic(), True)
        client = self._client()
        if client is not None:
            try:
                client.setex(self._key(token), _TTL_SECONDS, "1")
            except Exception as exc:  # broker hiccup must not 500 the endpoint
                log.warning("run registry: could not publish cancel to Redis: %s", exc)
        return was_active

    def is_cancelled(self, token: str) -> bool:
        if not token:
            return False
        if token in self._cancelled:  # same-worker fast path
            return True
        client = self._client()
        if client is None:
            return False
        now = time.monotonic()
        hit = self._cache.get(token)
        if hit is not None and now - hit[0] < self._poll_interval:
            return hit[1]
        try:
            flag = bool(client.exists(self._key(token)))
        except Exception:
            return hit[1] if hit is not None else False
        self._cache[token] = (now, flag)
        if flag:
            self._cancelled.add(token)
        return flag

    def clear(self, token: str) -> None:
        """Forget a finished run (called when a run returns or errors)."""
        self._active.discard(token)
        self._cancelled.discard(token)
        self._cache.pop(token, None)
        client = self._client()
        if client is not None and token:
            try:
                client.delete(self._key(token))
            except Exception:
                pass

    def reset(self) -> None:
        """Drop all local state. Test helper."""
        self._cancelled.clear()
        self._active.clear()
        self._seen.clear()
        self._cache.clear()


#: Process-wide registry used by the API layer and Celery tasks.
run_registry = RunRegistry()
