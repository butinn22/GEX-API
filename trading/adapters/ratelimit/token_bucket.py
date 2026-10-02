"""Token-bucket rate limiting.

``TokenBucket`` is a correct, thread-safe token bucket with an injectable clock
(so tests are deterministic). ``RateLimiter`` manages a set of named scopes.
``RedisTokenBucket`` provides the same interface backed by Redis + Lua so the
limit is shared across workers/processes; it is used when a Redis client is
available, with the in-memory bucket as a bounded fallback.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["TokenBucket", "RateLimiter", "RedisTokenBucket"]

Clock = Callable[[], float]


@dataclass
class TokenBucket:
    """Token bucket: capacity tokens, refilling at ``refill_rate`` tokens/sec."""

    capacity: float
    refill_rate: float
    clock: Clock = field(default=time.monotonic, repr=False)
    _tokens: float = field(init=False)
    _last: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.refill_rate <= 0:
            raise ValueError("capacity and refill_rate must be > 0")
        self._tokens = self.capacity
        self._last = self.clock()

    def try_acquire(self, tokens: float = 1.0) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)`` without blocking."""
        with self._lock:
            now = self.clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_rate)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True, 0.0
            return False, (tokens - self._tokens) / self.refill_rate


class RateLimiter:
    """Named-scope token buckets (one bucket per provider/endpoint scope)."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def bucket(self, scope: str, capacity: float, refill_rate: float) -> TokenBucket:
        with self._lock:
            b = self._buckets.get(scope)
            if b is None or b.capacity != capacity or b.refill_rate != refill_rate:
                b = TokenBucket(capacity, refill_rate, clock=self._clock or time.monotonic)
                self._buckets[scope] = b
            return b

    def acquire(self, scope: str, capacity: float, refill_rate: float, tokens: float = 1.0) -> tuple[bool, float]:
        return self.bucket(scope, capacity, refill_rate).try_acquire(tokens)


# Lua script: atomically refill + consume, returns the new token count (or -1 if
# the key didn't exist and had to be initialised). Redis semantics: HSET the
# capacity/rate, then run the token math under a single Lua invocation.
_LUA_ACQUIRE = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local tokens = redis.call('HGET', key, 'tokens')
if not tokens then
  redis.call('HSET', key, 'tokens', capacity, 'updated', now)
  tokens = capacity
end
tokens = tonumber(tokens)
local updated = tonumber(redis.call('HGET', key, 'updated'))
tokens = math.min(capacity, tokens + (now - updated) * rate)
if tokens >= cost then
  tokens = tokens - cost
  redis.call('HSET', key, 'tokens', tokens, 'updated', now)
  return tokens
end
redis.call('HSET', key, 'tokens', tokens, 'updated', now)
return -1
"""


class RedisTokenBucket:
    """Redis-backed token bucket (atomic via Lua). Same interface as ``TokenBucket``.

    ``client`` is a redis-py client exposing ``eval`` (sync or async). Kept
    separate so the trading context can point at the same Redis as ``gex``.
    """

    def __init__(self, client, key: str, capacity: float, refill_rate: float) -> None:
        self._client = client
        self._key = key
        self.capacity = capacity
        self.refill_rate = refill_rate

    def try_acquire(self, tokens: float = 1.0) -> tuple[bool, float]:
        now = time.time()
        result = self._client.eval(
            _LUA_ACQUIRE, 1, self._key, self.capacity, self.refill_rate, tokens, now
        )
        allowed = float(result) >= 0
        retry = 0.0 if allowed else (tokens / self.refill_rate)
        return allowed, retry
