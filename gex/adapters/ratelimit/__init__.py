"""Rate-limit adapters (ring: adapters) — единственный авторитет лимитов провайдеров (Redis + Lua).

Лимиты провайдеров обязаны быть распределёнными: per-process ведро умножает фактическую
частоту на число воркеров, то есть отдаёт провайдеру ровно то, от чего он защищает.
Per-IP лимитер тоже переведён на общий Redis (защита от перебора пароля), а
``gex.rate_limiter.IpRateLimiter`` остаётся фасадом с локальным фолбэком.
"""

from gex.adapters.ratelimit.redis_lua import (
    KEY_PREFIX,
    LUA_TOKEN_BUCKET,
    RedisIpLimiter,
    RedisRateLimiter,
    RedisTokenBucket,
    TokenBucketScript,
    bucket_ttl_ms,
    canonical_provider,
    ip_rate_limit_key,
    make_limits,
    rate_limit_key,
)

__all__ = [
    "KEY_PREFIX",
    "LUA_TOKEN_BUCKET",
    "RedisIpLimiter",
    "RedisRateLimiter",
    "RedisTokenBucket",
    "TokenBucketScript",
    "bucket_ttl_ms",
    "canonical_provider",
    "ip_rate_limit_key",
    "make_limits",
    "rate_limit_key",
]
