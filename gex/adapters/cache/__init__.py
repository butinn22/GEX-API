"""Cache adapters (ring: adapters) — Redis-реализация CachePort: версионированный envelope, SWR, single-flight.

Ключи строятся только через :mod:`gex.adapters.cache.keys` (провайдер всегда входит в ключ).
Конверт значения — :mod:`gex.adapters.cache.envelope`; схождение запросов —
:mod:`gex.adapters.cache.singleflight`.
"""

from gex.adapters.cache.envelope import (
    ENVELOPE_VERSION,
    LEGACY_VERSION,
    Envelope,
    RedisEnvelopeCache,
    etag_for,
)
from gex.adapters.cache.keys import (
    CacheKeyError,
    chain_key,
    clean_segment,
    commodity_key,
    count,
    hv_key,
    normalize_provider,
    ohlcv_key,
    spot_key,
    symbol,
    timeframe,
)
from gex.adapters.cache.singleflight import LOCK_PREFIX, LocalSingleFlight, RedisSingleFlight

__all__ = [
    "ENVELOPE_VERSION",
    "LEGACY_VERSION",
    "LOCK_PREFIX",
    "CacheKeyError",
    "Envelope",
    "LocalSingleFlight",
    "RedisEnvelopeCache",
    "RedisSingleFlight",
    "chain_key",
    "clean_segment",
    "commodity_key",
    "count",
    "etag_for",
    "hv_key",
    "normalize_provider",
    "ohlcv_key",
    "spot_key",
    "symbol",
    "timeframe",
]
