"""Совместимый реэкспорт построителя ключей кэша (ring: adapters → ports).

Форма ключа — контракт кэша, а не деталь адаптера: реализация переехала в
``gex.ports.cache_keys`` (чистый форматтер строк, кольцо портов). Этот модуль
оставлен, чтобы существующие вызывающие (адаптеры, роутеры, тесты) продолжали
работать без переписывания; новые импорты обязаны идти через ``gex.ports.cache_keys``.

Реэкспорт явный (без ``import *``): звёздочка лишает статические гейты
(``check_imports``, pyflakes) возможности проверять имена.
"""

from gex.ports.cache_keys import (  # noqa: F401
    KEY_ROOT,
    KNOWN_PROVIDERS,
    PROVIDER_BYBIT,
    PROVIDER_FINNHUB,
    PROVIDER_MOEX,
    PROVIDER_SEC,
    PROVIDER_SCOPED_KINDS,
    PROVIDER_WEBULL,
    PROVIDER_YFINANCE,
    CacheKeyError,
    bars_key,
    chain_key,
    chain_key_v2,
    clean_segment,
    commodity_key,
    count,
    hv_key,
    normalize_provider,
    ohlcv_key,
    page_key,
    spot_key,
    state_key,
    symbol,
    timeframe,
)

__all__ = [
    "PROVIDER_YFINANCE",
    "PROVIDER_BYBIT",
    "PROVIDER_MOEX",
    "PROVIDER_WEBULL",
    "PROVIDER_SEC",
    "PROVIDER_FINNHUB",
    "KNOWN_PROVIDERS",
    "PROVIDER_SCOPED_KINDS",
    "KEY_ROOT",
    "CacheKeyError",
    "clean_segment",
    "state_key",
    "page_key",
    "normalize_provider",
    "symbol",
    "timeframe",
    "count",
    "chain_key",
    "chain_key_v2",
    "ohlcv_key",
    "bars_key",
    "spot_key",
    "hv_key",
    "commodity_key",
]
