"""Application settings. Env-overridable with safe dev defaults.

All secrets (DB URL, signing keys, broker credentials) come from environment
variables; the defaults here are for local/dev only and must be overridden in
production (see ARCHITECTURE.md §deployment).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Storage
    database_url: str = _env("TRADING_DATABASE_URL", "sqlite+aiosqlite:///./trading.db")
    # Security (used for JWT signing and Fernet key encryption)
    secret_key: str = _env("TRADING_SECRET_KEY", "dev-secret-change-me")
    admin_username: str = _env("TRADING_ADMIN_USERNAME", "admin")
    admin_password: str = _env("TRADING_ADMIN_PASSWORD", "admin")
    access_token_expire_minutes: int = int(_env("TRADING_TOKEN_TTL_MIN", "480"))
    # Brokers
    bingx_base_url: str = _env("BINGX_BASE_URL", "https://open-api.bingx.com")
    tbank_token: str = _env("TBANK_TOKEN", "")
    tbank_sandbox: bool = _env("TBANK_SANDBOX", "true").lower() in ("1", "true", "yes")
    # Queue / cache
    redis_url: str = _env("TRADING_REDIS_URL", "redis://localhost:6379/0")
    redis_result_backend: str = _env(
        "TRADING_REDIS_RESULT_BACKEND", "redis://localhost:6379/1"
    )
    # Celery task guards (seconds)
    backtest_time_limit: int = int(_env("TRADING_BACKTEST_TIME_LIMIT", "900"))
    monte_carlo_time_limit: int = int(_env("TRADING_MC_TIME_LIMIT", "1200"))
    # API rate limiting (requests per minute per client IP)
    rate_limit_requests: int = int(_env("TRADING_RATE_LIMIT_REQUESTS", "300"))
    # Backtest defaults
    periods_per_year: int = int(_env("TRADING_PERIODS_PER_YEAR", "252"))
    # Market-data cache: seconds a loaded OHLCV window stays usable. Historical
    # bars don't change mid-session, so re-running a backtest (the usual
    # "tweak a param and re-run" loop) should not re-download anything. Set to 0
    # to disable caching entirely.
    data_cache_ttl: float = float(_env("TRADING_DATA_CACHE_TTL", "300"))


settings = Settings()
