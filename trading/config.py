"""Application settings. Env-overridable with safe dev defaults.

All secrets (DB URL, signing keys, broker credentials) come from environment
variables; the defaults here are for local/dev only and must be overridden in
production (see ARCHITECTURE.md §deployment). Setting ``TRADING_ENV=production``
makes the app **fail fast** at startup if any default secret is still in place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_DEV_SECRET = "dev-secret-change-me"
_DEV_USER = "admin"
_DEV_PASSWORD = "admin"
_PRODUCTION_ENVS = ("prod", "production")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    #: Deployment environment: "dev" (default) | "production"/"prod".
    env: str = _env("TRADING_ENV", "dev")
    # Storage
    database_url: str = _env("TRADING_DATABASE_URL", "sqlite+aiosqlite:///./trading.db")
    # Security (JWT signing)
    secret_key: str = _env("TRADING_SECRET_KEY", _DEV_SECRET)
    # Security (at-rest encryption of broker credentials). Falls back to
    # ``secret_key`` when unset so existing encrypted rows keep decoding.
    broker_key_secret: str = _env("TRADING_BROKER_KEY_SECRET", "")
    admin_username: str = _env("TRADING_ADMIN_USERNAME", _DEV_USER)
    admin_password: str = _env("TRADING_ADMIN_PASSWORD", _DEV_PASSWORD)
    access_token_expire_minutes: int = int(_env("TRADING_TOKEN_TTL_MIN", "480"))
    #: Static token required on ``GET /metrics`` in production (empty = open).
    metrics_token: str = _env("TRADING_METRICS_TOKEN", "")
    #: Serve /docs, /redoc, /openapi.json (disabled in production unless "1").
    enable_docs: bool = _env("TRADING_ENABLE_DOCS", "").lower() in ("1", "true", "yes")
    #: Trust X-Forwarded-For for client-IP extraction (only behind a known proxy).
    trust_proxy: bool = _env("TRADING_TRUST_PROXY", "").lower() in ("1", "true", "yes")
    #: Auth bucket size per client IP (requests/min) and login lockout policy.
    login_rate_limit: int = int(_env("TRADING_LOGIN_RATE_LIMIT", "10"))
    login_lockout_failures: int = int(_env("TRADING_LOGIN_LOCKOUT_FAILURES", "5"))
    login_lockout_seconds: float = float(_env("TRADING_LOGIN_LOCKOUT_SECONDS", "900"))
    #: Max concurrent WebSocket connections per client IP.
    ws_max_connections_per_ip: int = int(_env("TRADING_WS_MAX_CONNECTIONS", "20"))
    #: Shared secret required by the ``/ws/client`` handshake frame.
    local_client_token: str = _env("TRADING_LOCAL_CLIENT_TOKEN", "")
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

    @property
    def is_production(self) -> bool:
        return self.env.strip().lower() in _PRODUCTION_ENVS

    @property
    def encryption_secret(self) -> str:
        """Key material for Fernet at-rest encryption (broker credentials)."""
        return self.broker_key_secret or self.secret_key

    def __post_init__(self) -> None:
        # Fail fast in production if any default/dev secret is still present.
        if not self.is_production:
            return
        problems: list[str] = []
        if self.secret_key == _DEV_SECRET:
            problems.append("TRADING_SECRET_KEY is still the dev default")
        if self.admin_username == _DEV_USER and self.admin_password == _DEV_PASSWORD:
            problems.append("TRADING_ADMIN_USERNAME/PASSWORD are still 'admin'/'admin'")
        if problems:
            raise RuntimeError(
                "Refusing to start in production with insecure defaults: "
                + "; ".join(problems)
            )


settings = Settings()
