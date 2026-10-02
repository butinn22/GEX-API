"""Единая точка чтения конфигурации (ring: frameworks).

Зачем отдельный модуль, если есть ``gex.auth.config``
----------------------------------------------------
``gex/auth/config.py`` — это **загрузчик** (pydantic-settings + валидация секретов). Он остаётся
единственным, кто читает ``.env``/env. ``gex/settings.py`` — типизированный **фасад** над ним для
новых колец (adapters/application/domain): датаклассы с явными полями, чтобы код не лазил в
плоский объект ``settings`` по строкам.

Правила (почему так, а не иначе)
--------------------------------
1. **Никаких ``getattr(settings, "NAME", default)``.** Обращение к полю идёт напрямую
   (``settings.REDIS_HOST``). Опечатка или переименование обязаны падать громко: тихий фолбэк
   превращает неверную конфигурацию в «незаметно выключенную функциональность» — ровно то, что
   произошло с ``ORCHESTRATOR_ENABLED`` (аудит: F-01, ``gex/auth/config.py:123`` + ``.env:44``).
2. **Состояние обязано быть наблюдаемым.** ``effective_flags()`` возвращает действующие флаги для
   стартового лога и ``/health``: «выключено» должно быть видно, а не выводиться из поведения.
3. **Один источник значений.** Значения по-прежнему приходят из ``Settings``; здесь — только
   группировка и приведение типов. Дублирования дефолтов нет.

Использование::

    from gex.settings import load
    cfg = load()
    cfg.redis.host, cfg.redis.port, cfg.gateway.enabled
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

# Реэкспорт загрузчика, чтобы новые модули импортировали конфиг из одного места.
from gex.auth.config import Settings, get_settings, settings  # noqa: F401  (re-export)

__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "RedisSettings",
    "RateLimitSettings",
    "SECSettings",
    "ProviderSettings",
    "GatewaySettings",
    "HttpSettings",
    "YfSettings",
    "BackendSettings",
    "load",
    "reload",
    "effective_flags",
    "is_production",
    "is_worker_role",
    "Settings",
    "get_settings",
    "settings",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Группы настроек
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppSettings:
    """Идентификация процесса и режим исполнения."""

    name: str
    version: str
    env: str            # development | production
    role: str           # all | web | worker
    debug: bool
    debug_sql: bool
    testing: bool

    @property
    def production(self) -> bool:
        return self.env == "production"

    @property
    def worker_enabled(self) -> bool:
        """Фоновые сервисы (scheduler, очереди, сканеры) в этом процессе."""
        return self.role in ("worker", "all")

    @property
    def api_enabled(self) -> bool:
        """Обслуживание HTTP в этом процессе."""
        return self.role in ("web", "all")


@dataclass(frozen=True)
class DatabaseSettings:
    url: str
    allow_fallback: bool
    pool_size: int
    max_overflow: int

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


@dataclass(frozen=True)
class RedisSettings:
    host: str
    port: int
    db: int
    password: str | None
    ttl_default: int
    maxmemory: str
    maxmemory_policy: str

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class RateLimitSettings:
    """Per-deployment бюджеты обращения к внешним провайдерам.

    ВНИМАНИЕ: значения обязаны применяться распределённо (Redis+Lua). Per-process бакеты
    умножают лимит на число uvicorn-воркеров (аудит: PC-05, SVC-08, B-03).
    """

    sec_user_agent: str
    sec_rate_per_sec: float
    sec_rate_burst: int
    sec_facts_ttl_hours: int


@dataclass(frozen=True)
class SECSettings:
    user_agent: str
    rate_per_sec: float
    rate_burst: int
    facts_ttl_hours: int


@dataclass(frozen=True)
class ProviderSettings:
    finnhub_api_key: str
    telegram_bot_token: str
    telegram_bot_username: str
    telegram_chat_id: str
    telegram_polling: bool
    telegram_webhook_auto: bool
    telegram_webhook_secret: str


@dataclass(frozen=True)
class GatewaySettings:
    """Настройки центрального шлюза к внешним провайдерам (оркестратор).

    ``enabled=False`` — это **явное** решение, а не случайность: оно попадает в
    :func:`effective_flags` и в стартовый лог. Если шлюз выключен, лимиты/кэш провайдеров
    работают в режиме вырожденной деградации (см. ROADMAP §3.1).
    """

    enabled: bool
    redis_url: str
    default_interactive_timeout_ms: int
    default_background_timeout_ms: int
    workers_enabled: bool
    workers_concurrency: int
    cache_version: str
    inline_execution: bool
    providers_cache_ttl_seconds: int


@dataclass(frozen=True)
class HttpSettings:
    """Единая политика исходящих HTTP-вызовов (см. ``gex/adapters/transport/http.py``).

    Раньше эти числа жили в тридцати местах в виде литералов ``timeout=20`` — теперь у процесса
    одно мнение о том, сколько ждать и сколько раз повторять.
    """

    connect_timeout: float
    read_timeout: float
    max_attempts: int
    backoff_base: float
    backoff_max: float
    max_elapsed: float
    pool_maxsize: int


@dataclass(frozen=True)
class YfSettings:
    """Дедлайны yfinance (см. ``gex/adapters/transport/yf_transport.py``).

    У библиотеки нет параметра timeout: вызов может висеть бесконечно и держать воркер.
    Эти числа — единственное, что ограничивает ожидание.
    """

    deadline_seconds: float
    download_deadline_seconds: float
    max_deadline_seconds: float
    #: Жёсткий потолок одновременных вызовов yfinance (защита от взрыва тредов).
    max_concurrency: int
    #: Сколько ждать свободный слот перед отказом (backpressure, а не очередь).
    queue_wait_seconds: float
    #: Порог брошенных вызовов, после которого включается предохранитель.
    breaker_threshold: int
    #: Длительность отказа предохранителя (backlog должен стечь).
    breaker_cooldown_seconds: float


@dataclass(frozen=True)
class BackendSettings:
    """Сгруппированный доступ к конфигурации для колец adapters/application."""

    app: AppSettings
    database: DatabaseSettings
    redis: RedisSettings
    rate_limit: RateLimitSettings
    sec: SECSettings
    providers: ProviderSettings
    gateway: GatewaySettings
    http: HttpSettings
    yf: YfSettings
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    frontend_url: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Сборка из Settings (единственный загрузчик .env — gex.auth.config)
# ─────────────────────────────────────────────────────────────────────────────

def _build(s: Settings) -> BackendSettings:
    """Собирает типизированный фасад из pydantic-настроек.

    Все обращения — прямые (``s.REDIS_HOST``): отсутствующее поле должно падать, а не подменяться
    дефолтом (правило 1 в докстринге модуля).
    """
    app = AppSettings(
        name=s.APP_NAME,
        version=s.VERSION,
        env=s.APP_ENV,
        role=s.ROLE,
        debug=s.DEBUG,
        debug_sql=s.DEBUG_SQL,
        testing=s.TESTING,
    )
    database = DatabaseSettings(
        url=s.DATABASE_URL,
        allow_fallback=s.ALLOW_DB_FALLBACK,
        pool_size=s.DB_POOL_SIZE,
        max_overflow=s.DB_MAX_OVERFLOW,
    )
    redis = RedisSettings(
        host=s.REDIS_HOST,
        port=int(s.REDIS_PORT),
        db=int(s.REDIS_DB),
        password=s.REDIS_PASSWORD,
        ttl_default=int(s.REDIS_TTL_DEFAULT),
        maxmemory=s.REDIS_MAXMEMORY,
        maxmemory_policy=s.REDIS_MAXMEMORY_POLICY,
    )
    sec = SECSettings(
        user_agent=s.SEC_USER_AGENT,
        rate_per_sec=float(s.SEC_RATE_PER_SEC),
        rate_burst=int(s.SEC_RATE_BURST),
        facts_ttl_hours=int(s.SEC_FACTS_TTL_HOURS),
    )
    gateway_redis_url = s.ORCHESTRATOR_REDIS_URL or redis.url
    gateway = GatewaySettings(
        enabled=bool(s.ORCHESTRATOR_ENABLED),
        redis_url=gateway_redis_url,
        default_interactive_timeout_ms=int(s.ORCHESTRATOR_DEFAULT_INTERACTIVE_TIMEOUT_MS),
        default_background_timeout_ms=int(s.ORCHESTRATOR_DEFAULT_BACKGROUND_TIMEOUT_MS),
        workers_enabled=bool(s.ORCHESTRATOR_WORKERS_ENABLED),
        workers_concurrency=int(s.ORCHESTRATOR_WORKERS_CONCURRENCY),
        cache_version=str(s.ORCHESTRATOR_CACHE_VERSION),
        inline_execution=bool(s.ORCHESTRATOR_INLINE_EXECUTION),
        providers_cache_ttl_seconds=int(s.ORCHESTRATOR_PROVIDERS_CACHE_TTL_SECONDS),
    )
    return BackendSettings(
        app=app,
        database=database,
        redis=redis,
        rate_limit=RateLimitSettings(
            sec_user_agent=sec.user_agent,
            sec_rate_per_sec=sec.rate_per_sec,
            sec_rate_burst=sec.rate_burst,
            sec_facts_ttl_hours=sec.facts_ttl_hours,
        ),
        sec=sec,
        http=HttpSettings(
            connect_timeout=float(s.HTTP_CONNECT_TIMEOUT),
            read_timeout=float(s.HTTP_READ_TIMEOUT),
            max_attempts=int(s.HTTP_MAX_ATTEMPTS),
            backoff_base=float(s.HTTP_BACKOFF_BASE),
            backoff_max=float(s.HTTP_BACKOFF_MAX),
            max_elapsed=float(s.HTTP_MAX_ELAPSED),
            pool_maxsize=int(s.HTTP_POOL_MAXSIZE),
        ),
        yf=YfSettings(
            deadline_seconds=float(s.YF_DEADLINE_SECONDS),
            download_deadline_seconds=float(s.YF_DOWNLOAD_DEADLINE_SECONDS),
            max_deadline_seconds=float(s.YF_MAX_DEADLINE_SECONDS),
            max_concurrency=int(s.YF_MAX_CONCURRENCY),
            queue_wait_seconds=float(s.YF_QUEUE_WAIT_SECONDS),
            breaker_threshold=int(s.YF_BREAKER_THRESHOLD),
            breaker_cooldown_seconds=float(s.YF_BREAKER_COOLDOWN_SECONDS),
        ),
        providers=ProviderSettings(
            finnhub_api_key=s.FINNHUB_API_KEY,
            telegram_bot_token=s.TELEGRAM_BOT_TOKEN,
            telegram_bot_username=s.TELEGRAM_BOT_USERNAME,
            telegram_chat_id=s.TELEGRAM_CHAT_ID,
            telegram_polling=bool(s.TELEGRAM_POLLING),
            telegram_webhook_auto=bool(s.TELEGRAM_WEBHOOK_AUTO),
            telegram_webhook_secret=s.TELEGRAM_WEBHOOK_SECRET,
        ),
        gateway=gateway,
        cors_origins=tuple(s.CORS_ORIGINS),
        frontend_url=s.FRONTEND_URL,
    )


@lru_cache(maxsize=1)
def load() -> BackendSettings:
    """Собрать конфиг один раз за процесс (ленивая инициализация поверх get_settings())."""
    return _build(get_settings())


def reload() -> BackendSettings:
    """Сбросить кэш и перечитать (используется тестами и админ-переключением режимов)."""
    load.cache_clear()
    return load()


# ─────────────────────────────────────────────────────────────────────────────
#  Наблюдаемость
# ─────────────────────────────────────────────────────────────────────────────

def effective_flags() -> dict[str, object]:
    """Действующие флаги для стартового лога и /health.

    Требование аудита (F-01): состояние шлюза и ролей должно быть видно явно, а не выводиться
    из наблюдаемого поведения. Одна строка лога в startup закрывает вопрос.
    """
    cfg = load()
    return {
        "env": cfg.app.env,
        "role": cfg.app.role,
        "debug": cfg.app.debug,
        "testing": cfg.app.testing,
        "worker_services": cfg.app.worker_enabled,
        "api_serving": cfg.app.api_enabled,
        "db": "sqlite" if cfg.database.is_sqlite else "postgres",
        "db_fallback_allowed": cfg.database.allow_fallback,
        "redis": f"{cfg.redis.host}:{cfg.redis.port}/{cfg.redis.db}",
        "gateway_enabled": cfg.gateway.enabled,
        "gateway_inline_execution": cfg.gateway.inline_execution,
        "gateway_workers_enabled": cfg.gateway.workers_enabled,
        "gateway_workers_concurrency": cfg.gateway.workers_concurrency,
        "sec_rate_per_sec": cfg.sec.rate_per_sec,
        # Политика транспорта видна в логе: «сколько ждём и сколько повторяем» — единственное
        # число на весь процесс, а не шесть литералов timeout= в тридцати местах (аудит 05: F-02).
        "http": f"connect={cfg.http.connect_timeout}s read={cfg.http.read_timeout}s "
                f"attempts={cfg.http.max_attempts} pool={cfg.http.pool_maxsize}",
        "yf_deadline": f"{cfg.yf.deadline_seconds}s (download {cfg.yf.download_deadline_seconds}s)",
    }


def is_production() -> bool:
    return load().app.production


def is_worker_role() -> bool:
    return load().app.worker_enabled
