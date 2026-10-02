"""Application configuration via pydantic-settings.

Читает параметры из env-файла или переменных окружения.
Поддерживает SQLite (dev/test) и PostgreSQL (prod) с настройками пула.
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import timedelta
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Путь к .env: два уровня вверх от gex/auth/config.py → корень проекта
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DOTENV_PATH = os.path.join(_PROJECT_ROOT, ".env")


class Settings(BaseSettings):
    """Единая конфигурация приложения."""

    model_config = SettingsConfigDict(
        env_file=_DOTENV_PATH if os.path.isfile(_DOTENV_PATH) else ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("REDIS_PASSWORD", mode="before")
    @classmethod
    def _empty_redis_password_is_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "GEX Analytics API"
    DEBUG: bool = False
    DEBUG_SQL: bool = False
    TESTING: bool = False
    VERSION: str = "0.4.0"
    APP_ENV: str = "development"   # development | production (валидация секретов)
    # Роль процесса в деплое: all (dev, всё в одном) | web (только API,
    # без фоновых сервисов) | worker (фоновые сервисы: scheduler, очереди,
    # сканеры). Миграции запускаются отдельной командой alembic upgrade head.
    ROLE: str = "all"

    # ── Воркеры очереди задач ─────────────────────────────────────────
    # Что запускает worker-процесс: core (scheduler, сканеры, telegram) и/или
    # consumers (потребители очереди задач). Чистые consumer-процессы:
    # WORKER_COMPONENTS=consumers (+ QUEUE_CONSUMERS=fast|heavy) — так устроены
    # сервисы worker-fast/worker-heavy в docker-compose.prod.yml (профиль split).
    WORKER_COMPONENTS: str = "core,consumers"
    # Какие профили очередей потребляет процесс: all | fast | heavy | fast,heavy.
    # fast — свечи market overview; heavy — цепочки/расчёты, где ожидание не
    # критично (см. gex.application.jobs.CONSUMER_PROFILES).
    QUEUE_CONSUMERS: str = "all"
    # Динамическое масштабирование потребителей по глубине очереди: число
    # воркеров растёт при backlog выше порога и уменьшается при полном простое,
    # в диапазоне [QUEUE_WORKERS_MIN, QUEUE_WORKERS_MAX] на процесс.
    QUEUE_AUTOSCALE: bool = True
    QUEUE_WORKERS_MIN: int = 1
    QUEUE_WORKERS_MAX: int = 4

    # ── GeoIP / мультиязычность ────────────────────────────────────────
    GEOIP_ENABLED: bool = True   # false — отключить определение страны по IP
    GEOIP_DEFAULT_LANG: str = "ru"  # язык по умолчанию (не ломает текущий RU-вариант)

    # ── Database ─────────────────────────────────────────────────────
    #  sqlite:///./gex.db           — SQLite (dev/test, default)
    #  postgresql://user:pass@host:5432/dbname        — PostgreSQL (sync, psycopg2)
    #  postgresql+asyncpg://user:pass@host:5432/dbname — async (для будущего)
    DATABASE_URL: str = "sqlite:///./gex.db"
    ALLOW_DB_FALLBACK: bool = True  # False в production: fail-fast вместо тихого перехода на SQLite

    # Пул соединений (PostgreSQL)
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ── JWT ──────────────────────────────────────────────────────────
    JWT_SECRET: str = ""  # обязателен в .env для production; в dev генерируется случайно
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    @property
    def ACCESS_TOKEN_EXPIRE(self) -> timedelta:
        return timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

    @property
    def REFRESH_TOKEN_EXPIRE(self) -> timedelta:
        return timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)

    # ── CORS ─────────────────────────────────────────────────────────
    FRONTEND_URL: str = "http://localhost:8000"
    CORS_ORIGINS: list[str] = ["*"]

    # ── Reverse proxy (nginx) ────────────────────────────────────────
    # PROXY_HEADERS=true: доверять X-Forwarded-* только от перечисленных
    # FORWARDED_ALLOW_IPS (см. ProxyHeadersMiddleware). Обязательно для
    # деплоя за nginx: иначе request.client.host = 127.0.0.1 для ВСЕХ
    # клиентов и per-IP rate-limit (login/register) схлопывается в одно
    # общее ведро на весь сервис (DoS логинов + нет защиты от брутфорса).
    PROXY_HEADERS: bool = False
    FORWARDED_ALLOW_IPS: str = "127.0.0.1"

    # ── Master Admin ─────────────────────────────────────────────────
    MASTER_EMAIL: str = "sadisting"
    MASTER_PASSWORD: str = "password"

    # ── Telegram webhook: секрет подлинности update (X-Telegram-Bot-Api-Secret-Token)
    TELEGRAM_WEBHOOK_SECRET: str = ""

    # ── Demo-доступ (кнопка «Просмотр демо» на /register) ─────────────
    # Служебный аккаунт с максимально ограниченными правами: только
    # просмотр данных тикера DEMO_TICKER (ES) в разрешённых разделах.
    # Вход — POST /auth/demo (без пароля); НЕ использовать для реальных
    # пользователей: все data-ручки вне allowlist ему закрыты middleware.
    DEMO_EMAIL: str = "demo@gex-analytics.com"

    # ── Email (SMTP stub) ────────────────────────────────────────────
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 1025
    SMTP_USER: str = ""
    SMTP_PASS: str = ""
    FROM_EMAIL: str = "noreply@gex-analytics.com"

    # ── Redis cache ─────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    REDIS_TTL_DEFAULT: int = 600
    REDIS_MAXMEMORY: str = "512mb"
    #: Политика вытеснения, которую приложение выставляет на подключённом инстансе.
    #: Для инстанса, который ТАКЖЕ обслуживает очередь задач (брокер + результаты),
    #: ``allkeys-lru`` опасен: под давлением памяти Redis может вытеснить ключи очереди
    #: и результатов (см. gex/workers/config.py). Если очередь задач делит инстанс с
    #: кэшем — ставить ``noeviction``: кэш и так несёт явные TTL и деградирует штатно.
    REDIS_MAXMEMORY_POLICY: str = "allkeys-lru"

    # ── Перезапуск Redis из админки ─────────────────────────────────
    # Кнопка «Перезапустить Redis» в админке должна перезапускать сервис НА СЕРВЕРЕ,
    # а не в браузере. Команда задаётся здесь (не приходит от клиента) и выполняется
    # без shell — передать произвольную строку снаружи нельзя. Пустая строка =
    # «перезапуск не настроен», ручка отвечает 501.
    REDIS_RESTART_COMMAND: str = "docker compose restart redis"
    #: Рабочий каталог команды (пусто — каталог запуска приложения).
    REDIS_RESTART_CWD: str = ""
    #: Сколько ждать завершения команды, прежде чем считать перезапуск подвисшим.
    REDIS_RESTART_TIMEOUT_SECONDS: float = 60.0

    # ── Data orchestrator (central external API gateway) ────────────────
    ORCHESTRATOR_ENABLED: bool = False
    ORCHESTRATOR_REDIS_URL: str = ""
    ORCHESTRATOR_DEFAULT_INTERACTIVE_TIMEOUT_MS: int = 3000
    ORCHESTRATOR_DEFAULT_BACKGROUND_TIMEOUT_MS: int = 30000
    ORCHESTRATOR_WORKERS_ENABLED: bool = False
    ORCHESTRATOR_WORKERS_CONCURRENCY: int = 4
    ORCHESTRATOR_CACHE_VERSION: str = "v1"
    # True in dev/test: cache misses are executed inline by the current process.
    # Production should set this to false and run dedicated workers.
    ORCHESTRATOR_INLINE_EXECUTION: bool = True
    ORCHESTRATOR_PROVIDERS_CACHE_TTL_SECONDS: int = 30

    # ── HTTP transport (единый для всех провайдеров, аудит 05: F-02) ─────
    # Раньше каждый вызов сам выбирал таймаут: в коде зафиксированы 3/10/15/20/25/200 с.
    # Теперь значения одни на процесс и читаются только gex/adapters/transport/http.py.
    HTTP_CONNECT_TIMEOUT: float = 5.0
    HTTP_READ_TIMEOUT: float = 20.0
    HTTP_MAX_ATTEMPTS: int = 3
    HTTP_BACKOFF_BASE: float = 0.5
    HTTP_BACKOFF_MAX: float = 8.0
    # Общий бюджет на все попытки == HTTP_READ_TIMEOUT. Инвариант: повтор никогда не удлиняет
    # запрос сверх одного таймаута — если первый вызов выдержал все 20 с, второго не будет.
    # Зато при быстрых отказах (connection refused, мгновенный 5xx) повторы почти бесплатны.
    HTTP_MAX_ELAPSED: float = 20.0
    HTTP_POOL_MAXSIZE: int = 20

    # ── yfinance deadline (итерация 23: таймаута у библиотеки нет вообще) ────
    # yf.download/Ticker.history не принимают timeout и могут висеть минутами, занимая
    # воркер. Мы не можем прервать вызов, но можем перестать его ждать и отдать странице
    # последний payload. download — массовая операция, для неё дедлайн вдвое больше.
    YF_DEADLINE_SECONDS: float = 30.0
    YF_DOWNLOAD_DEADLINE_SECONDS: float = 60.0
    YF_MAX_DEADLINE_SECONDS: float = 120.0

    # ── yfinance concurrency guard (инцидент 2026-09-21: взрыв тредов) ──────
    # Deadline «перестаём ждать» не убивает поток — брошенные вызовы продолжали
    # жить и копились: процесс дошёл до 6533 тредов, GIL-конкуренция превращала
    # штатный вызов 1.6 с в таймаут 30 с, что рождало ещё больше брошенных тредов.
    # Теперь число одновременных вызовов yfinance жёстко ограничено, а при
    # исчерпании слотов вызов падает сразу (backpressure) вместо накопления.
    YF_MAX_CONCURRENCY: int = 16
    #: Сколько ждать свободный слот; дольше — отказ (не очередь без конца).
    YF_QUEUE_WAIT_SECONDS: float = 2.0
    #: Сколько брошенных вызовов открывает «предохранитель» (апстрим деградировал).
    YF_BREAKER_THRESHOLD: int = 8
    #: Как долго предохранитель отклоняет вызовы, давая backlog стечь.
    YF_BREAKER_COOLDOWN_SECONDS: float = 30.0

    # ── SEC EDGAR (company fundamentals) ────────────────────────────
    # SEC требует валидный User-Agent вида "Имя Контакт@домен" — без него
    # data.sec.gov отдаёт 403. Формат проверяется, реальность email — нет.
    SEC_USER_AGENT: str = "GEXAnalytics contact@gex-analytics.com"
    # Лимит SEC — 10 req/s; держим 5 req/s (burst 10) для запаса и вежливости
    SEC_RATE_PER_SEC: float = 5.0
    SEC_RATE_BURST: int = 10
    # Свежесть данных о фундаментали (часы): в течение этого окна не ходим
    # в EDGAR повторно (PG-строка + SWR-кэш ответа).
    SEC_FACTS_TTL_HOURS: int = 12

    # ── Finnhub (company profile2: shares outstanding fallback) ────────
    # Бесплатный тариф — 60 запросов/мин. Ключ в .env (не коммитить).
    FINNHUB_API_KEY: str = ""

    # ── Telegram-уведомления (пусто = уведомления отключены) ──────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    # Имя бота (@username) — для deep-link активации аккаунтов
    TELEGRAM_BOT_USERNAME: str = "GexAnalyticsBot"
    # Long-polling: бот сам забирает /start (getUpdates) — работает без домена/webhook.
    # Отключается автоматически, если у бота зарегистрирован webhook.
    TELEGRAM_POLLING: bool = True
    # Авторегистрация webhook при старте (прод): true + FRONTEND_URL=https://домен
    TELEGRAM_WEBHOOK_AUTO: bool = False

    
    # ── Payments ───────────────────────────────────────────────
    BASIC_PRICE_USD: float = 10.0
    EXTENDED_PRICE_USD: float = 15.0
    USD_TO_RUB_RATE: float = 85.0

    # SBP (Система Быстрых Платежей) — задаются в .env; дефолт = Т-Банк
    SBP_PHONE: str = "+79178131132"
    SBP_BANK: str = "Т-Банк"
    SBP_NAME: str = ""

    # Crypto — задаются в .env; пусто = метод недоступен
    CRYPTO_USDT_TRC20: str = ""
    CRYPTO_USDT_BEP20: str = ""

    # Банковский счёт (перевод по реквизитам, ₽) — стартовый сид для
    # payment_settings; дальше реквизиты правятся из админки (БД).
    BANK_NAME: str = ""
    BANK_BIC: str = ""
    BANK_ACCOUNT: str = ""
    BANK_RECIPIENT: str = ""

    # ── OAuth (stub) ─────────────────────────────────────────────────
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""


@lru_cache
def get_settings() -> Settings:
    """Кешированный singleton конфига + валидация секретов/режима."""
    s = Settings()

    # JWT_SECRET: обязателен в production; в dev генерируем случайный
    if not s.JWT_SECRET:
        if s.APP_ENV == "production":
            raise RuntimeError(
                "JWT_SECRET не задан. Укажите надёжный секрет в .env "
                "(production запрещено запускать с пустым секретом)."
            )
        s.JWT_SECRET = secrets.token_urlsafe(48)
        logging.getLogger(__name__).warning(
            "JWT_SECRET не задан в .env — сгенерирован случайный на время dev-сессии."
        )

    if s.APP_ENV == "production":
        problems = []
        if s.MASTER_PASSWORD in ("", "password"):
            problems.append("MASTER_PASSWORD — дефолтное значение, задайте в .env")
        if "@" not in s.MASTER_EMAIL:
            problems.append("MASTER_EMAIL — задайте реальный email администратора (не 'sadisting')")
        if s.CORS_ORIGINS == ["*"]:
            problems.append("CORS_ORIGINS=['*'] — задайте белый список")
        if s.TELEGRAM_WEBHOOK_AUTO and not s.TELEGRAM_WEBHOOK_SECRET:
            problems.append("TELEGRAM_WEBHOOK_SECRET — обязателен при TELEGRAM_WEBHOOK_AUTO=true")
        if s.ROLE not in ("all", "web", "worker"):
            problems.append(f"ROLE={s.ROLE!r} — допустимо: all, web, worker")
        if problems:
            raise RuntimeError("Production configuration error: " + "; ".join(problems))
    else:
        # Не-prod: дефолтные мастер-креды допустимы для локальной разработки,
        # но громко предупреждаем — инстанс может быть доступен по LAN (0.0.0.0).
        if s.MASTER_PASSWORD in ("", "password") or s.MASTER_EMAIL == "sadisting":
            logging.getLogger(__name__).warning(
                "⚠️  MASTER_EMAIL/MASTER_PASSWORD — значения по умолчанию "
                "(sadisting/password). Это приемлемо только для локальной разработки: "
                "задайте свои в .env (MASTER_EMAIL, MASTER_PASSWORD) и APP_ENV=production "
                "перед публикацией сервиса."
            )

    # Диапазон воркеров очереди: инвертированные значения — конфигурационная ошибка,
    # но не повод не подниматься (мягкая нормализация + громкое предупреждение).
    if s.QUEUE_WORKERS_MAX < 1:
        logging.getLogger(__name__).warning("QUEUE_WORKERS_MAX=%s < 1 — поднимаю до 1", s.QUEUE_WORKERS_MAX)
        s.QUEUE_WORKERS_MAX = 1
    if s.QUEUE_WORKERS_MIN < 1:
        logging.getLogger(__name__).warning("QUEUE_WORKERS_MIN=%s < 1 — поднимаю до 1", s.QUEUE_WORKERS_MIN)
        s.QUEUE_WORKERS_MIN = 1
    if s.QUEUE_WORKERS_MAX < s.QUEUE_WORKERS_MIN:
        logging.getLogger(__name__).warning(
            "QUEUE_WORKERS_MAX=%s меньше MIN=%s — меняю местами",
            s.QUEUE_WORKERS_MAX, s.QUEUE_WORKERS_MIN,
        )
        s.QUEUE_WORKERS_MIN, s.QUEUE_WORKERS_MAX = s.QUEUE_WORKERS_MAX, s.QUEUE_WORKERS_MIN

    return s


# Singleton для обратной совместимости (импорт `from .config import settings`)
settings = get_settings()
