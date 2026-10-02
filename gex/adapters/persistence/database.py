"""Database engine, session factory, and declarative base.

Поддерживает SQLite (dev/test) и PostgreSQL (prod) с одним и тем же
синхронным SQLAlchemy 2.0 интерфейсом.

Graceful degradation
--------------------
Если PostgreSQL недоступен (сервер не запущен, неверные credentials),
движок автоматически переключается на SQLite (файл ``gex.db``) с
предупреждением в логе.  Это позволяет запускать приложение локально
без поднятого PostgreSQL.

При старте с PostgreSQL движок проверяет соединение:
  - успех → работает с PostgreSQL;
  - неудача → fallback на SQLite + лог警告.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gex.auth.config import settings

logger = logging.getLogger(__name__)

#: Сколько соединение SQLite ждёт снятия чужой блокировки, прежде чем бросить
#: ``database is locked``.
#:
#: У драйвера по умолчанию 5 с — этого не хватает: ``get_session()`` коммитит
#: **в конце** обработчика, поэтому транзакция держит блокировку всё время работы
#: хендлера (сетевой фетч цепочек/котировок — секунды). Пять секунд истекали
#: ровно на середине такого фетча, и фоновые писатели (коллектор метрик, трекер
#: визитов, воркеры очереди) падали с «database is locked».
SQLITE_BUSY_TIMEOUT_MS = 30_000


# ================================================================= #
#  Declarative Base
# ================================================================= #
class Base(DeclarativeBase):
    pass


# ================================================================= #
#  Engine — отложенное создание + fallback
# ================================================================= #

def _build_engine_url() -> str:
    """Вернуть DATABASE_URL с корректным драйвером для синхронного SQLAlchemy.

    * ``postgresql+asyncpg://`` → ``postgresql+psycopg2://``
    * ``postgresql://``         → ``postgresql+psycopg2://``
    * ``sqlite:///...``         → без изменений
    """
    raw = settings.DATABASE_URL

    # Для тестов — in-memory sqlite
    if raw.startswith("sqlite") and settings.TESTING:
        return "sqlite:///:memory:"

    # Нормализуем PostgreSQL URL для синхронного драйвера
    if raw.startswith("postgresql+asyncpg://"):
        return raw.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg2://")

    return raw


def _engine_kwargs() -> dict:
    """Аргументы для create_engine в зависимости от типа БД."""
    kwargs: dict = {
        "pool_pre_ping": True,
        "echo": settings.DEBUG_SQL,
    }
    url = _build_engine_url()

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {
            "check_same_thread": False,
            # Параметр драйвера: дублирует PRAGMA busy_timeout на случай, если
            # PRAGMA не применится (например, БД на сетевом диске).
            "timeout": SQLITE_BUSY_TIMEOUT_MS / 1000,
        }
        # in-memory SQLite: одно общее соединение, иначе каждый поток/подключение
        # видит свою пустую БД (SingletonThreadPool по умолчанию)
        if ":memory:" in url or url == "sqlite://":
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_size"] = settings.DB_POOL_SIZE
        kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW

    return kwargs


def _is_postgres_url(url: str) -> bool:
    """True если URL указывает на PostgreSQL."""
    return "postgresql" in url


# ================================================================= #
#  PRAGMA для SQLite (WAL и прочее)
# ================================================================= #
def _is_file_sqlite(engine) -> bool:
    """True для файлового SQLite (in-memory и PostgreSQL — нет)."""
    return engine.dialect.name == "sqlite" and ":memory:" not in str(engine.url)


def _apply_sqlite_pragmas(dbapi_conn, _record) -> None:
    """Выставить PRAGMA на каждом новом соединении SQLite.

    * ``journal_mode=WAL`` — главное исправление. В режиме по умолчанию
      (rollback journal) **любой читатель держит SHARED-блокировку**, а писатель
      ждёт её снятия: один долгий ``SELECT`` (``get_db_stats`` считает строки по
      всем таблицам, ``get_metric_history`` — по снапшотам) блокировал каждую
      запись. В WAL читатели и писатель не мешают друг другу.
    * ``busy_timeout`` — ждать блокировку, а не падать сразу. В WAL остаётся
      только конкуренция писателей, и её это почти снимает.
    * ``synchronous=NORMAL`` — в WAL безопасно (потерять можно только последнюю
      транзакцию при аварии питания) и заметно ускоряет COMMIT.

    PRAGMA лучшее из возможного: на сетевых дисках (WAL требует shm-файл) и в
    read-only БД оно не применяется — тогда работаем как раньше, с предупреждением.
    """
    try:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute(f"PRAGMA busy_timeout={int(SQLITE_BUSY_TIMEOUT_MS)}")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()
    except Exception as exc:  # noqa: BLE001 — PRAGMA не должны ронять старт
        logger.warning("SQLite PRAGMA не применены: %s", exc)


def _attach_sqlite_pragmas(engine) -> None:
    """Подключить PRAGMA к движку (только файловый SQLite)."""
    if _is_file_sqlite(engine):
        event.listen(engine, "connect", _apply_sqlite_pragmas)


def _enable_wal_now(engine) -> None:
    """Перевести файл БД в WAL сразу на старте и записать режим в лог.

    ``journal_mode`` — постоянный PRAGMA (хранится в файле), так что одного
    применения хватает; здесь он нужен, чтобы режим был виден в логе, а не
    угадывался по отсутствию ошибок.
    """
    if not _is_file_sqlite(engine):
        return
    try:
        with engine.connect() as conn:
            mode = conn.execute(text("PRAGMA journal_mode=WAL")).scalar()
        logger.info(
            "SQLite: journal_mode=%s, busy_timeout=%dms", mode, int(SQLITE_BUSY_TIMEOUT_MS),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQLite: не удалось включить WAL (%s) — работаем как есть", exc)


def _check_connection(url: str, kwargs: dict) -> bool:
    """Проверить, отвечает ли БД по указанному URL."""
    try:
        test_engine = create_engine(url, **{k: v for k, v in kwargs.items() if k != "echo"})
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        test_engine.dispose()
        return True
    except OperationalError as e:
        logger.warning("Database connection check FAILED for %s: %s", url, e)
        return False
    except Exception as e:
        logger.warning("Database connection check error for %s: %s", url, e)
        return False


# ── engine / SessionLocal — создаются один раз при импорте ─────────────── #
# Но с возможностью fallback через ensure_db()

_actual_url: str | None = None       # какой URL реально используется
_fallback_active: bool = False       # True если активен fallback на SQLite

engine = create_engine(_build_engine_url(), **_engine_kwargs())
_attach_sqlite_pragmas(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
_actual_url = _build_engine_url()


# ================================================================= #
#  Public API
# ================================================================= #

def active_url() -> str:
    """Вернуть URL, который реально используется (с учётом fallback)."""
    return _actual_url or "unknown"


def is_fallback() -> bool:
    """True если был активирован fallback (PostgreSQL → SQLite)."""
    return _fallback_active


def _rebind(factory):  # type: ignore[no-untyped-def]
    """Переключить **единственный** объект SessionLocal на новый движок.

    ``SessionLocal = sessionmaker(...)`` создаёт новый объект, но модули, сделавшие
    ``from gex.adapters.persistence.database import SessionLocal`` на import-time
    (``subscription_watcher``, ``signal_scanner_service``, ``sec_fundamentals``),
    продолжают держать **старую** привязку.  Поэтому фабрика не пересоздаётся, а
    перенастраивается на месте — иначе fallback на SQLite выглядит включённым
    (``is_fallback() == True``), а запросы всё равно идут в недоступный PostgreSQL.
    """
    SessionLocal.configure(bind=factory.kw["bind"])


def ensure_db() -> None:
    """Проверить соединение с БД. Для PostgreSQL — fallback на SQLite при неудаче.

    Вызывается при старте приложения перед ``init_db()``.
    """
    global engine, _actual_url, _fallback_active  # noqa: PLW0603

    url = _build_engine_url()
    kwargs = _engine_kwargs()

    # SQLite — нечего проверять, оно всегда доступно
    if url.startswith("sqlite"):
        _fallback_active = False
        _actual_url = url
        return

    # PostgreSQL — проверяем соединение
    if _is_postgres_url(url):
        if _check_connection(url, kwargs):
            logger.info("PostgreSQL connection OK (%s)", url)
            _fallback_active = False
            _actual_url = url
            return

        # Fail-fast в production: тихий fallback рассинхронизирует данные
        if not settings.ALLOW_DB_FALLBACK:
            raise RuntimeError(
                f"PostgreSQL недоступен ({url}), а ALLOW_DB_FALLBACK=False. "
                "Запустите PostgreSQL или включите fallback только для dev."
            )

        # Fallback на SQLite (только dev/test)
        fallback_url = "sqlite:///./gex.db"
        logger.warning(
            "PostgreSQL unavailable (%s). Falling back to SQLite (%s). "
            "Start PostgreSQL to use the configured database.",
            url, fallback_url,
        )
        fallback_kwargs: dict = {
            "pool_pre_ping": True,
            "echo": settings.DEBUG_SQL,
            "connect_args": {
                "check_same_thread": False,
                "timeout": SQLITE_BUSY_TIMEOUT_MS / 1000,
            },
        }
        engine = create_engine(fallback_url, **fallback_kwargs)
        _attach_sqlite_pragmas(engine)
        _rebind(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        _actual_url = fallback_url
        _fallback_active = True


def check_db() -> dict:
    """Диагностика: проверить соединение и вернуть статус."""
    status = {
        "configured_url": settings.DATABASE_URL,
        "active_url": _actual_url,
        "fallback_active": _fallback_active,
        "db_type": db_type(),
    }
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["connected"] = True
    except Exception as e:
        status["connected"] = False
        status["error"] = str(e)
    return status


def get_session() -> Iterator[Session]:
    """FastAPI dependency: получить сессию БД (yield-генератор).

    Коммитит при успехе, откатывает при ошибке — изменения не теряются,
    даже если обработчик забыл вызвать ``db.commit()``.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ================================================================= #
#  Инициализация / миграции
# ================================================================= #
def init_db() -> None:
    """Создать таблицы через ``Base.metadata.create_all``.

    В production используйте Alembic (``alembic upgrade head``).
    """
    from gex.auth import models  # noqa: F401 - регистрирует модели в Base.metadata
    from gex.auth import user_instrument  # noqa: F401 - UserInstrument
    from gex.auth import user_settings  # noqa: F401 - UserDashboardSettings
    from gex.auth import user_scanner_settings  # noqa: F401 - UserScannerSettings
    from gex.auth import payment_models  # noqa: F401 - Payment
    from gex.adapters.middleware import visits  # noqa: F401 - PageVisit (аналитика посещений)
    from gex.adapters.persistence import sec_models  # noqa: F401 - CompanyRevenue (SEC EDGAR fundamentals)
    from gex.adapters.middleware import system_metrics  # noqa: F401 - SystemMetricSnapshot (история метрик)
    from gex.orchestrator import models as _orchestrator_models  # noqa: F401 - orchestrator config/logs
    from gex.adapters.persistence import gex_state as _gex_state  # noqa: F401 - GexState persistent GEX cache
    _enable_wal_now(engine)
    logger.info("Creating tables via Base.metadata.create_all (URL=%s)", _actual_url)
    Base.metadata.create_all(bind=engine)


def drop_all() -> None:
    """Удалить все таблицы (для тестов / пересоздания схемы).

    Модели регистрируются явно, чтобы drop учитывал FK-зависимости
    (user_instruments/payments ссылаются на users).
    """
    from gex.auth import models, payment_models, user_instrument, user_settings  # noqa: F401 - регистрация в Base.metadata
    from gex.auth import user_scanner_settings  # noqa: F401 - UserScannerSettings
    from gex.adapters.middleware import visits  # noqa: F401 - PageVisit
    from gex.adapters.persistence import sec_models  # noqa: F401 - CompanyRevenue
    from gex.adapters.middleware import system_metrics  # noqa: F401 - SystemMetricSnapshot
    from gex.orchestrator import models as _orchestrator_models  # noqa: F401 - orchestrator config/logs
    from gex.adapters.persistence import gex_state as _gex_state  # noqa: F401 - GexState persistent GEX cache
    Base.metadata.drop_all(bind=engine)


def recreate_tables() -> None:
    """Пересоздать таблицы (drop + create)."""
    drop_all()
    init_db()


def db_type() -> str:
    """Вернуть настроенный тип БД (из DATABASE_URL): 'sqlite' | 'postgresql'.

    NOTE: если был активирован fallback (PostgreSQL → SQLite), эта функция
    всё равно вернёт 'postgresql', чтобы отразить намерение пользователя.
    Для фактического URL используйте ``active_url()``.

    ВНИМАНИЕ: для выбора диалект-специфичного SQL эта функция **не годится** —
    после fallback она отвечает «postgresql», а запрос уйдёт в SQLite. Для этого
    есть ``active_dialect()``.
    """
    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        return "sqlite"
    if "postgresql" in url:
        return "postgresql"
    return "unknown"


def active_dialect() -> str:
    """Диалект БД, в которую **фактически** уходят запросы: 'sqlite' | 'postgresql'.

    Отличие от ``db_type()`` принципиально, когда сработал fallback: ``db_type()``
    продолжает отвечать «postgresql» (намерение), а запросы исполняет SQLite.
    Диалект-специфичный SQL (``func.timezone(...)`` есть только в PostgreSQL)
    обязан выбираться по этой функции, иначе административные страницы падают
    с ``OperationalError: no such function: timezone`` — ровно на том окружении
    (локальный запуск без PostgreSQL), для которого fallback и написан.
    """
    return "postgresql" if "postgresql" in active_url() else "sqlite"
