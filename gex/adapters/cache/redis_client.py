"""Redis-кэширование для фетчеров GEX Analytics.

Предоставляет:
  * ``RedisClient`` — обёртка над ``redis.Redis`` с пулом соединений;
  * ``@cached(ttl, key_builder)`` — декоратор для методов-фетчеров;
  * ``serialize_*`` / ``deserialize_*`` — компактная сериализация.

**Cache-Aside паттерн:**
  ``get(key)`` → если hit → вернуть, если miss → вызвать оригинал → ``set(key)`` → вернуть.

**Graceful degradation:**
  Любая Redis-ошибка (ConnectionError, TimeoutError, RedisError) ловится —
  ``get()`` возвращает None, ``set()`` молча скипается. Приложение продолжает
  работать без кэша, без 500-х ошибок.

**Политика вытеснения (Eviction):**
  Ожидается, что Redis настроен с ``maxmemory 512mb`` и ``allkeys-lru``.
  Все ключи дополнительно имеют TTL (по умолчанию 600 секунд).

Пример::

    import pandas as pd
    from gex.adapters.cache.redis_client import RedisClient, cached

    cache = RedisClient()

    @cached(ttl=600, key_builder=lambda tf, ticker: f"ohlcv:{ticker}:{tf}")
    def fetch_ohlcv(ticker: str, timeframe: str) -> pd.DataFrame:
        ...  # реальный запрос к yfinance
"""
from __future__ import annotations

import io
import json
import logging
import pickle
import time
import zlib
from functools import wraps
from typing import Any, Optional, TypeVar
from collections.abc import Callable

import pandas as pd

from gex.adapters.cache.keys import PROVIDER_SCOPED_KINDS, CacheKeyError
from gex.adapters.queue.streams_client import StreamsClientMixin

logger = logging.getLogger(__name__)

# ── Попытка импорта redis ─────────────────────────────────────────────
# Если redis не установлен → RedisClient работает в режиме no-op (всегда miss).
_REDIS_AVAILABLE = False
try:
    import redis as _redis
    from redis import RedisError, ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

    _REDIS_AVAILABLE = True
except ImportError:
    _redis = None  # type: ignore[assignment]
    RedisError = Exception  # заглушка для except
    RedisConnectionError = Exception
    RedisTimeoutError = Exception
    logger.warning("redis package not installed — caching disabled")

# ── Константы сериализации ────────────────────────────────────────────
_ZLIB_LEVEL = 4  # быстрый уровень сжатия (3-6 — золотая середина)
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

# ── Типы ──────────────────────────────────────────────────────────────
F = TypeVar("F", bound=Callable[..., Any])


# ====================================================================== #
#  Сериализация / десериализация
# ====================================================================== #
def serialize_df(df: pd.DataFrame) -> bytes:
    """DataFrame → pickle + zlib (компактное хранение OHLC-массивов)."""
    raw = pickle.dumps(df, protocol=_PICKLE_PROTOCOL)
    return zlib.compress(raw, level=_ZLIB_LEVEL)


def deserialize_df(data: bytes) -> pd.DataFrame:
    """zlib + pickle → DataFrame."""
    raw = zlib.decompress(data)
    return pickle.loads(raw)


def serialize_value(obj: Any) -> bytes:
    """Универсальная сериализация: число → str, DataFrame → pickle+zlib, dict → pickle.

    **Идемпотентность для уже сериализованного — не оптимизация, а исправление дефекта.**
    Значения проходят через сериализацию дважды: слой конверта
    (:meth:`gex.adapters.cache.envelope.RedisEnvelopeCache.write`) сам вызывает ``serialize_value``
    для payload'а и передаёт в :meth:`RedisClient.set` уже готовые ``bytes``. Без ветки ``bytes``
    они уходили в ``pickle.dumps``, и в Redis ложилось ``pickle(pickle(payload))``. Читатель
    снимал один слой, получал ``bytes``, :meth:`Envelope.from_payload` возвращал ``None``, и
    **каждый** read становился вечным промахом (спам «Конверт … неизвестной формы — игнорирую»,
    ``None`` у followers → HTTP 500).

    Правила:

    * ``pd.DataFrame`` → :func:`serialize_df` (pickle+zlib);
    * ``int`` / ``float`` → ``str(obj).encode("utf-8")`` (числа дёшевы и читаемы глазами);
    * ``bytes`` / ``bytearray`` → как есть: значение **уже сериализовано** (payload конверта,
      предкодированный блоб) — повторно не заворачиваем;
    * ``str`` → ``obj.encode("utf-8")`` **без pickle**: токены аренд и идентификаторы обязаны
      сравниваться после ``.decode()`` с исходной строкой (см. ``RedisSingleFlight._release``
      и ``RedisLease.release``) — pickled-строка это сравнение сломала бы;
    * всё остальное (dict, list, dataclass, namedtuple, OptionSnapshot) → ``pickle.dumps``.
    """
    if isinstance(obj, pd.DataFrame):
        return serialize_df(obj)
    if isinstance(obj, (int, float)):
        return str(obj).encode("utf-8")
    if isinstance(obj, (bytes, bytearray)):
        # Уже сериализовано (payload конверта, предкодированный блоб) — не заворачиваем повторно.
        return bytes(obj)
    if isinstance(obj, str):
        # Строки храним сырым utf-8: их сравнивают после .decode() (токены аренд, id).
        return obj.encode("utf-8")
    # dict, list, dataclass, namedtuple → pickle
    return pickle.dumps(obj, protocol=_PICKLE_PROTOCOL)


def deserialize_value(data: bytes, as_type: Optional[type] = None) -> Any:
    """Универсальная десериализация.

    Поддерживает:
    * pickle (dict, list, dataclass, OptionSnapshot)
    * zlib + pickle (сжатые DataFrame)
    * plain str (числа, спот-цены)

    Parameters
    ----------
    data : bytes
        Сырые байты из Redis.
    as_type : type, optional
        Если указан — подсказка для десериализации (сейчас не используется,
        автоопределение формата).

    Returns
    -------
    Any
        Восстановленный объект.
    """
    # 1. Пробуем zlib + pickle (сжатые DataFrame)
    if len(data) >= 2 and data[0] == 0x78:  # zlib magic: 0x78
        try:
            raw = zlib.decompress(data)
            return pickle.loads(raw)
        except (zlib.error, pickle.UnpicklingError, TypeError, ValueError):
            pass
    # 2. Попробовать pickle в первую очередь (он ловит большинство типов)
    try:
        return pickle.loads(data)
    except (pickle.UnpicklingError, TypeError, ValueError):
        pass
    # 3. Если не pickle — возможно, обычная строка (число)
    try:
        decoded = data.decode("utf-8")
        # Попробовать float/int
        try:
            return float(decoded) if "." in decoded else int(decoded)
        except (ValueError, TypeError):
            return decoded
    except UnicodeDecodeError:
        pass
    return data


def _apply_eviction(conn: Any, maxmemory: str, policy: str) -> None:
    """Установить maxmemory и политику вытеснения (best-effort).

    Вынесено из класса: это чистая функция от соединения и двух параметров —
    ей не нужно состояние клиента, а класс-клиент от этого не растёт.
    """
    try:
        conn.config_set("maxmemory", maxmemory)
        conn.config_set("maxmemory-policy", policy)
        logger.info("Redis eviction: maxmemory=%s, policy=%s", maxmemory, policy)
    except Exception as exc:
        logger.debug("Cannot set Redis eviction policy (may lack permissions): %s", exc)


# ====================================================================== #
#  Построитель ключей
# ====================================================================== #
#: Минимальный интервал между попытками переподключения к Redis (секунды).
#: ``ping()`` при недоступном Redis инициировал полное переподключение на КАЖДЫЙ вызов,
#: а клиент внутри делает несколько попыток с таймаутом — тривиальный ``/health`` стоил
#: ~4.5 с и занимал поток пула (инцидент 2026-09-21). Теперь попытка не чаще раза в
#: интервал: «Redis недоступен» — быстрый ответ, а не минута ожидания на каждом пинге.
RECONNECT_MIN_INTERVAL = 30.0


def cache_key(prefix: str, *parts: str) -> str:
    """Собрать ключ по схеме ``gex:{prefix}:{part1}:{part2}``.

    **Только для провайдер-независимых ключей** (кэш ответа эндпоинта, состояние,
    служебные счётчики). Для ключей, данные которых приходят от внешнего источника,
    обязателен :mod:`gex.adapters.cache.keys`: провайдер должен входить в ключ, иначе
    два источника делят одну запись (итер. 25). Вызов с видом из
    ``PROVIDER_SCOPED_KINDS`` или с ``commodity:*`` поднимает
    :class:`~gex.adapters.cache.keys.CacheKeyError` — неоднозначный ключ невозможно
    построить даже «на автомате».

    Тикеры (части только из букв) апперкейсятся. Таймфреймы (1h, 2h, 4h, 1d)
    и числовые параметры НЕ апперкейсятся.

    Примеры::

        cache_key("res", "ta", "SPY", 1000)     → "gex:res:TA:SPY:1000"
        cache_key("sigstate3", "u-42")          → "gex:sigstate3:u-42"

    Провайдер-зависимые ключи (``chain`` / ``ohlcv`` / ``spot`` / ``hv``) — только так::

        from gex.adapters.cache.keys import chain_key, PROVIDER_BYBIT
        chain_key("BTC", 5, provider=PROVIDER_BYBIT)   → "gex:chain:bybit:BTC:5"
    """
    if prefix in PROVIDER_SCOPED_KINDS or str(prefix).startswith("commodity:"):
        raise CacheKeyError(
            f"cache_key({prefix!r}, ...) не различает источники данных: используйте "
            f"gex.adapters.cache.keys "
            f"({'commodity_key' if str(prefix).startswith('commodity:') else prefix + '_key'} "
            f"с обязательным provider=...)"
        )

    cleaned: list[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p)
        # Если строка состоит только из букв (тикер) — uppercase
        if s.isalpha():
            s = s.upper()
        cleaned.append(s)
    return f"gex:{prefix}:" + ":".join(cleaned) if cleaned else f"gex:{prefix}"


# ====================================================================== #
#  Redis-клиент
# ====================================================================== #
class RedisClient(StreamsClientMixin):
    """Обёртка над sync redis.Redis с пулом соединений и graceful degradation.

    Parameters
    ----------
    host : str
        Redis host.
    port : int
        Redis port.
    db : int
        Номер БД.
    password : str, optional
        Пароль (если требуется).
    socket_timeout : int
        Таймаут на операции (сек).
    socket_connect_timeout : int
        Таймаут на соединение (сек).
    maxmemory : str
        Значение maxmemory (устанавливается при подключении).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: int = 2,
        socket_connect_timeout: int = 3,
        maxmemory: str = "512mb",
        maxmemory_policy: str = "allkeys-lru",
    ):
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._socket_timeout = socket_timeout
        self._socket_connect_timeout = socket_connect_timeout
        self._maxmemory = maxmemory
        self._maxmemory_policy = maxmemory_policy or "allkeys-lru"

        self._conn: Optional[_redis.Redis] = None
        self._connected = False
        self._pool: Any = None
        #: Когда в последний раз пытались подключиться (монотонные секунды).
        self._last_connect_attempt: float = 0.0

        if _REDIS_AVAILABLE:
            self._connect()
        else:
            logger.info("Redis library not available — running in no-op mode")

    # ------------------------------------------------------------------ #
    #  Соединение
    # ------------------------------------------------------------------ #
    def reconnect(self) -> bool:
        """Переподключиться немедленно, в обход троттлинга.

        Нужно после внешнего перезапуска Redis: обычный путь ждал бы
        ``RECONNECT_MIN_INTERVAL``, и админка показала бы «недоступен» на только что
        поднятом сервере. Возвращает фактическое состояние соединения.
        """
        self._last_connect_attempt = 0.0
        self._connect()
        return bool(self._connected)

    def _reconnect_allowed(self) -> bool:
        """Можно ли сейчас пробовать переподключение (защита от шторма попыток)."""
        return time.monotonic() - self._last_connect_attempt >= RECONNECT_MIN_INTERVAL

    def _connect(self) -> None:
        """Создать пул и подключиться. Попытка — не чаще раза в ``RECONNECT_MIN_INTERVAL``."""
        if not _REDIS_AVAILABLE or _redis is None:
            self._connected = False
            return
        if not self._reconnect_allowed():
            return
        self._last_connect_attempt = time.monotonic()
        try:
            self._pool = _redis.ConnectionPool(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=False,
                max_connections=10,
                protocol=2,
            )
            self._conn = _redis.Redis(connection_pool=self._pool)
            # Проверяем связь и устанавливаем maxmemory (если есть права)
            self._conn.ping()
            self._connected = True
            _apply_eviction(self._conn, self._maxmemory, self._maxmemory_policy)
            logger.info("Redis connected: %s:%s/%s", self._host, self._port, self._db)
        except Exception as exc:
            self._connected = False
            self._conn = None
            self._pool = None
            logger.warning("Redis connection failed: %s — running without cache", exc)

    @property
    def connected(self) -> bool:
        """Проверить, активно ли соединение с Redis."""
        return self._connected

    @property
    def pool(self):
        """Доступ к пулу (для тестов)."""
        return self._pool

    # ------------------------------------------------------------------ #
    #  Основные операции (graceful)
    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Optional[bytes]:
        """Получить значение из кэша.

        Returns
        -------
        bytes or None
            None при miss, ошибке соединения или отсутствии ключа.
        """
        if not self._connected or self._conn is None:
            return None
        try:
            return self._conn.get(key)  # type: ignore[no-any-return]
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            # Соединение умерло посреди работы (Redis молча завис или упал): помечаем клиент
            # отключённым немедленно. Иначе каждый следующий вызов платил бы socket_timeout
            # (2 с) на каждую операцию до следующего ping — деградация кэша обязана быть
            # мгновенной, а не «минус 2 секунды на каждый запрос».
            logger.debug("Redis GET error for key='%s': %s", key, exc)
            self._connected = False
            return None

    def set(
        self,
        key: str,
        value: Any,
        ex: int = 600,
        *,
        px: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        """Сохранить значение с TTL.

        Parameters
        ----------
        key : str
            Ключ.
        value : Any
            Значение (будет сериализовано через :func:`serialize_value`).
        ex : int
            TTL в секундах. Игнорируется, если задан ``px``.
        px : int, optional
            TTL в миллисекундах. Нужен для аренд single-flight (итер. 26): аренда живёт
            секунды, а не минуты.
        nx : bool
            Ставить значение, только если ключа нет. Это **единственный** способ атомарно
            выбрать лидера между процессами: проверка ``get`` + ``set`` не атомарна, и два
            воркера могут оба решить, что они лидеры.

        Returns
        -------
        bool
            True при успехе; False при ошибке или если ``nx=True`` и ключ уже существует.
        """
        if not self._connected or self._conn is None:
            return False
        try:
            data = serialize_value(value)
            if px is not None:
                result = self._conn.set(key, data, px=px, nx=nx)
            else:
                result = self._conn.set(key, data, ex=ex, nx=nx)
            return bool(result)
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis SET error for key='%s': %s", key, exc)
            self._connected = False
            return False

    def delete(self, key: str) -> bool:
        """Удалить ключ."""
        if not self._connected or self._conn is None:
            return False
        try:
            return bool(self._conn.delete(key))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis DEL error for key='%s': %s", key, exc)
            self._connected = False
            return False

    def exists(self, key: str) -> bool:
        """Проверить существование ключа."""
        if not self._connected or self._conn is None:
            return False
        try:
            return bool(self._conn.exists(key))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis EXISTS error for key='%s': %s", key, exc)
            self._connected = False
            return False

    # ------------------------------------------------------------------ #
    #  Lua: атомарные операции (лимитеры)
    # ------------------------------------------------------------------ #
    def script_load(self, script: str) -> Optional[str]:
        """Загрузить Lua-скрипт и получить его SHA1 для последующих ``EVALSHA``.

        Returns
        -------
        Optional[str]
            SHA1-хэш скрипта или ``None`` при ошибке/отсутствии соединения.
        """
        if not self._connected or self._conn is None:
            return None
        try:
            sha = self._conn.script_load(script)
            return sha.decode() if isinstance(sha, bytes) else str(sha)
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis SCRIPT LOAD error: %s", exc)
            return None

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> Optional[list]:
        """Выполнить загруженный скрипт по SHA1.

        ``None`` при ошибке — в том числе ``NOSCRIPT``: вызывающий обязан отличать
        «не выполнилось» от «выполнилось с отрицательным ответом». Для лимитера это
        критично: принять ``None`` за «разрешено» значит молча снять лимит.
        """
        if not self._connected or self._conn is None:
            return None
        try:
            return list(self._conn.evalsha(sha, numkeys, *keys_and_args))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis EVALSHA error: %s", exc)
            return None

    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> Optional[list]:
        """Выполнить Lua-скрипт целиком (без предварительной загрузки)."""
        if not self._connected or self._conn is None:
            return None
        try:
            return list(self._conn.eval(script, numkeys, *keys_and_args))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis EVAL error: %s", exc)
            return None

    def flush(self) -> None:
        """Полная очистка текущей БД (для тестов)."""
        if not self._connected or self._conn is None:
            return
        try:
            self._conn.flushdb()  # type: ignore[no-any-return]  # type: ignore[no-any-return]
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis FLUSHDB error: %s", exc)
            self._connected = False

    def ping(self) -> bool:
        """Проверить связь с Redis (используется в health-check)."""
        if not self._connected or self._conn is None:
            # Ленивое переподключение: Redis мог быть недоступен на старте
            # или был перезапущен позже — пробуем восстановить соединение.
            try:
                self._connect()
            except Exception:
                return False
        if not self._connected or self._conn is None:
            return False
        try:
            return bool(self._conn.ping())  # type: ignore[no-any-return]
        except (RedisError, RedisConnectionError, RedisTimeoutError):
            self._connected = False
            return False

    def close(self) -> None:
        """Закрыть соединение."""
        if not _REDIS_AVAILABLE or _redis is None:
            return
        self._connected = False
        if self._pool is not None:
            try:
                self._pool.disconnect()
            except Exception:  # noqa: E722
                pass
            self._pool = None
            self._conn = None

    def __repr__(self) -> str:
        return f"<RedisClient {self._host}:{self._port}/{self._db} connected={self._connected}>"


# ====================================================================== #
#  Декоратор @cached
# ====================================================================== #
def cached(
    ttl: int = 600,
    key_builder: Optional[Callable[..., str]] = None,
    cache_client: Optional[RedisClient] = None,
    as_type: Optional[type] = None,
) -> Callable[[F], F]:
    """Декоратор Cache-Aside для методов-фетчеров.

    Обёртывает функцию так, что перед вызовом оригинала проверяется Redis.
    При hit → возвращается кэшированное значение (десериализованное).
    При miss → вызывается оригинал, результат сохраняется в Redis и возвращается.

    Parameters
    ----------
    ttl : int
        TTL в секундах (по умолчанию 600 = 10 минут).
    key_builder : callable, optional
        Функция, принимающая те же аргументы, что и обёрнутая функция,
        и возвращающая строку-ключ Redis.
        Если не указана — ключ строится как ``gex:{func_name}:{args}``.
    cache_client : RedisClient, optional
        Инстанс RedisClient. Если не указан — используется глобальный
        ``redis_client`` (см. ниже).
    as_type : type, optional
        Тип для десериализации (если нужно переопределить автоопределение).

    Пример::

        cache = RedisClient()

        @cached(ttl=600, key_builder=lambda tf, t: f"ohlcv:{t}:{tf}")
        def fetch_single(ticker: str, timeframe: str) -> pd.DataFrame:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Определяем RedisClient
            client = cache_client if cache_client is not None else _default_client
            if client is None or not client.connected:
                return func(*args, **kwargs)

            # Строим ключ
            if key_builder is not None:
                key = key_builder(*args, **kwargs)
            else:
                # Авто-ключ: gex:{func.__qualname__}:{args}
                # Апперкейсим только буквенные части (тикер)
                parts = []
                for a in args:
                    if a is not None:
                        s = str(a)
                        if s.isalpha():
                            s = s.upper()
                        parts.append(s)
                key = f"gex:{func.__qualname__}:{':'.join(parts)}" if parts else f"gex:{func.__qualname__}"


            # Проверяем кэш
            cached_data = client.get(key)
            if cached_data is not None:
                try:
                    result = deserialize_value(cached_data, as_type=as_type)
                    logger.debug("Cache HIT for key='%s'", key)
                    return result
                except Exception as exc:
                    logger.debug("Cache deserialize error for key='%s': %s — refetching", key, exc)
                    # Десериализация не удалась → считаем miss

            # Miss: вызываем оригинал
            logger.debug("Cache MISS for key='%s' — fetching from provider", key)
            result = func(*args, **kwargs)

            # Сохраняем в кэш (best-effort)
            if result is not None:
                client.set(key, result, ex=ttl)

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ====================================================================== #
#  Глобальный инстанс (по умолчанию — no-op)
# ====================================================================== #
_default_client: Optional[RedisClient] = None


def init_redis(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    maxmemory: str = "512mb",
    maxmemory_policy: str = "allkeys-lru",
) -> RedisClient:
    """Создать и зарегистрировать глобальный RedisClient.

    Вызывается при старте приложения (в ``main.py``).
    """
    global _default_client
    client = RedisClient(
        host=host,
        port=port,
        db=db,
        password=password,
        maxmemory=maxmemory,
        maxmemory_policy=maxmemory_policy,
    )
    _default_client = client
    return client


def get_redis() -> Optional[RedisClient]:
    """Получить глобальный RedisClient (может быть None, если не инициализирован)."""
    return _default_client
