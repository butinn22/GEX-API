"""Token Bucket rate limiter per external provider.

Предотвращает таймауты и бан от провайдеров данных (yfinance, Bybit, MOEX ISS)
за счёт ограничения частоты запросов на провайдера.

Пример::

    limiter = RateLimiter()
    limiter.wait("yfinance")   # блокируется до доступа к yfinance
    fetch_yfinance_data()
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Конфигурация лимитов по умолчанию: запросов в секунду
_DEFAULT_LIMITS: dict[str, dict] = {
    "yfinance": {
        "rate": 4.0,       # 4 запроса/сек — yfinance банит >5/сек
        "burst": 8,
        "description": "Yahoo Finance (опционы + OHLCV)",
    },
    "bybit": {
        "rate": 8.0,       # 8 запросов/сек — Bybit V5 публичный API
        "burst": 16,
        "description": "Bybit V5 (крипто-опционы + klines)",
    },
    "moex_iss": {
        "rate": 2.0,       # 2 запроса/сек — MOEX ISS строго лимитирует
        "burst": 4,
        "description": "MOEX ISS (опционы + свечи FORTS)",
    },
    "cboe": {
        "rate": 3.0,       # 3 запроса/сек — CBOE CDN
        "burst": 6,
        "description": "CBOE CDN (COR1M история)",
    },
    "telegram": {
        "rate": 20.0,      # 20 сообщений/сек — Telegram Bot API
        "burst": 30,
        "description": "Telegram Bot API (нотификации)",
    },
    "finnhub": {
        "rate": 1.0,       # 1 запрос/сек — бесплатный тариф Finnhub 60 req/min
        "burst": 5,
        "description": "Finnhub (company profile2, shares outstanding)",
    },
}


def _build_limits() -> dict[str, dict]:
    """Дефолтные лимиты + bucket 'sec' из настроек (SEC EDGAR).

    SEC разрешает max 10 req/s; держим 5 req/s (burst 10) — запас для
    параллельных запросов и вежливости к data.sec.gov.
    """
    limits: dict[str, dict] = {k: dict(v) for k, v in _DEFAULT_LIMITS.items()}
    try:
        from gex.auth.config import settings

        limits["sec"] = {
            "rate": settings.SEC_RATE_PER_SEC,
            "burst": settings.SEC_RATE_BURST,
            "description": "SEC EDGAR (company fundamentals)",
        }
    except Exception:  # pragma: no cover — конфиг всегда доступен
        limits.setdefault("sec", {"rate": 5.0, "burst": 10, "description": "SEC EDGAR"})
    return limits


class TokenBucket:
    """Token Bucket для одного провайдера.

    Потокобезопасен (threading.Lock).
    """

    def __init__(self, rate: float, burst: int, name: str = ""):
        if rate <= 0:
            raise ValueError(f"rate должен быть > 0, получено {rate}")
        if burst < 1:
            raise ValueError(f"burst должен быть >= 1, получено {burst}")
        self.rate = float(rate)          # токенов в секунду
        self.burst = int(burst)          # максимальный запас
        self.name = name
        self._tokens = float(burst)      # текущий запас
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Пополнить токены по прошедшему времени."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0, blocking: bool = True) -> bool:
        """Забрать ``tokens`` из ведра.

        Parameters
        ----------
        tokens : float
            Сколько токенов забрать (1 = один запрос).
        blocking : bool
            Если True — блокируется до появления токенов.
            Если False — возвращает False при нехватке.

        Returns
        -------
        bool
            True если токены получены.
        """
        if tokens <= 0:
            return True

        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            if not blocking:
                return False

            # Сколько ждать до появления нужного числа токенов
            deficit = tokens - self._tokens
            wait_time = deficit / self.rate

        # Ждём вне блокировки
        time.sleep(max(wait_time, 0.01))  # минимум 10ms

        # Повторная попытка (один раз — после sleep токены должны появиться)
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            # Даже с небольшим минусом — забираем (перерасход вернётся пополнением)
            deficit_after = tokens - self._tokens
            self._tokens = 0.0
            if deficit_after <= tokens * 0.2:  # <20% погрешность
                return True
            return False

    @property
    def available(self) -> float:
        """Текущее количество доступных токенов."""
        with self._lock:
            self._refill()
            return self._tokens

    def reset(self) -> None:
        """Сбросить ведро до полного."""
        with self._lock:
            self._tokens = float(self.burst)
            self._last_refill = time.monotonic()


class RateLimiter:
    """Глобальный rate limiter по провайдерам.

    Потокобезопасен. Вёдра разделяются между **потоками одного процесса**; между
    воркерами — нет (лимит умножается на число воркеров). Единый авторитет для всех
    воркеров даёт :class:`~gex.adapters.ratelimit.redis_lua.RedisRateLimiter`, а
    :func:`get_rate_limiter` выбирает его, когда Redis доступен.
    """

    def __init__(self, limits: dict[str, dict] | None = None):
        self._buckets: dict[str, TokenBucket] = {}
        cfg = limits or _build_limits()
        for name, params in cfg.items():
            self._buckets[name] = TokenBucket(
                rate=params["rate"],
                burst=params["burst"],
                name=name,
            )

    def get_bucket(self, provider: str) -> TokenBucket | None:
        """Получить ведро для провайдера (или None если неизвестен)."""
        return self._buckets.get(provider)

    def acquire(self, provider: str, tokens: float = 1.0, blocking: bool = True) -> bool:
        """Забрать токены у провайдера.

        Parameters
        ----------
        provider : str
            Ключ провайдера (yfinance, bybit, moex_iss, ...).
        tokens : float
            Сколько токенов.
        blocking : bool
            True = ждать, False = не ждать.

        Returns
        -------
        bool
        """
        bucket = self._buckets.get(provider)
        if bucket is None:
            return True  # неизвестный провайдер — не ограничиваем
        return bucket.acquire(tokens, blocking=blocking)

    def wait(self, provider: str) -> None:
        """Блокирующий вызов: ждать доступа к провайдеру.

        Используется перед каждым HTTP-запросом к внешнему API.
        """
        self.acquire(provider, tokens=1.0, blocking=True)

    def reset_all(self) -> None:
        """Сбросить все вёдра (для тестов)."""
        for bucket in self._buckets.values():
            bucket.reset()

    @property
    def providers(self) -> list[str]:
        return list(self._buckets.keys())

    def __repr__(self) -> str:
        infos = []
        for name, bucket in sorted(self._buckets.items()):
            infos.append(f"{name}: {bucket.available:.1f}/{bucket.burst}")
        return f"<RateLimiter {' | '.join(infos)}>"


class IpRateLimiter:
    """Per-IP лимитер для критичных эндпоинтов (login/register).

    Два режима:

    * **Redis доступен** → ведро в Redis (``adapters/ratelimit/redis_lua.py``), общее
      для всех воркеров. Это важно для защиты от перебора пароля: при per-process
      ведре порог умножается на число воркеров (2 воркера → вдвое больше попыток
      на один IP).
    * **Redis недоступен** → локальные вёдра с автоочисткой (деградация подтверждается
      счётчиком ``degraded_calls`` и предупреждением в логе, а не молчанием).

    ``scope`` разделяет независимые лимиты: ручка логина и ручка проверки Telegram-кода
    не должны делить одно ведро на IP.
    """

    def __init__(
        self,
        rate: float,
        burst: int,
        ttl_seconds: float = 3600.0,
        scope: str = "default",
    ):
        self.rate = float(rate)
        self.burst = int(burst)
        self.ttl = float(ttl_seconds)
        self.scope = scope
        self._buckets: dict[str, tuple[float, TokenBucket]] = {}
        self._lock = threading.Lock()
        self._remote = None
        self.degraded_calls = 0
        self._warned = False

    def allow(self, ip: str) -> bool:
        """True если запрос разрешён (non-blocking)."""
        remote = self._remote_backend()
        if remote is not None:
            return remote.allow(ip)
        return self._allow_local(ip)

    def _remote_backend(self):
        """Redis-бэкенд, если он доступен; иначе ``None`` (и один раз пишем об этом).

        Импорт ``get_redis`` ленивый: модуль лимитера обязан импортироваться без pandas,
        иначе его нельзя проверить в наборе без внешних зависимостей.
        """
        if self._remote is not None:
            return self._remote
        try:
            from gex.adapters.cache.redis_client import get_redis

            redis = get_redis()
            if redis is None or not redis.connected:
                return None
            from gex.adapters.ratelimit.redis_lua import RedisIpLimiter

            self._remote = RedisIpLimiter(redis, self.scope, self.rate, self.burst)
            logger.info(
                "IP-лимитер %s: авторитет Redis (лимит общий для всех воркеров)", self.scope
            )
            return self._remote
        except ImportError:
            return None

    def _allow_local(self, ip: str) -> bool:
        self.degraded_calls += 1
        if not self._warned:
            self._warned = True
            logger.warning(
                "IP-лимитер %s: Redis недоступен — лимит per-process, "
                "порог перебора умножается на число воркеров", self.scope,
            )
        now = time.monotonic()
        with self._lock:
            if len(self._buckets) > 10000:
                for key in [k for k, (ts, _) in self._buckets.items() if now - ts > self.ttl]:
                    self._buckets.pop(key, None)
            entry = self._buckets.get(ip)
            if entry is None:
                # Новый IP: создаём ведро и СРАЗУ тратим первый токен
                # (иначе первый запрос был бы «бесплатным» сверх burst).
                bucket = TokenBucket(self.rate, self.burst)
                ok = bucket.acquire(1.0, blocking=False)
                self._buckets[ip] = (now, bucket)
                return ok
            ts, bucket = entry
            self._buckets[ip] = (now, bucket)
            return bucket.acquire(1.0, blocking=False)


class RateLimiterAuthority:
    """Единая точка доступа к лимитам: выбирает авторитет и умеет его повысить.

    Зачем фасад, а не просто «взять Redis-лимитер»

    * Worker может стартовать **раньше**, чем Redis станет доступен (или Redis
      перезапускают). Если решение принять один раз при первом запросе, процесс
      навсегда останется с per-process лимитом — то есть с исходным дефектом.
      Фасад периодически перепроверяет доступность и повышает авторитет.
    * Ссылку на лимитер сервисы берут в ``__init__`` (``get_rate_limiter()``), поэтому
      объект обязан оставаться одним и тем же.
    """

    #: Как часто перепроверять доступность Redis, если работаем на локальных вёдрах.
    RECHECK_INTERVAL_S = 30.0

    def __init__(
        self,
        limits: dict[str, dict] | None = None,
        *,
        redis_getter=None,
        clock=time.monotonic,
    ) -> None:
        self._limits = limits if limits is not None else _build_limits()
        self._local = RateLimiter(self._limits)
        self._remote = None
        self._redis_getter = redis_getter
        self._clock = clock
        self._next_recheck = 0.0
        self._upgraded = False

    @property
    def authority(self) -> str:
        """Текущий авторитет лимитов: ``redis`` или ``local`` (для диагностики/метрик)."""
        return "redis" if self._remote is not None else "local"

    @property
    def providers(self) -> list[str]:
        return list(self._limits.keys())

    def acquire(self, provider: str, tokens: float = 1.0, blocking: bool = True) -> bool:
        remote = self._current()
        if remote is not None:
            return remote.acquire(provider, tokens=tokens, blocking=blocking)
        return self._local.acquire(provider, tokens=tokens, blocking=blocking)

    def wait(self, provider: str) -> None:
        self.acquire(provider, tokens=1.0, blocking=True)

    def reset_all(self) -> None:
        self._local.reset_all()
        if self._remote is not None:
            self._remote.reset_all()

    def _current(self):
        """Redis-авторитет, если доступен; иначе ``None`` (локальные вёдра)."""
        if self._remote is not None:
            return self._remote
        now = self._clock()
        if now < self._next_recheck:
            return None
        self._next_recheck = now + self.RECHECK_INTERVAL_S
        try:
            getter = self._redis_getter
            if getter is None:
                from gex.adapters.cache.redis_client import get_redis

                getter = get_redis
            redis = getter()
            if redis is None or not redis.connected:
                return None
            from gex.adapters.ratelimit.redis_lua import RedisRateLimiter, make_limits

            self._remote = RedisRateLimiter(
                redis, make_limits(self._limits), degraded_limiter=self._local
            )
            if not self._upgraded:
                self._upgraded = True
                logger.info(
                    "Лимиты: авторитет Redis-Lua (единый для всех воркеров). "
                    "Доступные провайдеры: %s", sorted(self._limits),
                )
            return self._remote
        except ImportError:
            return None


# Глобальный инстанс (для обратной совместимости)
_default_limiter: RateLimiterAuthority | None = None


def get_rate_limiter() -> RateLimiterAuthority:
    """Получить/создать глобальный лимитер (единый авторитет: Redis-Lua, иначе per-process)."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiterAuthority()
    return _default_limiter


def reset_default_limiter() -> None:
    """Сбросить глобальный лимитер (тесты)."""
    global _default_limiter
    _default_limiter = None
