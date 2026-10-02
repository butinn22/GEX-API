"""Redis-Lua rate limiter — единственный авторитет лимитов между воркерами (ring: adapters).

Зачем переносить лимит в Redis
------------------------------
Текущий :class:`gex.rate_limiter.TokenBucket` держит запас токенов **в памяти процесса**.
Число воркеров больше одного, поэтому фактический лимит = заданный × число воркеров:
для yfinance «4 запроса/сек» при 4 воркерах превращаются в 16 — то есть провайдер,
который банит за >5/сек, получает ровно то, за что банит. Докстрока при этом утверждала,
что вёдра «разделяются между потоками/воркерами» — это было верно только для потоков.

Как устроено
------------
* **Вся арифметика — в одном Lua-скрипте.** Читать-изменять-записывать из Python нельзя:
  между ``HGET`` и ``HSET`` вклинится другой воркер, и оба получат «да». Скрипт в Redis
  исполняется целиком, поэтому гонки нет по построению.
* **Время берёт Redis** (``TIME`` внутри скрипта), а не воркеры: при расхождении часов
  между машинами лимит считался бы от разных «сейчас» и тихо разъезжался.
* **Ждать решает клиент** по ``wait_ms``, который вернул скрипт: сервер не блокируется
  (блокировка Lua на время ожидания остановила бы всех).

Ловушка Lua, из-за которой здесь строки
--------------------------------------
Lua-числа возвращаются в RESP как целые: ``return 3.7`` превратится в ``3``. Поэтому
остаток токенов отдаётся строкой (``tostring``), а ``allowed``/``wait_ms`` — целыми.
Иначе клиент видел бы остаток «3» при фактических 3.7 и расчёт ожидания разъезжался.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Optional, Protocol

from gex.adapters.cache.keys import CacheKeyError, clean_segment, normalize_provider

logger = logging.getLogger(__name__)


class RateLimitBackendError(RuntimeError):
    """Бэкенд лимитера не смог выполнить операцию (нет соединения, NOSCRIPT, мусорный ответ).

    Отдельный тип нужен, чтобы «не выполнено» никогда не смешивалось с «разрешено»:
    молча пропустить запрос при сломанном Redis — это снятие лимита, а не деградация.
    """


class ScriptRedis(Protocol):
    """Минимум Redis'а для Lua-лимитера (совместим с ``RedisClient`` и с fakeredis)."""

    def script_load(self, script: str) -> Optional[str]: ...
    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> Optional[list]: ...
    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> Optional[list]: ...
    def delete(self, *keys: str) -> object: ...


class LimiterLike(Protocol):
    """Лимитер, пригодный как фолбэк при недоступном Redis."""

    def acquire(self, provider: str, tokens: float = 1.0, blocking: bool = True) -> bool: ...

#: Ключ ведра лимита. Провайдер нормализуется тем же словарём, что и ключи кэша
#: (``moex`` → ``moex_iss``): иначе один провайдер получил бы два ведра и двойной лимит.
KEY_PREFIX = "gex:rl:"

#: Минимальный и максимальный TTL ведра (секунды) — простой ведра не должен оставлять
#: «вечный» ключ, но и не должен сбрасывать запас у активного провайдера.
MIN_BUCKET_TTL_S = 60
MAX_BUCKET_TTL_S = 3600

#: Один скрипт на все вёдра: токен-бакет с полным запасом ``burst`` и пополнением ``rate``/сек.
#:
#KEYS[1] = ключ ведра
#ARGV[1] = rate (токенов/сек), ARGV[2] = burst, ARGV[3] = запрошено токенов,
#ARGV[4] = ttl_ms
# Возврат: {allowed(0/1), tokens_left(string), wait_ms(int)}
LUA_TOKEN_BUCKET = """
if redis.replicate_commands then redis.replicate_commands() end
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local want = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])

local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local last_ms = tonumber(data[2])
if tokens == nil then tokens = burst end
if last_ms == nil then last_ms = now_ms end

local elapsed = now_ms - last_ms
if elapsed < 0 then elapsed = 0 end          -- часы Redis ушли назад: не «переполняем» ведро
tokens = math.min(burst, tokens + elapsed * rate / 1000.0)

local allowed = 0
local wait_ms = 0
if tokens >= want then
  tokens = tokens - want
  allowed = 1
else
  local deficit = want - tokens
  wait_ms = math.ceil(deficit * 1000.0 / rate)
  if wait_ms < 1 then wait_ms = 1 end
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now_ms))
redis.call('PEXPIRE', key, ttl_ms)
return {allowed, tostring(tokens), wait_ms}
"""

#: Проверка скрипта в живом Redis: возвращает «сколько доступно сейчас» (для тестов).
#: Отдельный скрипт, потому что ``available`` не должен списывать токены.
LUA_PEEK = """
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
return {data[1] or '', data[2] or ''}
"""


def canonical_provider(name: str) -> str:
    """Имя провайдера для ведра лимита.

    Словарь лимитов и словарь источников данных **пересекаются, но не совпадают**:
    ``yfinance``/``moex_iss``/``sec`` есть в обоих, а ``cboe`` (CDN) и ``telegram``
    (сокет уведомлений) — только в лимитах, потому что они ничего не кладут в кэш.
    Поэтому: если имя известно канонизатору ключей — берём канон (``moex`` → ``moex_iss``),
    иначе считаем его обычным сегментом. Расширять список провайдеров *кэша* именами
    вроде ``telegram`` было бы хуже: он описывает «кто произвёл значение», а не «кого
    мы ограничиваем».
    """
    try:
        return normalize_provider(name)
    except CacheKeyError:
        return clean_segment(name, name="provider")


def rate_limit_key(provider: str) -> str:
    """Ключ ведра провайдера (имя канонизируется: ``moex`` и ``moex_iss`` — одно ведро)."""
    return f"{KEY_PREFIX}{canonical_provider(provider)}"


def ip_rate_limit_key(scope: str, ip: str) -> str:
    """Ключ ведра per-IP лимита. ``scope`` разделяет независимые лимиты на одном IP.

    Область (``"auth"``, ``"sec_forecast"``) — это пространство имён приложения, а не
    провайдер, поэтому проверяется как обычный сегмент ключа.
    """
    safe_ip = str(ip).strip().replace(":", "_")  # IPv6-двоеточия: ':' разделяет сегменты
    return f"{KEY_PREFIX}ip:{clean_segment(scope, name='scope')}:{safe_ip}"


def bucket_ttl_ms(rate: float, burst: int) -> int:
    """TTL ведра: полный «перезаряд» с запасом ×2, в границах [60 с, 1 ч]."""
    refill_ms = (burst / rate) * 1000.0 * 2.0 if rate > 0 else MAX_BUCKET_TTL_S * 1000
    return int(min(max(refill_ms, MIN_BUCKET_TTL_S * 1000), MAX_BUCKET_TTL_S * 1000))


class TokenBucketScript:
    """Обёртка над Lua-скриптом: ``SCRIPT LOAD`` один раз, дальше ``EVALSHA``.

    ``NOSCRIPT`` (Redis перезапустили, кэш скриптов пуст) — не ошибка: скрипт
    перезагружается и вызов повторяется. Без этого после рестарта Redis лимитер
    начал бы падать на каждой проверке.
    """

    def __init__(self, redis: ScriptRedis) -> None:
        self._redis = redis
        self._sha: Optional[str] = None
        self._lock = threading.Lock()

    def _load(self) -> str:
        sha = self._redis.script_load(LUA_TOKEN_BUCKET)
        if not sha:
            # RedisClient при ошибке возвращает None (его соглашение — «не падать»).
            # Для лимитера это НЕ «ок»: превращаем в явную недоступность бэкенда.
            raise RateLimitBackendError("SCRIPT LOAD не вернул SHA (Redis недоступен?)")
        return sha

    def run(self, key: str, rate: float, burst: int, want: float, ttl_ms: int) -> list:
        with self._lock:
            if self._sha is None:
                self._sha = self._load()
            sha = self._sha
        args = (rate, burst, want, ttl_ms)
        try:
            reply = self._redis.evalsha(sha, 1, key, *args)
        except Exception as exc:  # noqa: BLE001 — NOSCRIPT или транспорт
            if "NOSCRIPT" not in str(exc).upper() and "No matching script" not in str(exc):
                raise RateLimitBackendError(str(exc)) from exc
            logger.info("Lua-скрипт лимитера не в кэше Redis — перезагружаю")
            with self._lock:
                self._sha = self._load()
                sha = self._sha
            reply = self._redis.evalsha(sha, 1, key, *args)
        return self._validate(key, reply)

    @staticmethod
    def _validate(key: str, reply: object) -> list:
        """Ответ обязан быть тройкой ``{allowed, tokens_left, wait_ms}``.

        ``None`` (клиент вернул «ошибка») или другая форма — это недоступность бэкенда:
        принять такое за «разрешено» значит снять лимит ровно тогда, когда Redis сломан.
        """
        if reply is None:
            raise RateLimitBackendError(f"EVALSHA вернул None для {key}")
        if not isinstance(reply, (list, tuple)):
            # Явная проверка формы, а не приведение: строка тоже «итерируема», и
            # `list("abc")` дал бы тройку символов, которую легко принять за ответ.
            raise RateLimitBackendError(f"нечитаемый ответ EVALSHA для {key}: {reply!r}")
        values = list(reply)
        if len(values) != 3:
            raise RateLimitBackendError(f"ответ EVALSHA для {key} не тройка: {values!r}")
        return values


class RedisTokenBucket:
    """Ведро провайдера, живущее в Redis (единый авторитет для всех воркеров)."""

    def __init__(
        self,
        redis: ScriptRedis,
        provider: str,
        rate: float,
        burst: int,
        *,
        script: Optional[TokenBucketScript] = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_wait_s: float = 5.0,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate должен быть > 0, получено {rate}")
        if burst < 1:
            raise ValueError(f"burst должен быть >= 1, получено {burst}")
        self._redis = redis
        self.provider = provider
        self.key = rate_limit_key(provider)
        self.rate = float(rate)
        self.burst = int(burst)
        self._script = script or TokenBucketScript(redis)
        self._sleep = sleeper
        self._max_wait_s = max_wait_s
        self._ttl_ms = bucket_ttl_ms(self.rate, self.burst)

    def acquire(self, tokens: float = 1.0, blocking: bool = True) -> bool:
        """Забрать ``tokens`` из ведра.

        Возвращает True, если лимит позволил запрос. Если ``blocking`` — ждёт по
        ``wait_ms`` от скрипта (ожидание ограничено ``max_wait_s``: лимитер не должен
        превращаться в бесконечную очередь).
        """
        if tokens <= 0:
            return True
        if tokens > self.burst:
            logger.warning(
                "Лимитер %s: запрошено %.2f токенов при burst=%d — запрос пропущен",
                self.provider, tokens, self.burst,
            )
            return False

        deadline = time.monotonic() + self._max_wait_s
        while True:
            allowed, _left, wait_ms = self._try(tokens)
            if allowed:
                return True
            if not blocking:
                return False
            if time.monotonic() >= deadline:
                logger.warning(
                    "Лимитер %s: ожидание больше %.1f с — запрос отклонён", self.provider, self._max_wait_s
                )
                return False
            self._sleep(min(max(wait_ms, 1) / 1000.0, self._max_wait_s))

    def _try(self, tokens: float) -> tuple[bool, float, int]:
        raw = self._script.run(self.key, self.rate, self.burst, tokens, self._ttl_ms)
        allowed = bool(int(raw[0]))
        left = float(raw[1]) if raw[1] not in (None, b"", "") else 0.0
        wait_ms = int(raw[2] or 0)
        return allowed, left, wait_ms

    @property
    def available(self) -> float:
        """Остаток токенов (приблизительно: считается между пополнениями)."""
        try:
            data = self._redis.eval(LUA_PEEK, 1, self.key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Лимитер %s: остаток недоступен (%s)", self.provider, exc)
            return 0.0
        if not data or data[0] in (b"", "", None):
            return float(self.burst)
        return float(data[0])

    def reset(self) -> bool:
        """Сбросить ведро (тесты и ручное вмешательство)."""
        try:
            return bool(self._redis.delete(self.key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Лимитер %s: сброс не удался (%s)", self.provider, exc)
            return False


class RedisRateLimiter:
    """Лимитер провайдеров на Redis-Lua. Интерфейс совпадает с локальным ``RateLimiter``.

    Лимиты передаются снаружи (словарь ``{provider: {rate, burst}}``): адаптер не должен
    знать про настройки приложения — иначе получится импорт из внешнего кольца внутрь.
    """

    def __init__(
        self,
        redis: ScriptRedis,
        limits: dict[str, dict],
        *,
        degraded_limiter: Optional[LimiterLike] = None,
    ) -> None:
        self._redis = redis
        self._degraded = degraded_limiter
        self._script = TokenBucketScript(redis)
        self._buckets: dict[str, RedisTokenBucket] = {
            name: RedisTokenBucket(
                redis, name, params["rate"], int(params["burst"]), script=self._script
            )
            for name, params in limits.items()
        }
        self.degraded_calls = 0
        self._warned: set[str] = set()

    def get_bucket(self, provider: str) -> Optional[RedisTokenBucket]:
        """Ведро провайдера. Имя канонизируется: иначе ``wait("moex")`` не нашёл бы
        ведро ``moex_iss`` и запрос ушёл бы **без лимита** — тихая дыра."""
        try:
            return self._buckets.get(canonical_provider(provider))
        except Exception:  # noqa: BLE001 — некорректное имя не должно ронять запрос
            return None

    def acquire(self, provider: str, tokens: float = 1.0, blocking: bool = True) -> bool:
        bucket = self.get_bucket(provider)
        if bucket is None:
            return True  # неизвестный провайдер — не ограничиваем (как и локальный лимитер)
        try:
            return bucket.acquire(tokens, blocking=blocking)
        except Exception as exc:  # noqa: BLE001 — Redis недоступен
            return self._degrade(provider, exc, tokens, blocking)

    def wait(self, provider: str) -> None:
        self.acquire(provider, tokens=1.0, blocking=True)

    def reset_all(self) -> None:
        for bucket in self._buckets.values():
            bucket.reset()

    @property
    def providers(self) -> list[str]:
        return list(self._buckets.keys())

    def _degrade(self, provider: str, exc: BaseException, tokens: float, blocking: bool) -> bool:
        """Redis недоступен: лимит переходит в per-process режим, но об этом **сообщается**.

        Молча вернуть «разрешено» нельзя: это снимает защиту провайдера целиком и никак
        не видно снаружи. Предупреждение пишется один раз на провайдера (иначе под
        нагрузкой лог превращается в поток), счётчик ``degraded_calls`` — всегда.
        """
        self.degraded_calls += 1
        if provider not in self._warned:
            self._warned.add(provider)
            logger.warning(
                "Лимитер %s: Redis недоступен (%s) — лимит сведён к per-process. "
                "Фактическая частота может превысить заданную в число воркеров раз.",
                provider, exc,
            )
        if self._degraded is None:
            return True
        return self._degraded.acquire(provider, tokens=tokens, blocking=blocking)

    def __repr__(self) -> str:
        return f"<RedisRateLimiter providers={sorted(self._buckets)} degraded={self.degraded_calls}>"


class RedisIpLimiter:
    """Per-IP лимитер на Redis-Lua (для login/register и подобных ручек).

    Отличие от локального :class:`gex.rate_limiter.IpRateLimiter` принципиальное: там
    ведро живёт в процессе, поэтому защита от перебора умножается на число воркеров.
    """

    def __init__(
        self,
        redis: ScriptRedis,
        scope: str,
        rate: float,
        burst: int,
        *,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._redis = redis
        self.scope = scope
        self.rate = float(rate)
        self.burst = int(burst)
        self._script = TokenBucketScript(redis)
        self._ttl_ms = bucket_ttl_ms(self.rate, self.burst)
        self._sleeper = sleeper
        self.degraded_calls = 0

    def allow(self, ip: str) -> bool:
        """True — запрос разрешён (без ожидания)."""
        key = ip_rate_limit_key(self.scope, ip)
        try:
            raw = self._script.run(key, self.rate, self.burst, 1.0, self._ttl_ms)
            return bool(int(raw[0]))
        except Exception as exc:  # noqa: BLE001 — Redis недоступен
            self.degraded_calls += 1
            logger.warning("IP-лимитер %s: Redis недоступен (%s) — пропускаю запрос", self.scope, exc)
            return True


def make_limits(limits: dict[str, dict]) -> dict[str, dict]:
    """Валидировать словарь лимитов до передачи в Redis-адаптер.

    Ошибка в конфиге (rate=0, burst=0) должна всплыть при сборке, а не при первом
    запросе пользователя: иначе деление на ноль окажется «пятьюстами» в проде.
    """
    checked: dict[str, dict] = {}
    origins: dict[str, str] = {}
    for name, params in limits.items():
        rate = float(params["rate"])
        burst = int(params["burst"])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"лимит {name}: rate={rate} должен быть > 0")
        if burst < 1:
            raise ValueError(f"лимит {name}: burst={burst} должен быть >= 1")
        canon = canonical_provider(name)
        if canon in checked:
            # Два имени одного провайдера = два ведра = двойной лимит. Это ошибка
            # конфигурации, и она обязана всплыть при сборке, а не в проде.
            raise ValueError(
                f"лимиты {origins[canon]!r} и {name!r} — один провайдер ({canon}): "
                f"получилось бы два ведра и двойной лимит"
            )
        origins[canon] = name
        checked[canon] = {"rate": rate, "burst": burst}
    return checked


__all__ = [
    "KEY_PREFIX",
    "LimiterLike",
    "RateLimitBackendError",
    "ScriptRedis",
    "LUA_PEEK",
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
