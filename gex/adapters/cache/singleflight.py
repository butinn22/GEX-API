"""Single-flight: N одновременных запросов одного ключа → 1 вычисление (ring: adapters).

Что не так с прежним поведением
-------------------------------
В :mod:`gex.result_cache` был только локальный ``threading.Lock`` на ключ, и это давало две
проблемы:

1. **Follower мог запустить loader сам.** При ``acquire(timeout=60)`` и таймауте код уходил
   в ``compute()`` напрямую — то есть под нагрузкой дублировал самую дорогую работу
   ровно тогда, когда она и так не успевает.
2. **Кросс-процессного схождения не было вовсе.** ``threading.Lock`` действует внутри
   одного процесса, а воркеров несколько: N воркеров = N вычислений одного и того же.

Как устроено здесь
------------------
* :class:`LocalSingleFlight` — один лидер на ключ внутри процесса; follower **получает
  результат лидера** (или его исключение) и не зовёт loader никогда.
* :class:`RedisSingleFlight` — лидер определяется арендой в Redis (``SET NX PX``); follower
  ждёт и перечитывает кэш, loader не запускает. Если лидер пропал (аренда истекла, процесс
  умер), один из ожидающих забирает аренду и считает — иначе запрос не завершится никогда.
* Деградация без Redis — явная и в логах: без общего хранилища схождение невозможно,
  поэтому лидером становится каждый (как сегодня), но об этом пишется предупреждение.

Инвариант, который проверяется тестом: **число вызовов loader'а при N параллельных
запросах равно 1**, а не N.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable, Optional

from .envelope import RedisLike

logger = logging.getLogger(__name__)

#: Префикс ключа аренды. Отдельно от ключа данных: аренда — про «кто считает сейчас»,
#: данные — про «что уже посчитано», и смешивать их значит терять данные при истечении аренды.
LOCK_PREFIX = "gex:lock:"

#: Минимальная аренда: меньше — лидер не успеет записать результат.
MIN_LEASE_MS = 100


class _Slot:
    """Ячейка результата для ожидающих потоков внутри процесса."""

    __slots__ = ("event", "value", "error", "done")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: object = None
        self.error: Optional[BaseException] = None
        self.done = False

    def publish(self, value: object) -> None:
        self.value, self.done = value, True
        self.event.set()

    def fail(self, error: BaseException) -> None:
        self.error, self.done = error, True
        self.event.set()

    def wait(self, timeout: float) -> bool:
        return self.event.wait(timeout)


class LocalSingleFlight:
    """Одно вычисление на ключ внутри процесса.

    Follower не вызывает ``loader``: он ждёт результат лидера. При таймауте ожидания
    follower всё равно **не** считает сам — он возвращает ``None`` (промах), и решение
    «считать или отдать устаревшее» остаётся за вызывающим слоем. Это осознанный выбор:
    дублировать дорогую работу под нагрузкой хуже, чем отдать stale.
    """

    def __init__(self, *, wait_timeout: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._wait_timeout = wait_timeout
        self._clock = clock
        self._slots: dict[str, _Slot] = {}
        self._guard = threading.Lock()
        self.loader_calls = 0
        self.follower_misses = 0

    def run(self, key: str, loader: Callable[[], object], *, on_miss: Optional[Callable[[], object]] = None) -> object:
        """Выполнить ``loader`` один раз для всех ожидающих по этому ключу.

        ``on_miss`` — что вернуть, если ожидание истекло (обычно устаревшее значение из кэша).
        """
        with self._guard:
            slot = self._slots.get(key)
            is_leader = slot is None
            if is_leader:
                slot = self._slots[key] = _Slot()
        assert slot is not None

        if is_leader:
            try:
                self.loader_calls += 1
                value = loader()
            except BaseException as exc:  # noqa: BLE001 — исключение транслируем ожидающим
                slot.fail(exc)
                raise
            else:
                slot.publish(value)
                return value
            finally:
                with self._guard:
                    self._slots.pop(key, None)

        # follower: ждём лидера, loader не запускаем
        if slot.wait(self._wait_timeout):
            if slot.error is not None:
                raise slot.error
            return slot.value
        self.follower_misses += 1
        logger.warning("Single-flight: ожидание истекло для %s (%.1f с)", key, self._wait_timeout)
        return on_miss() if on_miss is not None else None


class RedisSingleFlight:
    """Одно вычисление на ключ **между процессами** — через аренду в Redis.

    Порядок для лидера: взять аренду → (double-check кэша) → посчитать → записать в кэш →
    снять аренду. Follower: ждать появления свежего значения в кэше; loader не запускать.

    ``is_fresh`` по умолчанию — «в кэше есть что угодно». Если вызывающий передал свой
    предикат, follower ждёт именно свежести, а не факта записи.
    """

    def __init__(
        self,
        redis: RedisLike,
        *,
        lease_ms: int = 30_000,
        wait_ms: int = 5_000,
        poll_s: float = 0.02,
        max_poll_s: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._redis = redis
        self._lease_ms = max(int(lease_ms), MIN_LEASE_MS)
        self._wait_ms = int(wait_ms)
        # Интервал опроса растёт экспоненциально от ``poll_s`` до ``max_poll_s``:
        # фиксированные 20 мс давали до 250 чтений кэша на одного ожидающего за 5 с
        # (N ожидающих × 250 обращений к Redis при каждом схождении). Рост интервала
        # оставляет отклик быстрым на коротких ожиданиях и убирает шторм чтений на длинных.
        self._poll_s = max(float(poll_s), 1e-4)
        self._max_poll_s = max(float(max_poll_s), self._poll_s)
        self._clock = clock
        self._sleep = sleeper
        self.loader_calls = 0
        self.takeovers = 0

    @staticmethod
    def lock_key(key: str) -> str:
        return f"{LOCK_PREFIX}{key}"

    def _acquire(self, key: str, token: str) -> bool:
        try:
            return bool(self._redis.set(self.lock_key(key), token, nx=True, px=self._lease_ms))
        except Exception as exc:  # noqa: BLE001 — без Redis схождение невозможно
            logger.warning("Аренда single-flight недоступна (%s) — считаю без схождения", exc)
            return True

    def _release(self, key: str, token: str) -> None:
        """Снять **свою** аренду; сравнение токена терпимо к ``bytes``.

        Реальный ``RedisClient.get`` возвращает ``bytes`` (``decode_responses=False``), поэтому
        прежнее ``self._redis.get(...) == token`` (``bytes == str``) всегда ложно: аренда
        не снималась и висела весь срок (``REDIS_LEASE_MS`` = 60 с). Следующие ожидающие не
        могли её перехватить, получали ``None`` и роняли запрос в 500.

        Порядок: прочитать → декодировать байты как utf-8 → сравнить **как строки** → удалить.
        При ``UnicodeDecodeError`` (в ключе legacy-pickled токен) — это чужая аренда: логируем
        debug и **не удаляем**, она истечёт по TTL сама. Удалять не своё нельзя.
        """
        try:
            current = self._redis.get(self.lock_key(key))
            if current is None:
                return
            if isinstance(current, (bytes, bytearray)):
                try:
                    current = bytes(current).decode("utf-8")
                except UnicodeDecodeError:
                    # legacy pickled-токен: это не наша аренда — не трогаем, истечёт по TTL.
                    logger.debug("Аренда %s в неизвестном формате — не освобождаю", key)
                    return
            if str(current) != str(token):
                return
            self._redis.delete(self.lock_key(key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Аренда %s не снята: %s", key, exc)

    def run(
        self,
        key: str,
        loader: Callable[[], object],
        *,
        read: Callable[[], object],
        is_fresh: Optional[Callable[[object], bool]] = None,
    ) -> object:
        """``read()`` — чтение значения из кэша, ``loader()`` — вычисление + запись в кэш."""
        fresh = is_fresh or (lambda value: value is not None)
        token = uuid.uuid4().hex

        if self._acquire(key, token):
            try:
                cached = read()
                if fresh(cached):
                    return cached  # double-check: пока брали аренду, значение появилось
                self.loader_calls += 1
                return loader()
            finally:
                self._release(key, token)

        # follower: ждём значение, loader НЕ запускаем
        deadline = self._clock() + self._wait_ms / 1000.0
        pause = self._poll_s
        while self._clock() < deadline:
            cached = read()
            if fresh(cached):
                return cached
            self._sleep(pause)
            pause = min(pause * 2.0, self._max_poll_s)

        # Лидер не успел. Забираем аренду, если он её уже отпустил (или умер).
        if self._acquire(key, token):
            self.takeovers += 1
            try:
                cached = read()
                if fresh(cached):
                    return cached
                self.loader_calls += 1
                return loader()
            finally:
                self._release(key, token)

        # Аренда всё ещё у лидера, значение не появилось: отдаём что есть (или None).
        logger.warning("Single-flight: лидер не успел для %s — отдаю кэш как есть", key)
        return read()


__all__ = ["LOCK_PREFIX", "MIN_LEASE_MS", "LocalSingleFlight", "RedisSingleFlight"]
