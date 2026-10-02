"""Хранилище payload'ов страниц: реализация ``CachePort`` **и** ``SnapshotPort`` (ring: adapters).

Что это
-------
Порт ``CachePort`` описывает продукт целиком: «страница получает последнее известное значение
мгновенно, свежесть добирается фоном, вычисление одного ключа схлопывается между процессами».
До этой итерации реализации у порта не было: роутеры ходили в ``result_cache``, который не
умел ни etag, ни признака «идёт пересчёт», ни версии схемы значения.

Второй контракт — :class:`gex.ports.cache.SnapshotPort` (``peek``/``write``) — добавлен для
страниц, вынесенных из запроса (широта рынка, композит секторов, широта IMOEX). ``peek``
**никогда** не вычисляет: если значения нет, страница отвечает «данные готовятся» вместо того,
чтобы ждать yfinance (широта рынка — это ~500 бумаг и минуты, а не секунды).

Два яруса, и это не дублирование
--------------------------------
* **Redis** — общий ярус: значение, положенное фоновым пересчётом (воркер очереди),
  API. Ключ живёт ``2 x fresh``, поэтому устаревшее значение есть чем подменить ответ.
* **Память процесса** — страховка от «Redis недоступен». Ровно в этот момент кэш нужнее
  всего: без яруса памяти недоступность Redis означала бы не «страница стареет», а «каждый
  запрос считает всё заново» (инцидент 2026-09-21: 6533 треда, дашборд лёг).

Из чего собрано
---------------
* значение — конверт :mod:`gex.adapters.cache.envelope` (версия схемы, метка времени, etag,
  источник) — итер. 26;
* схождение вычислений — :class:`gex.adapters.cache.singleflight.RedisSingleFlight` (аренда
  в Redis) — итер. 26;
* окна свежести приходят **снаружи** (``gex.domain.freshness``): хранилище не знает, что
  «страница конуса» устаревает за 600 с, а «котировка» за 30 с.

Признак «идёт пересчёт»
-----------------------
``computing`` — это наличие аренды (``gex:lock:{key}``), а не отдельное поле в конверте:
аренда уже означает «кто-то считает прямо сейчас», и дублировать это состояние вторым флагом
значило бы заводить два источника правды, которые разъедутся при падении процесса.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, Protocol

from gex.adapters.cache.envelope import Envelope, RedisEnvelopeCache
from gex.adapters.cache.singleflight import RedisSingleFlight
from gex.ports.cache import CachedPayload, CacheStatus

logger = logging.getLogger(__name__)

#: Сколько ждать лидера, прежде чем вернуть устаревшее значение. Пользователь не должен
#: ждать пересчёта: если лидер не успел — отдаём то, что есть.
DEFAULT_MAX_WAIT_MS = 3_000

#: Срок аренды лидера: с запасом на самый тяжёлый эндпоинт (sector ~14 с) + запись.
DEFAULT_LEASE_MS = 60_000

#: Потолок записей яруса памяти процесса. Значений мало (по одному на страницу и режим), но
#: граница обязана быть: без неё словарь растёт вместе с разнообразием ключей.
DEFAULT_MEM_ENTRIES = 32


class EnvelopeRedis(Protocol):
    """Минимум клиента Redis, нужный хранилищу (совместим с ``RedisClient``).

    Протокол вместо нетипизированного параметра: он документирует контракт и не даёт
    разойтись тестовому фейку и продовому клиенту (этот класс дефекта встречался трижды).
    """

    def get(self, key: str) -> object: ...
    def set(self, key: str, value: object, ex: Optional[int] = None, *,
            px: Optional[int] = None, nx: bool = False) -> object: ...
    def delete(self, key: str) -> object: ...


class RedisPageStore:
    """SWR-хранилище значений поверх Redis-конверта (``CachePort`` + ``SnapshotPort``)."""

    def __init__(
        self,
        redis: Optional[EnvelopeRedis],
        *,
        max_wait_ms: int = DEFAULT_MAX_WAIT_MS,
        lease_ms: int = DEFAULT_LEASE_MS,
        clock: Optional[Callable[[], float]] = None,
        start_threads: bool = True,
        mem_entries: int = DEFAULT_MEM_ENTRIES,
    ) -> None:
        self._redis = redis
        self._max_wait_ms = max_wait_ms
        self._clock = clock
        self._start_threads = start_threads
        # Два разных времени, и путать их нельзя:
        #  * конверт живёт по **wall-clock** (возраст значения сравнивается с TTL);
        #  * ожидание лидера — по **монотонному** (иначе перевод часов сдвинул бы дедлайн).
        self._cache = RedisEnvelopeCache(redis, clock=clock) if clock else RedisEnvelopeCache(redis)
        self._singleflight = RedisSingleFlight(redis, lease_ms=lease_ms, wait_ms=max_wait_ms)
        self._bg: set[str] = set()
        self._bg_guard = threading.Lock()
        # Ярус памяти: key -> (stored_at, retention_s, value, etag). Пишется вместе с Redis
        # (не вместо него), чтобы недоступность Redis не превращалась в «кэша нет вовсе».
        self._mem: dict[str, tuple[float, int, Any, str]] = {}
        self._mem_guard = threading.Lock()
        self._mem_entries = max(4, int(mem_entries))
        # Счётчики: без них «кэш работает» и «кэш не работает» выглядят одинаково.
        self.stats: dict[str, int] = {
            "hit": 0, "stale": 0, "miss": 0, "computing": 0, "refresh_started": 0,
            "refresh_failed": 0, "peek_hit": 0, "peek_stale": 0, "peek_expired": 0,
            "peek_empty": 0, "peek_memory": 0, "write": 0,
        }

    # ------------------------------------------------------------------ #
    #  CachePort
    # ------------------------------------------------------------------ #
    def get(
        self,
        key: str,
        *,
        fresh: int,
        stale_max: int,
        compute: Callable[[], Any],
        max_wait_ms: Optional[int] = None,
    ) -> CachedPayload:
        """Вернуть значение по SWR-логике.

        * свежее (``age <= fresh``) — как есть, без пересчёта;
        * устаревшее, но годное (``fresh < age <= stale_max``) — отдаём сразу и обновляем
          в фоне; если фоновый пересчёт уже идёт в другом процессе, помечаем ``computing``;
        * отсутствующее или слишком старое — считаем синхронно под single-flight.
        """
        envelope = self._cache.read(key)
        now = self._now()
        stale_max = max(stale_max, fresh)

        if envelope is not None and not envelope.is_legacy():
            age = envelope.age(now)
            if age <= fresh:
                self.stats["hit"] += 1
                return self._payload(envelope, CacheStatus.HIT, fresh, stale_max)
            if age <= stale_max:
                computing = self._is_computing(key)
                self._refresh_async(key, fresh, compute)
                self.stats["stale"] += 1
                if computing:
                    self.stats["computing"] += 1
                return self._payload(
                    envelope,
                    CacheStatus.COMPUTING if computing else CacheStatus.STALE,
                    fresh,
                    stale_max,
                    computing=computing,
                )

        else:
            # Redis пуст или недоступен, но значение есть в памяти процесса: отдаём его как
            # устаревшее, а не считаем в запросе. Порядок предпочтений тот же, что и в
            # Redis-ветке: «устаревшее» → «пересчёт», и никогда наоборот.
            entry = self._mem_read(key, now)
            if entry is not None:
                stored_at, _retention, value, etag = entry
                age = now - stored_at
                if age <= stale_max:
                    self.stats["stale"] += 1
                    logger.debug(
                        "Кэш %s: Redis недоступен — отдаю значение из памяти (возраст %.0f c)", key, age
                    )
                    return CachedPayload(
                        value=value, status=CacheStatus.STALE, ts=stored_at, fresh=fresh,
                        stale_max=stale_max, etag=etag or None, age_s=age,
                    )

        value = self._recompute(key, fresh, compute, wait_ms=max_wait_ms)
        envelope = self._cache.read(key)  # перечитываем: конверт записал лидер
        self.stats["miss"] += 1
        if envelope is not None:
            return self._payload(envelope, CacheStatus.MISS, fresh, stale_max)
        # Redis недоступен: значение есть, метаданных нет — это не повод не отдать данные.
        return CachedPayload(
            value=value, status=CacheStatus.MISS, ts=now, fresh=fresh, stale_max=stale_max
        )

    def invalidate(self, key: str) -> None:
        """Сбросить ключ (например, после ручного обновления данных)."""
        self._cache.invalidate(key)
        with self._mem_guard:
            self._mem.pop(key, None)

    # ------------------------------------------------------------------ #
    #  SnapshotPort: чтение без вычисления + запись фоновым пересчётом
    # ------------------------------------------------------------------ #
    def peek(
        self,
        key: str,
        *,
        fresh: int,
        stale_max: int,
    ) -> Optional[CachedPayload]:
        """Последнее известное значение без вычисления (Redis → память процесса).

        Классификацию свежести задаёт **вызывающий** (``fresh``/``stale_max`` пришли из
        ``gex.domain.freshness`` и зависят от торговой сессии), а не ttl, с которым запись
        когда-то легла: вне сессии окна шире, и значение, записанное днём, вечером обязано
        считаться свежим. Поэтому ``ttl`` конверта для классификации не используется —
        только возраст против окон читателя.

        Возвращает ``None``, если значения нет нигде. ``EXPIRED`` возвращается со значением
        (диагностика «было, но протухло»), решение «отдавать или нет» — за вызывающим.
        """
        stale_max = max(stale_max, fresh)
        now = self._now()

        envelope = self._cache.read(key)
        if envelope is not None and not envelope.is_legacy():
            age = envelope.age(now)
            status = self._classify(age, fresh, stale_max)
            self._count_peek(status)
            return self._payload(envelope, status, fresh, stale_max, age=age)

        entry = self._mem_read(key, now)
        if entry is not None:
            stored_at, _retention, value, etag = entry
            age = now - stored_at
            status = self._classify(age, fresh, stale_max)
            self._count_peek(status)
            self.stats["peek_memory"] += 1
            logger.debug("Snapshot %s отдан из памяти процесса (возраст %.0f c)", key, age)
            return CachedPayload(
                value=value,
                status=status,
                ts=stored_at,
                fresh=fresh,
                stale_max=stale_max,
                etag=etag or None,
                age_s=age,
            )

        self.stats["peek_empty"] += 1
        return None

    def write(self, key: str, value: Any, *, fresh: int, source: str = "") -> None:
        """Записать значение в оба яруса.

        ``fresh`` — окно, до которого значение считается свежим (из ``gex.domain.freshness``);
        ключ в Redis живёт вдвое дольше, иначе устаревшее нечего было бы отдать под флагом
        ``STALE``. В память кладём с тем же сроком: этот ярус существует ровно для случая
        «Redis недоступен», и держать в нём вечные значения незачем.
        """
        retention = max(int(fresh), 1) * 2
        envelope = self._cache.write(key, value, fresh, source=source)
        etag = envelope.etag if envelope is not None else ""
        self._mem_write(key, value, retention, etag)
        self.stats["write"] += 1
        if envelope is None:
            logger.info(
                "Snapshot %s сохранён только в памяти процесса (Redis недоступен) — "
                "фон его пересчитает, страница получит устаревшее значение",
                key,
            )

    def last_written_at(self, key: str) -> Optional[float]:
        """Когда значение писалось в последний раз (для диагностики и админки)."""
        envelope = self._cache.read(key)
        if envelope is not None and not envelope.is_legacy():
            return envelope.stored_at
        entry = self._mem_read(key, self._now())
        return entry[0] if entry is not None else None

    # ------------------------------------------------------------------ #
    #  Наблюдаемость
    # ------------------------------------------------------------------ #
    def is_computing(self, key: str) -> bool:
        return self._is_computing(key)

    def describe(self) -> dict:
        return {"stats": dict(self.stats), "available": self._redis is not None}

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        return (self._clock or time.time)()

    def _payload(
        self,
        envelope: Envelope,
        status: CacheStatus,
        fresh: int,
        stale_max: int,
        *,
        computing: bool = False,
        age: float = 0.0,
    ) -> CachedPayload:
        return CachedPayload(
            value=envelope.value,
            status=status,
            ts=envelope.stored_at,
            fresh=fresh,
            stale_max=stale_max,
            version=envelope.version,
            etag=envelope.etag or None,
            computing=computing,
            age_s=age,
        )

    @staticmethod
    def _classify(age: float, fresh: int, stale_max: int) -> CacheStatus:
        """Статус по возрасту: свежее → годное устаревшее → протухшее."""
        if age <= fresh:
            return CacheStatus.HIT
        if age <= stale_max:
            return CacheStatus.STALE
        return CacheStatus.EXPIRED

    def _count_peek(self, status: CacheStatus) -> None:
        self.stats["peek_hit" if status is CacheStatus.HIT else
                   "peek_stale" if status is CacheStatus.STALE else "peek_expired"] += 1

    # ------------------------------------------------------------------ #
    #  Ярус памяти процесса
    # ------------------------------------------------------------------ #
    def _mem_read(self, key: str, now: float) -> Optional[tuple[float, int, Any, str]]:
        with self._mem_guard:
            entry = self._mem.get(key)
            if entry is None:
                return None
            stored_at, retention, _value, _etag = entry
            if now - stored_at > retention:
                # Просроченное забываем сразу: иначе память процесса превращается в архив,
                # из которого страница однажды отдаст «данные недельной давности».
                self._mem.pop(key, None)
                return None
            return entry

    def _mem_write(self, key: str, value: Any, retention: int, etag: str) -> None:
        with self._mem_guard:
            self._mem[key] = (self._now(), int(retention), value, etag)
            if len(self._mem) <= self._mem_entries:
                return
            oldest = sorted(self._mem.items(), key=lambda kv: kv[1][0])
            for stale_key, _ in oldest[: len(self._mem) - self._mem_entries]:
                self._mem.pop(stale_key, None)

    def _is_computing(self, key: str) -> bool:
        """Идёт ли пересчёт ключа (аренда single-flight занята).

        Ошибка чтения аренды не считается «идёт пересчёт»: ложный ``computing`` в ответе
        показал бы пользователю индикатор ревалидации, которой нет.
        """
        if self._redis is None:
            return False
        try:
            return bool(self._redis.get(RedisSingleFlight.lock_key(key)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Аренда %s недоступна: %s", key, exc)
            return False

    def _recompute(
        self, key: str, fresh: int, compute: Callable[[], Any], *, wait_ms: Optional[int]
    ) -> Any:
        """Считать под кросс-процессной арендой; follower берёт результат из кэша."""

        def loader() -> Any:
            value = compute()
            # ttl конверта = окно «ещё годно»: ключ живёт дольше свежести, иначе
            # устаревшее значение нечего было бы отдавать под флагом STALE.
            envelope = self._cache.write(key, value, fresh)
            # Дублируем в память процесса: если Redis упадёт между запросами, странице
            # будет что отдать (см. ветку ``_mem_read`` выше в ``get``).
            self._mem_write(key, value, max(int(fresh), 1) * 2, envelope.etag if envelope else "")
            return value

        def read() -> Optional[Any]:
            envelope = self._cache.read(key)
            if envelope is None or envelope.is_legacy():
                return None
            return envelope.value if envelope.age(self._now()) <= fresh else None

        if self._redis is None or not getattr(self._redis, "connected", True):
            return compute()
        try:
            return self._singleflight.run(key, loader, read=read, is_fresh=lambda v: v is not None)
        except Exception as exc:  # noqa: BLE001 — хранилище не имеет права ронять запрос
            logger.warning("Redis недоступен при пересчёте %s (%s) — считаю без кэша", key, exc)
            return compute()

    def _refresh_async(self, key: str, fresh: int, compute: Callable[[], Any]) -> None:
        """Фоновый пересчёт: один поток на ключ.

        Пользователь уже получил ответ, поэтому сбой обновления — это warning, а не ошибка
        запроса; но он обязан быть посчитан (``refresh_failed``), иначе деградация кэша
        останется незаметной.
        """
        if not self._start_threads:
            # Тестовый режим: обновление выполняет сам тест (см. refresh_now) — потоков нет,
            # и в набор «уже обновляется» ничего не попадает, иначе ключ застрял бы в нём.
            self.stats["refresh_started"] += 1
            return
        with self._bg_guard:
            if key in self._bg:
                return
            self._bg.add(key)
        self.stats["refresh_started"] += 1

        def worker() -> None:
            try:
                self._recompute(key, fresh, compute, wait_ms=self._max_wait_ms)
            except Exception as exc:  # noqa: BLE001
                self.stats["refresh_failed"] += 1
                logger.warning("Фоновое обновление %s не удалось: %s", key, exc)
            finally:
                with self._bg_guard:
                    self._bg.discard(key)

        threading.Thread(target=worker, daemon=True, name=f"page-refresh-{key[-20:]}").start()

    def refresh_now(self, key: str, fresh: int, compute: Callable[[], Any]) -> Any:
        """Фоновое обновление синхронно (тесты и предпрогрев)."""
        try:
            return self._recompute(key, fresh, compute, wait_ms=self._max_wait_ms)
        finally:
            with self._bg_guard:
                self._bg.discard(key)


__all__ = [
    "DEFAULT_LEASE_MS",
    "DEFAULT_MAX_WAIT_MS",
    "DEFAULT_MEM_ENTRIES",
    "EnvelopeRedis",
    "RedisPageStore",
]
