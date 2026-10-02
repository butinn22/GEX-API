"""Кэш результатов аналитики: Stale-While-Revalidate + single-flight.

Проблема: тяжёлые аналитические эндпоинты (macd ~2.9с, ext/gex ~1.4с,
signals, cone, sector ~14с) считаются заново при каждом запросе, даже
когда исходные данные (OHLCV/цепочки) уже в кэше. Это CPU-задержки,
которые нельзя убрать кэшированием данных.

Решение — кэш ФИНАЛЬНОГО результата в Redis:

* свежий (age <= ttl)           → вернуть мгновенно;
* устаревший (ttl < age <= max) → вернуть сразу (stale) + фоновый
  пересчёт (SWR) — пользователь никогда не ждёт;
* отсутствует/слишком старый    → вычислить под single-flight
  (N параллельных запросов = 1 вычисление, в т.ч. между воркерами).

Хранение: ``gex:res:…`` → конверт :class:`gex.adapters.cache.envelope.Envelope`
(версия схемы, метка времени, etag, источник). Записи прежней формы ``{"v","ts"}``
читаются как устаревшие и перезаписываются в новом виде — см. итер. 26.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from gex.adapters.cache.envelope import RedisEnvelopeCache
from gex.adapters.cache.singleflight import LocalSingleFlight, RedisSingleFlight
from gex.adapters.cache.redis_client import cache_key, get_redis, serialize_value  # noqa: F401  (реэкспорт: его зовут роутеры)

logger = logging.getLogger(__name__)

#: Сколько ждать лидера внутри процесса. Больше не нужно: SWR уже отдал stale.
LOCAL_WAIT_TIMEOUT = 30.0

#: Кросс-процессное ожидание. Короче локального: если лидер не успел, отдаём stale.
REDIS_WAIT_MS = 5_000

#: Аренда лидера с запасом на самый дорогой эндпоинт (sector ~14 с) + запись.
REDIS_LEASE_MS = 60_000


class ResultCache:
    """SWR-кэш результатов с single-flight (потокобезопасен).

    Redis — основной слой, но не единственный: при его недоступности включается
    внутрипроцессный TTL-кэш (:attr:`_mem`). Раньше при ``Redis == None`` кэш
    отключался целиком и каждый запрос шёл в живой апстрим — именно так дашборд
    положил процесс (6533 треда, инцидент 2026-09-21). Память ограничена
    :data:`MEM_MAX_ENTRIES` записями, вытесняются самые старые.
    """

    #: Потолок записей внутрипроцессного кэша (без него словарь растёт бесконтрольно).
    MEM_MAX_ENTRIES = 512

    def __init__(self) -> None:
        self._bg: set[str] = set()
        self._guard = threading.Lock()
        self._local = LocalSingleFlight(wait_timeout=LOCAL_WAIT_TIMEOUT)
        self._redis_sf: Optional[RedisSingleFlight] = None
        self._cache: Optional[RedisEnvelopeCache] = None
        self._mem: dict[str, tuple[float, Any]] = {}

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #
    def get(
        self,
        key: str,
        ttl: int,
        compute: Callable[[], Any],
        max_age: int | None = None,
    ) -> Any:
        """Вернуть результат (кэш или вычисление) по SWR-логике."""
        max_age = max_age or ttl * 2
        cache = self._envelope_cache()
        if cache is None:
            # Redis недоступен — это не повод отключать кэш: именно в такой момент
            # каждый запрос уходит в живой апстрим. Падаем на внутрипроцессный слой.
            return self._get_local(key, ttl, compute, max_age)

        now = time.time()
        envelope = cache.read(key)
        if envelope is not None and not envelope.is_legacy():
            if envelope.is_fresh(now):
                return envelope.value
            if envelope.is_usable_stale(max_age, now):
                # Stale-While-Revalidate: отдаём устаревшее, обновляем в фоне
                self._refresh_async(key, ttl, compute)
                return envelope.value
        # Нет значения, запись старой схемы или слишком старый → пересчёт под single-flight
        stale = envelope.value if envelope is not None else None
        return self._local.run(
            key,
            lambda: self._recompute(key, ttl, compute),
            on_miss=lambda: self._on_wait_timeout(key, ttl, compute, stale),
        )

    def invalidate(self, key: str) -> None:
        """Сбросить кэш ключа (например, после ручного обновления данных)."""
        cache = self._envelope_cache()
        if cache is not None:
            cache.invalidate(key)
        with self._guard:
            self._mem.pop(key, None)

    # ------------------------------------------------------------------ #
    #  Внутрипроцессный слой (работает без Redis)
    # ------------------------------------------------------------------ #
    def _mem_read(self, key: str) -> Optional[tuple[float, Any]]:
        with self._guard:
            return self._mem.get(key)

    def _mem_write(self, key: str, value: Any) -> None:
        with self._guard:
            self._mem[key] = (time.time(), value)
            if len(self._mem) <= self.MEM_MAX_ENTRIES:
                return
            # Вытесняем самые старые записи: кэш не должен расти без границы.
            oldest = sorted(self._mem.items(), key=lambda kv: kv[1][0])
            for stale_key, _ in oldest[: len(self._mem) - self.MEM_MAX_ENTRIES]:
                self._mem.pop(stale_key, None)

    def _recompute_local(self, key: str, ttl: int, compute: Callable[[], object]) -> object:
        """Пересчёт без Redis: считаем и кладём в память процесса."""
        value = compute()
        if value is not None:
            self._mem_write(key, value)
        return value

    def _get_local(
        self, key: str, ttl: int, compute: Callable[[], object], max_age: int
    ) -> object:
        """SWR по памяти процесса: тот же порядок предпочтений, что и у Redis-слоя."""
        now = time.time()
        hit = self._mem_read(key)
        if hit is not None:
            ts, value = hit
            age = now - ts
            if age <= ttl:
                return value
            if age <= max_age:
                # Отдаём устаревшее сразу, обновляем в фоне — как в Redis-слое.
                self._refresh_async(key, ttl, compute)
                return value
        stale = hit[1] if hit is not None else None
        return self._local.run(
            key,
            lambda: self._recompute_local(key, ttl, compute),
            on_miss=lambda: self._on_local_wait_timeout(key, ttl, compute, stale),
        )

    def _on_local_wait_timeout(
        self, key: str, ttl: int, compute: Callable[[], object], stale: object
    ) -> object:
        """Лидер не ответил: stale → пересчёт (тот же порядок, что в Redis-слое)."""
        if stale is not None:
            logger.warning("Кэш %s: лидер не ответил — отдаю устаревшее значение", key)
            return stale
        logger.warning("Кэш %s: лидер не ответил и устаревшего значения нет — считаю сам", key)
        return self._recompute_local(key, ttl, compute)

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _envelope_cache(self) -> Optional[RedisEnvelopeCache]:
        """Слой конвертов; ``None``, если Redis недоступен (кэш выключается целиком)."""
        redis = get_redis()
        if redis is None or not redis.connected:
            return None
        if self._cache is None:
            self._cache = RedisEnvelopeCache(redis)
            self._redis_sf = RedisSingleFlight(
                redis, lease_ms=REDIS_LEASE_MS, wait_ms=REDIS_WAIT_MS
            )
        return self._cache

    def _read_fresh(self, key: str) -> Callable[[], object]:
        """Предикат для single-flight: «в кэше есть годное значение».

        Ключ передаётся аргументом (а не читается из поля объекта): ``_read_fresh``
        вызывается из разных потоков на разные ключи, и общее изменяемое поле здесь
        означало бы, что один поток читает кэш другого.
        """
        cache = self._envelope_cache()
        if cache is None:
            return lambda: None

        def read() -> object:
            envelope = cache.read(key)
            if envelope is None or envelope.is_legacy() or not envelope.is_fresh():
                return None
            return envelope.value

        return read

    def _recompute(self, key: str, ttl: int, compute: Callable[[], object]) -> object:
        """Лидер процесса: кросс-процессная аренда → двойная проверка кэша → счёт → запись.

        Если мы были follower'ом и чужой лидер не отдал ни значения, ни годного stale
        (:meth:`RedisSingleFlight.run` вернул ``None``), считаем сами — **последним средством**,
        ровно как в :meth:`_on_wait_timeout`. Иначе ``None`` дошёл бы до FastAPI и дал
        ``ResponseValidationError`` → 500. Порядок предпочтений неизменен: stale → расчёт.
        """
        cache = self._envelope_cache()
        if cache is None:
            return compute()

        # Запускался ли **наш** loader: легитимно-None результат пересчитывать нельзя (иначе
        # получилось бы двойное вычисление), а вот «нас даже не спросили» — надо.
        ran: dict[str, bool] = {"loader": False}

        def loader() -> object:
            ran["loader"] = True
            value = compute()
            cache.write(key, value, ttl)
            return value

        read = self._read_fresh(key)
        sf = self._redis_sf
        if sf is None:
            return loader()
        result = sf.run(key, loader, read=read, is_fresh=lambda v: v is not None)
        if result is None and not ran["loader"]:
            logger.warning(
                "Кэш %s: лидер не ответил и устаревшего значения нет — считаю сам", key
            )
            return loader()
        return result

    def _on_wait_timeout(
        self, key: str, ttl: int, compute: Callable[[], object], stale: object
    ) -> object:
        """Лидер не ответил за отведённое время.

        Порядок предпочтений: устаревшее значение → пересчёт. Считать **последним**
        средством, а не первым: до этого места доходят только когда лидер пропал,
        и дублирование дорогой работы под нагрузкой — это то, от чего single-flight
        и защищает.
        """
        if stale is not None:
            logger.warning("Кэш %s: лидер не ответил — отдаю устаревшее значение", key)
            return stale
        logger.warning("Кэш %s: лидер не ответил и устаревшего значения нет — считаю сам", key)
        return self._recompute(key, ttl, compute)

    def _refresh_async(self, key: str, ttl: int, compute: Callable[[], object]) -> None:
        """Фоновый пересчёт: один поток на ключ."""
        with self._guard:
            if key in self._bg:
                return
            self._bg.add(key)
        threading.Thread(
            target=self._bg_worker,
            args=(key, ttl, compute),
            daemon=True,
            name=f"swr-{key[-24:]}",
        ).start()

    def _bg_worker(self, key: str, ttl: int, compute: Callable[[], object]) -> None:
        try:
            if self._envelope_cache() is None:
                self._recompute_local(key, ttl, compute)
            else:
                self._recompute(key, ttl, compute)
            logger.debug("SWR refresh done: %s", key)
        except Exception as exc:
            logger.warning("SWR refresh failed %s: %s", key, exc)
        finally:
            with self._guard:
                self._bg.discard(key)


# Единый инстанс для всего приложения
result_cache = ResultCache()
