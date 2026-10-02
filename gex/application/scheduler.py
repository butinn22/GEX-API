"""Планировщик фоновых задач (scheduled producer).

Публикует ``FetchTask`` в очередь (Redis Streams за портом) на регулярной основе, чтобы данные
всегда были свежими в Redis. Rate limiting осуществляется на стороне
consumer (``background_fetcher``), scheduler только ставит задачи.

Расписание (интервалы между одинаковыми типами задач):
  * OHLCV (US/индексы) — каждые 5 минут
  * OHLCV (MOEX, commodity) — каждые 15 минут
  * Option chains (US) — каждые 15 минут
  * Option chains (крипта) — каждые 20 минут
  * Option chains (MOEX) — каждые 30 минут
  * Vol индикаторы — каждые 15 минут
  * GEX профили (live) — каждые 30 минут
"""
from __future__ import annotations

import logging
import threading

from gex.adapters.cache.redis_client import get_redis
from gex.application.jobs import FetchTask
from gex.application.prewarm import PrewarmPlan, PrewarmSlot, PrewarmWorker
from gex.adapters.cache.lease import RedisLease
from gex.application.queue import TaskPublisher
from gex.deps import get_task_queue

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Конфигурация расписания
# ====================================================================== #
_SCHEDULE: dict[str, int] = {
    # Графики дашборда: 5 минут — компромисс между «данные не выглядят застывшими»
    # (карточки синхронизируются клиентским авто-рефрешем) и нагрузкой на yfinance
    # (18 тикеров x 2 запроса за тик ≈ 7 запросов/мин). MOEX/commodity реже —
    # ISS и товарные ряды не требуют высокой частоты, а ISS чувствителен к нагрузке.
    "ohlcv_us": 5 * 60,        # 5 минут
    "ohlcv_moex": 15 * 60,     # 15 минут
    "ohlcv_commodity": 15 * 60, # 15 минут
    "chain_yfinance": 15 * 60,  # 15 минут
    "chain_bybit": 20 * 60,    # 20 минут
    "chain_moex": 30 * 60,     # 30 минут
    "vol": 15 * 60,            # 15 минут
    "gex_profile": 30 * 60,    # 30 минут
    "chain_webull": 15 * 60,   # 15 минут
    "sector": 30 * 60,         # 30 минут
    "breadth": 30 * 60,        # 30 минут
    "composite": 30 * 60,      # 30 минут
}

# Фиксированные МСК-слоты (для /breadth-imoex — полный парсинг ISS только по расписанию,
# чтобы не попасть в бан) живут в домене: ``gex.domain.schedule.PERIODIC_SLOTS``. Оттуда их
# берут оба потребителя — слот прогрева (``fixed_msk`` у ``PrewarmSlot``) и расписание
# расписание фоновой очереди (``gex/workers``). Копии здесь нет намеренно: вторая таблица
# означала бы два разных «23:00».


# ====================================================================== #
#  План прогрева
# ====================================================================== #
def build_prewarm_plan(multiplier: float = 1.0) -> PrewarmPlan:
    """Объявленное расписание прогрева: слот → интервал → что публиковать.

    Раньше расписание жило в двух местах: интервалы в ``_SCHEDULE``, а списки тикеров —
    прямо в цикле, по одному `_publish_tasks` на слот. Здесь они сведены в объявления,
    поэтому расписание можно напечатать, проверить на полноту и передать воркеру.
    """
    from gex.application.background_fetcher import (
        COMMODITY_TICKERS_BG,
        CRYPTO_TICKERS,
        MOEX_TICKERS,
        OHLCV_ONLY_TICKERS,
        US_TICKERS,
    )

    def interval(name: str) -> int:
        return int(_SCHEDULE.get(name, 600) * multiplier)

    return PrewarmPlan(slots=(
        PrewarmSlot(
            "ohlcv_us", interval("ohlcv_us"),
            lambda: [FetchTask("ohlcv", "yfinance", t, priority=0)
                     for t in US_TICKERS + CRYPTO_TICKERS + OHLCV_ONLY_TICKERS],
            "Свечи US-акций и крипты через yfinance",
        ),
        PrewarmSlot(
            "ohlcv_moex", interval("ohlcv_moex"),
            lambda: [FetchTask("ohlcv", "moex_iss", t, priority=0) for t in MOEX_TICKERS],
            "Свечи MOEX через ISS",
        ),
        PrewarmSlot(
            "ohlcv_commodity", interval("ohlcv_commodity"),
            lambda: [FetchTask("ohlcv", "yfinance", t, priority=0) for t in COMMODITY_TICKERS_BG],
            "Свечи товарных фьючерсов через yfinance",
        ),
        PrewarmSlot(
            "chain_webull", interval("chain_webull"),
            lambda: [FetchTask("chain", "webull", t, {"max_expiries": 5}, priority=0)
                     for t in US_TICKERS],
            "Опционные цепочки US через Webull (реальный OI)",
        ),
        PrewarmSlot(
            "chain_yfinance", interval("chain_yfinance"),
            lambda: [FetchTask("chain", "yfinance", t, {"max_expiries": 5}, priority=0)
                     for t in US_TICKERS],
            "Опционные цепочки US через yfinance (фолбэк)",
        ),
        PrewarmSlot(
            "chain_bybit", interval("chain_bybit"),
            lambda: [FetchTask("chain", "bybit", t, {"max_expiries": 3}, priority=0)
                     for t in CRYPTO_TICKERS],
            "Опционные цепочки крипты через Bybit",
        ),
        PrewarmSlot(
            "chain_moex", interval("chain_moex"),
            lambda: [FetchTask("chain", "moex_iss", t, {"max_expiries": 3}, priority=0)
                     for t in MOEX_TICKERS],
            "Опционные цепочки MOEX через ISS",
        ),
        PrewarmSlot(
            "vol", interval("vol"),
            lambda: [FetchTask("vol", "yfinance", "ALL", priority=0)],
            "Индикаторы волатильности (VIX/PCR)",
        ),
        PrewarmSlot(
            "gex_profile", interval("gex_profile"),
            lambda: [FetchTask("gex_profile", "yfinance", t, {"days": 30}, priority=0)
                     for t in US_TICKERS[:5]],
            "GEX-профили топ-5 US (живой расчёт)",
        ),
        PrewarmSlot(
            "sector", interval("sector"),
            lambda: [FetchTask("sector", "yfinance", "ALL", priority=0)],
            "Секторная широта рынка",
        ),
        PrewarmSlot(
            "breadth", interval("breadth"),
            lambda: [FetchTask("breadth", "yfinance", "ALL", priority=0)],
            "Широта рынка (breadth)",
        ),
        PrewarmSlot(
            "composite", interval("composite"),
            lambda: [FetchTask("composite", "yfinance", "ALL", priority=0)],
            "Композит товарного рынка",
        ),
        # Широта MOEX (IMOEX) здесь БОЛЬШЕ НЕ ПЛАНИРУЕТСЯ.
        #
        # Слоты 23:00/08:00 МСК переехали в расписание фоновой очереди (``gex/workers``,
        # расписание — из таблицы домена ``PERIODIC_SLOTS``). Две причины:
        #   * суточная квота ISS — 3 обращения; два планировщика (потоковый и Beat)
        #     означали бы четыре парсинга в сутки, и один из них гарантированно получал бы
        #     отказ, оставляя страницу на данных предыдущего дня;
        #   * расписание фоновых расчётов теперь одно на кластер, и его видно в одном месте.
        # Окно вхождения слота по-прежнему считает домен, а ``fixed_msk`` остаётся
        # возможностью слота — ею пользуется тест ``test_prewarm``.
    ))


class Scheduler:
    """Планировщик превентивного фетчинга.

    В отдельном потоке публикует задачи в очередь по расписанию.

    Parameters
    ----------
    queue : TaskPublisher
        Redis-очередь для публикации задач.
    interval_multiplier : float
        Множитель интервалов (1.0 = нормально, 2.0 = вдвое реже).
    """

    def __init__(
        self,
        queue: TaskPublisher | None = None,
        interval_multiplier: float = 1.0,
    ):
        self._queue = queue or get_task_queue()
        self._multiplier = float(interval_multiplier)
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_event = threading.Event()
        self._plan = build_prewarm_plan(self._multiplier)
        # Аренда — то, чего не хватало: маркер `GET`+`SET` не атомарен, поэтому слот
        # публиковался каждой репликой сразу после старта.
        self._worker = PrewarmWorker(
            self._plan, self._queue, self._lease(), max_slots_per_tick=4
        )

    def _lease(self) -> RedisLease | None:
        """Аренда поверх Redis; при недоступном Redis — ``None`` (единственный процесс)."""
        try:
            redis = get_redis()
            if redis is None or not redis.connected:
                return None
            return RedisLease(redis)
        except Exception as exc:  # noqa: BLE001 — прогрев не должен падать без Redis
            logger.warning("Аренда прогрева недоступна: %s", exc)
            return None

    @property
    def plan(self) -> PrewarmPlan:
        """Объявленное расписание (для админки и проверок полноты)."""
        return self._plan

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Запустить планировщик в фоновом потоке."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="gex-scheduler",
        )
        self._thread.start()
        logger.info("Scheduler started (multiplier=%.1f)", self._multiplier)

    def stop(self) -> None:
        """Остановить планировщик."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    #  Публикация
    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        """Главный цикл: воркер сам решает, каким слотам пора.

        Решение «пора / не пора» больше не размазано по циклу: интервалы, аренда и
        ограничение на число слотов за тик живут в :class:`PrewarmWorker`. Цикл здесь
        только про «когда просыпаться».
        """
        logger.info("Scheduler loop started: слотов в плане %d", len(self._plan.slots))
        while self._running and not self._stop_event.is_set():
            try:
                fired = self._worker.run_once()
                if fired:
                    logger.info("Scheduler: выполнены слоты: %s", ", ".join(fired))
            except Exception as exc:  # noqa: BLE001 — цикл обязан выживать
                logger.error("Scheduler loop error: %s", exc)
            self._stop_event.wait(30)

    def run_once(self) -> list[str]:
        """Разовый тик (админка/тесты): какие слоты выполнились."""
        return self._worker.run_once()

    def describe(self) -> dict:
        """Состояние планировщика: план, что выполнено, что пропущено.

        ВНИМАНИЕ: ``slots`` обязан остаться словарём «имя слота → интервал», потому
        что именно так его читает админка (``auth/admin/system.py``). Раньше словарь
        затирался распаковкой ``**self._worker.describe()``, где под тем же ключом
        лежит **список имён** — данные об интервалах молча исчезали, а админка падала
        с ``AttributeError: 'list' object has no attribute 'items'``. Поэтому список
        имён выносится под отдельный ключ, а не распаковкой поверх.
        """
        worker = self._worker.describe()
        problems = self._plan.validate()
        return {
            "running": self._running,
            "multiplier": self._multiplier,
            "slots": self._plan.intervals(),
            "slot_names": worker.pop("slots", []),
            "problems": problems,
            **worker,
        }


# ====================================================================== #
#  Глобальный инстанс (для админ-панели)
# ====================================================================== #
_scheduler_instance: "Scheduler | None" = None


def set_scheduler(scheduler: "Scheduler | None") -> None:
    """Зарегистрировать глобальный инстанс планировщика (вызывается в main)."""
    global _scheduler_instance
    _scheduler_instance = scheduler


def get_scheduler() -> "Scheduler | None":
    """Получить глобальный инстанс планировщика (может быть None)."""
    return _scheduler_instance


# ====================================================================== #
#  API ручка: ручной триггер превентивного фетчинга
# ====================================================================== #
def trigger_prewarm(queue: TaskPublisher | None = None) -> dict:
    """Вручную запустить превентивный fetch всех инструментов.

    Returns
    -------
    dict
        ``{"published": N, "tasks": [...]}``
    """
    from gex.application.background_fetcher import build_prewarm_tasks

    q = queue or get_task_queue()
    tasks = build_prewarm_tasks(priority=1)
    ok = q.publish_many(tasks)
    return {
        "published": ok,
        "total": len(tasks),
        "task_types": list(set(t.task_type for t in tasks)),
    }
