"""ConsumerPool: эластичное число воркеров очереди по нагрузке (Фаза 2).

Проверяется логика автопарка: рост при backlog, сокращение при простое, cooldown
против осцилляций, неприкосновенность занятых воркеров и границы min..max.

Порт-заглушка управляет глубиной очереди и «поставками» задач, часы инжектируются —
cooldown проверяется детерминированно, без sleep на реальном времени. При этом
``TaskConsumer`` настоящие: они крутят цикл на заглушке и честно считают
``stats.processed`` — именно по счётчикам пул понимает, кто простаивал.

    python tests/test_consumer_pool.py
    pytest tests/test_consumer_pool.py -q
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.application.consumer_pool import ConsumerPool, PoolConfig, parse_profiles  # noqa: E402
from gex.ports.job_queue import Delivery, Job  # noqa: E402


# ====================================================================== #
#  Инструменты теста
# ====================================================================== #
class FakeClock:
    """Монотонные часы с ручным управлением — cooldown без реального времени."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class QueueState:
    """«Очередь»: глубина (backlog), доступные поставки задач, режим нон-стоп."""

    def __init__(self):
        self.depth = 0
        self.deliveries = 0
        self.endless = False
        self.lock = threading.Lock()


class StubPort:
    """Минимальная очередь с интерфейсом JobQueuePort (без Redis)."""

    def __init__(self, name: str, state: QueueState):
        self.name = name
        self._state = state

    def read(self, *, count: int = 10, block_ms=None):
        with self._state.lock:
            if self._state.endless:
                take = True
            elif self._state.deliveries > 0:
                self._state.deliveries -= 1
                take = True
            else:
                take = False
        if not take:
            return []
        return [self._delivery()]

    def _delivery(self) -> Delivery:
        return Delivery(
            stream="gex:q:ohlcv:bg",
            message_id=f"{self.name}-{time.monotonic_ns()}",
            job=Job(task_type="ohlcv", idempotency_key=f"k-{self.name}", payload={"ticker": "SPY"}),
        )

    def depth(self, queue=None):
        return {"gex:q:ohlcv:bg": self._state.depth}

    def pending(self, queue=None):
        return {}

    def should_process(self, job):
        return True

    def ack(self, delivery):
        return True

    def dead_letter(self, delivery, error):
        return True

    def claim_stale(self, *, min_idle_ms: int = 60_000, count: int = 10):
        return []


def make_pool(
    *, min_workers: int = 1, max_workers: int = 3, autoscale: bool = True,
    high: int = 5, cooldown: float = 10.0,
) -> tuple[ConsumerPool, QueueState, FakeClock]:
    state = QueueState()
    clock = FakeClock()

    pool = ConsumerPool(
        "fast",
        lambda name: StubPort(name, state),
        lambda task: None,
        PoolConfig(
            min_workers=min_workers, max_workers=max_workers, autoscale=autoscale,
            high_watermark=high, check_interval_s=0.05, cooldown_s=cooldown,
        ),
        clock=clock,
    )
    return pool, state, clock


# ====================================================================== #
#  Состав и границы
# ====================================================================== #
def test_starts_with_min_workers():
    pool, _state, _clock = make_pool(min_workers=2, max_workers=4)
    try:
        pool.start(autopilot=False)
        assert pool.size == 2
        assert pool.describe()["workers"] == ["fast-1", "fast-2"]
    finally:
        pool.stop()


def test_scales_up_with_backlog_and_respects_max():
    pool, state, clock = make_pool(min_workers=1, max_workers=3, high=5)
    try:
        pool.start(autopilot=False)
        state.depth = 10
        assert pool.autoscale_once() == "up"
        assert pool.size == 2
        # cooldown: сразу второй раз состав не меняем
        assert pool.autoscale_once() is None
        clock.advance(11)
        assert pool.autoscale_once() == "up"
        assert pool.size == 3
        # потолок: дальше не растём
        clock.advance(11)
        assert pool.autoscale_once() is None
        assert pool.size == 3
    finally:
        pool.stop()


def test_scales_down_when_idle_not_below_min():
    pool, state, clock = make_pool(min_workers=1, max_workers=3, high=5)
    try:
        pool.start(autopilot=False)
        state.depth = 10
        pool.autoscale_once()
        clock.advance(11)
        pool.autoscale_once()
        assert pool.size == 3

        state.depth = 0
        time.sleep(0.05)  # воркеры без поставок уходят в ожидание (дельта processed = 0)
        clock.advance(11)
        assert pool.autoscale_once() == "down"
        assert pool.size == 2
        clock.advance(11)
        assert pool.autoscale_once() == "down"
        assert pool.size == 1
        clock.advance(11)
        assert pool.autoscale_once() is None  # min — предел
        assert pool.size == 1
    finally:
        pool.stop()


def test_cooldown_prevents_flapping():
    pool, state, clock = make_pool(min_workers=1, max_workers=3, high=5)
    try:
        pool.start(autopilot=False)
        state.depth = 10
        assert pool.autoscale_once() == "up"
        state.depth = 0
        assert pool.autoscale_once() is None  # cooldown после изменения
        clock.advance(4)  # меньше cooldown (10 c)
        assert pool.autoscale_once() is None
        assert pool.size == 2
        clock.advance(7)  # суммарно 11 c — можно
        assert pool.autoscale_once() == "down"
        assert pool.size == 1
    finally:
        pool.stop()


def test_busy_workers_are_not_stopped():
    """Пул не сокращается, пока воркеры что-то обрабатывают — работу не прерываем."""
    pool, state, clock = make_pool(min_workers=1, max_workers=3, high=5)
    try:
        pool.start(autopilot=False)
        state.depth = 10
        pool.autoscale_once()
        clock.advance(11)
        pool.autoscale_once()
        assert pool.size == 3

        state.depth = 0
        workers = list(pool._workers)  # noqa: SLF001
        before = {w.name: w.consumer.stats.processed for w in workers}
        state.endless = True  # задачи идут нон-стоп: все воркеры заняты
        # TaskConsumer после пустого чтения уходит в idle-паузу (~1 c) — ждём,
        # пока каждый воркер реально начнёт обрабатывать (дельта processed > 0).
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not all(
            w.consumer.stats.processed > before[w.name] for w in workers
        ):
            time.sleep(0.05)

        clock.advance(11)
        assert pool.autoscale_once() is None
        assert pool.size == 3
    finally:
        pool.stop()


def test_autoscale_off_keeps_min():
    pool, state, _clock = make_pool(min_workers=1, max_workers=3, autoscale=False, high=5)
    try:
        pool.start(autopilot=False)
        state.depth = 100
        assert pool.autoscale_once() is None
        assert pool.size == 1
    finally:
        pool.stop()


def test_stop_releases_all_workers():
    pool, state, _clock = make_pool(min_workers=2, max_workers=3)
    pool.start(autopilot=False)
    state.depth = 10
    pool.autoscale_once()
    consumers = [w.consumer for w in pool._workers]  # noqa: SLF001 — проверяем «дожили ли потоки»
    assert len(consumers) == 3

    pool.stop()
    assert pool.size == 0
    assert all(c._thread is not None and not c._thread.is_alive() for c in consumers)  # noqa: SLF001


# ====================================================================== #
#  Разбор QUEUE_CONSUMERS
# ====================================================================== #
def test_parse_profiles_all_and_csv():
    assert parse_profiles("all") == ("fast", "heavy")
    assert parse_profiles("fast") == ("fast",)
    assert parse_profiles("heavy,fast") == ("heavy", "fast")
    assert parse_profiles(" HEAVY ") == ("heavy",)


def test_parse_profiles_unknown_falls_back_to_all():
    """Опечатка не должна оставить очередь без потребителей."""
    assert parse_profiles("xyz") == ("fast", "heavy")
    assert parse_profiles("fast,junk") == ("fast",)
    assert parse_profiles("") == ("fast", "heavy")


def test_pool_config_rejects_inverted_range():
    for kwargs in ({"min_workers": 3, "max_workers": 1}, {"min_workers": 0, "max_workers": 3}):
        try:
            PoolConfig(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"конфигурация {kwargs} должна быть отклонена")


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- consumer pool: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
