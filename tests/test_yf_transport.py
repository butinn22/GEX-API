"""Deadline-обёртка yfinance: зависший источник обязан отпустить поток.

Тесты идут без yfinance и без сети: тикер и ``download`` подставляются заглушками.
Проверяется именно механика дедлайна — «перестали ждать» или «зависли вместе с ним».

    python tests/test_yf_transport.py
    pytest tests/test_yf_transport.py -q
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex.adapters.transport.yf_transport import (  # noqa: E402
    DeadlinePolicy,
    DeadlineStats,
    YfDeadlineError,
    YfGate,
    YfTransport,
    run_with_deadline,
)


class _FakeTicker:
    """Тикер-заглушка: каждый I/O-метод спит заданное время."""

    def __init__(self, symbol: str, delay: float = 0.0, payload=None):
        self.symbol = symbol
        self.delay = delay
        self.payload = payload if payload is not None else f"data:{symbol}"

    def _wait(self):
        if self.delay:
            time.sleep(self.delay)
        return self.payload

    def history(self, **kwargs):
        return self._wait()

    def option_chain(self, date):
        return self._wait()

    @property
    def fast_info(self):
        return self._wait()

    @property
    def info(self):
        return self._wait()

    @property
    def options(self):
        return self._wait()


def _expect_deadline(func):
    """Требует, чтобы ``func`` упал с :class:`YfDeadlineError`.

    Вынесено в помощник не только ради краткости: в проекте метрика «глушение ошибок»
    считает ``except: pass``, и три одинаковых блока в тестах портили бы бейзлайн.
    """
    try:
        func()
    except YfDeadlineError:
        return
    raise AssertionError("ожидалась YfDeadlineError — источник не должен был отвечать")


def _transport(delay: float = 0.0, **policy_kw) -> YfTransport:
    policy = DeadlinePolicy(**policy_kw) if policy_kw else DeadlinePolicy(default_seconds=0.3)
    return YfTransport(
        policy=policy,
        ticker_factory=lambda symbol: _FakeTicker(symbol, delay=delay),
        downloader=lambda *a, **kw: "downloaded",
    )


def test_fast_call_returns_value():
    transport = _transport(delay=0.0)
    assert transport.history("AAPL") == "data:AAPL"


def test_hanging_source_raises_deadline_error():
    """Главный тест итерации 23: источник висит — мы поднимаем ошибку, а не ждём."""
    transport = _transport(delay=5.0)

    started = time.monotonic()

    def _call():
        try:
            transport.history("AAPL")
        except YfDeadlineError as exc:
            assert "AAPL" in str(exc)
            raise
        raise AssertionError("источник не должен был ответить")

    _expect_deadline(_call)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"ждали {elapsed:.2f} с при дедлайне 0.3 с — дедлайн не работает"


def test_deadline_does_not_block_caller_forever():
    """После дедлайна поток вызванного кода обязан быть свободен (это и есть смысл обёртки)."""
    transport = _transport(delay=3.0)
    # Если дедлайн не сработал, тест зависнет здесь на 3 с — это и есть проверяемое поведение.
    _expect_deadline(lambda: transport.info("SPY"))


def test_download_uses_heavier_deadline():
    """Массовая загрузка получает свой (увеличенный) дедлайн, а не общий."""
    policy = DeadlinePolicy(default_seconds=0.3, download_seconds=0.9)

    def _slow(*args, **kwargs):
        time.sleep(0.6)
        return "ok"

    transport = YfTransport(policy=policy, downloader=_slow)
    # 0.6 с > default (0.3), но < download (0.9) → должно успеть
    assert transport.download("AAPL", "MSFT") == "ok"


def test_explicit_seconds_are_capped_by_policy():
    """«Подожди час» — не дедлайн, а его отсутствие; потолок обязан работать."""
    policy = DeadlinePolicy(default_seconds=0.2, download_seconds=0.4, max_seconds=0.5)
    transport = YfTransport(policy=policy, ticker_factory=lambda s: _FakeTicker(s, delay=3.0))

    started = time.monotonic()
    _expect_deadline(lambda: transport.history("AAPL", seconds=3600))
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"потолок не сработал: ждали {elapsed:.2f} с"


def test_exception_from_source_propagates():
    """Ошибка источника обязана дойти до вызывающего в исходном виде."""

    def _boom():
        raise ValueError("bad ticker")

    try:
        run_with_deadline(_boom, seconds=1.0, name="boom")
    except ValueError as exc:
        assert str(exc) == "bad ticker"
    else:
        raise AssertionError("исключение из рабочего потока потеряно")


def test_abandoned_calls_are_observable():
    """Брошенные вызовы видны в статистике: иначе деградация снова окажется незаметной."""
    stats = DeadlineStats()
    transport = YfTransport(
        policy=DeadlinePolicy(default_seconds=0.2),
        ticker_factory=lambda s: _FakeTicker(s, delay=0.6),
        stats=stats,
    )

    _expect_deadline(lambda: transport.history("AAPL"))

    snapshot = stats.as_dict()
    assert snapshot["calls"] == 1
    assert snapshot["timeouts"] == 1
    assert snapshot["abandoned"] == 1, "пока брошенный поток работает, он должен быть виден"

    # Когда поток дойдёт до конца, счётчик висящих обязан уменьшиться.
    deadline = time.monotonic() + 5
    while stats.as_dict()["abandoned"] > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stats.as_dict()["abandoned"] == 0
    assert stats.as_dict()["completed"] == 1


def test_stats_are_thread_safe():
    stats = DeadlineStats()

    def _bump():
        for _ in range(200):
            stats.note_start()
            stats.note_completed(was_abandoned=False)

    threads = [threading.Thread(target=_bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert stats.as_dict()["calls"] == 800
    assert stats.as_dict()["completed"] == 800


def test_policy_validates_bounds():
    for bad in ({"default_seconds": 0}, {"default_seconds": 500.0}, {"download_seconds": -1}):
        try:
            DeadlinePolicy(**bad)
        except ValueError:
            continue
        raise AssertionError(f"ожидалась ошибка валидации для {bad}")


def test_deadline_error_is_runtime_error():
    """Совместимость: вызывающий код исторически ловил ``RuntimeError``."""
    assert issubclass(YfDeadlineError, RuntimeError)


# ====================================================================== #
#  Ограничение параллелизма (инцидент 2026-09-21: 6533 треда в процессе)
# ====================================================================== #
def test_gate_rejects_when_all_slots_are_busy():
    """Слоты заняты → немедленный отказ, а не бесконечная очередь заявок."""
    gate = YfGate(max_concurrency=2, queue_wait=0.05)
    release = threading.Event()
    gate.submit(lambda: release.wait(5))
    gate.submit(lambda: release.wait(5))
    try:
        try:
            gate.submit(lambda: None)
        except YfDeadlineError as exc:
            assert exc.operation == "backpressure"
        else:
            raise AssertionError("ожидался отказ при занятых слотах")
    finally:
        release.set()


def test_gate_keeps_thread_count_bounded_under_meltdown():
    """Главная регрессия: дедлайны не должны порождать поток без счёта.

    Раньше каждый вызов создавал отдельный ``threading.Thread``, а по истечении
    дедлайна поток бросался живым — их копились тысячи. Теперь живых вызовов
    не больше числа слотов.
    """
    gate = YfGate(max_concurrency=4, queue_wait=0.05)
    base = threading.active_count()
    for _ in range(20):
        try:
            run_with_deadline(lambda: time.sleep(4), seconds=0.15, name="slow", gate=gate)
        except YfDeadlineError:
            pass
    time.sleep(0.1)
    spawned = threading.active_count() - base
    assert spawned <= gate.max_concurrency + 2, (
        f"потоков порождено {spawned} при лимите {gate.max_concurrency} — взрыв тредов вернулся"
    )


def test_breaker_opens_after_repeated_timeouts():
    """Апстрим деградировал → новые вызовы отклоняются, пока backlog не стечёт."""
    gate = YfGate(
        max_concurrency=4, queue_wait=0.05, breaker_threshold=2, breaker_cooldown=5.0
    )
    stats = DeadlineStats()
    for _ in range(2):
        try:
            run_with_deadline(lambda: time.sleep(3), seconds=0.1, name="slow",
                              stats=stats, gate=gate)
        except YfDeadlineError:
            pass
    assert gate.is_open, "предохранитель должен открыться после порога брошенных вызовов"
    try:
        run_with_deadline(lambda: "ok", seconds=1.0, name="next", stats=stats, gate=gate)
    except YfDeadlineError as exc:
        assert exc.operation == "breaker"
    else:
        raise AssertionError("ожидался отказ предохранителя, а не обращение к сети")


def test_rejected_calls_are_observable():
    """Отказ без запуска считается отдельно от таймаута."""
    gate = YfGate(max_concurrency=1, queue_wait=0.05)
    stats = DeadlineStats()
    release = threading.Event()
    gate.submit(lambda: release.wait(5))
    try:
        try:
            run_with_deadline(lambda: None, seconds=1.0, name="x", stats=stats, gate=gate)
        except YfDeadlineError:
            pass
        assert stats.rejected == 1, f"rejected={stats.rejected}, ожидался 1"
        assert stats.timeouts == 0, "отказ по backpressure — не таймаут"
    finally:
        release.set()


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
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- yf transport: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
