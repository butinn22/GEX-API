"""AsyncBridge: один постоянный loop вместо ``asyncio.run`` на каждый вызов.

Главная проверка — не «корутина выполнилась», а **сохранность объектов, привязанных к loop**:
именно они и умирали от ``asyncio.run`` (аудит 05: F-10). Тесты идут на чистом asyncio.

    python tests/test_async_bridge.py
    pytest tests/test_async_bridge.py -q
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex.adapters.transport.loop import (  # noqa: E402
    AsyncBridge,
    AsyncBridgeError,
    BridgeTimeoutError,
)


def test_coroutine_runs_and_returns_value():
    bridge = AsyncBridge()
    try:
        async def _coro():
            await asyncio.sleep(0)
            return 42

        assert bridge.run(_coro()) == 42
    finally:
        bridge.shutdown()


def test_loop_survives_between_calls():
    """Ключевой тест: loop один и тот же, а объекты на нём остаются живыми."""
    bridge = AsyncBridge()
    try:
        async def _remember():
            # Объект, привязанный к loop: с asyncio.run он умер бы вместе с ним.
            lock = asyncio.Lock()
            _remember.lock = lock
            return id(asyncio.get_running_loop())

        first_loop = bridge.run(_remember())
        assert bridge.run(_remember()) == first_loop, "loop пересоздаётся между вызовами"

        async def _use_lock():
            async with _remember.lock:
                return "locked"

        assert bridge.run(_use_lock()) == "locked", (
            "объект, созданный в прошлом вызове, должен остаться рабочим"
        )
    finally:
        bridge.shutdown()


def test_concurrent_calls_from_threads():
    bridge = AsyncBridge()
    try:
        results = {}

        async def _work(n: int):
            await asyncio.sleep(0.01)
            return n * 2

        def _worker(n: int):
            results[n] = bridge.run(_work(n))

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == {i: i * 2 for i in range(8)}
    finally:
        bridge.shutdown()


def test_exception_propagates():
    bridge = AsyncBridge()
    try:
        async def _boom():
            raise ValueError("boom")

        try:
            bridge.run(_boom())
        except ValueError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("исключение из корутины потеряно")
    finally:
        bridge.shutdown()


def test_timeout_is_enforced():
    """Зависшая корутина не должна навсегда занимать вызывающий поток."""
    bridge = AsyncBridge()
    try:
        async def _hang():
            await asyncio.sleep(5)

        started = time.monotonic()
        try:
            bridge.run(_hang(), timeout=0.3)
        except BridgeTimeoutError:
            elapsed = time.monotonic() - started
        else:
            raise AssertionError("ожидался BridgeTimeoutError")

        assert elapsed < 2.0, f"ждали {elapsed:.2f} с при таймауте 0.3 с"
    finally:
        bridge.shutdown()


def _expect_bridge_error(func):
    """Требует :class:`AsyncBridgeError`. Помощник — чтобы не плодить ``except: pass``
    (в проекте это метрика «глушение ошибок»)."""
    try:
        func()
    except AsyncBridgeError:
        return
    raise AssertionError("ожидался AsyncBridgeError")


def test_run_from_running_loop_is_rejected():
    """Из async-контекста ``run`` блокировал бы loop — это обязано падать явно."""
    bridge = AsyncBridge()
    try:
        async def _outer():
            async def _inner():
                return 1

            bridge.run(_inner())

        _expect_bridge_error(lambda: bridge.run(_outer()))
    finally:
        bridge.shutdown()


def test_background_call_does_not_block():
    bridge = AsyncBridge()
    try:
        async def _slow():
            await asyncio.sleep(0.3)
            return "done"

        started = time.monotonic()
        future = bridge.run_in_background(_slow())
        assert time.monotonic() - started < 0.1, "run_in_background не должен ждать"
        assert future.result(5) == "done"
    finally:
        bridge.shutdown()


def test_shutdown_is_idempotent_and_restartable():
    bridge = AsyncBridge()
    bridge.start()
    assert bridge.is_running
    bridge.shutdown()
    bridge.shutdown()  # повторный вызов не должен падать
    assert not bridge.is_running

    async def _coro():
        return "again"

    assert bridge.run(_coro()) == "again", "после остановки мост обязан запуститься заново"
    bridge.shutdown()


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
    print(f"--- async bridge: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
