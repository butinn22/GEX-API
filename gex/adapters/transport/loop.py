"""Мост к asyncio для синхронного кода (кольцо ``adapters``).

Проблема: ``asyncio.run`` на каждый вызов
-----------------------------------------
Синхронным сервисам нужно дёрнуть асинхронный оркестратор, и раньше это делалось так::

    def _run(coro):
        return asyncio.run(coro)      # gex/orchestrator/sync_gateway.py:33

``asyncio.run`` **создаёт новый event loop и закрывает его по завершении**. Последствия:

1. объекты, привязанные к loop, умирают вместе с ним: ``httpx.AsyncClient``, асинхронный
   Redis, ``asyncio.Lock``, любые созданные таски — следующий вызов получает «Event loop is
   closed» или «attached to a different loop»;
2. каждый вызов платит за создание и разрушение loop;
3. фоновые задачи, запущенные внутри, обрываются на полуслове (аудит 05: F-10).

Решение: ``AsyncBridge``
------------------------
Один постоянный loop в отдельном daemon-потоке на процесс. Синхронный код отдаёт в него
корутину и ждёт результат — как раньше, но loop живёт между вызовами, поэтому loop-bound
клиенты остаются валидными.

Чего здесь нет: очередей, планировщика и ретраев. Ровно одно — «выполнить корутину из
синхронного контекста на постоянном loop».
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Coroutine, Optional

__all__ = [
    "AsyncBridgeError",
    "BridgeTimeoutError",
    "AsyncBridge",
    "get_shared_bridge",
    "reset_shared_bridge",
]

log = logging.getLogger(__name__)


class AsyncBridgeError(RuntimeError):
    """Мост не может выполнить корутину (не запущен, вызван из async-контекста)."""


class BridgeTimeoutError(TimeoutError):
    """Корутина не уложилась в отведённое время."""


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Текущий работающий loop или ``None`` (``get_running_loop`` бросает, а не возвращает)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class AsyncBridge:
    """Постоянный event loop в фоновом потоке + синхронная точка входа в него.

    Потокобезопасен: несколько рабочих потоков могут одновременно отдавать корутины,
    диспетчеризация идёт через ``run_coroutine_threadsafe``.
    """

    def __init__(self, *, thread_name: str = "gex-async-bridge", timeout: Optional[float] = None):
        self._thread_name = thread_name
        self._default_timeout = timeout
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._closing = False

    # ── жизненный цикл ──────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        loop = self._loop
        return loop is not None and loop.is_running()

    def start(self) -> asyncio.AbstractEventLoop:
        """Запускает фоновый поток с loop (повторный вызов ничего не делает)."""
        if self.is_running and self._loop is not None:
            return self._loop

        with self._lock:
            if self._loop is not None and not self._closing:
                return self._loop

            self._closing = False
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                try:
                    loop.run_forever()
                finally:
                    # Не даём «Event loop is closed» всплыть в чужих потоках при остановке.
                    try:
                        loop.close()
                    except Exception:  # noqa: BLE001 — остановка не должна падать
                        log.debug("ошибка закрытия loop моста", exc_info=True)

            self._thread = threading.Thread(target=_runner, name=self._thread_name, daemon=True)
            self._thread.start()
            ready.wait(10)
            self._loop = loop
            log.info("AsyncBridge запущен (поток %s)", self._thread_name)
            return loop

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Останавливает loop и поток. После этого мост можно запустить заново."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._closing = True
            self._loop = None
            self._thread = None

        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout)
        log.info("AsyncBridge остановлен")

    # ── выполнение ──────────────────────────────────────────────────────────

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._closing:
            return self.start()
        return self._loop

    def submit(self, coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
        """Ставит корутину в loop и сразу возвращает ``Future`` (не дожидаясь результата)."""
        loop = self.loop
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def run(self, coro: Coroutine[Any, Any, Any], *, timeout: Optional[float] = None) -> Any:
        """Выполняет корутину и возвращает результат (блокирует вызывающий поток).

        ``timeout`` ограничивает ожидание — по тем же причинам, что и дедлайн yfinance:
        зависшая корутина не должна навсегда занимать рабочий поток.
        """
        if _running_loop() is not None:
            # Закрываем корутину сами: иначе она останется не await'нутой и Python
            # предупредит «coroutine was never awaited» — шум, который маскирует реальные утечки.
            coro.close()
            raise AsyncBridgeError(
                "AsyncBridge.run вызван из работающего event loop — это заблокировало бы его. "
                "В async-коде awaiting'ите корутину напрямую."
            )

        future = self.submit(coro)
        limit = timeout if timeout is not None else self._default_timeout
        try:
            return future.result(limit)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise BridgeTimeoutError(f"корутина не завершилась за {limit} с") from exc

    def run_in_background(
        self, coro: Coroutine[Any, Any, Any], *, on_error=None
    ) -> concurrent.futures.Future:
        """Запускает корутину и не ждёт её — для фоновой ревалидации (SWR)."""
        future = self.submit(coro)
        if on_error is not None:
            future.add_done_callback(lambda f: None if f.cancelled() or not f.exception() else on_error(f.exception()))
        return future


_lock = threading.Lock()
_shared: Optional[AsyncBridge] = None


def get_shared_bridge() -> AsyncBridge:
    """Ленивый общий мост: один loop на процесс вместо ``asyncio.run`` на каждый вызов."""
    global _shared
    if _shared is None:
        with _lock:
            if _shared is None:
                _shared = AsyncBridge()
    return _shared


def reset_shared_bridge() -> None:
    """Останавливает общий мост (тесты, перезапуск конфигурации)."""
    global _shared
    with _lock:
        bridge, _shared = _shared, None
    if bridge is not None:
        bridge.shutdown()
