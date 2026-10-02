"""Deadline-обёртка над yfinance (кольцо ``adapters``).

Проблема, которую решает модуль
-------------------------------
``yfinance`` — синхронная библиотека **без параметра timeout**. ``yf.download`` и
``Ticker.history`` уходят в сеть и возвращаются, когда вернутся: если провайдер держит
соединение, поток висит минутами. В приложении это выглядит так: воркер занят, страница
ждёт, следующий запрос встаёт в очередь — и никто не может это прервать, потому что
прерывать нечем.

Python не умеет убивать поток, поэтому честная формулировка: **мы не прерываем вызов,
мы перестаём его ждать**. Вызов уходит в отдельный daemon-поток, основной ждёт ровно
``deadline`` секунд и, если ответа нет, поднимает :class:`YfDeadlineError`. Застрявший
поток остаётся висеть до своего собственного таймаута сокета, но:

* основной поток свободен и может отдать странице последний payload (политика SWR, P5);
* каждый такой случай **виден**: счётчик ``abandoned`` растёт и попадает в метрики
  (принцип P2 — «состояние обязано быть наблюдаемым», иначе деградация снова окажется
  незаметной, как уже было с ``ORCHESTRATOR_ENABLED``).

Чего здесь нет: кэша и ретраев. Это следующие итерации (26, 27).

Итерация «инцидент 2026-09-21»: одного дедлайна оказалось мало. Брошенные потоки
копились (6533 треда в процессе), GIL-конкуренция растягивала живой вызов за дедлайн,
и это рождало новые брошенные потоки — дашборд умирал, а апстрим был здоров (1.6 с
на свежем процессе). Теперь вызовы идут через :class:`YfGate`: ограниченный пул,
backpressure при занятых слотах и предохранитель при деградации апстрима.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "YfDeadlineError",
    "DeadlinePolicy",
    "DeadlineStats",
    "YfGate",
    "run_with_deadline",
    "YfTransport",
    "get_shared_yf_transport",
    "reset_shared_yf_transport",
    "get_shared_gate",
    "reset_shared_gate",
]

log = logging.getLogger(__name__)

#: По умолчанию ждём столько же, сколько транспорт ждёт чтения: дольше одного таймаута
#: пользователь всё равно не будет смотреть на спиннер.
DEFAULT_DEADLINE = 30.0
#: Массовая загрузка (``yf.download`` по сотням тикеров) объективно тяжелее.
DEFAULT_DOWNLOAD_DEADLINE = 60.0
#: Потолок: «подожди час» — это не deadline, а отсутствие deadline.
MAX_DEADLINE = 120.0

#: Сколько вызовов yfinance могут идти одновременно (инцидент 2026-09-21).
#: Раньше каждый вызов порождал отдельный поток, а по истечении дедлайна поток
#: **бросался** — Python не умеет убивать потоки. Брошенные копились: процесс
#: дошёл до 6533 тредов, GIL-конкуренция растягивала штатный вызов 1.6 с за
#: дедлайн 30 с, и это рождало новые брошенные потоки. Теперь число живых
#: вызовов равно числу слотов — взрыв структурно невозможен.
DEFAULT_MAX_CONCURRENCY = 16
#: Сколько ждать свободный слот. Дольше — немедленный отказ (backpressure).
DEFAULT_QUEUE_WAIT = 2.0
#: Сколько брошенных вызовов открывает предохранитель (апстрим деградировал).
DEFAULT_BREAKER_THRESHOLD = 8
#: Как долго предохранитель отклоняет вызовы, давая backlog стечь.
DEFAULT_BREAKER_COOLDOWN = 30.0


class YfDeadlineError(RuntimeError):
    """Источник не ответил за отведённое время.

    Наследник ``RuntimeError``: вызывающий код исторически ловил именно его и уходит
    в fallback (кешированное значение, ``None``, другая страница).
    """

    def __init__(self, message: str, *, seconds: float = 0.0, operation: str = ""):
        super().__init__(message)
        self.seconds = seconds
        self.operation = operation


@dataclass(frozen=True)
class DeadlinePolicy:
    """Сколько ждать yfinance. Единственное место, где эти числа определены."""

    default_seconds: float = DEFAULT_DEADLINE
    download_seconds: float = DEFAULT_DOWNLOAD_DEADLINE
    max_seconds: float = MAX_DEADLINE

    def __post_init__(self) -> None:
        if self.default_seconds <= 0 or self.download_seconds <= 0:
            raise ValueError("deadline должен быть положительным")
        if self.default_seconds > self.max_seconds or self.download_seconds > self.max_seconds:
            raise ValueError(f"deadline превышает потолок {self.max_seconds} с")

    def for_operation(self, seconds: Optional[float], *, heavy: bool = False) -> float:
        """Итоговое значение: явное, если передали; иначе профиль операции; иначе потолок."""
        chosen = seconds if seconds is not None else (
            self.download_seconds if heavy else self.default_seconds
        )
        return max(0.1, min(float(chosen), self.max_seconds))


class DeadlineStats:
    """Наблюдаемость дедлайнов.

    ``abandoned`` — **текущее** число вызовов, которые мы бросили ждать и которые ещё работают.
    Если оно растёт, источник деградировал, и это видно в метриках, а не только в жалобах
    пользователей. Счётчик честно уменьшается, когда брошенный поток всё-таки завершается.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.completed = 0
        self.timeouts = 0
        self.abandoned = 0
        #: Отклонено без запуска: слоты заняты или открыт предохранитель.
        self.rejected = 0

    def note_start(self) -> None:
        with self._lock:
            self.calls += 1

    def note_completed(self, *, was_abandoned: bool) -> None:
        with self._lock:
            self.completed += 1
            if was_abandoned:
                self.abandoned -= 1

    def note_timeout(self) -> None:
        with self._lock:
            self.timeouts += 1
            self.abandoned += 1

    def note_rejected(self) -> None:
        """Вызов не состоялся: backpressure (нет слота) или предохранитель."""
        with self._lock:
            self.rejected += 1

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "completed": self.completed,
                "timeouts": self.timeouts,
                "abandoned": self.abandoned,
                "rejected": self.rejected,
            }


class YfGate:
    """Ограничитель одновременных вызовов yfinance: пул + backpressure + предохранитель.

    Зачем слой поверх пула
    ----------------------
    Инцидент 2026-09-21: ``run_with_deadline`` бросал поток при таймауте, но брошенный
    поток продолжал работать, и их число росло безгранично — 6533 треда. Пул из
    ``max_workers`` слотов делает взрыв структурно невозможным: сколько слотов, столько
    и живых вызовов.

    Но одного пула мало: у ``ThreadPoolExecutor`` неограниченная очередь, поэтому при
    деградации апстрима заявки накапливались бы так же бесконтрольно, только в очереди.
    Слот занимается **до** подачи заявки, а при его отсутствии вызов отклоняется сразу —
    сервис деградирует (быстрый отказ), а не тонет.

    Предохранитель — вторая линия: когда брошенных вызовов становится больше порога,
    новые отклоняются на ``cooldown`` секунд, чтобы backlog успел стечь.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        queue_wait: float = DEFAULT_QUEUE_WAIT,
        breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
        breaker_cooldown: float = DEFAULT_BREAKER_COOLDOWN,
    ) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self._queue_wait = max(0.0, float(queue_wait))
        self._threshold = max(1, int(breaker_threshold))
        self._cooldown = max(0.0, float(breaker_cooldown))
        self._pool = ThreadPoolExecutor(
            max_workers=self.max_concurrency, thread_name_prefix="yf-deadline"
        )
        self._slots = threading.Semaphore(self.max_concurrency)
        self._open_until = 0.0
        self._guard = threading.Lock()

    @property
    def is_open(self) -> bool:
        """Предохранитель открыт → вызовы отклоняются без обращения к сети."""
        return time.monotonic() < self._open_until

    def trip(self, *, abandoned: int) -> None:
        """Открыть предохранитель, если брошенных вызовов больше порога."""
        if abandoned < self._threshold:
            return
        with self._guard:
            if time.monotonic() < self._open_until:
                return
            self._open_until = time.monotonic() + self._cooldown
        log.warning(
            "Предохранитель yfinance открыт на %.0f с: брошенных вызовов %d (порог %d). "
            "Новые вызовы отклоняются без сети, пока backlog не стечёт.",
            self._cooldown, abandoned, self._threshold,
        )

    def submit(
        self, func: Callable[[], Any], *, on_done: Optional[Callable[[Future], None]] = None
    ) -> Future:
        """Занять слот и подать заявку; при отсутствии слота — немедленный отказ."""
        if self.is_open:
            raise YfDeadlineError(
                "предохранитель yfinance открыт (апстрим деградировал)",
                seconds=0.0, operation="breaker",
            )
        if not self._slots.acquire(timeout=self._queue_wait):
            raise YfDeadlineError(
                f"нет свободного слота yfinance (лимит {self.max_concurrency}, "
                f"ожидание {self._queue_wait:.1f} с)",
                seconds=self._queue_wait, operation="backpressure",
            )
        try:
            future = self._pool.submit(func)
        except BaseException:
            self._slots.release()
            raise

        def _release(f: Future) -> None:
            self._slots.release()
            if on_done is not None:
                on_done(f)

        future.add_done_callback(_release)
        return future


def run_with_deadline(
    func: Callable[[], Any],
    *,
    seconds: float,
    name: str = "yf",
    stats: Optional[DeadlineStats] = None,
    gate: Optional[YfGate] = None,
) -> Any:
    """Выполняет ``func`` в потоке пула и ждёт его не дольше ``seconds``.

    Возвращает результат или пробрасывает исключение из рабочего потока. По истечении
    времени — :class:`YfDeadlineError`; рабочий поток остаётся жив, пока не дойдёт до
    своего таймаута сокета (убить поток в Python нельзя).

    Отличие от прежней версии: поток не создаётся на каждый вызов — их ровно
    ``max_concurrency``, а при занятых слотах вызов отклоняется сразу.
    """
    if seconds <= 0:
        raise ValueError("seconds должен быть положительным")

    gate = gate or get_shared_gate()
    if stats is not None:
        stats.note_start()

    box: dict[str, Any] = {"abandoned": False}

    def _on_done(_f: Future) -> None:
        if stats is not None:
            stats.note_completed(was_abandoned=box["abandoned"])

    try:
        future = gate.submit(func, on_done=_on_done)
    except YfDeadlineError:
        # Backpressure/предохранитель: вызова не было, потока не создано.
        if stats is not None:
            stats.note_rejected()
        raise

    started = time.monotonic()
    try:
        return future.result(timeout=seconds)
    except TimeoutError:
        # ``TimeoutError`` мог подняться и внутри func — различаем по реальному времени.
        if time.monotonic() - started < seconds * 0.9:
            raise
        box["abandoned"] = True
        if stats is not None:
            stats.note_timeout()
            gate.trip(abandoned=stats.abandoned)
        raise YfDeadlineError(
            f"{name} не ответил за {seconds:.1f} с", seconds=seconds, operation=name
        )


class YfTransport:
    """Тонкая обёртка над yfinance: те же операции, но с ограничением по времени.

    Зависимости (``ticker_factory``, ``downloader``) подставляются снаружи, поэтому
    логика дедлайна проверяется без установленного yfinance.
    """

    def __init__(
        self,
        *,
        policy: Optional[DeadlinePolicy] = None,
        ticker_factory: Optional[Callable[[str], Any]] = None,
        downloader: Optional[Callable[..., Any]] = None,
        stats: Optional[DeadlineStats] = None,
        gate: Optional[YfGate] = None,
    ):
        self._policy = policy or DeadlinePolicy()
        self._ticker_factory = ticker_factory
        self._downloader = downloader
        self.stats = stats or DeadlineStats()
        self._gate = gate or get_shared_gate()

    @property
    def policy(self) -> DeadlinePolicy:
        return self._policy

    # ── примитив ────────────────────────────────────────────────────────────

    def call(self, func: Callable[[], Any], *, seconds: Optional[float] = None, name: str = "yf") -> Any:
        """Любая операция yfinance под дедлайном (для случаев вне готовых методов ниже)."""
        return run_with_deadline(
            func, seconds=self._policy.for_operation(seconds), name=name,
            stats=self.stats, gate=self._gate,
        )

    # ── операции 1:1 с тем, что реально используется в коде ──────────────────

    def history(self, symbol: str, seconds: Optional[float] = None, **kwargs: Any) -> Any:
        """``yf.Ticker(symbol).history(**kwargs)`` под дедлайном."""
        return self.call(
            lambda: self._ticker(symbol).history(**kwargs),
            seconds=seconds,
            name=f"history({symbol})",
        )

    def fast_info(self, symbol: str, seconds: Optional[float] = None) -> Any:
        """``yf.Ticker(symbol).fast_info`` — lazy-свойство, которое тоже ходит в сеть."""
        return self.call(
            lambda: self._ticker(symbol).fast_info,
            seconds=seconds,
            name=f"fast_info({symbol})",
        )

    def info(self, symbol: str, seconds: Optional[float] = None) -> Any:
        """``yf.Ticker(symbol).info`` — самый медленный источник цены."""
        return self.call(
            lambda: self._ticker(symbol).info,
            seconds=seconds,
            name=f"info({symbol})",
        )

    def options(self, symbol: str, seconds: Optional[float] = None) -> Any:
        """Список дат экспирации."""
        return self.call(
            lambda: self._ticker(symbol).options,
            seconds=seconds,
            name=f"options({symbol})",
        )

    def option_chain(
        self, symbol: str, date: Optional[str] = None, seconds: Optional[float] = None
    ) -> Any:
        """Цепочка опционов на дату экспирации."""
        return self.call(
            lambda: self._ticker(symbol).option_chain(date),
            seconds=seconds,
            name=f"option_chain({symbol},{date})",
        )

    def download(self, *args: Any, seconds: Optional[float] = None, **kwargs: Any) -> Any:
        """``yf.download`` — массовая загрузка, для неё свой (увеличенный) дедлайн."""
        return run_with_deadline(
            lambda: self._downloader_fn()(*args, **kwargs),
            seconds=self._policy.for_operation(seconds, heavy=True),
            name="download",
            stats=self.stats,
            gate=self._gate,
        )

    def ticker(self, symbol: str) -> "DeadlineTicker":
        """Тикер с дедлайном: тот же интерфейс, но I/O не может повесить вызывающий поток.

        Именно этот метод — рекомендуемый путь миграции: он сохраняет **один** экземпляр
        ``Ticker`` (а значит и внутренний кэш yfinance), поэтому правка сводится к замене
        ``yf.Ticker(sym)`` на ``transport.ticker(sym)`` — остальной код не меняется.
        """
        return DeadlineTicker(self._ticker(symbol), self)

    # ── внутренности ────────────────────────────────────────────────────────

    def _ticker(self, symbol: str) -> Any:
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf  # локально: модуль импортируется без yfinance

        return yf.Ticker(symbol)

    def _downloader_fn(self) -> Callable[..., Any]:
        if self._downloader is not None:
            return self._downloader
        import yfinance as yf

        return yf.download


class DeadlineTicker:
    """Обертка над ``yf.Ticker``: сетевые вызовы идут под дедлайном, остальное — насквозь.

    Обёртка, а не наследник: ``yf.Ticker`` не обязан оставаться совместимым по конструктору
    между версиями, а нам нужен только его интерфейс.
    """

    __slots__ = ("_inner", "_transport")

    def __init__(self, inner: Any, transport: "YfTransport"):
        self._inner = inner
        self._transport = transport

    @property
    def symbol(self) -> str:
        return str(getattr(self._inner, "ticker", "?"))

    def __getattr__(self, name: str) -> Any:
        """Всё, кроме сетевых операций, отдаём как есть (без дедлайна и без магии)."""
        return getattr(self._inner, name)

    def history(self, **kwargs: Any) -> Any:
        return self._transport.call(
            lambda: self._inner.history(**kwargs), name=f"history({self.symbol})"
        )

    def option_chain(self, date: Optional[str] = None) -> Any:
        return self._transport.call(
            lambda: self._inner.option_chain(date), name=f"option_chain({self.symbol},{date})"
        )

    @property
    def fast_info(self) -> Any:
        return self._transport.call(
            lambda: self._inner.fast_info, name=f"fast_info({self.symbol})"
        )

    @property
    def info(self) -> Any:
        return self._transport.call(lambda: self._inner.info, name=f"info({self.symbol})")

    @property
    def options(self) -> Any:
        return self._transport.call(lambda: self._inner.options, name=f"options({self.symbol})")


_lock = threading.Lock()
_shared: Optional[YfTransport] = None
_gate_lock = threading.Lock()
_gate: Optional[YfGate] = None


def get_shared_gate() -> YfGate:
    """Общий ограничитель: один пул yfinance на процесс."""
    global _gate
    if _gate is None:
        with _gate_lock:
            if _gate is None:
                _gate = yf_gate_from_settings()
    return _gate


def reset_shared_gate() -> None:
    """Сбросить ограничитель (тесты, смена конфигурации)."""
    global _gate
    with _gate_lock:
        _gate = None


def get_shared_yf_transport() -> YfTransport:
    """Ленивый общий транспорт yfinance: одна политика дедлайнов на процесс."""
    global _shared
    if _shared is None:
        with _lock:
            if _shared is None:
                _shared = yf_transport_from_settings()
    return _shared


def reset_shared_yf_transport() -> None:
    """Сбрасывает общий экземпляр (тесты, смена конфигурации)."""
    global _shared
    with _lock:
        _shared = None


def yf_gate_from_settings() -> YfGate:
    """Собирает ограничитель параллелизма из фасада конфигурации."""
    from gex.settings import load

    cfg = load().yf
    return YfGate(
        max_concurrency=cfg.max_concurrency,
        queue_wait=cfg.queue_wait_seconds,
        breaker_threshold=cfg.breaker_threshold,
        breaker_cooldown=cfg.breaker_cooldown_seconds,
    )


def yf_transport_from_settings() -> YfTransport:
    """Собирает транспорт из фасада конфигурации (импорт — внутри, как в http.py)."""
    from gex.settings import load

    cfg = load().yf
    return YfTransport(
        policy=DeadlinePolicy(
            default_seconds=cfg.deadline_seconds,
            download_seconds=cfg.download_deadline_seconds,
            max_seconds=cfg.max_deadline_seconds,
        ),
        gate=get_shared_gate(),
    )
