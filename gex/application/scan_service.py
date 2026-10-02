"""Сервис автосканирования: ежедневный прогон TA + GEX по основным тикерам.

Запускает фоновый поток (daemon) при старте приложения, который периодически
проходит по списку тикеров (:data:`WATCHLIST`) и для каждого выполняет полный
технический анализ (через :class:`gex.ta_service.TAService`) и GEX-анализ
(через :class:`gex.service.GEXService.analyze_live`). Результаты хранятся
in-memory (последний успешный прогон на тикер) и доступны через эндпоинты.

Архитектура
-----------
Без внешних зависимостей по расписанию — используется ``threading`` из stdlib.
Поток-демон не блокирует shutdown. Тредобезопасность обеспечивается
``threading.Lock`` вокруг хранилища результатов.

Жизненный цикл
--------------
  * :meth:`start` — запускает фоновый цикл (первый прогон почти сразу, затем
    каждые ``interval_seconds``).
  * :meth:`stop` — мягко останавливает поток (через ``Event``).
  * :meth:`scan_all` — один полный прогон (можно вызывать вручную из эндпоинта).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from gex.application.service import GEXService
from gex.application.ta_service import TAService

logger = logging.getLogger(__name__)


# Список тикеров для ежедневного автосканирования (основные ETF + megacaps).
WATCHLIST: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "AVGO",
    "SMH", "RSP",
)

# Интервал между прогонами по умолчанию: 6 часов = 4 прогона/сутки.
DEFAULT_INTERVAL_SECONDS: int = 6 * 3600

# Задержка первого прогона после старта (чтобы сервис успел инициализироваться).
STARTUP_DELAY_SECONDS: float = 5.0


@dataclass
class ScanRecord:
    """Результат одного прогона по тикеру.

    Attributes
    ----------
    ticker : str
    scanned_at : datetime
        Время завершения прогона (UTC).
    status : str
        'ok' — оба анализа успешны; 'error' — хотя бы один упал.
    error : Optional[str]
        Текст ошибки, если status == 'error'.
    ta : Optional[Any]
        Полный TA-отчёт (TAAnalysisOut) или None при ошибке TA.
    gex : Optional[Any]
        Полный GEX-отчёт (GEXAnalysisOut) или None при ошибке GEX.
    """

    ticker: str
    scanned_at: datetime
    status: str = "ok"
    error: Optional[str] = None
    ta: Optional[Any] = None
    gex: Optional[Any] = None


@dataclass
class ScanReport:
    """Сводный отчёт по одному полному прогону всех тикеров.

    Attributes
    ----------
    scanned_at : datetime
        Время завершения прогона.
    interval_hours : float
        Текущий интервал между прогонами, часов.
    total, ok, failed : int
        Счётчики.
    records : list[ScanRecord]
        Результаты по каждому тикеру.
    """

    scanned_at: datetime
    interval_hours: float
    total: int = 0
    ok: int = 0
    failed: int = 0
    records: list[ScanRecord] = field(default_factory=list)


class ScanService:
    """Фоновый автосканер TA + GEX по списку тикеров.

    Parameters
    ----------
    ta_service : TAService
        Сервис технического анализа.
    gex_service : GEXService
        Сервис GEX-анализа (метод ``analyze_live``).
    watchlist : tuple[str, ...]
        Список тикеров. По умолчанию :data:`WATCHLIST`.
    interval_seconds : int
        Интервал между прогонами, секунд (по умолчанию 6 часов).
    startup_delay : float
        Задержка первого прогона после :meth:`start`, секунд.
    per_ticker_delay : float
        Пауза между тикерами внутри одного прогона, секунд. Смягчает
        rate-limiting Yahoo Finance при последовательных запросах.
    """

    def __init__(
        self,
        ta_service: TAService,
        gex_service: GEXService,
        watchlist: tuple[str, ...] = WATCHLIST,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        startup_delay: float = STARTUP_DELAY_SECONDS,
        per_ticker_delay: float = 2.0,
    ):
        self.ta_service = ta_service
        self.gex_service = gex_service
        self.watchlist = tuple(t.strip().upper() for t in watchlist)
        self.interval_seconds = int(interval_seconds)
        self.startup_delay = float(startup_delay)
        self.per_ticker_delay = float(per_ticker_delay)

        self._lock = threading.Lock()
        self._results: dict[str, ScanRecord] = {}
        self._last_report: Optional[ScanReport] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ================================================================== #
    #  Жизненный цикл
    # ================================================================== #
    def start(self) -> None:
        """Запустить фоновый поток сканирования (идемпотентно)."""
        if self._running:
            logger.warning("ScanService уже запущен")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="gex-scan", daemon=True
        )
        self._running = True
        self._thread.start()
        logger.info(
            "ScanService запущен: %d тикеров, интервал %dс",
            len(self.watchlist), self.interval_seconds,
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Мягко остановить фоновый поток."""
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("ScanService остановлен")

    @property
    def is_running(self) -> bool:
        return self._running

    # ================================================================== #
    #  Фоновый цикл
    # ================================================================== #
    def _run_loop(self) -> None:
        """Главный цикл: первый прогон, затем периодические с интервалом."""
        # Короткая задержка перед первым прогоном — пусть приложение стартует.
        if self._stop_event.wait(self.startup_delay):
            return  # остановили до старта

        while not self._stop_event.is_set():
            try:
                report = self.scan_all()
                logger.info(
                    "Прогон завершён: ok=%d failed=%d из %d",
                    report.ok, report.failed, report.total,
                )
            except Exception:  # pragma: no cover — защитный catch-all в цикле
                logger.exception("Непредвиденная ошибка в цикле сканера")

            # Ждём до следующего прогона (прерываемо через stop_event).
            if self._stop_event.wait(self.interval_seconds):
                break

    # ================================================================== #
    #  Один полный прогон
    # ================================================================== #
    def scan_all(self, auto_notify: bool = True) -> ScanReport:
        """Пройти по всему watchlist, обновить кэш, вернуть сводный отчёт.

        Каждый тикер обрабатывается в собственном try/except — ошибка одного
        не роняет остальные. Результаты складываются в ``_results`` под блокировкой.

        Parameters
        ----------
        auto_notify : bool
            Отправить сводный отчёт в Telegram после прогона. Фоновый цикл
            вызывает с ``True``; ручной эндпоинт ``POST /scans/run`` — с
            ``False`` (чтобы не дублировать отправку, которая делается через
            query-параметр ``notify`` самой ручкой).
        """
        logger.info("Начало прогона по %d тикерам", len(self.watchlist))
        records: list[ScanRecord] = []
        ok = failed = 0

        for ticker in self.watchlist:
            record = self._scan_one(ticker)
            records.append(record)
            if record.status == "ok":
                ok += 1
            else:
                failed += 1

            # Пауза между тикерами — смягчает rate-limiting Yahoo Finance.
            # Прерываемо через stop_event, чтобы shutdown не ждал.
            if self.per_ticker_delay > 0 and ticker != self.watchlist[-1]:
                if self._stop_event.wait(self.per_ticker_delay):
                    break

        report = ScanReport(
            scanned_at=datetime.now(timezone.utc),
            interval_hours=round(self.interval_seconds / 3600.0, 2),
            total=len(self.watchlist),
            ok=ok,
            failed=failed,
            records=records,
        )

        with self._lock:
            for r in records:
                if r.status == "ok":
                    # Сохраняем только успешные прогоны в кэш «последний успешный».
                    self._results[r.ticker] = r
                else:
                    # Если раньше был успешный — оставляем его; лишь логируем провал.
                    logger.warning("Тикер %s: %s", r.ticker, r.error)
            self._last_report = report

        # Автопуш сводки в Telegram (фоновый прогон). Ошибки транспорта
        # логируются и не роняют цикл сканера.
        if auto_notify:
            self._notify_report(report)

        return report

    def _notify_report(self, report: ScanReport) -> None:
        """Отправить сводный отчёт прогона в Telegram.

        Конвертирует внутренний :class:`ScanReport` в :class:`ScanReportOut`
        (через локальный маппер, чтобы не зависеть от ``main.py``) и вызывает
        ``notify_scan_report``. Любые ошибки транспорта логируются и не
        распространяются — фоновый сканер не должен падать из-за Telegram.
        """
        try:
            from gex.schemas import ScanReportOut, ScanRecordOut
            from gex.adapters.notifications.telegram_sender import notify_scan_report

            records_out = [
                ScanRecordOut(
                    ticker=r.ticker,
                    scanned_at=r.scanned_at,
                    status=r.status,
                    error=r.error,
                    ta=r.ta,
                    gex=r.gex,
                )
                for r in report.records
            ]
            report_out = ScanReportOut(
                scanned_at=report.scanned_at,
                interval_hours=report.interval_hours,
                running=self.is_running,
                total=report.total,
                ok=report.ok,
                failed=report.failed,
                records=records_out,
            )
            notify_scan_report(report_out)
        except Exception:  # noqa: BLE001 — транспорт не должен ронять сканер
            logger.exception("Не удалось отправить сводку сканера в Telegram")

    def _scan_one(self, ticker: str) -> ScanRecord:
        """Прогнать TA + GEX по одному тикеру, вернуть ScanRecord.

        TA и GEX считаются независимо: если TA упал, GEX всё равно считается
        (и наоборот). Статус 'ok' только если оба успешны.
        """
        ta_result = None
        gex_result = None
        error_parts: list[str] = []

        # --- TA ---
        try:
            ta_result = self.ta_service.analyze(ticker)
        except Exception as exc:  # noqa: BLE001 — перехватываем любой сбой
            error_parts.append(f"TA: {type(exc).__name__}: {exc}")
            logger.debug("TA %s упал: %s", ticker, exc)

        # --- GEX ---
        try:
            gex_result = self.gex_service.analyze_live(ticker)
        except Exception as exc:  # noqa: BLE001
            error_parts.append(f"GEX: {type(exc).__name__}: {exc}")
            logger.debug("GEX %s упал: %s", ticker, exc)

        status = "ok" if (ta_result is not None and gex_result is not None) else "error"
        return ScanRecord(
            ticker=ticker,
            scanned_at=datetime.now(timezone.utc),
            status=status,
            error=" | ".join(error_parts) if error_parts else None,
            ta=ta_result,
            gex=gex_result,
        )

    # ================================================================== #
    #  Доступ к результатам
    # ================================================================== #
    def get_latest(self, ticker: str) -> Optional[ScanRecord]:
        """Последний успешный прогон по тикеру или None."""
        with self._lock:
            return self._results.get(ticker.strip().upper())

    def list_records(self) -> list[ScanRecord]:
        """Все последние успешные прогоны (по тикерам из кэша)."""
        with self._lock:
            return list(self._results.values())

    def last_report(self) -> Optional[ScanReport]:
        """Последний сводный отчёт (для эндпоинта /scans)."""
        with self._lock:
            return self._last_report
