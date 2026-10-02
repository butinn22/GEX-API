"""
Dependency Injection container for GEX Analytics.

Architecture
------------
- ``Container`` — holds all service instances, wired at startup.
- ``provide_*()`` — FastAPI ``Depends()``-compatible provider functions.
- ``get_*_service()`` — legacy getters for backward compatibility (delegate to Container).

Usage in routers::

    from fastapi import Depends
    from gex.deps import provide_gex_service

    @router.get("/gex/{ticker}")
    def get_gex(ticker: str, svc = Depends(provide_gex_service)):
        return svc.analyze(ticker)

Синхронный доступ (без FastAPI-контекста) — только там, где Depends неприменим::

    from gex.deps import get_redis_client, get_task_queue

Легаси-геттеры вида ``get_<domain>_service()`` удалены: у них не было ни одного
потребителя (аудит 2026-09-16, отчёт 08-legacy-dead-code, F5). Живые исключения —
``get_auto_scanner_service``, ``get_scan_service``, ``get_redis_client``, ``get_task_queue``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gex.application.service import GEXService
    from gex.application.ta_service import TAService
    from gex.application.trendline_service import TrendlineService
    from gex.application.macd_trend_service import MacdTrendService
    from gex.application.signal_service import SignalService
    from gex.application.signal_scanner_service import SignalScannerService
    from gex.application.auto_scanner_service import AutoScannerService
    from gex.application.scan_service import ScanService
    from gex.application.sector_service import SectorBreadthService
    from gex.application.extended import ExtendedGEXAnalyzer
    from gex.application.commodity_dynamics import CommodityDynamicsService
    from gex.application.novel_candles import NovelCandlesService
    from gex.adapters.cache.page_store import RedisPageStore
    from gex.adapters.cache.redis_client import RedisClient
    from gex.application.queue import TaskPublisher


# ═══════════════════════════════════════════════════════════════════════
# Container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Container:
    """Application-wide service container.

    Services are set via ``wire()`` at startup, then accessed through
    ``provide_*()`` FastAPI dependencies or ``get_*_service()`` helpers.
    """

    gex_service: GEXService | None = None
    ta_service: TAService | None = None
    trendline_service: TrendlineService | None = None
    macd_trend_service: MacdTrendService | None = None
    signal_service: SignalService | None = None
    signal_scanner_service: SignalScannerService | None = None
    auto_scanner_service: AutoScannerService | None = None
    auto_scanner_ru_service: AutoScannerService | None = None
    auto_scanner_crypto_service: AutoScannerService | None = None
    auto_scanner_fx_service: AutoScannerService | None = None
    auto_scanner_sectors_service: AutoScannerService | None = None
    scan_service: ScanService | None = None
    sector_service: SectorBreadthService | None = None
    extended_gex_service: ExtendedGEXAnalyzer | None = None
    commodity_dynamics_service: CommodityDynamicsService | None = None
    novel_candles_service: NovelCandlesService | None = None
    redis_client: RedisClient | None = None
    task_queue: TaskPublisher | None = None
    #: Хранилище payload'ов страниц (``SnapshotPort``): чтение без вычисления для страниц,
    #: вынесенных из запроса. Живёт в контейнере, потому что один экземпляр на процесс —
    #: в него же пишет фоновый пересчёт, и ярус памяти процесса должен быть общим.
    page_store: RedisPageStore | None = None

    def wire(
        self,
        *,
        gex_svc=None,
        ta_svc=None,
        trendline_svc=None,
        macd_svc=None,
        sig_svc=None,
        sig_scanner_svc=None,
        auto_scanner_svc=None,
        auto_scanner_ru_svc=None,
        auto_scanner_crypto_svc=None,
        auto_scanner_fx_svc=None,
        auto_scanner_sectors_svc=None,
        scan_svc=None,
        sector_svc=None,
        ext_gex_svc=None,
        comm_dyn_svc=None,
        redis=None,
        tq=None,
        page_store=None,
    ) -> None:
        """Wire all services into the container. Called once at startup."""
        from gex.application.novel_candles import NovelCandlesService

        self.gex_service = gex_svc
        self.ta_service = ta_svc
        self.trendline_service = trendline_svc
        self.macd_trend_service = macd_svc
        self.signal_service = sig_svc
        self.signal_scanner_service = sig_scanner_svc
        self.auto_scanner_service = auto_scanner_svc
        self.auto_scanner_ru_service = auto_scanner_ru_svc
        self.auto_scanner_crypto_service = auto_scanner_crypto_svc
        self.auto_scanner_fx_service = auto_scanner_fx_svc
        self.auto_scanner_sectors_service = auto_scanner_sectors_svc
        self.scan_service = scan_svc
        self.sector_service = sector_svc
        self.extended_gex_service = ext_gex_svc
        self.commodity_dynamics_service = comm_dyn_svc
        self.novel_candles_service = NovelCandlesService(redis_client=redis)
        self.redis_client = redis
        self.task_queue = tq
        self.page_store = page_store


# ── Global container instance ────────────────────────────────────────

container = Container()


# ═══════════════════════════════════════════════════════════════════════
# FastAPI Depends() providers (use in route signatures)
# ═══════════════════════════════════════════════════════════════════════

def provide_gex_service() -> GEXService:
    return _assert(container.gex_service, "GEXService")


def provide_ta_service() -> TAService:
    return _assert(container.ta_service, "TAService")


def provide_trendline_service() -> TrendlineService:
    return _assert(container.trendline_service, "TrendlineService")


def provide_macd_trend_service() -> MacdTrendService:
    return _assert(container.macd_trend_service, "MacdTrendService")


def provide_signal_service() -> SignalService:
    return _assert(container.signal_service, "SignalService")


def provide_signal_scanner_service() -> SignalScannerService:
    return _assert(container.signal_scanner_service, "SignalScannerService")


def provide_auto_scanner_service() -> AutoScannerService:
    return _assert(container.auto_scanner_service, "AutoScannerService")


def provide_auto_scanner_ru_service() -> AutoScannerService:
    return _assert(container.auto_scanner_ru_service, "AutoScannerService (RU)")


def provide_auto_scanner_crypto_service() -> AutoScannerService:
    return _assert(container.auto_scanner_crypto_service, "AutoScannerService (crypto)")


def provide_auto_scanner_fx_service() -> AutoScannerService:
    return _assert(container.auto_scanner_fx_service, "AutoScannerService (FX)")


def provide_auto_scanner_sectors_service() -> AutoScannerService:
    return _assert(container.auto_scanner_sectors_service, "AutoScannerService (sectors)")


def provide_scan_service() -> ScanService:
    return _assert(container.scan_service, "ScanService")


def provide_sector_service() -> SectorBreadthService:
    return _assert(container.sector_service, "SectorBreadthService")


def provide_page_store():
    """Хранилище снапшотов страниц (``SnapshotPort``) для роутеров, вынесенных из запроса.

    Провайдер, а не прямой импорт: роутерам запрещено ходить в ``gex.adapters`` (кольцо R5),
    и это не формальность — именно через такой импорт в обработчик когда-то вернулась
    синхронная загрузка данных.
    """
    return _assert(container.page_store, "RedisPageStore")


def provide_extended_gex_service() -> ExtendedGEXAnalyzer:
    return _assert(container.extended_gex_service, "ExtendedGEXAnalyzer")


def provide_commodity_dynamics_service() -> CommodityDynamicsService:
    return _assert(container.commodity_dynamics_service, "CommodityDynamicsService")


def provide_novel_candles_service() -> NovelCandlesService:
    return _assert(container.novel_candles_service, "NovelCandlesService")


# ═══════════════════════════════════════════════════════════════════════
# Legacy getters — оставлены ТОЛЬКО те, у которых есть живые потребители.
#
# Аудит 2026-09-16 (отчёт 08-legacy-dead-code, F5): 13 геттеров-делегатов не
# использовались нигде, включая тесты, и удалены. Оставшиеся имеют конкретных
# вызывающих (проверено grep'ом по всему репозиторию):
#   get_auto_scanner_service — gex/auth/admin_router.py:1399
#   get_scan_service         — gex/auth/admin_router.py:1398, gex/system_metrics.py:240
#   get_redis_client         — 14 мест (роутеры/сервисы)
#   get_task_queue           — планировщик, фон-фетчер, админка, метрики
# Полный перевод роутеров на Depends(provide_*) — волна P6.E.

def get_auto_scanner_service() -> AutoScannerService:
    return provide_auto_scanner_service()


def get_scan_service() -> ScanService:
    return provide_scan_service()


def get_redis_client() -> RedisClient | None:
    return container.redis_client


def get_task_queue() -> TaskPublisher:
    """Публикатор задач: очередь (Redis Streams) за портом, наружу — задачи.

    Геттер сохранён под тем же именем: на него завязаны планировщик, админка, метрики
    и ручка фетчера, и замена транспорта не должна тянуть за собой их переписывание.
    """
    return _assert(container.task_queue, "TaskPublisher")


# ═══════════════════════════════════════════════════════════════════════
# Wire helper (replaces init_deps)
# ═══════════════════════════════════════════════════════════════════════

def wire_container(**kwargs) -> None:
    """Wire all services into the global container. Called once at startup."""
    container.wire(**kwargs)


# ── Internal ──────────────────────────────────────────────────────────

def _assert(obj, name: str):
    if obj is None:
        raise RuntimeError(f"{name} not wired into container. Call wire_container() at startup.")
    return obj
