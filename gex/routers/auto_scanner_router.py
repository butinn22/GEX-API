"""Auto Signal Scanner Router — автоматический сканер (только EXTENDED подписка)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from gex.auth.dependencies import get_current_user, require_subscription
from gex.auth.models import User
from gex.auth.settings_router import load_scanner_settings
from gex.adapters.persistence.database import get_session
from gex.deps import (
    provide_auto_scanner_service,
    provide_auto_scanner_ru_service,
    provide_auto_scanner_crypto_service,
    provide_auto_scanner_fx_service,
    provide_auto_scanner_sectors_service,
)
from gex.application.auto_scanner_service import AutoScannerService
from gex.domain.trend_regime import RegimeSliders, evaluate_regime, signal_allowed
from gex.workers import dispatch
from ._helpers import drop_unanchored_signals, handle

router = APIRouter(tags=["auto-scanner"])

require_extended = require_subscription("EXTENDED")

#: Допустимые универсумы сканера.
_UNIVERSES = {"us", "ru", "crypto", "fx", "sectors"}


def _pick_service(
    universe: str,
    us_svc: AutoScannerService,
    ru_svc: AutoScannerService,
    crypto_svc: AutoScannerService,
    fx_svc: AutoScannerService,
    sectors_svc: AutoScannerService,
) -> AutoScannerService:
    """Выбрать сервис по универсуму.

    ru → MOEX-акции, crypto → крипта, fx → валюты/металлы,
    sectors → секторальные ETF США (12 + RSP), иначе US-акции.
    """
    if universe == "ru":
        return ru_svc
    if universe == "crypto":
        return crypto_svc
    if universe == "fx":
        return fx_svc
    if universe == "sectors":
        return sectors_svc
    return us_svc


class SignalDict(BaseModel):
    action: str | None = None
    price: float | None = None
    entry_score: float | None = None
    confidence_class: str | None = None
    timestamp: str | None = None
    tp_price: float | None = None
    sl_price: float | None = None
    reason: str | None = None
    order_type: str | None = None
    gex_reason: str | None = None
    gex_multiplier: float | None = None
    verification_score: float | None = None


class AutoScanInstrumentOut(BaseModel):
    ticker: str
    timeframe: str
    signals: list[SignalDict] = Field(default_factory=list)
    last_scan: str | None = None
    last_signal_ts: str | None = None
    error: str | None = None
    #: Текущая позиция по машине состояний (side/avg_price/since)
    position: dict | None = None


class AutoScanStatusOut(BaseModel):
    running: bool = False
    total_tickers: int = 0
    total_instruments: int = 0
    scanned_count: int = 0
    total_fetches: int = 0
    completed_fetches: int = 0
    instruments_with_signals: int = 0
    instruments_with_errors: int = 0
    scanned_at: str | None = None
    last_error: str | None = None
    new_signals: int = 0
    done: bool | None = None


class TickerListOut(BaseModel):
    tickers: list[str]
    count: int
    names: dict[str, str] = Field(default_factory=dict)


@router.get("/scanner/auto/tickers", response_model=TickerListOut)
def get_tickers(
    universe: str = Query("us", description="us | ru | crypto | fx | sectors"),
    user: User = Depends(require_extended),
    us_svc: AutoScannerService = Depends(provide_auto_scanner_service),
    ru_svc: AutoScannerService = Depends(provide_auto_scanner_ru_service),
    crypto_svc: AutoScannerService = Depends(provide_auto_scanner_crypto_service),
    fx_svc: AutoScannerService = Depends(provide_auto_scanner_fx_service),
    sectors_svc: AutoScannerService = Depends(provide_auto_scanner_sectors_service),
):
    """Список всех тикеров автосканера (и названия компаний)."""
    svc = _pick_service(universe, us_svc, ru_svc, crypto_svc, fx_svc, sectors_svc)
    tickers = svc.get_tickers()
    return TickerListOut(tickers=tickers, count=len(tickers), names=svc.get_names())


@router.get("/scanner/auto/status", response_model=AutoScanStatusOut)
def get_status(
    universe: str = Query("us", description="us | ru | crypto | fx | sectors"),
    user: User = Depends(require_extended),
    us_svc: AutoScannerService = Depends(provide_auto_scanner_service),
    ru_svc: AutoScannerService = Depends(provide_auto_scanner_ru_service),
    crypto_svc: AutoScannerService = Depends(provide_auto_scanner_crypto_service),
    fx_svc: AutoScannerService = Depends(provide_auto_scanner_fx_service),
    sectors_svc: AutoScannerService = Depends(provide_auto_scanner_sectors_service),
):
    """Текущий статус сканера: прогресс, ошибки."""
    svc = _pick_service(universe, us_svc, ru_svc, crypto_svc, fx_svc, sectors_svc)
    return handle(lambda: svc.get_status(), error_src="auto-scanner")


@router.get("/scanner/auto/signals")
def get_signals(
    ticker: Optional[str] = Query(None, description="Фильтр по тикеру"),
    timeframe: Optional[str] = Query(None, description="Фильтр по таймфрейму (4h/1d)"),
    only_with_signals: bool = Query(False, description="Только инструменты с сигналами"),
    universe: str = Query("us", description="us | ru | crypto | fx | sectors"),
    flat_slider: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Превью слайдера боковика (перекрывает личный)"
    ),
    filter_signals: Optional[bool] = Query(
        None, description="Превью фильтра сигналов (перекрывает личный)"
    ),
    trail_pct: Optional[float] = Query(
        None, ge=0.0, le=25.0,
        description="Превью персонального трейлинг-стопа %, для текущего универсума",
    ),
    reverse: Optional[bool] = Query(
        None, description="Превью принудительного разворота (перекрывает личный)"
    ),
    user: User = Depends(require_extended),
    db: Session = Depends(get_session),
    us_svc: AutoScannerService = Depends(provide_auto_scanner_service),
    ru_svc: AutoScannerService = Depends(provide_auto_scanner_ru_service),
    crypto_svc: AutoScannerService = Depends(provide_auto_scanner_crypto_service),
    fx_svc: AutoScannerService = Depends(provide_auto_scanner_fx_service),
    sectors_svc: AutoScannerService = Depends(provide_auto_scanner_sectors_service),
):
    """Накопленные сигналы автосканера + верификация боковика.

    Возвращает плоский список инструментов с их сигналами. Поддерживает
    фильтрацию по тикеру, таймфрейму, и флаг «только с сигналами».

    **Верификация боковика (персональная):** вердикт ``UP/DOWN/FLAT``
    считается на чтении из слайдер-НЕЗАВИСИМЫХ метрик инструмента по
    личному слайдеру пользователя (``/auth/settings/scanner``). Query-параметры
    ``flat_slider``/``filter_signals`` перекрывают личные — для мгновенного
    превью слайдера без сохранения. Сигналы без подтверждённого тренда
    помечаются ``blocked`` (причина ``flat``/``direction_mismatch``); при
    ``filter_signals=true`` они отсекаются из ``signals`` (счётчик — в
    ``blocked_count``). Личный набор тикеров (``tickers[universe]``) сужает
    выдачу; пустой список = все тикеры сканера.
    """
    svc = _pick_service(universe, us_svc, ru_svc, crypto_svc, fx_svc, sectors_svc)

    def _get():
        settings = load_scanner_settings(db, user)
        sliders = _sliders_from_settings(settings)
        if flat_slider is not None:
            sliders = RegimeSliders(
                flat=flat_slider,
                atr=sliders.atr, bbw=sliders.bbw, pct=sliders.pct,
                flat_score_threshold=sliders.flat_score_threshold,
                trend_strength_low=sliders.trend_strength_low,
            )
        gate = bool(settings.get("gate_exits"))
        do_filter = settings.get("filter_signals", True)
        if filter_signals is not None:
            do_filter = filter_signals

        personal = {
            str(t).strip().upper()
            for t in (settings.get("tickers") or {}).get(universe, [])
        }

        # Персональный трейлинг-стоп % (по рынку) + принудительный разворот:
        # сигналы переигрываются из снапшота с этими параметрами на чтении.
        trail_map = settings.get("trailing_pct") or {}
        trail = float(trail_map.get(universe) or 0.0)
        if trail_pct is not None:
            trail = float(trail_pct)
        rev = bool(settings.get("reverse_close", True))
        if reverse is not None:
            rev = bool(reverse)

        signals = svc.get_signals(
            ticker=ticker, timeframe=timeframe, trailing_pct=trail, reverse=rev,
        )
        instruments: list[dict] = []
        total_blocked = 0
        total_with_signals = 0
        total_instruments = 0
        for item in signals:
            # Личный набор тикеров: пусто = все; иначе только выбранные
            if personal and str(item.get("ticker", "")).upper() not in personal:
                continue
            total_instruments += 1
            verdict = _verdict_for_instrument(item, sliders)
            raw_signals = list(item.get("signals") or [])
            # Помечаем заблокированные сигналы (флаг добавляется, сам сигнал не теряется)
            annotated: list[dict] = []
            for sig in raw_signals:
                allowed, reason = signal_allowed(
                    sig.get("order_type"), verdict, gate_exits=gate
                )
                annotated.append({**sig, "blocked": not allowed, "blocked_reason": reason})
            blocked_count = sum(1 for s in annotated if s.get("blocked"))
            if blocked_count:
                total_blocked += blocked_count
            # При filter_signals=true отсекаем заблокированные из выдачи
            shown = [s for s in annotated if not s.get("blocked")] if do_filter else annotated
            # Очерёдность: выход без видимого входа не показываем (фильтр мог
            # скрыть вход, оставив его выход — «SHORT EXIT» без «SHORT ENTRY»).
            shown = drop_unanchored_signals(shown)
            if shown:
                total_with_signals += 1
            instruments.append({
                "ticker": item.get("ticker"),
                "timeframe": item.get("timeframe"),
                "signals": shown,
                "last_scan": item.get("last_scan"),
                "last_signal_ts": item.get("last_signal_ts"),
                "error": item.get("error"),
                "regime": item.get("regime"),
                "position": item.get("position"),
                "verdict": verdict,
                "blocked_count": blocked_count,
                "signals_total": len(raw_signals),
            })
        if only_with_signals:
            # Инструмент «с сигналами» — если есть показанные сигналы ИЛИ
            # открыта позиция (позиция без свежих сигналов всё равно важна:
            # «что держим прямо сейчас» по тикеру).
            instruments = [
                i for i in instruments
                if i.get("signals")
                or str((i.get("position") or {}).get("side") or "flat") != "flat"
            ]
        return {
            "instruments": instruments,
            "total": len(instruments),
            "with_signals": sum(1 for i in instruments if i.get("signals")),
            "blocked_signals": total_blocked,
            "status": svc.get_status(),
        }

    return handle(_get, error_src="auto-scanner")


def _sliders_from_settings(settings: dict) -> RegimeSliders:
    """Собрать RegimeSliders из нормализованных настроек пользователя."""
    return RegimeSliders.from_dict({
        "flat": settings.get("flat_slider", 0.5),
        "atr": settings.get("slider_atr"),
        "bbw": settings.get("slider_bbw"),
        "pct": settings.get("slider_pct"),
        "flat_score_threshold": settings.get("flat_score_threshold", 60.0),
        "trend_strength_low": settings.get("trend_strength_low", 30.0),
    })


def _verdict_for_instrument(item: dict, sliders: RegimeSliders) -> Optional[dict]:
    """Вердикт режима по слайдер-независимым метрикам инструмента.

    ``None`` — метрик нет (мало истории): сигналы не режем (нечем
    верифицировать, см. gex.trend_regime.signal_allowed).
    """
    metrics = item.get("regime")
    if not isinstance(metrics, dict) or not metrics:
        return None
    return evaluate_regime(metrics, sliders)


#: Ручи могут ответить и 200 (успели), и 202 (в фоне), и 502 (фоновая задача
#: провалилась) — возвращается JSONResponse, поэтому схему надо объявить явно.
_ASYNC_RESPONSES = {
    202: {"description": "Принято в фоновую обработку; опрашивать GET /tasks/{task_id}"},
    502: {"description": "Фоновая задача завершилась ошибкой"},
}


@router.post("/scanner/auto/run", responses=_ASYNC_RESPONSES)
def run_scan(
    universe: str = Query("us", description="us | ru | crypto | fx | sectors"),
    user: User = Depends(require_extended),
    us_svc: AutoScannerService = Depends(provide_auto_scanner_service),
    ru_svc: AutoScannerService = Depends(provide_auto_scanner_ru_service),
    crypto_svc: AutoScannerService = Depends(provide_auto_scanner_crypto_service),
    fx_svc: AutoScannerService = Depends(provide_auto_scanner_fx_service),
    sectors_svc: AutoScannerService = Depends(provide_auto_scanner_sectors_service),
):
    """Запустить полное сканирование всех тикеров.

    Сканирование уходит в Celery-воркер (~60-90 сек для 113 тикеров — это бюджет
    распределённого лимитера провайдера, не CPU). Если результат готов быстрее
    ``CELERY_WAIT_TIMEOUT`` сек, отвечаем ``200`` с прежним телом. Иначе —
    ``202 Accepted`` и ``task_id``; клиент опрашивает ``GET /tasks/{task_id}``.

    При недоступности брокера задача выполняется синхронно, как раньше: очередь —
    оптимизация, а не новая точка отказа.
    """
    if dispatch.enabled():
        from gex.workers.tasks.scanner import scan_signature

        response = dispatch.dispatch_bounded(scan_signature(universe))
        if response is not None:
            return response

    svc = _pick_service(universe, us_svc, ru_svc, crypto_svc, fx_svc, sectors_svc)
    return handle(lambda: svc.run_scan(), error_src="auto-scanner")


@router.post("/scanner/auto/run/next", responses=_ASYNC_RESPONSES)
def run_next_batch(
    batch_size: int = Query(10, ge=1, le=50, description="Тикеров за вызов"),
    universe: str = Query("us", description="us | ru | crypto | fx | sectors"),
    user: User = Depends(require_extended),
    us_svc: AutoScannerService = Depends(provide_auto_scanner_service),
    ru_svc: AutoScannerService = Depends(provide_auto_scanner_ru_service),
    crypto_svc: AutoScannerService = Depends(provide_auto_scanner_crypto_service),
    fx_svc: AutoScannerService = Depends(provide_auto_scanner_fx_service),
    sectors_svc: AutoScannerService = Depends(provide_auto_scanner_sectors_service),
):
    """Прогрессивное сканирование: следующая пачка тикеров.

    Для UI с прогресс-баром — вызывается последовательно,
    пока done != true. Контракт ``200`` / ``202`` — как у ``/scanner/auto/run``.
    """
    if dispatch.enabled():
        from gex.workers.tasks.scanner import next_batch_signature

        response = dispatch.dispatch_bounded(next_batch_signature(universe, batch_size))
        if response is not None:
            return response

    svc = _pick_service(universe, us_svc, ru_svc, crypto_svc, fx_svc, sectors_svc)
    return handle(lambda: svc.run_next_batch(batch_size=batch_size), error_src="auto-scanner")


@router.post("/scanner/auto/reset")
def reset_scanner(
    universe: str = Query("us", description="us | ru | crypto | fx | sectors"),
    user: User = Depends(require_extended),
    us_svc: AutoScannerService = Depends(provide_auto_scanner_service),
    ru_svc: AutoScannerService = Depends(provide_auto_scanner_ru_service),
    crypto_svc: AutoScannerService = Depends(provide_auto_scanner_crypto_service),
    fx_svc: AutoScannerService = Depends(provide_auto_scanner_fx_service),
    sectors_svc: AutoScannerService = Depends(provide_auto_scanner_sectors_service),
):
    """Сбросить все накопленные сигналы."""
    svc = _pick_service(universe, us_svc, ru_svc, crypto_svc, fx_svc, sectors_svc)
    return handle(lambda: svc.reset(), error_src="auto-scanner")
