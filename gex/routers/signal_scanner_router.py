"""Signal Scanner Router — персональный сканер (только EXTENDED подписка)."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from gex.auth.dependencies import get_current_user, require_subscription
from gex.auth.models import User, subscription_is_active
from gex.auth.settings_router import load_scanner_settings
from gex.adapters.persistence.database import get_session
from gex.deps import provide_signal_scanner_service
from gex.application.signal_scanner_service import (
    MAX_INSTRUMENTS,
    POLL_INTERVAL_SECONDS,
    SignalScannerService,
    _esc_html,
    signal_line_html,
)
from gex.adapters.notifications.telegram_sender import send_telegram_message
from gex.domain.trend_regime import RegimeSliders, evaluate_regime, signal_allowed
from ._helpers import drop_unanchored_signals, handle

router = APIRouter(tags=["signal-scanner"])

# Барьер: только EXTENDED подписка
require_extended = require_subscription("EXTENDED")


class WatchlistItem(BaseModel):
    ticker: str
    timeframe: str


class WatchlistRequest(BaseModel):
    instruments: list[WatchlistItem] = Field(default_factory=list, max_length=MAX_INSTRUMENTS)


class InstrumentOut(BaseModel):
    ticker: str
    timeframe: str
    latest_signals: list[dict] = Field(default_factory=list)
    last_scan: str | None = None
    error: str | None = None
    #: Вердикт режима рынка по личному слайдеру (None — метрик нет)
    verdict: dict | None = None
    #: Сырые метрики режима (слайдер-независимые, диагностика)
    regime: dict | None = None
    #: Текущая позиция по машине состояний (side/avg_price/since)
    position: dict | None = None
    blocked_count: int = 0
    signals_total: int = 0


class SignalScannerOut(BaseModel):
    instruments: list[InstrumentOut] = Field(default_factory=list)
    max_instruments: int = MAX_INSTRUMENTS
    pool_interval_seconds: int = POLL_INTERVAL_SECONDS
    running: bool = False
    scanned_at: str | None = None
    blocked_signals: int = 0


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


def _verdict_for_instrument(instr: Any, sliders: RegimeSliders) -> dict | None:
    """Вердикт режима по слайдер-независимым метрикам инструмента."""
    metrics = getattr(instr, "regime", None)
    if not isinstance(metrics, dict) or not metrics:
        return None
    return evaluate_regime(metrics, sliders)


def _annotate_signals(
    raw_signals: list[dict], verdict: dict | None, gate: bool, do_filter: bool
) -> tuple[list[dict], int]:
    """Пометить сигналы blocked/blocked_reason и (при do_filter) отсечь."""
    annotated = []
    for sig in raw_signals:
        allowed, reason = signal_allowed(sig.get("order_type"), verdict, gate_exits=gate)
        annotated.append({**sig, "blocked": not allowed, "blocked_reason": reason})
    blocked = sum(1 for s in annotated if s.get("blocked"))
    shown = [s for s in annotated if not s.get("blocked")] if do_filter else annotated
    # Очерёдность: выход без видимого входа не показываем (фильтр мог скрыть
    # вход, оставив его выход). Список здесь «новые первыми» — разворачиваем
    # в хронологический порядок для учёта позиции и обратно.
    shown = list(reversed(drop_unanchored_signals(list(reversed(shown)))))
    return shown, blocked


def _build_response(svc: SignalScannerService, user: User, settings: dict) -> SignalScannerOut:
    instruments = svc.get_watchlist(user.id)
    report = svc.get_report(user.id)

    sliders = _sliders_from_settings(settings)
    gate = bool(settings.get("gate_exits"))
    do_filter = settings.get("filter_signals", True)

    instr_out: list[InstrumentOut] = []
    total_blocked = 0
    for instr in instruments:
        verdict = _verdict_for_instrument(instr, sliders)
        raw = []
        for sig in (instr.latest_signals or [])[:5]:
            raw.append({
                "action": getattr(sig, "action", None),
                "price": getattr(sig, "price", None),
                "entry_score": getattr(sig, "entry_score", None),
                "confidence_class": getattr(sig, "confidence_class", None),
                "timestamp": getattr(sig, "timestamp", None),
                "tp_price": getattr(sig, "tp_price", None),
                "sl_price": getattr(sig, "sl_price", None),
                "reason": getattr(sig, "reason", None),
                "order_type": getattr(sig, "order_type", None),
                "gex_reason": getattr(sig, "gex_reason", None),
                "gex_multiplier": getattr(sig, "gex_multiplier", None),
                "verification_score": getattr(sig, "verification_score", None),
            })
        shown, blocked = _annotate_signals(raw, verdict, gate, do_filter)
        total_blocked += blocked
        instr_out.append(InstrumentOut(
            ticker=instr.ticker,
            timeframe=instr.timeframe,
            latest_signals=shown,
            last_scan=instr.last_scan.isoformat() if instr.last_scan else None,
            error=instr.error,
            verdict=verdict,
            regime=dict(instr.regime) if instr.regime else None,
            position=dict(instr.position) if instr.position else None,
            blocked_count=blocked,
            signals_total=len(raw),
        ))

    return SignalScannerOut(
        instruments=instr_out,
        max_instruments=MAX_INSTRUMENTS,
        pool_interval_seconds=POLL_INTERVAL_SECONDS,
        running=svc.is_running,
        scanned_at=report.scanned_at.isoformat() if report.scanned_at else None,
        blocked_signals=total_blocked,
    )


@router.get("/scanner/signals", response_model=SignalScannerOut)
def get_signals(
    flat_slider: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Превью слайдера фильтрации (перекрывает личный)"
    ),
    filter_signals: Optional[bool] = Query(
        None, description="Превью фильтра сигналов (перекрывает личный)"
    ),
    user: User = Depends(require_extended),
    db: Session = Depends(get_session),
    svc: SignalScannerService = Depends(provide_signal_scanner_service),
):
    """Персональные сигналы + верификация боковика по личному слайдеру.

    Вердикт UP/DOWN/FLAT считается на чтении из слайдер-независимых метрик
    (сохранены при сканировании) по настройкам пользователя
    (``/auth/settings/scanner``). Query-параметры ``flat_slider``/``filter_signals``
    перекрывают личные — для мгновенного превью. Сигналы без подтверждённого
    тренда помечаются ``blocked``; при ``filter_signals=true`` отсекаются
    (счётчик — в ``blocked_signals``/``blocked_count``).
    """
    settings = load_scanner_settings(db, user)
    if flat_slider is not None:
        settings = {**settings, "flat_slider": flat_slider}
    if filter_signals is not None:
        settings = {**settings, "filter_signals": filter_signals}
    return handle(lambda: _build_response(svc, user, settings), error_src="signal-scanner")


@router.post("/scanner/signals/watchlist", response_model=SignalScannerOut)
def update_watchlist(
    body: WatchlistRequest,
    user: User = Depends(require_extended),
    db: Session = Depends(get_session),
    svc: SignalScannerService = Depends(provide_signal_scanner_service),
):
    def _update():
        pairs = [{"ticker": i.ticker, "timeframe": i.timeframe} for i in body.instruments]
        svc.set_watchlist(user.id, pairs)
        return _build_response(svc, user, load_scanner_settings(db, user))

    return handle(_update, error_src="signal-scanner")


@router.post("/scanner/signals/run", response_model=SignalScannerOut)
def run_signals_now(
    user: User = Depends(require_extended),
    db: Session = Depends(get_session),
    svc: SignalScannerService = Depends(provide_signal_scanner_service),
    notify: bool = Query(False),
    background_tasks: BackgroundTasks = None,
):
    """Принудительный прогон сканирования для пользователя.

    При notify=true — сводка уходит в Telegram (если подключён и включены
    персональные уведомления в профиле — тот же гейт, что в фоновом цикле).
    """

    def _run():
        # Ручной прогон: сервис сам уведомления НЕ шлёт (notify=False) —
        # сводка уходит ниже один раз, только при notify=true в запросе.
        svc.scan_now(user_id=user.id, notify=False)
        result = _build_response(svc, user, load_scanner_settings(db, user))

        if notify and user.telegram_chat_id and user.telegram_notify and subscription_is_active(user):
            instruments = result.instruments
            if instruments:
                lines = ["<b>Сигнальный сканер — сводка</b>", ""]
                for instr in instruments:
                    signals = instr.latest_signals or []
                    if signals:
                        line = signal_line_html(instr.ticker, instr.timeframe, signals[0], when=instr.last_scan)
                        if line:
                            lines.append(line)
                    elif instr.error:
                        lines.append(
                            f"  <b>{_esc_html(instr.ticker)}</b> "
                            f"[{_esc_html(instr.timeframe)}]: {_esc_html(instr.error)}"
                        )
                if len(lines) > 2:
                    if background_tasks:
                        background_tasks.add_task(send_telegram_message, "\n".join(lines), parse_mode="HTML", chat_id=user.telegram_chat_id)
                    else:
                        send_telegram_message("\n".join(lines), parse_mode="HTML", chat_id=user.telegram_chat_id)

        return result

    return handle(_run, error_src="signal-scanner")
