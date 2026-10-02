"""Персональные настройки дашборда: GET/PUT /auth/settings/dashboard.

GET — работает и без авторизации (возвращает пустые дефолты, фронт
подмешивает localStorage). PUT — требует авторизации: сохраняет настройки
в таблицу user_dashboard_settings (одна строка на пользователя).

Валидация:
- emas        — подмножество {20, 50, 100, 200}, без дубликатов, по возрастанию
- instruments — любые валидные тикеры (акции/ETF/индексы/крипта/MOEX):
  A-Z 0-9 и спецсимволы . _ - ^ =, до 20 символов, без дубликатов,
  максимум MAX_INSTRUMENTS. Пользователь сам добавляет любой тикер,
  выбор закрепляется за его аккаунтом.
- trendlines  — dict {TICKER: [до 5 линий]}, каждая линия — 4 конечных числа
  + опциональный цвет (строка ≤ 32 символов) + опциональные метки
  времени t1/t2 (epoch мс) — привязка к времени, а не к номеру бара:
  благодаря им линия пользователя переносится между таймфреймами
- timeframe   — таймфрейм графиков дашборда, один из ALLOWED_TIMEFRAMES
"""
from __future__ import annotations

import logging
import math
import re

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from gex.adapters.persistence.database import get_session
from gex.adapters.providers.catalog import CANONICAL_TIMEFRAMES

from .dependencies import get_current_user, get_optional_user
from .models import User
from .user_scanner_settings import UserScannerSettings
from .user_settings import UserDashboardSettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/settings", tags=["settings"])

# Базовые тикеры дашборда (используются новым React UI)
ALLOWED_EMAS = [20, 50, 100, 200]
# Любой валидный тикер: акции/ETF (AAPL, BRK.B), индексы (^GSPC), крипта (BTC,
# SOL, BTC-USD), MOEX (RTS, MIX, SI). Буквы/цифры + . _ - ^ =, до 20 символов.
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9._\-^=]{0,19}$")
MAX_INSTRUMENTS = 30
MAX_CUSTOM_LINES = 5
# Таймфреймы графиков дашборда. Раньше здесь был свой литерал «как у ohlcv_service,
# но не импортируем — тянет yfinance в роутер»; теперь источник — каталог (чистые
# данные, без SDK и сети), поэтому дублирование больше не нужно.
ALLOWED_TIMEFRAMES = CANONICAL_TIMEFRAMES
DEFAULT_TIMEFRAME = "1d"

# ── Авто-сканер: верификация боковика + личные тикеры ──────────
#: Универсумы авто-сканера (gex.routers.auto_scanner_router._UNIVERSES).
SCANNER_UNIVERSES = ("us", "ru", "crypto", "fx", "sectors")
#: Максимум личных тикеров на универсум (пусто = следить за всеми).
MAX_SCANNER_TICKERS = 60
DEFAULT_FLAT_SLIDER = 0.5
DEFAULT_FLAT_SCORE_THRESHOLD = 60.0
DEFAULT_TREND_STRENGTH_LOW = 30.0
#: Верхняя граница трейлинг-стопа, % (0 — выключен; защита от опечаток).
MAX_TRAILING_PCT = 25.0


# ================================================================= #
#  Pydantic-схемы
# ================================================================= #
class DashboardLine(BaseModel):
    """Одна кастомная трендовая линия (bar_index → цена).

    Поля ``opacity``/``label``/``kind`` — для GEX-стен (Call/Put Wall):
    полупрозрачные уровни S/R с подписью. Поля ``t1``/``t2`` — время
    концов линии (epoch мс): по ним линия перестраивается при смене
    таймфрейма (bar_index у 1d и 4h означает разное время).
    Все они опциональны — старые линии (x1/y1/x2/y2/color) работают.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    color: str | None = None
    opacity: float | None = None
    label: str | None = None
    kind: str | None = None
    t1: float | None = None
    t2: float | None = None


class DashboardBand(BaseModel):
    """Доверительный интервал GEX на графике (один на тикер)."""

    low: float
    high: float
    label: str | None = None
    color: str | None = None


class DashboardSettingsIn(BaseModel):
    """Входная модель PUT /auth/settings/dashboard."""

    emas: list[int] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    trendlines: dict[str, list[DashboardLine]] = Field(default_factory=dict)
    walls: dict[str, list[DashboardLine]] = Field(default_factory=dict)
    bands: dict[str, DashboardBand] = Field(default_factory=dict)
    timeframe: str = DEFAULT_TIMEFRAME
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Вклад инструментов в настроение рынка на дашборде "
        "(0..100 на тикер; пусто = все равны).",
    )


class DashboardSettingsOut(BaseModel):
    """Выходная модель GET/PUT /auth/settings/dashboard."""

    emas: list[int] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    trendlines: dict[str, list[dict]] = Field(default_factory=dict)
    walls: dict[str, list[dict]] = Field(default_factory=dict)
    bands: dict[str, dict] = Field(default_factory=dict)
    timeframe: str = DEFAULT_TIMEFRAME
    weights: dict[str, float] = Field(default_factory=dict)


class ScannerSettingsIn(BaseModel):
    """Входная модель PUT /auth/settings/scanner.

    ``flat_slider`` — главный слайдер верификации боковика (0 — флэтом
    считается только совсем мёртвый рынок, 1 — широкое определение флэта).
    ``slider_atr``/``slider_bbw``/``slider_pct`` — тонкая настройка по
    компонентам (``null`` → берётся главный). ``tickers`` — личный набор
    отслеживаемых тикеров по универсумам (пусто = все тикеры сканера).
    """

    flat_slider: float = DEFAULT_FLAT_SLIDER
    slider_atr: float | None = None
    slider_bbw: float | None = None
    slider_pct: float | None = None
    flat_score_threshold: float = DEFAULT_FLAT_SCORE_THRESHOLD
    trend_strength_low: float = DEFAULT_TREND_STRENGTH_LOW
    filter_signals: bool = True
    gate_exits: bool = False
    tickers: dict[str, list[str]] = Field(default_factory=dict)
    #: Трейлинг-стоп % по универсумам авто-сканера ({"us": 2.0, ...});
    #: отсутствие ключа/0 — выключен. Меняет место выходов на чтении.
    trailing_pct: dict[str, float] = Field(default_factory=dict)
    #: Трейлинг-стоп % личного сигнального сканера (единое значение, 0 — выкл).
    trailing_pct_personal: float = 0.0
    #: Принудительный разворот: противоположный вход закрывает позицию.
    reverse_close: bool = True


class ScannerSettingsOut(ScannerSettingsIn):
    """Выходная модель GET/PUT /auth/settings/scanner (та же форма)."""


# ================================================================= #
#  Нормализация
# ================================================================= #
def _defaults() -> dict:
    return {
        "emas": [],
        "instruments": [],
        "trendlines": {},
        "walls": {},
        "bands": {},
        "timeframe": DEFAULT_TIMEFRAME,
        "weights": {},
    }


def _normalize(data: dict) -> dict:
    """Привести входные настройки к безопасному каноническому виду."""
    # EMA: только разрешённые, без дубликатов, по возрастанию
    emas: list[int] = []
    for e in data.get("emas", []):
        try:
            ie = int(e)
        except (TypeError, ValueError):
            continue
        if ie in ALLOWED_EMAS and ie not in emas:
            emas.append(ie)
    emas.sort()

    # Инструменты: любые валидные тикеры (базовые + пользовательские),
    # без дубликатов, максимум MAX_INSTRUMENTS
    instruments = []
    for i in data.get("instruments", []):
        t = str(i).strip().upper()
        if TICKER_RE.match(t) and t not in instruments:
            instruments.append(t)
        if len(instruments) >= MAX_INSTRUMENTS:
            break

    # Кастомные линии: до MAX_CUSTOM_LINES на тикер, только конечные числа
    trendlines: dict[str, list[dict]] = {}
    for ticker, lines in (data.get("trendlines") or {}).items():
        t = str(ticker).strip().upper()
        if not t:
            continue
        valid = _normalize_lines(lines, MAX_CUSTOM_LINES)
        if valid:
            trendlines[t] = valid

    # GEX-стены (уровни с GEX-графика): отдельный список, ≤ MAX_CUSTOM_LINES
    walls: dict[str, list[dict]] = {}
    for ticker, lines in (data.get("walls") or {}).items():
        t = str(ticker).strip().upper()
        if not t:
            continue
        valid = _normalize_lines(lines, MAX_CUSTOM_LINES)
        if valid:
            walls[t] = valid

    # Доверительный интервал GEX: один на тикер, low < high
    bands: dict[str, dict] = {}
    for ticker, b in (data.get("bands") or {}).items():
        t = str(ticker).strip().upper()
        if not t or not isinstance(b, dict):
            continue
        try:
            low = float(b.get("low"))
            high = float(b.get("high"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            continue
        band: dict = {"low": low, "high": high}
        label = str(b.get("label") or "").strip()[:64] or None
        if label:
            band["label"] = label
        color = str(b.get("color") or "").strip()[:32] or None
        if color:
            band["color"] = color
        bands[t] = band

    # Таймфрейм графиков: только из разрешённого набора, иначе дефолт
    tf = str(data.get("timeframe") or "").strip().lower()
    if tf not in ALLOWED_TIMEFRAMES:
        tf = DEFAULT_TIMEFRAME

    # Веса инструментов для настроения рынка: тикер -> 0..100
    weights: dict[str, float] = {}
    for k, v in (data.get("weights") or {}).items():
        tk = str(k).strip().upper()
        if not TICKER_RE.match(tk):
            continue
        try:
            w = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(w) or w < 0 or w > 100:
            continue
        weights[tk] = round(w, 1)

    return {
        "emas": emas,
        "instruments": instruments,
        "trendlines": trendlines,
        "walls": walls,
        "bands": bands,
        "timeframe": tf,
        "weights": weights,
    }


def _normalize_lines(lines, limit: int) -> list[dict]:
    """Нормализовать список линий (кастомные/GEX-стены): конечные числа,
    color ≤ 32, opacity 0..1, label ≤ 64, kind ∈ {resistance, support}."""
    valid: list[dict] = []
    for ln in list(lines)[:limit]:
        try:
            x1, y1, x2, y2 = (
                float(ln["x1"]), float(ln["y1"]),
                float(ln["x2"]), float(ln["y2"]),
            )
        except (TypeError, KeyError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        color = str(ln.get("color") or "").strip()[:32] or None
        line: dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color}
        # GEX-стены: прозрачность 0..1, подпись, тип S/R
        try:
            op = ln.get("opacity")
            if op is not None:
                op = float(op)
                if not math.isfinite(op) or not (0.0 <= op <= 1.0):
                    op = None
        except (TypeError, ValueError):
            op = None
        if op is not None:
            line["opacity"] = op
        label = str(ln.get("label") or "").strip()[:64] or None
        if label:
            line["label"] = label
        kind = str(ln.get("kind") or "").strip().lower()[:16] or None
        if kind in ("resistance", "support"):
            line["kind"] = kind
        # Метки времени концов линии (epoch мс): по ним линия переносится
        # между таймфреймами. Пишем только когда обе корректны.
        stamps: dict[str, float] = {}
        for key in ("t1", "t2"):
            try:
                raw = ln.get(key)
                if raw is None:
                    continue
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val) and val > 0:
                stamps[key] = val
        if len(stamps) == 2:
            line.update(stamps)
        valid.append(line)
    return valid


# ================================================================= #
#  Авто-сканер: слайдеры флэта + личные тикеры
# ================================================================= #
def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


def _scanner_defaults() -> dict:
    return {
        "flat_slider": DEFAULT_FLAT_SLIDER,
        "slider_atr": None,
        "slider_bbw": None,
        "slider_pct": None,
        "flat_score_threshold": DEFAULT_FLAT_SCORE_THRESHOLD,
        "trend_strength_low": DEFAULT_TREND_STRENGTH_LOW,
        "filter_signals": True,
        "gate_exits": False,
        "tickers": {},
        "trailing_pct": {},
        "trailing_pct_personal": 0.0,
        "reverse_close": True,
    }


def _normalize_scanner(data: dict) -> dict:
    """Привести настройки авто-сканера к безопасному каноническому виду."""
    out = _scanner_defaults()

    def _slider(key: str) -> float | None:
        raw = data.get(key)
        if raw is None or raw == "":
            return None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return _clamp(v, 0.0, 1.0)

    def _threshold(key: str, default: float) -> float:
        raw = data.get(key, default)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(v):
            return default
        return _clamp(v, 0.0, 100.0)

    flat = _slider("flat_slider")
    out["flat_slider"] = DEFAULT_FLAT_SLIDER if flat is None else flat
    out["slider_atr"] = _slider("slider_atr")
    out["slider_bbw"] = _slider("slider_bbw")
    out["slider_pct"] = _slider("slider_pct")
    out["flat_score_threshold"] = _threshold("flat_score_threshold", DEFAULT_FLAT_SCORE_THRESHOLD)
    out["trend_strength_low"] = _threshold("trend_strength_low", DEFAULT_TREND_STRENGTH_LOW)

    # Фильтрация сигналов: bool из любого разумного представления
    fs = data.get("filter_signals", True)
    out["filter_signals"] = (
        fs is True or str(fs).strip().lower() in ("1", "true", "yes", "on", "да")
    )
    ge = data.get("gate_exits", False)
    out["gate_exits"] = (
        ge is True or str(ge).strip().lower() in ("1", "true", "yes", "on", "да")
    )

    # Личные тикеры по универсумам: только валидные, дедуп, кап
    tickers: dict[str, list[str]] = {}
    for univ, raw_list in (data.get("tickers") or {}).items():
        univ = str(univ).strip().lower()
        if univ not in SCANNER_UNIVERSES:
            continue
        seen: list[str] = []
        for t in (raw_list or []):
            s = str(t).strip().upper()
            if TICKER_RE.match(s) and s not in seen:
                seen.append(s)
            if len(seen) >= MAX_SCANNER_TICKERS:
                break
        tickers[univ] = seen
    out["tickers"] = tickers

    # Трейлинг-стоп: % по универсумам (авто-сканер; 0/отсутствует = выкл)
    trail: dict[str, float] = {}
    for univ, raw in (data.get("trailing_pct") or {}).items():
        u = str(univ).strip().lower()
        if u not in SCANNER_UNIVERSES:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            trail[u] = round(_clamp(v, 0.1, MAX_TRAILING_PCT), 2)
    out["trailing_pct"] = trail

    # Личный сигнальный сканер: единый % (0 = выключен)
    try:
        tp = float(data.get("trailing_pct_personal", 0.0))
    except (TypeError, ValueError):
        tp = 0.0
    out["trailing_pct_personal"] = (
        round(_clamp(tp, 0.0, MAX_TRAILING_PCT), 2) if math.isfinite(tp) else 0.0
    )

    # Принудительный разворот при противоположном входе
    rc = data.get("reverse_close", True)
    out["reverse_close"] = (
        rc is True or str(rc).strip().lower() in ("1", "true", "yes", "on", "да")
    )
    return out


# ================================================================= #
#  Роуты
# ================================================================= #
@router.get("/dashboard", response_model=DashboardSettingsOut)
def get_dashboard_settings(
    user: User | None = Depends(get_optional_user),
    db: Session = Depends(get_session),
):
    """Настройки дашборда текущего пользователя (или дефолт для анонима)."""
    if user is None:
        return _defaults()
    row = (
        db.query(UserDashboardSettings)
        .filter(UserDashboardSettings.user_id == user.id)
        .first()
    )
    if row is None:
        return _defaults()
    if not isinstance(row.data, dict):
        return _defaults()
    # Старые записи без timeframe — отдаём с дефолтным таймфреймом
    return {**_defaults(), **row.data}


@router.put("/dashboard", response_model=DashboardSettingsOut)
def put_dashboard_settings(
    body: DashboardSettingsIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Сохранить настройки дашборда пользователя (upsert)."""
    data = _normalize(body.model_dump())
    row = (
        db.query(UserDashboardSettings)
        .filter(UserDashboardSettings.user_id == user.id)
        .first()
    )
    if row is None:
        row = UserDashboardSettings(user_id=user.id, data=data)
        db.add(row)
    else:
        row.data = data
    db.commit()
    return data


# ── Авто-сканер: верификация боковика + личные тикеры ──────────
def load_scanner_settings(db: Session, user: User | None) -> dict:
    """Настройки авто-сканера для пользователя (дефолты для анонима).

    Используется и роутом ``GET /auth/settings/scanner``, и роутером
    авто-сканера (вердикт по личному слайдеру на чтении).
    """
    if user is None:
        return _scanner_defaults()
    row = (
        db.query(UserScannerSettings)
        .filter(UserScannerSettings.user_id == user.id)
        .first()
    )
    if row is None or not isinstance(row.data, dict):
        return _scanner_defaults()
    return {**_scanner_defaults(), **row.data}


@router.get("/scanner", response_model=ScannerSettingsOut)
def get_scanner_settings(
    user: User | None = Depends(get_optional_user),
    db: Session = Depends(get_session),
):
    """Настройки авто-сканера текущего пользователя (или дефолт)."""
    return load_scanner_settings(db, user)


@router.put("/scanner", response_model=ScannerSettingsOut)
def put_scanner_settings(
    body: ScannerSettingsIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Сохранить настройки авто-сканера пользователя (upsert)."""
    data = _normalize_scanner(body.model_dump())
    row = (
        db.query(UserScannerSettings)
        .filter(UserScannerSettings.user_id == user.id)
        .first()
    )
    if row is None:
        row = UserScannerSettings(user_id=user.id, data=data)
        db.add(row)
    else:
        row.data = data
    db.commit()
    return data
