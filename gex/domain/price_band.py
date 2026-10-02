"""Доверительный интервал цены по ключевым GEX-уровням (price band).

Опциональная метрика для отображения на GEX-профиле: диапазон, в котором цена
инструмента с высокой вероятностью останется, ограниченный **ключевыми**
GEX-уровнями по объёму (Open Interest), а не крайними страйками цепочки.

Логика (по ТЗ):
  * Берём страйки в окне ±10% (индексы) / ±15-20% (акции и крипта) от spot.
  * Не используем самые крайние значения — ищем **ключевые** уровни по OI/AG:
    сильнейший положительный (Call Wall — сопротивление сверху) и сильнейший
    отрицательный (Put Wall — поддержка снизу) внутри окна.
  * Доверительный интервал = [ключевая поддержка, ключевое сопротивление].
  * Дополнительно: медиана/центр диапазона, ширина в %, мощность границ
    (доля OI/AG относительно окна).

Опора на GEX + ключевые объёмы даёт «жёсткие» границы: там, где сосредоточен
открытый интерес и где гамма меняет знак, дилеры активно хеджируют и цена
тормозит/разворачивается.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gex.application.extended import ExtendedGEXReport, ExtendedStrike

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Пороговые константы
# ====================================================================== #
# Ширина окна поиска по умолчанию (в долях от spot) по типу актива.
_WINDOW_PCT = {
    "crypto": 0.20,   # ±20% для крипты (высокая волатильность)
    "stock": 0.15,    # ±15% для акций
    "index": 0.10,    # ±10% для индексов (SPX/SPY/QQQ — низкая вола)
}
# Минимальное число страйков в окне для построения полосы.
_MIN_STRIKES = 4
# Доля OI/AG страйка от максимума в окне, чтобы считаться «ключевым».
_KEY_LEVEL_OI_SHARE = 0.15    # ≥15% от max OI в окне
_KEY_LEVEL_AG_SHARE = 0.30    # ≥30% от max AG в окне


# ====================================================================== #
#  Dataclass результата
# ====================================================================== #
@dataclass
class PriceBand:
    """Доверительный интервал цены по ключевым GEX-уровням."""

    low: float                          # нижняя граница (ключевая поддержка)
    high: float                         # верхняя граница (ключевое сопротивление)
    center: float                       # центр диапазона
    width_pct: float                    # ширина полосы в % от spot
    low_strength: float                 # |Net GEX| нижней границы
    high_strength: float                # |Net GEX| верхней границы
    low_label: str                      # тип нижней границы («Put Wall», «Positive island»…)
    high_label: str                     # тип верхней границы
    window_pct: float                   # использованное окно (±% от spot)
    n_strikes_in_window: int            # число страйков в окне
    key_levels: list[dict] = field(default_factory=list)  # топ-ключевые уровни
    methodology: str = ""               # описание метода


# ====================================================================== #
#  Главная функция
# ====================================================================== #
def compute_price_band(report: ExtendedGEXReport) -> Optional[PriceBand]:
    """Построить доверительный интервал цены по ключевым GEX-уровням.

    Returns
    -------
    PriceBand | None
        ``None``, если в окне слишком мало страйков для устойчивой оценки.
    """
    spot = report.spot
    if not np.isfinite(spot) or spot <= 0 or not report.per_strike:
        return None

    # --- 1. Выбор окна по типу актива ---
    window_pct = _window_for(report)
    lo_price = spot * (1 - window_pct)
    hi_price = spot * (1 + window_pct)

    # --- 2. Страйки в окне ---
    in_window = [
        s for s in report.per_strike
        if lo_price <= s.strike <= hi_price
    ]
    if len(in_window) < _MIN_STRIKES:
        logger.debug(
            "price_band: мало страйков в окне ±%.0f%% для %s (%d < %d)",
            window_pct * 100, report.symbol, len(in_window), _MIN_STRIKES,
        )
        return None

    # --- 3. Ключевые уровни по OI/AG ---
    # Сопротивление сверху: сильнейший страйк с положительным Net GEX выше spot.
    # Поддержка снизу: сильнейший страйк с отрицательным Net GEX ниже spot.
    above = [s for s in in_window if s.strike >= spot]
    below = [s for s in in_window if s.strike <= spot]

    resistance = _strongest_wall(above, positive=True, spot=spot)
    support = _strongest_wall(below, positive=False, spot=spot)

    # Fallback: если нет положительной стены выше — берём Call Wall из отчёта
    # (если он в окне); аналогично для поддержки.
    if resistance is None and np.isfinite(report.call_wall):
        cw = report.call_wall
        if lo_price <= cw <= hi_price:
            resistance = _level_from_strike(
                next((s for s in in_window if math.isclose(s.strike, cw, rel_tol=1e-4)),
                     in_window[-1]),
                label="Call Wall (report)",
            )
    if support is None and np.isfinite(report.put_wall):
        pw = report.put_wall
        if lo_price <= pw <= hi_price:
            support = _level_from_strike(
                next((s for s in in_window if math.isclose(s.strike, pw, rel_tol=1e-4)),
                     in_window[0]),
                label="Put Wall (report)",
            )

    # Если всё ещё нет одной из границ — берём край окна как мягкую границу.
    if support is None:
        support = _soft_boundary(in_window[0], "граница окна (нет Put Wall)")
    if resistance is None:
        resistance = _soft_boundary(in_window[-1], "граница окна (нет Call Wall)")

    low = support["price"]
    high = resistance["price"]
    if high <= low:
        # вырожденный случай — раздвинем на минимум
        high = max(high, low * 1.001)

    center = (low + high) / 2.0
    width_pct = (high - low) / spot * 100.0

    # --- 4. Топ-ключевые уровни для отображения (по AG, до 5) ---
    key_levels = _top_key_levels(in_window, k=5)

    return PriceBand(
        low=float(low),
        high=float(high),
        center=float(center),
        width_pct=float(width_pct),
        low_strength=float(support["strength"]),
        high_strength=float(resistance["strength"]),
        low_label=support["label"],
        high_label=resistance["label"],
        window_pct=float(window_pct * 100),
        n_strikes_in_window=len(in_window),
        key_levels=key_levels,
        methodology=(
            "Границы = ключевые GEX-уровни по объёму (OI/AG) в окне ±"
            f"{window_pct*100:.0f}% от spot. Не крайние страйки, а сильнейшие "
            "Call Wall (сопротивление) и Put Wall (поддержка) внутри окна."
        ),
    )


# ====================================================================== #
#  Вспомогательные функции
# ====================================================================== #
def _window_for(report: ExtendedGEXReport) -> float:
    """Выбор окна по типу актива (crypto/stock/index).

    Эвристика «индекс»: тикеры SPX/SPY/QQQ/IWM/DIA/^VIX (низкая вола).
    Прочие US-акции — stock. Крипта — по report.source.
    """
    if report.source == "crypto":
        return _WINDOW_PCT["crypto"]
    sym = (report.symbol or "").upper()
    index_tickers = {"SPX", "SPY", "QQQ", "IWM", "DIA", "^VIX", "VIX", "VVIX",
                     "MDY", "RSP", "SMH"}
    if sym in index_tickers:
        return _WINDOW_PCT["index"]
    return _WINDOW_PCT["stock"]


def _strongest_wall(
    strikes: list[ExtendedStrike],
    positive: bool,
    spot: float,
) -> Optional[dict]:
    """Найти сильнейшую стену заданного знака среди страйков.

    «Сила» = OI-взвешенная: учитываем и |Net GEX|, и суммарный OI. Берём страйк
    с макс |Net GEX| среди удовлетворяющих знаку — это и есть ключевая стена.
    Дополнительно проверяем, что уровень достаточно «плотный» (OI/AG ≥ порога).
    """
    if not strikes:
        return None
    candidates = [s for s in strikes if (s.gex_net > 0) == positive]
    if not candidates:
        return None
    # Сильнейший по |Net GEX|.
    best = max(candidates, key=lambda s: abs(s.gex_net))
    # Проверка «ключевости» по AG-доли (опционально — если не выполняется,
    # всё равно возвращаем, это лучшая стена в окне).
    max_ag = max((s.ag for s in strikes), default=0.0)
    is_key = max_ag > 0 and (best.ag / max_ag) >= _KEY_LEVEL_AG_SHARE

    label = ("Call Wall" if positive else "Put Wall") + (
        " (ключевой)" if is_key else " (лучшая в окне)"
    )
    return {
        "price": float(best.strike),
        "strength": float(abs(best.gex_net)),
        "oi": float(best.oi_call + best.oi_put),
        "ag": float(best.ag),
        "label": label,
        "is_key": is_key,
    }


def _level_from_strike(s: ExtendedStrike, label: str) -> dict:
    return {
        "price": float(s.strike),
        "strength": float(abs(s.gex_net)),
        "oi": float(s.oi_call + s.oi_put),
        "ag": float(s.ag),
        "label": label,
        "is_key": True,
    }


def _soft_boundary(s: ExtendedStrike, label: str) -> dict:
    """Мягкая граница окна — используется, когда ключевой стены нет."""
    return {
        "price": float(s.strike),
        "strength": float(abs(s.gex_net)),
        "oi": float(s.oi_call + s.oi_put),
        "ag": float(s.ag),
        "label": label,
        "is_key": False,
    }


def _top_key_levels(strikes: list[ExtendedStrike], k: int = 5) -> list[dict]:
    """Топ-k страйков по AG (Aggregate Gamma) в окне — для отображения на чарте."""
    sorted_by_ag = sorted(strikes, key=lambda s: s.ag, reverse=True)[:k]
    return [
        {
            "strike": float(s.strike),
            "gex_net": float(s.gex_net),
            "ag": float(s.ag),
            "oi_total": float(s.oi_call + s.oi_put),
            "type": "RESISTANCE" if s.gex_net > 0 else "SUPPORT",
        }
        for s in sorted_by_ag
    ]
