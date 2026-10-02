"""Отношения и производные метрики из XBRL: чистые функции (ring: application).

Почему отдельный модуль
-----------------------
Метрики считались внутри ``sec_forecast.py`` вместе с регрессиями, оценкой и сборкой
ответа. Здесь только арифметика отчётности, и она проверяется без данных из сети.

Общее правило всех функций: **``None`` вместо мнимого числа**. Делить на ноль, брать
логарифм отрицательного или «рост» из одной точки нельзя — но и падать из-за этого
нельзя: отсутствующая метрика это норма (компания могла не подать статью), и вызывающий
обязан увидеть пропуск, а не правдоподобное число.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = [
    "cagr",
    "debt_to_equity",
    "free_cash_flow",
    "growth",
    "margin",
    "pe_ratio",
    "safe_div",
]


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Деление, которое не падает и не выдаёт бесконечность.

    ``None`` при нулевом/отсутствующем знаменателе: «долг / 0» — это не «очень много долга»,
    а отсутствие смысла, и подставлять сюда большое число значит испортить сортировку.
    """
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return None


def margin(
    numerator: Optional[float],
    revenue: Optional[float],
    *,
    percent: bool = True,
) -> Optional[float]:
    """Отношение статьи к выручке (по умолчанию в процентах).

    Убыток даёт отрицательную маржу — это факт, а не ошибка данных, поэтому знак не
    отбрасывается.
    """
    value = safe_div(numerator, revenue)
    if value is None:
        return None
    return round(value * 100.0, 2) if percent else round(value, 4)


def growth(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Рост относительно предыдущего периода, в процентах.

    Отрицательный знаменатель не считается: «рост к убытку» меняет знак и читается
    наоборот, поэтому такое отношение отдаётся как ``None``.
    """
    if current is None or previous is None or previous <= 0:
        return None
    return round((float(current) - float(previous)) / float(previous) * 100.0, 2)


def cagr(series: Sequence[float], period_years: Optional[float] = None) -> Optional[float]:
    """Среднегодовой рост, в процентах.

    ``period_years`` — длина периода; по умолчанию ``len(series) - 1`` (ряд по годам).
    Считать CAGR нельзя при неположительных концах ряда: корень степени из отрицательного
    значения не определён, а приближение «на глаз» даёт неверный знак.
    """
    if len(series) < 2:
        return None
    first, last = float(series[0]), float(series[-1])
    years = float(period_years) if period_years else float(len(series) - 1)
    if years <= 0 or first <= 0 or last <= 0:
        return None
    return round(((last / first) ** (1.0 / years) - 1.0) * 100.0, 2)


def debt_to_equity(debt: Optional[float], equity: Optional[float]) -> Optional[float]:
    """Долг к собственному капиталу. При неположительном капитале — ``None``.

    Компания с отрицательным капиталом технически «должна больше, чем стоит»; отношение
    здесь превращается в бессмысленное отрицательное число, а не в знак здоровья.
    """
    if equity is None or equity <= 0:
        return None
    return round(safe_div(debt, equity) or 0.0, 4) if debt is not None else None


def free_cash_flow(operating: Optional[float], capex: Optional[float]) -> Optional[float]:
    """Свободный поток: операционный минус капитальные затраты.

    ``capex`` у SEC подаётся положительным числом (расход), поэтому он вычитается как есть.
    """
    if operating is None:
        return None
    if capex is None:
        return float(operating)
    return float(operating) - abs(float(capex))


def pe_ratio(price: Optional[float], eps: Optional[float]) -> Optional[float]:
    """P/E. При нулевой или отрицательной прибыли на акцию — ``None`` (не «отрицательный P/E»)."""
    if eps is None or eps <= 0:
        return None
    return round(safe_div(price, eps) or 0.0, 2) if price is not None else None
