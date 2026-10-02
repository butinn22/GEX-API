"""Трендовые линии + анализ тренда (порт Pinescript v5 «Trend lines Andreu»).

Перенос стратегии построения трендовых линий из Pinescript v5
(``trendlines.TXT``) на Python + два способа определения тренда:

1. **По углу трендовых линий** — линия, построенная через пару экстремумов
   ``(x1, p1) → (x2, p2)``, экстраполируется на текущий бар; угол наклона
   ``atan(Δprice / Δbars)`` (нормированный через ATR) даёт направление тренда.
   Восходящая линия поддержки (положительный угол) — бычий сигнал, нисходящая
   линия сопротивления (отрицательный угол) — медвежий.

2. **По фракталам HigherHigh/HigherLow (HH/HL) и LowerHigh/LowerLow (LH/LL)** —
   классическая свинг-структура: последовательные максимумы выше (HH), а
   минимумы выше (HL) → восходящий тренд; LH+LL → нисходящий. Реализован через
   ``ta.pivothigh/pivotlow``-логику (fractal detection по окну ``left/right``).

Канон Pinescript-алгоритма
--------------------------
* ``resolution`` (``x1``) — окно поиска экстремумов, по умолчанию 6;
* ``minimums/maximums`` — смещение (в барах назад) до ближайшего минимума/
  максимума в окне ``x1``; сбрасывается в ``x1//2``, когда минимум окна
  приходится ровно на центр;
* линия валидна, если её экстраполяция на текущий бар лежит **ниже** high[1]
  (поддержка) или **выше** low[1] (сопротивление), и **не пересекается**
  телами промежуточных свечей;
* ограничение ``max_support_lines`` / ``max_resistance_lines`` (по умолчанию 5).

Все функции работают с DataFrame ``Open, High, Low, Close`` (как
:data:`gex.ta_fetcher.TIMEFRAMES`-фетчеры) и не имеют побочных эффектов.

Пример::

    from gex.domain.trendlines import analyze_trendlines
    result = analyze_trendlines(df, timeframe="1d")
    result.trend_direction   # "BULLISH" | "BEARISH" | "RANGE"
    result.fractal_trend     # "BULLISH" | "BEARISH" | "RANGE"
    result.support_lines     # list[Trendline]
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────── #
#  Dataclasses — результат анализа трендовых линий по одному таймфрейму
# ────────────────────────────────────────────────────────────────────── #
@dataclass
class Trendline:
    """Одна построенная трендовая линия (поддержка или сопротивление).

    Координаты — в системе (bar_index, price): ``x1`` всегда дальше в прошлое
    (старше), ``x2`` — ближе к настоящему. Линия экстраполируется вправо
    (``extend=right``), как в оригинальном Pinescript.

    Attributes
    ----------
    kind : str
        ``"support"`` (поддержка, зелёная) или ``"resistance"`` (сопротивление,
        розовая).
    x1, x2 : int
        Бар-индексы точек линии (``bar_index``, как в Pine — 0..N-1 от начала
        серии). ``x1 < x2``.
    price1, price2 : float
        Цены (low/high) в точках линии.
    angle_deg : float
        Наклон линии в градусах (``atan(Δprice/Δbars)``). Положительный угол =
        линия смотрит вверх, отрицательный = вниз.
    slope : float
        Чистый наклон ``Δprice / Δbars`` (price per bar).
    current_price : float
        Экстраполированное значение линии на последний бар.
    """

    kind: str  # "support" | "resistance"
    x1: int
    x2: int
    price1: float
    price2: float
    angle_deg: float
    slope: float
    current_price: float

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x1": int(self.x1),
            "x2": int(self.x2),
            "price1": float(self.price1),
            "price2": float(self.price2),
            "angle_deg": round(float(self.angle_deg), 4),
            "slope": float(self.slope),
            "current_price": float(self.current_price),
        }


@dataclass
class FractalPoint:
    """Свинг-точка (фрактал) для HH/HL/LH/LL-анализа."""

    idx: int            # бар-индекс
    price: float        # цена свинга (high или low)
    kind: str           # "high" | "low"


@dataclass
class FractalStructure:
    """Структура свингов: последовательности HH/HL/LH/LL."""

    swing_highs: list[FractalPoint] = field(default_factory=list)
    swing_lows: list[FractalPoint] = field(default_factory=list)
    higher_highs: int = 0
    lower_highs: int = 0
    higher_lows: int = 0
    lower_lows: int = 0

    @property
    def last_high(self) -> Optional[FractalPoint]:
        return self.swing_highs[-1] if self.swing_highs else None

    @property
    def last_low(self) -> Optional[FractalPoint]:
        return self.swing_lows[-1] if self.swing_lows else None


@dataclass
class TrendlineAnalysis:
    """Полный результат анализа по одному таймфрейму.

    Объединяет два независимых определения тренда:
    * :attr:`trend_direction` — по углу трендовых линий (порт Pine-логики);
    * :attr:`fractal_trend` — по свинг-структуре HH/HL/LH/LL.

    Attributes
    ----------
    timeframe : str
    last_close : float
        Цена закрытия последнего бара.
    support_lines, resistance_lines : list[Trendline]
        Построенные линии (до ``max_*_lines``).
    strongest_support, strongest_resistance : Optional[Trendline]
        Самая значимая (по модулю угла / близости к цене) линия каждого типа.
    trend_direction : str
        Консенсус по трендовым линиям: ``BULLISH`` / ``BEARISH`` / ``RANGE``.
    trend_strength : float
        Сила тренда по линиям, 0..100 (по модулю нормированного угла и числу
        согласных линий).
    line_angle_deg : float
        Усреднённый угол основных линий (знак задаёт направление).
    fractals : FractalStructure
        Свинг-структура.
    fractal_trend : str
        Тренд по фракталам HH/HL/LH/LL.
    fractal_strength : float
        Сила фрактального тренда, 0..100.
    combined_trend : str
        Итоговый тренд (согласие обоих методов), ``BULLISH`` / ``BEARISH`` /
        ``RANGE``.
    atr : float
        ATR(14) — для нормировки углов и оценки ширины зон.
    """

    timeframe: str
    last_close: float
    support_lines: list[Trendline] = field(default_factory=list)
    resistance_lines: list[Trendline] = field(default_factory=list)
    strongest_support: Optional[Trendline] = None
    strongest_resistance: Optional[Trendline] = None
    trend_direction: str = "RANGE"
    trend_strength: float = 0.0
    line_angle_deg: float = 0.0
    fractals: FractalStructure = field(default_factory=FractalStructure)
    fractal_trend: str = "RANGE"
    fractal_strength: float = 0.0
    combined_trend: str = "RANGE"
    combined_strength: float = 0.0
    atr: float = 0.0
    n_bars: int = 0


# ────────────────────────────────────────────────────────────────────── #
#  Геометрия: цена на линии, угол, проверка пересечений
# ────────────────────────────────────────────────────────────────────── #
def price_at(t1: float, p1: float, t2: float, p2: float, t3: float) -> float:
    """Экстраполяция линейной функции по двум точкам.

    Прямой порт ``price_at(t1, p1, t2, p2, t3)`` из Pinescript::

        p1 + (p2 - p1) * (t3 - t1) / (t2 - t1)

    При ``t2 == t1`` (вырожденная линия) возвращается ``p1``.
    """
    if t2 == t1:
        return float(p1)
    return float(p1 + (p2 - p1) * (t3 - t1) / (t2 - t1))


def line_angle_deg(x1: int, p1: float, x2: int, p2: float) -> float:
    """Угол наклона линии в градусах: ``atan(Δprice / Δbars)``.

    Положительный угол = линия смотрит вверх, отрицательный = вниз.
    """
    dx = x2 - x1
    if dx == 0:
        return 0.0
    return math.degrees(math.atan((p2 - p1) / dx))


def line_slope(x1: int, p1: float, x2: int, p2: float) -> float:
    """Наклон линии (price per bar): ``(p2 - p1) / (x2 - x1)``."""
    dx = x2 - x1
    if dx == 0:
        return 0.0
    return (p2 - p1) / dx


# ────────────────────────────────────────────────────────────────────── #
#  Экстремумы: минимумы/максимумы по окну (порт Pine minimums/maximums)
# ────────────────────────────────────────────────────────────────────── #
def _compute_minimums(low: np.ndarray, resolution: int) -> np.ndarray:
    """Серия ``minimums`` — смещение (в барах назад) до ближайшего минимума.

    Прямой порт Pinescript::

        minimums := ta.lowestbars(min_values, x1) == -x2 ? x2 : minimums[1] + 1

    ``ta.lowestbars(low, x1)`` возвращает отрицательное смещение (в барах назад)
    до минимального значения в окне длиной ``x1``. Если этот минимум приходится
    ровно на ``-x2`` (центр окна), сбрасываем ``minimums = x2``; иначе
    увеличиваем на 1 (минимум «уходит» вправо).

    Возвращает массив целых смещений той же длины, что ``low``.
    """
    n = len(low)
    x2 = resolution // 2
    minimums = np.zeros(n, dtype=np.int64)
    for i in range(n):
        # Окно low[i - resolution + 1 .. i] (x1 баров, включая i).
        lo = i - resolution + 1
        if lo < 0:
            # Недостаточно истории в начале серии — не подтверждённый минимум.
            minimums[i] = minimums[i - 1] + 1 if i > 0 else 0
            continue
        window = low[lo : i + 1]
        # ta.lowestbars возвращает смещение от конца окна (отрицательное).
        # argmin даёт позицию минимума от начала окна; смещение от конца =
        # -(resolution - 1 - argmin) = argmin - (resolution - 1).
        argmin = int(np.argmin(window))
        offset = argmin - (resolution - 1)  # отрицательное, как lowestbars
        if offset == -x2:
            minimums[i] = x2
        else:
            minimums[i] = (minimums[i - 1] + 1) if i > 0 else 0
    return minimums


def _compute_maximums(high: np.ndarray, resolution: int) -> np.ndarray:
    """Серия ``maximums`` — смещение до ближайшего максимума (порт Pine)."""
    n = len(high)
    x2 = resolution // 2
    maximums = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lo = i - resolution + 1
        if lo < 0:
            maximums[i] = maximums[i - 1] + 1 if i > 0 else 0
            continue
        window = high[lo : i + 1]
        argmax = int(np.argmax(window))
        offset = argmax - (resolution - 1)
        if offset == -x2:
            maximums[i] = x2
        else:
            maximums[i] = (maximums[i - 1] + 1) if i > 0 else 0
    return maximums


# ────────────────────────────────────────────────────────────────────── #
#  Построение трендовых линий (порт Pine — Support/Resistance)
# ────────────────────────────────────────────────────────────────────── #
def _build_support_lines(
    low: np.ndarray,
    high: np.ndarray,
    open_: np.ndarray,
    close_: np.ndarray,
    minimums: np.ndarray,
    *,
    history_bars: int,
    max_lines: int,
) -> list[Trendline]:
    """Построение линий поддержки — порт блока ``// Support`` из Pinescript.

    Алгоритм ищет пары подтверждённых минимумов (``minimum1`` ближе к настоящему,
    ``minimum2`` дальше в прошлое), проводит линию и проверяет:
    1. Экстраполяция на текущий бар лежит **ниже** ``high[1]`` (максимум
       предыдущего бара).
    2. Линия не пересекается телами промежуточных свечей (``min(open, close)``).
    Берётся самая верхняя валидная линия для каждой ближней точки (как в
    оригинале — ``last_line`` c заменой по ``bar1 == x2``).
    """
    n = len(low)
    last_bar = n - 1
    prev_bar = last_bar - 1
    lines: list[Trendline] = []

    # Pine-цикл: minimum1, minimum2 — смещения «назад» от last_bar.
    minimum1 = 0
    for _ in range(51):
        if minimum1 >= history_bars:
            break
        minimum1 += int(minimums[last_bar - minimum1])
        if minimum1 <= 0 or minimum1 >= history_bars:
            break
        minimum2 = minimum1 * 2
        for _ in range(51):
            if minimum2 >= minimum1 * 8 or minimum2 >= history_bars:
                break
            minimum2 += int(minimums[last_bar - minimum2])

            if minimum1 >= history_bars or minimum2 >= history_bars:
                break
            if minimum2 <= minimum1:
                continue

            bar1 = last_bar - minimum1
            bar2 = last_bar - minimum2
            price1 = float(low[bar1])
            price2 = float(low[bar2])

            current_price = price_at(bar2, price2, bar1, price1, last_bar)
            # Pine: ``if current_price < high[1]`` — линия ниже максимума prev бара.
            if prev_bar < 0 or current_price >= float(high[prev_bar]):
                continue

            # Проверка пересечений телами свечей (Pine: medium-цикл по minimums).
            medium = 0
            crossed = False
            for _ in range(51):
                if medium >= minimum2:
                    break
                medium += int(minimums[last_bar - medium])
                if medium >= minimum2:
                    break
                line_val = price_at(bar2, price2, bar1, price1, last_bar - medium)
                body_min = min(float(open_[last_bar - medium]), float(close_[last_bar - medium]))
                if line_val > body_min:
                    crossed = True
                    break

            if crossed:
                continue

            angle = line_angle_deg(bar2, price2, bar1, price1)
            slope = line_slope(bar2, price2, bar1, price1)
            tl = Trendline(
                kind="support",
                x1=bar2, x2=bar1, price1=price2, price2=price1,
                angle_deg=angle, slope=slope, current_price=current_price,
            )
            # Дедупликация: среди линий с тем же ближним якорем bar1 оставляем
            # самую верхнюю (Pine: ``if current_price > last_price: set_xy``).
            replaced = False
            for k, ex in enumerate(lines):
                if ex.x2 == bar1:
                    if current_price > ex.current_price:
                        lines[k] = tl
                    replaced = True
                    break
            if not replaced:
                lines.append(tl)
                if len(lines) > max_lines:
                    lines.pop(0)
    return lines


def _build_resistance_lines(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close_: np.ndarray,
    maximums: np.ndarray,
    *,
    history_bars: int,
    max_lines: int,
) -> list[Trendline]:
    """Построение линий сопротивления — порт блока ``// Resistance`` из Pine.

    Зеркально к :func:`_build_support_lines`: линия должна лежать **выше**
    ``low[1]`` (минимум предыдущего бара) и не пересекаться телами свечей.
    """
    n = len(high)
    last_bar = n - 1
    prev_bar = last_bar - 1
    lines: list[Trendline] = []

    maximum1 = 0
    for _ in range(101):
        if maximum1 >= history_bars:
            break
        maximum1 += int(maximums[last_bar - maximum1])
        if maximum1 <= 0 or maximum1 >= history_bars:
            break
        maximum2 = maximum1 * 2
        for _ in range(51):
            if maximum2 >= maximum1 * 8 or maximum2 >= history_bars:
                break
            maximum2 += int(maximums[last_bar - maximum2])

            if maximum1 >= history_bars or maximum2 >= history_bars:
                break
            if maximum2 <= maximum1:
                continue

            bar1 = last_bar - maximum1
            bar2 = last_bar - maximum2
            price1 = float(high[bar1])
            price2 = float(high[bar2])

            current_price = price_at(bar2, price2, bar1, price1, last_bar)
            # Pine: ``if current_price > low[1]``.
            if prev_bar < 0 or current_price <= float(low[prev_bar]):
                continue

            # Проверка пересечений телами свечей.
            medium = 0
            crossed = False
            for _ in range(101):
                if medium >= maximum2:
                    break
                medium += int(maximums[last_bar - medium])
                if medium >= maximum2:
                    break
                line_val = price_at(bar2, price2, bar1, price1, last_bar - medium)
                body_max = max(float(open_[last_bar - medium]), float(close_[last_bar - medium]))
                if line_val < body_max:
                    crossed = True
                    break

            if crossed:
                continue

            angle = line_angle_deg(bar2, price2, bar1, price1)
            slope = line_slope(bar2, price2, bar1, price1)
            tl = Trendline(
                kind="resistance",
                x1=bar2, x2=bar1, price1=price2, price2=price1,
                angle_deg=angle, slope=slope, current_price=current_price,
            )
            # Дедупликация: среди линий с тем же ближним якорем bar1 оставляем
            # самую нижнюю (Pine: ``if current_price < last_price: set_xy``).
            replaced = False
            for k, ex in enumerate(lines):
                if ex.x2 == bar1:
                    if current_price < ex.current_price:
                        lines[k] = tl
                    replaced = True
                    break
            if not replaced:
                lines.append(tl)
                if len(lines) > max_lines:
                    lines.pop(0)
    return lines


# ────────────────────────────────────────────────────────────────────── #
#  ATR (для нормировки углов и силы тренда)
# ────────────────────────────────────────────────────────────────────── #
def _wilder_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR Уайлдера как число для последнего бара (для нормировки)."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    if not np.isfinite(val) or val <= 0:
        val = float((high - low).tail(period).mean()) if len(df) >= 1 else 1.0
    return val if val > 0 else 1.0


# ────────────────────────────────────────────────────────────────────── #
#  Фракталы: pivot high / pivot low (HH/HL/LH/LL)
# ────────────────────────────────────────────────────────────────────── #
def find_pivot_highs(high: np.ndarray, left: int, right: int) -> list[FractalPoint]:
    """Найти swing-highs: бар, чей high максимален в окне ``[i-left, i+right]``.

    Прямой аналог ``ta.pivothigh(source, left, right)`` из Pinescript: точка
    подтверждается только когда прошло ``right`` баров вправо. Возвращает
    подтверждённые фракталы, отсортированные по времени.
    """
    n = len(high)
    pts: list[FractalPoint] = []
    for i in range(left, n - right):
        center = high[i]
        if np.isnan(center):
            continue
        window = high[i - left : i + right + 1]
        # Строго больше всех соседей (как Pine pivothigh: левая часть строго
        # меньше, правая — меньше либо равна, но канонически используем >=).
        if center >= window.max() and np.sum(window == center) == 1:
            pts.append(FractalPoint(idx=int(i), price=float(center), kind="high"))
    return pts


def find_pivot_lows(low: np.ndarray, left: int, right: int) -> list[FractalPoint]:
    """Найти swing-lows — аналог ``ta.pivotlow(source, left, right)``."""
    n = len(low)
    pts: list[FractalPoint] = []
    for i in range(left, n - right):
        center = low[i]
        if np.isnan(center):
            continue
        window = low[i - left : i + right + 1]
        if center <= window.min() and np.sum(window == center) == 1:
            pts.append(FractalPoint(idx=int(i), price=float(center), kind="low"))
    return pts


def classify_fractals(
    swing_highs: list[FractalPoint], swing_lows: list[FractalPoint]
) -> FractalStructure:
    """Классифицировать свинги в HH/HL/LH/LL-структуру.

    * Higher High (HH): текущий свинг-хай выше предыдущего.
    * Lower High (LH): текущий ниже предыдущего.
    * Higher Low (HL): текущий свинг-лоу выше предыдущего.
    * Lower Low (LL): текущий ниже предыдущего.
    """
    struct = FractalStructure(swing_highs=list(swing_highs), swing_lows=list(swing_lows))
    for j in range(1, len(swing_highs)):
        if swing_highs[j].price > swing_highs[j - 1].price:
            struct.higher_highs += 1
        elif swing_highs[j].price < swing_highs[j - 1].price:
            struct.lower_highs += 1
    for j in range(1, len(swing_lows)):
        if swing_lows[j].price > swing_lows[j - 1].price:
            struct.higher_lows += 1
        elif swing_lows[j].price < swing_lows[j - 1].price:
            struct.lower_lows += 1
    return struct


def fractal_trend_direction(struct: FractalStructure) -> tuple[str, float]:
    """Определить направление и силу тренда по фрактальной структуре.

    Логика (канон HH/HL/LH/LL):
    * **BULLISH**: есть хотя бы один HH и один HL (последовательное повышение).
    * **BEARISH**: есть хотя бы один LH и один LL (последовательное понижение).
    * иначе **RANGE**.

    Сила 0..100 считается как долю согласующихся свингов от общего числа.
    """
    bull = struct.higher_highs + struct.higher_lows
    bear = struct.lower_highs + struct.lower_lows
    total = bull + bear
    if total == 0:
        return "RANGE", 0.0

    is_bull = struct.higher_highs >= 1 and struct.higher_lows >= 1
    is_bear = struct.lower_highs >= 1 and struct.lower_lows >= 1

    if is_bull and not is_bear:
        direction = "BULLISH"
    elif is_bear and not is_bull:
        direction = "BEARISH"
    elif is_bull and is_bear:
        # Смешанная структура — победитель по большинству.
        direction = "BULLISH" if bull > bear else ("BEARISH" if bear > bull else "RANGE")
    else:
        direction = "BULLISH" if bull > bear else ("BEARISH" if bear > bull else "RANGE")

    strength = max(bull, bear) / total * 100.0
    return direction, round(strength, 1)


# ────────────────────────────────────────────────────────────────────── #
#  Тренд по углу трендовых линий
# ────────────────────────────────────────────────────────────────────── #
def lines_trend_direction(
    support_lines: list[Trendline],
    resistance_lines: list[Trendline],
    last_close: float,
    atr: float,
) -> tuple[str, float, float]:
    """Определить тренд по углам трендовых линий.

    Наклон каждой линии переводится в «нормированный на ATR» показатель
    (чтобы сравнивать активы с разной волатильностью)::

        norm_slope = (slope / atr) * 100   # процентов ATR за бар

    Знак наклона **сам по себе** кодирует направление: восходящая линия
    (положительный угол) = бычья, нисходящая (отрицательный угол) = медвежья —
    это справедливо и для поддержки, и для сопротивления. Все линии голосуют
    своим нормированным наклоном; сумма голосов даёт направление и силу.

    Возвращает ``(direction, strength_0_100, avg_angle_deg)``.
    """
    if atr <= 0:
        atr = 1.0
    votes: list[float] = []  # знаковые веса (нормированный наклон каждой линии)
    angles: list[float] = []

    for tl in support_lines:
        votes.append((tl.slope / atr) * 100.0)
        angles.append(tl.angle_deg)
    for tl in resistance_lines:
        votes.append((tl.slope / atr) * 100.0)
        angles.append(tl.angle_deg)

    if not votes:
        return "RANGE", 0.0, 0.0

    avg_angle = float(np.mean(angles)) if angles else 0.0
    net = float(np.sum(votes))

    # Порог нормированного наклона ~0.1% ATR/бар — граница «содержательного»
    # тренда (эмпирически). Ниже — считаем RANGE.
    threshold = 0.1
    if net > threshold:
        direction = "BULLISH"
    elif net < -threshold:
        direction = "BEARISH"
    else:
        direction = "RANGE"

    # Сила: насыщающаяся функция от модуля net (1% ATR/бар ≈ сильный тренд).
    strength = min(100.0, abs(net) / 1.0 * 100.0)
    return direction, round(strength, 1), round(avg_angle, 2)


# ────────────────────────────────────────────────────────────────────── #
#  Главный API: полный анализ по одному таймфрейму
# ────────────────────────────────────────────────────────────────────── #
def analyze_trendlines(
    df: pd.DataFrame,
    *,
    timeframe: str,
    resolution: int = 6,
    history_bars: int = 300,
    max_support_lines: int = 5,
    max_resistance_lines: int = 5,
    pivot_left: int = 5,
    pivot_right: int = 5,
) -> TrendlineAnalysis:
    """Полный анализ трендовых линий и тренда по одному таймфрейму.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV с колонками ``Open, High, Low, Close`` (минимум 2*resolution
        баров; рекомендуется 200+ для сходимости фракталов).
    timeframe : str
        Метка таймфрейма (``"1h"`` / ``"4h"`` / ``"1d"`` …) — попадает в ответ.
    resolution : int
        Окно поиска экстремумов ``x1`` (Pine). По умолчанию 6.
    history_bars : int
        Глубина истории для построения линий (Pine ``history_bars``), по
        умолчанию 300.
    max_support_lines, max_resistance_lines : int
        Лимит линий каждого типа (Pine). По умолчанию 5/5.
    pivot_left, pivot_right : int
        Окно фрактальной разворотной логики (HH/HL/LH/LL). По умолчанию 5/5
        (сопоставимо с Pine-``left/right`` из блока Bjorgum).

    Returns
    -------
    TrendlineAnalysis
        Линии поддержки/сопротивления, угол-тренд, фрактальный тренд и
        объединённый (combined) вердикт.

    Raises
    ------
    ValueError
        Если DataFrame пустой или не содержит OHLC-колонок.
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame не содержит колонки: {missing}")
    if len(df) < 2 * resolution:
        raise ValueError(
            f"Недостаточно баров ({len(df)}) для анализа трендовых линий "
            f"(нужно ≥ {2 * resolution})."
        )

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    n = len(df)
    last_close = float(df["Close"].iloc[-1])

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    close_ = df["Close"].to_numpy(dtype=float)

    # 1. Серии minimums/maximums (порт Pine).
    minimums = _compute_minimums(low, resolution)
    maximums = _compute_maximums(high, resolution)

    # 2. Линии поддержки/сопротивления (порт Pine-блоков).
    hb = min(history_bars, n)
    support_lines = _build_support_lines(
        low, high, open_, close_, minimums,
        history_bars=hb, max_lines=max_support_lines,
    )
    resistance_lines = _build_resistance_lines(
        high, low, open_, close_, maximums,
        history_bars=hb, max_lines=max_resistance_lines,
    )

    # 3. ATR для нормировки углов.
    atr = _wilder_atr(df, period=14)

    # 4. Самые значимые линии (ближайшие к цене по current_price).
    strongest_support = None
    if support_lines:
        strongest_support = min(support_lines, key=lambda t: abs(t.current_price - last_close))
    strongest_resistance = None
    if resistance_lines:
        strongest_resistance = min(resistance_lines, key=lambda t: abs(t.current_price - last_close))

    # 5. Тренд по углу линий.
    trend_dir, trend_strength, avg_angle = lines_trend_direction(
        support_lines, resistance_lines, last_close, atr
    )

    # 6. Фракталы (HH/HL/LH/LL).
    swing_highs = find_pivot_highs(high, pivot_left, pivot_right)
    swing_lows = find_pivot_lows(low, pivot_left, pivot_right)
    fractals = classify_fractals(swing_highs, swing_lows)
    fractal_dir, fractal_strength = fractal_trend_direction(fractals)

    # 7. Combined-вердикт: согласие двух методов.
    combined_dir, combined_strength = _combine_trends(
        (trend_dir, trend_strength), (fractal_dir, fractal_strength)
    )

    return TrendlineAnalysis(
        timeframe=timeframe,
        last_close=last_close,
        support_lines=support_lines,
        resistance_lines=resistance_lines,
        strongest_support=strongest_support,
        strongest_resistance=strongest_resistance,
        trend_direction=trend_dir,
        trend_strength=trend_strength,
        line_angle_deg=avg_angle,
        fractals=fractals,
        fractal_trend=fractal_dir,
        fractal_strength=fractal_strength,
        combined_trend=combined_dir,
        combined_strength=combined_strength,
        atr=atr,
        n_bars=n,
    )


def _combine_trends(
    a: tuple[str, float], b: tuple[str, float], w_a: float = 0.5, w_b: float = 0.5
) -> tuple[str, float]:
    """Объединить два независимых определения тренда.

    Если оба метода согласны — направление подтверждено, сила = среднее.
    Если расходятся — RANGE с ослабленной силой (по доминирующему).
    """
    dir_a, str_a = a
    dir_b, str_b = b
    if dir_a == dir_b and dir_a != "RANGE":
        strength = (str_a * w_a + str_b * w_b) / (w_a + w_b)
        return dir_a, round(min(100.0, strength), 1)
    # Один из методов RANGE или они расходятся.
    if dir_a == "RANGE" and dir_b != "RANGE":
        return dir_b, round(str_b * 0.5, 1)
    if dir_b == "RANGE" and dir_a != "RANGE":
        return dir_a, round(str_a * 0.5, 1)
    # Оба RANGE или противоположны.
    return "RANGE", round(max(str_a, str_b) * 0.3, 1)
