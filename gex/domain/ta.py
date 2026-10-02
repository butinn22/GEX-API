"""Технический анализ базового актива: индикаторы, тренд, вероятность разворота.

Модуль не зависит от сторонних TA-библиотек (как и весь пакет ``gex``) —
индикаторы вычисляются напрямую через pandas/numpy/scipy. Это даёт
полную прозрачность формул и контроль над сглаживанием.

Состав
------
* :func:`compute_indicators`   — EMA20/50/200, RSI(14) Wilder, MACD(12,26,9);
* :func:`detect_trend`         — тренд через Higher-High / Lower-Low + стек EMA;
* :func:`reversal_probability` — **комбинированная** стохастическая вероятность
  смены тренда (эмпирическая цепь Маркова + Монте-Карло GBM-путей);
* :func:`analyze_timeframe`    — сборка индикаторы + тренд + вероятность.

Все функции принимают DataFrame OHLCV (столбцы ``Open, High, Low, Close,
Volume``), как их отдаёт :class:`gex.ta_fetcher.TATimeframesFetcher`.

Соглашения о трендах
--------------------
Классификация свингов по Dow Theory:
  * **BULLISH** — последовательность Higher High + Higher Low;
  * **BEARISH** — Lower High + Lower Low;
  * **RANGE**  — смешанная/нечёткая структура.

Эти состояния образуют алфавит эмпирической цепи Маркова.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from gex.domain.timeframes import CANONICAL_TIMEFRAMES

# Канонические примитивы — из домена, а не второй копией здесь. Раньше этот модуль был
# единственным местом с индикаторами, а `domain/indicators/*` писались как их точная
# транскрипция и остались **никем не используемыми** (числились сиротами): тесты ядра
# сравнивали ядро с самим собой. Реэкспорт делает связь явной и снимает сиротство.
from gex.domain.indicators.frames import atr_series_wilder, compute_atr  # noqa: F401
from gex.domain.indicators.primitives import (  # noqa: F401
    pine_ema,
    pine_rma,
    pine_sma,
    pine_stdev,
)

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Параметры индикаторов (классические значения по умолчанию)
# ====================================================================== #
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Свинг-окно: сколько баров слева/справа для подтверждения фрактала.
SWING_K = 2

# Окно momentum-оценки силы тренда по свечам (последние N баров).
MOMENTUM_LOOKBACK = 20

# Таймфреймы, упорядоченные от младшего к старшему (для multi-TF подтверждения).
TIMEFRAMES_ORDER: tuple[str, ...] = CANONICAL_TIMEFRAMES


# ====================================================================== #
#  Результаты: индикаторы
# ====================================================================== #
@dataclass
class TAIndicators:
    """Текущие значения технических индикаторов на последнем баре.

    Attributes
    ----------
    ema20, ema50, ema200 : float
        Экспоненциальные скользящие средние.
    rsi : float
        RSI(14) Wilder, 0..100.
    macd, macd_signal, macd_hist : float
        Линия MACD, сигнальная и гистограмма.
    macd_bull_cross, macd_bear_cross : bool
        Пересечения MACD на последнем баре.
    ema_bull_stack, ema_bear_stack : bool
        Идеальный стек EMA (20>50>200 / 20<50<200).
    """

    ema20: float
    ema50: float
    ema200: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_bull_cross: bool
    macd_bear_cross: bool
    ema_bull_stack: bool
    ema_bear_stack: bool


# ====================================================================== #
#  Результаты: candle-momentum (сила тренда по свечам)
# ====================================================================== #
@dataclass
class MomentumStrength:
    """Сила тренда по форме и объёму последних N свечей.

    Оценка построена на двух процентных метриках движения (см.
    :func:`compute_momentum_strength`):

      * ``close_to_close_pct`` — среднее %-изменение закрытия бар-к-бару
        (направленный импульс);
      * ``range_pct`` — среднее %-расширение от предыдущего Low до текущего
        High (амплитуда свинга в сторону движения).

    Сторона движения (``side``) определяется знаком close-to-close; объём
    модулирует итоговую силу: рост объёма в сторону тренда усиливает оценку,
    падение — ослабляет («объём подтверждает тренд»).

    Attributes
    ----------
    side : str
        'BULLISH' | 'BEARISH' | 'NEUTRAL' — сторона импульса.
    strength : float
        Итоговая сила 0..100 (импульс × подтверждение объёмом).
    close_to_close_pct : float
        Среднее %-изменение Close (бар к бару) по окну.
    range_pct : float
        Среднее %-расширение (High_i − Low_{i-1}) / Low_{i-1} по окну.
    avg_body_pct : float
        Средний %-размер тела свечи |Close−Open|/Open.
    volume_trend : float
        Коэффициент 0..1+: отношение среднего объёма «трендовых» баров к
        среднему объёму всех баров окна. >1 = объём подтверждает тренд.
    net_move_pct : float
        Полное %-движение от первого Close окна к последнему.
    n_bars : int
        Число баров в окне.
    """

    side: str
    strength: float
    close_to_close_pct: float
    range_pct: float
    avg_body_pct: float
    volume_trend: float
    net_move_pct: float
    n_bars: int


# ====================================================================== #
#  Результаты: тренд
# ====================================================================== #
@dataclass
class TrendInfo:
    """Оценка рыночного тренда: свинги HH/LL + momentum + стек EMA.

    Attributes
    ----------
    direction : str
        'BULLISH' | 'BEARISH' | 'RANGE'.
    strength : float
        Уверенность в тренде, 0..100 — комбинация candle-momentum, согласия
        свингов и EMA-стека.
    recent_high, recent_low : float
    swing_highs, swing_lows : list[float]
        Подтверждённые свинг-пивоты.
    higher_highs, higher_lows : int
        Число HH / HL среди последних свингов.
    lower_highs, lower_lows : int
        Число LH / LL.
    momentum : Optional[MomentumStrength]
        Candle-momentum оценка (None, если окно слишком короткое).
    """

    direction: str
    strength: float
    recent_high: float
    recent_low: float
    swing_highs: list[float] = field(default_factory=list)
    swing_lows: list[float] = field(default_factory=list)
    higher_highs: int = 0
    higher_lows: int = 0
    lower_highs: int = 0
    lower_lows: int = 0
    momentum: Optional[MomentumStrength] = None


# ====================================================================== #
#  Результаты: вероятность разворота
# ====================================================================== #
@dataclass
class ReversalProb:
    """Стохастическая вероятность смены тренда.

    Комбинированная оценка из двух независимых моделей:
      * ``p_markov`` — эмпирическая вероятность перехода в противоположное
        состояние по исторической цепи Маркова (3 состояния: UP/DOWN/RANGE);
      * ``p_mc``     — доля Монте-Карло GBM-путей, на которых быстрый
        тренд-сигнал сменил знак к концу горизонта.

    Attributes
    ----------
    p_reversal : float
        Итоговая вероятность разворота тренда, 0..1.
        ``p_reversal = w_markov*p_markov + w_mc*p_mc`` с поправкой на
        истощение (RSI в зонах перекупленности/перепроданности).
    p_markov, p_mc : float
        Компонентные оценки.
    method : str
        Использованный метод ('Markov+MonteCarlo').
    n_paths : int
        Число смоделированных путей.
    horizon_bars : int
        Горизонт прогноза в барах данного таймфрейма.
    weights : dict
        Веса компонентов (для аудита).
    """

    p_reversal: float
    p_markov: float
    p_mc: float
    method: str
    n_paths: int
    horizon_bars: int
    weights: dict = field(default_factory=dict)


# ====================================================================== #
#  Результаты: подтверждение старшего TF младшими
# ====================================================================== #
@dataclass
class TimeframeConfirmation:
    """Подтверждение тренда на старшем таймфрейме младшими.

    Идея классического multi-timeframe анализа: направление старшего TF
    «истинное», но его нужно подтверждать согласием младших TF. Если младшие
    TF расходятся со старшим — сила тренда снижается (вероятнее коррекция/флэт),
    если согласны — подтверждается (усиливается).

    Attributes
    ----------
    base_timeframe : str
        Старший TF, для которого считается подтверждение.
    base_direction : str
        Направление базового (старшего) TF.
    agreeing : list[str]
        Младшие TF, чьё направление совпадает с базовым.
    disagreeing : list[str]
        Младшие TF с противоположным направлением.
    neutral : list[str]
        Младшие TF в RANGE.
    confirmation_ratio : float
        0..1 — доля согласных младших TF (с учётом их силы).
    adjustment : float
        −50..+50 — поправка к силе тренда старшего TF (отрицательная при
        расхождении, положительная при полном согласии).
    """

    base_timeframe: str
    base_direction: str
    agreeing: list[str] = field(default_factory=list)
    disagreeing: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)
    confirmation_ratio: float = 0.0
    adjustment: float = 0.0


# ====================================================================== #
#  Результаты: дивергенции осцилляторов
# ====================================================================== #
@dataclass
class Divergence:
    """Одна дивергенция осциллятора с ценой.

    Attributes
    ----------
    oscillator : str
        'RSI' или 'MACD'.
    type : str
        'BULLISH' (бычья — цена обновляет LL, осциллятор HL) или
        'BEARISH' (медвежья — цена HH, осциллятор LH).
    strength : float
        Сила дивергенции 0..100 (по величине расхождения наклонов).
    bars_ago : int
        Сколько баров назад сформировалась дивергенция (0 = на последнем баре).
    """

    oscillator: str
    type: str
    strength: float
    bars_ago: int


@dataclass
class DivergenceInfo:
    """Сводка дивергенций осцилляторов на таймфрейме.

    Attributes
    ----------
    divergences : list[Divergence]
        Найденные дивергенции (RSI/MACD), отсортированы по свежести и силе.
    has_bullish, has_bearish : bool
        Наличие хотя бы одной бычьей / медвежьей дивергенции.
    net_signal : str
        'BULLISH' | 'BEARISH' | 'NEUTRAL' — преобладающий сигнал дивергенций.
    """

    divergences: list[Divergence] = field(default_factory=list)
    has_bullish: bool = False
    has_bearish: bool = False
    net_signal: str = "NEUTRAL"


# ====================================================================== #
#  Результаты: один таймфрейм целиком
# ====================================================================== #
@dataclass
class TimeframeAnalysis:
    """Полный анализ одного таймфрейма: индикаторы + тренд + вероятность.

    Поля ``confirmation`` и ``divergence`` заполняются post-hoc: первое —
    в сервисе при multi-TF сводке, второе — в :func:`analyze_timeframe`.
    """

    timeframe: str
    last_close: float
    n_bars: int
    indicators: TAIndicators
    trend: TrendInfo
    reversal: ReversalProb
    confirmation: Optional[TimeframeConfirmation] = None
    divergence: Optional[DivergenceInfo] = None


# ====================================================================== #
#  1. Индикаторы
# ====================================================================== #
def compute_indicators(df: pd.DataFrame) -> TAIndicators:
    """Вычислить технические индикаторы на OHLCV-датафрейме.

    Parameters
    ----------
    df : pd.DataFrame
        Столбцы ``Open, High, Low, Close, Volume``; индекс — время.

    Returns
    -------
    TAIndicators
        Значения на последнем баре.

    Raises
    ------
    ValueError
        Если датафрейм пуст или нет столбца ``Close``.
    """
    if df is None or len(df) == 0:
        raise ValueError("Пустой датафрейм — невозможно вычислить индикаторы.")
    if "Close" not in df.columns:
        raise ValueError("Датафрейм должен содержать столбец 'Close'.")

    close = df["Close"].astype(float)

    # --- EMA (exponential, adjust=False — классическая рекурсивная форма) ---
    # ВНИМАНИЕ: это НЕ `pine_ema` из домена. Здесь EMA pandas (`adjust=False`, без
    # SMA-сида), в домене — канонический `ta.ema` со SMA-сидом. Числа расходятся: на
    # 300 барах у EMA200 расхождение достигает 0.45 и на последнем баре НЕ исчезает,
    # а именно последний бар здесь и отдаётся. Поэтому делегирование невозможно без
    # изменения чисел (а значит, и `ema_bull_stack`), и оно НЕ сделано: выбор варианта
    # — решение о модели, а не рефакторинг. Расхождение закреплено тестом
    # `tests/test_ta_domain_parity.py::test_ema_variants_differ_and_are_not_interchangeable`,
    # чтобы замена одного на другое не прошла незаметно.
    ema20 = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema50 = close.ewm(span=EMA_MID, adjust=False).mean()
    ema200 = close.ewm(span=EMA_SLOW, adjust=False).mean()

    # --- RSI (Wilder's smoothing) ---
    rsi = _wilder_rsi(close, RSI_PERIOD)

    # --- MACD ---
    macd_line = close.ewm(span=MACD_FAST, adjust=False).mean() - \
        close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # Значения на последнем баре
    i = len(df) - 1
    e20 = float(ema20.iloc[i])
    e50 = float(ema50.iloc[i])
    e200 = float(ema200.iloc[i])
    r = float(rsi.iloc[i])
    m = float(macd_line.iloc[i])
    ms = float(macd_signal.iloc[i])
    mh = float(macd_hist.iloc[i])

    # Пересечения MACD на последнем баре
    if len(macd_hist) >= 2:
        prev_h = float(macd_hist.iloc[i - 1])
        macd_bull_cross = (prev_h <= 0) and (mh > 0)
        macd_bear_cross = (prev_h >= 0) and (mh < 0)
    else:
        macd_bull_cross = macd_bear_cross = False

    # Идеальный стек EMA
    ema_bull_stack = (e20 > e50) and (e50 > e200)
    ema_bear_stack = (e20 < e50) and (e50 < e200)

    return TAIndicators(
        ema20=e20,
        ema50=e50,
        ema200=e200,
        rsi=r,
        macd=m,
        macd_signal=ms,
        macd_hist=mh,
        macd_bull_cross=macd_bull_cross,
        macd_bear_cross=macd_bear_cross,
        ema_bull_stack=ema_bull_stack,
        ema_bear_stack=ema_bear_stack,
    )


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI по Уайлдеру — каноническая реализация из домена.

    Здесь было 40 строк собственной реализации (SMA-сид, сглаживание Уайлдера, политики
    разогрева и плоского ряда). Та же логика уже жила в ``domain/indicators/rsi.py`` с
    **явными** политиками (``warmup``/``flat``), и её docstring прямо ссылался на эту
    функцию как на источник политики. Две реализации одного индикатора — это не дублирование
    ради краткости, а расхождение: правку в одной из них вторая бы не увидела.

    Паритет проверен по всей длине ряда и для обеих политик
    (``tests/test_ta_domain_parity.py``): расхождение ровно 0.0, поэтому замена —
    поведенчески ничего не меняющий шаг, а не «примерно то же самое».

    Серия возвращается как ``pd.Series`` с исходным индексом: вызывающий код работает с
    индексом (обращения по позиции и ``.iloc``), и потеря индекса была бы тихой поломкой.
    """
    from gex.domain.indicators import rsi as _rsi

    values = _rsi.wilder_rsi(close.to_numpy(dtype=float), period)
    return pd.Series(values, index=close.index)

# ====================================================================== #
#  Candle-momentum: сила тренда по размеру/движению последних N свечей
# ====================================================================== #
def compute_momentum_strength(
    df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
) -> Optional[MomentumStrength]:
    """Сила тренда по форме и объёму последних ``lookback`` свечей.

    Метрики движения (в процентах, таймфрейм-независимые):

      1. **close-to-close** — среднее %-изменение закрытия бар к бару::

             pct_i = 100 * (Close_i − Close_{i-1}) / Close_{i-1}

         Знак среднего задаёт сторону импульса, модуль — его величину.

      2. **range expansion** — среднее %-расширение от *прошлого Low* до
         *нового High* (амплитуда свинга в сторону движения)::

             rng_i = 100 * (High_i − Low_{i-1}) / Low_{i-1}

         Положительная и растущая амплитуда в сторону тренда = сильный
         directional thrust.

      3. **body size** — средний |Close−Open|/Open (тело свечи): большие тела
         в сторону тренда = уверенное движение, маленькие = нерешительность.

    Подтверждение объёмом: средний объём баров, закрывшихся в сторону тренда,
    делится на средний объём всех баров окна (``volume_trend``). >1 = объём
    подтверждает тренд, <1 = расходится (объём на откат больше, чем на импульс).

    Итоговая ``strength`` (0..100) — сигмоид от произведения направленного
    движения на объёмное подтверждение, масштабированный под 0..100::

        raw = |close_to_close_pct| * sigmoid(volume_trend) * (1 + body_factor)
        strength = 100 * sigmoid(k * raw)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV (нужны Open, High, Low, Close, Volume).
    lookback : int
        Число последних баров для окна (по умолчанию 20).

    Returns
    -------
    MomentumStrength or None
        None, если данных меньше 2 баров в окне.
    """
    required = {"Open", "High", "Low", "Close"}
    if df is None or not required.issubset(df.columns) or len(df) < 2:
        return None

    tail = df.tail(lookback)
    o = tail["Open"].astype(float).values
    h = tail["High"].astype(float).values
    low = tail["Low"].astype(float).values
    c = tail["Close"].astype(float).values
    vol = tail["Volume"].astype(float).values if "Volume" in tail.columns else None
    n = len(c)

    # --- 1. close-to-close % (со 2-го бара) ---
    c2c = np.diff(c) / c[:-1] * 100.0
    avg_c2c = float(np.mean(c2c)) if n > 1 else 0.0

    # --- 2. range expansion: (High_i − Low_{i-1}) / Low_{i-1} ---
    rng = np.zeros(max(n - 1, 0))
    prev_low = low[:-1]
    cur_high = h[1:]
    nz = prev_low > 0
    rng[nz] = (cur_high[nz] - prev_low[nz]) / prev_low[nz] * 100.0
    avg_range = float(np.mean(rng)) if len(rng) else 0.0

    # --- 3. body size |Close−Open|/Open ---
    body = np.abs(c - o) / np.where(o > 0, o, np.nan) * 100.0
    body = body[np.isfinite(body)]
    avg_body = float(np.mean(body)) if body.size else 0.0

    # --- Сторона движения ---
    if avg_c2c > 1e-9:
        side = "BULLISH"
    elif avg_c2c < -1e-9:
        side = "BEARISH"
    else:
        side = "NEUTRAL"

    # --- 4. Подтверждение объёмом ---
    volume_trend = 1.0
    if vol is not None and side != "NEUTRAL" and n > 1:
        # «Трендовые» бары — закрылись в сторону импульса
        bar_dir = np.sign(np.diff(c))  # sign 1/-1/0
        if side == "BULLISH":
            trend_mask = np.append(bar_dir > 0, c[-1] >= o[-1])
        else:
            trend_mask = np.append(bar_dir < 0, c[-1] < o[-1])
        trend_mask = trend_mask.astype(bool)
        if trend_mask.any() and (~trend_mask).any():
            avg_vol_trend = float(np.mean(vol[trend_mask]))
            avg_vol_other = float(np.mean(vol[~trend_mask]))
            volume_trend = avg_vol_trend / avg_vol_other if avg_vol_other > 0 else 1.0
        elif trend_mask.all():
            # Все бары в сторону тренда → сильное подтверждение
            volume_trend = 1.3

    # --- 5. Итоговая сила 0..100 ---
    # Нормируем движения к «ожидаемой» шкале через сигмоид.
    # body_factor: большие тела усиливают, но ограниченно.
    body_factor = np.tanh(avg_body / 1.0)  # ~0..1 для типичных 0..2% тел
    vol_factor = 1.0 / (1.0 + np.exp(-(volume_trend - 1.0) * 2.5))  # sigmoid(volume_trend)
    raw = abs(avg_c2c) * vol_factor * (1.0 + body_factor)
    # k подобран так, чтобы ~0.3% средних бар-к-бару с подтверждающим объёмом
    # давало силу ~70-80, а ~0.05% — ~20-30 (адекватно для интрадей/дней).
    strength = 100.0 * (1.0 / (1.0 + np.exp(-8.0 * (raw - 0.15))))

    return MomentumStrength(
        side=side,
        strength=float(np.clip(strength, 0.0, 100.0)),
        close_to_close_pct=float(avg_c2c),
        range_pct=float(avg_range),
        avg_body_pct=float(avg_body),
        volume_trend=float(volume_trend),
        net_move_pct=float((c[-1] - c[0]) / c[0] * 100.0) if c[0] > 0 else 0.0,
        n_bars=int(n),
    )


# ====================================================================== #
#  Дивергенции осцилляторов (RSI / MACD) с ценой
# ====================================================================== #
def detect_divergences(df: pd.DataFrame, lookback: int = 60) -> DivergenceInfo:
    """Найти дивергенции RSI и MACD с ценой за последние ``lookback`` баров.

    Дивергенция — расхождение направления цены и осциллятора на двух
    соседних свинг-экстремумах:

      * **Бычья** (BULLISH): цена обновляет Lower Low, а осциллятор —
        Higher Low (нисходящий импульс слабеет → возможен разворот вверх);
      * **Медвежья** (BEARISH): цена обновляет Higher High, а осциллятор —
        Lower High (восходящий импульс слабеет → возможен разворот вниз).

    Алгоритм:
      1. Вычисляются ряды цены (Close) и осцилляторов (RSI, MACD-гистограмма).
      2. Находятся свинг-экстремумы цены фрактальным методом.
      3. Для каждых двух последовательных одноимённых свингов (два лоя или
         два хая) сравниваются значения осциллятора в этих точках.
      4. Сила дивергенции — нормированное расхождение наклонов цены и
         осциллятора (0..100).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV.
    lookback : int
        Глубина поиска (по умолчанию 60 баров).

    Returns
    -------
    DivergenceInfo
        Сводка: список дивергенций, флаги наличия, результирующий сигнал.
    """
    if df is None or len(df) < EMA_MID + MACD_SIGNAL + 2:
        return DivergenceInfo()

    tail = df.tail(lookback)
    close = tail["Close"].astype(float)
    rsi = _wilder_rsi(close, RSI_PERIOD)
    macd_line = close.ewm(span=MACD_FAST, adjust=False).mean() - \
        close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_hist = macd_line - macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()

    high = tail["High"].astype(float).values
    low = tail["Low"].astype(float).values
    close_v = close.values
    rsi_v = rsi.values
    macd_v = macd_hist.values

    # Свинг-экстремумы цены (фракталы, окно 2)
    k = 2
    swing_highs_idx = []
    swing_lows_idx = []
    for i in range(k, len(close_v) - k):
        if high[i] == high[i - k : i + k + 1].max() and \
                np.sum(high[i - k : i + k + 1] == high[i]) == 1:
            swing_highs_idx.append(i)
        if low[i] == low[i - k : i + k + 1].min() and \
                np.sum(low[i - k : i + k + 1] == low[i]) == 1:
            swing_lows_idx.append(i)

    divergences: list[Divergence] = []

    # --- Медвежьи дивергенции: по двум последовательным свинг-хаям ---
    for j in range(1, len(swing_highs_idx)):
        i0, i1 = swing_highs_idx[j - 1], swing_highs_idx[j]
        if close_v[i1] > close_v[i0]:  # цена: Higher High
            for osc_name, osc_v in (("RSI", rsi_v), ("MACD", macd_v)):
                d0, d1 = osc_v[i0], osc_v[i1]
                if np.isfinite(d0) and np.isfinite(d1) and d1 < d0:
                    # Сила: относительное расхождение цены vs осциллятора
                    price_pct = abs((close_v[i1] - close_v[i0]) /
                                    max(close_v[i0], 1e-9))
                    osc_pct = abs(d1 - d0) / (abs(d0) + 1e-9)
                    raw = min(price_pct + osc_pct, 1.0)
                    strength = float(np.clip(raw * 100.0, 0.0, 100.0))
                    divergences.append(Divergence(
                        oscillator=osc_name, type="BEARISH",
                        strength=strength, bars_ago=len(close_v) - 1 - i1,
                    ))

    # --- Бычьи дивергенции: по двум последовательным свинг-лоям ---
    for j in range(1, len(swing_lows_idx)):
        i0, i1 = swing_lows_idx[j - 1], swing_lows_idx[j]
        if close_v[i1] < close_v[i0]:  # цена: Lower Low
            for osc_name, osc_v in (("RSI", rsi_v), ("MACD", macd_v)):
                d0, d1 = osc_v[i0], osc_v[i1]
                if np.isfinite(d0) and np.isfinite(d1) and d1 > d0:
                    price_pct = abs((close_v[i1] - close_v[i0]) /
                                    max(close_v[i0], 1e-9))
                    osc_pct = abs(d1 - d0) / (abs(d0) + 1e-9)
                    raw = min(price_pct + osc_pct, 1.0)
                    strength = float(np.clip(raw * 100.0, 0.0, 100.0))
                    divergences.append(Divergence(
                        oscillator=osc_name, type="BULLISH",
                        strength=strength, bars_ago=len(close_v) - 1 - i1,
                    ))

    # Берём самые свежие/сильные, не более 4 штук
    divergences.sort(key=lambda d: (-d.bars_ago * -1, -d.strength))  # свежие + сильные
    divergences = divergences[:4]

    has_bullish = any(d.type == "BULLISH" for d in divergences)
    has_bearish = any(d.type == "BEARISH" for d in divergences)
    bull_strength = sum(d.strength for d in divergences if d.type == "BULLISH")
    bear_strength = sum(d.strength for d in divergences if d.type == "BEARISH")
    if bull_strength > bear_strength:
        net_signal = "BULLISH"
    elif bear_strength > bull_strength:
        net_signal = "BEARISH"
    else:
        net_signal = "NEUTRAL"

    return DivergenceInfo(
        divergences=divergences,
        has_bullish=has_bullish,
        has_bearish=has_bearish,
        net_signal=net_signal,
    )


# ====================================================================== #
#  2. Тренд: свинги HH/LL + momentum + подтверждение EMA
# ====================================================================== #
def detect_trend(df: pd.DataFrame, k: int = SWING_K) -> TrendInfo:
    """Определить тренд: свинги HH/LL + candle-momentum + стек EMA.

    Три независимых сигнала объединяются:

    1. **Свинг-структура** (Dow Theory): фрактальные свинг-хай/лоу,
       классификация HH+HL → BULLISH, LH+LL → BEARISH.
    2. **Candle-momentum** (:func:`compute_momentum_strength`): сторона и
       сила движения по последним 20 свечам (% close-to-close, амплитуда
       Low→High, тела свечей, подтверждение объёмом).
    3. **EMA-стек**: идеальный стек 20>50>200 (вверх) / 20<50<200 (вниз).

    Направление: голос сигналов 1+2 (EMA лишь подтверждает). Если свинги и
    momentum согласованы — берётся их направление; если расходятся — RANGE.

    Сила тренда (0..100) = взвешенное сочетание::

        strength = 0.5*momentum.strength      (свечная динамика + объём)
                 + 0.3*swing_agreement         (согласие свингов)
                 + 0.2*ema_bonus               (стек EMA: +бонус/штраф)

    При расхождении стороны momentum и финального направления сила
    дополнительно штрафуется (движение не подтверждает тренд).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV.
    k : int
        Полуокно фрактала (баров слева/справа).
    """
    if df is None or len(df) < 2 * k + 1:
        # Недостаточно данных — нейтральный тренд
        close = float(df["Close"].iloc[-1]) if df is not None and len(df) else float("nan")
        return TrendInfo(
            direction="RANGE", strength=0.0,
            recent_high=close, recent_low=close,
        )

    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values

    swing_highs_idx = []
    swing_lows_idx = []
    for i in range(k, len(high) - k):
        win_h = high[i - k : i + k + 1]
        win_l = low[i - k : i + k + 1]
        if high[i] == win_h.max() and np.sum(win_h == high[i]) == 1:
            swing_highs_idx.append(i)
        if low[i] == win_l.min() and np.sum(win_l == low[i]) == 1:
            swing_lows_idx.append(i)

    swing_highs = [float(high[i]) for i in swing_highs_idx]
    swing_lows = [float(low[i]) for i in swing_lows_idx]

    # Считаем HH/LH по свинг-хаям, HL/LL по свинг-лоям
    higher_highs = lower_highs = 0
    for j in range(1, len(swing_highs)):
        if swing_highs[j] > swing_highs[j - 1]:
            higher_highs += 1
        elif swing_highs[j] < swing_highs[j - 1]:
            lower_highs += 1
    higher_lows = lower_lows = 0
    for j in range(1, len(swing_lows)):
        if swing_lows[j] > swing_lows[j - 1]:
            higher_lows += 1
        elif swing_lows[j] < swing_lows[j - 1]:
            lower_lows += 1

    # --- 1b. Сводные свинг-счётчики ---
    bull_score = higher_highs + higher_lows
    bear_score = lower_highs + lower_lows
    total_swings = bull_score + bear_score

    # --- 2. Candle-momentum (последние N свечей) ---
    momentum = compute_momentum_strength(df, lookback=MOMENTUM_LOOKBACK)

    # --- Классификация: голос свингов + momentum ---
    swing_dir = "RANGE"
    if total_swings > 0 and bull_score > bear_score and higher_highs >= 1 and higher_lows >= 1:
        swing_dir = "BULLISH"
    elif total_swings > 0 and bear_score > bull_score and lower_highs >= 1 and lower_lows >= 1:
        swing_dir = "BEARISH"

    mom_dir = momentum.side if momentum is not None else "NEUTRAL"

    if swing_dir != "RANGE" and (mom_dir == swing_dir or mom_dir == "NEUTRAL"):
        # Свинги и momentum согласованы (или momentum нейтрален)
        direction = swing_dir
    elif mom_dir in ("BULLISH", "BEARISH") and swing_dir == "RANGE":
        # Свингов нет, но есть явный импульс — доверяем momentum
        direction = mom_dir
    elif swing_dir != "RANGE" and mom_dir in ("BULLISH", "BEARISH") and mom_dir != swing_dir:
        # Прямое расхождение свингов и momentum → тренд под вопросом
        direction = "RANGE"
    else:
        direction = swing_dir  # RANGE

    # --- Сила тренда 0..100: blend momentum + свинги + EMA ---
    swing_agreement = (abs(bull_score - bear_score) / total_swings * 100.0) \
        if total_swings > 0 else 0.0

    momentum_strength = momentum.strength if momentum is not None else 0.0

    # EMA-стек: подтверждение или расхождение
    ema_bonus = 0.0
    close_s = df["Close"].astype(float)
    if len(close_s) >= EMA_SLOW:
        e20 = close_s.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
        e50 = close_s.ewm(span=EMA_MID, adjust=False).mean().iloc[-1]
        e200 = close_s.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
        last = float(close_s.iloc[-1])
        ema_confirms_bull = (e20 > e50 > e200) and (last > e50)
        ema_confirms_bear = (e20 < e50 < e200) and (last < e50)
        if direction == "BULLISH" and ema_confirms_bull:
            ema_bonus = 100.0
        elif direction == "BEARISH" and ema_confirms_bear:
            ema_bonus = 100.0
        elif direction in ("BULLISH", "BEARISH"):
            ema_bonus = 40.0   # EMA не выстроены идеально, но частично согласны
        else:
            ema_bonus = 20.0   # RANGE — слабое базовое

    strength = (
        0.5 * momentum_strength
        + 0.3 * swing_agreement
        + 0.2 * ema_bonus
    )

    # Штраф: если сторона momentum противоречит финальному направлению,
    # значит движение не подтверждает тренд → снижаем силу.
    if direction in ("BULLISH", "BEARISH") and mom_dir in ("BULLISH", "BEARISH") \
            and mom_dir != direction:
        strength *= 0.5

    strength = float(np.clip(strength, 0.0, 100.0))

    recent_high = float(np.nanmax(high)) if len(high) else float("nan")
    recent_low = float(np.nanmin(low)) if len(low) else float("nan")

    return TrendInfo(
        direction=direction,
        strength=strength,
        recent_high=recent_high,
        recent_low=recent_low,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        higher_highs=higher_highs,
        higher_lows=higher_lows,
        lower_highs=lower_highs,
        lower_lows=lower_lows,
        momentum=momentum,
    )


# ====================================================================== #
#  3. Вероятность разворота: Markov + Monte-Carlo
# ====================================================================== #
def reversal_probability(
    df: pd.DataFrame,
    current_trend: str,
    horizon_bars: int = 20,
    n_paths: int = 10_000,
    w_markov: float = 0.5,
    w_mc: float = 0.5,
    seed: int = 42,
) -> ReversalProb:
    """Стохастическая вероятность смены тренда (Markov + Monte-Carlo).

    Компоненты
    ----------
    1. **Эмпирический Марков**: история цен разбивается на состояния по
       быстрому тренд-сигналу (cross EMA20/50 + наклон). Строится матрица
       переходов 3×3, ``p_markov = P[current → opposite]``. Это непараметрическая
       оценка «частоты разворотов» на конкретном активе.

    2. **Monte-Carlo GBM**: калибровка ``mu, sigma`` по последним лог-доходностям,
       симуляция ``n_paths`` путей длиной ``horizon_bars`` из текущего ``spot``.
       На каждом пути пересчитывается быстрый тренд-сигнал; доля путей со
       сменой знака = ``p_mc``. Это учитывает текущую волатильность и дрейф.

    Финальная вероятность::

        p_reversal = w_markov*p_markov + w_mc*p_mc

    с поправкой на истощение: если RSI>70 при BULLISH (или <30 при BEARISH)
    вероятность разворота повышается (импульс истощён).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV истории.
    current_trend : str
        'BULLISH' | 'BEARISH' | 'RANGE' (из :func:`detect_trend`).
    horizon_bars : int
        Горизонт прогноза в барах текущего таймфрейма.
    n_paths : int
        Число Монте-Карло путей.
    w_markov, w_mc : float
        Веса компонентов (нормируются к 1).
    seed : int
        Зерно генератора (воспроизводимость MC).
    """
    if df is None or len(df) < EMA_MID + MACD_SIGNAL:
        raise ValueError(
            f"Недостаточно истории ({len(df) if df is not None else 0} баров) "
            f"для оценки вероятности разворота (нужно > {EMA_MID + MACD_SIGNAL})."
        )
    if horizon_bars < 1:
        horizon_bars = 1
    if n_paths < 100:
        n_paths = 100

    # Нормируем веса
    w_sum = w_markov + w_mc
    w_markov, w_mc = w_markov / w_sum, w_mc / w_sum

    # --- 1. Эмпирический Марков ---
    p_markov = _markov_reversal_probability(df)

    # --- 2. Monte-Carlo GBM ---
    p_mc = _mc_reversal_probability(df, horizon_bars, n_paths, seed)

    # --- Комбинация ---
    p_reversal = w_markov * p_markov + w_mc * p_mc

    # --- Поправка на истощение импульса (RSI) ---
    rsi_now = float(_wilder_rsi(df["Close"].astype(float), RSI_PERIOD).iloc[-1])
    if current_trend == "BULLISH" and rsi_now > RSI_OVERBOUGHT:
        # Перекупленность → повышаем вероятность разворота вниз
        excess = (rsi_now - RSI_OVERBOUGHT) / (100.0 - RSI_OVERBOUGHT)
        p_reversal += 0.10 * excess
    elif current_trend == "BEARISH" and rsi_now < RSI_OVERSOLD:
        excess = (RSI_OVERSOLD - rsi_now) / RSI_OVERSOLD
        p_reversal += 0.10 * excess

    p_reversal = float(np.clip(p_reversal, 0.0, 1.0))

    return ReversalProb(
        p_reversal=p_reversal,
        p_markov=float(np.clip(p_markov, 0.0, 1.0)),
        p_mc=float(np.clip(p_mc, 0.0, 1.0)),
        method="Markov+MonteCarlo",
        n_paths=n_paths,
        horizon_bars=horizon_bars,
        weights={"markov": round(w_markov, 3), "mc": round(w_mc, 3)},
    )


def _trend_signal_series(close: pd.Series) -> pd.Series:
    """Быстрый тренд-сигнал по истории: +1 (вверх) / -1 (вниз) / 0 (боковик).

    Комбинация: позиция EMA20 относительно EMA50 и знак MACD-гистограммы.
    Согласованный bullish → +1, согласованный bearish → −1, иначе 0.
    Используется для построения цепи Маркова и для проверки путей MC.
    """
    e20 = close.ewm(span=EMA_FAST, adjust=False).mean()
    e50 = close.ewm(span=EMA_MID, adjust=False).mean()
    macd_line = close.ewm(span=MACD_FAST, adjust=False).mean() - \
        close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_hist = macd_line - macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()

    bull = (e20 > e50) & (macd_hist > 0)
    bear = (e20 < e50) & (macd_hist < 0)
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[bull] = 1
    sig[bear] = -1
    return sig


def _markov_reversal_probability(df: pd.DataFrame) -> float:
    """Эмпирическая P(переход из текущего тренда в противоположный).

    Строит матрицу переходов 3×3 по последовательности быстрого тренд-сигнала
    и возвращает вероятность перехода из текущего состояния в противоположное.
    В состояниях RANGE разворот оценивается как среднее переходов в оба конца.
    """
    close = df["Close"].astype(float)
    sig = _trend_signal_series(close)
    # Сжимаем серии одинаковых состояний → «пребывание» не считается переходом
    states = sig.values
    # Убираем leading/trailing нули и подряд идущие дубликаты
    compressed = []
    prev = None
    for s in states:
        if s != prev:
            compressed.append(int(s))
            prev = s
    if len(compressed) < 2:
        return 0.3  # недостаточно переходов — мягкий априорный приор

    # Алфавит: -1 (DOWN), 0 (RANGE), 1 (UP)
    idx = {-1: 0, 0: 1, 1: 2}
    trans = np.zeros((3, 3), dtype=float)
    for a, b in zip(compressed[:-1], compressed[1:]):
        trans[idx[a], idx[b]] += 1.0

    # Нормируем по строкам (сглаживание Лапласа против деления на 0)
    trans = trans + 1e-6
    row_sums = trans.sum(axis=1, keepdims=True)
    P = trans / row_sums

    # Текущее состояние — последнее в сжатой последовательности
    current = compressed[-1]
    ci = idx[current]

    # Противоположное: для UP (1) → DOWN (0), для DOWN (-1) → UP (2),
    # для RANGE (0) → среднее P(→UP)+P(→DOWN) (любой выход из боковика)
    if current == 1:
        return float(P[ci, idx[-1]])           # UP → DOWN
    elif current == -1:
        return float(P[ci, idx[1]])             # DOWN → UP
    else:
        return float((P[ci, idx[1]] + P[ci, idx[-1]]) / 2.0)  # RANGE → тренд


def _mc_reversal_probability(
    df: pd.DataFrame,
    horizon_bars: int,
    n_paths: int,
    seed: int,
) -> float:
    """Доля Монте-Карло GBM-путей со сменой тренд-сигнала.

    Калибруем геометрическое броуновское движение по недавним лог-доходностям
    (окно ~ 3×EMA_MID для адаптивности к текущему режиму), затем симулируем
    ``n_paths`` путей длиной ``horizon_bars`` и проверяем быстрый тренд-сигнал
    на конце пути против текущего.
    """
    close = df["Close"].astype(float).values
    spot = float(close[-1])

    # Калибровка GBM по недавним лог-дохам
    window = min(len(close), max(EMA_MID * 3, EMA_MID + MACD_SIGNAL))
    recent = close[-window:]
    log_rets = np.diff(np.log(recent))
    log_rets = log_rets[np.isfinite(log_rets)]
    if len(log_rets) < 5:
        return 0.3

    mu = float(np.mean(log_rets))
    sigma = float(np.std(log_rets, ddof=1))
    sigma = max(sigma, 1e-5)

    # Текущий быстрый тренд-сигнал (по полной истории)
    sig_series = _trend_signal_series(df["Close"].astype(float)).values
    current_sig = int(sig_series[-1])

    rng = np.random.default_rng(seed)
    # Симуляция: log-приращения, GBM
    drift = (mu - 0.5 * sigma ** 2) * np.ones((n_paths, 1))
    shocks = rng.standard_normal((n_paths, horizon_bars))
    steps = drift + sigma * shocks
    log_paths = np.log(spot) + np.cumsum(steps, axis=1)
    # Полный путь для пересчёта индикаторов: [spot, ...simulated]
    full = np.empty((n_paths, horizon_bars + 1))
    full[:, 0] = np.log(spot)
    full[:, 1:] = log_paths
    price_paths = np.exp(full)

    # На каждом пути: быстрый сигнал на конце = sign(EMA20-EMA50) на последних
    #EMA50/EMA20 на коротком участке пути неустойчиво, поэтому используем
    # взвешенную комбинацию: наклон последних цен и MACD-подобную разность.
    flipped = 0
    for p in range(n_paths):
        end_sig = _fast_signal_from_path(price_paths[p])
        # Считаем сменой: переход в противоположный знак либо выход из 0 в противоположное
        if current_sig > 0 and end_sig < 0:
            flipped += 1
        elif current_sig < 0 and end_sig > 0:
            flipped += 1
        elif current_sig == 0 and end_sig != 0:
            flipped += 1

    return float(flipped / n_paths)


def _fast_signal_from_path(path: np.ndarray) -> int:
    """Быстрый тренд-сигнал на коротком смоделированном пути.

    На малом числе баров полноценные EMA/MACD нестабильны, поэтому используем
    робастную прокси-комбинацию:
      * наклон (линейная регрессия лог-цен: знак slope);
      * положение последней цены относительно короткой EMA;
      * MACD-подобная разность коротких EMA.

    Согласованный bullish → +1, bearish → −1, иначе 0.
    """
    n = len(path)
    if n < 4:
        return 0
    x = np.arange(n, dtype=float)
    y = path
    # Знак наклона МНК (без свободного члена достаточно ковариации)
    slope = np.cov(x, y, bias=False)[0, 1] / (np.var(x, ddof=1) or 1e-12)

    e_fast = pd.Series(path).ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    e_mid = pd.Series(path).ewm(span=EMA_MID, adjust=False).mean().iloc[-1]
    last = float(path[-1])

    bull = (slope > 0) and (last > e_mid) and (e_fast > e_mid)
    bear = (slope < 0) and (last < e_mid) and (e_fast < e_mid)
    if bull:
        return 1
    if bear:
        return -1
    return 0


# ====================================================================== #
#  4. Сборка одного таймфрейма
# ====================================================================== #
def analyze_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    horizon_bars: int = 20,
    n_paths: int = 10_000,
    seed: int = 42,
) -> TimeframeAnalysis:
    """Полный анализ одного таймфрейма: индикаторы + тренд + вероятность.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV данного таймфрейма.
    timeframe : str
        Метка ('1h', '2h', '4h', '1d').
    horizon_bars : int
        Горизонт прогноза для вероятности разворота, в барах.
    """
    indicators = compute_indicators(df)
    trend = detect_trend(df)
    try:
        reversal = reversal_probability(
            df,
            current_trend=trend.direction,
            horizon_bars=horizon_bars,
            n_paths=n_paths,
            seed=seed,
        )
    except ValueError as exc:
        # Недостаточно истории для стохастики (короткий таймфрейм / новый тикер).
        # Не роняем весь анализ — возвращаем слабый априор вместо оценки.
        logger.warning("reversal_probability fallback на %s: %s", timeframe, exc)
        reversal = ReversalProb(
            p_reversal=0.5,
            p_markov=0.5,
            p_mc=0.5,
            method="insufficient-history",
            n_paths=0,
            horizon_bars=horizon_bars,
            weights={},  # нет компонентов: мало истории для Markov+MC
        )
    divergence = detect_divergences(df)
    return TimeframeAnalysis(
        timeframe=timeframe,
        last_close=float(df["Close"].iloc[-1]),
        n_bars=int(len(df)),
        indicators=indicators,
        trend=trend,
        reversal=reversal,
        divergence=divergence,
    )


# ====================================================================== #
#  Multi-timeframe подтверждение: младшие TF → старший
# ====================================================================== #
def build_timeframe_confirmations(
    analyses: list[TimeframeAnalysis],
) -> dict[str, TimeframeConfirmation]:
    """Построить подтверждение каждого таймфрейма младшими для него.

    Принцип классического multi-timeframe анализа: тренд старшего TF —
    «истинный», младшие TF его подтверждают или опровергают.

    Для каждого TF в ``analyses`` берутся все *младшие* TF (раньше в порядке
    :data:`TIMEFRAMES_ORDER`) и считается:

      * ``confirmation_ratio`` — взвешенная по силе доля согласных младших TF
        (1.0 = все согласны, 0.0 = все против);
      * ``adjustment`` — поправка к силе тренда данного TF: −50..+50.
        При полном согласии младших — бонус (до +25..+50), при расхождении —
        штраф (до −50). RANGE-младшие TF не штрафуют (нейтральны).

    Согласие взвешивается: ближайший младший TF весомее дальнего
    (экспоненциально затухающие веса).

    Parameters
    ----------
    analyses : list[TimeframeAnalysis]
        Уже проанализированные TF (поле ``trend.direction`` используется как
        голос). Порядок — произвольный; функция сортирует по
        :data:`TIMEFRAMES_ORDER`.

    Returns
    -------
    dict[str, TimeframeConfirmation]
        Ключ — TF, значение — его подтверждение младшими. TF без младших
        (самый младший) получает пустое подтверждение с adjustment=0.
    """
    # Индексируем по TF
    by_tf: dict[str, TimeframeAnalysis] = {a.timeframe: a for a in analyses}
    order = [tf for tf in TIMEFRAMES_ORDER if tf in by_tf]

    result: dict[str, TimeframeConfirmation] = {}
    for tf in order:
        base = by_tf[tf]
        base_dir = base.trend.direction
        # Младшие TF — те, что идут раньше в порядке
        lower = [t for t in order if t != tf and _tf_rank(t) < _tf_rank(tf)]

        agreeing: list[str] = []
        disagreeing: list[str] = []
        neutral: list[str] = []

        if base_dir in ("BULLISH", "BEARISH"):
            opposite = "BEARISH" if base_dir == "BULLISH" else "BULLISH"
            weighted_agree = 0.0
            weighted_total = 0.0
            for j, lt in enumerate(lower):
                ldir = by_tf[lt].trend.direction
                # Вес: ближе (больший ранг среди младших) — весомее.
                # Затухание 0.5^расстояние от базового TF.
                dist = _tf_rank(tf) - _tf_rank(lt) - 1
                w = 0.6 ** dist
                lstrength = by_tf[lt].trend.strength / 100.0
                if ldir == base_dir:
                    agreeing.append(lt)
                    weighted_agree += w * lstrength
                    weighted_total += w * lstrength
                elif ldir == opposite:
                    disagreeing.append(lt)
                    weighted_total += w * lstrength
                else:
                    neutral.append(lt)
            confirmation_ratio = (
                weighted_agree / weighted_total if weighted_total > 0 else 0.5
            )
            # adjustment: [-50, +50]. ratio=1 → +50, ratio=0 → -50, ratio=0.5 → 0
            adjustment = (confirmation_ratio - 0.5) * 100.0
        else:
            # Базовый TF в RANGE — подтверждение не считается
            confirmation_ratio = 0.0
            adjustment = 0.0
            for lt in lower:
                ldir = by_tf[lt].trend.direction
                if ldir in ("BULLISH", "BEARISH"):
                    neutral.append(lt)  # при базовом RANGE все младшие «нейтральны»
                else:
                    neutral.append(lt)

        result[tf] = TimeframeConfirmation(
            base_timeframe=tf,
            base_direction=base_dir,
            agreeing=agreeing,
            disagreeing=disagreeing,
            neutral=neutral,
            confirmation_ratio=float(np.clip(confirmation_ratio, 0.0, 1.0)),
            adjustment=float(np.clip(adjustment, -50.0, 50.0)),
        )
    return result


def apply_confirmation(
    analyses: list[TimeframeAnalysis],
    confirmations: dict[str, TimeframeConfirmation],
) -> list[TimeframeAnalysis]:
    """Применить multi-TF поправку к силе тренда каждого TF (in-place override).

    Возвращает новый список с обновлённым ``trend.strength`` (старое значение
    не теряется — оно остаётся в ``trend.momentum.strength`` и свингах).
    Сила ограничивается 0..100. Если подтверждение меняет силу так, что она
    падает ниже порога (e.g. расхождение сильное) — направление может стать
    RANGE (тренд не подтверждён младшими).
    """
    import dataclasses

    out: list[TimeframeAnalysis] = []
    for a in analyses:
        conf = confirmations.get(a.timeframe)
        if conf is None:
            out.append(a)
            continue
        new_strength = float(np.clip(a.trend.strength + conf.adjustment, 0.0, 100.0))
        # Если младшие TF решительно против и сила упала низко — понижаем до RANGE
        new_dir = a.trend.direction
        if (
            a.trend.direction in ("BULLISH", "BEARISH")
            and conf.adjustment < -30.0
            and new_strength < 25.0
        ):
            new_dir = "RANGE"
        new_trend = dataclasses.replace(a.trend, strength=new_strength, direction=new_dir)
        out.append(
            dataclasses.replace(a, trend=new_trend, confirmation=conf)
        )
    return out


def _tf_rank(tf: str) -> int:
    """Порядковый индекс таймфрейма (младший → 0). Неизвестный → большой."""
    try:
        return TIMEFRAMES_ORDER.index(tf)
    except ValueError:
        return len(TIMEFRAMES_ORDER)

