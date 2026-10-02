"""RSI Novel Candles — порт PineScript v5 «RSI CANDLES NOVEL BY ME» (by butinn22).

Индикатор строит RSI-свечи на базе Novel Candles: сначала стандартные свечи
сливаются с Heikin-Ashi (см. :mod:`gex.novel_candles`), затем RSI считается
отдельно для каждой цены O/H/L/C с нормализацией приращений к средней пары
баров. Дополнительно считаются скользящие средние RSI, линейная регрессия
(ta.linreg + кастомная 100-периодная регрессия с MAD), динамические уровни
(перекупленность/перепроданность/середина на основе RSI) и полосы Боллинджера.

Финальные линии:
  * ``resistance`` — усреднение верхних границ (ema(ubb,9), upperRS, upper20, upperclassic)
  * ``support``    — усреднение нижних границ (ema(lbbb,9), lowerRS, lower20, lowerclassic)
  * ``mid``        — RMA(14) от усреднённой середины

Семантика свечей: RSI_close > RSI_close[1] → бычья (зелёная), иначе медвежья.

В PineScript оригинальные графики support/resistance/mid рисуются с
``offset=1`` (сдвиг на один бар вправо) — фронтенд применяет сдвиг сам,
поэтому API отдаёт «сырые» ряды.

Источники OHLCV переиспользуют существующие фетчеры (yfinance / Bybit /
MOEX ISS) через :class:`NovelCandlesService` — без дублирования кода.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from gex.application.novel_candles import NovelCandlesService
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher, TIMEFRAMES
from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher

logger = logging.getLogger(__name__)

#: Значения по умолчанию индикатора (соответствуют input(...) в PineScript).
DEFAULT_LENN = 14          # Length (RSI)
DEFAULT_LENG = 100         # Linear Regression length
DEFAULT_PERIOD100 = 100    # Период кастомной регрессии
DEFAULT_OB_LEVEL = 75.0    # RSI Overbought
DEFAULT_OS_LEVEL = 25.0    # RSI Oversold
DEFAULT_OM_LEVEL = 50.0    # RSI Middle


# ══════════════════════════════════════════════════════════════════════
#  Pine-совместимые примитивы (ta.rma / ta.ema / ta.sma / ta.stdev /
#  ta.bb / ta.linreg) — точный порт эталонной реализации.
# ══════════════════════════════════════════════════════════════════════

def pine_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's Moving Average, семантически эквивалентный ``ta.rma()``.

    Инициализация: SMA первых ``length`` валидных значений, далее
    ``rma = alpha * value + (1 - alpha) * rma_prev``, ``alpha = 1 / length``.
    """
    series = pd.Series(series, dtype="float64")
    values = series.to_numpy()
    result = np.full(len(values), np.nan, dtype=float)

    if length <= 0:
        return pd.Series(result, index=series.index)

    valid_positions = np.flatnonzero(~np.isnan(values))
    if len(valid_positions) < length:
        return pd.Series(result, index=series.index)

    # Первое окно с length валидными значениями (Pine: RMA стартует с SMA)
    start = valid_positions[length - 1]
    initial_window = values[start - length + 1 : start + 1]

    if np.isnan(initial_window).any():
        # Fallback для рядов с пропусками: ищем первое сплошное окно
        for i in range(length - 1, len(values)):
            window = values[i - length + 1 : i + 1]
            if not np.isnan(window).any():
                start = i
                initial_window = window
                break
        else:
            return pd.Series(result, index=series.index)

    result[start] = np.mean(initial_window)
    alpha = 1.0 / length

    for i in range(start + 1, len(values)):
        value = values[i]
        if np.isnan(value):
            result[i] = np.nan
        elif np.isnan(result[i - 1]):
            result[i] = value
        else:
            result[i] = alpha * value + (1.0 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def pine_ema(series: pd.Series, length: int) -> pd.Series:
    """EMA, семантически эквивалентная ``ta.ema()``.

    ``alpha = 2 / (length + 1)``, сид — SMA первых ``length`` валидных значений.
    """
    series = pd.Series(series, dtype="float64")
    values = series.to_numpy()
    result = np.full(len(values), np.nan, dtype=float)

    if length <= 0:
        return pd.Series(result, index=series.index)

    valid_positions = np.flatnonzero(~np.isnan(values))
    if len(valid_positions) < length:
        return pd.Series(result, index=series.index)

    start = valid_positions[length - 1]
    initial_window = values[start - length + 1 : start + 1]

    if np.isnan(initial_window).any():
        for i in range(length - 1, len(values)):
            window = values[i - length + 1 : i + 1]
            if not np.isnan(window).any():
                start = i
                initial_window = window
                break
        else:
            return pd.Series(result, index=series.index)

    result[start] = np.mean(initial_window)
    alpha = 2.0 / (length + 1.0)

    for i in range(start + 1, len(values)):
        value = values[i]
        if np.isnan(value):
            result[i] = np.nan
        elif np.isnan(result[i - 1]):
            result[i] = value
        else:
            result[i] = alpha * value + (1.0 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def pine_sma(series: pd.Series, length: int) -> pd.Series:
    """Простое скользящее среднее, ``ta.sma()``."""
    return series.rolling(window=length, min_periods=length).mean()


def pine_stdev(series: pd.Series, length: int) -> pd.Series:
    """Стандартное отклонение, ``ta.stdev()`` (по умолчанию biased, ddof=0)."""
    return series.rolling(window=length, min_periods=length).std(ddof=0)


def pine_bb(series: pd.Series, length: int, mult: float):
    """Полосы Боллинджера, ``ta.bb(source, length, mult)``.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (basis, upper, lower) = (SMA, SMA + mult*stdev, SMA - mult*stdev).
    """
    basis = pine_sma(series, length)
    deviation = mult * pine_stdev(series, length)
    return basis, basis + deviation, basis - deviation


def pine_linreg(series: pd.Series, length: int, offset: int = 0) -> pd.Series:
    """Линейная регрессия, ``ta.linreg(source, length, offset)``.

    Регрессия оценивается в точке ``x = length - 1 - offset`` окна
    (offset=0 → последний бар окна).
    """
    series = pd.Series(series, dtype="float64")
    values = series.to_numpy()
    result = np.full(len(values), np.nan, dtype=float)

    if length <= 0:
        return pd.Series(result, index=series.index)

    x = np.arange(length, dtype=float)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x * x)
    denominator = length * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return pd.Series(result, index=series.index)

    for i in range(length - 1, len(values)):
        y = values[i - length + 1 : i + 1]
        if np.isnan(y).any():
            continue
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        slope = (length * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / length
        result[i] = intercept + slope * (length - 1 - offset)

    return pd.Series(result, index=series.index)


# ══════════════════════════════════════════════════════════════════════
#  Кастомная линейная регрессия + MAD (calc_linreg / calc_mad из Pine)
# ══════════════════════════════════════════════════════════════════════

def calc_linreg_custom(x, y) -> tuple[float, float]:
    """Наклон и intercept регрессии y = slope*x + intercept (МНК)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return np.nan, np.nan

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def calc_mad_custom(x, y, slope: float, intercept: float) -> float:
    """Mean Absolute Deviation регрессионных остатков (``calc_mad`` из Pine)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return np.nan
    predicted = slope * x + intercept
    return float(np.mean(np.abs(y - predicted)))


# ══════════════════════════════════════════════════════════════════════
#  Сервис
# ══════════════════════════════════════════════════════════════════════

class RsiNovelService:
    """Сервис RSI Novel Candles: OHLCV → novel-свечи → RSI-свечи → линии.

    Параметры
    ---------
    redis_client : RedisClient | None
        Клиент Redis для кэширования (пробрасывается в фетчеры).
    """

    def __init__(self, redis_client: Optional[RedisClient] = None):
        # Переиспользуем NovelCandlesService: детект типа актива, фетч OHLCV
        # (yfinance / Bybit / MOEX) и сам алгоритм novel-трансформации.
        self._novel_svc = NovelCandlesService(redis_client=redis_client)
        self._ta_fetcher = create_timeframes_fetcher(redis_client=redis_client)

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #
    def fetch_and_analyze(
        self,
        ticker: str,
        timeframe: str = "1d",
        limit: int = 500,
        lenn: int = DEFAULT_LENN,
        wicks: bool = True,
        leng: int = DEFAULT_LENG,
        period100: int = DEFAULT_PERIOD100,
        ob_level: float = DEFAULT_OB_LEVEL,
        os_level: float = DEFAULT_OS_LEVEL,
        om_level: float = DEFAULT_OM_LEVEL,
        length12: int = DEFAULT_LENN,
    ) -> dict:
        """Получить OHLCV → novel-свечи → RSI Novel и сериализовать.

        Returns
        -------
        dict
            Ключи: ``ticker``, ``timeframe``, ``asset_type``, ``n_bars``,
            ``bars`` (RSI-свечи), ``rsi_close``, ``rsi_open``, ``rsi_avg``,
            ``rsi_ma_fast``, ``rsi_ma``, ``rsi_ma3``, ``trend_ma``,
            ``linear_reg_curve``, ``resistance``, ``support``, ``mid``,
            ``slope``, ``mad``, ``last_rsi``, ``signal``, ``rsi_length``,
            ``levels``.
        """
        ticker = ticker.strip().upper()
        tf = timeframe.strip().lower()
        if tf not in TIMEFRAMES:
            raise ValueError(
                f"Неподдерживаемый таймфрейм '{timeframe}'. Доступно: {', '.join(TIMEFRAMES)}"
            )

        asset_type = NovelCandlesService._detect_asset_type(ticker)
        logger.info(
            "RsiNovel: ticker=%s, tf=%s, asset=%s, limit=%d, lenn=%d, wicks=%s",
            ticker, tf, asset_type, limit, lenn, wicks,
        )

        # --- 1. OHLCV через существующие фетчеры ---
        df_raw = self._novel_svc._fetch_ohlcv(ticker, tf, asset_type, limit)
        if df_raw is None or len(df_raw) == 0:
            raise ValueError(f"Нет OHLCV данных для '{ticker}' [{tf}]")

        # --- 2. Novel-трансформация (переиспользуем алгоритм Novel Candles) ---
        novel_df = NovelCandlesService.compute_novel_candles(df_raw)

        # --- 3. RSI Novel ---
        rdf = self.compute_rsi_novel(
            novel_df,
            lenn=lenn,
            wicks=wicks,
            leng=leng,
            period100=period100,
            ob_level=ob_level,
            os_level=os_level,
            om_level=om_level,
            length12=length12,
        )

        return self._serialize(ticker, tf, asset_type, novel_df, rdf, ob_level, os_level, om_level, lenn)

    # ------------------------------------------------------------------ #
    #  Алгоритм (порт PineScript)
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_rsi_novel(
        novel_df: pd.DataFrame,
        lenn: int = DEFAULT_LENN,
        wicks: bool = True,
        leng: int = DEFAULT_LENG,
        period100: int = DEFAULT_PERIOD100,
        ob_level: float = DEFAULT_OB_LEVEL,
        os_level: float = DEFAULT_OS_LEVEL,
        om_level: float = DEFAULT_OM_LEVEL,
        length12: int = DEFAULT_LENN,
    ) -> pd.DataFrame:
        """Рассчитать RSI-свечи и линии по DataFrame novel-свечей.

        Ожидаются столбцы ``Open, High, Low, Close`` (хронологический порядок,
        старые → новые). Возвращает DataFrame со всеми сериями индикатора.

        Raises
        ------
        ValueError
            Если отсутствуют обязательные столбцы или DataFrame пуст.
        """
        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in novel_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if novel_df.empty:
            raise ValueError("DataFrame пуст — не из чего строить RSI-свечи")

        df = novel_df.copy()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        src_open = df["Open"]
        src_high = df["High"]
        src_low = df["Low"]
        src_close = df["Close"]

        # ---------------------------------------------------------------- #
        #  RSI для каждой цены: 50 + 50 * rma(gain/norm) / rma(|gain|/norm)
        # ---------------------------------------------------------------- #
        def _rsi(src: pd.Series, use_wicks_norm: bool) -> pd.Series:
            if use_wicks_norm:
                norm = (src + src.shift(1)) / 2.0
            else:
                norm = (src_close + src_close.shift(1)) / 2.0
            gain_loss = pine_change(src) / norm
            with np.errstate(divide="ignore", invalid="ignore"):
                rsi = 50.0 + 50.0 * pine_rma(gain_loss, lenn) / pine_rma(gain_loss.abs(), lenn)
            return rsi.replace([np.inf, -np.inf], np.nan)

        RSI_close = _rsi(src_close, True)
        RSI_open = _rsi(src_open, wicks)
        RSI_high = _rsi(src_high, wicks)
        RSI_low = _rsi(src_low, wicks)

        # Коррекция high/low: свеча всегда с корректным порядком
        RSI_high_fixed = pd.concat([RSI_high, RSI_open, RSI_close], axis=1).max(axis=1)
        RSI_low_fixed = pd.concat([RSI_low, RSI_open, RSI_close], axis=1).min(axis=1)

        # ---------------------------------------------------------------- #
        #  Скользящие средние RSI
        # ---------------------------------------------------------------- #
        rsi_avg = (RSI_close + RSI_low + RSI_high + RSI_open) / 4.0
        rsiMA = pine_ema(rsi_avg, 20)
        rsiMA3 = pine_ema(rsi_avg, 50)
        rsiMAfast = pine_ema(rsi_avg, 10)

        # ---------------------------------------------------------------- #
        #  Линейная регрессия
        # ---------------------------------------------------------------- #
        close_price = rsi_avg  # Pine: rsiMA2 = avg(RSI O/H/L/C)
        linear_reg = pine_linreg(close_price, leng, 0)

        # Кастомная period100-регрессия с MAD (Pine: calc_linreg/calc_mad)
        slope_series = pd.Series(np.nan, index=df.index, dtype=float)
        intercept_series = pd.Series(np.nan, index=df.index, dtype=float)
        mad_series = pd.Series(np.nan, index=df.index, dtype=float)
        lin_reg_series = pd.Series(np.nan, index=df.index, dtype=float)

        if period100 >= 1 and len(df) >= period100:
            x_arr = np.arange(period100 - 1, -1, -1, dtype=float)
            for i in range(period100 - 1, len(df)):
                # Pine: unshift(i) → x = [period-1..0],
                # y = rsiMA3[i-(period-1)] .. rsiMA3[i] (обратный порядок)
                y_arr = np.array(
                    [rsiMA3.iloc[i - j] for j in range(period100 - 1, -1, -1)],
                    dtype=float,
                )
                if np.isnan(y_arr).any():
                    continue
                slope, intercept = calc_linreg_custom(x_arr, y_arr)
                if np.isnan(slope):
                    continue
                slope_series.iloc[i] = slope
                intercept_series.iloc[i] = intercept
                mad_series.iloc[i] = calc_mad_custom(x_arr, y_arr, slope, intercept)
                lin_reg_series.iloc[i] = slope * (period100 - 1) + intercept

        # Итоговая кривая регрессии: avg(linear_reg, rsiMA3, linear_reg, rsiMA3, lin_reg)
        linear_reg_curve = (
            linear_reg + rsiMA3 + linear_reg + rsiMA3 + lin_reg_series
        ) / 5.0

        trend_ma = pine_ema(close_price, 33)

        # ---------------------------------------------------------------- #
        #  Динамические уровни (ubb / lbbb / lm) на основе RSI
        # ---------------------------------------------------------------- #
        ep = 2 * length12 - 1
        auc = pine_ema((RSI_close - RSI_close.shift(1)).clip(lower=0), ep)
        adc = pine_ema((RSI_close.shift(1) - RSI_close).clip(lower=0), ep)

        def _dyn_level(level: float) -> pd.Series:
            x = (length12 - 1) * (adc * level / (100.0 - level) - auc)
            return pd.Series(
                np.where(x >= 0, RSI_close + x, RSI_close + x * (100.0 - level) / level),
                index=df.index,
                dtype=float,
            )

        ubb = _dyn_level(ob_level)
        lbbb = _dyn_level(os_level)
        lm = _dyn_level(om_level)

        # ---------------------------------------------------------------- #
        #  Полосы Боллинджера
        # ---------------------------------------------------------------- #
        middle_source = rsi_avg
        middlelll, upperRS, lowerRS = pine_bb(middle_source, 20, 2.618)
        middle, upper20, lower20 = pine_bb(RSI_close, 20, 2.0)
        middleclassic, upperclassic, lowerclassic = pine_bb(RSI_close, 5, 4.0)

        # ---------------------------------------------------------------- #
        #  Финальные линии: Resistance / Support / Mid
        # ---------------------------------------------------------------- #
        ema_ubb_9 = pine_ema(ubb, 9)
        resistance = (
            ema_ubb_9 + upperRS + upperRS + ema_ubb_9 + upperRS + upperRS + upper20 + upperclassic
        ) / 8.0

        ema_lbbb_9 = pine_ema(lbbb, 9)
        support = (
            ema_lbbb_9 + lowerRS + lowerRS + ema_lbbb_9 + lowerRS + lowerRS + lower20 + lowerclassic
        ) / 8.0

        ema_lm_9 = pine_ema(lm, 9)
        mid_source = (
            ema_lm_9 + middlelll + middlelll + ema_lm_9 + middlelll + middlelll + middle + middleclassic
        ) / 8.0
        mid = pine_rma(mid_source, 14)

        candle_direction = np.where(RSI_close > RSI_close.shift(1), 1, -1)

        return pd.DataFrame(
            {
                "novel_open": src_open,
                "novel_high": src_high,
                "novel_low": src_low,
                "novel_close": src_close,
                "RSI_open": RSI_open,
                "RSI_high": RSI_high,
                "RSI_low": RSI_low,
                "RSI_close": RSI_close,
                "RSI_high_fixed": RSI_high_fixed,
                "RSI_low_fixed": RSI_low_fixed,
                "rsi_avg": rsi_avg,
                "rsiMA": rsiMA,
                "rsiMA3": rsiMA3,
                "rsiMAfast": rsiMAfast,
                "linear_reg": linear_reg,
                "custom_slope": slope_series,
                "custom_intercept": intercept_series,
                "custom_mad": mad_series,
                "lin_reg": lin_reg_series,
                "linear_reg_curve": linear_reg_curve,
                "trend_ma": trend_ma,
                "ubb": ubb,
                "lbbb": lbbb,
                "lm": lm,
                "middlelll": middlelll,
                "upperRS": upperRS,
                "lowerRS": lowerRS,
                "middle": middle,
                "upper20": upper20,
                "lower20": lower20,
                "middleclassic": middleclassic,
                "upperclassic": upperclassic,
                "lowerclassic": lowerclassic,
                "resistance": resistance,
                "support": support,
                "mid": mid,
                "candle_direction": candle_direction,
            },
            index=df.index,
        )

    # ------------------------------------------------------------------ #
    #  Сериализация
    # ------------------------------------------------------------------ #
    @staticmethod
    def _serialize(
        ticker: str,
        tf: str,
        asset_type: str,
        novel_df: pd.DataFrame,
        rdf: pd.DataFrame,
        ob_level: float,
        os_level: float,
        om_level: float,
        lenn: int,
    ) -> dict:
        """Преобразовать DataFrame в JSON-совместимый ответ API."""
        n = len(rdf)

        def _col(name: str) -> list[Optional[float]]:
            if name not in rdf.columns:
                return [None] * n
            return [
                None if pd.isna(v) else round(float(v), 4)
                for v in rdf[name].tolist()
            ]

        bars = []
        for idx, (ts, row) in enumerate(rdf.iterrows()):
            bars.append({
                "time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": _clean(row["RSI_open"]),
                "high": _clean(row["RSI_high_fixed"]),
                "low": _clean(row["RSI_low_fixed"]),
                "close": _clean(row["RSI_close"]),
            })

        # Последние значения для stat-карточек
        last_rsi = _last_finite(rdf["RSI_close"])
        slope = _last_finite(rdf["custom_slope"])
        mad = _last_finite(rdf["custom_mad"])

        if last_rsi is None:
            signal = None
        elif last_rsi >= ob_level:
            signal = "overbought"
        elif last_rsi <= os_level:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "ticker": ticker,
            "timeframe": tf,
            "asset_type": asset_type,
            "n_bars": n,
            "bars": bars,
            "rsi_close": _col("RSI_close"),
            "rsi_open": _col("RSI_open"),
            "rsi_avg": _col("rsi_avg"),
            "rsi_ma_fast": _col("rsiMAfast"),
            "rsi_ma": _col("rsiMA"),
            "rsi_ma3": _col("rsiMA3"),
            "trend_ma": _col("trend_ma"),
            "linear_reg_curve": _col("linear_reg_curve"),
            "resistance": _col("resistance"),
            "support": _col("support"),
            "mid": _col("mid"),
            "slope": None if slope is None else round(slope, 6),
            "mad": None if mad is None else round(mad, 6),
            "last_rsi": last_rsi,
            "signal": signal,
            "rsi_length": lenn,
            "levels": {"ob": ob_level, "os": os_level, "om": om_level},
        }


# ══════════════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════

def pine_change(series: pd.Series, length: int = 1) -> pd.Series:
    """``ta.change(source, length)`` → ``source - source[length]``."""
    return series - series.shift(length)


def _clean(v) -> Optional[float]:
    """float с округлением до 4 знаков; NaN/Inf → None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, 4)


def _last_finite(series: pd.Series) -> Optional[float]:
    """Последнее конечное значение серии (NaN/Inf пропускаются)."""
    for v in reversed(series.tolist()):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            return round(f, 4)
    return None
