"""Проверка эквивалентности индикаторных ядер эталонным реализациям.

Зачем: при переносе математики в `gex/domain/**` главный риск — незаметно изменить числа.
Этот тест сравнивает numpy-ядра (`gex.domain.indicators._kernels`) с **дословной транскрипцией**
прежних реализаций (`gex/rsi_novel.py:53-151`, версия до рефакторинга) на ряде краевых случаев.

Ядра — numpy-only, поэтому тест запускается даже там, где нет pandas:

    python tests/test_indicator_kernels.py     # standalone, печатает отчёт
    pytest tests/test_indicator_kernels.py -q  # на финальном этапе, вместе с остальными

Ссылки: аудит 2026-09-16, отчёт 02-ta-indicators (D-1: ATR×4 и RSI×3 — расходящиеся копии).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex.domain.indicators import _kernels as k  # noqa: E402


# ── Эталонные реализации (транскрипция gex/rsi_novel.py:53-151) ──────────────

def reference_recursive(values: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """Дословный перенос тела pine_rma/pine_ema (без pandas-обвязки)."""
    result = np.full(len(values), np.nan, dtype=float)
    if length <= 0:
        return result
    valid_positions = np.flatnonzero(~np.isnan(values))
    if len(valid_positions) < length:
        return result
    start = int(valid_positions[length - 1])
    initial_window = values[start - length + 1 : start + 1]
    if np.isnan(initial_window).any():
        for i in range(length - 1, len(values)):
            window = values[i - length + 1 : i + 1]
            if not np.isnan(window).any():
                start = i
                initial_window = window
                break
        else:
            return result
    result[start] = np.mean(initial_window)
    for i in range(start + 1, len(values)):
        value = values[i]
        if np.isnan(value):
            result[i] = np.nan
        elif np.isnan(result[i - 1]):
            result[i] = value
        else:
            result[i] = alpha * value + (1.0 - alpha) * result[i - 1]
    return result


def reference_rma(values: np.ndarray, length: int) -> np.ndarray:
    return reference_recursive(values, length, 1.0 / length) if length > 0 else np.full(len(values), np.nan)


def reference_ema(values: np.ndarray, length: int) -> np.ndarray:
    return reference_recursive(values, length, 2.0 / (length + 1.0)) if length > 0 else np.full(len(values), np.nan)


def reference_rolling_mean(values: np.ndarray, length: int) -> np.ndarray:
    """pandas rolling(window=length, min_periods=length).mean() в numpy-виде."""
    out = np.full(len(values), np.nan, dtype=float)
    if length <= 0:
        return out
    for i in range(length - 1, len(values)):
        window = values[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = np.mean(window)
    return out


def reference_rolling_std(values: np.ndarray, length: int) -> np.ndarray:
    """pandas rolling(min_periods=length).std(ddof=0) в numpy-виде."""
    out = np.full(len(values), np.nan, dtype=float)
    if length <= 0:
        return out
    for i in range(length - 1, len(values)):
        window = values[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = np.std(window, ddof=0)
    return out


# ── Наборы входных данных (включая краевые случаи) ───────────────────────────

def sample_series() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    flat = np.full(50, 7.0)
    with_nans = rng.normal(100, 5, 80)
    with_nans[[3, 4, 5, 20, 21]] = np.nan
    late_start = np.concatenate([np.full(10, np.nan), rng.normal(50, 2, 40)])
    return {
        "random": rng.normal(100, 10, 120),
        "flat": flat,
        "with_nans": with_nans,
        "late_start": late_start,
        "short": np.array([1.0, 2.0, 3.0]),
        "monotonic": np.arange(1.0, 61.0),
    }


def _assert_allclose(actual: np.ndarray, expected: np.ndarray, ctx: str) -> None:
    assert actual.shape == expected.shape, f"{ctx}: форма {actual.shape} != {expected.shape}"
    nan_actual, nan_expected = np.isnan(actual), np.isnan(expected)
    assert np.array_equal(nan_actual, nan_expected), f"{ctx}: NaN-маски различаются"
    both = ~nan_actual
    if both.any():
        assert np.allclose(actual[both], expected[both], rtol=0, atol=1e-12), (
            f"{ctx}: max|Δ| = {np.max(np.abs(actual[both] - expected[both]))}"
        )


# ── Тесты ────────────────────────────────────────────────────────────────────

def test_rma_matches_reference():
    for name, values in sample_series().items():
        for length in (1, 2, 5, 14, 200):
            _assert_allclose(k.rma(values, length), reference_rma(values, length), f"rma[{name},len={length}]")


def test_ema_matches_reference():
    for name, values in sample_series().items():
        for length in (1, 2, 5, 14, 200):
            _assert_allclose(k.ema(values, length), reference_ema(values, length), f"ema[{name},len={length}]")


def test_sma_stdev_match_rolling():
    for name, values in sample_series().items():
        for length in (1, 3, 14, 20):
            _assert_allclose(k.sma(values, length), reference_rolling_mean(values, length), f"sma[{name},len={length}]")
            _assert_allclose(k.stdev(values, length), reference_rolling_std(values, length), f"stdev[{name},len={length}]")


def test_bollinger_composition():
    values = sample_series()["random"]
    basis, upper, lower = k.bollinger(values, 20, 2.0)
    _assert_allclose(basis, reference_rolling_mean(values, 20), "bb.basis")
    dev = 2.0 * reference_rolling_std(values, 20)
    _assert_allclose(upper, reference_rolling_mean(values, 20) + dev, "bb.upper")
    _assert_allclose(lower, reference_rolling_mean(values, 20) - dev, "bb.lower")


def test_edge_cases_no_crash_and_nan():
    """length <= 0, ряд короче окна, полностью NaN — не падаем и не выдумываем числа."""
    for values in (np.array([]), np.array([1.0, 2.0]), np.full(10, np.nan)):
        for fn in (k.rma, k.ema, k.sma, k.stdev):
            out = fn(values, 14)
            assert out.shape == (len(values),), f"{fn.__name__}: форма результата"
            assert np.isnan(out).all(), f"{fn.__name__}: ожидались NaN на неполном ряде"
    assert np.isnan(k.rma(np.arange(10.0), 0)).all()
    assert np.isnan(k.ema(np.arange(10.0), -3)).all()


def test_rma_seed_is_sma_of_first_solid_window():
    """Документируем ключевую семантику: сид = SMA первого сплошного окна."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = k.rma(values, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    seed = np.mean(values[:3])
    assert np.isclose(out[2], seed), "сид RMA должен быть равен SMA первых трёх значений"
    # alpha = 1/3: rma[3] = alpha*x[3] + (1-alpha)*rma[2]; сравниваем с допуском,
    # потому что (1 - 1/3) и 2/3 различаются на 1 ulp (а не из-за ошибки алгоритма)
    expected_next = seed + (values[3] - seed) / 3
    assert np.isclose(out[3], expected_next, rtol=1e-15), f"rma[3]={out[3]} != {expected_next}"


# ── ATR: сверка канона со всеми прежними реализациями (8 мест, 2 математики) ──

from gex.domain.indicators import atr as a  # noqa: E402


def _ohlc(seed: int = 7, n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0.5, 0.3, n))
    low = close - np.abs(rng.normal(0.5, 0.3, n))
    return high, low, close


def reference_tr_loop(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """TR как в hybrid_trend.compute_atr: tr[0] = h[0]-l[0], далее максимум трёх разностей."""
    n = len(high)
    tr = np.empty(n, dtype=float)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return tr


def reference_atr_loop(high, low, close, period: int) -> np.ndarray:
    """Транскрипция hybrid_trend.compute_atr / novel_candles.compute_atr / trading_algorithm._atr."""
    tr = reference_tr_loop(high, low, close)
    n = len(tr)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    alpha = 1.0 / float(period)
    out[0] = tr[0]
    for i in range(1, n):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return out


def reference_atr_ewm_min_periods(high, low, close, period: int) -> np.ndarray:
    """Семантика pandas ``ewm(alpha=1/p, adjust=False, min_periods=p)`` на NaN-free TR.

    Значения совпадают с рекурсией; первые ``period-1`` помечаются NaN. Так работали
    ``trendlines._wilder_atr``, ``trend_regime`` (колонка atr) и ``breadth_imoex_service``.
    """
    out = reference_atr_loop(high, low, close, period).copy()
    head = min(period - 1, len(out))
    out[:head] = np.nan
    return out


def reference_atr_mean_tail(high, low, close, period: int) -> float | None:
    """Транскрипция gexcone.atr_14: простое среднее TR за period, None при нехватке истории."""
    tr = reference_tr_loop(high, low, close)
    tr = tr[~np.isnan(tr)]
    n = len(high)
    if n < period + 2 or len(tr) < period:
        return None
    value = float(np.mean(tr[-period:]))
    return value if np.isfinite(value) and value > 0 else None


def test_atr_matches_loop_family():
    """#1/#2/#7 (hybrid_trend, novel_candles с period=200, trading_algorithm) — без прогрева."""
    high, low, close = _ohlc()
    for period in (14, 200):
        _assert_allclose(
            a.atr(high, low, close, period, method="wilder", warmup="none"),
            reference_atr_loop(high, low, close, period),
            f"atr.loop[period={period}]",
        )


def test_atr_matches_ewm_min_periods_family():
    """#3/#4/#5 (trendlines, trend_regime, breadth_imoex) — прогрев = period."""
    high, low, close = _ohlc()
    for period in (5, 14, 20):
        _assert_allclose(
            a.atr(high, low, close, period, method="wilder", warmup="period"),
            reference_atr_ewm_min_periods(high, low, close, period),
            f"atr.ewm[period={period}]",
        )


def test_true_range_first_bar_is_hl():
    high, low, close = _ohlc()
    tr = a.true_range(high, low, close)
    assert tr[0] == high[0] - low[0], "на первом баре TR = H-L (нет предыдущего close)"
    _assert_allclose(tr, reference_tr_loop(high, low, close), "true_range")


def test_atr_scalar_trendlines_policy():
    """#3: скаляр с фоллбэком mean(H-L) → 1.0, включая короткие ряды."""
    high, low, close = _ohlc()
    got = a.atr_last(high, low, close, 14, method="wilder", fallback="hl_mean_then_one")
    expected_series = reference_atr_ewm_min_periods(high, low, close, 14)
    assert got is not None and abs(got - float(expected_series[-1])) < 1e-12

    short_h, short_l, short_c = high[:5], low[:5], close[:5]
    got_short = a.atr_last(short_h, short_l, short_c, 14, method="wilder", fallback="hl_mean_then_one")
    assert got_short is not None and got_short > 0, "фоллбэк обязан вернуть положительное число"

    flat = a.atr_last(np.full(30, 5.0), np.full(30, 5.0), np.full(30, 5.0), 14,
                      method="wilder", fallback="hl_mean_then_one")
    assert flat == 1.0, "нулевой размах → фоллбэк 1.0 (как в trendlines._wilder_atr)"


def test_atr_scalar_gexcone_mean_policy():
    """#6: /gexcone считает среднее TR и возвращает None при нехватке истории."""
    high, low, close = _ohlc()
    assert a.atr_last(high, low, close, 14, method="mean", fallback="none_if_insufficient") == \
        reference_atr_mean_tail(high, low, close, 14)
    short_h, short_l, short_c = high[:15], low[:15], close[:15]
    assert a.atr_last(short_h, short_l, short_c, 14, method="mean",
                      fallback="none_if_insufficient") is None


def test_atr_scalar_macd_policy_none_when_insufficient():
    """#8: macd_trend._atr требует n >= period+1 и иначе возвращает None."""
    high, low, close = _ohlc()
    assert a.atr_last(high[:14], low[:14], close[:14], 14, method="wilder", fallback="none") is None
    assert a.atr_last(high, low, close, 14, method="wilder", fallback="none") is not None


def test_atr_methods_diverge_documented():
    """Фиксируем ФАКТ расхождения: mean-family и wilder-family не совпадают (это и был баг).

    Тест существует, чтобы расхождение нельзя было «случайно устранить» без решения владельца:
    если числа начнут совпадать, тест упадёт и заставит обновить решение.
    """
    high, low, close = _ohlc()
    wilder = a.atr_last(high, low, close, 14, method="wilder", fallback="none")
    mean = a.atr_last(high, low, close, 14, method="mean", fallback="none_if_insufficient")
    assert wilder is not None and mean is not None
    assert abs(wilder - mean) > 1e-9, "методы обязаны различаться — это разные определения ATR"


# ── RSI: сверка канона с тремя прежними реализациями ─────────────────────────

from gex.domain.indicators import rsi as r  # noqa: E402


def reference_rsi_ta(close: np.ndarray, period: int) -> np.ndarray:
    """Транскрипция ta._wilder_rsi (без pandas): прогрев 50, плоский ряд 50, чистый рост 100."""
    n = len(close)
    if n <= period:
        return np.full(n, 50.0)
    delta = np.empty(n, dtype=float)
    delta[0] = np.nan
    delta[1:] = np.diff(close)
    gain = np.where(np.isnan(delta), np.nan, np.clip(delta, 0.0, None))
    loss = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0.0, None))

    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0.0, np.nan, avg_gain / avg_loss)
        out = 100.0 - (100.0 / (1.0 + rs))
    warm = np.isnan(avg_gain) & np.isnan(avg_loss)          # прогрев → 50
    out = np.where(warm, 50.0, out)
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)            # плоский ряд → 50
    out = np.where(flat, 50.0, out)
    out = np.where(np.isnan(out), 100.0, out)               # только рост → 100
    return np.clip(out, 0.0, 100.0)


def reference_rsi_volatility_cone(close: np.ndarray, period: int) -> np.ndarray:
    """Транскрипция volatility_cone.compute_rsi_wilder: прогрев NaN, avg_loss==0 → 100."""
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.sum(gain[1 : period + 1]) / period
    avg_loss = np.sum(loss[1 : period + 1]) / period
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _rsi_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    return {
        "random_walk": 100 + np.cumsum(rng.normal(0, 1, 150)),
        "uptrend": np.linspace(10, 60, 80),
        "downtrend": np.linspace(60, 10, 80),
        "flat": np.full(80, 42.0),
        "short": np.array([1.0, 2.0, 3.0]),
    }


def test_wilder_rsi_matches_ta_reference():
    for name, close in _rsi_samples().items():
        for period in (2, 14, 30):
            _assert_allclose(
                r.wilder_rsi(close, period, warmup="neutral_50", flat="neutral_50"),
                reference_rsi_ta(close, period),
                f"rsi.ta[{name},p={period}]",
            )


def test_wilder_rsi_matches_volatility_cone_reference():
    for name, close in _rsi_samples().items():
        for period in (2, 14, 30):
            _assert_allclose(
                r.wilder_rsi(close, period, warmup="nan", flat="hundred"),
                reference_rsi_volatility_cone(close, period),
                f"rsi.vc[{name},p={period}]",
            )


def test_wilder_rsi_variants_diverge_documented():
    """Расхождение ta vs volatility_cone — факт, зафиксированный аудитом (02 D-6).

    Тест падает, если расхождение исчезнет молча: тогда нужно обновить решение владельца.
    """
    flat = _rsi_samples()["flat"]
    ta_variant = r.wilder_rsi(flat, 14, warmup="neutral_50", flat="neutral_50")
    vc_variant = r.wilder_rsi(flat, 14, warmup="nan", flat="hundred")
    assert np.nanmax(np.abs(ta_variant - vc_variant)) > 1.0, "варианты обязаны различаться"

    short = _rsi_samples()["short"]
    assert np.isnan(r.wilder_rsi(short, 14, warmup="nan", flat="hundred")).all(), "vc-вариант: NaN"
    assert (r.wilder_rsi(short, 14, warmup="neutral_50", flat="neutral_50") == 50.0).all(), "ta-вариант: 50"


def test_normalized_rsi_formula_and_finiteness():
    """Pine-нормированный RSI: значения конечны там, где оригинал даёт число, и лежат вокруг 50."""
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, 120))
    out = r.normalized_rsi(close, 20)
    finite = out[np.isfinite(out)]
    assert finite.size > 0, "ожидались посчитанные значения"
    assert finite.min() >= -200 and finite.max() <= 300, "значения не должны «улетать»"

    # нормировка по другой серии (в rsi_novel это close при wicks=True) — тоже валидный вызов
    out_norm = r.normalized_rsi(close, 20, norm_src=close)
    _assert_allclose(out_norm, r.normalized_rsi(close, 20), "normalized_rsi[default norm == src]")


def test_wilder_rsi_extreme_series():
    """Только рост → 100; только падение → 0; плоский → 50 (для ta-варианта)."""
    up = np.linspace(1.0, 50.0, 60)
    down = np.linspace(50.0, 1.0, 60)
    assert np.isclose(r.wilder_rsi(up, 14)[-1], 100.0)
    assert np.isclose(r.wilder_rsi(down, 14)[-1], 0.0)
    assert np.isclose(r.wilder_rsi(np.full(60, 5.0), 14)[-1], 50.0)


# ── Heikin-Ashi и линрег: сверка с прежними реализациями ─────────────────────

from gex.domain.indicators import candles as cd  # noqa: E402


def reference_ha_midpoint(o, h, l, c):
    """Транскрипция novel_candles.compute_heikin_ashi (полный OHLC, сид (O+C)/2)."""
    n = len(o)
    ha_close = np.zeros(n)
    ha_open = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)
    for i in range(n):
        ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        if i == 0:
            ha_open[i] = (o[i] + c[i]) / 2.0
        else:
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close[i])
    return ha_open, ha_high, ha_low, ha_close


def reference_ha_volatility_cone(o, h, l, c):
    """Транскрипция volatility_cone.compute_heikin_ashi (пара, сид ha_open[0] = O)."""
    n = len(o)
    ha_close = np.empty(n, dtype=np.float64)
    ha_open = np.empty(n, dtype=np.float64)
    ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    ha_open[0] = o[0]
    for i in range(1, n):
        ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return ha_open, ha_close


def reference_pine_linreg(values: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    """Транскрипция rsi_novel.pine_linreg."""
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < length:
        return out
    x = np.arange(length, dtype=float)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x * x)
    denominator = length * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return out
    for i in range(length - 1, n):
        window = values[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        sum_y = np.sum(window)
        sum_xy = np.sum(x * window)
        slope = (length * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / length
        out[i] = intercept + slope * (length - 1 - offset)
    return out


def _candles(n: int = 60, seed: int = 5):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    # первый open намеренно НЕ равен close[0]: иначе сиды Heikin-Ashi (O+C)/2 и O совпадут
    # и тест на расхождение семантик пройдёт впустую
    open_ = np.concatenate(([close[0] - 1.0], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.2, n))
    return open_, high, low, close


def test_heikin_ashi_matches_novel_candles_reference():
    o, h, l, c = _candles()
    got = cd.heikin_ashi(o, h, l, c, seed="midpoint")
    exp = reference_ha_midpoint(o, h, l, c)
    for idx, label in enumerate(("ha_open", "ha_high", "ha_low", "ha_close")):
        _assert_allclose(got[idx], exp[idx], f"heikin_ashi.midpoint[{label}]")


def test_heikin_ashi_matches_volatility_cone_reference():
    o, h, l, c = _candles()
    ha_open, _, _, ha_close = cd.heikin_ashi(o, h, l, c, seed="open")
    exp_open, exp_close = reference_ha_volatility_cone(o, h, l, c)
    _assert_allclose(ha_open, exp_open, "heikin_ashi.open[ha_open]")
    _assert_allclose(ha_close, exp_close, "heikin_ashi.open[ha_close]")


def test_heikin_ashi_seed_divergence_documented():
    """Сид (O+C)/2 против O: расхождение на первом баре, затухает вдвое за бар.

    Тест фиксирует, что обе семантики сохранены и что они реально разные — «молчаливая»
    унификация одного из вариантов станет падением этого теста.
    """
    o, h, l, c = _candles(n=12)
    mid_open, *_ = cd.heikin_ashi(o, h, l, c, seed="midpoint")
    open_open, *_ = cd.heikin_ashi(o, h, l, c, seed="open")
    assert abs(mid_open[0] - open_open[0]) > 0, "на первом баре сиды обязаны различаться"
    diff_first = abs(mid_open[0] - open_open[0])
    diff_tenth = abs(mid_open[9] - open_open[9])
    assert diff_tenth < diff_first, "разница обязана затухать (половина на бар)"


def test_linreg_matches_pine_reference():
    _, _, _, close = _candles()
    for length in (5, 11, 20):
        for offset in (0, 2):
            _assert_allclose(
                k.linreg(close, length, offset),
                reference_pine_linreg(close, length, offset),
                f"linreg[len={length},offset={offset}]",
            )


def test_linear_fit_matches_calc_linreg_custom():
    x = np.arange(20, dtype=float)
    y = 3.5 * x + 7.0
    slope, intercept = k.linear_fit(x, y)
    assert np.isclose(slope, 3.5) and np.isclose(intercept, 7.0)
    # вырожденный случай (все x одинаковы) → NaN, как в оригинале
    slope_deg, intercept_deg = k.linear_fit(np.full(5, 2.0), np.arange(5.0))
    assert np.isnan(slope_deg) and np.isnan(intercept_deg)


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- indicator kernels: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
