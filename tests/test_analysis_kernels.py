"""Проверка эквивалентности анализа (пивоты/структура) прежним реализациям.

Как и `test_indicator_kernels.py`: сравнивает канон `gex.domain.analysis.*` с дословной
транскрипцией прежних фрагментов (`ta.detect_trend`, `trendlines.find_pivot_*`,
`hybrid_trend._fractals`). Запускается без pandas:

    python tests/test_analysis_kernels.py
    pytest tests/test_analysis_kernels.py -q     # на финальном этапе
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex.domain.analysis import pivots as pv  # noqa: E402


def _assert_allclose(actual: np.ndarray, expected: np.ndarray, ctx: str) -> None:
    """Сравнение массивов с проверкой NaN-маски (тот же хелпер, что в test_indicator_kernels)."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{ctx}: форма {actual.shape} != {expected.shape}"
    nan_actual, nan_expected = np.isnan(actual), np.isnan(expected)
    assert np.array_equal(nan_actual, nan_expected), f"{ctx}: NaN-маски различаются"
    both = ~nan_actual
    if both.any():
        assert np.allclose(actual[both], expected[both], rtol=0, atol=1e-12), (
            f"{ctx}: max|Δ| = {np.max(np.abs(actual[both] - expected[both]))}"
        )


# ── Эталоны (транскрипции прежних реализаций) ────────────────────────────────

def reference_ta_detect_trend(high: np.ndarray, low: np.ndarray, k: int) -> tuple[list[int], list[int]]:
    """Транскрипция блока ta.py:749-755 (детект тренда)."""
    swing_highs_idx, swing_lows_idx = [], []
    for i in range(k, len(high) - k):
        win_h = high[i - k : i + k + 1]
        win_l = low[i - k : i + k + 1]
        if high[i] == win_h.max() and np.sum(win_h == high[i]) == 1:
            swing_highs_idx.append(i)
        if low[i] == win_l.min() and np.sum(win_l == low[i]) == 1:
            swing_lows_idx.append(i)
    return swing_highs_idx, swing_lows_idx


def reference_ta_detect_divergences(high: np.ndarray, low: np.ndarray, k: int = 2) -> tuple[list[int], list[int]]:
    """Транскрипция блока ta.py:631-640 (внутри detect_divergences, k=2 захардкожен)."""
    sh, sl = [], []
    for i in range(k, len(high) - k):
        if high[i] == high[i - k : i + k + 1].max() and np.sum(high[i - k : i + k + 1] == high[i]) == 1:
            sh.append(i)
        if low[i] == low[i - k : i + k + 1].min() and np.sum(low[i - k : i + k + 1] == low[i]) == 1:
            sl.append(i)
    return sh, sl


def reference_trendlines_pivot(series: np.ndarray, left: int, right: int, kind: str) -> list[int]:
    """Транскрипция trendlines.find_pivot_highs/find_pivot_lows (:494-526)."""
    out = []
    n = len(series)
    for i in range(left, n - right):
        center = series[i]
        if np.isnan(center):
            continue
        window = series[i - left : i + right + 1]
        if kind == "high" and center >= window.max() and np.sum(window == center) == 1:
            out.append(i)
        if kind == "low" and center <= window.min() and np.sum(window == center) == 1:
            out.append(i)
    return out


def reference_hybrid_fractals(series: np.ndarray, left: int, right: int, strict: bool) -> np.ndarray:
    """Транскрипция hybrid_trend._fractals (:294-319) для верхней серии."""
    n = len(series)
    mask = np.zeros(n, dtype=bool)
    if n < left + right + 1 or (left == 0 and right == 0):
        return mask
    for i in range(left, n - right):
        window = series[i - left : i + right + 1]
        center = window[left]
        others = np.concatenate([window[:left], window[left + 1 :]])
        if len(others) == 0:
            continue
        mx = float(np.max(others))
        mn = float(np.min(others))
        mask[i] = bool(center > mx) if strict else bool(center >= mx and center > mn)
    return mask


# ── Данные ───────────────────────────────────────────────────────────────────

def _series(n: int = 120, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0, 1.4, n))
    high = base + np.abs(rng.normal(0.5, 0.3, n))
    low = base - np.abs(rng.normal(0.5, 0.3, n))
    return high, low


def _with_plateau() -> np.ndarray:
    """Серия с плато на вершине (равные соседи) — различает strict и plateau."""
    return np.array([1.0, 2.0, 5.0, 5.0, 3.0, 2.0, 1.0, 4.0, 2.0, 1.0])


# ── Тесты ────────────────────────────────────────────────────────────────────

def test_pivots_match_ta_detect_trend():
    high, low = _series()
    for k in (1, 2, 3, 5):
        exp_h, exp_l = reference_ta_detect_trend(high, low, k)
        assert list(pv.pivot_indices(high, k, k, kind="high")) == exp_h, f"swing-highs при k={k}"
        assert list(pv.pivot_indices(low, k, k, kind="low")) == exp_l, f"swing-lows при k={k}"


def test_pivots_match_ta_detect_divergences_k2():
    """Копия внутри detect_divergences давала k=2 — сверяем с той же семантикой."""
    high, low = _series()
    exp_h, exp_l = reference_ta_detect_divergences(high, low, 2)
    assert list(pv.pivot_indices(high, 2, 2, kind="high")) == exp_h
    assert list(pv.pivot_indices(low, 2, 2, kind="low")) == exp_l


def test_pivots_match_trendlines_asymmetric_windows():
    """У trendlines окна left/right задаются раздельно (в вызовах 5/5) — проверяем и асимметрию."""
    high, low = _series()
    for left, right in ((5, 5), (2, 5), (5, 2), (1, 3)):
        assert list(pv.pivot_indices(high, left, right, kind="high")) == reference_trendlines_pivot(high, left, right, "high")
        assert list(pv.pivot_indices(low, left, right, kind="low")) == reference_trendlines_pivot(low, left, right, "low")


def test_pivots_match_hybrid_trend_strict_mode():
    high, low = _series()
    for left, right in ((5, 5), (3, 3)):
        exp = reference_hybrid_fractals(high, left, right, strict=True)
        got = pv.pivot_mask(high, left, right, mode="strict")
        assert np.array_equal(got, exp), f"hybrid strict при left={left}, right={right}"


def test_pivots_match_hybrid_trend_plateau_mode():
    high, _ = _series()
    for left, right in ((5, 5), (3, 3)):
        exp = reference_hybrid_fractals(high, left, right, strict=False)
        got = pv.pivot_mask(high, left, right, mode="plateau")
        assert np.array_equal(got, exp), f"hybrid plateau при left={left}, right={right}"


def test_strict_and_plateau_differ_on_plateau_series():
    """Плато на вершине: strict его не считает пивотом, plateau — считает.

    Тест фиксирует, что режимы различны и что ни один не «унифицирован» молча.
    """
    series = _with_plateau()
    strict = pv.pivot_indices(series, 1, 1, mode="strict")
    plateau = pv.pivot_indices(series, 1, 1, mode="plateau")
    assert 2 not in strict and 3 not in strict, "при плато строгий режим не должен давать пивот"
    assert 2 in plateau and 3 in plateau, "нестрогий режим обязан увидеть плато"

    flat = np.full(20, 7.0)
    assert pv.pivot_indices(flat, 2, 2, mode="plateau").size == 0, "полностью плоское окно — не пивот"


def test_pivot_prices_alignment():
    high, _ = _series()
    idx = pv.pivot_indices(high, 3, 3)
    prices = pv.pivot_prices(high, 3, 3)
    assert len(prices) == len(idx)
    assert np.allclose(prices, high[idx])


def test_pivots_with_nan_and_short_series():
    short = np.array([1.0, 2.0, 3.0])
    assert pv.pivot_indices(short, 5, 5).size == 0, "недостаток данных → пустой результат"
    with_nan = np.array([1.0, np.nan, 5.0, 2.0, 1.0, 3.0, 1.0])
    assert pv.pivot_indices(with_nan, 1, 1).size >= 0  # не падаем на NaN
    assert pv.pivot_indices(np.arange(10.0), 0, 0).size == 0, "left=right=0 — вырожденное окно"


# ── Структура HH/HL/LH/LL: сверка с двумя правилами ──────────────────────────

from gex.domain.analysis import structure as st  # noqa: E402


def reference_count_swings(hh_prices, ll_prices):
    """Транскрипция подсчёта из ta.py:760-773 и trendlines.py:529-550 (идентичны)."""
    higher_highs = lower_highs = higher_lows = lower_lows = 0
    for j in range(1, len(hh_prices)):
        if hh_prices[j] > hh_prices[j - 1]:
            higher_highs += 1
        elif hh_prices[j] < hh_prices[j - 1]:
            lower_highs += 1
    for j in range(1, len(ll_prices)):
        if ll_prices[j] > ll_prices[j - 1]:
            higher_lows += 1
        elif ll_prices[j] < ll_prices[j - 1]:
            lower_lows += 1
    return higher_highs, lower_highs, higher_lows, lower_lows


def reference_ta_swing_direction(c) -> str:
    """Транскрипция ta.py:774-787."""
    hh, lh, hl, ll = c
    bull, bear = hh + hl, lh + ll
    total = bull + bear
    if total > 0 and bull > bear and hh >= 1 and hl >= 1:
        return "BULLISH"
    if total > 0 and bear > bull and lh >= 1 and ll >= 1:
        return "BEARISH"
    return "RANGE"


def reference_trendlines_direction(c) -> tuple[str, float]:
    """Транскрипция trendlines.py:553-583."""
    hh, lh, hl, ll = c
    bull, bear = hh + hl, lh + ll
    total = bull + bear
    if total == 0:
        return "RANGE", 0.0
    is_bull = hh >= 1 and hl >= 1
    is_bear = lh >= 1 and ll >= 1
    if is_bull and not is_bear:
        direction = "BULLISH"
    elif is_bear and not is_bull:
        direction = "BEARISH"
    elif is_bull and is_bear:
        direction = "BULLISH" if bull > bear else ("BEARISH" if bear > bull else "RANGE")
    else:
        direction = "BULLISH" if bull > bear else ("BEARISH" if bear > bull else "RANGE")
    return direction, round(max(bull, bear) / total * 100.0, 1)


def _counts_grid():
    return [(hh, lh, hl, ll) for hh in range(3) for lh in range(3) for hl in range(3) for ll in range(3)]


def test_count_swings_matches_reference():
    rng = np.random.default_rng(23)
    for _ in range(50):
        highs = np.sort(rng.normal(100, 5, 6))
        lows = np.sort(rng.normal(95, 5, 6))
        c = st.count_swings(highs, lows)
        exp = reference_count_swings(highs, lows)
        assert (c.higher_highs, c.lower_highs, c.higher_lows, c.lower_lows) == exp


def test_ta_direction_matches_reference_on_all_combinations():
    for hh, lh, hl, ll in _counts_grid():
        counts = st.SwingCounts(hh, lh, hl, ll)
        assert st.ta_swing_direction(counts) == reference_ta_swing_direction((hh, lh, hl, ll)), (
            f"ta-правило при HH={hh},LH={lh},HL={hl},LL={ll}"
        )


def test_trendlines_direction_matches_reference_on_all_combinations():
    for hh, lh, hl, ll in _counts_grid():
        counts = st.SwingCounts(hh, lh, hl, ll)
        got = (st.trendlines_direction(counts), st.trendlines_strength(counts))
        assert got == reference_trendlines_direction((hh, lh, hl, ll)), (
            f"trendlines-правило при HH={hh},LH={lh},HL={hl},LL={ll}"
        )


def test_direction_rules_diverge_documented():
    """Один и тот же набор свингов → разные ответы страниц (находка аудита 02 D-4).

    Пример ``HH=1, HL=1, LH=3, LL=0``: ``/trendlines`` скажет BULLISH (есть HH и HL, медвежьей
    структуры нет), а ``/ta`` — RANGE (сумма свингов 2 против 3 не в пользу быков).
    Тест фиксирует расхождение: молчаливая унификация станет его падением.
    """
    counts = st.SwingCounts(higher_highs=1, lower_highs=3, higher_lows=1, lower_lows=0)
    assert st.trendlines_direction(counts) == "BULLISH"
    assert st.ta_swing_direction(counts) == "RANGE"
    divergences = [
        (hh, lh, hl, ll)
        for hh, lh, hl, ll in _counts_grid()
        if st.ta_swing_direction(st.SwingCounts(hh, lh, hl, ll)) != st.trendlines_direction(st.SwingCounts(hh, lh, hl, ll))
    ]
    assert len(divergences) >= 6, f"ожидалось несколько расхождений, найдено {len(divergences)}"


def test_momentum_veto_matches_ta_block():
    """Momentum-veto (ta.py:791-801): согласие, нейтраль, отсутствие свингов, конфликт → RANGE."""
    assert st.apply_momentum_veto("BULLISH", "BULLISH") == "BULLISH"
    assert st.apply_momentum_veto("BULLISH", "NEUTRAL") == "BULLISH"
    assert st.apply_momentum_veto("BULLISH", None) == "BULLISH"
    assert st.apply_momentum_veto("RANGE", "BEARISH") == "BEARISH"
    assert st.apply_momentum_veto("BULLISH", "BEARISH") == "RANGE"
    assert st.apply_momentum_veto("RANGE", "NEUTRAL") == "RANGE"


# ── Моментум и числовые хелперы ──────────────────────────────────────────────

from gex.domain.analysis import momentum as mo  # noqa: E402
from gex.domain.analysis import numeric as nm  # noqa: E402


def reference_sigmoid(x: float) -> float:
    """Транскрипция direction._sigmoid (:113-119)."""
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def reference_clip_z(value: float, lo: float = -3.0, hi: float = 3.0) -> float:
    """Транскрипция direction._clip_z (:106-110)."""
    if not np.isfinite(value):
        return 0.0
    return float(max(lo, min(hi, value)))


def reference_momentum_strength(o, h, low, c, vol):
    """Транскрипция ta.compute_momentum_strength (:503-577) без pandas."""
    n = len(c)
    if n < 2:
        return None
    c2c = np.diff(c) / c[:-1] * 100.0
    avg_c2c = float(np.mean(c2c))

    rng = np.zeros(max(n - 1, 0))
    prev_low = low[:-1]
    cur_high = h[1:]
    nz = prev_low > 0
    rng[nz] = (cur_high[nz] - prev_low[nz]) / prev_low[nz] * 100.0
    avg_range = float(np.mean(rng)) if len(rng) else 0.0

    body = np.abs(c - o) / np.where(o > 0, o, np.nan) * 100.0
    body = body[np.isfinite(body)]
    avg_body = float(np.mean(body)) if body.size else 0.0

    if avg_c2c > 1e-9:
        side = "BULLISH"
    elif avg_c2c < -1e-9:
        side = "BEARISH"
    else:
        side = "NEUTRAL"

    volume_trend = 1.0
    if vol is not None and side != "NEUTRAL" and n > 1:
        bar_dir = np.sign(np.diff(c))
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
            volume_trend = 1.3

    body_factor = np.tanh(avg_body / 1.0)
    vol_factor = 1.0 / (1.0 + np.exp(-(volume_trend - 1.0) * 2.5))
    raw = abs(avg_c2c) * vol_factor * (1.0 + body_factor)
    strength = 100.0 * (1.0 / (1.0 + np.exp(-8.0 * (raw - 0.15))))
    return side, float(np.clip(strength, 0.0, 100.0)), avg_c2c, avg_range, avg_body, float(volume_trend)


def _ohlcv(n: int = 25, drift: float = 0.4, seed: int = 31):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(drift, 1.0, n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.2, n))
    volume = np.abs(rng.normal(1_000_000, 150_000, n))
    return open_, high, low, close, volume


def test_sigmoid_matches_direction_reference():
    for x in (-50.0, -3.0, -0.5, 0.0, 0.5, 3.0, 50.0):
        assert abs(nm.sigmoid(x) - reference_sigmoid(x)) < 1e-12, f"sigmoid({x})"


def test_clip_z_matches_reference_and_handles_nan():
    for value in (-5.0, -3.0, 0.0, 1.234, 3.0, 9.9, float("nan"), float("inf")):
        assert nm.clip_z(value) == reference_clip_z(value), f"clip_z({value})"
    assert nm.clip_z(5.0, lo=-2.0, hi=2.0) == 2.0
    assert nm.clip_z(float("nan"), lo=-2.0, hi=2.0) == 0.0


def test_momentum_strength_matches_ta_reference():
    for drift in (0.5, -0.5, 0.0):
        o, h, low, c, vol = _ohlcv(drift=drift)
        for use_volume in (True, False):
            got = mo.momentum_strength(o, h, low, c, vol if use_volume else None)
            exp = reference_momentum_strength(o, h, low, c, vol if use_volume else None)
            assert got is not None and exp is not None
            assert got.side == exp[0], f"сторона (drift={drift}, vol={use_volume})"
            assert abs(got.strength - exp[1]) < 1e-9, f"сила (drift={drift}, vol={use_volume})"
            assert abs(got.close_to_close_pct - exp[2]) < 1e-12
            assert abs(got.range_pct - exp[3]) < 1e-12
            assert abs(got.avg_body_pct - exp[4]) < 1e-12
            assert abs(got.volume_trend - exp[5]) < 1e-12


def test_momentum_strength_all_bars_in_trend_uses_volume_base():
    """Все бары закрылись в сторону тренда → volume_trend = 1.3 (ветка оригинала)."""
    close = np.linspace(100.0, 130.0, 20)
    open_ = close - 0.5
    high = close + 0.2
    low = close - 0.7
    volume = np.full(20, 500.0)
    got = mo.momentum_strength(open_, high, low, close, volume)
    assert got is not None and got.side == "BULLISH" and abs(got.volume_trend - 1.3) < 1e-12


def test_momentum_strength_short_input_returns_none():
    assert mo.momentum_strength([1.0], [1.0], [1.0], [1.0]) is None


def test_atr_distance_momentum_matches_hybrid_reference():
    src = np.array([10.0, 12.0, 11.0, 15.0, 15.0])
    ema = np.array([11.0, 11.0, 11.0, 11.0, 20.0])
    atr = np.array([1.0, 1.0, 0.0, 2.0, 1.0])
    mult = 1.5
    expected = np.array([
        float(np.clip((src[t] - ema[t]) / (atr[t] * mult), -1, 1)) if atr[t] > 0 else 0.0
        for t in range(len(src))
    ])
    _assert_allclose(mo.atr_distance_momentum(src, ema, atr, atr_mult=mult), expected, "atr_distance_momentum")


def test_neutralizing_price_matches_reference():
    o, h, low, c, _ = _ohlcv()
    expected = np.where(o > c, (o + low) / 2.0, (c + h) / 2.0)
    _assert_allclose(mo.neutralizing_price(o, h, low, c), expected, "neutralizing_price")


def test_direction_momentum_signal_matches_reference_and_booster_is_explicit():
    """z-вклад: EMA-часть + бустер; booster_weight=0 убирает двойной учёт входа."""
    fast, slow = 101.5, 100.0
    rel = (fast - slow) / slow
    ema_z_only = float(np.tanh(rel / 0.003) * 1.5)

    got_no_booster = mo.direction_momentum_signal(fast, slow, booster_weight=0.0)
    assert abs(got_no_booster - float(np.clip(ema_z_only, -2.0, 2.0))) < 1e-12

    got_with_booster = mo.direction_momentum_signal(fast, slow, ta_strength=80.0, ta_side="BULLISH")
    assert got_with_booster > got_no_booster, "бустер обязан усиливать вклад"

    got_opposite = mo.direction_momentum_signal(fast, slow, ta_strength=80.0, ta_side="BEARISH")
    assert got_opposite < got_no_booster, "противоположный бустер обязан ослаблять вклад"
    assert -2.0 <= got_opposite <= 2.0 and -2.0 <= got_with_booster <= 2.0, "диапазон z ∈ [-2, +2]"

    assert mo.direction_momentum_signal(float("nan"), slow) == 0.0
    assert mo.direction_momentum_signal(fast, 0.0) == 0.0


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
    print(f"--- analysis kernels: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
