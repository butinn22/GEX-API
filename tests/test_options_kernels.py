"""Проверка эквивалентности опционной математики (GEX-масштаб, уровни, стены).

Сверяет канон `gex.domain.options.*` с транскрипциями прежних реализаций
(`metrics._find_gamma_flip_cumulative`, `extended._zero_gamma_level`, `metrics._walls_by_*`,
`extended._wall`, `metrics.py:194-197` / `gexcone.py:588-590` / `extended.py:677`).

    python tests/test_options_kernels.py
    pytest tests/test_options_kernels.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex.domain.options import gex_scale as gs  # noqa: E402
from gex.domain.options import levels as lv  # noqa: E402


def _assert_allclose(actual, expected, ctx: str) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{ctx}: форма {actual.shape} != {expected.shape}"
    assert np.allclose(actual, expected, rtol=0, atol=1e-12), (
        f"{ctx}: max|Δ| = {np.max(np.abs(actual - expected))}"
    )


# ── Масштаб GEX ──────────────────────────────────────────────────────────────

def test_scale_methods_match_originals():
    """metrics/gexcone: Γ·pc·S²·0.01; extended: Γ·pc·100 (проверяем оба, как в оригинале)."""
    gamma = np.array([0.012, 0.031, 0.0045])
    spot, per_contract = 615.0, 100
    metrics_like = gamma * per_contract * (spot ** 2) * 0.01
    extended_like = gamma * per_contract * 100.0
    _assert_allclose(gs.per_contract_gex(gamma, spot, per_contract, method="dollar_gamma_1pct"),
                     metrics_like, "dollar_gamma_1pct")
    _assert_allclose(gs.per_contract_gex(gamma, spot, per_contract, method="contract_scale_100"),
                     extended_like, "contract_scale_100")


def test_scale_methods_coincide_only_at_spot_100():
    """Ключевая находка: масштабы совпадают лишь при S=100, иначе расходятся как S²·0.01/100."""
    gamma = np.array([0.02])
    for spot, expected_ratio in ((100.0, 1.0), (600.0, 36.0), (10.0, 0.01), (50.0, 0.25)):
        a = gs.per_contract_gex(gamma, spot, 100, method="dollar_gamma_1pct")
        b = gs.per_contract_gex(gamma, spot, 100, method="contract_scale_100")
        assert np.isclose(a[0] / b[0], expected_ratio), f"ratio при S={spot}"
        assert np.isclose(gs.scale_ratio(spot), expected_ratio), f"scale_ratio при S={spot}"


def test_gex_values_includes_sign_and_oi():
    gamma = np.array([0.01, 0.02])
    oi = np.array([1000.0, 500.0])
    sign = np.array([1.0, -1.0])
    got = gs.gex_values(gamma, oi, sign, 500.0, 100)
    expected = sign * gamma * 100 * (500.0 ** 2) * 0.01 * oi
    _assert_allclose(got, expected, "gex_values")


def test_scale_rejects_unknown_method():
    try:
        gs.per_contract_gex([0.01], 100.0, method="whatever")
    except ValueError as exc:
        assert "method must be one of" in str(exc)
    else:
        raise AssertionError("неизвестный метод должен отклоняться")


# ── Gamma flip / zero gamma ──────────────────────────────────────────────────

def reference_gamma_flip(strikes: np.ndarray, gex_net: np.ndarray, *, extended_variant: bool):
    """Транскрипция metrics._find_gamma_flip_cumulative / extended._zero_gamma_level."""
    order = np.argsort(strikes, kind="stable")
    s, v = strikes[order], gex_net[order]
    if s.size == 0:
        return None
    cum = np.cumsum(v)
    sign_change = np.diff(np.sign(cum)) != 0
    idx = np.flatnonzero(sign_change)
    if idx.size == 0:
        return None
    i = int(idx[0]) + 1
    if extended_variant and i == 0:
        return float(s[0])
    s0, s1 = float(s[i - 1]), float(s[i])
    g0, g1 = float(cum[i - 1]), float(cum[i])
    if np.isclose(g1 - g0, 0.0):
        return float(0.5 * (s0 + s1))
    return float(s0 - g0 * (s1 - s0) / (g1 - g0))


def test_gamma_flip_matches_both_originals():
    strikes = np.array([480.0, 490.0, 500.0, 510.0, 520.0, 530.0])
    # кумулятив обязан реально пересечь ноль: -1.0, -1.5, -1.8, +0.2, +0.7, +0.9 (×1e9)
    gex_net = np.array([-1.0e9, -0.5e9, -0.3e9, 2.0e9, 0.5e9, 0.2e9])
    got = lv.gamma_flip_cumulative(strikes, gex_net)
    assert got is not None, "при таком профиле переход знака обязан быть найден"
    assert abs(got - reference_gamma_flip(strikes, gex_net, extended_variant=False)) < 1e-9
    assert abs(got - reference_gamma_flip(strikes, gex_net, extended_variant=True)) < 1e-9
    assert 500.0 < got < 510.0, f"переход знака между 500 и 510, получено {got}"


def test_gamma_flip_returns_none_without_sign_change():
    strikes = np.array([100.0, 110.0, 120.0])
    only_positive = np.array([1.0, 2.0, 3.0])
    assert lv.gamma_flip_cumulative(strikes, only_positive) is None
    assert lv.gamma_flip_cumulative(np.array([]), np.array([])) is None


def test_gamma_flip_unsorted_input_is_handled():
    """Оригинал сортирует по страйку — канон обязан делать то же."""
    strikes = np.array([520.0, 480.0, 510.0, 490.0, 500.0])
    gex_net = np.array([1.9e9, -2.0e9, 0.8e9, -1.5e9, -0.4e9])
    sorted_strikes = np.array([480.0, 490.0, 500.0, 510.0, 520.0])
    sorted_gex = np.array([-2.0e9, -1.5e9, -0.4e9, 0.8e9, 1.9e9])
    assert lv.gamma_flip_cumulative(strikes, gex_net) == lv.gamma_flip_cumulative(sorted_strikes, sorted_gex)


# ── Стены ────────────────────────────────────────────────────────────────────

def _per_strike():
    strikes = np.array([480.0, 490.0, 500.0, 510.0, 520.0])
    gex_net = np.array([-2.0e9, 0.5e9, -0.3e9, 1.7e9, 0.2e9])
    oi_call = np.array([100.0, 900.0, 200.0, 1500.0, 300.0])
    oi_put = np.array([2200.0, 400.0, 1100.0, 300.0, 150.0])
    return strikes, gex_net, oi_call, oi_put


def test_primary_walls_match_metrics_reference():
    strikes, gex_net, _, _ = _per_strike()
    pos = gex_net > 0
    neg = gex_net < 0
    exp_call = strikes[pos][np.argmax(gex_net[pos])]
    exp_put = strikes[neg][np.argmin(gex_net[neg])]
    assert lv.primary_walls_by_gex(strikes, gex_net) == (float(exp_call), float(exp_put))


def test_walls_by_oi_match_metrics_reference():
    strikes, _, oi_call, oi_put = _per_strike()
    assert lv.walls_by_oi(strikes, oi_call, oi_put) == (
        float(strikes[int(np.argmax(oi_call))]),
        float(strikes[int(np.argmax(oi_put))]),
    )


def test_ranked_walls_match_metrics_reference():
    strikes, gex_net, _, _ = _per_strike()
    exp_call = [float(x) for x in strikes[gex_net > 0][np.argsort(-np.abs(gex_net[gex_net > 0]))]]
    exp_put = [float(x) for x in strikes[gex_net < 0][np.argsort(-np.abs(gex_net[gex_net < 0]))]]
    call, put = lv.ranked_walls(strikes, gex_net, top_n=3)
    assert call == exp_call and put == exp_put


def test_wall_with_strength_matches_extended_reference():
    strikes, gex_net, _, _ = _per_strike()
    total_abs = float(np.abs(gex_net).sum())
    for side, mask in (("call", gex_net > 0), ("put", gex_net < 0)):
        sub_strikes = strikes[mask]
        sub_gex = gex_net[mask]
        if side == "call":
            j = int(np.argmax(sub_gex))
        else:
            j = int(np.argmin(sub_gex))
        exp = (float(sub_strikes[j]), abs(float(sub_gex[j])) / total_abs)
        got = lv.wall_with_strength(strikes, gex_net, side=side)
        assert abs(got[0] - exp[0]) < 1e-9 and abs(got[1] - exp[1]) < 1e-12, f"{side}: {got} != {exp}"


def test_walls_empty_sides_return_nan():
    strikes = np.array([100.0, 110.0])
    all_positive = np.array([1.0, 2.0])
    call, put = lv.primary_walls_by_gex(strikes, all_positive)
    assert np.isnan(put) and not np.isnan(call)
    strike, strength = lv.wall_with_strength(strikes, all_positive, side="put")
    assert np.isnan(strike) and strength == 0.0
    assert lv.ranked_walls(np.array([]), np.array([])) == ([], [])


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
    print(f"--- options kernels: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
