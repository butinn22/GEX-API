"""Tests for :mod:`gex.macd_trend` and the MACD-trend FastAPI endpoint.

Run with::

    python -m pytest tests/test_macd_trend.py -q

These tests are hermetic (no network) except the two explicitly network-marked
endpoint tests, which are skipped automatically when offline.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from gex.domain.macd_trend import (
    AnalyzerConfig,
    BarResult,
    MacdTrendAnalyzer,
    Quadrant,
    analyze_history,
    compute_macd,
)


# ====================================================================== #
#  Fixtures
# ====================================================================== #
@pytest.fixture
def rising_avg() -> tuple[pd.Series, pd.Series]:
    """Strictly rising AVG line above zero → bullish strengthening."""
    macd = pd.Series(np.linspace(0.1, 5.0, 80))
    signal = macd.copy()  # avg = macd, strictly rising
    return macd, signal


@pytest.fixture
def falling_avg_below() -> tuple[pd.Series, pd.Series]:
    """Strictly falling AVG line below zero → bearish strengthening."""
    macd = pd.Series(np.linspace(-0.1, -5.0, 80))
    signal = macd.copy()
    return macd, signal


def _seed_analyzer(macd: pd.Series, signal: pd.Series, **cfg) -> MacdTrendAnalyzer:
    """Feed a full series through a streaming analyzer and return it."""
    an = MacdTrendAnalyzer(AnalyzerConfig(**cfg))
    for m, s in zip(macd, signal):
        an.update(float(m), float(s))
    return an


# ====================================================================== #
#  1. Synthetic rising/falling → angle sign + quadrant
# ====================================================================== #
def test_rising_avg_positive_angle_and_bullish_quadrant(rising_avg):
    """Linearly rising AVG → angle_degrees > 0 and BULLISH_STRENGTHENING."""
    macd, signal = rising_avg
    df = analyze_history(macd, signal, AnalyzerConfig(M=10, H=50))

    # First N bars the angle is undefined; take the last finite value.
    angles = df["angle_degrees"].dropna().astype(float)
    assert len(angles) > 0
    assert float(angles.iloc[-1]) > 0.0

    quads = df["quadrant"].dropna()
    assert quads.iloc[-1] == Quadrant.BULLISH_STRENGTHENING.value


def test_falling_avg_negative_angle_and_bearish_quadrant(falling_avg_below):
    """Linearly falling AVG below zero → angle < 0 and BEARISH_STRENGTHENING."""
    macd, signal = falling_avg_below
    df = analyze_history(macd, signal, AnalyzerConfig(M=10, H=50))

    angles = df["angle_degrees"].dropna().astype(float)
    assert len(angles) > 0
    assert float(angles.iloc[-1]) < 0.0

    quads = df["quadrant"].dropna()
    assert quads.iloc[-1] == Quadrant.BEARISH_STRENGTHENING.value


def test_flat_avg_is_flat_quadrant():
    """Constant AVG line → FLAT quadrant (angle ~0).

    With a perfectly constant series the normalization factor (std) is ~0, so
    the angle is undefined (None) by design — we accept either FLAT quadrants
    where the angle could be resolved, or an all-None quadrant column.
    """
    const = pd.Series([1.0] * 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = analyze_history(const, const, AnalyzerConfig(M=10, H=30))
    quads = df["quadrant"].dropna().tolist()
    # If any quadrant was resolved, it must be FLAT.
    assert all(q == Quadrant.FLAT.value for q in quads)


# ====================================================================== #
#  2. Streaming vs batch identical
# ====================================================================== #
def test_streaming_and_batch_identical():
    """Both usage modes must produce identical results on the same data."""
    rng = np.random.default_rng(123)
    n = 300
    macd = pd.Series(np.sin(np.linspace(0, 18, n)) * 2 + rng.normal(0, 0.4, n))
    signal = macd.rolling(9, min_periods=1).mean()

    cfg = AnalyzerConfig()
    df_batch = analyze_history(macd, signal, cfg)

    an = MacdTrendAnalyzer(cfg)
    stream_rows = [an.update(float(m), float(s)).as_dict() for m, s in zip(macd, signal)]
    df_stream = pd.DataFrame.from_records(stream_rows)

    cols = [
        "angle_degrees", "strength_instant", "strength_percentile",
        "composite_score", "conviction_multiplier", "final_trend_score",
        "norm_slope", "avg_value",
    ]
    for col in cols:
        b = df_batch[col].astype(float).values
        s = df_stream[col].astype(float).values
        mask = ~(np.isnan(b) | np.isnan(s))
        if not mask.any():
            continue
        diff = float(np.max(np.abs(b[mask] - s[mask])))
        assert diff < 1e-9, f"streaming vs batch differ on {col}: {diff}"

    # Quadrants identical too.
    assert list(df_batch["quadrant"]) == list(df_stream["quadrant"])


# ====================================================================== #
#  3. MWU: an always-right expert's weight grows
# ====================================================================== #
def test_mwu_weight_grows_for_always_right_expert():
    """When one expert always predicts the realized direction, its MWU weight
    should grow relative to the others over enough bars.

    Construction: feed a strictly rising AVG. Expert A (position sign) and the
    realized outcome ``sign(avg[t]-avg[t-1])`` are both persistently +1, so
    Expert A is never penalised. We verify its weight ends up strictly above
    the uniform 1/3 and above the (penalised) Expert B/C weights.
    """
    macd = pd.Series(np.linspace(0.5, 8.0, 120))
    signal = macd * 0.95  # avg slightly positive and rising
    an = _seed_analyzer(macd, signal, M=10, H=50, eta=0.3)

    w = an._weights  # noqa: SLF001 — internal, intentional for the test
    assert w[0] > 1.0 / 3.0, f"Expert A weight {w[0]} did not grow above uniform"
    assert w[0] == max(w), "Expert A should be the heaviest expert"


# ====================================================================== #
#  4. Markov transition rows sum to 1
# ====================================================================== #
def test_markov_rows_sum_to_one():
    """Each populated Markov row must sum to 1 (within tolerance)."""
    rng = np.random.default_rng(7)
    n = 400
    # Alternating regimes to populate several rows.
    macd = pd.Series(np.where(np.arange(n) % 40 < 20,
                              rng.normal(2, 0.3, n), rng.normal(-2, 0.3, n)))
    signal = macd.rolling(9, min_periods=1).mean()
    an = _seed_analyzer(macd, signal, M=10, H=50, markov_min_obs=5)

    counts = an._trans_counts  # noqa: SLF001
    row_sums = counts.sum(axis=1)
    nz = row_sums > 0
    # Each non-empty row, normalised, sums to exactly 1.
    for i in np.where(nz)[0]:
        probs = counts[i] / row_sums[i]
        assert abs(probs.sum() - 1.0) < 1e-9, f"row {i} sums to {probs.sum()}"

    # next-state probabilities for the current state also sum to 1.
    last_result = BarResult  # placeholder for readability
    # Re-fetch the last emitted probs via a fresh update-free probe:
    an2 = _seed_analyzer(macd, signal, M=10, H=50, markov_min_obs=5)
    probs = an2._markov_next_probs(an2._prev_state, 5)  # noqa: SLF001
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert len(probs) == 5
    del last_result


def test_markov_uniform_until_min_obs():
    """Below ``markov_min_obs`` the next-state distribution is uniform."""
    macd = pd.Series(np.linspace(1.0, 2.0, 30))
    signal = macd.copy()
    an = MacdTrendAnalyzer(AnalyzerConfig(markov_min_obs=100, M=5, H=20))
    for m, s in zip(macd, signal):
        an.update(float(m), float(s))
    probs = an._markov_next_probs(an._prev_state, 100)  # noqa: SLF001
    for v in probs.values():
        assert abs(v - 0.2) < 1e-9


# ====================================================================== #
#  5. Edge cases never raise
# ====================================================================== #
def test_too_few_bars_does_not_raise():
    """Very short input → no exception, results gracefully None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = analyze_history([0.1, 0.2, 0.3], [0.0, 0.1, 0.2], AnalyzerConfig())
    assert len(df) == 3
    assert df["angle_degrees"].isna().all()  # need N bars first


def test_nan_inputs_do_not_raise():
    """NaN in macd/signal → degenerate BarResult, no exception."""
    an = MacdTrendAnalyzer(AnalyzerConfig(M=5, H=20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = an.update(float("nan"), 0.1)
    assert res.avg_value is None
    assert res.final_trend_score is None
    # Analyzer stays usable afterwards.
    res2 = an.update(0.5, 0.4)
    assert res2.avg_value is not None


def test_zero_volatility_does_not_raise():
    """Constant AVG → ~0 normalization → angle None, no crash."""
    const = pd.Series([1.0] * 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = analyze_history(const, const, AnalyzerConfig(M=10, H=30))
    assert len(df) == 50
    # After N bars the angle is defined but degenerate → FLAT or None.
    assert df["quadrant"].iloc[-1] in (Quadrant.FLAT.value, None, float("nan"))


def test_length_mismatch_raises():
    """Unequal macd/signal lengths → ValueError (caller error, not edge case)."""
    with pytest.raises(ValueError):
        analyze_history([1.0, 2.0, 3.0], [1.0, 2.0], AnalyzerConfig())


# ====================================================================== #
#  6. compute_macd helper + strength bounds
# ====================================================================== #
def test_compute_macd_shapes_and_types():
    close = pd.Series(np.linspace(100, 120, 100) + np.random.default_rng(0).normal(0, 0.5, 100))
    macd, signal = compute_macd(close)
    assert len(macd) == len(close)
    assert len(signal) == len(close)
    assert macd.dtype == float


def test_strength_instant_bounded_in_0_1():
    rng = np.random.default_rng(1)
    macd = pd.Series(rng.normal(0, 2, 200))
    signal = macd.rolling(9, min_periods=1).mean()
    df = analyze_history(macd, signal, AnalyzerConfig())
    s = df["strength_instant"].dropna().astype(float)
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_final_trend_score_bounded_minus1_plus1():
    rng = np.random.default_rng(2)
    macd = pd.Series(rng.normal(0, 3, 300))
    signal = macd.rolling(9, min_periods=1).mean()
    df = analyze_history(macd, signal, AnalyzerConfig())
    f = df["final_trend_score"].dropna().astype(float)
    assert (f >= -1.0).all() and (f <= 1.0).all()


# ====================================================================== #
#  7. ATR normalization + linear_regression line method
# ====================================================================== #
def test_atr_normalization_path():
    """ATR mode with OHLC supplied produces finite angles (no exception)."""
    rng = np.random.default_rng(5)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    high = close + rng.uniform(0.1, 1.5, 100)
    low = close - rng.uniform(0.1, 1.5, 100)
    macd, signal = compute_macd(close)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = analyze_history(
            macd, signal, AnalyzerConfig(normalization_method="atr"),
            close=close, high=high, low=low,
        )
    assert df["angle_degrees"].dropna().astype(float).size > 0


def test_atr_falls_back_when_no_ohlc(caplog=None):
    """ATR mode without OHLC falls back to rolling_std (warning emitted)."""
    rng = np.random.default_rng(6)
    macd = pd.Series(rng.normal(0, 2, 100))
    signal = macd.rolling(9, min_periods=1).mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = analyze_history(macd, signal, AnalyzerConfig(normalization_method="atr"))
    assert len(df) == 100


def test_linear_regression_line_method():
    macd = pd.Series(np.linspace(0.1, 4.0, 60))
    signal = macd.copy()
    df = analyze_history(
        macd, signal, AnalyzerConfig(line_method="linear_regression", M=10, H=30)
    )
    angles = df["angle_degrees"].dropna().astype(float)
    assert len(angles) > 0
    assert float(angles.iloc[-1]) > 0.0  # rising → positive


# ====================================================================== #
#  8. Kelly fraction (research)
# ====================================================================== #
def test_kelly_fraction_when_enabled():
    rng = np.random.default_rng(8)
    macd = pd.Series(rng.normal(1, 2, 200))
    signal = macd.rolling(9, min_periods=1).mean()
    df = analyze_history(macd, signal, AnalyzerConfig(kelly_enabled=True, kelly_b=2.0))
    k = df["kelly_fraction"].dropna().astype(float)
    assert len(k) > 0
    # Kelly fraction must lie within [-1, 1] for b>=1.
    assert (k >= -1.0).all() and (k <= 1.0).all()


def test_kelly_none_when_disabled():
    macd = pd.Series(np.linspace(0.1, 3.0, 50))
    signal = macd.copy()
    df = analyze_history(macd, signal, AnalyzerConfig(kelly_enabled=False))
    assert df["kelly_fraction"].isna().all()


# ====================================================================== #
#  9. Zero-crossing detection
# ====================================================================== #
def test_zero_cross_detection():
    """AVG line crossing zero is flagged bull/bear at the right bars."""
    # Build a series that goes + -> - -> +.
    vals = np.concatenate([
        np.linspace(0.5, 2.0, 40),
        np.linspace(2.0, -2.0, 40),
        np.linspace(-2.0, 1.0, 40),
    ])
    macd = pd.Series(vals)
    signal = macd.copy()
    df = analyze_history(macd, signal, AnalyzerConfig(N=5, M=10, H=50))
    crosses = df["zero_cross"].dropna().tolist()
    assert "bear_cross" in crosses  # + -> -
    assert "bull_cross" in crosses  # - -> +


# ====================================================================== #
#  10. Endpoint tests (network — skipped when offline)
# ====================================================================== #
def _has_network() -> bool:
    import socket
    try:
        socket.create_connection(("api.bybit.com", 443), timeout=3)
        return True
    except OSError:
        return False


def _auth_headers(client) -> dict:
    """Создать таблицы, засеять Master Admin и вернуть Bearer-заголовки."""
    from gex.auth.config import settings as app_settings
    from gex.auth.router import seed_master_admin
    from gex.adapters.persistence.database import SessionLocal, recreate_tables

    recreate_tables()
    db = SessionLocal()
    try:
        seed_master_admin(db)
    finally:
        db.close()
    r = client.post(
        "/auth/login",
        json={"email": app_settings.MASTER_EMAIL, "password": app_settings.MASTER_PASSWORD},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.mark.skipif(not _has_network(), reason="no network")
def test_endpoint_stock_returns_200():
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    headers = _auth_headers(client)
    r = client.get("/macd/trend/AAPL", params={"timeframe": "1d", "H": 100, "M": 15}, headers=headers)
    assert r.status_code == 200
    d = r.json()
    assert d["symbol"] == "AAPL"
    assert d["asset_type"] == "stock"
    assert d["consensus_trend"] in ("BULLISH", "BEARISH", "RANGE")
    assert len(d["timeframes"]) >= 1
    assert d["summarize"] is not None


@pytest.mark.skipif(not _has_network(), reason="no network")
def test_endpoint_validation_422():
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    headers = _auth_headers(client)
    r = client.get("/macd/trend/AAPL", params={"normalization_method": "bogus"}, headers=headers)
    assert r.status_code == 422
