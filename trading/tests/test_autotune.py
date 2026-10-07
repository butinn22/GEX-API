"""Auto-tuning pipeline + volatility-adaptive risk profiles."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.application.autotune import (
    PROFILE_RISK,
    RiskProfile,
    auto_stop_take,
    autotune,
    compute_volatility,
    ema50_confirmed,
)
from trading.domain import Bar, Side

T0 = datetime(2023, 1, 1, tzinfo=timezone.utc)


def make_bars(closes, range_pct: float = 0.01) -> list[Bar]:
    bars = []
    for i, c in enumerate(closes):
        half = c * range_pct / 2
        bars.append(Bar(
            timestamp=T0 + timedelta(days=i),
            open=c, high=c + half, low=c - half, close=c, volume=1000.0,
        ))
    return bars


class TestComputeVolatility:
    def test_atr_and_adr_positive(self):
        bars = make_bars(np.linspace(100, 150, 60))
        vol = compute_volatility(bars)
        assert vol.atr14 > 0
        assert vol.adr > 0
        assert vol.atr_pct == pytest.approx(vol.atr14 / bars[-1].close)

    def test_adr_is_mean_of_last_period_ranges(self):
        closes = np.full(30, 100.0)
        bars = make_bars(closes, range_pct=0.02)  # every range = 2.0
        vol = compute_volatility(bars, period=14)
        assert vol.adr == pytest.approx(2.0)

    def test_adapts_to_asset_volatility(self):
        calm = compute_volatility(make_bars(np.linspace(100, 110, 60), range_pct=0.005))
        wild = compute_volatility(make_bars(np.linspace(100, 110, 60), range_pct=0.05))
        assert wild.atr14 > calm.atr14 * 5

    def test_too_few_bars_raises(self):
        with pytest.raises(ValueError):
            compute_volatility(make_bars([100.0] * 5))


class TestAutoStopTake:
    def test_low_profile_tight_stops(self):
        t = auto_stop_take(100.0, Side.BUY, atr=2.0, profile=RiskProfile.LOW)
        assert t.stop_loss == pytest.approx(98.0)   # 1.0 x ATR
        assert t.take_profit == pytest.approx(103.0)  # 1.5 x ATR

    def test_medium_profile_moderate_stops(self):
        t = auto_stop_take(100.0, Side.BUY, atr=2.0, profile=RiskProfile.MEDIUM)
        assert t.stop_loss == pytest.approx(96.0)   # 2.0 x ATR
        assert t.take_profit == pytest.approx(106.0)  # 3.0 x ATR

    def test_high_profile_wide_breakout_stops(self):
        t = auto_stop_take(100.0, Side.BUY, atr=2.0, profile=RiskProfile.HIGH)
        assert t.stop_loss == pytest.approx(94.0)   # 3.0 x ATR
        assert t.take_profit == pytest.approx(110.0)  # 5.0 x ATR

    def test_short_side_mirrored(self):
        t = auto_stop_take(100.0, Side.SELL, atr=2.0, profile=RiskProfile.MEDIUM)
        assert t.stop_loss == pytest.approx(104.0)
        assert t.take_profit == pytest.approx(94.0)

    def test_profiles_are_ordered_by_width(self):
        assert (PROFILE_RISK[RiskProfile.LOW].stop_atr
                < PROFILE_RISK[RiskProfile.MEDIUM].stop_atr
                < PROFILE_RISK[RiskProfile.HIGH].stop_atr)

    def test_only_high_profile_requires_ema50(self):
        assert PROFILE_RISK[RiskProfile.HIGH].ema50_confirm is True
        assert PROFILE_RISK[RiskProfile.LOW].ema50_confirm is False
        assert PROFILE_RISK[RiskProfile.MEDIUM].ema50_confirm is False


class TestEma50Confirmed:
    def test_uptrend_confirms_long_only(self):
        bars = make_bars(np.linspace(100, 200, 120))
        assert ema50_confirmed(bars, Side.BUY) is True
        assert ema50_confirmed(bars, Side.SELL) is False

    def test_downtrend_confirms_short_only(self):
        bars = make_bars(np.linspace(200, 100, 120))
        assert ema50_confirmed(bars, Side.SELL) is True
        assert ema50_confirmed(bars, Side.BUY) is False

    def test_insufficient_data_is_not_confirmed(self):
        bars = make_bars(np.linspace(100, 110, 20))
        assert ema50_confirmed(bars, Side.BUY) is False


class TestAutotune:
    def test_end_to_end_picks_params_and_targets(self):
        rng = np.random.default_rng(7)
        trend = np.linspace(100, 160, 300)
        closes = trend + rng.normal(0, 1.0, 300)
        bars = make_bars(closes)
        result = autotune(
            "SYNTH", "sma_crossover", bars, RiskProfile.MEDIUM,
            grid={"fast": [3, 5], "slow": [10, 20]},
        )
        assert result.symbol == "SYNTH"
        assert result.profile is RiskProfile.MEDIUM
        assert result.best_params["fast"] in (3, 5)
        assert result.best_params["slow"] in (10, 20)
        assert result.volatility.atr14 > 0
        long_t = result.targets_for(Side.BUY)
        assert long_t.stop_loss < long_t.entry < long_t.take_profit
        short_t = result.targets_for(Side.SELL)
        assert short_t.take_profit < short_t.entry < short_t.stop_loss

    def test_high_profile_reports_ema50_confirmation(self):
        bars = make_bars(np.linspace(100, 200, 300))
        result = autotune(
            "SYNTH", "sma_crossover", bars, RiskProfile.HIGH,
            grid={"fast": [3], "slow": [10]},
        )
        assert result.ema50_confirmed_long is True
        assert result.ema50_confirmed_short is False

    def test_serializable(self):
        bars = make_bars(np.linspace(100, 120, 300))
        result = autotune(
            "SYNTH", "sma_crossover", bars, RiskProfile.LOW,
            grid={"fast": [3], "slow": [10]},
        )
        d = result.as_dict()
        assert d["risk_profile"] == "low"
        assert "volatility" in d and "best_params" in d
        assert d["long_targets"]["stop_loss"] < d["long_targets"]["entry"]


class TestAutotuneEndpoint:
    async def test_post_autotune(self):
        import httpx

        from trading.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            login = await c.post(
                "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
            )
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
            r = await c.post("/api/v1/backtest/autotune", json={
                "strategy": "sma_crossover",
                "symbol": "SYNTH",
                "risk_profile": "high",
                "grid": {"fast": [3, 5], "slow": [10]},
                "limit": 300,
            }, headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["risk_profile"] == "high"
        assert body["symbol"] == "SYNTH"
        assert body["volatility"]["atr14"] > 0
        assert body["long_targets"]["profile"] == "high"
        assert "ema50_confirmed_long" in body
