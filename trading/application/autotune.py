"""Automated pre-live tuning: parameter optimization + volatility-adaptive risk.

Pipeline (runs **before** live trading):

1. ``compute_volatility`` measures the asset — ATR(14) and average daily
   range — so stops scale with what the instrument actually does, not a
   fixed percentage.
2. ``optimize_strategy`` (existing train/validation grid search) picks the
   strategy parameters out of sample.
3. ``auto_stop_take`` derives Stop Loss / Take Profit from the selected risk
   profile in ATR multiples:

   * **Low** — tight stops (1.0× ATR SL / 1.5× ATR TP);
   * **Medium** — moderate stops (2.0× / 3.0×);
   * **High** — breakout/trend-following (3.0× / 5.0×) and entries require
     **EMA50 confirmation** (longs above, shorts below).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from trading.application.cancellation import CancelToken
from trading.application.indicators import atr, ema
from trading.domain import Bar, Side

from .backtest.engine import BacktestConfig
from .backtest.optimize import OptimizeResult, optimize_strategy

__all__ = [
    "RiskProfile",
    "ProfileRisk",
    "PROFILE_RISK",
    "VolatilityStats",
    "RiskTargets",
    "AutoTuneResult",
    "compute_volatility",
    "auto_stop_take",
    "ema50_confirmed",
    "autotune",
]


class RiskProfile(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ProfileRisk:
    stop_atr: float  # stop distance in ATR multiples
    tp_atr: float  # take-profit distance in ATR multiples
    ema50_confirm: bool  # require EMA50 trend confirmation for entries


PROFILE_RISK: dict[RiskProfile, ProfileRisk] = {
    RiskProfile.LOW: ProfileRisk(stop_atr=1.0, tp_atr=1.5, ema50_confirm=False),
    RiskProfile.MEDIUM: ProfileRisk(stop_atr=2.0, tp_atr=3.0, ema50_confirm=False),
    RiskProfile.HIGH: ProfileRisk(stop_atr=3.0, tp_atr=5.0, ema50_confirm=True),
}


@dataclass(frozen=True)
class VolatilityStats:
    atr14: float
    adr: float  # average daily range: mean(high - low) over the ATR window
    atr_pct: float  # atr14 as a fraction of the last close


@dataclass(frozen=True)
class RiskTargets:
    entry: float
    stop_loss: float
    take_profit: float
    side: Side
    profile: RiskProfile
    atr: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry": round(self.entry, 8),
            "stop_loss": round(self.stop_loss, 8),
            "take_profit": round(self.take_profit, 8),
            "side": self.side.value,
            "profile": self.profile.value,
            "atr": round(self.atr, 8),
        }


def compute_volatility(bars: Sequence[Bar], period: int = 14) -> VolatilityStats:
    """ATR(period) + average daily range of the asset's recent history."""
    bars = sorted(bars, key=lambda b: b.timestamp)
    if len(bars) < period:
        raise ValueError(f"need >= {period} bars for volatility, got {len(bars)}")
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    atr_val = float(atr(highs, lows, closes, period)[-1])
    adr = float(np.mean([h - l for h, l in zip(highs[-period:], lows[-period:])]))
    last_close = closes[-1]
    return VolatilityStats(
        atr14=atr_val,
        adr=adr,
        atr_pct=atr_val / last_close if last_close > 0 else 0.0,
    )


def auto_stop_take(
    entry: float, side: Side, *, atr: float, profile: RiskProfile
) -> RiskTargets:
    """SL/TP for an entry, scaled to the asset's ATR by risk profile."""
    risk = PROFILE_RISK[profile]
    if side is Side.BUY:
        return RiskTargets(
            entry=entry, stop_loss=entry - risk.stop_atr * atr,
            take_profit=entry + risk.tp_atr * atr, side=side, profile=profile, atr=atr,
        )
    return RiskTargets(
        entry=entry, stop_loss=entry + risk.stop_atr * atr,
        take_profit=entry - risk.tp_atr * atr, side=side, profile=profile, atr=atr,
    )


def ema50_confirmed(bars: Sequence[Bar], side: Side, period: int = 50) -> bool:
    """Trend filter for the High profile: longs above / shorts below EMA50."""
    closes = np.array([b.close for b in bars], dtype=float)
    ema_val = ema(closes, period)[-1]
    if not np.isfinite(ema_val):
        return False
    last = closes[-1]
    return bool(last > ema_val) if side is Side.BUY else bool(last < ema_val)


@dataclass
class AutoTuneResult:
    symbol: str
    strategy: str
    profile: RiskProfile
    volatility: VolatilityStats
    best_params: dict[str, Any]
    optimize: OptimizeResult
    last_close: float
    ema50_confirmed_long: bool
    ema50_confirmed_short: bool

    def targets_for(self, side: Side, entry: float | None = None) -> RiskTargets:
        return auto_stop_take(
            entry if entry is not None else self.last_close,
            side, atr=self.volatility.atr14, profile=self.profile,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "risk_profile": self.profile.value,
            "volatility": {
                "atr14": round(self.volatility.atr14, 8),
                "adr": round(self.volatility.adr, 8),
                "atr_pct": round(self.volatility.atr_pct, 6),
            },
            "best_params": self.best_params,
            "best": self.optimize.best,
            "leaderboard": [c.as_dict() for c in self.optimize.leaderboard],
            "ema50_confirmed_long": self.ema50_confirmed_long,
            "ema50_confirmed_short": self.ema50_confirmed_short,
            "long_targets": self.targets_for(Side.BUY).as_dict(),
            "short_targets": self.targets_for(Side.SELL).as_dict(),
        }


def autotune(
    symbol: str,
    strategy: str,
    bars: Sequence[Bar],
    profile: RiskProfile,
    *,
    base_params: Mapping[str, Any] | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    cfg: BacktestConfig | None = None,
    cancel: CancelToken | None = None,
) -> AutoTuneResult:
    """Optimize a strategy's parameters on the asset's history and derive
    volatility-adaptive risk targets. Synchronous — offload with
    ``asyncio.to_thread`` at the API layer."""
    bars = sorted(bars, key=lambda b: b.timestamp)
    volatility = compute_volatility(bars)
    opt = optimize_strategy(
        strategy, symbol, bars,
        base_params=base_params, grid=grid, cfg=cfg, cancel=cancel,
    )
    return AutoTuneResult(
        symbol=symbol,
        strategy=strategy,
        profile=profile,
        volatility=volatility,
        best_params=opt.best_params,
        optimize=opt,
        last_close=bars[-1].close,
        ema50_confirmed_long=ema50_confirmed(bars, Side.BUY),
        ema50_confirmed_short=ema50_confirmed(bars, Side.SELL),
    )
