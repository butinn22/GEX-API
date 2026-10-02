"""GEX EMF+ADL strategy adapter — wraps ``gex.strategy.EMAFilterTrendStrategy``.

This is the SAME strategy the auto_scanner / auto_signals use (the Pine→Python
port of the combined EMA/ADL strategy). It is driven through the trading
``Strategy`` port so it can be backtested by the event-driven engine.

Why ``prepare`` exists
----------------------
The underlying strategy is **batch-oriented**: ``calculate`` needs the full OHLCV
window and builds a ~110-column feature frame. The first version of this adapter
called ``calculate`` on every ``on_bar``, over the whole accumulated buffer. That
is O(n) work per bar — O(n²) per replay — and it measured **~405 s (6.8 min) for
1,500 bars**, with 51% of it in the VWAP state machine.

The fix is structural, not cosmetic: ``prepare(bars)`` computes the frame **once**
(O(n)) before the replay and ``on_bar`` becomes an O(1) row lookup. The engine
calls ``prepare`` for every strategy (default no-op), so nothing else changes.

Correctness rests on the frame being **causal**: every indicator in the pipeline
looks only backwards (``shift``/``rolling``/``ewm``, the VWAP pivot recursion, the
ADL ``cumsum``), so row *i* of the full-history frame is identical to row *i* of
the frame built on the prefix ``bars[:i+1]``. That is precisely the equality the
old per-bar recompute depended on, which is why the single precompute cannot
change a single signal. The streaming path below is kept for live trading, where
the future is genuinely unknown and ``prepare`` is never called.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from trading.domain import Bar, Price, Side, Signal, Tick
from trading.ports import Strategy

__all__ = ["GexEMFStrategy"]

MIN_BARS = 60


def _flag(row: pd.Series, column: str) -> bool:
    value = row.get(column)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


class GexEMFStrategy(Strategy):
    name = "gex_emf"

    def __init__(
        self,
        symbol: str,
        *,
        settings: dict | None = None,
        use_risk_exits: bool = False,
    ) -> None:
        from gex.strategy.settings import StrategySettings
        from gex.strategy.trading_algorithm import EMAFilterTrendStrategy

        self.symbol = symbol
        self._settings = StrategySettings.from_dict(settings) if settings else StrategySettings()
        self._gex = EMAFilterTrendStrategy(settings=self._settings)
        self._bars: list[Bar] = []
        self._ts: list[datetime] = []
        self._side = "flat"
        self._processed = 0
        # Batch mode, populated by ``prepare``: the whole-replay feature frame and
        # the timestamp of each of its rows. ``None`` means "streaming".
        self._features: pd.DataFrame | None = None
        self._cursor = 0
        #: Close of each prepared bar — the O(1) divergence probe in ``on_bar``.
        self._closes: np.ndarray | None = None
        #: Set when a prepared run was invalidated and we had to fall back.
        self.fallback_count = 0
        # ── take-profit / trailing-stop overlay ─────────────────────────
        # The strategy's ``combined_*_exit`` columns are pure indicator crossovers;
        # the TP/trailing rules live in ``_RiskMixin`` and are only reached by
        # ``EMAFilterTrendStrategy.evaluate``, which a column-driven backtest never
        # calls. So ``tp_percent``/``trailing_percent`` were silently inert. With
        # ``use_risk_exits`` the adapter applies those same rules itself.
        self._use_risk_exits = bool(use_risk_exits)
        self._entry_price: float | None = None
        self._entry_atr: float | None = None
        self._best: float | None = None  # highest high (long) / lowest low (short)

    def _reset_state(self) -> None:
        self._bars = []
        self._ts = []
        self._side = "flat"
        self._processed = 0
        self._features = None
        self._closes = None
        self._cursor = 0
        self._entry_price = None
        self._entry_atr = None
        self._best = None

    async def start(self) -> None:
        self._reset_state()

    # ── batch path ──────────────────────────────────────────────────────

    async def prepare(self, bars: Sequence[Bar]) -> None:
        """Precompute the feature frame once for the whole replay (O(n)).

        ``on_bar`` then only reads row *i*, which makes a replay O(n) overall.
        Falls back to streaming automatically if the bars that arrive do not
        line up with what was prepared here.
        """
        bars = list(bars)
        self._reset_state()
        if len(bars) < MIN_BARS:
            return
        self._features = self._gex.calculate(
            self._to_ohlcv(bars), include_decorative=False
        )
        self._ts = [b.timestamp for b in bars]
        self._closes = np.array([float(b.close) for b in bars], dtype=float)

    def _to_ohlcv(self, bars: Sequence[Bar]) -> pd.DataFrame:
        return pd.DataFrame({
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        })

    def _drain(self, feature_idx: int, frame: pd.DataFrame) -> list[Signal]:
        """Emit every not-yet-emitted signal from row ``_processed`` to ``feature_idx``.

        The position state machine is applied strictly in bar order, exactly as
        the per-bar loop used to; only *when* the frame was computed changed.
        """
        signals: list[Signal] = []
        while self._processed <= feature_idx:
            j = self._processed
            signals.extend(self._signal_at(frame, j, self._ts[j]))
            self._processed += 1
        return signals

    # ── signal mapping ──────────────────────────────────────────────────

    def _signal_at(self, features: pd.DataFrame, i: int, timestamp) -> list[Signal]:
        row = features.iloc[i]
        close = float(row["close"])
        price = Price(close)
        out: list[Signal] = []
        if self._side == "flat":
            if _flag(row, "long_entry_signal"):
                self._open("long", close, row)
                out.append(Signal(self.symbol, Side.BUY, self.name, "long_entry",
                                  strength=1.0, price=price, timestamp=timestamp))
            elif _flag(row, "short_entry_signal"):
                self._open("short", close, row)
                out.append(Signal(self.symbol, Side.SELL, self.name, "short_entry",
                                  strength=1.0, price=price, timestamp=timestamp))
        else:
            # Risk exits take precedence: a breached stop is a hard exit regardless
            # of what the indicators say. The engine fills both a stop and an
            # indicator exit at the *next* bar's open, so this is the same
            # convention as every other signal — the level is judged on the close.
            risk = self._risk_exit(row, close, price, timestamp) if self._use_risk_exits else None
            if risk is not None:
                out.append(risk)
                self._close_position()
                return out
            if self._side == "long":
                if _flag(row, "long_exit_signal"):
                    self._close_position()
                    out.append(Signal(self.symbol, Side.SELL, self.name, "long_exit",
                                      strength=1.0, price=price, timestamp=timestamp))
            elif self._side == "short":
                if _flag(row, "short_exit_signal"):
                    self._close_position()
                    out.append(Signal(self.symbol, Side.BUY, self.name, "short_exit",
                                      strength=1.0, price=price, timestamp=timestamp))

        if self._side != "flat":
            # Track the extremes *after* the exit checks, so the stop level tested
            # against this bar's close never depends on this bar's own high/low.
            high, low = float(row["high"]), float(row["low"])
            if self._side == "long":
                self._best = high if self._best is None else max(self._best, high)
            else:
                self._best = low if self._best is None else min(self._best, low)
        return out

    # ── take-profit / trailing stop ─────────────────────────────────────

    def _open(self, side: str, close: float, row: pd.Series) -> None:
        """Record the entry so the risk levels can be computed for this position."""
        self._side = side
        self._entry_price = close
        self._best = close
        atr = row.get("atr")
        try:
            value = float(atr) if atr is not None and not pd.isna(atr) else None
        except (TypeError, ValueError):
            value = None
        self._entry_atr = value if value and value > 0 else None

    def _close_position(self) -> None:
        self._side = "flat"
        self._entry_price = None
        self._entry_atr = None
        self._best = None

    def _risk_exit(self, row: pd.Series, close: float, price: Price, timestamp) -> Signal | None:
        """Take-profit / trailing-stop exit for the open position, if breached.

        The levels come from the project's own ``_RiskMixin``
        (``take_profit_price`` / ``trailing_stop_price``) so the overlay adds no new
        risk maths: ``use_atr_stops`` selects ATR-scaled levels
        (``atr_tp_mult``/``atr_sl_mult``), otherwise the ``tp_percent`` /
        ``trailing_percent`` percentages apply. The ATR measured at entry fixes the
        trade's risk budget, so the level does not jump when volatility moves.
        """
        entry = self._entry_price
        if entry is None:  # pragma: no cover - defensive
            return None
        settings = self._settings
        atr = self._entry_atr  # ``None`` → _RiskMixin falls back to percentages
        long = self._side == "long"
        direction = "long" if long else "short"

        if long:
            stop = self._gex.trailing_stop_price(
                entry, direction, highest_price=self._best, atr_value=atr
            )
            if settings.use_trailing and stop is not None and close <= stop:
                return Signal(self.symbol, Side.SELL, self.name, "trailing_stop",
                              strength=1.0, price=price, timestamp=timestamp)
            tp = self._gex.take_profit_price(entry, direction, atr)
            if settings.use_take_profit and close >= tp:
                return Signal(self.symbol, Side.SELL, self.name, "take_profit",
                              strength=1.0, price=price, timestamp=timestamp)
        else:
            stop = self._gex.trailing_stop_price(
                entry, direction, lowest_price=self._best, atr_value=atr
            )
            if settings.use_trailing and stop is not None and close >= stop:
                return Signal(self.symbol, Side.BUY, self.name, "trailing_stop",
                              strength=1.0, price=price, timestamp=timestamp)
            tp = self._gex.take_profit_price(entry, direction, atr)
            if settings.use_take_profit and close <= tp:
                return Signal(self.symbol, Side.BUY, self.name, "take_profit",
                              strength=1.0, price=price, timestamp=timestamp)
        return None

    # ── streaming path ──────────────────────────────────────────────────

    async def on_bar(self, bar: Bar) -> list[Signal]:
        if self._features is not None and self._cursor < len(self._ts):
            if self._matches_prepared(self._cursor, bar):
                idx = self._cursor
                self._cursor += 1
                if idx + 1 < MIN_BARS:  # same warm-up gate as the streaming path
                    return []
                return self._drain(idx, self._features)
            # The bar stream diverged from what prepare() saw (replay resumed on a
            # different series). Drop the prepared frame and stream instead of
            # emitting signals indexed against the wrong bars.
            self.fallback_count += 1
            self._features = None
            self._closes = None
            self._cursor = 0

        self._bars.append(bar)
        self._ts.append(bar.timestamp)
        if len(self._bars) < MIN_BARS:
            return []
        features = self._gex.calculate(
            self._to_ohlcv(self._bars), include_decorative=False
        )
        return self._drain(len(self._bars) - 1, features)

    def _matches_prepared(self, idx: int, bar: Bar) -> bool:
        """Does ``bar`` correspond to prepared row ``idx``?

        Checks the timestamp **and** the close. A timestamp alone is not enough:
        a corrected/revised series has the same stamps but different prices, and
        using the prepared rows would then compute signals from stale indicator
        values. O(1), so it costs nothing per bar.
        """
        if self._ts[idx] != bar.timestamp:
            return False
        closes = self._closes
        if closes is None:  # pragma: no cover - defensive
            return False
        expected = closes[idx]
        actual = float(bar.close)
        return abs(expected - actual) <= 1e-9 * max(1.0, abs(expected))

    async def on_tick(self, tick: Tick) -> list[Signal]:
        return []

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        bars = list(bars)
        if len(bars) < MIN_BARS:
            return []
        self._side = "flat"
        self._processed = 0
        self._ts = [b.timestamp for b in bars]
        self._features = None
        self._closes = None
        self._entry_price = None
        self._entry_atr = None
        self._best = None
        features = self._gex.calculate(
            self._to_ohlcv(bars), include_decorative=False
        )
        signals: list[Signal] = []
        for i in range(len(bars)):
            signals.extend(self._signal_at(features, i, self._ts[i]))
        return signals
