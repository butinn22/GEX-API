"""Trend direction & strength from MACD (MACD line vs Signal line).

This module is a self-contained, dependency-light (``numpy``/``pandas``/stdlib
only) engine that turns two already-computed series — the MACD line and its
Signal line — into a rich, *causal* (no look-ahead) trend estimate.

The design layers three independent ideas on top of plain MACD geometry:

Part 1 — Geometry (the core)
    * An ``AVG`` line = mean(MACD, Signal).
    * A secant (or OLS regression) slope over a window ``N``.
    * A *normalized* slope (divided by a rolling volatility / ATR) → an
      interpretable ``angle_degrees`` in ``(-90°, 90°)`` that is comparable
      across instruments and timeframes.
    * A small 5-state ``Quadrant`` enum describing the regime.

Part 2 — Probability
    * Rolling *percentile* of the current angle vs the instrument's own history
      (``H`` bars) — a statistically grounded "how strong is this trend?".
    * An empirical first-order Markov chain over the 5 quadrants → next-state
      probabilities.

Part 3 — Game theory
    * Multiplicative Weights Update (Hedge) over 3 "expert" voters playing a
      repeated game vs the market → ``composite_score``.
    * A *conviction multiplier* from the MACD–Signal spread (market
      "equilibrium / disequilibrium").
    * ``final_trend_score`` = ``composite_score`` scaled by conviction.

Part 4 — Kelly (optional, off by default)
    * A research-only position-sizing fraction from the Markov same-camp
      probability. **Not financial advice.**

Two equivalent usage modes are provided and are guaranteed to produce
identical results on the same data:

    * Streaming — :class:`MacdTrendAnalyzer` with ``.update()``.
    * Batch — :func:`analyze_history` returning a ``pd.DataFrame``.

All public code is fully typed and documented (Google/NumPy docstrings).
Edge cases (too few bars, ``NaN`` inputs, near-zero volatility) never raise —
they emit :func:`warnings.warn` and return ``None``/``NaN`` instead.
"""
from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections.abc import Sequence

import numpy as np
import pandas as pd

# Numerical guard: anything below this in absolute value is treated as "zero"
# to avoid blow-ups when dividing (e.g. near-flat volatility).
_EPS: float = 1e-12


# ====================================================================== #
#  Configuration
# ====================================================================== #
@dataclass
class AnalyzerConfig:
    """All tunable parameters of :class:`MacdTrendAnalyzer`.

    Attributes
    ----------
    N : int
        Secant / regression window length (bars between points A and B).
        Larger → smoother but laggier slope. Default 5.
    M : int
        Rolling window for the ``rolling_std`` normalization factor. Default 20.
    H : int
        Rolling window for the angle-strength percentile. Default 200.
    flat_threshold_deg : float
        Angle magnitude (degrees) below which the regime is ``FLAT``. Default 3.
    eta : float
        MWU/Hedge learning rate. Default 0.1.
    markov_min_obs : int
        Minimum transitions observed for a Markov row before using it (else a
        uniform distribution is returned). Default 30.
    kelly_enabled : bool
        Compute the (research-only) Kelly fraction. Default False.
    kelly_b : float
        Net odds ``b`` used in the Kelly formula. Default 2.0.
    normalization_method : str
        ``"rolling_std"`` (std of the AVG line, default) or ``"atr"`` (ATR(14)
        of the price; requires high/low/close to be supplied to ``update``).
    line_method : str
        ``"two_point"`` (secant through the window ends, default) or
        ``"linear_regression"`` (OLS over the whole window — more robust to
        edge noise).
    """

    N: int = 5
    M: int = 20
    H: int = 200
    flat_threshold_deg: float = 3.0
    eta: float = 0.1
    markov_min_obs: int = 30
    kelly_enabled: bool = False
    kelly_b: float = 2.0
    normalization_method: str = "rolling_std"
    line_method: str = "two_point"


# ====================================================================== #
#  Quadrant enum — the (small) state space of the Markov chain
# ====================================================================== #
class Quadrant(enum.Enum):
    """Five mutually-exclusive trend regimes.

    The alphabet is intentionally tiny: a small state space makes the empirical
    Markov transition matrix converge faster (Part 2). ``FLAT`` is the catch-all
    when the normalized angle is within ``±flat_threshold_deg``.
    """

    BULLISH_STRENGTHENING = "BULLISH_STRENGTHENING"  # above zero & rising
    BULLISH_WEAKENING = "BULLISH_WEAKENING"          # above zero & falling
    BEARISH_STRENGTHENING = "BEARISH_STRENGTHENING"  # below zero & falling
    BEARISH_WEAKENING = "BEARISH_WEAKENING"          # below zero & rising
    FLAT = "FLAT"                                    # |angle| <= threshold


# Stable ordering of states used everywhere (matrix rows/cols, percentile, …).
_STATES: tuple[Quadrant, ...] = (
    Quadrant.BULLISH_STRENGTHENING,
    Quadrant.BULLISH_WEAKENING,
    Quadrant.BEARISH_STRENGTHENING,
    Quadrant.BEARISH_WEAKENING,
    Quadrant.FLAT,
)
_STATE_INDEX: dict[Quadrant, int] = {s: i for i, s in enumerate(_STATES)}


def _camp(state: Quadrant) -> int:
    """Return the trend camp of a state: +1 bullish, -1 bearish, 0 flat."""
    if state in (Quadrant.BULLISH_STRENGTHENING, Quadrant.BULLISH_WEAKENING):
        return 1
    if state in (Quadrant.BEARISH_STRENGTHENING, Quadrant.BEARISH_WEAKENING):
        return -1
    return 0


# ====================================================================== #
#  Per-bar result
# ====================================================================== #
@dataclass
class BarResult:
    """Everything computed for a single bar (causal — uses data up to ``t``).

    Attributes
    ----------
    avg_value : float
        ``(macd_line + signal_line) / 2`` at this bar.
    position : str
        ``"above"`` if ``avg >= 0`` else ``"below"``.
    zero_cross : Optional[str]
        ``"bull_cross"`` / ``"bear_cross"`` / ``None`` (crossing of the AVG
        line through zero over the ``N``-bar window).
    norm_slope : Optional[float]
        Normalized slope of the AVG line (raw slope / normalization factor).
    angle_degrees : Optional[float]
        ``degrees(arctan(norm_slope))`` ∈ (-90, 90).
    quadrant : Optional[Quadrant]
        Regime bucket, or ``None`` if the angle is undefined.
    strength_instant : Optional[float]
        Instantaneous strength ``min(|angle|/90, 1.0)`` ∈ [0, 1].
    strength_percentile : Optional[float]
        Percentile (0-100) of ``|angle|`` vs the last ``H`` bars.
    markov_next_state_probs : dict[str, float]
        Empirical ``P(state[t+1] | state[t])`` over the 5 quadrants.
    composite_score : Optional[float]
        MWU-weighted vote of the 3 experts ∈ [-1, 1].
    conviction_multiplier : Optional[float]
        Spread-based conviction ``min(|spread_angle|/90, 1.0)`` ∈ [0, 1].
    final_trend_score : Optional[float]
        ``composite_score * (0.5 + 0.5 * conviction)`` ∈ [-1, 1].
    kelly_fraction : Optional[float]
        Research-only Kelly fraction, or ``None`` when disabled.
    """

    avg_value: Optional[float]
    position: Optional[str]
    zero_cross: Optional[str]
    norm_slope: Optional[float]
    angle_degrees: Optional[float]
    quadrant: Optional[Quadrant]
    strength_instant: Optional[float]
    strength_percentile: Optional[float]
    markov_next_state_probs: dict[str, float]
    composite_score: Optional[float]
    conviction_multiplier: Optional[float]
    final_trend_score: Optional[float]
    kelly_fraction: Optional[float] = None

    def as_dict(self) -> dict:
        """Dataclass → plain dict, with ``quadrant`` rendered as its value."""
        d = asdict(self)
        if self.quadrant is not None:
            d["quadrant"] = self.quadrant.value
        return d


# ====================================================================== #
#  Helper: compute MACD from a close series (for self-contained tests/demo)
# ====================================================================== #
def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Standard MACD via EMA (``close.ewm(span=...).mean()``).

    Provided for self-contained tests/demo only — the analyzer is designed to
    consume MACD/Signal computed by an external source.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    fast, slow, signal : int
        Classic MACD parameters (12/26/9).

    Returns
    -------
    (macd_line, signal_line) : tuple[pd.Series, pd.Series]
        ``macd_line = EMA_fast − EMA_slow``; ``signal_line = EMA(macd, signal)``.
        Uses ``adjust=False`` (the canonical recursive EMA form, matching the
        rest of the ``gex`` package).
    """
    close = pd.Series(close).astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# ====================================================================== #
#  Streaming analyzer
# ====================================================================== #
class MacdTrendAnalyzer:
    """Stateful, causal MACD-trend analyzer.

    Feed it one ``(macd_value, signal_value)`` per bar via :meth:`update` and
    get a :class:`BarResult` back. Internal state (AVG history, Markov counts,
    MWU weights, percentile buffer) is maintained incrementally.

    For ATR normalization, pass ``high_value``/``low_value``/``close_value`` to
    :meth:`update`; otherwise the analyzer falls back to ``rolling_std`` with a
    warning.

    Parameters
    ----------
    config : AnalyzerConfig, optional
        Configuration. Defaults are used if omitted.
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()

        # --- AVG line history ---
        self._avg: list[float] = []

        # --- Price history for ATR (only used when normalization_method="atr") ---
        self._high: list[float] = []
        self._low: list[float] = []
        self._close: list[float] = []

        # --- Angles history (for the rolling percentile) ---
        self._abs_angles: list[float] = []

        # --- Markov 5x5 transition counts ---
        self._trans_counts: np.ndarray = np.zeros(
            (len(_STATES), len(_STATES)), dtype=float
        )
        self._prev_state: Optional[Quadrant] = None

        # --- MWU / Hedge expert weights ---
        self._weights: np.ndarray = np.full(3, 1.0 / 3.0)
        # Votes emitted on the *previous* bar (to be scored on the next bar).
        self._prev_votes: Optional[np.ndarray] = None

        self._bar_index: int = 0

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        macd_value: float,
        signal_value: float,
        close_value: Optional[float] = None,
        high_value: Optional[float] = None,
        low_value: Optional[float] = None,
    ) -> BarResult:
        """Process one bar and return its :class:`BarResult`.

        No look-ahead: only data up to and including this bar is used. Any
        ``NaN`` input produces a degenerate result (``None``/``NaN`` fields)
        and a warning, but never raises.

        Parameters
        ----------
        macd_value, signal_value : float
            MACD line and Signal line values for this bar.
        close_value, high_value, low_value : float, optional
            Required only for ``normalization_method="atr"`` (true ATR uses
            high/low/close). If omitted in ATR mode, falls back to rolling_std.
        """
        cfg = self.config

        # --- Validate / sanitize inputs ----------------------------------
        if not _is_finite(macd_value) or not _is_finite(signal_value):
            warnings.warn(
                "MacdTrendAnalyzer.update received non-finite macd/signal; "
                "returning a degenerate (None) BarResult.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._degenerate_result()

        avg = (float(macd_value) + float(signal_value)) / 2.0
        self._avg.append(avg)

        # Price history (kept regardless; cheap and needed for ATR fallback).
        self._high.append(float(high_value) if _is_finite(high_value) else float("nan"))
        self._low.append(float(low_value) if _is_finite(low_value) else float("nan"))
        self._close.append(float(close_value) if _is_finite(close_value) else float("nan"))

        t = self._bar_index

        # --- Part 1: geometry --------------------------------------------
        position = "above" if avg >= 0.0 else "below"

        zero_cross = self._zero_cross(cfg.N)

        norm_factor = self._normalization_factor(cfg)
        raw_slope = self._raw_slope(cfg.N, cfg.line_method)
        norm_slope: Optional[float]
        if raw_slope is None or norm_factor is None or abs(norm_factor) < _EPS:
            if raw_slope is not None and (norm_factor is None or abs(norm_factor) < _EPS):
                warnings.warn(
                    "Normalization factor ~0; angle/slope set to None.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            norm_slope = None
        else:
            norm_slope = float(raw_slope / norm_factor)

        angle: Optional[float]
        if norm_slope is None:
            angle = None
        else:
            angle = float(np.degrees(np.arctan(norm_slope)))

        quadrant = self._classify(angle, cfg.flat_threshold_deg)
        strength_instant: Optional[float]
        if angle is None:
            strength_instant = None
        else:
            strength_instant = float(min(abs(angle) / 90.0, 1.0))

        # --- Part 2: probabilistic layer ---------------------------------
        if angle is not None:
            self._abs_angles.append(abs(angle))
        else:
            self._abs_angles.append(float("nan"))
        strength_percentile = self._strength_percentile(cfg.H, angle)

        # Markov chain: register the transition prev_state -> quadrant, then
        # emit next-state probabilities conditioned on the current state.
        if quadrant is not None and self._prev_state is not None:
            i = _STATE_INDEX[self._prev_state]
            j = _STATE_INDEX[quadrant]
            self._trans_counts[i, j] += 1.0
        next_probs = self._markov_next_probs(quadrant, cfg.markov_min_obs)
        if quadrant is not None:
            self._prev_state = quadrant

        # --- Part 3: game-theoretic layer (MWU + conviction) --------------
        composite_score, conviction_multiplier, final_trend_score = \
            self._game_layer(avg, macd_value, signal_value, norm_factor, next_probs)

        # --- Part 4 (optional): Kelly ------------------------------------
        kelly_fraction = self._kelly_fraction(next_probs, quadrant) \
            if cfg.kelly_enabled else None

        self._bar_index += 1

        return BarResult(
            avg_value=avg,
            position=position,
            zero_cross=zero_cross,
            norm_slope=norm_slope,
            angle_degrees=angle,
            quadrant=quadrant,
            strength_instant=strength_instant,
            strength_percentile=strength_percentile,
            markov_next_state_probs=next_probs,
            composite_score=composite_score,
            conviction_multiplier=conviction_multiplier,
            final_trend_score=final_trend_score,
            kelly_fraction=kelly_fraction,
        )

    # ------------------------------------------------------------------ #
    #  Internal: geometry helpers
    # ------------------------------------------------------------------ #
    def _zero_cross(self, N: int) -> Optional[str]:
        """Detect a zero-crossing of the AVG line over the last ``N`` bars."""
        avg = self._avg
        t = len(avg) - 1
        if t < N or N <= 0:
            return None
        a, b = avg[t - N], avg[t]
        if not (_is_finite(a) and _is_finite(b)):
            return None
        above_a, above_b = a >= 0.0, b >= 0.0
        if above_a == above_b:
            return None
        # below -> above = bull_cross ; above -> below = bear_cross
        return "bull_cross" if (not above_a and above_b) else "bear_cross"

    def _raw_slope(self, N: int, line_method: str) -> Optional[float]:
        """Raw (un-normalized) slope of the AVG line over the window."""
        avg = self._avg
        t = len(avg) - 1
        if t < N or N <= 0:
            return None
        window = avg[t - N: t + 1]
        if any(not _is_finite(v) for v in window):
            return None

        if line_method == "linear_regression" and len(window) >= 2:
            # OLS slope over the whole window [t-N .. t].
            x = np.arange(len(window), dtype=float)
            y = np.asarray(window, dtype=float)
            xvar = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
            if xvar < _EPS:
                return None
            slope = float(np.cov(x, y, ddof=1)[0, 1] / xvar)
            # Express per-bar (window covers N bars → divide by N like the
            # two-point secant, so both methods share the same units).
            return slope
        elif line_method == "linear_regression":
            return None

        # two_point (default): secant through A=(t-N, avg[t-N]), B=(t, avg[t]).
        return (window[-1] - window[0]) / float(N)

    def _normalization_factor(self, cfg: AnalyzerConfig) -> Optional[float]:
        """Volatility used to normalize the raw slope.

        ``rolling_std``  — sample std of the AVG line over the last ``M`` bars.
        ``atr``          — Wilder ATR(14) of the price (needs high/low/close);
                           falls back to rolling_std if OHLC was not supplied.
        """
        method = cfg.normalization_method
        if method == "atr":
            factor = self._atr(14)
            if factor is None:
                warnings.warn(
                    "ATR normalization unavailable (no/insufficient OHLC); "
                    "falling back to rolling_std.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._rolling_std(self._avg, cfg.M)
            return factor
        # default rolling_std
        return self._rolling_std(self._avg, cfg.M)

    @staticmethod
    def _rolling_std(series: Sequence[float], M: int) -> Optional[float]:
        """Sample std of the last ``M`` finite values of ``series``."""
        if M <= 1:
            return None
        tail = [v for v in series[-M:] if _is_finite(v)]
        if len(tail) < 2:
            return None
        return float(np.std(tail, ddof=1))

    def _atr(self, period: int = 14) -> Optional[float]:
        """Wilder-style ATR(period) from stored high/low/close.

        Returns ``None`` if OHLC is missing/insufficient or contains NaNs.
        Uses Wilder smoothing (``alpha = 1/period`` EMA of True Range), the
        convention used elsewhere in the ``gex`` package.
        """
        n = len(self._close)
        if n < period + 1:
            return None
        high = np.asarray(self._high, dtype=float)
        low = np.asarray(self._low, dtype=float)
        close = np.asarray(self._close, dtype=float)
        if np.isnan(high).any() or np.isnan(low).any() or np.isnan(close).any():
            return None
        # True Range (needs the previous close).
        pc = close[:-1]
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - pc),
            np.abs(low[1:] - pc),
        ])
        if len(tr) < period:
            return None
        # Wilder: seed with SMA of first `period` TRs, then recursive smoothing.
        atr = float(np.mean(tr[:period]))
        for k in range(period, len(tr)):
            atr = (atr * (period - 1) + tr[k]) / period
        return atr

    def _classify(self, angle: Optional[float], flat_threshold_deg: float) -> Optional[Quadrant]:
        """Map (current AVG position, angle) → Quadrant. ``None`` if undefined.

        The position ("above"/"below" zero) is read from the last appended AVG
        value (already stored on ``self``).
        """
        avg = self._avg[-1] if self._avg else None
        if angle is None or avg is None:
            return None
        above = avg >= 0.0
        if abs(angle) <= flat_threshold_deg:
            return Quadrant.FLAT
        if above:
            return Quadrant.BULLISH_STRENGTHENING if angle > 0 else Quadrant.BULLISH_WEAKENING
        # below zero
        return Quadrant.BEARISH_STRENGTHENING if angle < 0 else Quadrant.BEARISH_WEAKENING

    # ------------------------------------------------------------------ #
    #  Internal: probabilistic layer
    # ------------------------------------------------------------------ #
    def _strength_percentile(self, H: int, angle: Optional[float]) -> Optional[float]:
        """Percentile (0-100) of ``|angle|`` vs the last ``H`` finite values."""
        if angle is None:
            return None
        cur = abs(angle)
        tail = [v for v in self._abs_angles[-H:] if _is_finite(v)]
        if len(tail) == 0:
            return None
        arr = np.asarray(tail, dtype=float)
        # "Percentile" = fraction of history that is <= current value, ×100.
        frac = float(np.mean(arr <= cur))
        return frac * 100.0

    def _markov_next_probs(
        self, current: Optional[Quadrant], min_obs: int
    ) -> dict[str, float]:
        """``P(state[t+1] | state[t])`` as a dict keyed by quadrant value.

        Until ``min_obs`` transitions have been observed for the current row,
        a uniform distribution is returned (a non-informative prior).
        """
        n_states = len(_STATES)
        if current is None:
            return {s.value: 1.0 / n_states for s in _STATES}
        i = _STATE_INDEX[current]
        row = self._trans_counts[i]
        total = float(row.sum())
        if total < min_obs:
            return {s.value: 1.0 / n_states for s in _STATES}
        probs = row / total
        return {s.value: float(probs[j]) for j, s in enumerate(_STATES)}

    # ------------------------------------------------------------------ #
    #  Internal: game-theoretic layer
    # ------------------------------------------------------------------ #
    def _game_layer(
        self,
        avg: float,
        macd_value: float,
        signal_value: float,
        norm_factor: Optional[float],
        next_probs: dict[str, float],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """MWU expert voting + spread conviction → composite / final scores.

        Returns ``(composite_score, conviction_multiplier, final_trend_score)``.
        """
        cfg = self.config

        # --- 1. Score the *previous* votes against the realized outcome ---
        if self._prev_votes is not None and len(self._avg) >= 2:
            outcome = np.sign(avg - self._avg[-2])  # +1 / -1 / 0
            votes = self._prev_votes
            # loss_i: 0 if sign match, 1 if mismatch, 0.5 if expert was neutral.
            match = (np.sign(votes) == outcome)
            neutral = (np.sign(votes) == 0)
            loss = np.where(match, 0.0, np.where(neutral, 0.5, 1.0))
            # Guard against weight collapse to exactly zero.
            self._weights = self._weights * np.exp(-cfg.eta * loss)
            wsum = float(self._weights.sum())
            if wsum < _EPS:
                # Reset to uniform if everything collapsed.
                self._weights = np.full(3, 1.0 / 3.0)
            else:
                self._weights = self._weights / wsum

        # --- 2. Form the new votes (prediction t -> t+1) -----------------
        # Expert A: sign of position.
        vote_a = 1.0 if avg >= 0.0 else -1.0
        # Expert B: sign of angle with a flat dead-zone.
        angle = self._last_angle()
        if angle is not None and angle > cfg.flat_threshold_deg:
            vote_b = 1.0
        elif angle is not None and angle < -cfg.flat_threshold_deg:
            vote_b = -1.0
        else:
            vote_b = 0.0
        # Expert C: sign of the most probable next Markov state's camp.
        vote_c = _vote_from_markov(next_probs)

        votes = np.array([vote_a, vote_b, vote_c], dtype=float)
        self._prev_votes = votes

        composite_score = float(np.dot(self._weights, votes))
        # Домены обещает [-1, 1], и схема (`le=1.0`) это проверяет. Плавающая точка
        # это обещание нарушает: сумма весов, равная единице «на бумаге», даёт
        # 1.0000000000000002, а pydantic отвергает такое значение — ответ уходил не
        # 422, а 404 от обработчика, то есть дефект выглядел как «страницы нет».
        # Поэтому результат приводится к объявленному диапазону явно, а не «обычно
        # он и так в норме».
        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        # --- 3. Conviction multiplier from the MACD-Signal spread --------
        if norm_factor is None or abs(norm_factor) < _EPS:
            warnings.warn(
                "Normalization factor ~0; conviction set to None.",
                RuntimeWarning,
                stacklevel=2,
            )
            return composite_score, None, None
        spread = float(macd_value) - float(signal_value)
        spread_angle = float(np.degrees(np.arctan(spread / norm_factor)))
        conviction = float(min(abs(spread_angle) / 90.0, 1.0))

        final_trend_score = composite_score * (0.5 + 0.5 * conviction)
        return composite_score, conviction, final_trend_score

    # ------------------------------------------------------------------ #
    #  Internal: Kelly (research only)
    # ------------------------------------------------------------------ #
    def _kelly_fraction(
        self, next_probs: dict[str, float], current: Optional[Quadrant]
    ) -> Optional[float]:
        """Kelly fraction ``f* = p − (1−p)/b`` from the same-camp probability.

        .. warning::
           Research/experimental component, **not financial advice**. Requires
           separate validation before any real use.

        ``p`` = Markov probability that the next state stays in the same camp
        (bullish/bearish) as the current state.
        """
        cfg = self.config
        if current is None or _camp(current) == 0:
            return None
        camp = _camp(current)
        p = 0.0
        for s in _STATES:
            if _camp(s) == camp:
                p += next_probs.get(s.value, 0.0)
        p = float(np.clip(p, 0.0, 1.0))
        b = cfg.kelly_b
        if b <= _EPS:
            return None
        return float(p - (1.0 - p) / b)

    # ------------------------------------------------------------------ #
    #  Internal: small instance bridges / degenerate fallback
    # ------------------------------------------------------------------ #
    def _last_avg(self) -> Optional[float]:
        return self._avg[-1] if self._avg else None

    def _last_angle(self) -> Optional[float]:
        return self._abs_angles[-1] if (self._abs_angles and _is_finite(self._abs_angles[-1])) else None

    def _degenerate_result(self) -> BarResult:
        """All-None result used when the bar's inputs are unusable."""
        return BarResult(
            avg_value=None,
            position=None,
            zero_cross=None,
            norm_slope=None,
            angle_degrees=None,
            quadrant=None,
            strength_instant=None,
            strength_percentile=None,
            markov_next_state_probs={s.value: 1.0 / len(_STATES) for s in _STATES},
            composite_score=None,
            conviction_multiplier=None,
            final_trend_score=None,
            kelly_fraction=None,
        )


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #
def _is_finite(x: Optional[float]) -> bool:
    """True iff ``x`` is a finite number (rejects None / NaN / inf)."""
    if x is None:
        return False
    try:
        return bool(np.isfinite(x))
    except (TypeError, ValueError):
        return False


def _vote_from_markov(next_probs: dict[str, float]) -> float:
    """Expert C vote: camp of the single most probable next state.

    Ties are broken toward ``FLAT`` (→ 0): a genuinely ambiguous forecast is
    treated as no-vote rather than a coin flip.
    """
    # argmax over the (state_value -> prob) dict, FLAT wins ties.
    best_state = Quadrant.FLAT
    best_p = -1.0
    for s in _STATES:
        p = next_probs.get(s.value, 0.0)
        if p > best_p + _EPS:
            best_p = p
            best_state = s
    return float(_camp(best_state))


# ====================================================================== #
#  Batch mode
# ====================================================================== #
def analyze_history(
    macd_line: "pd.Series | Sequence[float]",
    signal_line: "pd.Series | Sequence[float]",
    config: Optional[AnalyzerConfig] = None,
    close: "Optional[pd.Series | Sequence[float]]" = None,
    high: "Optional[pd.Series | Sequence[float]]" = None,
    low: "Optional[pd.Series | Sequence[float]]" = None,
) -> pd.DataFrame:
    """Run the streaming analyzer over a whole history → tidy DataFrame.

    Produces **exactly** the same results as calling :meth:`update` bar-by-bar
    on the same data (this function is a thin loop around it).

    Parameters
    ----------
    macd_line, signal_line : array-like
        Equal-length MACD / Signal series.
    config : AnalyzerConfig, optional
    close, high, low : array-like, optional
        Required only for ``normalization_method="atr"``.

    Returns
    -------
    pd.DataFrame
        One row per bar, indexed ``0..n-1``, with columns matching
        :class:`BarResult` (``quadrant`` is stored as its string value).
    """
    macd_line = pd.Series(macd_line).astype(float)
    signal_line = pd.Series(signal_line).astype(float)
    if len(macd_line) != len(signal_line):
        raise ValueError(
            f"macd_line and signal_line must have equal length "
            f"({len(macd_line)} != {len(signal_line)})."
        )

    n = len(macd_line)
    close_v = _optional_series(close, n)
    high_v = _optional_series(high, n)
    low_v = _optional_series(low, n)

    analyzer = MacdTrendAnalyzer(config)
    records: list[dict] = []
    for t in range(n):
        result = analyzer.update(
            macd_value=float(macd_line.iloc[t]),
            signal_value=float(signal_line.iloc[t]),
            close_value=close_v[t],
            high_value=high_v[t],
            low_value=low_v[t],
        )
        records.append(result.as_dict())

    df = pd.DataFrame.from_records(records)
    return df


def _optional_series(
    s: "Optional[pd.Series | Sequence[float]]", n: int
) -> list[Optional[float]]:
    """Coerce an optional array-like into a list[Optional[float]] of length n."""
    if s is None:
        return [None] * n
    s = pd.Series(s).astype(float)
    return [None if not _is_finite(v) else float(v) for v in s.tolist()]


# ====================================================================== #
#  Demo (synthetic sine + noise)
# ====================================================================== #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 600
    t = np.arange(n)
    # Smooth synthetic price: uptrend + sine cycle + noise.
    price = 100.0 + 0.05 * t + 8.0 * np.sin(2 * np.pi * t / 120.0) + rng.normal(0, 0.5, n)
    close = pd.Series(price)

    macd_line, signal_line = compute_macd(close)
    df = analyze_history(macd_line, signal_line)

    last = df.iloc[-1]
    print("=== Last bar ===")
    print(f"  quadrant            : {last['quadrant']}")
    print(f"  angle_degrees       : {last['angle_degrees']:.2f}")
    print(f"  strength_instant    : {last['strength_instant']:.3f}")
    print(f"  strength_percentile : {last['strength_percentile']:.1f}")
    print(f"  composite_score     : {last['composite_score']:+.3f}")
    print(f"  conviction          : {last['conviction_multiplier']:.3f}")
    print(f"  final_trend_score   : {last['final_trend_score']:+.3f}")
    print(f"  rows                : {len(df)}")

    # Optional plot (matplotlib not required for the engine itself).
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(close.values, label="close", color="black")
        axes[0].set_title("Synthetic close")
        axes[1].plot(macd_line.values, label="MACD")
        axes[1].plot(signal_line.values, label="Signal")
        axes[1].axhline(0, color="grey", lw=0.5)
        axes[1].legend(loc="upper left")
        axes[2].plot(df["final_trend_score"].astype(float).values, label="final_trend_score")
        axes[2].axhline(0, color="grey", lw=0.5)
        axes[2].set_title("Final trend score [-1, 1]")
        axes[2].legend(loc="upper left")
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")
