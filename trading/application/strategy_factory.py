"""Strategy factory — one place that turns ``(name, symbol, params)`` into a ``Strategy``.

The parameter *shape* for each strategy (which keys it accepts, how a nested
``long``/``short`` block maps onto flat ``long_fast``/``short_slow`` keys, …) is
knowledge that must not leak into the API or the backtest engines. Both the
single-symbol backtest and the multi-ticker portfolio backtest build their
strategies through here, so a strategy tuned in one path behaves identically in
the other.

Design notes
------------
* Every strategy is reachable by its registered name and is built with plain,
  JSON-serialisable ``params`` — that is what makes per-ticker individual
  settings possible straight from the API request.
* Bad parameters raise :class:`~trading.domain.StrategyError` (a domain error)
  rather than ``ValueError``/``KeyError`` so the API can map it to a 4xx.
"""
from __future__ import annotations

from typing import Any, Mapping

from trading.domain import StrategyError
from trading.ports import Strategy

__all__ = ["STRATEGY_NAMES", "build_strategy"]

#: Names accepted by :func:`build_strategy` (kept in sync with the registry).
STRATEGY_NAMES: tuple[str, ...] = (
    "sma_crossover",
    "sma_crossover_ls",
    "buy_and_hold",
    "mean_reversion",
    "momentum",
    "gex_emf",
)


def _pick(nested: Mapping[str, Any], flat: Mapping[str, Any], *keys: str, default: Any) -> Any:
    """First non-``None`` value for ``keys`` in ``nested`` then ``flat``."""
    for k in keys:
        if k in nested and nested[k] is not None:
            return nested[k]
    for k in keys:
        if k in flat and flat[k] is not None:
            return flat[k]
    return default


def _to_int(value: Any, key: str, minimum: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"parameter '{key}' must be an integer, got {value!r}") from exc
    if out < minimum:
        raise StrategyError(f"parameter '{key}' must be >= {minimum}, got {out}")
    return out


def _to_float(value: Any, key: str, *, gt: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"parameter '{key}' must be a number, got {value!r}") from exc
    if gt is not None and out <= gt:
        raise StrategyError(f"parameter '{key}' must be > {gt}, got {out}")
    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _side(nested: Mapping[str, Any], flat: Mapping[str, Any], prefix: str,
          default_fast: int, default_slow: int) -> tuple[bool, int, int, float]:
    """Extract ``(enabled, fast, slow, strength)`` for one leg of the dual strategy."""
    enabled = _to_bool(_pick(nested, flat, "enabled", f"{prefix}_enabled", default=True))
    fast = _to_int(_pick(nested, flat, "fast", f"{prefix}_fast", default=default_fast),
                   f"{prefix}_fast", 1)
    slow = _to_int(_pick(nested, flat, "slow", f"{prefix}_slow", default=default_slow),
                   f"{prefix}_slow", 2)
    strength = _to_float(_pick(nested, flat, "strength", f"{prefix}_strength", default=1.0),
                         f"{prefix}_strength", gt=0.0)
    return enabled, fast, slow, strength


def build_strategy(name: str, symbol: str, params: Mapping[str, Any] | None = None) -> Strategy:
    """Build a registered strategy for ``symbol`` with per-strategy ``params``.

    ``params`` is a flat mapping; the dual strategy additionally accepts nested
    ``long`` / ``short`` blocks (as sent by the API for per-side settings).
    """
    p: dict[str, Any] = dict(params or {})

    if name == "sma_crossover":
        from trading.application.strategies.sma_crossover import SmaCrossover

        fast = _to_int(_pick({}, p, "fast", default=20), "fast", 1)
        slow = _to_int(_pick({}, p, "slow", default=50), "slow", 2)
        if fast >= slow:
            raise StrategyError(f"sma_crossover requires fast < slow (got {fast} >= {slow})")
        return SmaCrossover(symbol, fast=fast, slow=slow)

    if name == "sma_crossover_ls":
        from trading.application.strategies.dual_sma_crossover import DualSmaCrossover

        long_cfg = p.get("long") if isinstance(p.get("long"), Mapping) else {}
        short_cfg = p.get("short") if isinstance(p.get("short"), Mapping) else {}
        l_en, l_fast, l_slow, l_str = _side(long_cfg, p, "long", 10, 20)
        s_en, s_fast, s_slow, s_str = _side(short_cfg, p, "short", 10, 20)
        return DualSmaCrossover(
            symbol,
            long_enabled=l_en, long_fast=l_fast, long_slow=l_slow, long_strength=l_str,
            short_enabled=s_en, short_fast=s_fast, short_slow=s_slow, short_strength=s_str,
        )

    if name == "buy_and_hold":
        from trading.application.strategies.buy_and_hold import BuyAndHold

        return BuyAndHold(symbol)

    if name == "mean_reversion":
        from trading.application.strategies.mean_reversion import MeanReversion

        return MeanReversion(symbol, period=_to_int(_pick({}, p, "period", default=20), "period", 2))

    if name == "momentum":
        from trading.application.strategies.momentum import Momentum

        return Momentum(symbol, period=_to_int(_pick({}, p, "period", default=10), "period", 1))

    if name == "gex_emf":
        from trading.application.strategies.gex_emf import GexEMFStrategy

        nested = p.get("settings")
        settings: dict[str, Any] = dict(nested) if isinstance(nested, Mapping) else {
            k: v for k, v in p.items() if k not in ("fast", "slow", "period")
        }
        # ``use_risk_exits`` is an adapter switch, not a StrategySettings field, so
        # read it (from either place) and drop it before constructing the settings —
        # otherwise it would be silently filtered out by ``from_dict``.
        flag = settings.pop("use_risk_exits", None)
        if flag is None:
            flag = p.get("use_risk_exits")
        return GexEMFStrategy(
            symbol, settings=settings or None, use_risk_exits=bool(flag)
        )

    raise StrategyError(f"unknown strategy '{name}' (known: {', '.join(STRATEGY_NAMES)})")
