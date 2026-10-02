"""AUTO-scope domain logic for the GEX Details DEPTH/EXPIRIES toggle.

This module is intentionally **pure**: no I/O, no FastAPI, no provider clients.
It owns the three decisions the AUTO mode makes on the server:

1. **What "wide" means** — the scope maxima (:data:`AUTO_MAX_DAYS` /
   :data:`AUTO_MAX_EXPIRIES`) and the cache lifetime (:data:`AUTO_CACHE_TTL`).
2. **Is the built profile thin?** — :func:`detect_sparse`, a pure function of the
   assembled per-strike profile with *named* thresholds, so it is unit-testable with
   synthetic chains (see design doc §5.1). Reasons: ``no_data``, ``total_oi_zero``,
   ``low_strikes``, ``low_atm_strikes``, ``missing_call_wall``, ``missing_put_wall``
   and ``low_expiries`` (fewer than :data:`AUTO_MIN_EXPIRIES` distinct usable expiry
   buckets — the primary provider returned a single expiry bucket).
3. **How to combine two providers without double counting** — :func:`merge_chains`
   performs a chain-level union keyed ``(strike, type, round(T*365))`` where the
   primary row wins in full; and :func:`fallback_source_for` maps a primary source
   to the one provider that may be consulted for escalation.

Keeping this logic in one dependency-free module is deliberate: the escalation
decision spends a second (up to ~15 s) network fetch, so it must be a measurable
function of the data — not a heuristic scattered across the router and the UI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

# ====================================================================== #
#  Constants (design doc §5.1 / §5.3 — thresholds are asserted by name in tests)
# ====================================================================== #
#: Widest scope the AUTO toggle ever requests (the API maximum).
AUTO_MAX_DAYS: float = 90.0
#: Expiry cap for the widest scope (US / crypto / futures).
AUTO_MAX_EXPIRIES: int = 20
#: Result-cache lifetime in seconds (locked requirement: ~10 minutes).
AUTO_CACHE_TTL: int = 600

#: Absolute floor of distinct strikes below which the profile is "thin".
AUTO_MIN_STRIKES: int = 8
#: Half-width of the "near ATM" band, as a fraction of spot (±10%).
AUTO_ATM_BAND_PCT: float = 0.10
#: Distinct strikes required *inside* the ATM band.
AUTO_MIN_ATM_STRIKES: int = 5
#: Minimum distinct usable expiry buckets (rounded T-days); below → under-covered.
AUTO_MIN_EXPIRIES: int = 3

#: Reason codes emitted by :func:`detect_sparse` (machine-readable, for tooltips / tests).
REASON_NO_DATA = "no_data"
REASON_TOTAL_OI_ZERO = "total_oi_zero"
REASON_LOW_STRIKES = "low_strikes"
REASON_LOW_ATM_STRIKES = "low_atm_strikes"
REASON_MISSING_CALL_WALL = "missing_call_wall"
REASON_MISSING_PUT_WALL = "missing_put_wall"
REASON_LOW_EXPIRIES = "low_expiries"


# ====================================================================== #
#  Value objects
# ====================================================================== #
@dataclass
class StrikeLite:
    """Provider-agnostic view of one per-strike row consumed by :func:`detect_sparse`.

    Callers adapt their own per-strike objects (``ExtendedStrike`` on the ext path,
    ``StrikeProfileOut`` on the MOEX path) into this shape, so the sparse rule does
    not depend on any particular report class.
    """

    strike: float
    oi_call: float
    oi_put: float
    gex_net: float


@dataclass
class AutoCoverage:
    """Coverage metadata attached to a report produced in AUTO mode (design §5)."""

    mode: str = "auto"
    resolved_days: float = AUTO_MAX_DAYS
    resolved_expiries: int = AUTO_MAX_EXPIRIES
    sources_used: list[str] = field(default_factory=list)
    primary_source: str = ""
    fallback_used: bool = False
    escalated: bool = False
    partial: bool = False
    expirations_merged: int = 0
    strike_count: int = 0
    total_oi: float = 0.0
    sparse: bool = False
    sparse_reasons: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    # Additive Phase-4 coverage metadata (optional; None/absent = not computed).
    strike_min: Optional[float] = None
    strike_max: Optional[float] = None
    nearest_expiry_days: Optional[float] = None
    # Верхняя граница использованного диапазона данных (аудит 2026-09-17:
    # «data range used» — ближняя экспирации без дальней неполна).
    furthest_expiry_days: Optional[float] = None


# ====================================================================== #
#  Sparse detection (design §5.1)
# ====================================================================== #
def _wall_missing(value: object) -> bool:
    """True if a wall price is ``None``, ``NaN``/``Inf`` or ``<= 0`` (edge case 14)."""
    if value is None:
        return True
    try:
        f = float(value)
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(f)) or f <= 0.0


def detect_sparse(
    strikes: Sequence[StrikeLite],
    spot: float,
    call_wall: Optional[float],
    put_wall: Optional[float],
    expiries: Optional[int] = None,
) -> list[str]:
    """Return the **union** of reason codes for a thin profile; ``[]`` means usable.

    Parameters
    ----------
    strikes : Sequence[StrikeLite]
        The assembled per-strike profile (any object exposing ``strike``,
        ``oi_call``, ``oi_put``; a dataclass is expected but only attributes are read).
    spot : float
        Pricing reference. ``spot <= 0`` makes the ATM test unsatisfiable.
    call_wall, put_wall : Optional[float]
        The wall prices from the assembled report; ``None``/``NaN``/``<= 0`` count
        as missing.
    expiries : Optional[int]
        Number of distinct usable expiry buckets (rounded T-days) in the chain
        (see :func:`count_expiries`). When provided and ``< AUTO_MIN_EXPIRIES`` the
        profile is under-covered and :data:`REASON_LOW_EXPIRIES` is appended. This
        parameter is **optional**: omitting it (``None``) skips the expiry check so
        existing callers keep byte-identical behaviour.

    Returns
    -------
    list[str]
        Empty ⇒ not sparse. Otherwise the reasons from the design table, in a
        stable order so tests can assert them by name. The expiry reason, when
        present, is appended **last**.
    """
    reasons: list[str] = []

    n = len(strikes)
    if n == 0:
        reasons.append(REASON_NO_DATA)

    total_oi = 0.0
    for s in strikes:
        try:
            total_oi += float(s.oi_call) + float(s.oi_put)
        except (TypeError, ValueError):
            continue
    if total_oi <= 0.0:
        reasons.append(REASON_TOTAL_OI_ZERO)

    if n < AUTO_MIN_STRIKES:
        reasons.append(REASON_LOW_STRIKES)

    # Distinct strikes inside the ±AUTO_ATM_BAND_PCT band of spot.
    atm_ok = False
    if spot is not None and math.isfinite(float(spot)) and float(spot) > 0.0:
        band = float(spot) * AUTO_ATM_BAND_PCT
        atm_count = 0
        for s in strikes:
            try:
                k = float(s.strike)
            except (TypeError, ValueError):
                continue
            if math.isfinite(k) and abs(k - float(spot)) <= band:
                atm_count += 1
        atm_ok = atm_count >= AUTO_MIN_ATM_STRIKES
    if not atm_ok:
        reasons.append(REASON_LOW_ATM_STRIKES)

    if _wall_missing(call_wall):
        reasons.append(REASON_MISSING_CALL_WALL)
    if _wall_missing(put_wall):
        reasons.append(REASON_MISSING_PUT_WALL)

    # Under-covered: too few distinct usable expiry buckets. Optional input, so
    # callers that do not pass ``expiries`` are unaffected (backward compatible).
    if expiries is not None and expiries < AUTO_MIN_EXPIRIES:
        reasons.append(REASON_LOW_EXPIRIES)

    return reasons


# ====================================================================== #
#  Merge (design §5.2) — chain-level union, no double counting
# ====================================================================== #
def merge_chains(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Union of two option chains without double counting.

    Union key : ``(strike, type, round(T * 365))`` — the exact granularity at which
    ``_build_per_strike`` aggregates, so after de-duplication every contract appears
    once and contributes exactly one weighted GEX term.

    Priority  : the **primary row wins in full** on a key collision (entire row, all
    fields — never per-field mixing, never summing overlapping rows). Fallback-unique
    rows are added; that is the point of escalation.

    ``spot``/``symbol`` are **not** part of the chain frame: the caller keeps the
    primary snapshot's pricing reference (see ``analyze_auto``).
    """
    pri = primary.assign(_prio=0)
    fb = fallback.assign(_prio=1)
    both = pd.concat([pri, fb], ignore_index=True)
    both["_tday"] = (both["T"].astype(float) * 365.0).round().astype("int64")
    # Primary first, stable → ``keep="first"`` keeps the primary row on collisions.
    both = both.sort_values("_prio", kind="stable")
    both = both.drop_duplicates(subset=["strike", "type", "_tday"], keep="first")
    return both.drop(columns=["_prio", "_tday"]).reset_index(drop=True)


def count_expiries(chain: Optional[pd.DataFrame]) -> int:
    """Number of distinct expiry buckets (rounded T-days) in a chain frame."""
    if chain is None or len(chain) == 0 or "T" not in chain.columns:
        return 0
    try:
        return int((chain["T"].astype(float) * 365.0).round().nunique())
    except (TypeError, ValueError):
        return 0


# ====================================================================== #
#  Fallback mapping (design §5.2 table)
# ====================================================================== #
#: Primary sources that have exactly one symmetric counterpart for escalation.
_SYMMETRIC_FALLBACK: dict[str, str] = {"webull": "yfinance", "yfinance": "webull"}


def fallback_source_for(primary_source: str, symbol: str) -> Optional[str]:
    """The one source AUTO may consult for escalation, or ``None`` if single-provider.

    Crypto (Bybit), futures (yfinance proxy), the unclassifiable ``stock`` net and
    MOEX ISS have no second provider — AUTO then does **max scope only**. For
    ``webull``/``yfinance`` the *other* of the pair is returned, so pinning a source
    while AUTO is ON still guarantees completeness (design decisions, Q1).
    """
    src = (primary_source or "").strip().lower()
    return _SYMMETRIC_FALLBACK.get(src)
