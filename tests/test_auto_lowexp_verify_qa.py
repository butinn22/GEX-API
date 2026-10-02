"""Independent QA verification for the ``low_expiries`` escalation fix (AUTO mode).

Written by QA (严过关) **independently** of the engineer's suites. It re-derives the
contract from the fix description and probes the boundaries adversarially:

* ``detect_sparse`` expiry boundaries (0/1/2 flagged, 3/4 clean, omitted ⇒ legacy);
* the new ``expiries`` kwarg is genuinely **optional** (4-positional call unchanged);
* ``low_expiries`` is appended **last** to the reason list;
* a healthy multi-bucket chain ⇒ exactly ONE fetch, ``escalated=False``, ``sparse=False``;
* an under-covered (2-bucket) but otherwise rich primary ⇒ exactly TWO fetches,
  successful union, ``low_expiries`` cleared;
* under-covered + fallback failure ⇒ ``partial`` and primary data, no exception;
* MOEX: 6 buckets clean; a synthetic 2-bucket MOEX-like profile is flagged but
  never escalates (ISS has no fallback);
* every ``detect_sparse`` call site under ``gex/`` passes the new ``expiries`` kwarg.

No network: every provider interaction is an in-process fake / synthetic chain.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gex.application.auto_scope import (
    AUTO_MIN_EXPIRIES,
    REASON_LOW_EXPIRIES,
    StrikeLite,
    count_expiries,
    detect_sparse,
)
from gex.application.extended import ExtendedGEXAnalyzer
from gex.application.moex_service import MOEXGEXService, MOEX_PRIMARY_SOURCE
from gex.domain.data_loader import OptionSnapshot


# ====================================================================== #
#  Fixtures (independent of the engineer's helpers)
# ====================================================================== #
def _rich_lite(spot: float = 100.0, n: int = 10) -> list[StrikeLite]:
    """Rich by every non-expiry rule: n strikes inside the ATM band, positive OI."""
    return [
        StrikeLite(strike=95.0 + i, oi_call=10.0, oi_put=10.0, gex_net=1.0)
        for i in range(n)
    ]


def _snapshot(
    spot: float = 100.0,
    symbol: str = "AAPL",
    expiry_days: tuple[float, ...] = (30.0,),
) -> OptionSnapshot:
    """Synthetic chain with one row per (strike, side) per ``expiry_days`` bucket.

    OI slopes so both a finite Call Wall and Put Wall appear; 11 strikes straddle
    spot so the ATM / strike-count rules never fire on their own.
    """
    strikes = np.linspace(80.0, 120.0, 11)
    rows = []
    for days in expiry_days:
        T = days / 365.0
        for k in strikes:
            k = float(k)
            diff = k - spot
            oi_c = max(100.0, 1000.0 + diff * 40.0)
            oi_p = max(100.0, 1000.0 - diff * 40.0)
            iv = 0.2 * (1.0 + 2.0 * ((k - spot) / spot) ** 2)
            rows.append({"strike": k, "type": "C", "oi": oi_c, "iv": iv, "T": T})
            rows.append({"strike": k, "type": "P", "oi": oi_p, "iv": iv, "T": T})
    chain = pd.DataFrame(rows, columns=["strike", "type", "oi", "iv", "T"])
    return OptionSnapshot(symbol=symbol, spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)


def _patch_fetch(analyzer, primary, primary_src, fallback=None,
                 fallback_force="yfinance", fallback_raises=False):
    """Replace ``_fetch`` and record every ``(ticker, max_expiries, force_source)``."""
    calls: list[tuple] = []

    def fake_fetch(ticker, max_expiries, force_source=None, max_days=None):
        calls.append((ticker, max_expiries, force_source))
        if fallback_force is not None and force_source == fallback_force:
            if fallback_raises:
                raise RuntimeError("fallback provider down (synthetic)")
            return fallback, fallback_force, 100, 0.0
        return primary, primary_src, 100, 0.0

    analyzer._fetch = fake_fetch  # type: ignore[assignment]
    return calls


# ====================================================================== #
#  1. detect_sparse — expiry boundaries + backward compatibility
# ====================================================================== #
class TestDetectSparseExpiryBoundaries:
    """< AUTO_MIN_EXPIRIES ⇒ flagged; >= ⇒ clean; omitted ⇒ legacy behaviour."""

    @pytest.mark.parametrize("expiries", [0, 1, 2])
    def test_below_threshold_is_flagged(self, expiries):
        reasons = detect_sparse(_rich_lite(), 100.0, 105.0, 95.0, expiries=expiries)
        assert reasons == [REASON_LOW_EXPIRIES], reasons

    @pytest.mark.parametrize("expiries", [3, 4, 5, 20])
    def test_at_or_above_threshold_is_clean(self, expiries):
        assert detect_sparse(_rich_lite(), 100.0, 105.0, 95.0, expiries=expiries) == []

    def test_threshold_is_three(self):
        assert AUTO_MIN_EXPIRIES == 3

    def test_omitted_param_matches_explicit_none(self):
        # Backward compat: the 4-arg positional call must equal expiries=None.
        four = detect_sparse(_rich_lite(), 100.0, 105.0, 95.0)
        five_none = detect_sparse(_rich_lite(), 100.0, 105.0, 95.0, expiries=None)
        assert four == five_none == []

    def test_omitted_param_does_not_suppress_other_reasons(self):
        # A profile that is thin for other reasons keeps those reasons when expiries
        # is omitted — and equals the expiries=None form byte-for-byte.
        strikes = _rich_lite()
        assert detect_sparse(strikes, 100.0, None, 95.0) == ["missing_call_wall"]
        assert detect_sparse(strikes, 100.0, None, 95.0, expiries=None) == ["missing_call_wall"]

    def test_optional_default_is_none_via_signature(self):
        import inspect

        sig = inspect.signature(detect_sparse)
        assert list(sig.parameters) == ["strikes", "spot", "call_wall", "put_wall", "expiries"]
        assert sig.parameters["expiries"].default is None

    def test_low_expiries_appended_last(self):
        reasons = detect_sparse(_rich_lite(), 100.0, None, 95.0, expiries=1)
        assert reasons == ["missing_call_wall", REASON_LOW_EXPIRIES]
        assert reasons[-1] == REASON_LOW_EXPIRIES

    def test_low_expiries_is_last_among_all_reasons(self):
        # Empty profile + missing walls + low expiries → full union, expiry reason last.
        reasons = detect_sparse([], 100.0, None, None, expiries=2)
        assert reasons == [
            "no_data",
            "total_oi_zero",
            "low_strikes",
            "low_atm_strikes",
            "missing_call_wall",
            "missing_put_wall",
            REASON_LOW_EXPIRIES,
        ]


# ====================================================================== #
#  2. count_expiries sanity (feeds the new rule)
# ====================================================================== #
class TestCountExpiries:
    def test_two_and_six_buckets(self):
        two = _snapshot(expiry_days=(30.0, 60.0))
        six = _snapshot(expiry_days=(7.0, 14.0, 30.0, 60.0, 90.0, 180.0))
        assert count_expiries(two.chain) == 2
        assert count_expiries(six.chain) == 6

    def test_rounds_into_buckets(self):
        # 30.2d and 30.4d round to the same 30-day bucket; 31.4d is a distinct bucket.
        df = pd.DataFrame({"T": [30.2 / 365, 30.4 / 365, 31.4 / 365]})
        assert count_expiries(df) == 2


# ====================================================================== #
#  3. analyze_auto — healthy (no false positive) and under-covered escalation
# ====================================================================== #
class TestAnalyzeAutoExpiryEscalation:

    def test_healthy_multi_bucket_does_not_escalate(self):
        """≥3 buckets + rich profile ⇒ exactly one fetch, no escalation, not sparse."""
        an = ExtendedGEXAnalyzer()
        healthy = _snapshot(expiry_days=(7.0, 30.0, 60.0))
        calls = _patch_fetch(an, healthy, "webull")

        rep = an.analyze_auto("AAPL")
        cov = rep.coverage

        assert len(calls) == 1, "healthy profile must not escalate"
        assert calls[0][2] is None              # primary: auto resolution
        assert cov.escalated is False
        assert cov.fallback_used is False
        assert cov.partial is False
        assert cov.sparse is False
        assert cov.sparse_reasons == []
        assert cov.sources_used == ["webull"]
        assert cov.expirations_merged == 3

    def test_undercovered_primary_escalates_and_clears_low_expiries(self):
        """2-bucket rich primary ⇒ exactly two fetches; union ≥3 buckets, not sparse."""
        an = ExtendedGEXAnalyzer()
        primary = _snapshot(expiry_days=(30.0, 60.0))                       # 2 buckets
        fallback = _snapshot(expiry_days=(7.0, 30.0, 60.0))                 # adds bucket 7
        calls = _patch_fetch(an, primary, "webull", fallback=fallback, fallback_force="yfinance")

        rep = an.analyze_auto("AAPL")
        cov = rep.coverage

        assert count_expiries(primary.chain) == 2
        assert len(calls) == 2, "exactly one extra fetch"
        assert calls[0][2] is None
        assert calls[1][2] == "yfinance"
        assert calls[1][1] == 20                     # fallback also fetched at max scope
        assert cov.escalated is True
        assert cov.fallback_used is True
        assert cov.partial is False
        assert cov.sources_used == ["webull", "yfinance"]
        assert cov.expirations_merged >= 3
        assert cov.sparse is False
        assert REASON_LOW_EXPIRIES not in cov.sparse_reasons

    def test_upgrade_by_extra_bucket_only(self):
        """The escalation is driven *solely* by low_expiries, not by strike/OI/wall rules."""
        an = ExtendedGEXAnalyzer()
        primary = _snapshot(expiry_days=(30.0, 60.0))
        fallback = _snapshot(expiry_days=(7.0,))    # a lone distinct bucket is enough
        _patch_fetch(an, primary, "webull", fallback=fallback, fallback_force="yfinance")

        rep = an.analyze_auto("AAPL")
        cov = rep.coverage
        assert cov.escalated is True
        assert cov.fallback_used is True
        assert cov.expirations_merged >= 3

    def test_fallback_failure_is_partial_and_low_expiries_persists(self):
        """Under-covered + fallback raising ⇒ partial, primary data, no exception."""
        an = ExtendedGEXAnalyzer()
        primary = _snapshot(expiry_days=(30.0, 60.0))
        calls = _patch_fetch(an, primary, "webull", fallback_raises=True)

        rep = an.analyze_auto("AAPL")               # must NOT raise
        cov = rep.coverage

        assert len(calls) == 2
        assert cov.partial is True
        assert cov.fallback_used is False
        assert cov.escalated is True                # attempt was made
        assert cov.sources_used == ["webull"]
        assert REASON_LOW_EXPIRIES in cov.sparse_reasons
        assert cov.sparse is True
        assert len(rep.per_strike) > 0              # primary data returned

    def test_single_bucket_webull_style_escalates(self):
        """Real-world shape: webull returns ONE usable bucket ⇒ escalate."""
        an = ExtendedGEXAnalyzer()
        primary = _snapshot(expiry_days=(30.0,))
        fallback = _snapshot(expiry_days=(7.0, 30.0, 60.0))
        calls = _patch_fetch(an, primary, "webull", fallback=fallback, fallback_force="yfinance")

        an.analyze_auto("AAPL")
        assert len(calls) == 2


# ====================================================================== #
#  4. MOEX — informational flag, never escalates
# ====================================================================== #
class TestMoexLowExpiries:

    @staticmethod
    def _service():
        return MOEXGEXService(repo=object(), runner=object())

    @staticmethod
    def _out(spot: float = 100.0, call_wall: float = 120.0, put_wall: float = 80.0):
        strikes = [
            SimpleNamespace(strike=float(k), oi_call=100.0, oi_put=100.0, gex_net=1.0)
            for k in np.linspace(80.0, 120.0, 11)
        ]
        profile = SimpleNamespace(per_strike=strikes, call_wall=call_wall, put_wall=put_wall)
        return SimpleNamespace(symbol="RTS", spot=spot, profile=profile)

    def test_six_bucket_chain_clean(self):
        svc = self._service()
        snap = _snapshot(symbol="RTS", expiry_days=(7.0, 14.0, 30.0, 60.0, 90.0, 180.0))
        cov = svc._auto_coverage(snap, self._out(), days=90.0, max_expiries=0, elapsed_ms=1)
        assert cov.expirations_merged == 6
        assert REASON_LOW_EXPIRIES not in cov.sparse_reasons
        assert cov.sparse is False
        assert cov.escalated is False

    def test_two_bucket_flagged_but_never_escalates(self):
        """ISS has no fallback ⇒ sparse is informational; escalated/fallback stay False."""
        svc = self._service()
        snap = _snapshot(symbol="RTS", expiry_days=(30.0, 60.0))
        cov = svc._auto_coverage(snap, self._out(), days=90.0, max_expiries=0, elapsed_ms=1)
        assert cov.expirations_merged == 2
        assert REASON_LOW_EXPIRIES in cov.sparse_reasons
        assert cov.sparse is True
        assert cov.escalated is False
        assert cov.fallback_used is False
        assert cov.sources_used == [MOEX_PRIMARY_SOURCE]


# ====================================================================== #
#  5. Static guard: every detect_sparse call site passes the new kwarg
# ====================================================================== #
def _detect_sparse_call_sites() -> list[tuple[str, int, bool]]:
    root = Path(__file__).resolve().parents[1] / "gex"
    sites: list[tuple[str, int, bool]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            else:
                continue
            if name == "detect_sparse":
                kwargs = {k.arg for k in node.keywords}
                sites.append((path.relative_to(root.parent).as_posix(), node.lineno,
                              "expiries" in kwargs))
    return sites


class TestCallSiteCoverage:

    def test_every_call_site_passes_expiries(self):
        sites = _detect_sparse_call_sites()
        assert sites, "expected at least one detect_sparse call site under gex/"
        missing = [(f, ln) for f, ln, ok in sites if not ok]
        assert not missing, f"detect_sparse call sites missing expiries= kwarg: {missing}"

    def test_expected_call_sites_present(self):
        files = {f for f, _ln, _ok in _detect_sparse_call_sites()}
        assert "gex/application/extended.py" in files
        assert "gex/application/moex_service.py" in files
        assert len(_detect_sparse_call_sites()) >= 3
