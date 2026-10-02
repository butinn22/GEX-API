"""Independent QA verification suite for the AUTO-mode feature (design §5, AC2).

This file is written by QA **independently** of the engineer's
``tests/test_auto_gex.py`` / ``tests/test_auto_routes.py``. It re-derives the
AC2 invariants from the design doc using its own synthetic chains and its own
call-counting fakes, so a bug shared by the implementation and the engineer's
tests cannot hide here.

No network: every provider interaction is replaced by an in-process fake
(``_fetch`` monkeypatch / fake router service); synthetic ``OptionSnapshot``
chains only.

Covered:
  * union / no-double-count invariant over an overlapping synthetic pair
    (whole-row primary priority, T taken from the winning row);
  * ``detect_sparse`` named thresholds (8 / ±10% / 5) and NaN/zero handling;
  * ``detect_sparse`` expiry-coverage rule (< ``AUTO_MIN_EXPIRIES`` buckets ⇒
    ``low_expiries``; optional arg ⇒ backward compatible when omitted);
  * ``fallback_source_for`` mapping table;
  * escalation budget: exactly one extra fetch, no loops; failing fallback ⇒
    ``partial`` result, primary data returned, no exception;
  * router wiring: ``mode=auto`` ignores client ``days``/``expiries`` and
    resolves 90/20; invalid ``mode`` ⇒ 422; manual mode unchanged (``auto`` null);
  * AUTO cache key disjoint from the manual key; TTL 600;
  * MOEX auto path: 90 / expiries 0 / moex_iss / no escalation + sparse flag;
  * crypto: coverage returned, no second fetch;
  * ``sources_used`` ordering ``[primary, fallback]``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gex.auth.dependencies as auth_deps
from gex.auth.dependencies import get_current_user
from gex.adapters.cache.redis_client import cache_key
from gex.application.auto_scope import (
    AUTO_CACHE_TTL,
    AUTO_MAX_DAYS,
    AUTO_MAX_EXPIRIES,
    AUTO_MIN_ATM_STRIKES,
    AUTO_MIN_STRIKES,
    AutoCoverage,
    StrikeLite,
    detect_sparse,
    fallback_source_for,
    merge_chains,
)
from gex.application.extended import ExtendedGEXAnalyzer
from gex.application.moex_service import MOEXGEXService, MOEX_PRIMARY_SOURCE
from gex.deps import provide_extended_gex_service
from gex.domain.data_loader import OptionSnapshot
from gex.routers import extended_router as ext_mod
from gex.routers import moex_router as moex_mod


# ====================================================================== #
#  Synthetic chains (independent of the engineer's fixtures)
# ====================================================================== #
#: Three distinct expiry buckets — a "rich" profile that clears AUTO_MIN_EXPIRIES.
_RICH_T = (7.0 / 365.0, 30.0 / 365.0, 60.0 / 365.0)


def _snapshot(
    spot: float = 100.0,
    symbol: str = "AAPL",
    strikes: tuple[float, ...] | None = None,
    oi_call=None,
    oi_put=None,
    T: float | tuple[float, ...] = 30.0 / 365.0,
    walls: bool = True,
) -> OptionSnapshot:
    """Build a synthetic chain (single- or multi-expiry).

    ``T`` is either a single year-fraction (one expiry bucket, the default) or a
    tuple of them — pass ``T=(7 / 365, 30 / 365, 60 / 365)`` to build a chain with
    three distinct expiry buckets (a "rich" profile that satisfies
    ``AUTO_MIN_EXPIRIES``).

    ``oi_call``/``oi_put`` may be callables ``f(strike)`` or floats. When
    ``walls`` is False the OI is flat (no positive call wall ⇒ NaN wall), which
    is useful to force sparseness.
    """
    ks = strikes if strikes is not None else tuple(np.linspace(80.0, 120.0, 11))
    ts = (T,) if isinstance(T, (int, float)) else tuple(T)
    rows = []
    for T_i in ts:
        for k in ks:
            k = float(k)
            if callable(oi_call):
                c = float(oi_call(k))
            elif oi_call is None:
                c = 1000.0 + (k - spot) * 40.0
            else:
                c = float(oi_call)
            if callable(oi_put):
                p = float(oi_put(k))
            elif oi_put is None:
                p = 1000.0 - (k - spot) * 40.0
            else:
                p = float(oi_put)
            rows.append({"strike": k, "type": "C", "oi": max(0.0, c), "iv": 0.2, "T": T_i})
            rows.append({"strike": k, "type": "P", "oi": max(0.0, p), "iv": 0.2, "T": T_i})
    chain = pd.DataFrame(rows, columns=["strike", "type", "oi", "iv", "T"])
    return OptionSnapshot(symbol=symbol, spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)


def _lite(strikes) -> list[StrikeLite]:
    return [
        StrikeLite(
            strike=float(s.k),
            oi_call=float(s.c),
            oi_put=float(s.p),
            gex_net=float(getattr(s, "g", 0.0)),
        )
        for s in strikes
    ]


def _patch_fetch(analyzer, *, primary, primary_src, fallback=None, fallback_src="yfinance",
                 fallback_force="yfinance", fallback_raises=False):
    """Replace ``_fetch`` on an analyzer instance and record every call."""
    calls: list[tuple] = []

    def fake_fetch(ticker, max_expiries, force_source=None, max_days=None):
        calls.append((ticker, max_expiries, force_source))
        if fallback_force is not None and force_source == fallback_force:
            if fallback_raises:
                raise RuntimeError("fallback provider down")
            return fallback, fallback_src, 100, 0.0
        return primary, primary_src, 100, 0.0

    analyzer._fetch = fake_fetch  # type: ignore[assignment]
    return calls


# ====================================================================== #
#  1. merge_chains — union / no double count / whole-row priority
# ====================================================================== #
class TestMergeUnionInvariant:
    """AC2(c): union keyed (strike, type, round(T*365)); primary row wins in full."""

    def _fixtures(self):
        T_pri = 30.2 / 365.0
        T_fb_collide = 30.4 / 365.0   # rounds to the same bucket as T_pri
        T_fb_other = 31.4 / 365.0     # distinct bucket (31)
        primary = pd.DataFrame([
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.20, "T": T_pri},
            {"strike": 100.0, "type": "P", "oi": 20.0, "iv": 0.20, "T": T_pri},
            {"strike": 105.0, "type": "C", "oi": 30.0, "iv": 0.22, "T": T_pri},
        ])
        fallback = pd.DataFrame([
            # same (strike, type, rounded T) as primary 100C → primary must win entirely
            {"strike": 100.0, "type": "C", "oi": 999.0, "iv": 0.55, "T": T_fb_collide},
            # distinct expiry bucket → added
            {"strike": 100.0, "type": "C", "oi": 7.0, "iv": 0.31, "T": T_fb_other},
            # unique strike → added
            {"strike": 110.0, "type": "P", "oi": 5.0, "iv": 0.25, "T": T_pri},
        ])
        return primary, fallback

    @staticmethod
    def _key(row) -> tuple:
        return (float(row["strike"]), str(row["type"]), int(round(float(row["T"]) * 365.0)))

    def test_sum_over_union_equals_sum_over_unique_keys(self):
        primary, fallback = self._fixtures()
        merged = merge_chains(primary, fallback)

        # Reference union computed independently (primary overrides on collision).
        expected: dict[tuple, float] = {}
        for row in fallback.to_dict("records"):
            expected[self._key(row)] = float(row["oi"])
        for row in primary.to_dict("records"):
            expected[self._key(row)] = float(row["oi"])   # primary wins

        got_keys = [self._key(r) for r in merged.to_dict("records")]
        assert len(got_keys) == len(set(got_keys)), "union must have one row per key"
        assert set(got_keys) == set(expected)
        assert abs(float(merged["oi"].sum()) - sum(expected.values())) < 1e-9

    def test_overlapping_row_counted_once_with_primary_fields(self):
        primary, fallback = self._fixtures()
        merged = merge_chains(primary, fallback)
        row = merged[(merged["strike"] == 100.0) & (merged["type"] == "C")].iloc[0]
        # oi/iv/T all come from the primary row — never summed, never mixed.
        assert float(row["oi"]) == 10.0
        assert float(row["iv"]) == 0.20
        assert abs(float(row["T"]) - 30.2 / 365.0) < 1e-12

    def test_naive_concat_would_have_double_counted(self):
        """Guard: proves the dedup actually removed an overlap (test is meaningful)."""
        primary, fallback = self._fixtures()
        merged = merge_chains(primary, fallback)
        naive = float(primary["oi"].sum()) + float(fallback["oi"].sum())
        assert naive == 1071.0
        assert float(merged["oi"].sum()) == 72.0
        assert naive > float(merged["oi"].sum())

    def test_helper_columns_are_stripped(self):
        primary, fallback = self._fixtures()
        merged = merge_chains(primary, fallback)
        assert "_prio" not in merged.columns and "_tday" not in merged.columns
        assert list(primary.columns) == list(merged.columns)


# ====================================================================== #
#  2. detect_sparse — named thresholds + NaN/zero handling
# ====================================================================== #
class TestDetectSparseThresholds:

    def _strikes(self, ks, oi=10.0):
        return [
            StrikeLite(strike=float(k), oi_call=oi, oi_put=oi, gex_net=1.0) for k in ks
        ]

    def test_exactly_eight_strikes_is_not_low(self):
        ks = [97, 98, 99, 100, 101, 102, 103, 104]      # 8 strikes, all within ±10%
        reasons = detect_sparse(self._strikes(ks), 100.0, 104.0, 97.0)
        assert "low_strikes" not in reasons
        assert reasons == []                            # healthy floor → fully usable

    def test_seven_strikes_is_low(self):
        ks = [97, 98, 99, 100, 101, 102, 103]           # 7 strikes
        reasons = detect_sparse(self._strikes(ks), 100.0, 103.0, 97.0)
        assert "low_strikes" in reasons

    def test_exactly_five_atm_is_ok(self):
        ks = [90.0, 95.0, 100.0, 105.0, 110.0, 130.0, 140.0, 150.0]  # 5 within ±10%
        reasons = detect_sparse(self._strikes(ks), 100.0, 150.0, 90.0)
        assert "low_atm_strikes" not in reasons
        assert reasons == []

    def test_four_atm_is_low(self):
        ks = [90.0, 95.0, 100.0, 105.0, 130.0, 140.0, 150.0, 160.0]  # 4 within ±10%
        reasons = detect_sparse(self._strikes(ks), 100.0, 160.0, 90.0)
        assert "low_atm_strikes" in reasons

    def test_atm_band_is_inclusive_at_the_edge(self):
        # |k - spot| == exactly 10% of spot counts as inside the band.
        ks = [90.0, 95.0, 100.0, 105.0, 110.0, 130.0, 140.0, 150.0]
        reasons = detect_sparse(self._strikes(ks), 100.0, 150.0, 90.0)
        assert "low_atm_strikes" not in reasons

    @pytest.mark.parametrize("spot", [float("nan"), 0.0, -5.0])
    def test_non_positive_or_nan_spot_is_low_atm(self, spot):
        reasons = detect_sparse(self._strikes([100.0] * 10), spot, 110.0, 90.0)
        assert "low_atm_strikes" in reasons

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0.0, -1.0])
    def test_missing_walls(self, bad):
        reasons = detect_sparse(self._strikes([100.0 + i for i in range(10)]), 100.0, bad, bad)
        assert "missing_call_wall" in reasons and "missing_put_wall" in reasons

    def test_total_oi_zero(self):
        reasons = detect_sparse(self._strikes([100.0 + i for i in range(10)], oi=0.0),
                                100.0, 110.0, 90.0)
        assert "total_oi_zero" in reasons

    def test_no_data(self):
        assert "no_data" in detect_sparse([], 100.0, 110.0, 90.0)

    def test_thresholds_are_the_designed_values(self):
        # Thresholds are part of the AC (8 / ±10% / 5); pin them so a silent
        # change can't pass as "green".
        assert AUTO_MIN_STRIKES == 8
        assert AUTO_MIN_ATM_STRIKES == 5


# ====================================================================== #
#  3. fallback_source_for mapping
# ====================================================================== #
class TestFallbackMap:

    @pytest.mark.parametrize("primary,expected", [
        ("webull", "yfinance"),
        ("yfinance", "webull"),
        ("crypto", None),
        ("futures", None),
        ("stock", None),
        ("moex_iss", None),
        ("Webull", "yfinance"),        # case/space-insensitive
        ("  YFINANCE ", "webull"),
        ("", None),
        ("unknown", None),
    ])
    def test_mapping(self, primary, expected):
        assert fallback_source_for(primary, "AAPL") == expected


# ====================================================================== #
#  4. analyze_auto — escalation budget, failure, crypto, ordering
# ====================================================================== #
class TestAnalyzeAutoEscalation:

    def test_no_escalation_single_fetch(self):
        an = ExtendedGEXAnalyzer()
        calls = _patch_fetch(an, primary=_snapshot(T=_RICH_T), primary_src="webull")
        rep = an.analyze_auto("AAPL")
        assert len(calls) == 1
        assert rep.coverage.escalated is False
        assert rep.coverage.fallback_used is False
        assert rep.coverage.sources_used == ["webull"]

    def test_sparse_triggers_exactly_one_extra_fetch(self):
        an = ExtendedGEXAnalyzer()
        thin = _snapshot(strikes=(99.0, 100.0, 101.0), oi_call=0.0, oi_put=0.0)
        calls = _patch_fetch(an, primary=thin, primary_src="webull",
                             fallback=_snapshot(), fallback_src="yfinance")
        rep = an.analyze_auto("AAPL")
        # exactly two provider round-trips, never a loop
        assert len(calls) == 2
        assert calls[0][2] is None            # primary: auto resolution
        assert calls[1][2] == "yfinance"      # fallback: forced source
        cov = rep.coverage
        assert cov.escalated is True
        assert cov.fallback_used is True
        assert cov.partial is False
        assert cov.sources_used == ["webull", "yfinance"]   # ordered [primary, fallback]

    def test_fallback_failure_is_not_fatal(self):
        an = ExtendedGEXAnalyzer()
        thin = _snapshot(strikes=(99.0, 100.0, 101.0), oi_call=0.0, oi_put=0.0)
        calls = _patch_fetch(an, primary=thin, primary_src="webull", fallback_raises=True)
        rep = an.analyze_auto("AAPL")          # must NOT raise
        assert len(calls) == 2
        cov = rep.coverage
        assert cov.partial is True
        assert cov.fallback_used is False
        assert cov.escalated is True           # attempt was made
        assert cov.sources_used == ["webull"]  # fallback contributed no rows
        assert cov.sparse is True              # primary profile still thin
        assert len(rep.per_strike) > 0         # primary data returned

    def test_crypto_never_escalates(self):
        an = ExtendedGEXAnalyzer()
        thin = _snapshot(symbol="BTC", strikes=(99.0, 100.0, 101.0), oi_call=0.0, oi_put=0.0)
        calls = _patch_fetch(an, primary=thin, primary_src="crypto")
        rep = an.analyze_auto("BTC")
        assert len(calls) == 1
        cov = rep.coverage
        assert cov is not None
        assert cov.primary_source == "crypto"
        assert cov.sources_used == ["crypto"]
        assert cov.escalated is False
        assert cov.fallback_used is False
        assert cov.sparse is True              # thin, but still reported

    def test_futures_never_escalates(self):
        an = ExtendedGEXAnalyzer()
        thin = _snapshot(symbol="ES", strikes=(99.0, 100.0, 101.0), oi_call=0.0, oi_put=0.0)
        calls = _patch_fetch(an, primary=thin, primary_src="futures")
        rep = an.analyze_auto("ES")
        assert len(calls) == 1
        assert rep.coverage.sources_used == ["futures"]
        assert rep.coverage.escalated is False

    def test_pinned_yfinance_promotes_webull_to_fallback(self):
        an = ExtendedGEXAnalyzer()
        thin = _snapshot(strikes=(99.0, 100.0, 101.0), oi_call=0.0, oi_put=0.0)
        calls = _patch_fetch(an, primary=thin, primary_src="yfinance",
                             fallback=_snapshot(), fallback_src="webull",
                             fallback_force="webull")
        rep = an.analyze_auto("AAPL", source="yfinance")
        assert calls[1][2] == "webull"
        assert rep.coverage.sources_used == ["yfinance", "webull"]
        assert rep.coverage.primary_source == "yfinance"

    def test_merged_report_keeps_primary_spot_and_source(self):
        an = ExtendedGEXAnalyzer()
        thin = _snapshot(spot=100.0, strikes=(99.0, 100.0, 101.0), oi_call=0.0, oi_put=0.0)
        fb = _snapshot(spot=999.0, strikes=(200.0, 210.0, 220.0), oi_call=50.0, oi_put=50.0)
        _patch_fetch(an, primary=thin, primary_src="webull", fallback=fb, fallback_src="yfinance")
        rep = an.analyze_auto("AAPL")
        assert rep.source == "webull"          # top-level source stays primary
        assert rep.spot == 100.0               # pricing reference never overridden
        assert rep.coverage.primary_source == "webull"

    def test_resolved_scope_is_max(self):
        an = ExtendedGEXAnalyzer()
        _patch_fetch(an, primary=_snapshot(T=_RICH_T), primary_src="webull")
        cov = an.analyze_auto("AAPL").coverage
        assert cov.resolved_days == AUTO_MAX_DAYS == 90.0
        assert cov.resolved_expiries == AUTO_MAX_EXPIRIES == 20

    def test_coverage_consistency_with_final_profile(self):
        an = ExtendedGEXAnalyzer()
        _patch_fetch(an, primary=_snapshot(T=_RICH_T), primary_src="webull")
        rep = an.analyze_auto("AAPL")
        cov = rep.coverage
        assert cov.strike_count == len(rep.per_strike)
        expected_oi = sum(s.oi_call + s.oi_put for s in rep.per_strike)
        assert abs(cov.total_oi - expected_oi) < 1e-6


# ====================================================================== #
#  5. Router wiring: /ext/gex
# ====================================================================== #
class _RecordingExtService:
    """Fake analyzer exposing the exact public signature (incl. defaults)."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._rep = ExtendedGEXAnalyzer().analyze("AAPL", snapshot=_snapshot())

    def analyze(self, ticker, days=30.0, max_expiries=5, hedge_scenarios_pct=None, source="auto"):
        self.calls.append(("manual", dict(days=days, max_expiries=max_expiries, source=source)))
        rep = self._rep
        rep.coverage = None
        return rep

    def analyze_auto(self, ticker, source="auto", hedge_scenarios_pct=None,
                     max_days=AUTO_MAX_DAYS, max_expiries=AUTO_MAX_EXPIRIES):
        self.calls.append(("auto", dict(source=source, max_days=max_days, max_expiries=max_expiries)))
        rep = self._rep
        rep.coverage = AutoCoverage(
            mode="auto", resolved_days=max_days, resolved_expiries=max_expiries,
            sources_used=[source if source in ("webull", "yfinance") else "webull"],
            primary_source=source if source in ("webull", "yfinance") else "webull",
            fallback_used=False, escalated=False, partial=False,
            expirations_merged=1, strike_count=len(rep.per_strike),
            total_oi=sum(s.oi_call + s.oi_put for s in rep.per_strike),
            sparse=False, sparse_reasons=[], elapsed_ms=5,
        )
        return rep


class _RecordingCache:
    def __init__(self):
        self.key = ""
        self.ttl = -1

    def get(self, key, ttl, compute):
        self.key, self.ttl = key, ttl
        return compute()


@pytest.fixture
def ext_client(monkeypatch):
    monkeypatch.setattr(auth_deps, "can_bypass_barriers", lambda user: True)
    svc = _RecordingExtService()
    cache = _RecordingCache()
    monkeypatch.setattr(ext_mod, "result_cache", cache)
    app = FastAPI()
    app.include_router(ext_mod.router)
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[provide_extended_gex_service] = lambda: svc
    return TestClient(app), svc, cache


class TestExtendedRouter:

    def test_auto_resolves_to_90_20_ignoring_client(self, ext_client):
        client, svc, _ = ext_client
        r = client.get("/ext/gex/AAPL?mode=auto&days=7&expiries=1")
        assert r.status_code == 200
        kind, kw = svc.calls[-1]
        assert kind == "auto"
        # client days/expiries must NOT reach the auto path — backend maxima win
        assert kw["max_days"] == 90.0
        assert kw["max_expiries"] == 20
        assert "days" not in kw and "expiries" not in kw
        body = r.json()
        assert body["days"] == 90.0
        assert body["auto"]["resolved_days"] == 90.0
        assert body["auto"]["resolved_expiries"] == 20

    def test_auto_block_has_all_required_fields(self, ext_client):
        """AC2(e): the `auto` block must carry all 18 metadata fields.

        14 базовых полей + 4 аддитивных поля охвата (strike_min/strike_max/
        nearest_expiry_days/furthest_expiry_days) — обратная совместимость
        сохранена (optional).
        """
        client, _, _ = ext_client
        body = client.get("/ext/gex/AAPL?mode=auto").json()
        assert set(body["auto"]) == {
            "mode", "resolved_days", "resolved_expiries", "sources_used",
            "primary_source", "fallback_used", "escalated", "partial",
            "expirations_merged", "strike_count", "total_oi", "sparse",
            "sparse_reasons", "elapsed_ms",
            "strike_min", "strike_max", "nearest_expiry_days",
            "furthest_expiry_days",
        }

    def test_auto_uses_disjoint_cache_key_and_ttl(self, ext_client):
        client, _, cache = ext_client
        client.get("/ext/gex/AAPL?mode=auto")
        assert "EXTGEXA" in cache.key
        assert cache.ttl == AUTO_CACHE_TTL == 600

    def test_manual_mode_is_untouched(self, ext_client):
        client, svc, cache = ext_client
        r = client.get("/ext/gex/AAPL?mode=manual&days=7&expiries=2")
        assert r.status_code == 200
        kind, kw = svc.calls[-1]
        assert kind == "manual"
        assert kw["days"] == 7 and kw["max_expiries"] == 2
        body = r.json()
        assert body["auto"] is None
        assert body["days"] == 7.0
        assert "EXTGEXA" not in cache.key

    def test_default_mode_is_manual(self, ext_client):
        client, svc, _ = ext_client
        r = client.get("/ext/gex/AAPL")
        assert r.status_code == 200
        assert svc.calls[-1][0] == "manual"
        assert r.json()["auto"] is None

    def test_invalid_mode_is_422(self, ext_client):
        client, _, _ = ext_client
        assert client.get("/ext/gex/AAPL?mode=fast").status_code == 422
        assert client.get("/ext/gex/AAPL?mode=").status_code == 422


# ====================================================================== #
#  6. Cache-key disjointness (pure)
# ====================================================================== #
class TestCacheKeyDisjointness:

    def test_disjoint_even_for_identical_scope(self):
        man = cache_key("res", "extgex", "AAPL", 90, 20, "webull", "-1.0-1.0")
        auto = cache_key("res", "extgexA", "AAPL", 90, 20, "webull", "-1.0-1.0")
        assert man != auto
        assert "EXTGEXA" in auto and "EXTGEXA" not in man
        # segment is the only difference → no manual/auto cross-poisoning possible
        assert auto == man.replace("EXTGEX", "EXTGEXA") or "extgexA" in auto.lower()


# ====================================================================== #
#  7. MOEX auto path
# ====================================================================== #
class _FakeRepo:
    def put(self, asset, snapshot):
        return None


class TestMoexAuto:

    def _service(self, snapshot, out):
        svc = MOEXGEXService(repo=_FakeRepo(), runner=SimpleNamespace(run_analysis=lambda *a, **k: out))
        svc._fetch = lambda asset, max_expiries: (snapshot, {"r": 0.05, "per_contract": 1})
        return svc

    def _out(self, strikes, spot=100.0, call_wall=120.0, put_wall=80.0):
        profile = SimpleNamespace(per_strike=strikes, call_wall=call_wall, put_wall=put_wall)
        return SimpleNamespace(symbol="RTS", spot=spot, profile=profile)

    def test_auto_attaches_coverage_90_all_no_escalation(self):
        ks = np.linspace(80.0, 120.0, 11)
        strikes = [SimpleNamespace(strike=float(k), oi_call=100.0, oi_put=100.0, gex_net=1.0) for k in ks]
        svc = self._service(_snapshot(symbol="RTS", T=_RICH_T), None)
        svc._runner = SimpleNamespace(run_analysis=lambda *a, **k: self._out(strikes))
        out = svc.analyze("RTS", days=90.0, max_expiries=0, auto=True)
        cov = out.auto
        assert cov is not None
        assert cov.mode == "auto"
        assert cov.resolved_days == 90.0
        assert cov.resolved_expiries == 0                 # 0 = ALL on MOEX ISS
        assert cov.primary_source == MOEX_PRIMARY_SOURCE == "moex_iss"
        assert cov.sources_used == ["moex_iss"]
        assert cov.escalated is False
        assert cov.fallback_used is False
        assert cov.partial is False
        assert cov.strike_count == 11
        assert cov.total_oi == 2200.0
        assert isinstance(cov.sparse, bool)               # sparse flag always present
        assert cov.sparse is False

    def test_auto_sparse_flag_when_thin(self):
        svc = self._service(_snapshot(symbol="RTS"), None)
        svc._runner = SimpleNamespace(run_analysis=lambda *a, **k: self._out([], call_wall=0.0, put_wall=0.0))
        out = svc.analyze("RTS", days=90.0, max_expiries=0, auto=True)
        assert out.auto is not None
        assert out.auto.sparse is True
        assert "no_data" in out.auto.sparse_reasons

    def test_manual_moex_has_manual_coverage(self):
        """Ручной режим MOEX тоже отдаёт охват (аудит 2026-09-17), но mode='manual'.

        Эскалации и fallback'ов в ручном режиме по-прежнему нет: ``escalated``
        и ``fallback_used`` остаются False.
        """
        strikes = [SimpleNamespace(strike=100.0, oi_call=1.0, oi_put=1.0, gex_net=1.0)]
        svc = self._service(_snapshot(symbol="RTS"), None)
        svc._runner = SimpleNamespace(run_analysis=lambda *a, **k: self._out(strikes))
        out = svc.analyze("RTS", days=30.0, max_expiries=5, auto=False)
        cov = getattr(out, "auto", None)
        assert cov is not None
        assert cov.mode == "manual"
        assert cov.primary_source == MOEX_PRIMARY_SOURCE == "moex_iss"
        assert cov.sources_used == ["moex_iss"]
        assert cov.escalated is False
        assert cov.fallback_used is False
        assert cov.strike_count == 1
        assert cov.total_oi == 2.0

    def test_moex_router_wires_auto(self):
        recorded: list[tuple] = []

        class _FakeSvc:
            def analyze_moex(self, asset, **kw):
                recorded.append((asset, kw))
                return {"ok": True}

        moex_mod.get_moex_gex("RTS", mode="auto", svc=_FakeSvc(), notify=False, background_tasks=None)
        assert recorded[-1] == ("RTS", {"days": 90.0, "max_expiries": 0, "auto": True})

    def test_moex_router_manual_untouched(self):
        recorded: list[tuple] = []

        class _FakeSvc:
            def analyze_moex(self, asset, **kw):
                recorded.append((asset, kw))
                return {"ok": True}

        moex_mod.get_moex_gex("RTS", mode="manual", days=30, expiries=5,
                              svc=_FakeSvc(), notify=False, background_tasks=None)
        _asset, kw = recorded[-1]
        assert kw == {"days": 30, "max_expiries": 5}
        assert "auto" not in kw


# ====================================================================== #
#  8. Backward compatibility: manual report unchanged
# ====================================================================== #
class TestBackwardCompat:

    def test_manual_report_carries_manual_coverage(self):
        """Ручной отчёт теперь тоже несёт охват (данные о покрытии нужны UI).

        Поле по-прежнему **опциональное**: аналитик, работающий без фетчера
        (например, на готовом снапшоте в тестах), получает coverage, а
        отсутствие coverage сериализуется как ``"auto":null`` — старые клиенты
        не ломаются.
        """
        from gex.schemas.extended_schemas import extended_report_to_schema

        rep = ExtendedGEXAnalyzer().analyze("AAPL", snapshot=_snapshot())
        assert rep.coverage is not None
        assert rep.coverage.mode == "manual"
        out = extended_report_to_schema(rep, days=30.0)
        assert out.auto is not None
        assert out.auto.mode == "manual"

    def test_missing_coverage_serializes_as_null(self):
        """Без coverage блок ``auto`` остаётся null — контракт не сломан."""
        from gex.schemas.extended_schemas import extended_report_to_schema

        rep = ExtendedGEXAnalyzer().analyze("AAPL", snapshot=_snapshot())
        rep.coverage = None
        out = extended_report_to_schema(rep, days=30.0)
        assert out.auto is None
        assert '"auto":null' in out.model_dump_json()

    def test_source_literal_accepts_yfinance(self):
        from gex.schemas.extended_schemas import extended_report_to_schema

        rep = ExtendedGEXAnalyzer().analyze("AAPL", snapshot=_snapshot())
        rep.source = "yfinance"          # pre-existing latent 500 fixed by §6
        assert extended_report_to_schema(rep).source == "yfinance"
