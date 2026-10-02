"""Tests for the /breadth-imoex feature (MOEX ISS breadth).

Covers, incrementally:
  * orchestrator registration (data type, TTLs, daily quota rule);
  * ISS payload parsing (universe, prefs filter, top-10, completed-day filter);
  * breadth computation (EW basket, EMA breadth, McClellan A/D, diagnosis);
  * service fallback (Redis -> file snapshot);
  * orchestrator force-refresh and 24h quota;
  * API route behavior (no data -> 503, cached -> 200, no external fetch).
"""
from __future__ import annotations

from gex.orchestrator.cache import default_ttl_for
from gex.orchestrator.constants import DATA_TYPES
from gex.orchestrator.rate_limiter import default_rate_rules


# ----------------------------------------------------------------------
# Increment A: orchestrator registration
# ----------------------------------------------------------------------
class TestOrchestratorRegistration:
    def test_imoex_breadth_data_type_is_registered(self):
        assert "imoex_breadth" in DATA_TYPES

    def test_imoex_breadth_default_ttls(self):
        fresh, stale = default_ttl_for("imoex_breadth", {})
        assert fresh == 6 * 3600
        assert stale == 48 * 3600

    def test_imoex_breadth_daily_quota_rule(self):
        rules = default_rate_rules()
        rule = next(
            (r for r in rules if r.provider == "iss" and r.endpoint_pattern == "imoex_breadth"),
            None,
        )
        assert rule is not None, "expected iss:imoex_breadth daily quota rule"
        assert rule.window_seconds == 86400
        assert rule.max_requests == 3
        assert rule.scope == "endpoint"



# ----------------------------------------------------------------------
# Increment B: ISS payload parsing
# ----------------------------------------------------------------------
import pandas as pd  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from gex.adapters.fetchers.imoex_breadth_fetcher import (  # noqa: E402
    FALLBACK_UNIVERSE,
    drop_forming_session,
    parse_candles_block,
    parse_universe_payload,
    top10_secids,
)


def _analytics_payload() -> dict:
    return {
        "analytics": {
            "columns": [
                "indexid", "ticker", "secid", "shortnames", "weight", "capitalization",
            ],
            "data": [
                ["IMOEX", "SBER", "SBER", "Сбербанк", 14.2, 5_200_000_000_000],
                ["IMOEX", "SBERP", "SBERP", "Сбербанк-п", 1.1, 400_000_000_000],
                ["IMOEX", "LKOH", "LKOH", "ЛУКОЙЛ", 13.8, 4_900_000_000_000],
                ["IMOEX", "GAZP", "GAZP", "ГАЗПРОМ ао", 12.9, 4_200_000_000_000],
                ["IMOEX", "SNGSP", "SNGSP", "Сургутнефтегаз-п", 2.0, 500_000_000_000],
                ["IMOEX", "TATN", "TATN", "Татнефть им. В.Д. Шашина", 4.1, 1_300_000_000_000],
                ["IMOEX", "TATNP", "TATNP", "Татнефть 3 ап", 1.2, 300_000_000_000],
                ["IMOEX", "TRNFP", "TRNFP", "Транснефть ап", 1.5, 200_000_000_000],
            ],
        }
    }


class TestIssParsing:
    def test_parse_universe_payload_filters_preferred_shares(self):
        universe = parse_universe_payload(_analytics_payload())
        secids = [u["secid"] for u in universe]
        assert "SBER" in secids
        assert "LKOH" in secids
        assert "TATN" in secids
        for pref in ("SBERP", "SNGSP", "TATNP", "TRNFP"):
            assert pref not in secids

    def test_parse_universe_payload_accepts_lowercase_columns(self):
        payload = {"analytics": {"columns": ["secid", "shortnames", "weight", "capitalization"],
                                 "data": [["sber", "Сбербанк", "13.4", "4,5E12"]]}}
        universe = parse_universe_payload(payload)
        assert universe == [{"secid": "SBER", "weight": 13.4, "capitalization": 0.0}]

    def test_top10_secids_sorted_by_capitalization(self):
        universe = parse_universe_payload(_analytics_payload())
        top = top10_secids(universe)
        assert top[0] == "SBER"
        assert top[1] == "LKOH"
        assert top[2] == "GAZP"
        assert "SBERP" not in top

    def test_top10_secids_falls_back_to_fixed_list(self):
        top = top10_secids([])
        assert len(top) >= 10
        assert top[:3] == ["SBER", "LKOH", "GAZP"]

    def test_top10_secids_uses_fixed_order_without_caps_or_weights(self):
        universe = [{"secid": s, "weight": 0.0, "capitalization": 0.0}
                    for s in ("GAZP", "LKOH", "SBER", "ROSN", "GMKN", "NVTK", "TATN", "PLZL", "YDEX", "MOEX")]
        top = top10_secids(universe)
        assert top[:3] == ["SBER", "LKOH", "GAZP"]

    def test_fallback_universe_has_no_preferred_shares(self):
        secids = [u["secid"] for u in FALLBACK_UNIVERSE]
        for pref in ("SBERP", "SNGSP", "TATNP", "TRNFP"):
            assert pref not in secids

    def test_parse_candles_block_builds_ohlcv_dataframe(self):
        payload = {
            "candles": {
                "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
                "data": [
                    [100.0, 101.0, 102.0, 99.0, 1000.0, 100.0, "2026-09-08 10:00:00", "2026-09-08 18:50:00"],
                    [101.0, 102.0, 103.0, 100.0, 1100.0, 110.0, "2026-09-09 10:00:00", "2026-09-09 18:50:00"],
                ],
            }
        }
        df = parse_candles_block(payload)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 2
        assert float(df["Close"].iloc[-1]) == 102.0

    def test_drop_forming_session_drops_today_before_close_only(self):
        idx = pd.to_datetime(
            ["2026-09-08 10:00:00", "2026-09-09 10:00:00"],
            format="mixed",
        )
        df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
        now_noon = datetime(2026, 9, 9, 12, 0, tzinfo=timezone(timedelta(hours=3)))
        trimmed = drop_forming_session(df, now_msk=now_noon)
        assert len(trimmed) == 1
        assert trimmed["Close"].iloc[-1] == 100.0

        now_night = datetime(2026, 9, 9, 23, 0, tzinfo=timezone(timedelta(hours=3)))
        kept = drop_forming_session(df, now_msk=now_night)
        assert len(kept) == 2
        assert kept["Close"].iloc[-1] == 101.0


class TestGenericTqbrFetch:
    def test_fetch_constituent_closes_uses_generic_tqbr_endpoint(self, monkeypatch):
        from gex.adapters.fetchers.imoex_breadth_fetcher import ImoexBreadthFetcher

        fetcher = ImoexBreadthFetcher()
        fetched: list = []
        idx = pd.bdate_range("2025-01-02", periods=300, tz="UTC")

        def fake_daily(url, secid):
            fetched.append((url, secid))
            return pd.DataFrame({"Open": [100.0] * 300, "High": [101.0] * 300, "Low": [99.0] * 300, "Close": list(range(300)), "Volume": [1000.0] * 300}, index=idx)

        monkeypatch.setattr(fetcher, "_fetch_daily_candles", fake_daily)
        series = fetcher.fetch_constituent_closes(["SBER", "XYZNEW"])

        assert set(series.keys()) == {"SBER", "XYZNEW"}
        assert [s for _, s in fetched] == ["SBER", "XYZNEW"]
        assert all("TQBR" in url and "stock" in url for url, _ in fetched)

    def test_fetch_constituent_ohlc_returns_full_ohlcv(self, monkeypatch):
        from gex.adapters.fetchers.imoex_breadth_fetcher import ImoexBreadthFetcher

        fetcher = ImoexBreadthFetcher()
        idx = pd.bdate_range("2025-01-02", periods=300, tz="UTC")
        df = pd.DataFrame({"Open": [100.0] * 300, "High": [101.0] * 300, "Low": [99.0] * 300,
                           "Close": [100.5] * 300, "Volume": [1000.0] * 300}, index=idx)
        monkeypatch.setattr(fetcher, "_fetch_daily_candles", lambda url, secid: df)
        frames = fetcher.fetch_constituent_ohlc(["SBER"])
        assert set(frames) == {"SBER"}
        assert list(frames["SBER"].columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_fetch_rvi_daily_uses_sndx_index_endpoint(self, monkeypatch):
        from gex.adapters.fetchers.imoex_breadth_fetcher import ImoexBreadthFetcher

        fetcher = ImoexBreadthFetcher()
        calls: list = []
        idx = pd.bdate_range("2025-01-02", periods=300, tz="UTC")
        df = pd.DataFrame({"Open": [40.0] * 300, "High": [41.0] * 300, "Low": [39.0] * 300,
                           "Close": [40.0] * 300, "Volume": [0.0] * 300}, index=idx)

        def fake_daily(url, secid):
            calls.append((url, secid))
            return df

        monkeypatch.setattr(fetcher, "_fetch_daily_candles", fake_daily)
        out = fetcher.fetch_rvi_daily()
        assert (calls[0][1]) == "RVI"
        assert "markets/index" in calls[0][0] and "SNDX" in calls[0][0]
        assert len(out) == 300

    def test_fetch_payload_includes_ohlc_and_rvi(self, monkeypatch):
        from gex.adapters.fetchers.imoex_breadth_fetcher import FALLBACK_UNIVERSE, ImoexBreadthFetcher

        fetcher = ImoexBreadthFetcher()
        idx = pd.bdate_range("2025-01-02", periods=300, tz="UTC")
        df = pd.DataFrame({"Open": [100.0] * 300, "High": [101.0] * 300, "Low": [99.0] * 300,
                           "Close": [100.5] * 300, "Volume": [1000.0] * 300}, index=idx)
        monkeypatch.setattr(fetcher, "fetch_universe", lambda: [dict(u) for u in FALLBACK_UNIVERSE])
        monkeypatch.setattr(fetcher, "fetch_index_daily", lambda: df)
        monkeypatch.setattr(fetcher, "fetch_constituent_ohlc", lambda secids: {s: df for s in secids})
        monkeypatch.setattr(fetcher, "fetch_rvi_daily", lambda: df)

        payload = fetcher.fetch_payload()
        assert "rvi" in payload and payload["rvi"]["bars"]
        assert payload["stocks"]
        first_bar = payload["stocks"][0]["bars"][0]
        assert "h" in first_bar and "l" in first_bar and "c" in first_bar





# ----------------------------------------------------------------------
# Increment C: service computation + fallback storage
# ----------------------------------------------------------------------
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gex.application.breadth_imoex_service import (  # noqa: E402
    compute_result,
    get_latest,
    store_latest,
)


@pytest.fixture
def snapshot_dir():
    """Локальная временная папка внутри репозитория (tmp_path заблокирован песочницей)."""
    directory = Path(__file__).parent / "_tmp_imoex_breadth"
    directory.mkdir(exist_ok=True)
    yield directory
    for item in directory.glob("*.json"):
        try:
            item.unlink()
        except OSError:
            pass


def _synthetic_raw_payload(n_days: int = 320, n_stocks: int = 6) -> dict:
    """IMOEX + N акций с детерминированными дневными доходностями."""
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    imoex_close = 1000.0 * np.cumprod(np.full(n_days, 1.01))
    stocks = []
    for i in range(n_stocks):
        r = 0.005 + i * 0.0005
        closes = 100.0 * np.cumprod(np.full(n_days, 1.0 + r))
        bars = [{"t": d.isoformat(), "c": float(c)} for d, c in zip(dates, closes)]
        stocks.append({"symbol": f"T{i}", "bars": bars})
    imoex_bars = [{"t": d.isoformat(), "c": float(c)} for d, c in zip(dates, imoex_close)]
    return {
        "universe": [{"secid": f"T{i}", "weight": 1.0, "capitalization": 1.0} for i in range(n_stocks)],
        "top10": [f"T{i}" for i in range(min(n_stocks, 10))],
        "imoex": {"bars": imoex_bars},
        "stocks": stocks,
    }


def _flat_ohlc_raw_payload(n_days: int = 320, n_stocks: int = 6, *, with_rvi: bool = True) -> dict:
    """Бумаги с постоянным True Range=4 (H=102,L=98,C=100) + RVI=40."""
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    stocks = []
    for i in range(n_stocks):
        bars = [
            {"t": d.isoformat(), "o": 100.0, "h": 102.0, "l": 98.0, "c": 100.0, "v": 1000.0}
            for d in dates
        ]
        stocks.append({"symbol": f"T{i}", "bars": bars})
    imoex_bars = [{"t": d.isoformat(), "c": 3000.0} for d in dates]
    payload = {
        "universe": [{"secid": f"T{i}", "weight": 1.0, "capitalization": 1.0} for i in range(n_stocks)],
        "top10": [f"T{i}" for i in range(min(n_stocks, 10))],
        "imoex": {"bars": imoex_bars},
        "stocks": stocks,
    }
    if with_rvi:
        payload["rvi"] = {"bars": [{"t": d.isoformat(), "c": 40.0} for d in dates]}
    return payload


class TestComputeResult:
    def test_compute_result_shape_and_ew_basket(self):
        out = compute_result(_synthetic_raw_payload(), warmup=20)
        market = out["market"]
        assert market["benchmark"] == "IMOEX"
        assert market["leaders"] == "TOP10"
        assert len(market["dates"]) >= 280
        assert len(market["imoex"]) == len(market["dates"])
        assert len(market["ew"]) == len(market["dates"])
        # EW-корзина стартует со 100 и растёт (средняя доходность 1.75%/день).
        assert market["ew"][0] == 100.0
        assert market["ew"][-1] > market["ew"][0]
        assert market["rel_ew"][0] is not None
        assert market["rel_leaders"][0] is not None
        assert out["stocks"]["n"] == 6
        assert out["meta"]["last_completed_day"] == market["dates"][-1]
        assert out["meta"]["source"] == "iss"

    def test_compute_result_ema_and_mcclellan(self):
        out = compute_result(_synthetic_raw_payload(), warmup=20)
        stocks = out["stocks"]
        assert stocks["n"] == 6
        # Все 6 бумаг растут — после прогрева доля выше EMA близка к 100%.
        assert stocks["above20"][-1] == 100.0
        assert stocks["above50"][-1] == 100.0
        assert stocks["above200"][-1] == 100.0
        # Постоянный A/D-импульс: осциллятор сходится к нулю, суммация ~10000.
        assert abs(stocks["mcc_osc"][-1]) < 0.05
        assert abs(stocks["mcc_sum"][-1] - 10000.0) < 0.5
        # Диагноз: IMOEX растёт >0.5%/5д, EW и Топ-10 совпадают -> UPCOHER.
        assert out["current"]["state"] == "UPCOHER"
        assert out["current"]["imoex_5d_pct"] is not None

    def test_compute_result_raises_without_index(self):
        import pytest
        with pytest.raises(ValueError):
            compute_result({"imoex": {"bars": []}, "stocks": [], "top10": []})

    def test_compute_result_volatility_atr_and_rvi(self):
        out = compute_result(_flat_ohlc_raw_payload(), warmup=20)
        vol = out["volatility"]
        assert vol["atr_period"] == 14
        assert vol["n"] == 6
        assert len(vol["dates"]) == len(vol["atr_pct"]) == len(vol["rvi_daily"]) == len(vol["ratio"])
        # Постоянный True Range=4 при Close=100 -> ATR%=4.0.
        assert abs(vol["atr_pct"][-1] - 4.0) < 0.02
        # RVI(40% годовых) -> дневная 40/sqrt(252).
        expected_rvi_daily = 40.0 / (252 ** 0.5)
        assert abs(vol["rvi_daily"][-1] - expected_rvi_daily) < 0.01
        assert abs(vol["ratio"][-1] - (4.0 / expected_rvi_daily)) < 0.01
        assert vol["rvi_annual"][-1] == 40.0

    def test_compute_result_volatility_without_rvi(self):
        raw = _flat_ohlc_raw_payload(with_rvi=False)
        out = compute_result(raw, warmup=20)
        vol = out["volatility"]
        assert abs(vol["atr_pct"][-1] - 4.0) < 0.02
        assert vol["rvi_daily"][-1] is None
        assert vol["ratio"][-1] is None



class _FakeRedis:
    def __init__(self, data: dict | None = None, *, connected: bool = True):
        self.data = dict(data or {})
        self.connected = connected
        self.written: dict = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.written[key] = value
        self.data[key] = value


class TestLatestStorage:
    def test_store_and_get_latest_via_snapshot(self, monkeypatch, snapshot_dir):
        import gex.application.breadth_imoex_service as svc

        cache = _FakeRedis()
        monkeypatch.setattr(svc, "_cache", cache)
        snapshot = snapshot_dir / "snapshot.json"
        monkeypatch.setattr(svc, "_SNAPSHOT_FILE", snapshot)

        payload = {"market": {"benchmark": "IMOEX"}, "meta": {"updated_at": "t"}}
        assert store_latest(payload) is True
        assert snapshot.exists()
        assert json.loads(snapshot.read_text(encoding="utf-8")) == payload
        assert svc._KEY_LATEST in cache.written

        # Эмуляция: Redis недоступен — читаем файл-снапшот.
        cache.connected = False
        assert get_latest() == payload

    def test_get_latest_returns_none_without_any_data(self, monkeypatch, snapshot_dir):
        import gex.application.breadth_imoex_service as svc

        monkeypatch.setattr(svc, "_cache", _FakeRedis(connected=False))
        monkeypatch.setattr(svc, "_SNAPSHOT_FILE", snapshot_dir / "missing.json")
        assert get_latest() is None

    def test_store_latest_true_when_only_file_works(self, monkeypatch, snapshot_dir):
        import gex.application.breadth_imoex_service as svc

        monkeypatch.setattr(svc, "_cache", _FakeRedis(connected=False))
        snapshot = snapshot_dir / "file_only.json"
        monkeypatch.setattr(svc, "_SNAPSHOT_FILE", snapshot)

        assert store_latest({"market": {}}) is True
        assert snapshot.exists()




# ----------------------------------------------------------------------
# Increment D: orchestrator force-refresh, adapter, sync gateway, quota
# ----------------------------------------------------------------------
import asyncio  # noqa: E402
import uuid  # noqa: E402

from fakeredis.aioredis import FakeRedis  # noqa: E402

from gex.orchestrator.adapters.base import BaseProviderAdapter, ProviderAdapterRegistry  # noqa: E402
from gex.orchestrator.adapters.iss_adapter import ISSAdapter  # noqa: E402
from gex.orchestrator.exceptions import RateLimitedError  # noqa: E402
from gex.orchestrator.schemas import (  # noqa: E402
    OrchestratorTask,
    ProviderResponse,
    RateLimitRuleConfig,
)
from gex.orchestrator.service import Orchestrator  # noqa: E402
import gex.orchestrator.sync_gateway as sync_gateway_module  # noqa: E402
from gex.orchestrator.sync_gateway import sync_fetch_imoex_breadth  # noqa: E402


@pytest.fixture
async def fake_redis():
    redis = FakeRedis()
    yield redis
    await redis.aclose()


class IssCountingAdapter(BaseProviderAdapter):
    provider_code = "iss"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, task: OrchestratorTask) -> ProviderResponse:
        self.calls += 1
        await asyncio.sleep(0.01)
        return ProviderResponse(
            provider=self.provider_code,
            data_type=task.data_type,
            data={"calls": self.calls},
            status_code=200,
        )


def _orchestrator_with_adapter(redis, adapter, *, rate_rules=None):
    registry = ProviderAdapterRegistry()
    registry.register(adapter)
    return Orchestrator(
        redis,
        adapter_registry=registry,
        inline_execution=True,
        rate_rules=rate_rules,
    )


class TestOrchestratorImoexBreadth:
    async def test_force_refresh_bypasses_fresh_cache(self, fake_redis):
        adapter = IssCountingAdapter()
        orch = _orchestrator_with_adapter(fake_redis, adapter)

        first = await orch.fetch_imoex_breadth(force_refresh=False)
        second = await orch.fetch_imoex_breadth(force_refresh=False)
        assert adapter.calls == 1
        assert first.data == second.data

        forced = await orch.fetch_imoex_breadth(force_refresh=True)
        assert adapter.calls == 2
        assert forced.data["calls"] == 2

    async def test_daily_quota_blocks_fourth_forced_run(self, fake_redis):
        adapter = IssCountingAdapter()
        quota_rule = RateLimitRuleConfig(
            name="iss:imoex_breadth:daily",
            provider="iss",
            endpoint_pattern="imoex_breadth",
            scope="endpoint",
            window_seconds=86400,
            max_requests=3,
            cost=1,
        )
        orch = _orchestrator_with_adapter(fake_redis, adapter, rate_rules=[quota_rule])

        for _ in range(3):
            await orch.fetch_imoex_breadth(force_refresh=True)
        assert adapter.calls == 3

        with pytest.raises(RateLimitedError):
            await orch.fetch_imoex_breadth(force_refresh=True)
        assert adapter.calls == 3

    async def test_admin_bypass_skips_daily_quota(self, fake_redis):
        adapter = IssCountingAdapter()
        quota_rule = RateLimitRuleConfig(
            name="iss:imoex_breadth:daily",
            provider="iss",
            endpoint_pattern="imoex_breadth",
            scope="endpoint",
            window_seconds=86400,
            max_requests=3,
            cost=1,
        )
        orch = _orchestrator_with_adapter(fake_redis, adapter, rate_rules=[quota_rule])

        for _ in range(3):
            await orch.fetch_imoex_breadth(force_refresh=True)
        assert adapter.calls == 3

        admin_run = await orch.fetch_imoex_breadth(force_refresh=True, bypass_rate_limit=True)
        assert admin_run.data["calls"] == 4
        assert adapter.calls == 4


class TestIssAdapterImoexBreadth:
    async def test_adapter_returns_fetcher_payload(self, monkeypatch):
        import gex.adapters.fetchers.imoex_breadth_fetcher as fetcher_module

        class FakeFetcher:
            def __init__(self, timeout=30.0):
                pass

            def fetch_payload(self):
                return {
                    "universe": [{"secid": "SBER"}],
                    "top10": ["SBER"],
                    "imoex": {"bars": [{"t": "2026-09-08T07:00:00+00:00", "c": 3000.0}]},
                    "stocks": [],
                }

        monkeypatch.setattr(fetcher_module, "ImoexBreadthFetcher", FakeFetcher)
        adapter = ISSAdapter()
        task = OrchestratorTask(
            request_id=uuid.uuid4(),
            provider="iss",
            data_type="imoex_breadth",
            symbol="IMOEX",
            params={},
            cache_key="orchestrator:cache:iss:imoex_breadth:IMOEX:-:-:-:test:v1",
        )
        response = await adapter.execute(task)
        assert response.status_code == 200
        assert response.data["top10"] == ["SBER"]


class TestSyncGatewayImoexBreadth:
    def test_sync_fetch_imoex_breadth_calls_orchestrator(self, monkeypatch):
        class DummyResult:
            data = {"ok": True}

        class DummyOrchestrator:
            def __init__(self) -> None:
                self.called: dict = {}

            async def fetch_imoex_breadth(self, *, force_refresh=False, max_wait_ms=120000, bypass_rate_limit=False):
                self.called = {"force_refresh": force_refresh, "max_wait_ms": max_wait_ms, "bypass_rate_limit": bypass_rate_limit}
                return DummyResult()

        dummy = DummyOrchestrator()
        monkeypatch.setattr(sync_gateway_module, "is_orchestrator_ready", lambda: True)
        monkeypatch.setattr(sync_gateway_module, "get_orchestrator", lambda: dummy)

        assert sync_fetch_imoex_breadth(force_refresh=True, bypass_rate_limit=True) == {"ok": True}
        assert dummy.called == {"force_refresh": True, "max_wait_ms": 120000, "bypass_rate_limit": True}



# ----------------------------------------------------------------------
# Increment E: GET /breadth-imoex route
# ----------------------------------------------------------------------
from fastapi import HTTPException  # noqa: E402


class TestBreadthImoexRoute:
    """Маршрут страницы: чтение готового расчёта, ни одного обращения к ISS.

    С итер. 38 обработчики асинхронные и читают снапшот (``SnapshotPort``), поэтому тест
    проверяет наблюдаемое: ответ 200 с данными сервиса и **ноль** обращений к ``ensure_warm``.
    Холодный случай (нет ни снапшота, ни расчёта) требует порта хранилища и общей обвязки —
    он покрыт ``tests/test_market_pages_cache.py`` (``TestImoexRoute``).
    """

    def test_route_returns_latest_and_never_fetches(self, monkeypatch):
        import asyncio

        import gex.application.breadth_imoex_service as svc
        from gex.routers import breadth_sector_router as mod
        from starlette.responses import Response

        payload = {"market": {"benchmark": "IMOEX"}, "meta": {"source": "iss"}}
        fetch_calls: list = []
        monkeypatch.setattr(svc, "get_latest", lambda: payload)
        monkeypatch.setattr(svc, "ensure_warm", lambda *a, **k: fetch_calls.append(1) or True)
        monkeypatch.setattr(mod, "_peek", _no_snapshot)

        body = asyncio.run(mod.get_breadth_imoex(response=Response(), store=object()))

        assert body["market"] == payload["market"]
        assert body["meta"]["page"] == "breadth-imoex"
        assert fetch_calls == []


async def _no_snapshot(store, key, page):
    """Пустой снапшот: проверяем путь «данных страницы нет, читаем хранилище сервиса»."""
    return None


class TestBreadthImoexAdminRefresh:
    """Ручное обновление: ставит фоновую задачу и отвечает 202, не парся ISS в запросе."""

    class _Signature:
        """Сигнатура задачи: ``apply_async`` обязан вызываться без повторов."""

        def __init__(self, *, fail: bool = False, queue: str | None = None):
            self.fail = fail
            self.queue = queue

        def set(self, **options):
            self.queue = options.get("queue")
            return self

        def apply_async(self, retry=True):
            assert retry is False, "повтор публикации внутри запроса недопустим"
            if self.fail:
                raise ConnectionError("broker down")

            class _Result:
                id = "imoex-task"

            return _Result()

    def test_admin_refresh_dispatches_and_returns_202(self, monkeypatch):
        import asyncio

        import gex.application.breadth_imoex_service as svc
        from gex.routers import breadth_sector_router as mod
        from gex.workers import local_refresh
        from gex.workers.tasks import market_pages as task_mod

        local_refresh.reset()
        signature = self._Signature()
        monkeypatch.setattr(task_mod, "admin_signature", lambda page, mode=None: signature)
        # ``ensure_warm`` в запросе вызываться не должен: парсинг ISS ушёл в воркер.
        monkeypatch.setattr(
            svc, "ensure_warm", lambda *a, **k: pytest.fail("ISS не парсится в запросе")
        )

        result = asyncio.run(mod.refresh_breadth_imoex(user=object()))

        assert result["status"] == "accepted"
        assert result["channel"] == "celery"
        assert result["task_id"] == "imoex-task"
        assert "status_url" in result

    def test_admin_refresh_falls_back_to_local_when_broker_down(self, monkeypatch):
        import asyncio

        from gex.routers import breadth_sector_router as mod
        from gex.workers import local_refresh
        from gex.workers.tasks import market_pages as task_mod

        local_refresh.reset()  # дебаунс/остывание живут в модуле — между тестами их не переносим
        local_calls: list = []
        monkeypatch.setattr(
            task_mod, "admin_signature", lambda page, mode=None: self._Signature(fail=True)
        )
        monkeypatch.setattr(task_mod, "refresh_page_now", lambda *a, **k: local_calls.append(k))

        result = asyncio.run(mod.refresh_breadth_imoex(user=object()))

        assert result["status"] == "accepted"
        assert result["channel"] == "local"
        deadline = time.monotonic() + 2.0
        while not local_calls and time.monotonic() < deadline:
            time.sleep(0.02)
        assert local_calls and local_calls[0]["bypass_rate_limit"] is True




# ----------------------------------------------------------------------
# Increment F: fixed MSK schedule + task queue + background handler
# ----------------------------------------------------------------------
from gex.application.scheduler import Scheduler  # noqa: E402
from gex.application.jobs import FetchTask  # noqa: E402
from gex.application.prewarm import PrewarmSlot  # noqa: E402
from gex.domain.schedule import PERIODIC_SLOTS, slot_occurrence  # noqa: E402


class TestFixedMskSchedule:
    def test_slot_due_only_inside_windows(self):
        """Окно вхождения считает домен; таблица слотов — ``PERIODIC_SLOTS``.

        Слоты IMOEX переехали в Celery Beat, но семантика окна не изменилась: слот «23:00»
        действует в ``[23:00, 24:00)`` МСК и не догоняется позже.
        """
        msk = timezone(timedelta(hours=3))
        slots = PERIODIC_SLOTS["breadth-imoex"]
        assert slots == ((23, 0), (8, 0)), "слоты МСК — источник расписания и для Beat"

        assert slot_occurrence(slots, datetime(2026, 9, 9, 22, 59, tzinfo=msk)) is None
        assert slot_occurrence(slots, datetime(2026, 9, 9, 23, 0, tzinfo=msk)) == "2026-09-09 23:00"
        assert slot_occurrence(slots, datetime(2026, 9, 9, 23, 59, 59, tzinfo=msk)) == "2026-09-09 23:00"
        assert slot_occurrence(slots, datetime(2026, 9, 10, 0, 1, tzinfo=msk)) is None
        assert slot_occurrence(slots, datetime(2026, 9, 10, 8, 5, tzinfo=msk)) == "2026-09-10 08:00"
        assert slot_occurrence(slots, datetime(2026, 9, 10, 9, 30, tzinfo=msk)) is None

    def test_scheduler_publishes_once_per_occurrence(self):
        """Фиксированный слот публикуется один раз за вхождение, и это держит **аренда**.

        Прежняя версия звала приватный ``Scheduler._publish_fixed`` и проверяла маркер
        ``gex:prewarm:fixed:...`` в Redis. После итер. 32 публикацию ведёт воркер прогрева,
        а защита от повторов — это ``RedisLease`` (``prewarm:fixed:{slot}:{occurrence}``):
        маркер «уже публиковали» через GET→publish→SET был не атомарен, поэтому после
        старта прогрев публиковала **каждая** реплика. Проверяем наблюдаемое поведение:
        первый тик публикует, повторный — нет, следующее вхождение — снова публикует.
        """
        from gex.application.prewarm import PrewarmPlan
        from gex.application.scheduler import Scheduler

        class FakeQueue:
            def __init__(self) -> None:
                self.published: list = []

            def publish_many(self, tasks):
                self.published.extend(tasks)
                return len(tasks)

        from gex.application.prewarm import PrewarmWorker

        msk = timezone(timedelta(hours=3))
        queue = FakeQueue()
        sched = Scheduler(queue=queue)
        # План сужаем до одного фиксированного слота: тест про механику аренды, а не про
        # состав расписания проекта (у IMOEX слоты теперь в Celery Beat). Слот строится
        # здесь явно — проверяется способность ``PrewarmWorker`` не публиковать дважды,
        # и она должна проверяться независимо от того, кто сегодня задаёт слоты.
        narrow = PrewarmPlan(slots=[PrewarmSlot(
            "breadth_imoex", 24 * 3600,
            lambda: [FetchTask("breadth_imoex", "iss", "IMOEX", priority=0)],
            "Широта MOEX: фиксированные слоты",
            fixed_msk=PERIODIC_SLOTS["breadth-imoex"],
        )])
        assert [s.name for s in narrow.slots] == ["breadth_imoex"]
        # Аренда — **единственный** механизм, снимающий повтор для фиксированного слота:
        # `_is_due` для таких слотов всегда True, а `_last_run` намеренно не участвует.
        # Без аренды воркер работает в режиме «единственный процесс» и публикует каждый тик —
        # поэтому тест обязан дать аренду, иначе он проверяет не то поведение, что в проде.
        class FakeLease:
            def __init__(self) -> None:
                self.held: dict = {}

            def acquire(self, name, ttl):
                if name in self.held:
                    return None  # занято другим «процессом»
                self.held[name] = object()
                return self.held[name]

            def release(self, name, token):
                self.held.pop(name, None)

        sched._worker = PrewarmWorker(narrow, queue, lease=FakeLease(), max_slots_per_tick=4)
        # Окно вхождения считается по **стенным** часам воркера (`_wall_clock`), а не по
        # аргументу `now`: именно так это работает в проде (`datetime.fromtimestamp(...)`).
        # Подставляем часы, чтобы тест был детерминированным и не зависел от текущего времени.
        current = {"ts": datetime(2026, 9, 9, 23, 5, tzinfo=msk).timestamp()}
        sched._worker._wall_clock = lambda: current["ts"]

        fired = sched._worker.run_once(now=current["ts"])
        assert "breadth_imoex" in fired, f"слот не сработал в окне 23:00: {fired}"
        assert len(queue.published) >= 1

        # Тот же слот, то же вхождение → повторной публикации нет (аренда уже взята).
        before = len(queue.published)
        sched._worker.run_once(now=current["ts"])
        assert len(queue.published) == before, "повторная публикация в том же вхождении"

        # Следующее вхождение (08:00) — публикация снова разрешена.
        current["ts"] = datetime(2026, 9, 10, 8, 2, tzinfo=msk).timestamp()
        sched._worker.run_once(now=current["ts"])
        assert len(queue.published) > before, "новое вхождение не опубликовало задачи"


class TestBreadthImoexTaskWiring:
    def test_fetch_task_queue_mapping(self):
        """Вид очереди задаёт ``FetchTask.queue`` (``queue_name`` был до итер. 28).

        Само имя потока ``gex:q:{kind}`` — деталь адаптера (``redis_streams.stream_name``),
        поэтому проверяется и он: порт обещает «вид задачи», транспорт решает форму имени.
        """
        from gex.adapters.queue.redis_streams import stream_name
        from gex.ports.job_queue import Priority

        task = FetchTask("breadth_imoex", "iss", "IMOEX", priority=0)
        assert task.queue == "vol", f"breadth_imoex обязан идти в очередь vol, а не {task.queue!r}"
        assert stream_name(task.queue, Priority.BACKGROUND) == "gex:q:vol:bg"

    def test_background_handler_dispatches_to_service(self, monkeypatch):
        import gex.application.background_fetcher as bg
        import gex.application.breadth_imoex_service as svc

        calls: list = []
        monkeypatch.setattr(svc, "ensure_warm", lambda force_refresh=True: calls.append(force_refresh) or True)
        monkeypatch.setattr(bg, "_bump_stats", lambda *a, **k: None)
        monkeypatch.setattr(bg, "_push_recent", lambda *a, **k: None)

        bg._fetch_breadth_imoex()
        assert calls == [True]



# ----------------------------------------------------------------------
# Increment G: ensure_warm failure modes + DB seed
# ----------------------------------------------------------------------
import gex.orchestrator.sync_gateway as gateway_module  # noqa: E402
from gex.orchestrator.repository import seed_defaults, rate_limit_rules  # noqa: E402


class TestEnsureWarm:
    def test_keep_previous_on_fetch_failure(self, monkeypatch):
        import gex.application.breadth_imoex_service as svc

        stored: list = []
        monkeypatch.setattr(gateway_module, "sync_fetch_imoex_breadth", lambda **k: None)
        monkeypatch.setattr(svc, "store_latest", lambda payload: stored.append(payload) or True)

        assert svc.ensure_warm(force_refresh=True) is False
        assert stored == []

    def test_computes_and_stores_on_success(self, monkeypatch):
        import gex.application.breadth_imoex_service as svc

        stored: list = []
        monkeypatch.setattr(gateway_module, "sync_fetch_imoex_breadth", lambda **k: _synthetic_raw_payload(n_stocks=30))
        monkeypatch.setattr(svc, "store_latest", lambda payload: stored.append(payload) or True)

        assert svc.ensure_warm(force_refresh=True) is True
        assert len(stored) == 1
        assert stored[0]["meta"]["source"] == "iss"
        assert stored[0]["market"]["benchmark"] == "IMOEX"

    def test_partial_load_below_threshold_keeps_previous(self, monkeypatch):
        import gex.application.breadth_imoex_service as svc

        raw = _synthetic_raw_payload()
        raw["universe"] = [{"secid": f"T{i}"} for i in range(10)]
        raw["stocks"] = raw["stocks"][:3]  # 3 из 10 — ниже 70%/30

        stored: list = []
        monkeypatch.setattr(gateway_module, "sync_fetch_imoex_breadth", lambda **k: raw)
        monkeypatch.setattr(svc, "store_latest", lambda payload: stored.append(payload) or True)

        assert svc.ensure_warm(force_refresh=True) is False
        assert stored == []

    def test_ensure_warm_accepts_small_fallback_universe(self, monkeypatch):
        import gex.application.breadth_imoex_service as svc

        raw = _synthetic_raw_payload(n_stocks=10)
        stored: list = []
        monkeypatch.setattr(gateway_module, "sync_fetch_imoex_breadth", lambda **k: raw)
        monkeypatch.setattr(svc, "store_latest", lambda payload: stored.append(payload) or True)

        assert svc.ensure_warm(force_refresh=True) is True
        assert len(stored) == 1

    def test_failure_reason_exposed_on_fetch_failure(self, monkeypatch):
        import gex.application.breadth_imoex_service as svc

        monkeypatch.setattr(gateway_module, "sync_fetch_imoex_breadth", lambda **k: None)
        assert svc.ensure_warm(force_refresh=True) is False
        assert svc.last_failure_reason() is not None

    def test_failure_reason_cleared_on_success(self, monkeypatch):
        import gex.application.breadth_imoex_service as svc

        monkeypatch.setattr(gateway_module, "sync_fetch_imoex_breadth", lambda **k: _synthetic_raw_payload(n_stocks=30))
        monkeypatch.setattr(svc, "store_latest", lambda payload: True)
        assert svc.ensure_warm(force_refresh=True) is True
        assert svc.last_failure_reason() is None



@pytest.fixture
def seeded_db_session():
    from gex.adapters.persistence.database import Base, SessionLocal, engine

    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        seed_defaults(session)
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


class TestRepositorySeed:
    def test_imoex_breadth_quota_rule_seeded_idempotently(self, seeded_db_session):
        rules = rate_limit_rules(seeded_db_session)
        daily = [r for r in rules if r["name"] == "iss:imoex_breadth:daily"]
        assert len(daily) == 1
        assert daily[0]["endpoint_pattern"] == "imoex_breadth"
        assert daily[0]["window_seconds"] == 86400
        assert daily[0]["max_requests"] == 3

        seed_defaults(seeded_db_session)
        rules_again = rate_limit_rules(seeded_db_session)
        assert len([r for r in rules_again if r["name"] == "iss:imoex_breadth:daily"]) == 1



class TestOrchestratorRedisHandshake:
    def test_create_async_redis_forces_resp2(self, monkeypatch):
        """Локальный Redis без RESP3/HELLO должен подключаться (protocol=2)."""
        import asyncio
        import gex.orchestrator.redis as redis_module

        captured: dict = {}

        class DummyClient:
            async def ping(self):
                return True

            async def aclose(self):
                return None

        def fake_from_url(url, **kwargs):
            captured.update(kwargs)
            return DummyClient()

        monkeypatch.setattr(redis_module, "from_url", fake_from_url)
        client = asyncio.run(redis_module.create_async_redis("redis://localhost:6379/0"))
        assert client is not None
        assert captured.get("protocol") == 2
