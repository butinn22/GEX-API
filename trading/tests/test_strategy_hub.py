"""Strategy Hub service tests: versioning, go-live gate, promote/rollback/
demote, deployable resolution, live-engine hot-swap and signal-key resolution.

No network: the engine is stubbed with a synthetic (real-shaped) fetcher,
mirroring ``test_signal_engine.py``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.fetchers.registry import FetcherRegistry
from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import (
    KeySignalRow,
    KeyTradeRow,
    SignalKeyRow,
    SignalPositionRow,
    StrategyPresetRow,
)
from trading.application.presets import (
    PresetService,
    PresetValidationError,
)
from trading.application.signal_engine import (
    SignalEngine,
    SignalEngineConfig,
)
from trading.domain import Bar, Exchange

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

_METRICS = {
    "total_return": 0.21, "sharpe": 1.42, "max_drawdown": 0.07,
    "win_rate": 0.58, "n_trades": 14,
}


def _bars(n: int = 400) -> list[Bar]:
    out: list[Bar] = []
    price = 100.0
    state = 11
    for i in range(n):
        state = (state * 1103515245 + 12345) % (2 ** 31)
        noise = ((state / 2 ** 31) - 0.5) / 100.0
        drift = 0.0025 if (i // 12) % 5 < 4 else -0.0015
        price *= 1.0 + drift + noise
        out.append(Bar(timestamp=_T0 + timedelta(hours=4 * i), open=price,
                       high=price * 1.001, low=price * 0.999, close=price,
                       volume=1000.0))
    return out


def _key_bars(n: int = 400, *, seed: float = 100.0, drift: float = 0.0015) -> list[Bar]:
    """Sinusoidal trending series (the shape the key-generation tests use)."""
    import math

    out: list[Bar] = []
    price = seed
    for i in range(n):
        ret = drift + 0.012 * math.sin(i / 9.0) - 0.006
        prev = price
        price = max(1.0, price * (1 + ret))
        out.append(Bar(timestamp=_T0 + timedelta(days=i), open=prev,
                       high=max(prev, price) * 1.004, low=min(prev, price) * 0.996,
                       close=price, volume=1000.0))
    return out


class _StubFetcher:
    exchange = Exchange.BYBIT

    def __init__(self, bars: list[Bar]) -> None:
        self._bars = list(bars)

    async def get_ohlcv(self, symbol, timeframe, *, start=None, end=None, limit=500):
        return list(self._bars)[-limit:]

    async def get_instruments(self):
        return []

    async def get_orderbook(self, symbol, *, depth=20):
        raise NotImplementedError

    async def get_trades(self, symbol, *, limit=50):
        raise NotImplementedError


class _StubRegistry(FetcherRegistry):
    def __init__(self, fetcher) -> None:
        super().__init__()
        self._fetcher = fetcher
        self.register(fetcher)

    async def get_ohlcv(self, exchanges, symbol, timeframe, *, start=None, end=None, limit=500):
        return await self._fetcher.get_ohlcv(symbol, timeframe, limit=limit)


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        for table in (KeySignalRow, KeyTradeRow, SignalKeyRow,
                      SignalPositionRow, StrategyPresetRow):
            await s.execute(delete(table))
        await s.commit()
        yield s


@pytest_asyncio.fixture
async def engine():
    await db.init_db()
    async with db._session_factory() as s:
        for table in (KeySignalRow, SignalPositionRow, StrategyPresetRow):
            await s.execute(delete(table))
        await s.commit()
    return SignalEngine(session_factory=db.session_factory(),
                        registry=_StubRegistry(_StubFetcher(_bars())))


def _save_valid(svc: PresetService, **kw):
    """Save a version that passes the go-live gate (params + evidence)."""
    kw.setdefault("symbol", "BTC")
    kw.setdefault("params", {"zone_atr": 0.5})
    kw.setdefault("optimizer_run_id", "run-1")
    kw.setdefault("metrics", _METRICS)
    return svc.save(**kw)


# ── go-live gate ──────────────────────────────────────────────────────


async def test_gate_codes_params_missing(session):
    svc = PresetService(session)
    row = await svc.save(symbol="BTC", params={})
    ok, reasons = await svc.validate_for_live(row.id)
    assert ok is False
    codes = {r["code"] for r in reasons}
    assert "params_missing" in codes
    assert "no_backtest_evidence" in codes
    for r in reasons:
        assert set(r) == {"code", "message"} and r["message"]


async def test_gate_code_build_failed(session):
    svc = PresetService(session)
    row = await svc.save(symbol="BTC", strategy="no_such_strategy",
                         params={"whatever": 1}, optimizer_run_id="run-2",
                         metrics=_METRICS)
    ok, reasons = await svc.validate_for_live(row.id)
    assert ok is False
    assert [r["code"] for r in reasons] == ["build_failed"]
    assert "no_such_strategy" in reasons[0]["message"]


async def test_gate_code_no_backtest_evidence(session):
    svc = PresetService(session)
    row = await svc.save(symbol="BTC", params={"zone_atr": 0.5})  # no metrics
    ok, reasons = await svc.validate_for_live(row.id)
    assert ok is False
    assert [r["code"] for r in reasons] == ["no_backtest_evidence"]

    # metrics present but no provenance → also refused (never fabricated)
    row2 = await svc.save(symbol="BTC", params={"zone_atr": 0.6},
                          metrics=None)
    with pytest.raises(ValueError):
        await svc.save(symbol="BTC", params={"zone_atr": 0.6}, metrics=_METRICS)


# ── promote / rollback / demote lifecycle ─────────────────────────────


async def test_promote_runs_gate_and_sets_single_live(session):
    svc = PresetService(session)
    v1 = await _save_valid(svc, symbol="BTC", strategy_name="alpha")
    with pytest.raises(PresetValidationError) as exc:
        await svc.promote((await svc.save(symbol="BTC", strategy_name="alpha",
                                          params={"zone_atr": 0.9},
                                          is_default=False)).id)
    assert exc.value.code == "validation_failed"
    assert exc.value.reasons and exc.value.reasons[0]["code"] == "no_backtest_evidence"

    live = await svc.promote(v1.id)
    assert live.status == "live_enabled" and live.is_default
    with pytest.raises(ValueError):  # already live
        await svc.promote(v1.id)


async def test_one_live_per_symbol_strategy_across_names(session):
    svc = PresetService(session)
    alpha = await _save_valid(svc, symbol="BTC", strategy_name="alpha")
    beta = await _save_valid(svc, symbol="BTC", strategy_name="beta")

    await svc.promote(alpha.id)
    assert (await svc.promote(beta.id)).status == "live_enabled"

    # alpha was demoted — exactly one live row for (BTC, strategy) remains
    rows = await svc.list(symbol="BTC")
    live_rows = [r for r in rows if r.status == "live_enabled"]
    assert len(live_rows) == 1 and live_rows[0].id == beta.id

    demoted = await svc.demote(beta.id)
    assert demoted.status == "backtest_only"
    with pytest.raises(ValueError):  # not live any more
        await svc.demote(beta.id)


async def test_rollback_re_runs_gate_and_restores_prior_version(session):
    svc = PresetService(session)
    v1 = await _save_valid(svc, symbol="BTC", strategy_name="alpha")
    v2 = await _save_valid(svc, symbol="BTC", strategy_name="alpha")
    assert (v1.version, v2.version) == (1, 2)  # monotonic per group

    await svc.promote(v2.id)
    back = await svc.rollback(v1.id)  # gate re-runs on the old version
    assert back.id == v1.id and back.status == "live_enabled"
    assert back.is_default and not (await svc.get(v2.id)).is_default

    # a version that lost its evidence must not be re-activatable
    v3 = await svc.save(symbol="BTC", strategy_name="alpha",
                        params={"zone_atr": 0.7})
    with pytest.raises(PresetValidationError):
        await svc.rollback(v3.id)


async def test_get_deployable_resolution_order(session):
    svc = PresetService(session)
    named = await _save_valid(svc, symbol="BTC", strategy_name="named")
    plain = await _save_valid(svc, symbol="BTC", strategy_name="")

    # nothing pinned → the unnamed group's default (no live row yet)
    dep = await svc.get_deployable("BTC", "trend_confluence_unified")
    assert dep["preset_id"] == plain.id and dep["version"] == plain.version

    await svc.promote(named.id)
    # live wins over the group default, across names
    dep = await svc.get_deployable("BTC", "trend_confluence_unified")
    assert dep["preset_id"] == named.id
    assert dep["params"] == svc.params_of(named)
    assert dep["status"] == "live_enabled"

    # a pinned preset_id wins over everything
    dep = await svc.get_deployable("BTC", "trend_confluence_unified",
                                   preset_id=plain.id)
    assert dep["preset_id"] == plain.id

    # symbol upper-casing matches the repository convention
    assert (await svc.get_deployable("btc", "trend_confluence_unified"))["preset_id"] == named.id

    with pytest.raises(ValueError):
        await svc.get_deployable("MISSING", "trend_confluence_unified")


# ── live engine: config, hot-swap, signal rows ────────────────────────


def test_engine_config_parses_params_by_symbol_and_preset_ids():
    cfg = SignalEngineConfig.from_dict({
        "symbols": ["BTC", "ETH"],
        "params_by_symbol": {"btc": {"n_break": 12}},
        "preset_ids": {"BTC": 7},
    })
    assert cfg.params_by_symbol == {"BTC": {"n_break": 12}}
    assert cfg.preset_ids == {"BTC": 7}
    out = cfg.to_dict()
    assert out["params_by_symbol"] == {"BTC": {"n_break": 12}}
    assert out["preset_ids"] == {"BTC": 7}


def test_engine_config_rejects_malformed_overrides():
    with pytest.raises(ValueError):
        SignalEngineConfig.from_dict({"symbols": ["BTC"], "params_by_symbol": "nope"})
    with pytest.raises(ValueError):
        SignalEngineConfig.from_dict({"symbols": ["BTC"],
                                      "params_by_symbol": {"BTC": 5}})
    with pytest.raises(ValueError):
        SignalEngineConfig.from_dict({"symbols": ["BTC"], "preset_ids": [1, 2]})


def test_build_strategy_prefers_per_symbol_params():
    eng = SignalEngine(session_factory=db.session_factory())
    eng._config = SignalEngineConfig(
        symbols=["BTC"], params={"n_break": 30},
        params_by_symbol={"BTC": {"n_break": 12}},
    )
    assert eng._build_strategy("BTC")._p.n_break == 12   # per-ticker wins
    assert eng._build_strategy("ETH")._p.n_break == 30  # global fallback


async def test_reload_ticker_hot_swaps_from_the_store(engine, session):
    from trading.application.presets import PresetService as PS

    cfg = SignalEngineConfig.from_dict({
        "symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": 5, "bars": 400,
    })
    await engine.start(cfg)
    try:
        # the warm-up replay processes 400 bars — give it time to finish
        for _ in range(300):
            await asyncio.sleep(0.1)
            state = engine._tickers["ETHUSDT"]
            if state.polls and not state.first_poll:
                break
        else:
            pytest.fail("first poll did not complete in time")
        state = engine._tickers["ETHUSDT"]
        old_strategy = state.strategy
        assert state.seen

        svc = PS(session)
        row = await svc.save(symbol="ETHUSDT", strategy="confluence_breakout",
                             params={"preset": "alligator_4h", "n_break": 20},
                             optimizer_run_id="run-live-1", metrics=_METRICS)
        await svc.promote(row.id)

        assert await engine.reload_ticker("ethusdt") is True
        assert state.strategy is not None and state.strategy is not old_strategy
        assert state.strategy._p.n_break == 20
        assert state.preset_id == row.id
        assert state.first_poll is True          # warm-up replay re-runs
        assert state.seen == set()               # fetch window replays
        assert state.last_error == ""
        assert engine._config.preset_ids["ETHUSDT"] == row.id
    finally:
        await engine.stop()


async def test_engine_records_preset_id_on_signal_rows(engine, session):
    from trading.application.presets import PresetService as PS

    svc = PS(session)
    row = await svc.save(symbol="ETHUSDT", strategy="confluence_breakout",
                         params={"preset": "alligator_4h", "n_break": 20},
                         optimizer_run_id="run-live-3", metrics=_METRICS)

    cfg = SignalEngineConfig.from_dict({
        "symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": 5, "bars": 400,
        "params_by_symbol": {"ETHUSDT": {"n_break": 20}},
        "preset_ids": {"ETHUSDT": row.id},
    })
    await engine.start(cfg)
    try:
        assert engine._tickers["ETHUSDT"].strategy._p.n_break == 20
        sigs = []
        for _ in range(300):
            await asyncio.sleep(0.1)
            sigs = await engine.list_signals(symbol="ETHUSDT")
            if sigs:
                break
        assert sigs, "no signal was persisted in time"
        assert all(s.preset_id == row.id for s in sigs)
    finally:
        await engine.stop()


async def test_reload_ticker_when_stopped_or_foreign_symbol(engine):
    assert await engine.reload_ticker("ETHUSDT") is False  # engine not running
    cfg = SignalEngineConfig.from_dict({
        "symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": 5, "bars": 400,
    })
    await engine.start(cfg)
    try:
        assert await engine.reload_ticker("NOPE") is False  # not a tracked ticker
    finally:
        await engine.stop()


async def test_reload_ticker_without_deployable_preset_stops_ticker(engine, session):
    from trading.application.presets import PresetService as PS

    cfg = SignalEngineConfig.from_dict({
        "symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": 5, "bars": 400,
    })
    await engine.start(cfg)
    try:
        # the store has no confluence_breakout preset for ETHUSDT — but the
        # engine was started with explicit params, so the ticker keeps running
        # until a reload is requested for that strategy class
        svc = PS(session)
        row = await svc.save(symbol="ETHUSDT", strategy="confluence_breakout",
                             params={"preset": "alligator_4h", "n_break": 20},
                             optimizer_run_id="run-live-2", metrics=_METRICS)
        await svc.promote(row.id)
        await svc.demote(row.id)
        # delete the group's only version → nothing deployable remains
        await svc.delete(row.id)
        for r in await svc.list(symbol="ETHUSDT", strategy="confluence_breakout"):
            await svc.delete(r.id)

        assert await engine.reload_ticker("ETHUSDT") is False
        state = engine._tickers["ETHUSDT"]
        assert state.last_error == "no deployable preset"
    finally:
        await engine.stop()


# ── signal-key resolution (follow-live / pinned) ──────────────────────


async def test_signal_key_follows_live_version(session):
    import json as _json

    from trading.application.signal_keys import SignalKeyService

    svc = PresetService(session)
    v1 = await _save_valid(svc, symbol="CCC",
                            params={"emf_mode": "bonus", "momentum_period": 10})
    v2 = await _save_valid(svc, symbol="CCC",
                            params={"emf_mode": "bonus", "momentum_period": 20})

    keys = SignalKeyService(session)
    row, _ = await keys.create(exchange="tbank", tickers=["ccc"])
    config = _json.loads(row.config_json)
    tick = next(t for t in config["tickers"] if t["symbol"] == "CCC")
    # follow-live: no embedded params copy, no pinned version
    assert tick["params"] is None and tick["preset_id"] is None

    await svc.promote(v1.id)
    await keys.generate(key=row, refresh=False,
                        bars_by_symbol={"CCC": _key_bars(400)})
    trades = await keys.trades(row.id)
    assert all(t.preset_id == v1.id for t in trades)

    # promote a newer version → the same key serves it on the next generate
    await svc.promote(v2.id)
    await keys.generate(key=row, refresh=False,
                        bars_by_symbol={"CCC": _key_bars(400)})
    trades = await keys.trades(row.id)
    assert all(t.preset_id == v2.id for t in trades)


async def test_signal_key_pinned_version(session):
    import json as _json

    from trading.application.signal_keys import SignalKeyService

    svc = PresetService(session)
    v1 = await _save_valid(svc, symbol="DDD",
                            params={"emf_mode": "bonus", "momentum_period": 10})
    v2 = await _save_valid(svc, symbol="DDD",
                            params={"emf_mode": "bonus", "momentum_period": 20})

    keys = SignalKeyService(session)
    row, _ = await keys.create(exchange="tbank", tickers=["DDD"])
    # pin v1 by editing the stored reference (the deploy tab offers this)
    config = _json.loads(row.config_json)
    config["tickers"][0]["preset_id"] = v1.id
    row.config_json = _json.dumps(config)
    await session.commit()

    await svc.promote(v2.id)  # live moves on, but the key stays pinned
    await keys.generate(key=row, refresh=False,
                        bars_by_symbol={"DDD": _key_bars(400)})
    trades = await keys.trades(row.id)
    assert all(t.preset_id == v1.id for t in trades)


async def test_signal_key_explicit_params_stay_embedded(session):
    import json as _json

    from trading.application.signal_keys import SignalKeyService

    keys = SignalKeyService(session)
    row, _ = await keys.create(exchange="tbank", tickers=["EEE"],
                               params_by_ticker={"EEE": {"zone_atr": 0.4}})
    config = _json.loads(row.config_json)
    assert config["tickers"][0] == {"symbol": "EEE", "params": {"zone_atr": 0.4},
                                    "preset_id": None}
    await keys.generate(key=row, refresh=False,
                        bars_by_symbol={"EEE": _key_bars(400)})
    trades = await keys.trades(row.id)
    assert all(t.preset_id is None for t in trades)
