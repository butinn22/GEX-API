"""Local signal client sessions: handshake, heartbeat, subscriptions, routing."""
from __future__ import annotations

import pytest

from trading.application.local_client import (
    ClientState,
    HandshakeRequiredError,
    LocalClientDispatcher,
    LocalClientRegistry,
    MaxTickersError,
    UnknownSessionError,
)
from trading.application.signal_hub import SignalHub


@pytest.fixture
def clock():
    t = [1000.0]
    return t


@pytest.fixture
def registry(clock):
    return LocalClientRegistry(heartbeat_timeout=10.0, clock=lambda: clock[0])


class TestHandshake:
    def test_connect_starts_in_connected_state(self, registry):
        s = registry.connect("bot-1")
        assert s.state is ClientState.CONNECTED
        assert s.client_id == "bot-1"
        assert registry.session_count == 1

    def test_session_ids_are_unique(self, registry):
        assert registry.connect().session_id != registry.connect().session_id

    def test_handshake_activates(self, registry):
        s = registry.connect()
        registry.handshake(s.session_id, "bot-2")
        assert registry.get(s.session_id).state is ClientState.ACTIVE
        assert registry.get(s.session_id).client_id == "bot-2"

    def test_handshake_unknown_session_raises(self, registry):
        with pytest.raises(UnknownSessionError):
            registry.handshake("nope", "x")

    def test_disconnect_forgets_session(self, registry):
        s = registry.connect()
        registry.disconnect(s.session_id)
        assert registry.session_count == 0
        with pytest.raises(UnknownSessionError):
            registry.get(s.session_id)


class TestHeartbeat:
    def test_touch_resets_liveness(self, registry, clock):
        s = registry.connect()
        registry.handshake(s.session_id, "b")
        clock[0] += 9.0
        registry.touch(s.session_id)
        clock[0] += 9.0
        assert not registry.is_stale(s.session_id)

    def test_session_goes_stale_after_timeout(self, registry, clock):
        s = registry.connect()
        clock[0] += 10.1
        assert registry.is_stale(s.session_id)
        assert registry.stale_sessions()[0].session_id == s.session_id

    def test_stale_uses_last_seen_not_connected_at(self, registry, clock):
        s = registry.connect()
        clock[0] += 8.0
        registry.touch(s.session_id)
        clock[0] += 5.0
        assert not registry.is_stale(s.session_id)


class TestSubscriptions:
    def test_subscribe_requires_handshake(self, registry):
        s = registry.connect()
        with pytest.raises(HandshakeRequiredError):
            registry.subscribe(s.session_id, ["BTC-USDT"])

    def test_subscribe_normalizes_and_dedupes(self, registry):
        s = registry.connect()
        registry.handshake(s.session_id, "b")
        out = registry.subscribe(s.session_id, ["btc-usdt", " SBER ", "BTC-USDT", ""])
        assert out == ["BTC-USDT", "SBER"]

    def test_subscribe_is_capped_at_max_tickers(self, registry):
        s = registry.connect()
        registry.handshake(s.session_id, "b")
        registry.subscribe(s.session_id, [f"T{i}" for i in range(20)])
        with pytest.raises(MaxTickersError):
            registry.subscribe(s.session_id, ["ONE-MORE"])

    def test_unsubscribe(self, registry):
        s = registry.connect()
        registry.handshake(s.session_id, "b")
        registry.subscribe(s.session_id, ["A", "B"])
        out = registry.unsubscribe(s.session_id, ["A"])
        assert out == ["B"]

    def test_custom_max_tickers(self, clock):
        reg = LocalClientRegistry(max_tickers=2, clock=lambda: clock[0])
        s = reg.connect()
        reg.handshake(s.session_id, "b")
        reg.subscribe(s.session_id, ["A", "B"])
        with pytest.raises(MaxTickersError):
            reg.subscribe(s.session_id, ["C"])


class TestRouting:
    def test_sessions_for_symbol_filters_by_ticker(self, registry):
        a = registry.connect(); registry.handshake(a.session_id, "a")
        b = registry.connect(); registry.handshake(b.session_id, "b")
        registry.subscribe(a.session_id, ["BTC-USDT"])
        registry.subscribe(b.session_id, ["SBER"])
        assert [s.client_id for s in registry.sessions_for("BTC-USDT")] == ["a"]

    def test_inactive_sessions_receive_nothing(self, registry):
        a = registry.connect()  # never handshakes, no tickers
        assert registry.sessions_for("BTC-USDT") == []

    def test_snapshot_reports_state_and_tickers(self, registry):
        s = registry.connect()
        registry.handshake(s.session_id, "bot")
        registry.subscribe(s.session_id, ["BTC-USDT"])
        snap = registry.snapshot()
        assert snap == [{
            "session_id": s.session_id,
            "client_id": "bot",
            "state": "active",
            "tickers": ["BTC-USDT"],
            "last_seen_age_s": 0.0,
            "stale": False,
        }]


class TestDispatcher:
    def test_fans_out_only_to_subscribed_sessions(self, registry):
        hub = SignalHub()
        disp = LocalClientDispatcher(registry, hub)
        a = registry.connect(); registry.handshake(a.session_id, "a")
        b = registry.connect(); registry.handshake(b.session_id, "b")
        registry.subscribe(a.session_id, ["BTC-USDT"])
        registry.subscribe(b.session_id, ["SBER"])

        disp.dispatch({"symbol": "BTC-USDT", "side": "buy"})
        assert not a.outbox.empty()
        assert b.outbox.empty()
        msg = a.outbox.get_nowait()
        assert msg["type"] == "signal"
        assert msg["ticker"] == "BTC-USDT"
        assert msg["signal"]["side"] == "buy"

    def test_ignores_messages_without_symbol(self, registry):
        hub = SignalHub()
        disp = LocalClientDispatcher(registry, hub)
        a = registry.connect(); registry.handshake(a.session_id, "a")
        registry.subscribe(a.session_id, ["BTC-USDT"])
        disp.dispatch({"type": "heartbeat"})
        assert a.outbox.empty()
