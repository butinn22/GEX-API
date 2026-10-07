"""Multi-account ('wallet') routing: settings, instrument filter, broker build."""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.brokers.bingx import BingxBroker
from trading.adapters.brokers.tbank import TbankBroker
from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import ApiKeyRow
from trading.application.account_router import (
    AccountSettings,
    AccountView,
    build_broker,
    select_accounts,
)
from trading.application.keys_service import KeysService

SECRET = "test-secret-key"


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(ApiKeyRow))
        await s.commit()
        yield s


def make_account(**kw) -> AccountView:
    kw.setdefault("key_id", 1)
    kw.setdefault("exchange", "bingx")
    kw.setdefault("label", "main")
    kw.setdefault("credentials", {"api_key": "k", "api_secret": "s", "extra": {}})
    kw.setdefault("settings", AccountSettings())
    return AccountView(**kw)


class TestAccountSettings:
    def test_defaults(self):
        s = AccountSettings()
        assert s.enabled is True
        assert s.risk_profile == "medium"
        assert s.max_position_pct == 1.0
        assert s.leverage == 1.0
        assert s.instruments == ()

    def test_from_extra_normalizes_instruments(self):
        s = AccountSettings.from_extra({"instruments": ["btc-usdt", " SBER "]})
        assert s.instruments == ("BTC-USDT", "SBER")

    def test_from_extra_ignores_junk(self):
        assert AccountSettings.from_extra({"nonsense": 1}) == AccountSettings()
        assert AccountSettings.from_extra(None) == AccountSettings()

    def test_covers_wildcard_when_no_instruments(self):
        assert AccountSettings().covers("ANYTHING") is True

    def test_covers_matches_case_insensitively(self):
        s = AccountSettings(instruments=("BTC-USDT",))
        assert s.covers("btc-usdt") is True
        assert s.covers("ETH-USDT") is False

    def test_updated_merges_and_validates(self):
        s = AccountSettings().updated({"risk_profile": "high", "leverage": 5})
        assert s.risk_profile == "high"
        assert s.leverage == 5.0
        assert s.max_position_pct == 1.0  # untouched

    def test_updated_rejects_bad_risk_profile(self):
        with pytest.raises(ValueError):
            AccountSettings().updated({"risk_profile": "yolo"})

    def test_updated_rejects_bad_leverage(self):
        with pytest.raises(ValueError):
            AccountSettings().updated({"leverage": 0})

    def test_updated_rejects_bad_position_pct(self):
        with pytest.raises(ValueError):
            AccountSettings().updated({"max_position_pct": 1.5})

    def test_roundtrip_via_dict(self):
        s = AccountSettings(instruments=("A",), risk_profile="low", enabled=False)
        assert AccountSettings.from_extra(s.as_dict()) == s


class TestSelectAccounts:
    def test_filters_by_instrument_and_enabled(self):
        crypto = make_account(key_id=1, settings=AccountSettings(instruments=("BTC-USDT",)))
        wild = make_account(key_id=2)
        off = make_account(key_id=3, settings=AccountSettings(enabled=False))
        stocks = make_account(key_id=4, settings=AccountSettings(instruments=("SBER",)))
        out = select_accounts([crypto, wild, off, stocks], "BTC-USDT")
        assert [a.key_id for a in out] == [1, 2]


class TestBuildBroker:
    def test_bingx_broker_uses_account_credentials(self):
        broker = build_broker(make_account(
            credentials={"api_key": "PUB", "api_secret": "SEC", "extra": {}}
        ))
        assert isinstance(broker, BingxBroker)
        assert broker.client.api_key == "PUB"
        assert broker.client.api_secret == "SEC"

    def test_tbank_broker_uses_token_and_account(self):
        broker = build_broker(make_account(
            exchange="tbank",
            credentials={"api_key": "tok", "api_secret": "",
                         "extra": {"account_id": "acc-9", "sandbox": True}},
        ))
        assert isinstance(broker, TbankBroker)
        assert broker.token == "tok"
        assert broker.account_id == "acc-9"
        assert broker.sandbox is True

    def test_unknown_exchange_raises(self):
        with pytest.raises(ValueError):
            build_broker(make_account(exchange="coinbase"))


class FakeBroker:
    """Minimal BrokerAdapter stand-in that records the intents it receives."""

    def __init__(self, exchange="bingx", fail=False):
        self.exchange = exchange
        self.placed = []
        self.fail = fail

    async def place_order(self, intent):
        if self.fail:
            from trading.domain import BrokerError

            raise BrokerError("boom")
        self.placed.append(intent)
        from trading.domain import Order, OrderStatus

        return Order(id=f"fake-{len(self.placed)}", symbol=intent.symbol,
                     side=intent.side, quantity=intent.quantity.value,
                     order_type=intent.order_type, status=OrderStatus.PENDING)


class TestApplyRisk:
    def test_scales_down_to_notional_cap(self):
        from trading.application.account_router import apply_risk
        from trading.domain import OrderIntent, OrderType, Price, Quantity, Side

        intent = OrderIntent(symbol="BTC-USDT", side=Side.BUY, quantity=Quantity(250),
                            order_type=OrderType.LIMIT, limit_price=Price(100.0))
        settings = AccountSettings(max_position_pct=0.5, leverage=2.0)
        out = apply_risk(intent, settings, equity=10_000.0)  # cap = 10 000 → 100 units
        assert out.quantity.value == pytest.approx(100.0)

    def test_leaves_small_intent_untouched(self):
        from trading.application.account_router import apply_risk
        from trading.domain import OrderIntent, OrderType, Price, Quantity, Side

        intent = OrderIntent(symbol="X", side=Side.BUY, quantity=Quantity(5),
                            order_type=OrderType.LIMIT, limit_price=Price(100.0))
        out = apply_risk(intent, AccountSettings(max_position_pct=0.5), equity=10_000.0)
        assert out is intent  # untouched, not merely equal

    def test_no_price_returns_intent_unchanged(self):
        from trading.application.account_router import apply_risk
        from trading.domain import OrderIntent, OrderType, Quantity, Side

        intent = OrderIntent(symbol="X", side=Side.BUY, quantity=Quantity(5),
                            order_type=OrderType.MARKET)
        assert apply_risk(intent, AccountSettings(max_position_pct=0.1), equity=1000.0) is intent


class TestAccountRouter:
    def make_router(self, accounts, broker_factory=None):
        from trading.application.account_router import AccountRouter
        return AccountRouter(accounts, broker_factory=broker_factory or (lambda a: FakeBroker()))

    def test_brokers_for_filters_and_caches(self):
        calls = []
        def factory(a):
            return (calls.append(a.key_id), FakeBroker())[1]
        router = self.make_router([
            make_account(key_id=1, settings=AccountSettings(instruments=("BTC-USDT",))),
            make_account(key_id=2, settings=AccountSettings(instruments=("SBER",))),
            make_account(key_id=3, settings=AccountSettings(enabled=False)),
        ], broker_factory=factory)
        pairs = router.brokers_for("BTC-USDT")
        assert [a.key_id for a, _ in pairs] == [1]
        router.brokers_for("BTC-USDT")
        assert calls == [1]  # broker built once, then cached

    async def test_place_multi_routes_and_scales(self):
        from trading.domain import OrderIntent, OrderType, Price, Quantity, Side

        a1 = make_account(key_id=1, label="crypto",
                          settings=AccountSettings(instruments=("BTC-USDT",), max_position_pct=0.5))
        a2 = make_account(key_id=2, label="all")
        b1, b2 = FakeBroker(), FakeBroker()
        router = self.make_router([a1, a2], broker_factory=lambda a: {1: b1, 2: b2}[a.key_id])
        intent = OrderIntent(symbol="BTC-USDT", side=Side.BUY, quantity=Quantity(250),
                            order_type=OrderType.LIMIT, limit_price=Price(100.0))
        results = await router.place_multi(intent, equities={1: 10_000.0, 2: 10_000.0})
        assert [r.key_id for r in results] == [1, 2]
        assert all(r.ok for r in results)
        assert b1.placed[0].quantity.value == pytest.approx(50.0)  # 0.5 × 10 000 / 100
        assert b2.placed[0].quantity.value == pytest.approx(100.0)  # 1.0 × 10 000 / 100
        assert len(b2.placed) == 1

    async def test_place_multi_isolates_failures(self):
        from trading.domain import OrderIntent, OrderType, Quantity, Side

        a1 = make_account(key_id=1, label="broken")
        a2 = make_account(key_id=2, label="healthy")
        router = self.make_router([a1, a2], broker_factory=lambda a: FakeBroker(fail=a.key_id == 1))
        intent = OrderIntent(symbol="X", side=Side.BUY, quantity=Quantity(1),
                            order_type=OrderType.MARKET)
        results = await router.place_multi(intent)
        assert results[0].ok is False and "boom" in (results[0].error or "")
        assert results[1].ok is True

    async def test_place_multi_with_no_matching_accounts(self):
        from trading.domain import OrderIntent, OrderType, Quantity, Side

        router = self.make_router([make_account(settings=AccountSettings(instruments=("SBER",)))])
        intent = OrderIntent(symbol="BTC-USDT", side=Side.BUY, quantity=Quantity(1),
                            order_type=OrderType.MARKET)
        assert await router.place_multi(intent) == []


class TestExecutionEngineMultiAccount:
    async def test_place_multi_through_engine(self):
        from trading.application.account_router import AccountRouter
        from trading.application.execution import BrokerRouter, ExecutionEngine
        from trading.domain import OrderIntent, OrderType, Quantity, Side

        broker = FakeBroker()
        account_router = AccountRouter([make_account(key_id=1)], broker_factory=lambda a: broker)
        engine = ExecutionEngine(BrokerRouter(), account_router=account_router)
        intent = OrderIntent(symbol="X", side=Side.BUY, quantity=Quantity(2),
                            order_type=OrderType.MARKET)
        results = await engine.place_multi(intent)
        assert len(results) == 1 and results[0].ok
        assert broker.placed[0].quantity.value == 2

    async def test_place_multi_requires_account_router(self):
        from trading.application.execution import BrokerRouter, ExecutionEngine
        from trading.domain import OrderIntent, OrderType, Quantity, Side

        engine = ExecutionEngine(BrokerRouter())
        intent = OrderIntent(symbol="X", side=Side.BUY, quantity=Quantity(1),
                            order_type=OrderType.MARKET)
        with pytest.raises(ValueError):
            await engine.place_multi(intent)


class TestKeysServiceIntegration:
    async def test_update_settings_persists(self, session):
        svc = KeysService(SECRET)
        row = await svc.add_key(session, exchange="bingx", label="a",
                                api_key="k", api_secret="s")
        updated = await svc.update_settings(
            session, row.id,
            {"instruments": ["BTC-USDT"], "risk_profile": "high", "leverage": 3},
        )
        import json
        extra = json.loads(updated.extra_json)
        assert extra["instruments"] == ["BTC-USDT"]
        assert extra["risk_profile"] == "high"
        assert extra["leverage"] == 3.0

    async def test_update_settings_rejects_bad_values(self, session):
        svc = KeysService(SECRET)
        row = await svc.add_key(session, exchange="bingx", label="a",
                                api_key="k", api_secret="s")
        with pytest.raises(ValueError):
            await svc.update_settings(session, row.id, {"risk_profile": "extreme"})

    async def test_build_account_router_from_db(self, session):
        from trading.application.account_router import AccountRouter

        svc = KeysService(SECRET)
        await svc.add_key(session, exchange="bingx", label="crypto",
                          api_key="k1", api_secret="s1",
                          extra={"instruments": ["BTC-USDT"]})
        await svc.add_key(session, exchange="tbank", label="stocks",
                          api_key="k2", api_secret="",
                          extra={"account_id": "acc-2", "instruments": ["SBER"]})
        router = await svc.build_account_router(session)
        assert isinstance(router, AccountRouter)
        assert [a.label for a in router.accounts] == ["crypto", "stocks"]
        assert [a.label for a, _ in router.brokers_for("SBER")] == ["stocks"]

    async def test_resolve_accounts_for_symbol(self, session):
        svc = KeysService(SECRET)
        await svc.add_key(session, exchange="bingx", label="crypto",
                          api_key="k1", api_secret="s1",
                          extra={"instruments": ["BTC-USDT"]})
        await svc.add_key(session, exchange="bingx", label="all",
                          api_key="k2", api_secret="s2")
        await svc.add_key(session, exchange="tbank", label="stocks",
                          api_key="k3", api_secret="",
                          extra={"account_id": "acc-3", "instruments": ["SBER"], "enabled": False})
        accounts = await svc.resolve_accounts_for_symbol(session, "BTC-USDT")
        labels = [a.label for a in accounts]
        assert labels == ["crypto", "all"]  # disabled tbank account excluded
        assert accounts[0].credentials["api_key"] == "k1"
