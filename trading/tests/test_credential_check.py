"""Tests for the credential/connector check (no network)."""
from __future__ import annotations

import asyncio

from trading.application.credential_check import build_broker, check_broker
from trading.domain import Account, BrokerError


class _Broker:
    """Minimal duck-typed broker: only ``get_accounts`` is exercised."""

    def __init__(self, *, accounts=None, exc=None, delay=0.0, dry_run=False):
        self._accounts = accounts or []
        self._exc = exc
        self._delay = delay
        self.dry_run = dry_run

    async def get_accounts(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._accounts


async def test_check_broker_ok():
    ok, msg = await check_broker(
        _Broker(accounts=[Account(id="1", currency="USDT", cash=123.5)])
    )
    assert ok is True
    assert "USDT" in msg and "123.5" in msg


async def test_check_broker_broker_error_is_reported_not_raised():
    ok, msg = await check_broker(_Broker(exc=BrokerError("bad signature")))
    assert ok is False
    assert "rejected" in msg.lower()


async def test_check_broker_timeout():
    ok, msg = await check_broker(_Broker(delay=0.25), timeout=0.01)
    assert ok is False
    assert "timed out" in msg


async def test_check_broker_dry_run_is_not_a_false_positive():
    ok, msg = await check_broker(_Broker(accounts=[Account(id="dry")], dry_run=True))
    assert ok is False
    assert "dry-run" in msg


def test_build_broker_unsupported_exchange():
    assert build_broker("nope", "k", "s") is None


def test_build_broker_builds_bingx():
    broker = build_broker("bingx", "k", "s", {})
    assert broker is not None and broker.exchange.value == "bingx"
