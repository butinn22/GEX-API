"""Tests for the TBANK stream bridge and Timescale storage layer."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.brokers.tbank import TbankBroker
from trading.adapters.brokers.tbank_stream import TbankStream, candle_to_bar, orderbook_to_domain
from trading.adapters.persistence import database as db
from trading.adapters.persistence.timescale import (
    HYPERTABLE_DDL,
    BarRow,
    TradeRow,
    bulk_insert_bars,
)
from trading.domain import Bar, DataFetchError


class _Candle:
    time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    open_price = 1.0
    close_price = 2.0
    highest_price = 3.0
    lowest_price = 0.5
    volume = 100.0


class _OrderBook:
    bids = [[100.0, 2.0]]
    asks = [[101.0, 1.0]]


def test_candle_to_bar():
    bar = candle_to_bar(_Candle())
    assert bar.open == 1.0 and bar.high == 3.0 and bar.close == 2.0


def test_orderbook_to_domain():
    book = orderbook_to_domain(_OrderBook())
    assert book.best_bid == 100.0 and book.best_ask == 101.0


def test_tbank_stream_dry_run_raises():
    stream = TbankStream(TbankBroker(""))  # no token → dry-run
    with pytest.raises(DataFetchError):
        stream.subscribe_candles("figi", None, lambda b: None)


def test_hypertable_ddl_present():
    assert any("create_hypertable('bars'" in d for d in HYPERTABLE_DDL)
    assert any("create_hypertable('trades'" in d for d in HYPERTABLE_DDL)


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(BarRow))
        await s.execute(delete(TradeRow))
        await s.commit()
        yield s


async def test_bulk_insert_bars(session):
    bars = [Bar(datetime(2024, 1, 1, tzinfo=timezone.utc), 1.0, 2.0, 0.5, 1.5, 10.0)]
    n = await bulk_insert_bars(session, "moex", "SBER", "1d", bars)
    assert n == 1
