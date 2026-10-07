"""QA Round-3 — INDEPENDENT UTCDateTime verification (C15).

Proves a value stored from a *naive* datetime comes back **tz-aware UTC** on
SQLite (the backend that previously returned naive datetimes), that the impl is
identical (no DDL change), and that the previously-crashing
``aware_now - row.dt`` computation no longer raises.

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_tz.py -q
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest_asyncio
from sqlalchemy import DateTime, select
from sqlalchemy import delete as sa_delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import OrderRow
from trading.adapters.persistence.types import UTCDateTime


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(sa_delete(OrderRow))
        await s.commit()
        yield s


def test_impl_matches_datetime_tz_no_ddl_change():
    # The storage type must be the unchanged DateTime(timezone=True) → no DDL.
    assert isinstance(UTCDateTime().impl, DateTime)
    assert UTCDateTime().impl.timezone is True


def test_result_value_normalises_naive_to_utc():
    t = UTCDateTime()
    naive = datetime(2024, 6, 1, 12, 0, 0)
    out = t.process_result_value(naive, None)
    assert out.tzinfo is not None
    assert out.utcoffset() == timedelta(0)


async def test_roundtrip_is_aware_utc_on_sqlite(session):
    oid = f"tz-{uuid.uuid4().hex}"
    session.add(
        OrderRow(
            id=oid, exchange="bingx", symbol="BTC-USDT", side="buy",
            quantity=1.0, order_type="market", status="open",
            filled_quantity=0.0,
            created_at=datetime(2024, 6, 1, 12, 0, 0),  # NAIVE on purpose
        )
    )
    await session.commit()
    session.expire_all()  # force a fresh DB read

    got = (
        await session.execute(select(OrderRow).where(OrderRow.id == oid))
    ).scalar_one()
    assert got.created_at.tzinfo is not None, "read returned a naive datetime"
    assert got.created_at.utcoffset() == timedelta(0), "read is not UTC"

    # the round-1 crash path: aware-now minus the stored value must not raise
    age = datetime.now(UTC) - got.created_at
    assert age.total_seconds() > 0


async def test_timezone_offsets_are_normalised_to_utc(session):
    oid = f"tz2-{uuid.uuid4().hex}"
    plus3 = timezone(timedelta(hours=3))
    session.add(
        OrderRow(
            id=oid, exchange="bingx", symbol="ETH-USDT", side="buy",
            quantity=1.0, order_type="market", status="open",
            filled_quantity=0.0,
            created_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=plus3),
        )
    )
    await session.commit()
    session.expire_all()
    got = (
        await session.execute(select(OrderRow).where(OrderRow.id == oid))
    ).scalar_one()
    assert got.created_at.utcoffset() == timedelta(0)
    assert got.created_at.hour == 9  # 12:00+03:00 → 09:00Z
