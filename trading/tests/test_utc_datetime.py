"""Timezone regression: read-back datetimes are aware-UTC on every backend (A5).

SQLite returns **naive** datetimes for ``DateTime(timezone=True)`` while
Postgres returns aware ones; the ``UTCDateTime`` decorator normalises both. The
tests below exercise the real round-trip through the (SQLite) test DB and would
raise ``TypeError`` on the pre-decorator code when subtracting an aware ``now``
from a naive stored row.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import KeySignalRow
from trading.adapters.persistence.types import UTCDateTime


def test_type_decorator_normalises_to_utc():
    dec = UTCDateTime()
    assert dec.process_bind_param(None, None) is None
    assert dec.process_result_value(None, None) is None
    naive = datetime(2026, 1, 1, 12, 0, 0)
    assert dec.process_bind_param(naive, None).utcoffset() == timedelta(0)
    assert dec.process_result_value(naive, None).utcoffset() == timedelta(0)


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(KeySignalRow))
        await s.commit()
        yield s


async def test_read_back_datetime_is_aware_utc(session):
    session.add(KeySignalRow(
        symbol="BTC-USDT", side="buy", state="long_entry", strategy="sma_crossover",
        timestamp=datetime(2026, 1, 2, 3, 4, 5),  # naive on write
    ))
    await session.commit()
    session.expunge_all()  # force the value to come back off the DB
    row = (await session.execute(select(KeySignalRow))).scalar_one()
    assert row.timestamp.tzinfo is not None
    assert row.timestamp.utcoffset() == timedelta(0)


async def test_subtracting_two_read_back_datetimes_does_not_raise(session):
    session.add(KeySignalRow(
        symbol="X", side="buy", state="long_entry", strategy="s",
        timestamp=datetime(2026, 1, 2, 3, 0, 0),
    ))
    session.add(KeySignalRow(
        symbol="X", side="sell", state="long_exit", strategy="s",
        timestamp=datetime(2026, 1, 2, 4, 30, 0),
    ))
    await session.commit()
    session.expunge_all()
    rows = (
        await session.execute(select(KeySignalRow).order_by(KeySignalRow.timestamp))
    ).scalars().all()
    # Pre-fix these were naive → ``aware_now - naive_row`` raised TypeError.
    assert datetime.now(timezone.utc) - rows[0].timestamp > timedelta(0)
    assert rows[1].timestamp - rows[0].timestamp == timedelta(hours=1, minutes=30)
