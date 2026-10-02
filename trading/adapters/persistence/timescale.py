"""TimescaleDB storage: bars/trades models + hypertable DDL + bulk write.

The bars/trades tables are created by SQLAlchemy ``create_all`` (they work as
plain tables on SQLite); the hypertable conversion is Postgres/Timescale-only
and mirrors ``docker/timescale-init/001_hypertables.sql``.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, text
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base
from .bulk import bulk_insert

__all__ = ["BarRow", "TradeRow", "HYPERTABLE_DDL", "create_hypertables", "bulk_insert_bars"]


class BarRow(Base):
    __tablename__ = "bars"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    exchange: Mapped[str] = mapped_column(String(16), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(8), primary_key=True)
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    close: Mapped[float]
    volume: Mapped[float] = mapped_column(default=0.0)


class TradeRow(Base):
    __tablename__ = "trades"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    exchange: Mapped[str] = mapped_column(String(16), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    price: Mapped[float]
    quantity: Mapped[float]
    side: Mapped[str] = mapped_column(String(8))


HYPERTABLE_DDL = [
    "SELECT create_hypertable('bars', 'time', if_not_exists => TRUE);",
    "SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);",
    "CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON bars (symbol, time DESC);",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, time DESC);",
]


async def create_hypertables(engine) -> None:
    """Convert bars/trades to hypertables. Postgres/Timescale ONLY."""
    async with engine.begin() as conn:
        for ddl in HYPERTABLE_DDL:
            await conn.execute(text(ddl))


async def bulk_insert_bars(session, exchange: str, symbol: str, timeframe: str, bars) -> int:
    """Insert domain Bars into the ``bars`` table; returns the count."""
    rows = [
        {
            "time": b.timestamp, "exchange": exchange, "symbol": symbol, "timeframe": timeframe,
            "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume,
        }
        for b in bars
    ]
    return await bulk_insert(session, BarRow, rows)
