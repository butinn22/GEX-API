"""Async database engine/session (SQLAlchemy 2.0 + async driver).

SQLite (aiosqlite) for dev/tests, Postgres (asyncpg) in production — same
interface, configured via ``TRADING_DATABASE_URL``.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from trading.config import settings
from .models import Base

__all__ = ["configure", "init_db", "dispose", "get_session"]

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def configure(url: str | None = None) -> AsyncEngine:
    global _engine, _session_factory
    _engine = create_async_engine(url or settings.database_url, future=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def init_db(url: str | None = None) -> None:
    """Create tables (idempotent). Call once at startup."""
    engine = _engine or configure(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose() -> None:
    if _engine is not None:
        await _engine.dispose()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields an async session per request."""
    if _session_factory is None:
        configure()
    async with _session_factory() as session:
        yield session
