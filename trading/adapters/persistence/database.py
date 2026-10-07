"""Async database engine/session (SQLAlchemy 2.0 + async driver).

SQLite (aiosqlite) for dev/tests, Postgres (asyncpg) in production — same
interface, configured via ``TRADING_DATABASE_URL``.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from trading.config import settings

from .models import Base

__all__ = [
    "configure",
    "init_db",
    "run_migrations",
    "dispose",
    "get_session",
    "session_factory",
]

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def configure(url: str | None = None) -> AsyncEngine:
    global _engine, _session_factory
    _engine = create_async_engine(url or settings.database_url, future=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def init_db(url: str | None = None) -> None:
    """Create tables directly (``create_all``, idempotent).

    **Test/bootstrap only — production uses :func:`run_migrations`.** The app
    lifespan runs ``alembic upgrade head``; this helper exists so ~20 test files
    can build a fresh schema fast without shelling out to Alembic.
    """
    engine = _engine or configure(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _sync_url() -> str:
    """Translate the async driver URL to the sync driver Alembic needs.

    Mirrors ``alembic/env.py`` so the app and the CLI migrate the same database.
    """
    url = settings.database_url
    return (
        url.replace("sqlite+aiosqlite:///", "sqlite:///")
        .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgresql://", "postgresql+psycopg2://")
    )


async def run_migrations() -> None:
    """Bring the database up to ``head`` via Alembic (idempotent).

    Alembic is synchronous, so the upgrade runs in a worker thread to keep the
    event loop free. Multi-replica deployments additionally run
    ``alembic upgrade head`` before the app starts (see docker-compose), so this
    call usually finds the DB already at head.

    A database that was bootstrapped by :func:`init_db` (``create_all`` — the
    test path, or a legacy app-created DB) already has the tables but no
    ``alembic_version`` marker; replaying the migration chain there would fail
    with "table already exists", so such a DB is **stamped** at head instead.
    """
    def _upgrade() -> None:
        from alembic.config import Config
        from sqlalchemy import create_engine, inspect
        from sqlalchemy.pool import NullPool

        from alembic import command

        url = _sync_url()
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", url)

        engine = create_engine(url, poolclass=NullPool)
        try:
            tables = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()
        has_tables = bool({"api_keys", "orders"} & tables)
        if has_tables and "alembic_version" not in tables:
            # create_all-managed DB: the schema already mirrors the models.
            command.stamp(cfg, "head")
            return
        command.upgrade(cfg, "head")

    await asyncio.to_thread(_upgrade)


async def dispose() -> None:
    if _engine is not None:
        await _engine.dispose()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields an async session per request."""
    if _session_factory is None:
        configure()
    async with _session_factory() as session:
        yield session


def session_factory() -> async_sessionmaker[AsyncSession]:
    """The configured factory, creating the engine on first use.

    Background tasks (the live signal engine) need their own sessions outside
    of the request dependency; this is the same factory, not a second engine.
    """
    global _session_factory
    if _session_factory is None:
        configure()
    assert _session_factory is not None
    return _session_factory
