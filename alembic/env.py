"""Alembic environment — sync engine, targets the SQLAlchemy ``Base`` metadata.

The URL is read from ``TRADING_DATABASE_URL`` (async) and translated to a sync
driver: aiosqlite → sqlite, asyncpg → psycopg2.
"""
from __future__ import annotations

from sqlalchemy import create_engine, pool

from alembic import context

from trading.adapters.persistence.models import Base
from trading.config import settings

target_metadata = Base.metadata


def _sync_url() -> str:
    url = settings.database_url
    return (
        url.replace("sqlite+aiosqlite:///", "sqlite:///")
        .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgresql://", "postgresql+psycopg2://")
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_sync_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
