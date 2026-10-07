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


def _configure(**kwargs) -> None:
    """Shared Alembic context config.

    ``compare_type=False``: our models infer ``Double`` from ``Mapped[float]``
    while the migrations spell the same storage as ``sa.Float()`` — FLOAT and
    Double are identical on our SQLite/Postgres targets, so we compare structure
    and nullability, not type spelling.

    ``render_as_batch=True``: emit ``batch_alter_table`` ops on autogenerate so
    ``--autogenerate`` produces SQLite-compatible ALTERs instead of failing.
    """
    context.configure(
        target_metadata=target_metadata,
        compare_type=False,
        render_as_batch=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    _configure(
        url=_sync_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_sync_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
