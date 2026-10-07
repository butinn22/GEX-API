"""Custom SQLAlchemy column types for the trading persistence layer."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator

__all__ = ["UTCDateTime"]


class UTCDateTime(TypeDecorator):
    """A ``DateTime(timezone=True)`` that is **aware-UTC on read everywhere**.

    Round-1 proved the backend difference: SQLite returns **naive** datetimes
    for ``DateTime(timezone=True)`` while Postgres (``timestamptz``) returns
    **aware** ones — so ``aware_now - row.dt`` raised ``TypeError`` on SQLite
    (e.g. the holding-seconds computation). Normalising on bind (store UTC) and
    on read (attach UTC when the backend dropped tzinfo) makes every column
    behave identically on both backends.

    The storage type (``impl``) is unchanged, so there is **no DDL change** and
    ``alembic check`` stays clean; on Postgres this is effectively a no-op.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
