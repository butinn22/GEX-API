"""tighten timestamp nullability: created_at / updated_at → NOT NULL

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-08

The ORM models declare every ``created_at`` / ``updated_at`` column as a
non-optional ``Mapped[datetime]`` (NOT NULL), but the original migrations
created them **nullable** — a real model↔schema drift that ``alembic check``
flags. Every row already has a value (``server_default=func.now()`` fills it on
INSERT), so tightening to NOT NULL is safe.

``batch_alter_table`` is used because SQLite cannot ``ALTER COLUMN`` in place:
it recreates the table. On Postgres the batch context emits a plain ``ALTER``.

If a batch operation is interrupted on SQLite it can leave a
``_alembic_tmp_<table>`` scratch table behind; drop any such table before
retrying the upgrade.
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

#: (table, [timestamp columns]) — every column is NOT NULL in the models.
_TABLES: list[tuple[str, list[str]]] = [
    ("api_keys", ["created_at"]),
    ("orders", ["created_at"]),
    ("backtest_results", ["created_at"]),
    ("strategy_presets", ["created_at", "updated_at"]),
    ("signal_keys", ["created_at"]),
    ("key_trades", ["created_at"]),
    ("signal_positions", ["created_at", "updated_at"]),
]


def upgrade() -> None:
    for table, cols in _TABLES:
        with op.batch_alter_table(table) as batch:
            for col in cols:
                batch.alter_column(
                    col,
                    existing_type=sa.DateTime(timezone=True),
                    nullable=False,
                    existing_server_default=sa.func.now(),
                )


def downgrade() -> None:
    for table, cols in _TABLES:
        with op.batch_alter_table(table) as batch:
            for col in cols:
                batch.alter_column(
                    col,
                    existing_type=sa.DateTime(timezone=True),
                    nullable=True,
                    existing_server_default=sa.func.now(),
                )
