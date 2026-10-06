"""strategy hub: versioned strategy store columns on strategy_presets

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-08

Adds the Strategy Hub columns to ``strategy_presets`` (group identity,
per-group monotonic version, validated timeframe, metrics snapshot,
deployment status, backtest provenance), backfills sequential version numbers
per group so the new unique index holds, and creates the group-version
unique index plus the (symbol, strategy, status) index used by the
live-resolution path.
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── strategy_presets: versioning / status / provenance columns ──
    # All adds carry server defaults so NOT NULL is painless on existing rows
    # (plain add_column is fine on SQLite; only drops/alters need batch mode).
    op.add_column("strategy_presets",
                  sa.Column("strategy_name", sa.String(64), nullable=False,
                            server_default=""))
    op.add_column("strategy_presets",
                  sa.Column("version", sa.Integer(), nullable=False,
                            server_default="1"))
    op.add_column("strategy_presets",
                  sa.Column("timeframe", sa.String(16), nullable=False,
                            server_default=""))
    op.add_column("strategy_presets",
                  sa.Column("metrics_json", sa.String(1024), nullable=False,
                            server_default="{}"))
    op.add_column("strategy_presets",
                  sa.Column("status", sa.String(16), nullable=False,
                            server_default="backtest_only"))
    op.add_column("strategy_presets",
                  sa.Column("backtest_ref", sa.String(64), nullable=True))

    # Backfill versions: sequential per group (symbol, strategy, strategy_name)
    # ordered by id, so the unique group-version index is satisfiable. SQLite-
    # safe correlated subquery (no window functions over the updated table).
    op.execute(
        """
        UPDATE strategy_presets
        SET version = (
            SELECT COUNT(*) FROM strategy_presets AS p2
            WHERE p2.symbol = strategy_presets.symbol
              AND p2.strategy = strategy_presets.strategy
              AND p2.strategy_name = strategy_presets.strategy_name
              AND p2.id <= strategy_presets.id
        )
        """
    )

    op.create_index(
        "uq_strategy_presets_group_version",
        "strategy_presets",
        ["symbol", "strategy", "strategy_name", "version"],
        unique=True,
    )
    op.create_index(
        "ix_strategy_presets_status",
        "strategy_presets",
        ["symbol", "strategy", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_strategy_presets_status", table_name="strategy_presets")
    op.drop_index("uq_strategy_presets_group_version", table_name="strategy_presets")
    # SQLite cannot drop a column in place — batch mode recreates the table.
    with op.batch_alter_table("strategy_presets") as batch:
        for name in ("backtest_ref", "status", "metrics_json", "timeframe",
                     "version", "strategy_name"):
            batch.drop_column(name)
