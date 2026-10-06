"""live signal engine: signal trade-plan columns + signal positions table

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-06
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The live signal engine writes signals that are not tied to a subscription
    # key, so ``key_id`` must be nullable (the key-driven generation path still
    # always supplies one). SQLite cannot ALTER a column in place, so this runs
    # through batch mode (recreate the table) — data is copied across.
    with op.batch_alter_table("key_signals") as batch:
        batch.alter_column("key_id", existing_type=sa.Integer(), nullable=True)

    # ── key_signals: the trade plan a strategy emits with a signal ──
    for name, col in (
        ("entry_price", sa.Float()),
        ("stop_loss", sa.Float()),
        ("take_profit", sa.Float()),
        ("position_size", sa.Float()),
        ("risk_pct", sa.Float()),
        ("risk_amount", sa.Float()),
    ):
        op.add_column("key_signals", sa.Column(name, col, nullable=True))
    op.add_column("key_signals", sa.Column("timeframe", sa.String(16), nullable=True))
    op.add_column("key_signals", sa.Column("bar_time", sa.DateTime(timezone=True), nullable=True))

    # ── signal_positions: full lifecycle of every live position ──
    op.create_table(
        "signal_positions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("strategy", sa.String(48), nullable=False, server_default=""),
        sa.Column("strategy_version", sa.String(16), nullable=False, server_default=""),
        sa.Column("preset", sa.String(32), nullable=False, server_default=""),
        sa.Column("timeframe", sa.String(16), nullable=False, server_default=""),
        sa.Column("source", sa.String(16), nullable=False, server_default="live"),
        sa.Column("side", sa.String(8), nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("quantity", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("initial_stop", sa.Float(), nullable=True),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("take_profit", sa.Float(), nullable=True),
        sa.Column("trail_price", sa.Float(), nullable=True),
        sa.Column("best_price", sa.Float(), nullable=True),
        sa.Column("worst_price", sa.Float(), nullable=True),
        sa.Column("mfe_r", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("bars_held", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.String(64), nullable=False, server_default=""),
        sa.Column("risk_amount", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("risk_pct", sa.Float(), nullable=True),
        sa.Column("gross_pnl", sa.Float(), nullable=True),
        sa.Column("net_pnl", sa.Float(), nullable=True),
        sa.Column("pnl_r", sa.Float(), nullable=True),
        sa.Column("pct_return", sa.Float(), nullable=True),
        sa.Column("unrealised_pnl", sa.Float(), nullable=True),
        sa.Column("mark_price", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_signal_positions_symbol", "signal_positions", ["symbol"])
    op.create_index("ix_signal_positions_status", "signal_positions", ["status"])
    op.create_index("ix_signal_positions_key_id", "signal_positions", ["key_id"])


def downgrade() -> None:
    op.drop_index("ix_signal_positions_key_id", table_name="signal_positions")
    op.drop_index("ix_signal_positions_status", table_name="signal_positions")
    op.drop_index("ix_signal_positions_symbol", table_name="signal_positions")
    op.drop_table("signal_positions")

    # Restore NOT NULL (live rows without a key are deleted first).
    op.execute("DELETE FROM key_signals WHERE key_id IS NULL")
    with op.batch_alter_table("key_signals") as batch:
        batch.alter_column("key_id", existing_type=sa.Integer(), nullable=False)
    for name in ("bar_time", "timeframe", "risk_amount", "risk_pct", "position_size",
                 "take_profit", "stop_loss", "entry_price"):
        op.drop_column("key_signals", name)
