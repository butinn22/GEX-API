"""presets + signal keys + key signals/trades (unified strategy workflow)

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-04
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_presets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("strategy", sa.String(48), nullable=False),
        sa.Column("strategy_version", sa.String(16), nullable=False, server_default=""),
        sa.Column("params_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("source", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("optimizer_run_id", sa.String(64), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.String(255), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_strategy_presets_symbol", "strategy_presets", ["symbol"])
    op.create_index("ix_strategy_presets_strategy", "strategy_presets", ["strategy"])

    op.create_table(
        "signal_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("exchange", sa.String(16), nullable=False),
        sa.Column("label", sa.String(64), nullable=False, server_default=""),
        sa.Column("config_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_signal_keys_key", "signal_keys", ["key"], unique=True)
    op.create_index("ix_signal_keys_exchange", "signal_keys", ["exchange"])

    op.create_table(
        "key_signals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(255), nullable=False, server_default=""),
        sa.Column("strength", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("price", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("strategy", sa.String(48), nullable=False),
        sa.Column("strategy_version", sa.String(16), nullable=False, server_default=""),
        sa.Column("preset_id", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="live"),
        sa.Column("indicators_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_key_signals_key_id", "key_signals", ["key_id"])
    op.create_index("ix_key_signals_timestamp", "key_signals", ["timestamp"])

    op.create_table(
        "key_trades",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("gross_pnl", sa.Float(), nullable=False),
        sa.Column("net_pnl", sa.Float(), nullable=False),
        sa.Column("pct_return", sa.Float(), nullable=False),
        sa.Column("holding_seconds", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("exit_reason", sa.String(64), nullable=False, server_default=""),
        sa.Column("strategy", sa.String(48), nullable=False),
        sa.Column("strategy_version", sa.String(16), nullable=False, server_default=""),
        sa.Column("preset_id", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="live"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_key_trades_key_id", "key_trades", ["key_id"])


def downgrade() -> None:
    op.drop_index("ix_key_trades_key_id", table_name="key_trades")
    op.drop_table("key_trades")
    op.drop_index("ix_key_signals_timestamp", table_name="key_signals")
    op.drop_index("ix_key_signals_key_id", table_name="key_signals")
    op.drop_table("key_signals")
    op.drop_index("ix_signal_keys_exchange", table_name="signal_keys")
    op.drop_index("ix_signal_keys_key", table_name="signal_keys")
    op.drop_table("signal_keys")
    op.drop_index("ix_strategy_presets_strategy", table_name="strategy_presets")
    op.drop_index("ix_strategy_presets_symbol", table_name="strategy_presets")
    op.drop_table("strategy_presets")
