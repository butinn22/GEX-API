"""referential integrity: non-destructive FKs on derived signal rows

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-07

Adds ``ON DELETE SET NULL`` foreign keys from the derived signal tables to their
parents, and makes ``key_trades.key_id`` nullable so the SET NULL action is
valid. This is **non-destructive**: existing orphan rows are *reported* to the
log, never silently deleted, and the FK only fires on a future hard delete of the
parent (which the application avoids — signal keys are soft-revoked and
signal/trade rows are purged explicitly in one transaction).

Why SET NULL and not CASCADE: the audit chose the non-destructive posture. The
derived rows keep their history and merely lose the dangling link, which an
operator can then reconcile from the orphan report.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")

#: (child table, child column, parent table) — every FK added by this revision.
_FKS = [
    ("key_signals", "key_id", "signal_keys"),
    ("key_signals", "preset_id", "strategy_presets"),
    ("signal_positions", "key_id", "signal_keys"),
    ("key_trades", "key_id", "signal_keys"),
    ("key_trades", "preset_id", "strategy_presets"),
]

_NAMES = {
    ("key_signals", "key_id"): "fk_key_signals_key_id",
    ("key_signals", "preset_id"): "fk_key_signals_preset_id",
    ("signal_positions", "key_id"): "fk_signal_positions_key_id",
    ("key_trades", "key_id"): "fk_key_trades_key_id",
    ("key_trades", "preset_id"): "fk_key_trades_preset_id",
}


def _report_orphans() -> None:
    """Log (never delete) rows whose FK target does not exist."""
    conn = op.get_bind()
    for child, col, parent in _FKS:
        sql = sa.text(
            f"SELECT COUNT(*) FROM {child} c WHERE c.{col} IS NOT NULL "
            f"AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.id = c.{col})"
        )
        try:
            n = int(conn.execute(sql).scalar() or 0)
        except sa.exc.OperationalError:
            continue  # a fresh/partial DB may not have every table yet
        if n:
            log.warning("FK orphan report: %s.%s -> %s has %d orphan row(s) "
                        "(left intact; reconcile manually)", child, col, parent, n)
        else:
            log.info("FK orphan report: %s.%s -> %s clean", child, col, parent)


def _add_fk(table: str, col: str, parent: str) -> None:
    with op.batch_alter_table(table) as batch:
        batch.create_foreign_key(_NAMES[(table, col)], parent, [col], ["id"],
                                 ondelete="SET NULL")


def _drop_fk(table: str, col: str) -> None:
    with op.batch_alter_table(table) as batch:
        batch.drop_constraint(_NAMES[(table, col)], type_="foreignkey")


def upgrade() -> None:
    _report_orphans()
    # key_trades.key_id must be nullable for ON DELETE SET NULL to be valid.
    with op.batch_alter_table("key_trades") as batch:
        batch.alter_column("key_id", existing_type=sa.Integer(), nullable=True)
    for child, col, parent in _FKS:
        _add_fk(child, col, parent)


def downgrade() -> None:
    for child, col, _parent in reversed(_FKS):
        _drop_fk(child, col)
    # Restore NOT NULL for key_trades.key_id (rollback path only: drop rows that a
    # prior SET NULL left without a key, mirroring revision 0005's pattern).
    op.execute("DELETE FROM key_trades WHERE key_id IS NULL")
    with op.batch_alter_table("key_trades") as batch:
        batch.alter_column("key_id", existing_type=sa.Integer(), nullable=False)
