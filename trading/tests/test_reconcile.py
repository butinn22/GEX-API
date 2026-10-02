"""Tests for position reconciliation."""
from __future__ import annotations

from trading.application.reconcile import reconcile_positions
from trading.domain import Position, PositionSide


def _long(symbol, qty, avg=100.0) -> Position:
    return Position(symbol, PositionSide.LONG, qty, avg)


def test_reconcile_insert_close_unchanged():
    broker = [_long("A", 10), _long("B", 5)]
    local = {"A": _long("A", 10), "C": _long("C", 3)}
    diff = reconcile_positions(broker, local)
    assert [p.symbol for p in diff.to_insert] == ["B"]
    assert diff.to_close == ["C"]
    assert diff.to_update == []
    assert diff.unchanged == 1
    assert diff.has_changes


def test_reconcile_update_on_quantity_difference():
    diff = reconcile_positions([_long("A", 12)], {"A": _long("A", 10)})
    assert [p.symbol for p in diff.to_update] == ["A"]
    assert not diff.to_insert and not diff.to_close


def test_reconcile_no_changes():
    diff = reconcile_positions([_long("A", 10)], {"A": _long("A", 10)})
    assert not diff.has_changes and diff.unchanged == 1
