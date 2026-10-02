"""Tests for the audit log."""
from __future__ import annotations

from trading.application.audit import AuditLog


def test_audit_log_records_and_filters():
    log = AuditLog()
    log.record("key_added", actor="admin", exchange="bingx")
    log.record("order_placed", actor="admin", symbol="BTC-USDT")
    log.record("key_added", actor="admin", exchange="tbank")

    assert len(log) == 3
    assert len(log.entries("key_added")) == 2
    assert log.entries("key_added")[0].detail["exchange"] == "bingx"
    assert log.entries("order_placed")[0].actor == "admin"
