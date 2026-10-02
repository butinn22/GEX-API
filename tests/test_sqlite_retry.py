"""Tests for gex.adapters.persistence.sqlite_retry.

Покрывает ровно то поведение, из-за которого «database is locked» долгое время
оставался незамеченным: повтор ТОЛЬКО на блокировку, повтор всей единицы
работы, отказ после исчерпания попыток.
"""
from __future__ import annotations

import pytest

from gex.adapters.persistence.sqlite_retry import (
    DEFAULT_ATTEMPTS,
    is_lock_error,
    run_with_retry,
)


def _lock_error(msg="database is locked"):
    from sqlalchemy.exc import OperationalError

    return OperationalError("stmt", {}, Exception(msg))


def test_is_lock_error_recognises_busy_sqlite():
    for msg in ("(sqlite3.OperationalError) database is locked",
                "database table is locked",
                "database is busy"):
        assert is_lock_error(Exception(msg)), msg


def test_is_lock_error_rejects_real_failures():
    """Неблокировочный сбой повторять нельзя — он должен пробрасываться сразу."""
    for msg in ("no such table: system_metric_snapshots",
                "UNIQUE constraint failed",
                "disk I/O error",
                "connection reset by peer"):
        assert not is_lock_error(Exception(msg)), msg


def test_returns_result_without_retry_when_it_succeeds():
    calls = []

    def op():
        calls.append(1)
        return "ok"

    assert run_with_retry(op, what="test") == "ok"
    assert len(calls) == 1


def test_retries_lock_error_then_succeeds():
    calls = []
    sleeps = []

    def op():
        calls.append(1)
        if len(calls) < 3:
            raise _lock_error()
        return "done"

    result = run_with_retry(op, what="test", sleep=sleeps.append, log=_quiet_logger())

    assert result == "done"
    assert len(calls) == 3
    assert len(sleeps) == 2  # пауза только между попытками


def test_backoff_grows_and_is_capped():
    sleeps = []

    def op():
        raise _lock_error()

    with pytest.raises(Exception):
        run_with_retry(op, attempts=6, base_delay=0.1, max_delay=0.5,
                       sleep=sleeps.append, log=_quiet_logger())

    # Пауза между попытками: attempts попыток → attempts-1 пауз.
    assert len(sleeps) == 5
    # Паузы растут, но не превышают max_delay (+ джиттер до 30%).
    assert all(0 < s <= 0.5 * 1.31 for s in sleeps)
    assert sleeps[0] < sleeps[-1]


def test_non_lock_error_is_not_retried():
    calls = []

    def op():
        calls.append(1)
        raise RuntimeError("no such table")

    with pytest.raises(RuntimeError):
        run_with_retry(op, attempts=5, sleep=lambda _s: None, log=_quiet_logger())

    assert len(calls) == 1  # повтор здесь только скрыл бы настоящую причину


def test_gives_up_after_attempts():
    calls = []

    def op():
        calls.append(1)
        raise _lock_error()

    with pytest.raises(Exception):
        run_with_retry(op, attempts=3, sleep=lambda _s: None, log=_quiet_logger())

    assert len(calls) == 3


def _quiet_logger():
    import logging

    log = logging.getLogger("test.sqlite_retry")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log
