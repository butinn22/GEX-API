"""Коллектор системных метрик: запись снапшота и повтор при блокировке БД.

Покрывает ровно тот сбой, который приходил в лог каждую минуту::

    sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked
    [SQL: DELETE FROM system_metric_snapshots WHERE ...]

Сценарий: фоновый поток пишет снапшот, пока обработчик держит транзакцию
открытой на время сетевого фетча. Раньше коллектор просто терял снимок и
писал предупреждение; теперь запись повторяется.
"""
from __future__ import annotations

from sqlalchemy.exc import OperationalError

from gex.adapters.middleware import system_metrics as sm
from gex.adapters.persistence.database import init_db


def _lock_error(msg="database is locked") -> OperationalError:
    """То, что выбрасывает SQLAlchemy при блокировке SQLite."""
    return OperationalError("stmt", {}, Exception(msg))


def test_collect_once_writes_a_snapshot():
    """Снимок попадает в БД и виден в истории."""
    init_db()
    collector = sm.SystemMetricCollector()

    collector.collect_once()

    assert collector.last_snapshot() is not None
    assert len(sm.get_metric_history(days=1)["points"]) == 1


def test_collect_once_retries_when_db_is_locked(monkeypatch):
    """«database is locked» — не потерянный снимок, а повтор записи."""
    init_db()
    collector = sm.SystemMetricCollector()

    calls = []
    real_write = collector._write_snapshot

    def flaky(now, payload):
        calls.append(1)
        if len(calls) == 1:
            raise _lock_error()
        return real_write(now, payload)

    monkeypatch.setattr(collector, "_write_snapshot", flaky)

    collector.collect_once()

    assert len(calls) == 2, "блокировку обязаны повторить, а не потерять снимок"
    assert collector.last_snapshot() is not None


def test_collect_once_survives_permanent_failure(monkeypatch):
    """БД лежит по-настоящему — поток коллектора продолжает работать."""
    init_db()
    collector = sm.SystemMetricCollector()

    def broken(now, payload):
        raise _lock_error()

    monkeypatch.setattr(collector, "_write_snapshot", broken)

    collector.collect_once()  # не должно бросить: цикл обязан выживать

    assert collector.last_snapshot() is None


def test_gather_does_not_fail_without_wired_services():
    """Метрики собираются, даже если контейнер сервисов не собран."""
    init_db()
    payload = sm.SystemMetricCollector()._gather()
    assert isinstance(payload, dict)
