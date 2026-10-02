"""Tests for the Celery offloading layer.

These run **without a broker**: Celery's eager mode executes tasks inline, so the
suite stays hermetic. What is covered here is the part that is easy to get wrong
and expensive to debug in production — queue routing by rate-limit bucket, the
``200`` / ``202`` / ``502`` dispatch contract, broker-down fallback, JSON-safety of
task payloads, and the Beat lease that enforces at-most-one scan per window.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.responses import JSONResponse

from gex.workers import config, dispatch
from gex.workers.celery_app import (
    ANALYTICS_QUEUE,
    SCAN_QUEUES,
    app,
    queue_for_universe,
)
from gex.workers.tasks import scanner as sc


# ── worker-side service factory ────────────────────────────────────────────


def test_services_lock_is_reentrant():
    """Regression: ``build_auto_scanner`` used to hold ``_lock`` and then call
    ``get_redis_client()``, which takes the same lock. ``threading.Lock`` is not
    reentrant, so **every worker deadlocked on startup**. The lock must be an
    RLock (the nested call is also hoisted out of the critical section now)."""
    from gex.workers import services

    assert services._lock.acquire(timeout=2), "lock busy — test isolation broken"
    try:
        assert services._lock.acquire(timeout=2), "lock не реентерабелен: воркер зависнет"
        services._lock.release()
    finally:
        services._lock.release()


def test_build_auto_scanner_returns_service_for_universe():
    """The worker rebuilds services itself — it never runs main.py's DI container."""
    from gex.workers.services import build_auto_scanner

    svc = build_auto_scanner("us")
    try:
        tickers = svc.get_tickers()
    except Exception as exc:  # noqa: BLE001 — universe file/CSV missing in this env
        pytest.skip(f"вселенная us недоступна в этом окружении: {exc}")
    assert tickers, "пустой список тикеров — каталог вселенных не загрузился"
    assert svc._universe == "us"


# ── routing ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "universe,expected",
    [
        ("us", "gex_scan_yf"),
        ("fx", "gex_scan_yf"),  # shares the yfinance budget with `us`
        ("sectors", "gex_scan_yf"),  # sector ETFs are fetched through yfinance too
        ("ru", "gex_scan_moex"),
        ("crypto", "gex_scan_bybit"),
        ("unknown", "gex_scan_yf"),  # never lose a task to an unknown universe
        ("", "gex_scan_yf"),
        (None, "gex_scan_yf"),
    ],
)
def test_queue_routing_follows_rate_bucket(universe, expected):
    assert queue_for_universe(universe) == expected


def test_us_and_fx_share_one_queue_because_one_provider_budget():
    """us, fx and sectors all hit yfinance — putting them on separate queues would
    let two workers draw on the same budget concurrently and breach the provider limit."""
    assert queue_for_universe("us") == queue_for_universe("fx")
    assert queue_for_universe("us") == queue_for_universe("sectors")
    assert queue_for_universe("ru") != queue_for_universe("us")


def test_scan_signature_is_routed_to_its_bucket_queue():
    sig = sc.scan_signature("crypto")
    assert sig.options.get("queue") == SCAN_QUEUES["bybit"]
    assert sig.args == ("crypto",) or sig.kwargs == {"universe": "crypto"}


def test_next_batch_signature_carries_batch_size():
    sig = sc.next_batch_signature("ru", 25)
    assert sig.options.get("queue") == SCAN_QUEUES["moex_iss"]
    assert sig.kwargs["batch_size"] == 25


# ── payload safety ─────────────────────────────────────────────────────────


def test_jsonify_makes_datetimes_serialisable():
    payload = {"scanned_at": datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc), "n": 1}
    out = sc._jsonify(payload)
    assert out["scanned_at"] == "2026-09-22 10:00:00+00:00"
    assert out["n"] == 1


def test_jsonify_survives_non_serialisable_objects():
    class Exotic:
        pass

    out = sc._jsonify({"x": Exotic()})
    assert isinstance(out["x"], str)


def test_backoff_grows_and_is_capped():
    delays = [sc._backoff(i) for i in range(6)]
    assert delays[0] < delays[3]
    assert all(d <= 310 for d in delays)  # cap 300 + jitter 10


def test_permanent_errors_are_not_retried():
    """``ValueError`` means 'no data for this ticker' — retrying would burn the
    provider budget on a request that will fail again."""
    assert ValueError not in sc.TRANSIENT_ERRORS
    assert RuntimeError in sc.TRANSIENT_ERRORS


# ── tasks (eager mode) ─────────────────────────────────────────────────────


class _FakeScanner:
    """Stands in for AutoScannerService so tests never touch the network."""

    def __init__(self, payload=None):
        self.payload = payload or {
            "scanned_count": 3,
            "scanned_at": datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc),
            "new_signals": 1,
        }
        self.run_scan_calls = 0
        self.run_batch_calls = []

    def run_scan(self):
        self.run_scan_calls += 1
        return self.payload

    def run_next_batch(self, batch_size=10):
        self.run_batch_calls.append(batch_size)
        return {**self.payload, "batch_size": batch_size}


@pytest.fixture
def eager(monkeypatch):
    """Run tasks inline, restoring the real setting afterwards."""
    monkeypatch.setattr(app.conf, "task_always_eager", True)
    monkeypatch.setattr(app.conf, "task_eager_propagates", True)
    return app


@pytest.fixture
def fake_scanner(monkeypatch):
    fake = _FakeScanner()
    monkeypatch.setattr(
        "gex.workers.services.build_auto_scanner", lambda universe: fake
    )
    return fake


def test_scan_task_returns_json_safe_payload(eager, fake_scanner):
    result = sc.scan_auto_run.apply(kwargs={"universe": "us"}).get()
    assert result["scanned_count"] == 3
    assert isinstance(result["scanned_at"], str)  # datetime was stringified
    assert fake_scanner.run_scan_calls == 1


def test_next_batch_task_passes_batch_size(eager, fake_scanner):
    result = sc.scan_auto_next_batch.apply(
        kwargs={"universe": "crypto", "batch_size": 7}
    ).get()
    assert result["batch_size"] == 7
    assert fake_scanner.run_batch_calls == [7]


def test_permanent_failure_returns_structured_payload(eager, monkeypatch):
    """A permanent failure (bad config, missing universe file) must come back as a
    structured payload instead of being retried until the budget is wasted."""

    def boom(universe):
        raise Exception("service exploded")

    monkeypatch.setattr("gex.workers.services.build_auto_scanner", boom)
    result = sc.scan_auto_run.apply(kwargs={"universe": "us"}, throws=False).get()
    assert result["failed"] is True
    assert "service exploded" in result["error"]


def test_transient_failure_is_retried_with_backoff(eager, monkeypatch):
    """Provider/transport errors are transient — they must be retried, not
    reported as a definitive failure."""
    from celery.exceptions import Retry

    def boom(universe):
        raise RuntimeError("provider 503")

    monkeypatch.setattr("gex.workers.services.build_auto_scanner", boom)
    with pytest.raises(Retry):  # eager mode surfaces the retry immediately
        sc.scan_auto_run.apply(kwargs={"universe": "us"}, throws=True)


# ── dispatch contract ──────────────────────────────────────────────────────


class _FakeResult:
    """Эмулирует ``AsyncResult`` для контракта dispatch.

    ``ready_after``: сколько внутренних опросов должно пройти, прежде чем задача
    «готова». ``get(timeout=…)`` эмулирует Celery: готовую задачу возвращает сразу,
    незавершённую — держит до исчерпания бюджета и поднимает celery-TimeoutError.
    """

    def __init__(self, ready_after=0, payload=None, fail=False, task_id="task-42"):
        self.id = task_id
        self.ready_after = ready_after
        self.result = RuntimeError("kaboom") if fail else (payload or {"ok": True})
        self.state = "FAILURE" if fail else "SUCCESS"
        self.get_calls = 0
        self.ready_calls = 0

    def ready(self):
        self.ready_calls += 1
        return self.ready_after <= 0

    def failed(self):
        return isinstance(self.result, BaseException)

    def get(self, propagate=False, timeout=None, interval=0.5):
        from celery.exceptions import TimeoutError as CeleryTimeoutError

        self.get_calls += 1
        if self.ready_after <= 0:
            return self.result
        raise CeleryTimeoutError("still running")


class _FakeSig:
    def __init__(self, result):
        self._result = result

    def apply_async(self):
        return self._result


class _BrokenSig:
    def apply_async(self):
        raise ConnectionError("broker unreachable")


def test_dispatch_returns_200_when_task_finishes_in_time():
    result = _FakeResult(ready_after=0)
    response = dispatch.dispatch_bounded(_FakeSig(result))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert b'"ok"' in response.body
    # Ожидание — один блокирующий get(), а не цикл ready(): каждый лишний
    # round-trip к result backend — это лишнее обращение к Redis под нагрузкой.
    assert result.get_calls == 1
    assert result.ready_calls == 0


def test_dispatch_returns_202_when_task_is_still_running():
    result = _FakeResult(ready_after=10_000)
    response = dispatch.dispatch_bounded(
        _FakeSig(result), timeout_s=0, poll_interval_s=0
    )
    assert response.status_code == 202
    assert b"task_id" in response.body
    assert b'"accepted"' in response.body
    assert response.headers["Location"] == "/tasks/task-42"
    assert response.headers["Retry-After"] == "2"
    assert result.get_calls == 1


def test_dispatch_returns_502_when_task_failed():
    result = _FakeResult(ready_after=0, fail=True)
    response = dispatch.dispatch_bounded(_FakeSig(result), timeout_s=1)
    assert response.status_code == 502
    assert b"kaboom" in response.body


def test_dispatch_returns_202_for_retrying_task():
    """Задача в состоянии RETRY — «ещё выполняется», а не провал: как и прежний
    цикл опроса, отвечаем 202, а не 502 (иначе транзиентный сбой провайдера
    выглядел бы для клиента как окончательная ошибка)."""
    from celery.exceptions import Retry

    result = _FakeResult(ready_after=0)
    result.result = Retry("transient failure", exc=RuntimeError("provider 503"))
    result.state = "RETRY"
    response = dispatch.dispatch_bounded(_FakeSig(result), timeout_s=1)
    assert response.status_code == 202
    assert b"task_id" in response.body
    assert result.get_calls == 1


def test_dispatch_falls_back_to_inline_when_broker_is_down():
    """The queue is an optimisation, never a new failure mode: if the broker is
    unreachable the route must run the work itself, as it did before Celery."""
    assert dispatch.dispatch_bounded(_BrokenSig()) is None


def test_publish_uses_bounded_pool_not_a_thread_per_call():
    """Под штормом публикаций при мёртвом брокере живых попыток не больше пула,
    а остальные падают в фолбэк немедленно, не порождая новые потоки.

    Раньше каждый вызов создавал отдельный поток: всплеск запросов к страницам
    в момент недоступности брокера порождал столько же брошенных потоков.
    """
    import threading
    import time as _time

    dispatch.reset_dispatch_state()
    started: list[int] = []
    _guard = threading.Lock()

    def slow_signature():
        with _guard:
            started.append(1)
        _time.sleep(0.15)  # публикация «висит» дольше бюджета вызова
        raise ConnectionError("broker unreachable")

    try:
        # Первые вызовы занимают слоты пула и упираются в бюджет ожидания.
        for _ in range(4):
            task_id, error = dispatch._publish_bounded(slow_signature, timeout_s=0.02)
            assert task_id == ""
            assert "timeout" in error
        # Пятый вызов: все слоты заняты — немедленный фолбэк без нового потока.
        t0 = _time.monotonic()
        task_id, error = dispatch._publish_bounded(slow_signature, timeout_s=1.0)
        assert _time.monotonic() - t0 < 0.1
        assert task_id == ""
        assert error == "publish pool busy"
        assert len(started) == 4
    finally:
        # Дождаться освобождения слотов и сбросить пул, чтобы не влиять на другие тесты.
        _time.sleep(0.25)
        dispatch.reset_dispatch_state()


def test_enabled_is_false_when_switched_off(monkeypatch):
    monkeypatch.setattr(config, "ENABLED", False)
    assert dispatch.enabled() is False


# ── beat scheduling / lease ─────────────────────────────────────────────────


def test_scheduled_scan_skips_when_another_worker_holds_the_lease(monkeypatch):
    """Beat has no cross-host singleton guarantee, so the lease is what makes the
    periodic scan run at most once per window."""
    monkeypatch.setattr("gex.workers.services.get_redis_client", lambda: object())
    monkeypatch.setattr(
        "gex.adapters.cache.lease.RedisLease.acquire",
        lambda self, name, ttl_s: None,  # someone else owns it
    )

    def must_not_run(**kwargs):
        raise AssertionError("scan must not start while the lease is held")

    monkeypatch.setattr(sc.scan_auto_run, "apply_async", must_not_run)

    assert sc.scan_auto_scheduled(universe="us")["skipped"] is True


def test_scheduled_scan_releases_lease_even_on_failure(monkeypatch):
    released = []
    monkeypatch.setattr("gex.workers.services.get_redis_client", lambda: object())
    monkeypatch.setattr(
        "gex.adapters.cache.lease.RedisLease.acquire",
        lambda self, name, ttl_s: "token-1",
    )
    monkeypatch.setattr(
        "gex.adapters.cache.lease.RedisLease.release",
        lambda self, name, token: released.append((name, token)) or True,
    )

    class _Res:
        def get(self, propagate=False):
            raise RuntimeError("scan blew up")

    monkeypatch.setattr(
        sc.scan_auto_run, "apply_async", lambda **kw: _Res()
    )

    with pytest.raises(RuntimeError):
        sc.scan_auto_scheduled(universe="crypto")

    assert released == [("scan:auto:crypto", "token-1")]
