"""Страницы широты рынка: ответ из кэша, а не из запроса.

Что здесь проверяется и почему именно это
-----------------------------------------
Требование к трём страницам (``/breadth``, ``/sector/breadth``, ``/breadth-imoex``) —
«пользователь не ждёт провайдера». Из него следуют проверяемые инварианты:

1. **В запросе нет вычислений.** Обработчик отвечает из снапшота; если снапшота нет, он
   отвечает «готовится» и просит фоновый пересчёт. Тест ломает вычисление (подменяет
   ``build_payload`` на падающий) и убеждается, что ответ всё равно есть.
2. **Устаревшее лучше пустого.** Снапшот старше окна свежести, но моложе ``stale_max``,
   отдаётся с ``200`` и признаком ``X-Cache: stale``, а обновление просится фоном.
3. **Запрос не висит.** Просьба о пересчёте отправляется асинхронно: даже если брокер
   «задумался» (в тесте — ``sleep`` больше окна ожидания), обработчик отвечает в пределах
   холодного окна. Это ровно та жалоба, из-за которой страницы переписывались.
4. **Расписание существует.** Beat содержит пятиминутный пересчёт страниц и слоты МСК
   для широты IMOEX (единственной страницы с суточной квотой провайдера).

Redis-хранилище подменяется: тесты проверяют проводку и контракт, а не сеть.
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gex.auth.dependencies as auth_deps
import gex.routers.breadth_sector_router as br_mod
import gex.workers.config as workers_config
import gex.workers.local_refresh as local_refresh
from gex.adapters.cache.keys import page_key
from gex.adapters.cache.page_store import RedisPageStore
from gex.application import market_pages
from gex.auth.dependencies import get_current_user
from gex.deps import provide_page_store
from gex.ports.cache import CacheStatus


# ====================================================================== #
#  Двойники хранилища и очереди
# ====================================================================== #
class FakeRedis:
    """Минимальный RedisClient-совместимый фейк (``get``/``set``/``delete``)."""

    def __init__(self, *, fail: bool = False):
        self.kv: dict[str, bytes] = {}
        self.fail = fail
        self.connected = True

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("redis down")

    def get(self, key):
        self._check()
        return self.kv.get(key)

    def set(self, key, value, ex=None, *, px=None, nx=False):
        self._check()
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def delete(self, key):
        self._check()
        return self.kv.pop(key, None) is not None


class Clock:
    """Управляемые часы: возраст значения задаётся тестом, а не ожиданием."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _store(redis=None, clock=None) -> RedisPageStore:
    return RedisPageStore(redis, clock=clock or Clock(), start_threads=False)


def _disable_cold_wait(monkeypatch, seconds: float = 0.05) -> None:
    """Холодное ожидание — из конфигурации; в тестах оно не должно стоить секунд."""
    monkeypatch.setattr(workers_config, "MARKET_COLD_WAIT_S", seconds)


@pytest.fixture(autouse=True)
def _clean_local_refresh():
    """Аварийный пересчёт — состояние процесса: между тестами его нельзя переносить."""
    local_refresh.reset()
    yield
    local_refresh.reset()


@pytest.fixture
def dispatch_log(monkeypatch):
    """Заглушка отправки фонового пересчёта: пишет вызовы, не ходит никуда."""
    calls: list[tuple[str, str | None, bool]] = []

    def fake_dispatch(page, mode=None, *, force=False):
        calls.append((page, mode, force))
        from gex.workers.dispatch import RefreshTicket

        return RefreshTicket(page=page, mode=mode, channel="celery", started=True, pending=False)

    monkeypatch.setattr(br_mod, "dispatch_refresh", fake_dispatch)
    return calls


def _wait_for(calls: list, expected: int, timeout: float = 2.0) -> None:
    """Дождаться фоновой просьбы о пересчёте.

    Просьба уходит в поток (обработчик её не ждёт — в этом весь смысл), поэтому тест обязан
    подождать её появления, а не проверять лог сразу после ответа.
    """
    deadline = time.monotonic() + timeout
    while len(calls) < expected and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(calls) >= expected, f"фоновая просьба о пересчёте не отправлена: {calls}"


@pytest.fixture
def make_client(monkeypatch):
    """Фабрика тестовых клиентов с подменёнными барьерами авторизации и хранилищем.

    Клиент создаётся **внутри** ``with TestClient(...)`` и живёт до конца теста. Без этого
    каждый запрос поднимает и гасит свой портал событий, а гашение ждёт фоновые потоки —
    и тест «обработчик не ждёт брокера» мерил бы уже не обработчик, а остановку портала.
    """
    opened: list = []

    def _make(store, *, admin: bool = True) -> TestClient:
        monkeypatch.setattr(auth_deps, "can_bypass_barriers", lambda user: admin)
        app = FastAPI()
        app.include_router(br_mod.router)
        app.dependency_overrides[get_current_user] = lambda: object()
        app.dependency_overrides[provide_page_store] = lambda: store
        context = TestClient(app)
        client = context.__enter__()
        opened.append(context)
        return client

    yield _make
    for context in opened:
        context.__exit__(None, None, None)


def _seed(store, page: str, mode: str | None, payload: dict, *, age_s: float = 0.0):
    """Положить снапшот страницы и состарить его на ``age_s`` секунд."""
    key = page_key(page, mode) if mode else page_key(page)
    store.write(key, payload, fresh=market_pages.retention_s(page), source="test")
    if age_s:
        # Возраст двигаем сдвигом часов хранилища: метка записи — wall-clock.
        store._clock = lambda: store._mem[key][0] + age_s  # noqa: SLF001 — тестовый доступ
        store._cache._clock = store._clock  # noqa: SLF001
    return key


BREADTH_PAYLOAD = {
    "market": {"dates": ["2026-09-22"], "es": [1.0]},
    "stocks": {"dates": ["2026-09-22"], "above50": [55.0]},
    "current": {"day": "2026-09-22", "state": "UPCOHER"},
}
SECTOR_PAYLOAD = {"composite": {"dates": ["2026-09-22"], "values": [1.0], "current": 1.0},
                  "per_sector": {}, "trend": {}, "macd": {}, "ema20": []}


# ====================================================================== #
#  1. Хранилище снапшотов: peek не считает, память спасает при Redis-down
# ====================================================================== #
class TestSnapshotStore:

    def test_peek_returns_none_when_nothing_stored(self):
        store = _store()
        assert store.peek(page_key("sector"), fresh=600, stale_max=3600) is None
        assert store.stats["peek_empty"] == 1

    def test_write_then_peek_is_hit(self):
        store = _store()
        key = page_key(market_pages.PAGE_SECTOR)
        store.write(key, SECTOR_PAYLOAD, fresh=600, source="test")
        snap = store.peek(key, fresh=600, stale_max=3600)
        assert snap is not None and snap.status is CacheStatus.HIT
        assert snap.value == SECTOR_PAYLOAD and snap.age_s == pytest.approx(0.0, abs=0.05)

    def test_peek_marks_stale_inside_stale_window(self):
        clock = Clock()
        store = _store(clock=clock)
        key = page_key(market_pages.PAGE_BREADTH, "mags")
        store.write(key, BREADTH_PAYLOAD, fresh=300, source="test")
        clock.advance(600)
        snap = store.peek(key, fresh=300, stale_max=3600)
        assert snap is not None and snap.status is CacheStatus.STALE

    def test_peek_marks_expired_beyond_stale_window(self):
        """Протухшее значение различимо от отсутствующего: по нему видно, что данные были."""
        clock = Clock()
        store = _store(FakeRedis(), clock=clock)
        key = page_key(market_pages.PAGE_BREADTH, "mags")
        store.write(key, BREADTH_PAYLOAD, fresh=300, source="test")
        clock.advance(10_000)
        snap = store.peek(key, fresh=300, stale_max=3600)
        assert snap is not None and snap.status is CacheStatus.EXPIRED
        assert snap.value == BREADTH_PAYLOAD

    def test_memory_tier_serves_after_redis_failure(self):
        """Redis упал — страница обязана отдать последнее значение, а не считать заново."""
        redis = FakeRedis()
        store = _store(redis)
        key = page_key(market_pages.PAGE_SECTOR)
        store.write(key, SECTOR_PAYLOAD, fresh=600, source="test")

        redis.fail = True  # Redis недоступен: конверт не читается
        snap = store.peek(key, fresh=600, stale_max=3600)
        assert snap is not None and snap.value == SECTOR_PAYLOAD

    def test_memory_tier_respects_retention(self):
        """Память не архив: слишком старое значение забывается, а не отдаётся вечно."""
        clock = Clock()
        store = _store(clock=clock)
        key = page_key(market_pages.PAGE_SECTOR)
        store.write(key, SECTOR_PAYLOAD, fresh=600, source="test")
        clock.advance(600 * 2 + 10)  # retention = 2 x fresh
        assert store.peek(key, fresh=600, stale_max=86_400) is None

    def test_write_without_redis_is_memory_only(self):
        store = _store(None)
        key = page_key(market_pages.PAGE_BREADTH, "top10")
        store.write(key, BREADTH_PAYLOAD, fresh=300, source="test")
        assert store.peek(key, fresh=300, stale_max=3600).value == BREADTH_PAYLOAD

    def test_invalidate_forgets_both_tiers(self):
        store = _store()
        key = page_key(market_pages.PAGE_SECTOR)
        store.write(key, SECTOR_PAYLOAD, fresh=600, source="test")
        store.invalidate(key)
        assert store.peek(key, fresh=600, stale_max=3600) is None


# ====================================================================== #
#  2. Маршруты: вычислений в запросе нет
# ====================================================================== #
class TestRoutesServeCacheOnly:

    def test_breadth_returns_cached_payload(self, make_client, dispatch_log):
        store = _store()
        _seed(store, market_pages.PAGE_BREADTH, "mags", BREADTH_PAYLOAD)
        client = make_client(store)

        r = client.get("/breadth?giants=mags")

        assert r.status_code == 200
        assert r.json()["current"]["day"] == "2026-09-22"
        assert r.json()["meta"]["served_from_cache"] is True
        assert r.headers["X-Cache"] == "hit"
        assert "X-Cache-Age" in r.headers
        # Свежее значение — фоновый пересчёт не нужен и не просится.
        assert dispatch_log == []

    def test_stale_payload_is_served_and_refresh_requested(self, make_client, dispatch_log):
        store = _store()
        _seed(store, market_pages.PAGE_BREADTH, "mags", BREADTH_PAYLOAD, age_s=400)
        client = make_client(store)

        r = client.get("/breadth?giants=mags")

        assert r.status_code == 200
        assert r.headers["X-Cache"] == "stale"
        assert r.json()["meta"]["stale"] is True
        _wait_for(dispatch_log, 1)
        assert dispatch_log == [(market_pages.PAGE_BREADTH, "mags", False)]

    def test_cold_page_answers_503_without_computing(self, monkeypatch, make_client, dispatch_log):
        """Холодный старт: ответ «готовится» + просьба о пересчёте. Вычислений нет."""
        store = _store()
        client = make_client(store)
        _disable_cold_wait(monkeypatch)

        def explode(*args, **kwargs):
            raise AssertionError("расчёт не имеет права выполняться внутри запроса")

        monkeypatch.setattr(market_pages, "build_payload", explode)

        r = client.get("/breadth?giants=mags")

        assert r.status_code == 503
        assert r.headers["X-Cache"] == "empty"
        assert r.headers["Retry-After"]
        _wait_for(dispatch_log, 1)
        assert dispatch_log == [(market_pages.PAGE_BREADTH, "mags", False)]

    def test_cold_page_does_not_wait_for_slow_broker(self, monkeypatch, make_client, dispatch_log):
        """Ключевое требование: «висит до таймаута» больше невозможно.

        Просьба о пересчёте уходит в поток, а ответ ограничен холодным окном: даже если
        публикация в очередь «задумалась» на пять секунд, обработчик отвечает раньше.
        """
        store = _store()
        client = make_client(store)
        _disable_cold_wait(monkeypatch, 0.2)

        def slow_dispatch(page, mode=None, *, force=False):
            time.sleep(1.5)
            from gex.workers.dispatch import RefreshTicket

            return RefreshTicket(page=page, mode=mode, channel="celery", started=True, pending=False)

        monkeypatch.setattr(br_mod, "dispatch_refresh", slow_dispatch)

        started = time.monotonic()
        r = client.get("/sector/breadth")
        elapsed = time.monotonic() - started

        assert r.status_code == 503
        assert elapsed < 1.0, f"обработчик ждал брокера {elapsed:.1f} c"

    def test_sector_uses_own_snapshot(self, monkeypatch, make_client, dispatch_log):
        store = _store()
        _seed(store, market_pages.PAGE_SECTOR, None, SECTOR_PAYLOAD)
        client = make_client(store)

        r = client.get("/sector/breadth")

        assert r.status_code == 200
        assert r.json()["composite"]["current"] == 1.0
        assert dispatch_log == []

    def test_breadth_modes_are_separate_keys(self, monkeypatch, make_client, dispatch_log):
        store = _store()
        _seed(store, market_pages.PAGE_BREADTH, "mags", BREADTH_PAYLOAD)
        client = make_client(store)
        _disable_cold_wait(monkeypatch)

        assert client.get("/breadth?giants=mags").status_code == 200
        r_top10 = client.get("/breadth?giants=top10")
        assert r_top10.status_code == 503          # своего снапшота у top10 ещё нет
        _wait_for(dispatch_log, 1)
        assert dispatch_log == [(market_pages.PAGE_BREADTH, "top10", False)]

    def test_invalid_mode_is_rejected_by_schema(self, make_client):
        store = _store()
        client = make_client(store)
        assert client.get("/breadth?giants=hackers").status_code == 422


# ====================================================================== #
#  3. IMOEX: снапшот → хранилище сервиса → 503
# ====================================================================== #
class TestImoexRoute:

    def test_snapshot_wins(self, make_client, dispatch_log):
        store = _store()
        _seed(store, market_pages.PAGE_IMOEX, None, {"current": {"day": "2026-09-22"}})
        client = make_client(store)

        r = client.get("/breadth-imoex")

        assert r.status_code == 200
        assert r.headers["X-Cache"] == "hit"
        assert dispatch_log == []

    def test_service_storage_is_the_fallback(self, monkeypatch, make_client, dispatch_log):
        """Очередь недоступна, снапшота нет — но расчёт сервиса обязан спасти страницу."""
        store = _store()
        client = make_client(store)

        async def fake_read():
            return {"current": {"day": "2026-09-21"}, "meta": {"last_completed_day": "2026-09-21"}}

        monkeypatch.setattr(br_mod, "_read_imoex", fake_read)

        r = client.get("/breadth-imoex")

        assert r.status_code == 200
        assert r.headers["X-Cache"] == "service"
        assert r.json()["current"]["day"] == "2026-09-21"
        assert dispatch_log == []

    def test_cold_imoex_requests_refresh(self, monkeypatch, make_client, dispatch_log):
        store = _store()
        client = make_client(store)

        async def empty_read():
            return None

        monkeypatch.setattr(br_mod, "_read_imoex", empty_read)

        r = client.get("/breadth-imoex")

        assert r.status_code == 503
        _wait_for(dispatch_log, 1)
        assert dispatch_log == [(market_pages.PAGE_IMOEX, None, False)]

    def test_admin_refresh_is_accepted_not_blocking(self, monkeypatch, make_client):
        """Раньше этот POST парсил ISS внутри запроса и висел минутами."""
        from gex.workers import dispatch
        from gex.workers.tasks import market_pages as task_mod

        store = _store()
        client = make_client(store)
        dispatch.reset_dispatch_state()

        class FakeSignature:
            def apply_async(self, retry=False):
                assert retry is False, "повтор публикации внутри запроса недопустим"

                class _Result:
                    id = "task-123"

                return _Result()

        monkeypatch.setattr(task_mod, "admin_signature", lambda page, mode=None: FakeSignature())

        r = client.post("/breadth-imoex/refresh")

        assert r.status_code == 202
        assert r.json()["task_id"] == "task-123"
        assert r.json()["channel"] == "celery"

    def test_admin_refresh_falls_back_when_broker_is_down(self, monkeypatch, make_client):
        from gex.workers import dispatch
        from gex.workers.tasks import market_pages as task_mod

        store = _store()
        client = make_client(store)
        dispatch.reset_dispatch_state()

        def broken_signature(page, mode=None):
            raise ConnectionError("broker down")

        local_calls: list = []
        monkeypatch.setattr(task_mod, "admin_signature", broken_signature)
        monkeypatch.setattr(task_mod, "refresh_page_now", lambda *a, **k: local_calls.append(k))

        r = client.post("/breadth-imoex/refresh")

        assert r.status_code == 202
        assert r.json()["channel"] == "local"
        _wait_for(local_calls, 1)
        assert local_calls[0]["bypass_rate_limit"] is True


# ====================================================================== #
#  4. Пересчёт: пишет снапшот, плановый режим уважает свежесть
# ====================================================================== #
class TestRefreshTask:

    def test_refresh_writes_snapshot_with_meta(self, monkeypatch):
        from gex.workers.tasks import market_pages as task_mod

        store = _store()
        monkeypatch.setattr(
            market_pages, "build_payload",
            lambda page, mode=None, **kw: {"current": {"day": "2026-09-22", "state": "UPCOHER"}},
        )

        result = task_mod.refresh_page_now(
            market_pages.PAGE_BREADTH, "mags", store=store
        )

        assert result["ok"] is True
        assert result["data_day"] == "2026-09-22"
        snap = store.peek(page_key(market_pages.PAGE_BREADTH, "mags"), fresh=300, stale_max=3600)
        assert snap is not None
        assert snap.value["meta"]["page"] == market_pages.PAGE_BREADTH
        assert snap.value["meta"]["data_day"] == "2026-09-22"

    def test_planned_refresh_skips_fresh_value(self, monkeypatch):
        from gex.workers.tasks import market_pages as task_mod

        store = _store()
        _seed(store, market_pages.PAGE_SECTOR, None, SECTOR_PAYLOAD)

        def explode(*args, **kwargs):
            raise AssertionError("свежее значение пересчитывать не нужно")

        monkeypatch.setattr(market_pages, "build_payload", explode)

        result = task_mod.refresh_page_now(market_pages.PAGE_SECTOR, None, force=False, store=store)

        assert result["ok"] is True and result["skipped"] is True
        assert result["reason"] == "fresh"

    def test_planned_refresh_rebuilds_stale_value(self, monkeypatch):
        from gex.workers.tasks import market_pages as task_mod

        store = _store()
        _seed(store, market_pages.PAGE_SECTOR, None, SECTOR_PAYLOAD, age_s=40_000)
        monkeypatch.setattr(
            market_pages, "build_payload", lambda page, mode=None, **kw: SECTOR_PAYLOAD
        )

        result = task_mod.refresh_page_now(market_pages.PAGE_SECTOR, None, force=False, store=store)

        assert result["ok"] is True and not result.get("skipped")

    def test_failed_refresh_reports_error_instead_of_raising(self, monkeypatch):
        from gex.workers.tasks import market_pages as task_mod

        store = _store()

        def explode(page, mode=None, **kw):
            raise RuntimeError("провайдер не ответил")

        monkeypatch.setattr(market_pages, "build_payload", explode)

        result = task_mod.refresh_page_now(market_pages.PAGE_BREADTH, "mags", store=store)

        assert result["ok"] is False
        assert "провайдер не ответил" in result["error"]


# ====================================================================== #
#  5. Расписание Beat и аварийный пересчёт
# ====================================================================== #
class TestScheduleAndFallback:

    def test_beat_refreshes_pages_every_five_minutes(self):
        from gex.workers.celery_app import MARKET_QUEUE, app

        entry = app.conf.beat_schedule["market-refresh-every-5m"]
        assert entry["task"] == "gex.market.refresh_all"
        assert float(entry["schedule"]) == 300.0
        assert entry["options"]["queue"] == MARKET_QUEUE

    def test_beat_has_msk_slots_for_periodic_pages(self):
        """Слоты МСК нельзя задавать «по часам»: Beat живёт в UTC."""
        from gex.domain.schedule import PERIODIC_SLOTS
        from gex.workers.celery_app import app

        entry = app.conf.beat_schedule["market-refresh-breadth-imoex-msk-slots"]
        assert entry["kwargs"] == {"page": "breadth-imoex", "force": True}
        assert PERIODIC_SLOTS["breadth-imoex"] == ((23, 0), (8, 0))
        # 23:00 и 08:00 МСК == 20:00 и 05:00 UTC.
        assert entry["schedule"].hour == {5, 20}
        assert entry["schedule"].minute == {0}

    def test_market_tasks_are_routed_to_market_queue(self):
        from gex.workers.celery_app import MARKET_QUEUE, TASK_ROUTES

        assert TASK_ROUTES["gex.market.*"]["queue"] == MARKET_QUEUE

    def test_admin_signature_uses_dedicated_imoex_task(self):
        """Ручное обновление IMOEX — отдельная задача: она ходит в ISS в обход квоты."""
        from gex.workers.celery_app import MARKET_QUEUE
        from gex.workers.tasks import market_pages as task_mod

        sig = task_mod.admin_signature(market_pages.PAGE_IMOEX)
        assert sig.task == "gex.market.refresh_imoex_admin"
        assert sig.options.get("queue") == MARKET_QUEUE

    def test_local_refresh_dedupes_and_cools_down(self):
        calls: list[str] = []
        key = "gex:page:sector"

        assert local_refresh.submit(key, lambda: calls.append("a"))[0] is True
        # Пока пересчёт идёт, второй запуск не нужен: это защита бюджета провайдера.
        started, reason = local_refresh.submit(key, lambda: calls.append("b"))
        assert started is False and reason in {"already_running", "cooldown"}

    def test_local_refresh_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(workers_config, "MARKET_LOCAL_REFRESH", False)
        started, reason = local_refresh.submit("k", lambda: None)
        assert started is False and reason == "disabled"

    def test_hung_local_refresh_releases_its_claim(self, monkeypatch):
        """Зависший пересчёт не имеет права держать страницу в «данных нет» навсегда.

        Поток Python прервать нельзя, поэтому единственное, что можно сделать, — снять метку
        «идёт пересчёт» по таймауту: следующий запрос запустит попытку заново.
        """
        monkeypatch.setattr(workers_config, "MARKET_LOCAL_REFRESH_MAX_S", 0.05)
        key = "gex:page:breadth:mags"

        started, _ = local_refresh.submit(key, lambda: time.sleep(5.0))
        assert started is True
        assert local_refresh.is_pending(key) is True

        time.sleep(0.1)
        assert local_refresh.is_pending(key) is False, "зависший пересчёт обязан освободить ключ"

    def test_dispatch_prefers_celery_then_falls_back_locally(self, monkeypatch):
        from gex.workers import dispatch
        from gex.workers.tasks import market_pages as task_mod

        monkeypatch.setattr(task_mod, "refresh_signature", None)

        def broken_signature(*args, **kwargs):
            raise ConnectionError("broker down")

        monkeypatch.setattr(task_mod, "refresh_signature", broken_signature)
        monkeypatch.setattr(task_mod, "refresh_page_now", lambda *a, **k: {"ok": True})
        monkeypatch.setattr(dispatch, "RefreshTicket", dispatch.RefreshTicket)

        ticket = dispatch.dispatch_refresh(market_pages.PAGE_SECTOR, None)

        assert ticket.channel == "local" and ticket.started is True
