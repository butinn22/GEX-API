"""Payload страницы: свежесть по классу, etag/304 и SWR (итерация 31).

Критерий итерации — «юнит-тесты по классам свежести». Проверяется главное продуктовое
обещание: **второй визит отдаёт payload мгновенно**, а свежесть добирается фоном.

Два уровня, и оба нужны:

* :class:`PagePayloadService` — превращает «страница + параметры» в окна свежести её класса
  (``fresh``/``stale_max``) и в заголовки ответа. Проверяется **по классам**: живая котировка,
  тяжёлый расчёт, периодическая страница — у каждого свои окна, и вне торговой сессии они шире
  (часы не делают данные свежее);
* :class:`RedisPageStore` — SWR и etag поверх конверта. Проверяется, что устаревшее отдаётся
  **сразу** (без ожидания пересчёта), что повторная выдача не запускает второй пересчёт,
  и что признак «идёт пересчёт» виден.

Сети нет: Redis подменяется фейком, время — управляемыми часами.

    python tests/test_page_payload.py
    pytest tests/test_page_payload.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from gex.adapters.cache.page_store import RedisPageStore  # noqa: E402
from gex.application.page_payload import PageNotFoundError, PagePayloadService  # noqa: E402
from gex.ports.cache import CachedPayload, CacheStatus  # noqa: E402


# ====================================================================== #
#  Двойники: скриптованный порт и Redis-фейк с управляемыми часами
# ====================================================================== #
class FakePort:
    """Порт кэша, который отдаёт заранее заданный статус и записывает окна."""

    def __init__(self, status=CacheStatus.MISS, value="payload", etag="e1", computing=False):
        self.status = status
        self.value = value
        self.etag = etag
        self.computing = computing
        self.calls: list[dict] = []
        self.invalidated: list[str] = []

    def get(self, key, *, fresh, stale_max, compute, max_wait_ms=None):
        self.calls.append(
            {"key": key, "fresh": fresh, "stale_max": stale_max, "max_wait_ms": max_wait_ms}
        )
        return CachedPayload(
            value=self.value,
            status=self.status,
            ts=1000.0,
            fresh=fresh,
            stale_max=stale_max,
            version=2,
            etag=self.etag,
            computing=self.computing,
        )

    def invalidate(self, key):
        self.invalidated.append(key)


class FakeRedis:
    """RedisClient-совместимый фейк: строки + SET NX PX (нужен single-flight) + сбой по флагу."""

    def __init__(self):
        self.kv: dict[str, bytes] = {}
        self.expiry: dict[str, float] = {}
        self.fail = False
        self.connected = True

    def _check(self):
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


def _store(redis, clock, **kwargs) -> RedisPageStore:
    return RedisPageStore(redis, clock=clock, start_threads=False, **kwargs)


class Clock:
    """Управляемые часы: ``advance`` двигает и wall-clock, и монотонное время."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ====================================================================== #
# 1. Политика по классам свежести
# ====================================================================== #
def _service(port: FakePort, **kwargs) -> PagePayloadService:
    return PagePayloadService(port, lambda *parts: "gex:res:" + ":".join(str(p) for p in parts), **kwargs)


def test_live_intraday_windows_in_session():
    """Живые данные в сессии: узкие окна — иначе страница показывала бы вчерашнюю цену."""
    port = FakePort()
    service = _service(port)
    service.get("ohlcv", lambda: "v", params=("SPY", "1h"), market_open=True)
    assert port.calls[0]["fresh"] == 30
    assert port.calls[0]["stale_max"] == 300


def test_live_intraday_windows_off_session():
    """Вне торгов окна шире: часы ничего не обновляют, частый пересчёт был бы впустую."""
    port = FakePort()
    _service(port).get("ohlcv", lambda: "v", params=("SPY",), market_open=False)
    assert port.calls[0]["fresh"] == 900
    assert port.calls[0]["stale_max"] == 7200


def test_computed_slow_windows():
    """Тяжёлый расчёт (конус): свежесть минутами — он не обязан пересчитываться каждую минуту."""
    port = FakePort()
    _service(port).get("gexcone", lambda: "v", params=("SPY",), market_open=True)
    assert port.calls[0]["fresh"] == 180
    assert port.calls[0]["stale_max"] == 3600


def test_periodic_scheduled_windows():
    """Периодическая страница обновляется по расписанию (6 ч), а не по запросу."""
    port = FakePort()
    _service(port).get("breadth-imoex", lambda: "v", market_open=True)
    assert port.calls[0]["fresh"] == 6 * 3600
    assert port.calls[0]["stale_max"] == 48 * 3600


def test_page_specific_windows_override_class_default():
    """У ``quote`` свои окна (3/30), а не классовые (30/300): котировка устаревает быстро."""
    port = FakePort()
    _service(port).get("quote", lambda: "v", params=("SPY",), market_open=True)
    assert port.calls[0]["fresh"] == 3
    assert port.calls[0]["stale_max"] == 30


def test_unknown_page_is_rejected():
    """Неизвестная страница — ошибка: молчаливый TTL «по умолчанию» и был источником литералов."""
    try:
        _service(FakePort()).get("несуществующая-страница", lambda: "v")
    except PageNotFoundError as exc:
        assert "не описана" in str(exc)
        return
    raise AssertionError("неизвестная страница принята")


def test_session_is_taken_from_injected_callable():
    """Сессию можно задать снаружи: иначе тест зависел бы от текущих часов."""
    state = {"open": False}
    port = FakePort()
    service = PagePayloadService(
        port, lambda *parts: "k", market_open=lambda: state["open"]
    )
    service.get("ohlcv", lambda: "v")
    assert port.calls[0]["fresh"] == 900, "взяты окна сессии, а не внесессионные"

    state["open"] = True
    service.get("ohlcv", lambda: "v")
    assert port.calls[1]["fresh"] == 30


def test_key_contains_page_and_params():
    port = FakePort()
    _service(port).get("gexcone", lambda: "v", params=("SPY", 5, "bybit"))
    assert port.calls[0]["key"] == "gex:res:gexcone:SPY:5:bybit"


def test_key_suffix_is_appended():
    port = FakePort()
    _service(port).get("gexcone", lambda: "v", params=("SPY",), key_suffix="v2")
    assert port.calls[0]["key"].endswith(":v2")


def test_page_class_is_exposed():
    assert _service(FakePort()).page_class("gexcone") == "computed_slow"


# ====================================================================== #
# 2. Заголовки ответа и 304
# ====================================================================== #
def test_stale_response_advertises_its_state():
    """Фронт должен знать, что данные устаревшие: индикатор ревалидации вместо догадок."""
    port = FakePort(status=CacheStatus.STALE, etag="abc")
    result = _service(port).get("gexcone", lambda: "v", market_open=True)
    headers = result.headers()
    assert headers["X-Cache"] == "stale"
    assert headers["ETag"] == '"abc"'
    assert headers["Cache-Control"] == "max-age=180"


def test_computing_is_visible_in_headers():
    port = FakePort(status=CacheStatus.COMPUTING, etag="abc", computing=True)
    result = _service(port).get("gexcone", lambda: "v", market_open=True)
    assert result.computing is True
    assert result.headers()["X-Computing"] == "1"
    assert result.headers()["X-Cache"] == "computing"


def test_304_when_etag_matches():
    port = FakePort(status=CacheStatus.HIT, etag="abc")
    result = _service(port).get("gexcone", lambda: "v", market_open=True)
    assert result.is_not_modified("abc") is True
    assert result.is_not_modified("другой") is False


def test_no_304_without_etag():
    """Пустой etag не должен означать «не изменилось»: иначе клиент остался бы без тела."""
    port = FakePort(status=CacheStatus.HIT, etag=None)
    result = _service(port).get("gexcone", lambda: "v", market_open=True)
    assert result.etag is None
    assert result.is_not_modified("abc") is False
    assert "ETag" not in result.headers()


def test_status_is_passed_through():
    for status in (CacheStatus.HIT, CacheStatus.STALE, CacheStatus.MISS, CacheStatus.COMPUTING):
        port = FakePort(status=status)
        assert _service(port).get("gexcone", lambda: "v", market_open=True).status == status


# ====================================================================== #
# 3. Хранилище: SWR, etag, признак пересчёта
# ====================================================================== #
def _page_store(redis=None, clock=None, **kwargs):
    redis = redis if redis is not None else FakeRedis()
    clock = clock or Clock()
    return _store(redis, clock, **kwargs), redis, clock


def test_miss_computes_once_and_caches():
    store, _, _ = _page_store()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"data": 42}

    first = store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    assert first.status == CacheStatus.MISS
    assert first.value == {"data": 42}
    assert first.etag, "etag не выставлен — 304 будет невозможен"

    second = store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    assert second.status == CacheStatus.HIT
    assert calls["n"] == 1, f"пересчёт при свежем значении: {calls['n']}"


def test_stale_is_returned_immediately_and_refreshed_in_background():
    """Главное обещание UX: пользователь не ждёт пересчёта."""
    store, _, clock = _page_store()
    values = {"n": 0}

    def compute():
        values["n"] += 1
        return {"v": values["n"]}

    store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    clock.advance(120)  # устарело, но ещё годно

    stale = store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    assert stale.status == CacheStatus.STALE
    assert stale.value == {"v": 1}, "отдано не прежнее значение"
    assert store.stats["refresh_started"] == 1, "фоновое обновление не запущено"

    # В тестовом режиме потоков нет — обновление выполняем явно.
    store.refresh_now("gex:res:x", 30, compute)
    fresh_again = store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    assert fresh_again.status == CacheStatus.HIT
    assert fresh_again.value == {"v": 2}


def test_too_old_recomputes_synchronously():
    store, _, clock = _page_store()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"v": calls["n"]}

    store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    clock.advance(400)  # за пределом stale_max

    assert store.get("gex:res:x", fresh=30, stale_max=300, compute=compute).status == CacheStatus.MISS
    assert calls["n"] == 2


def test_computing_flag_comes_from_another_process_lease():
    """Признак «идёт пересчёт» — это занятая аренда, а не второй источник правды."""
    from gex.adapters.cache.singleflight import RedisSingleFlight

    redis = FakeRedis()
    store, _, clock = _page_store(redis)
    store.get("gex:res:x", fresh=30, stale_max=300, compute=lambda: {"v": 1})
    clock.advance(120)
    redis.kv[RedisSingleFlight.lock_key("gex:res:x")] = b"another-worker"  # лидер — другой процесс

    result = store.get("gex:res:x", fresh=30, stale_max=300, compute=lambda: {"v": 2})
    assert result.status == CacheStatus.COMPUTING and result.computing is True
    assert store.stats["computing"] == 1


def test_legacy_envelope_is_not_fresh():
    """Запись прежней схемы (без версии) не считается свежей — её пересчитывают."""
    store, redis, _ = _page_store()
    from gex.adapters.cache.envelope import Envelope

    legacy = Envelope(value={"old": True}, stored_at=store._now(), ttl=0, version=0)
    redis.kv["gex:res:x"] = json.dumps(legacy.as_payload()).encode()
    result = store.get("gex:res:x", fresh=30, stale_max=300, compute=lambda: {"new": True})
    assert result.status == CacheStatus.MISS
    assert result.value == {"new": True}


def test_invalidate_forces_recompute():
    store, _, _ = _page_store()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"v": calls["n"]}

    store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    store.invalidate("gex:res:x")
    assert store.get("gex:res:x", fresh=30, stale_max=300, compute=compute).status == CacheStatus.MISS
    assert calls["n"] == 2


def test_store_without_redis_still_serves_payload():
    """Redis недоступен — это не отказ: считаем и отдаём, но без кэша."""
    store = RedisPageStore(None, start_threads=False)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"ok": True}

    result = store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    assert result.value == {"ok": True} and result.status == CacheStatus.MISS
    assert calls["n"] == 1
    assert store.describe()["available"] is False


def test_redis_failure_does_not_break_the_response():
    """Сбой Redis на пересчёте не должен ломать запрос: значение считается и отдаётся."""
    redis = FakeRedis()
    store, _, _ = _page_store(redis)
    redis.fail = True
    result = store.get("gex:res:x", fresh=30, stale_max=300, compute=lambda: {"v": 1})
    assert result.value == {"v": 1}


def test_stats_make_degradation_visible():
    """Без счётчиков «кэш работает» и «кэш не работает» выглядят одинаково."""
    store, _, clock = _page_store()
    compute = lambda: {"v": 1}  # noqa: E731
    store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    clock.advance(120)
    store.get("gex:res:x", fresh=30, stale_max=300, compute=compute)
    assert store.stats["miss"] == 1 and store.stats["hit"] == 1 and store.stats["stale"] == 1


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- page payload: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
