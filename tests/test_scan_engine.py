"""Сканер под арендой: один сканер на кластер, состояние в Redis (итерация 33).

Что здесь проверяется и почему это важно
---------------------------------------
В рабочей роли стартуют шесть фоновых циклов. До этой итерации каждая реплика запускала
свои, поэтому провайдер получал нагрузку ×N, а Telegram — ×N одинаковых уведомлений
(пользователю это видно как дубли сообщений). Плюс состояние сканера жило в памяти
процесса, поэтому API-роль (`web`) всегда отвечала «ещё не сканировали».

Проверки:

* **владелец один из N** — сканирует ровно одна реплика, остальные не сканируют вовсе;
* **состояние читается из другого процесса** — «роутер читает Redis»: у читателя нет
  ни аренды, ни сканера, а состояние он видит;
* **продление аренды** — долгий прогон не теряет владение;
* **перехват после смерти владельца** — работа продолжается без ручного вмешательства;
* **потеря аренды останавливает сервис** — иначе прежний владелец сканировал бы параллельно
  с новым, то есть дефект вернулся бы в худшем виде.

    python tests/test_scan_engine.py
    pytest tests/test_scan_engine.py -q
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.cache.lease import RedisLease  # noqa: E402
from gex.application.scan import ScanEngine, ScanState, supervise  # noqa: E402

LEASE_TTL = 120


class FakeRedis:
    """Redis с ``SET NX PX`` и истечением по управляемым часам."""

    def __init__(self, clock):
        self.kv: dict[str, tuple[bytes, float | None]] = {}
        self._clock = clock
        self.fail = False

    def _purge(self, key):
        entry = self.kv.get(key)
        if entry and entry[1] is not None and self._clock() >= entry[1]:
            self.kv.pop(key, None)

    def set(self, key, value, ex=None, *, px=None, nx=False):
        if self.fail:
            raise ConnectionError("redis down")
        self._purge(key)
        if nx and key in self.kv:
            return None
        ttl = None
        if px is not None:
            ttl = self._clock() + px / 1000.0
        elif ex is not None:
            ttl = self._clock() + ex
        self.kv[key] = (value if isinstance(value, bytes) else str(value).encode(), ttl)
        return True

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        self._purge(key)
        entry = self.kv.get(key)
        return entry[0] if entry else None

    def delete(self, key):
        self._purge(key)
        return self.kv.pop(key, None) is not None


class Clock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


class SharedState:
    """Хранилище состояния, общее для «процессов» (в проде это Redis)."""

    def __init__(self):
        self.payload: dict | None = None

    def read(self):
        return self.payload

    def write(self, payload):
        self.payload = payload


class CountingScan:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return _Report(ok=3, failed=1, total=4)


class _Report:
    def __init__(self, ok, failed, total):
        self.ok, self.failed, self.total = ok, failed, total


def _engine(name, redis, clock, state, **kwargs) -> ScanEngine:
    return ScanEngine(
        name, RedisLease(redis), read_state=state.read, write_state=state.write,
        clock=clock, lease_ttl_s=LEASE_TTL, **kwargs
    )


# ====================================================================== #
# 1. Один сканер на кластер
# ====================================================================== #
def test_only_one_worker_scans():
    """Три реплики на одном Redis: прогон выполняет ровно одна.

    Имя аренды — это **вид работы**, а не реплика: все три движка берут одно имя ``scan``.
    Если дать разные имена, аренды будут независимыми и сканировать станут все три —
    именно так выглядела ошибка в первой версии этого теста.
    """
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    scans = [CountingScan() for _ in range(3)]
    engines = [_engine("scan", redis, clock, state) for _ in range(3)]

    results = [e.run_once(s) for e, s in zip(engines, scans)]
    assert sum(1 for r in results if r is not None) == 1, "сканировали не одна реплика"
    assert [s.calls for s in scans] == [1, 0, 0]
    assert engines[1].skipped_not_owner == 1 and engines[2].skipped_not_owner == 1


def test_different_scan_names_are_independent():
    """Разные виды работы не мешают друг другу: аренда на вид, а не на процесс."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    a, b = _engine("scan:auto_us", redis, clock, state), _engine("scan:auto_ru", redis, clock, state)
    assert a.run_once(CountingScan()) is not None
    assert b.run_once(CountingScan()) is not None, "второй вид работы заблокирован чужой арендой"


def test_non_owner_does_not_scan_even_repeatedly():
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    owner, follower = _engine("scan", redis, clock, state), _engine("scan", redis, clock, state)
    owner_calls, follower_calls = CountingScan(), CountingScan()

    owner.run_once(owner_calls)
    for _ in range(3):
        follower.run_once(follower_calls)
    assert owner_calls.calls == 1
    assert follower_calls.calls == 0, "не-владелец всё-таки сканировал"


def test_owner_without_lease_scans():
    """Аренда не настроена (нет Redis) — работаем как единственный процесс."""
    clock = Clock()
    engine = ScanEngine("scan", None, clock=clock)
    assert engine.try_become_owner() is True and engine.is_owner is True
    assert engine.run_once(CountingScan()) is not None


# ====================================================================== #
# 2. Состояние видно из другого процесса
# ====================================================================== #
def test_state_is_readable_from_a_process_without_scanner():
    """API-роль не запускает сканеры, но обязана видеть состояние — иначе «ещё не сканировали»."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()

    worker = _engine("scan:auto_us", redis, clock, state)
    assert worker.run_once(CountingScan()) is not None

    reader = ScanEngine("scan:auto_us", None, read_state=state.read)  # у читателя нет аренды
    published = reader.read_state()
    assert published is not None
    assert published.name == "scan:auto_us"
    assert published.scanned == 3 and published.total == 4 and published.failed == 1
    assert published.running is False


def test_state_marks_running_during_the_scan():
    """Пока прогон идёт, состояние обязано это показывать (иначе индикатор врёт)."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    engine = _engine("scan", redis, clock, state)
    seen = {}

    def scan():
        seen.update(state.payload or {})
        return _Report(ok=1, failed=0, total=1)

    engine.run_once(scan)
    assert seen.get("running") is True, "во время прогона состояние не помечено running"
    assert (state.payload or {}).get("running") is False


def test_failed_run_is_published():
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    engine = _engine("scan", redis, clock, state)

    def boom():
        raise RuntimeError("провайдер недоступен")

    try:
        engine.run_once(boom)
    except RuntimeError:
        pass
    else:
        raise AssertionError("исключение проглотано")
    assert "провал" in (state.payload or {}).get("note", "") or \
           "упал" in (state.payload or {}).get("note", "")


def test_corrupt_state_is_ignored():
    """Мусор в состоянии не должен ронять чтение: это подсказка для UI, а не источник правды."""
    assert ScanState.from_payload(None) is None
    assert ScanState.from_payload({"version": 1}) is None
    assert ScanState.from_payload({"version": 99, "name": "x"}) is None
    assert ScanState.from_payload({"version": 1, "name": ""}) is None


# ====================================================================== #
# 3. Продление и перехват
# ====================================================================== #
def test_long_run_keeps_ownership_when_heartbeat_renews():
    """Пока идёт прогон, аренда продлевается сердцебиением — владение не теряется."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    engine = ScanEngine(
        "scan", RedisLease(redis), read_state=state.read, write_state=state.write,
        clock=clock, lease_ttl_s=2,  # короткая аренда: пульс продлевает её многократно
    )
    renews = []
    original_renew = engine._lease.renew

    def counting_renew(name, token, ttl_s):
        renews.append(name)
        return original_renew(name, token, ttl_s)

    engine._lease.renew = counting_renew

    def slow_scan():
        import time as _t

        _t.sleep(0.25)  # прогон дольше TTL (2 с? нет — TTL=2с, пульс 0.66с: продлений ≥1)
        return _Report(ok=1, failed=0, total=1)

    engine.run_once(slow_scan)
    assert len(renews) >= 1, "сердцебиение не продлевало аренду во время прогона"
    assert engine.is_owner is True
    follower = _engine("scan", redis, clock, state)
    assert follower.try_become_owner() is False, "владение ушло другой реплике"


def test_renewal_failure_drops_ownership():
    """Продление не удалось — владение отпускается: иначе два сканера пойдут параллельно."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    engine = _engine("scan", redis, clock, state)
    engine.try_become_owner()
    engine._lease.renew = lambda *a, **k: False  # аренда «истекла и перехвачена»

    engine.run_once(CountingScan())
    assert engine.is_owner is False, "движок остался владельцем без аренды"

    # Аренду НЕ удаляем: ключ может уже принадлежать другому процессу, а удалять чужую
    # аренду — это ровно тот дефект, от которого защищает сверка токена. Ключ истечёт сам.
    follower = _engine("scan", redis, clock, state)
    assert follower.try_become_owner() is False, "чужая аренда снята"
    clock.advance(LEASE_TTL + 1)
    assert follower.try_become_owner() is True, "после истечения аренды владение не перешло"


def test_takeover_after_owner_dies():
    """Владелец пропал (аренда истекла) — работу продолжает живая реплика."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    dead = _engine("scan", redis, clock, state)
    alive = _engine("scan", redis, clock, state)

    dead_calls = CountingScan()
    assert dead.run_once(dead_calls) is not None
    alive_calls = CountingScan()
    assert alive.run_once(alive_calls) is None, "аренда ещё жива"

    clock.advance(LEASE_TTL + 1)  # владелец «умер», аренда истекла
    assert alive.run_once(alive_calls) is not None, "перехват не состоялся"
    assert alive.takeovers == 1
    assert dead_calls.calls == 1 and alive_calls.calls == 1


def test_release_frees_the_lease_for_others():
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    owner = _engine("scan", redis, clock, state)
    other = _engine("scan", redis, clock, state)

    owner.try_become_owner()
    assert other.try_become_owner() is False
    owner.release()
    assert other.try_become_owner() is True


def test_redis_failure_does_not_grant_ownership():
    """Redis недоступен — владение не берём: лучше не сканировать, чем сканировать N раз."""
    clock = Clock()
    redis = FakeRedis(clock)
    redis.fail = True
    engine = _engine("scan", redis, clock, SharedState())
    assert engine.try_become_owner() is False


# ====================================================================== #
# 4. Цикл и супервизор сервисов
# ====================================================================== #
def test_loop_scans_on_interval_not_every_heartbeat():
    """Владелец сканирует по интервалу, а не на каждый пульс (иначе нагрузка ×N)."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    engine = _engine("scan", redis, clock, state)
    scans = CountingScan()
    stop = threading.Event()

    def tick():
        engine.loop(stop, scans, interval_s=600, heartbeat_s=10, startup_delay_s=0)
        return

    import time as _time

    class _FakeEvent(threading.Event):
        """Событие, которое «просыпается» мгновенно, но двигает часы — цикл без реального сна."""

        def __init__(self, clock):
            super().__init__()
            self._clock = clock
            self.waits = 0

        def wait(self, timeout=None):
            self._clock.advance(timeout or 1)
            self.waits += 1
            if self.waits >= 4:
                self.set()
            return self.is_set()

    fake_stop = _FakeEvent(clock)
    engine.loop(fake_stop, scans, interval_s=600, heartbeat_s=10, startup_delay_s=0)
    assert scans.calls == 1, f"прогонов {scans.calls}, ожидался 1 за 4 пульса с интервалом 600 с"


class FakeService:
    """Сервис с циклом внутри (как scan_service/auto_scanner): start/stop/is_running."""

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self._running = False

    @property
    def is_running(self):
        return self._running

    def start(self):
        self.started += 1
        self._running = True

    def stop(self):
        self.stopped += 1
        self._running = False


def test_supervise_starts_service_only_for_the_owner():
    """Супервизор запускает сервис у владельца и не запускает у остальных реплик."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    owner_svc, follower_svc = FakeService(), FakeService()
    owner = _engine("scan", redis, clock, state)
    follower = _engine("scan", redis, clock, state)
    stop = threading.Event()

    class _Once(threading.Event):
        def __init__(self):
            super().__init__()
            self.n = 0

        def wait(self, timeout=None):
            self.n += 1
            if self.n >= 2:
                self.set()
            return self.is_set()

    # Владелец уже держит аренду: супервизор follower'а обязан НЕ запускать сервис.
    assert owner.try_become_owner() is True
    supervise(follower_svc, follower, _Once(), check_interval_s=1)
    assert follower_svc.started == 0, "не-владелец запустил сервис — это и есть N сканеров"

    # А владелец — запускает и аккуратно останавливает при завершении.
    supervise(owner_svc, owner, _Once(), check_interval_s=1)
    assert owner_svc.started == 1 and owner_svc.stopped == 1


def test_supervise_stops_service_when_lease_is_lost():
    """Потеря аренды обязана останавливать сервис: иначе два сканера пойдут параллельно."""
    clock = Clock()
    redis = FakeRedis(clock)
    state = SharedState()
    svc = FakeService()
    engine = _engine("scan", redis, clock, state)
    stop = threading.Event()

    class _Steps(threading.Event):
        def __init__(self):
            super().__init__()
            self.n = 0

        def wait(self, timeout=None):
            self.n += 1
            if self.n == 1:
                # Пока супервизор спит, аренду забирает «другая реплика».
                token = engine._token
                engine._token = None
                engine._lease.release("scan", token)
                other = _engine("scan", redis, clock, state)
                other.try_become_owner()
            if self.n >= 3:
                self.set()
            return self.is_set()

    supervise(svc, engine, _Steps(), check_interval_s=1)
    assert svc.started == 1 and svc.stopped >= 1, "сервис не остановлен при потере аренды"


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
    print(f"--- scan engine: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
