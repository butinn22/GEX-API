"""Single-flight: N одновременных запросов одного ключа → 1 вычисление (итерация 26).

Главная проверка — **slow-leader**: пока лидер считает, все остальные ждут и получают
его результат, а не запускают дорогую работу повторно. Именно это было сломано в прежнем
``result_cache``: при ``acquire(timeout)`` и таймауте код уходил в ``compute()`` напрямую,
то есть под нагрузкой дублировал работу ровно тогда, когда она не успевает.

Запуск (без pandas/redis — Redis подменяется фейком):
    python tests/test_singleflight.py
    pytest tests/test_singleflight.py -q
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.cache.singleflight import LOCK_PREFIX, LocalSingleFlight, RedisSingleFlight  # noqa: E402


class FakeRedis:
    """Redis с семантикой ``SET NX PX``, истечением аренды и потокобезопасным хранилищем.

    Нужны именно NX и PX: на них держится выбор лидера между процессами и *замещение*
    лидера, который умер. Фейк без истечения аренды проверял бы не то поведение, что
    работает в проде: там лидер, умерший без снятия аренды, перестаёт быть лидером
    ровно потому, что аренда истекает.
    """

    def __init__(self):
        self.store: dict = {}
        self.expiry: dict = {}
        self._lock = threading.Lock()
        self.fail = False

    def _purge(self, key: str) -> None:
        exp = self.expiry.get(key)
        if exp is not None and time.monotonic() >= exp:
            self.store.pop(key, None)
            self.expiry.pop(key, None)

    def set(self, key, value, ex=None, nx=False, px=None):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            self._purge(key)
            if nx and key in self.store:
                return None
            self.store[key] = value
            if px:
                self.expiry[key] = time.monotonic() + px / 1000.0
            elif ex:
                self.expiry[key] = time.monotonic() + ex
            else:
                self.expiry.pop(key, None)
            return True

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            self._purge(key)
            return self.store.get(key)

    def delete(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            self.expiry.pop(key, None)
            return self.store.pop(key, None) is not None

    def plant(self, key, value, *, px: int) -> None:
        """Положить «чужую» аренду с заданным сроком жизни (имитация другого процесса)."""
        self.store[key] = value
        self.expiry[key] = time.monotonic() + px / 1000.0


class BytesFakeRedis(FakeRedis):
    """Redis, хранящий ``str`` как utf-8 ``bytes`` — как настоящий ``RedisClient``.

    Настоящий клиент сериализует строку-токен в ``b"token"``, и ``get`` возвращает ``bytes``.
    Фейк, хранивший ``str`` как ``str``, **скрывал** дефект: ``_release`` сравнивал
    ``self._redis.get(...) == token`` (``bytes == str``) — в проде это всегда ложно, аренда
    не снималась и висела весь срок. Проверка на байтовом фейке ловит эту регрессию.
    """

    def set(self, key, value, ex=None, nx=False, px=None):
        stored = value.encode("utf-8") if isinstance(value, str) else value
        return super().set(key, stored, ex=ex, nx=nx, px=px)


def _slowed(counter: dict, delay: float = 0.15, value: str = "result"):
    """Loader со счётчиком вызовов и задержкой (имитация дорогого расчёта)."""

    def loader():
        counter["calls"] = counter.get("calls", 0) + 1
        time.sleep(delay)
        return value

    return loader


# ====================================================================== #
# 1. Локальный single-flight
# ====================================================================== #
def test_slow_leader_computes_once():
    """8 параллельных запросов одного ключа → ровно 1 вызов loader'а, всем — его результат."""
    sf = LocalSingleFlight(wait_timeout=5.0)
    counter: dict = {}
    loader = _slowed(counter, delay=0.2, value={"data": 42})

    results: list = []
    errors: list = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait(timeout=5)
            results.append(sf.run("gex:res:macd:SPY", loader))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"ошибки у ожидающих: {errors}"
    assert counter["calls"] == 1, f"loader вызван {counter['calls']} раз вместо 1"
    assert len(results) == 8, f"результат получили {len(results)} из 8"
    assert all(r == {"data": 42} for r in results), "ожидающие получили не результат лидера"
    assert sf.follower_misses == 0


def test_follower_gets_leader_exception():
    """Ошибка лидера транслируется ожидающим: иначе они получат мусор вместо исключения."""
    sf = LocalSingleFlight(wait_timeout=5.0)

    def loader():
        time.sleep(0.1)
        raise RuntimeError("источник недоступен")

    errors: list = []
    barrier = threading.Barrier(4)

    def worker():
        try:
            barrier.wait(timeout=5)
            sf.run("k", loader)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(errors) == 4, f"исключение получили {len(errors)} из 4"
    assert all(isinstance(e, RuntimeError) for e in errors)


def test_different_keys_are_not_coalesced():
    """Схождение — только по одному ключу: иначе запрос отдал бы чужие данные."""
    sf = LocalSingleFlight(wait_timeout=5.0)
    counter: dict = {}

    a = sf.run("key-a", _slowed(counter, delay=0.05, value="A"))
    b = sf.run("key-b", _slowed(counter, delay=0.05, value="B"))

    assert (a, b) == ("A", "B")
    assert counter["calls"] == 2


def test_slot_is_released_after_leader_finishes():
    """После лидера ячейка освобождается: следующий вызов снова считает."""
    sf = LocalSingleFlight(wait_timeout=5.0)
    counter: dict = {}
    loader = _slowed(counter, delay=0.01, value="v")

    sf.run("k", loader)
    sf.run("k", loader)
    assert counter["calls"] == 2
    assert sf._slots == {}, "ячейка не освобождена — утечка памяти на каждый ключ"


def test_wait_timeout_does_not_rerun_loader():
    """Истёк таймаут ожидания → отдаём on_miss, но loader повторно НЕ запускаем."""
    sf = LocalSingleFlight(wait_timeout=0.01)
    counter: dict = {}
    release = threading.Event()

    def slow_loader():
        counter["calls"] = counter.get("calls", 0) + 1
        release.wait(timeout=5)
        return "leader"

    follower_result: list = []
    leader = threading.Thread(target=lambda: sf.run("k", slow_loader), daemon=True)
    leader.start()
    time.sleep(0.05)  # лидер уже внутри loader'а

    follower = threading.Thread(
        target=lambda: follower_result.append(sf.run("k", slow_loader, on_miss=lambda: "stale")),
        daemon=True,
    )
    follower.start()
    follower.join(timeout=5)

    assert follower_result == ["stale"], follower_result
    assert counter["calls"] == 1, f"ожидающий запустил loader ({counter['calls']} вызовов)"
    assert sf.follower_misses == 1
    release.set()
    leader.join(timeout=5)


# ====================================================================== #
# 2. Кросс-процессный single-flight (аренда в Redis)
# ====================================================================== #
def test_redis_leader_computes_once_and_follower_reads_cache():
    """Два «процесса» на одном Redis: считает только лидер, follower читает кэш."""
    redis = FakeRedis()
    counter: dict = {}
    cache: dict = {}
    loader = _slowed(counter, delay=0.2, value="from-leader")

    def read():
        return cache.get("k")

    def load_writing():
        value = loader()
        cache["k"] = value
        return value

    sf_leader = RedisSingleFlight(redis, lease_ms=5000, wait_ms=3000)
    sf_follower = RedisSingleFlight(redis, lease_ms=5000, wait_ms=3000)

    out: dict = {}
    leader = threading.Thread(target=lambda: out.__setitem__("leader", sf_leader.run("k", load_writing, read=read)))
    leader.start()
    time.sleep(0.05)  # лидер уже держит аренду и считает
    follower_value = sf_follower.run("k", load_writing, read=read)
    leader.join(timeout=5)

    assert counter["calls"] == 1, f"loader вызван {counter['calls']} раз вместо 1"
    assert out["leader"] == "from-leader"
    assert follower_value == "from-leader", "follower не получил результат лидера"
    assert sf_follower.loader_calls == 0, "follower запустил loader"
    assert redis.store.get(f"{LOCK_PREFIX}k") is None, "аренда не снята"


def test_redis_lease_released_when_client_returns_bytes():
    """Аренда обязана сниматься, даже если ``get`` возвращает ``bytes`` (как реальный клиент).

    Регрессия (Fix 2): ``_release`` сравнивал ``self._redis.get(...) == token`` (``bytes`` ==
    ``str``) — всегда ложно, аренда не удалялась и висела весь срок (``REDIS_LEASE_MS`` = 60 с),
    блокируя замещение лидера и роняя followers в 500. Идентификатор ключа — как в боевом
    репро (``gex:res:TESTREPRO:sf``).
    """
    redis = BytesFakeRedis()
    sf = RedisSingleFlight(redis, lease_ms=5000, wait_ms=300, poll_s=0.01)
    lock_key = f"{LOCK_PREFIX}gex:res:TESTREPRO:sf"

    out = sf.run("gex:res:TESTREPRO:sf", lambda: "leader-value", read=lambda: None)

    assert out == "leader-value"
    assert redis.store.get(lock_key) is None, "аренда не снята (bytes != str в _release)"
    assert redis.expiry.get(lock_key) is None, "TTL аренды остался — она будет замещать лидера зря"


def test_redis_double_check_after_acquiring_lease():
    """Пока брали аренду, значение мог записать другой: тогда loader не нужен вовсе."""
    redis = FakeRedis()
    sf = RedisSingleFlight(redis, lease_ms=5000, wait_ms=100)
    counter: dict = {}
    called = sf.run("k", _slowed(counter, value="fresh-from-other"), read=lambda: "already-fresh")
    assert called == "already-fresh"
    assert counter.get("calls", 0) == 0, "double-check не сработал: значение посчитано зря"


def test_redis_takeover_after_leader_vanishes():
    """Лидер умер (аренда истекла, значения нет) → ожидающий забирает аренду.

    Иначе запрос не завершится никогда: ждать больше некого. Проверяется именно
    замещение по истечении аренды — так в проде и происходит, когда воркер упал
    посреди расчёта.
    """
    redis = FakeRedis()
    sf = RedisSingleFlight(redis, lease_ms=50, wait_ms=250, poll_s=0.01)
    counter: dict = {}
    loader = _slowed(counter, delay=0.0, value="rescued")

    out = sf.run("k", loader, read=lambda: None)  # аренда свободна → лидер
    assert out == "rescued" and counter["calls"] == 1

    # «Чужой лидер»: аренда занята и истечёт через 50 мс, значения он не запишет.
    redis.plant(f"{LOCK_PREFIX}k2", "dead-owner", px=50)
    out2 = sf.run("k2", loader, read=lambda: None)

    assert out2 == "rescued", "зависший лидер не был замещён"
    assert sf.takeovers == 1, f"замещений: {sf.takeovers}"
    assert counter["calls"] == 2, f"loader вызван {counter['calls']} раз вместо 2"


def test_redis_returns_stale_when_leader_still_holds_lease():
    """Лидер считает дольше бюджета → отдаём устаревшее, а не None и не дубль расчёта."""
    redis = FakeRedis()
    sf = RedisSingleFlight(redis, lease_ms=5000, wait_ms=100, poll_s=0.01)
    redis.plant(f"{LOCK_PREFIX}k", "alive-owner", px=60_000)
    counter: dict = {}
    stale = {"old": True}

    out = sf.run("k", _slowed(counter, value="should-not-be-used"), read=lambda: stale)
    assert out is stale, "не отдано устаревшее значение"
    assert counter.get("calls", 0) == 0, "follower запустил loader при живом лидере"


def test_follower_reads_are_bounded_by_backoff():
    """Фиксированный интервал опроса давал ~40 чтений кэша за 400 мс ожидания
    (и до 250 за 5 с — помноженные на число ожидающих). Экспоненциальный рост
    интервала ограничивает шторм чтений Redis, сохраняя быстрый отклик на
    коротких ожиданиях."""
    reads = {"n": 0}

    class _LeaderBusy:
        def set(self, *a, **k):
            return False  # аренда всегда у лидера

    sf = RedisSingleFlight(
        _LeaderBusy(), lease_ms=60_000, wait_ms=400, poll_s=0.01, max_poll_s=0.08
    )

    def read():
        reads["n"] += 1
        return None

    result = sf.run("backoff:key", loader=lambda: "computed", read=read)
    assert result is None  # лидер так и не отдал значение
    # С фиксированным poll_s=0.01 чтений было бы ~40; бэкофф укладывается в ~10.
    assert reads["n"] <= 12


def test_redis_unavailable_fails_open():
    """Без Redis схождение невозможно: считаем сами, но значение обязано вернуться."""
    redis = FakeRedis()
    redis.fail = True
    sf = RedisSingleFlight(redis, lease_ms=1000, wait_ms=100)
    counter: dict = {}
    out = sf.run("k", _slowed(counter, delay=0.0, value="solo"), read=lambda: None)
    assert out == "solo"
    assert counter["calls"] == 1



# ====================================================================== #
#  3. Контракт с RedisClient: adapter не может опираться на то, чего нет
# ====================================================================== #
def test_redis_client_declares_the_capabilities_we_use():
    """``SET NX PX`` обязан поддерживаться клиентом, иначе аренда молча не берётся.

    Этот дефект был реальным: ``RedisSingleFlight`` вызывал ``set(..., nx=True, px=…)``,
    а ``RedisClient.set`` принимал только ``ex`` — то есть в проде аренда *никогда* не
    бралась, схождение между воркерами отсутствовало, а в логах это выглядело как
    «Redis недоступен». Проверка идёт по AST: набор тестов обязан работать без pandas.
    """
    import ast

    src = (ROOT / "gex" / "adapters" / "cache" / "redis_client.py").read_text(encoding="utf-8")
    params: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "RedisClient":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "set":
                    params = {a.arg for a in item.args.args + item.args.kwonlyargs}
    assert params, "не найден RedisClient.set"
    missing = {"nx", "px"} - params
    assert not missing, f"RedisClient.set не принимает {sorted(missing)} — аренда не будет взята"


def test_adapter_uses_only_declared_client_capabilities():
    """Все именованные аргументы, которые адаптер передаёт в Redis, должны быть у клиента.

    Ловит класс дефекта «адаптер опирается на несуществующий параметр»: он не падает
    на импорте и не виден в тестах с фейком, который принимает ``**kwargs``.
    """
    import ast

    client_src = (ROOT / "gex" / "adapters" / "cache" / "redis_client.py").read_text(encoding="utf-8")
    client_params: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(client_src)):
        if isinstance(node, ast.ClassDef) and node.name == "RedisClient":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    client_params[item.name] = {a.arg for a in item.args.args + item.args.kwonlyargs}

    adapter_src = (ROOT / "gex" / "adapters" / "cache" / "singleflight.py").read_text(encoding="utf-8")
    used: set[str] = set()
    for node in ast.walk(ast.parse(adapter_src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"set", "get", "delete"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr in {"_redis", "redis"}
        ):
            used |= {kw.arg for kw in node.keywords if kw.arg}

    assert used, "не найдено ни одного вызова Redis в адаптере — проверка потеряла смысл"
    client_methods = set(client_params)
    assert {"set", "get", "delete"} <= client_methods
    undeclared = used - client_params["set"]
    assert not undeclared, f"адаптер передаёт параметры, которых нет у RedisClient.set: {sorted(undeclared)}"


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
    print(f"--- singleflight: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
