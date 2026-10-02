"""`result_cache` целиком: SWR, конверт и single-flight вместе (итерация 26).

Проверяется стык трёх слоёв, а не каждый по отдельности: свежий ответ не считает,
устаревший отдаётся сразу и обновляется в фоне, отсутствующий считается один раз
даже при параллельных запросах, а запись прежней схемы распознаётся и перезаписывается.

`gex.adapters.cache.redis_client` подменяется заглушкой **до** импорта `result_cache`: настоящий модуль
тянет pandas, а эта логика обязана проверяться в stdlib-наборе.

    python tests/test_result_cache.py
    pytest tests/test_result_cache.py -q
"""
from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ====================================================================== #
#  Заглушка gex.adapters.cache.redis_client (без pandas)
# ====================================================================== #
class FakeRedisClient:
    """Клиент с интерфейсом RedisClient, достаточным для result_cache."""

    def __init__(self):
        self.store: dict = {}
        self.connected = True

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None, *, px=None, nx=False):
        """Как RedisClient: поддержка ``NX`` (выбор лидера) и ``PX`` (аренда single-flight).

        Без NX single-flight не мог бы выбрать лидера между вызовами: проверка+set не атомарны.
        """
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def delete(self, key):
        return self.store.pop(key, None) is not None


_stub = types.ModuleType("gex.adapters.cache.redis_client")
_stub.CURRENT = FakeRedisClient()
_stub.get_redis = lambda: _stub.CURRENT
_stub.cache_key = lambda prefix, *parts: f"gex:{prefix}:" + ":".join(str(p) for p in parts)
_stub.serialize_value = lambda value: json.dumps(value, sort_keys=True).encode()
_stub.deserialize_value = lambda raw: json.loads(raw.decode() if isinstance(raw, bytes) else raw)

#: Заглушка живёт в ``sys.modules`` **весь модуль**, но снимается после прогона.
#: ``result_cache`` импортирует из ``redis_client`` имена (``cache_key``, ``get_redis``,
#: ``serialize_value``), а ``gex.adapters.cache.serialize`` резолвит пару сериализаторов
#: **лениво** — поэтому убрать заглушку сразу после импорта нельзя: тогда serialize
#: подхватит настоящий модуль (с pickle+zlib) и «коверт в JSON» перестанет совпадать
#: с тем, что проверяет набор (ровно это и случилось: UnicodeDecodeError на 0x80 —
#: это читался zlib-заголовок).
#:
#: Правильная изоляция — снять заглушку по окончании набора (fixture autouse ниже):
#: наборы, идущие после, получают настоящий модуль и не падают с
#: ``ImportError: cannot import name 'RedisClient'``.
_real_redis_client = sys.modules.get("gex.adapters.cache.redis_client")
sys.modules["gex.adapters.cache.redis_client"] = _stub

import gex.adapters.cache.result_cache as _rc_mod  # noqa: E402
from gex.adapters.cache.result_cache import ResultCache  # noqa: E402
import gex.adapters.cache.serialize as _serialize_mod  # noqa: E402

#: Пара сериализаторов **пинится на заглушку** до конца набора.
#: ``serialize.resolve_serializers()`` резолвит пару лениво и кэширует её; conftest
#: восстанавливает настоящий ``redis_client`` при переходе к следующему файлу, поэтому
#: без пина набор начинал писать настоящим сериализатором (pickle+zlib) и падал на чтении
#: JSON (``UnicodeDecodeError`` на 0x80 — это zlib-заголовок). Пин делает набор
#: независимым от порядка запуска: каким бы ни был ``sys.modules`` позже, этот модуль
#: видит ровно ту пару, на которую рассчитаны его проверки.
_serialize_mod._pair = (_stub.serialize_value, _stub.deserialize_value)  # noqa: SLF001


def _restore_real_module():
    """Вернуть настоящий ``redis_client`` в ``sys.modules`` (для следующих наборов)."""
    if _real_redis_client is not None:
        sys.modules["gex.adapters.cache.redis_client"] = _real_redis_client
    else:
        sys.modules.pop("gex.adapters.cache.redis_client", None)


try:
    import pytest as _pytest

    @_pytest.fixture(autouse=True)
    def _pin_stub_each_test():
        """Ставить заглушку **перед каждым тестом** набора.

        Двух вещей, которых здесь не хватало, достаточно, чтобы набор падал в общем прогоне:

        1. ``ResultCache`` зовёт ``get_redis()`` **в рантайме**, а не при импорте. Модуль
           ``main`` (его импортирует соседний набор вроде ``test_admin_backup_restore``)
           выставляет в настоящем ``redis_client`` глобальный ``_default_client`` — то есть
           реальное соединение. Подмена ``sys.modules`` эту ссылку не перебивает: у
           ``ResultCache`` свой объект из настоящего модуля, и он видит настоящий Redis.
           Поэтому ``get_redis`` заглушки возвращает **её** клиент безусловно.
        2. Пара сериализаторов резолвится лениво и кэшируется, поэтому её тоже нужно
           переставлять перед каждым тестом (иначе читаются zlib-байты вместо JSON).
        """
        sys.modules["gex.adapters.cache.redis_client"] = _stub
        _stub.get_redis = lambda: _stub.CURRENT
        # ``result_cache`` делает ``from ...redis_client import get_redis`` — то есть
        # связывает имя **в своём** пространстве имён при импорте. Патч модуля-источника
        # до этой привязки не достаёт, поэтому переставляем и её.
        _rc_mod.get_redis = lambda: _stub.CURRENT
        _serialize_mod._pair = (_stub.serialize_value, _stub.deserialize_value)  # noqa: SLF001
        yield

    @_pytest.fixture(autouse=True, scope="module")
    def _isolate_redis_stub():
        """Снять заглушку по завершении набора, чтобы не ломать последующие."""
        sys.modules["gex.adapters.cache.redis_client"] = _stub
        _stub.get_redis = lambda: _stub.CURRENT
        _rc_mod.get_redis = lambda: _stub.CURRENT
        _serialize_mod._pair = (_stub.serialize_value, _stub.deserialize_value)  # noqa: SLF001
        yield
        _restore_real_module()
        _serialize_mod.reset_cache()
except ImportError:  # pytest недоступен — набор запускается напрямую
    import atexit

    atexit.register(_restore_real_module)


def _client() -> FakeRedisClient:
    return _stub.CURRENT


def _fresh_cache() -> ResultCache:
    _stub.CURRENT = FakeRedisClient()
    return ResultCache()


def _counting(value="computed", delay=0.0):
    counter = {"calls": 0}

    def compute():
        counter["calls"] += 1
        if delay:
            time.sleep(delay)
        return value

    return compute, counter


# ====================================================================== #
#  1. Свежее значение
# ====================================================================== #
def test_fresh_entry_does_not_recompute():
    cache = _fresh_cache()
    compute, counter = _counting("v1")
    assert cache.get("gex:res:k", ttl=300, compute=compute) == "v1"
    assert cache.get("gex:res:k", ttl=300, compute=compute) == "v1"
    assert counter["calls"] == 1, f"свежее значение пересчитано ({counter['calls']})"


def test_missing_entry_is_computed_and_written_as_envelope():
    cache = _fresh_cache()
    compute, counter = _counting("v1")
    assert cache.get("gex:res:k", ttl=300, compute=compute) == "v1"
    raw = _client().store["gex:res:k"]
    payload = json.loads(raw.decode())
    assert payload["__env__"] == 2, f"записан не конверт: {payload}"
    assert payload["v"] == "v1"
    assert payload["ttl"] == 300
    assert payload["etag"], "etag обязателен для ревалидации"
    assert counter["calls"] == 1


# ====================================================================== #
#  2. Старые записи и перезапись
# ====================================================================== #
def test_legacy_entry_triggers_recompute_without_crashing():
    """В работающем Redis лежат записи прежней формы — читатель обязан их переварить."""
    cache = _fresh_cache()
    _client().store["gex:res:k"] = json.dumps({"v": "old", "ts": time.time()}).encode()
    compute, counter = _counting("new")

    assert cache.get("gex:res:k", ttl=300, compute=compute) == "new"
    assert counter["calls"] == 1, "запись старой схемы принята за свежую"
    # и перезаписана в новом виде
    assert json.loads(_client().store["gex:res:k"].decode())["__env__"] == 2


def test_corrupt_entry_is_recomputed():
    cache = _fresh_cache()
    _client().store["gex:res:k"] = b"not-json"
    compute, counter = _counting("v")
    assert cache.get("gex:res:k", ttl=300, compute=compute) == "v"
    assert counter["calls"] == 1


# ====================================================================== #
#  3. Окна SWR
# ====================================================================== #
def _write_entry(cache_key: str, value, *, age: float, ttl: int) -> None:
    payload = {"__env__": 2, "v": value, "ts": time.time() - age, "ttl": ttl, "etag": "x"}
    _client().store[cache_key] = json.dumps(payload).encode()


def test_stale_within_max_age_returns_immediately_and_refreshes_in_background():
    """Главный сценарий UX: пользователь никогда не ждёт пересчёта."""
    cache = _fresh_cache()
    _write_entry("gex:res:k", "stale", age=400, ttl=300)
    compute, counter = _counting("fresh")

    started = time.monotonic()
    value = cache.get("gex:res:k", ttl=300, max_age=600, compute=compute)
    elapsed = time.monotonic() - started

    assert value == "stale", "отдано не устаревшее значение"
    assert elapsed < 0.05, f"ответ ждал пересчёта ({elapsed:.3f} с)"

    deadline = time.monotonic() + 3.0
    while counter["calls"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert counter["calls"] == 1, "фоновый пересчёт не запустился"
    assert json.loads(_client().store["gex:res:k"].decode())["v"] == "fresh"


def test_entry_older_than_max_age_is_recomputed():
    cache = _fresh_cache()
    _write_entry("gex:res:k", "ancient", age=900, ttl=300)
    compute, counter = _counting("new")
    assert cache.get("gex:res:k", ttl=300, max_age=600, compute=compute) == "new"
    assert counter["calls"] == 1


def test_max_age_defaults_to_double_ttl():
    cache = _fresh_cache()
    _write_entry("gex:res:k", "stale", age=250, ttl=300)  # свежее
    compute, counter = _counting("new")
    assert cache.get("gex:res:k", ttl=300, compute=compute) == "stale"
    assert counter["calls"] == 0


# ====================================================================== #
#  4. Отказы и деградация
# ====================================================================== #
def test_redis_unavailable_computes_without_crash():
    cache = ResultCache()
    client = FakeRedisClient()
    client.connected = False
    _stub.CURRENT = client
    compute, counter = _counting("v")
    assert cache.get("gex:res:k", ttl=300, compute=compute) == "v"
    assert counter["calls"] == 1
    assert client.store == {}, "при выключенном Redis запись не делается"


def test_redis_unavailable_still_caches_in_process():
    """Инцидент 2026-09-21: без Redis кэш отключался целиком.

    Каждый запрос уходил в живой апстрим, и дашборд положил процесс (6533 треда).
    При недоступном Redis значение обязано переиспользоваться из памяти процесса.
    """
    cache = ResultCache()
    client = FakeRedisClient()
    client.connected = False
    _stub.CURRENT = client
    compute, counter = _counting("v")
    assert cache.get("gex:res:noredis", ttl=300, compute=compute) == "v"
    assert cache.get("gex:res:noredis", ttl=300, compute=compute) == "v"
    assert counter["calls"] == 1, "без Redis значение обязано браться из памяти процесса"
    assert client.store == {}, "в Redis по-прежнему ничего не пишем"


def test_in_process_cache_is_bounded():
    """Внутрипроцессный слой обязан иметь потолок, иначе словарь растёт вечно."""
    cache = ResultCache()
    client = FakeRedisClient()
    client.connected = False
    _stub.CURRENT = client
    limit = ResultCache.MEM_MAX_ENTRIES
    for i in range(limit + 20):
        cache.get(f"gex:res:bound{i}", ttl=300, compute=lambda i=i: f"v{i}")
    assert len(cache._mem) <= limit, (
        f"в памяти {len(cache._mem)} записей при потолке {limit}"
    )


def test_invalidate_forces_recompute():
    cache = _fresh_cache()
    compute, counter = _counting("v")
    cache.get("gex:res:k", ttl=300, compute=compute)
    cache.invalidate("gex:res:k")
    cache.get("gex:res:k", ttl=300, compute=compute)
    assert counter["calls"] == 2, "invalidate не сбросил значение"


# ====================================================================== #
#  5. Схождение параллельных запросов (стык single-flight и конверта)
# ====================================================================== #
def test_parallel_requests_compute_once():
    """6 одновременных запросов одного ключа → 1 расчёт, все получают его результат."""
    cache = _fresh_cache()
    compute, counter = _counting("shared", delay=0.2)

    results: list = []
    errors: list = []
    barrier = threading.Barrier(6)

    def worker():
        try:
            barrier.wait(timeout=5)
            results.append(cache.get("gex:res:k", ttl=300, compute=compute))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"ошибки: {errors}"
    assert counter["calls"] == 1, f"параллельные запросы вызвали compute {counter['calls']} раз"
    assert results == ["shared"] * 6, results


def test_foreign_lock_without_stale_still_returns_value():
    """Чужой лидер и нет stale → считаем сами, а не возвращаем ``None`` (Fix 3).

    ``RedisSingleFlight.run`` в этой ситуации отдаёт ``None``; до Fix 3 ``_recompute``
    транслировал его наружу, ``None`` доходил до FastAPI и давал ``ResponseValidationError``
    → HTTP 500. Теперь расчёт выполняется локально — последним средством, как в
    ``_on_wait_timeout`` (порядок «stale → расчёт» неизменен) — и ровно один раз.
    """
    cache = _fresh_cache()
    cache._envelope_cache()  # создать ленивый RedisSingleFlight
    cache._redis_sf._wait_ms = 150  # окно ожидания не под этим тестом — не тянем 5 с в набор
    compute, counter = _counting("computed")

    # Чужая аренда: лидер из другого процесса уже держит ключ и значения не пишет.
    _client().store["gex:lock:gex:res:k"] = b"foreign"

    assert cache.get("gex:res:k", ttl=300, compute=compute) == "computed"
    assert counter["calls"] == 1, f"compute вызван {counter['calls']} раз вместо 1"

    # Аренда освободилась — второй запрос обслуживается из кэша, без повторного расчёта.
    del _client().store["gex:lock:gex:res:k"]
    assert cache.get("gex:res:k", ttl=300, compute=compute) == "computed"
    assert counter["calls"] == 1, "значение пересчитано вместо чтения из кэша"


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
    print(f"--- result cache: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
