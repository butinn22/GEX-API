"""Конверт кэша: версия схемы, etag, терпимость к старым записям (итерация 26).

Почему это отдельный набор проверок, а не «часть result_cache»
--------------------------------------------------------------
Конверт — это форма данных, лежащих в Redis. Ошибка в нём не проявляется как исключение:
она проявляется как молчаливая подмена (читаем чужое значение), как вечный промах (форма
не совпала) или как падение запроса на битых байтах. Поэтому проверяются именно границы:
старая схема, будущая схема, мусор, несериализуемое значение, недоступный Redis.

Запуск (без pandas/redis — сериализатор подставляется):
    python tests/test_cache_envelope.py
    pytest tests/test_cache_envelope.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.cache.envelope import (  # noqa: E402
    ENVELOPE_VERSION,
    LEGACY_VERSION,
    Envelope,
    RedisEnvelopeCache,
    etag_for,
)


def _json_serialize(value):
    return json.dumps(value, sort_keys=True).encode()


def _json_deserialize(raw):
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


class FakeRedis:
    """Минимум Redis'а: get/set/delete, сбой по флагу (проверка деградации)."""

    def __init__(self):
        self.store: dict = {}
        self.fail = False
        self.expirations: dict = {}

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value
        self.expirations[key] = ex
        return True

    def delete(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.pop(key, None) is not None


def _cache(redis=None, clock=None):
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return RedisEnvelopeCache(
        redis if redis is not None else FakeRedis(),
        serialize=_json_serialize,
        deserialize=_json_deserialize,
        **kwargs,
    )


# ====================================================================== #
# 1. Круговой рейс и метаданные
# ====================================================================== #
def test_roundtrip_keeps_metadata():
    redis = FakeRedis()
    cache = _cache(redis, clock=lambda: 1000.0)
    env = cache.write("gex:res:ta:SPY", {"a": 1}, ttl=300, source="yfinance")
    assert env is not None

    got = cache.read("gex:res:ta:SPY")
    assert got is not None
    assert got.value == {"a": 1}
    assert got.ttl == 300
    assert got.stored_at == 1000.0
    assert got.version == ENVELOPE_VERSION
    assert got.source == "yfinance"
    assert got.etag and got.etag == env.etag


def test_env_key_lives_twice_the_ttl():
    """Ключ обязан переживать свой срок свежести — иначе stale отдавать нечего."""
    redis = FakeRedis()
    _cache(redis).write("k", "v", ttl=300)
    assert redis.expirations["k"] == 600


def test_etag_is_stable_for_equal_values_and_changes_with_value():
    """etag — для ревалидации (304): «то же» должно давать тот же хэш."""
    value = {"date": "2026-09-16", "rows": [1, 2, 3]}
    same = {"date": "2026-09-16", "rows": [1, 2, 3]}
    other = {"date": "2026-09-17", "rows": [1, 2, 3]}

    def tag(v):
        return _cache(FakeRedis()).write("k", v, ttl=60).etag

    assert tag(value) == tag(same), "равные значения дали разные etag"
    assert tag(value) != tag(other), "изменённое значение сохранило etag"


def test_etag_for_hashes_bytes_not_repr():
    """Разные байты — разный etag, даже если текстовое представление совпадает."""
    assert etag_for(b"1") != etag_for(b"1.0")
    assert etag_for(b"same") == etag_for(b"same")
    assert len(etag_for(b"x")) == 16


# ====================================================================== #
# 2. Границы: старая схема, будущая схема, мусор
# ====================================================================== #
def test_legacy_payload_is_read_as_stale():
    """Записи прежней формы обязаны читаться: иначе выкат = холодный кэш на ровном месте."""
    redis = FakeRedis()
    redis.store["k"] = _json_serialize({"v": {"old": True}, "ts": 500.0})
    env = _cache(redis).read("k")
    assert env is not None
    assert env.value == {"old": True}
    assert env.version == LEGACY_VERSION
    assert env.is_legacy()
    assert not env.is_fresh(now=500.0), "legacy-запись не может считаться свежей"
    assert env.ttl == 0


def test_future_envelope_version_is_ignored():
    """Конверт новее кода: о форме значения не гадаем — лучше пересчитать."""
    payload = {"__env__": ENVELOPE_VERSION + 1, "v": "?", "ts": 1.0, "ttl": 60}
    assert Envelope.from_payload(payload) is None


def test_corrupt_and_unknown_payloads_return_none():
    for payload in [
        None, "not-a-dict", 123, [], {},
        {"v": 1},                        # нет ts
        {"ts": 1.0},                     # нет значения
        {"v": 1, "ts": "вчера"},         # ts не число
        {"__env__": ENVELOPE_VERSION, "ts": 1.0},   # нет v
    ]:
        assert Envelope.from_payload(payload) is None, f"принят мусор: {payload!r}"


def test_broken_bytes_do_not_raise():
    redis = FakeRedis()
    redis.store["k"] = b"\xff\xfe broken"
    assert _cache(redis).read("k") is None


def test_redis_failure_degrades_quietly():
    """Кэш не имеет права ронять запрос: недоступный Redis — это промах, не ошибка."""
    redis = FakeRedis()
    redis.fail = True
    cache = _cache(redis)
    assert cache.read("k") is None
    assert cache.write("k", "v", ttl=60) is None
    assert cache.invalidate("k") is False


def test_unserializable_value_is_not_cached():
    cache = RedisEnvelopeCache(
        FakeRedis(),
        serialize=lambda v: (_ for _ in ()).throw(TypeError("нельзя сериализовать")),
        deserialize=_json_deserialize,
    )
    assert cache.write("k", object(), ttl=60) is None


# ====================================================================== #
# 3. Свежесть и границы окон
# ====================================================================== #
def test_freshness_windows():
    env = Envelope(value="v", stored_at=1000.0, ttl=300)
    assert env.age(now=1100.0) == 100.0
    assert env.is_fresh(now=1300.0), "на границе ttl значение ещё свежее"
    assert not env.is_fresh(now=1300.1)
    # ttl < age <= max_age → отдаём устаревшее и обновляем в фоне
    assert env.is_usable_stale(600, now=1400.0)
    assert not env.is_usable_stale(600, now=1600.1)
    assert not env.is_usable_stale(600, now=1200.0), "свежее не является stale"


def test_negative_age_is_tolerated():
    """Сдвиг часов назад не должен ломать логику свежести."""
    env = Envelope(value="v", stored_at=2000.0, ttl=300)
    assert env.age(now=1900.0) == -100.0
    assert env.is_fresh(now=1900.0)


def test_to_stale_forces_recompute():
    env = Envelope(value="v", stored_at=1000.0, ttl=300)
    assert env.to_stale().ttl == 0
    assert not env.to_stale().is_fresh(now=1000.0)
    assert env.ttl == 300, "исходный конверт не мутируется"


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
    print(f"--- cache envelope: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
