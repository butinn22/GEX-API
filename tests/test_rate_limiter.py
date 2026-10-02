"""Лимиты провайдеров: один авторитет на все воркеры (итерация 27).

Проверяется то, чего не было: лимит **общий** для воркеров. До итерации ведро жило в
памяти процесса, поэтому фактическая частота равнялась заданной × число воркеров —
для yfinance «4/с» при 4 воркерах превращались в 16/с, то есть провайдер, банящий
за >5/с, получал ровно то, за что банит.

Наборы без внешних зависимостей не могут исполнять Lua, поэтому здесь используется
**эмуляция контракта скрипта** на Python: тот же алгоритм токен-бакета, тот же формат
ответа `{allowed, tokens_left(строка), wait_ms}`. Реальный Lua проверяется отдельно —
`scripts/quality/_audit_rl_check.py` (нужен `fakeredis[lua]`), он же измеряет частоту
при двух воркерах.

    python tests/test_rate_limiter.py
    pytest tests/test_rate_limiter.py -q
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.ratelimit.redis_lua import (  # noqa: E402
    RedisIpLimiter,
    RedisRateLimiter,
    RedisTokenBucket,
    TokenBucketScript,
    bucket_ttl_ms,
    canonical_provider,
    ip_rate_limit_key,
    make_limits,
    rate_limit_key,
)
from gex.adapters.ratelimit.rate_limiter import RateLimiter, RateLimiterAuthority, TokenBucket  # noqa: E402


class FakeRedisWithLua:
    """Redis с **эмуляцией** контракта ``LUA_TOKEN_BUCKET`` (тот же алгоритм).

    Хранилище общее для всех инстансов лимитера — именно это и проверяется: два
    «воркера» на одном Redis обязаны делить одно ведро.
    """

    def __init__(self, clock=None):
        self.connected = True
        self.hashes: dict[str, dict] = {}
        self.expiry: dict[str, int] = {}
        self.fail = False
        self._lock = threading.Lock()
        self._clock = clock or (lambda: time.time() * 1000.0)
        self.script_loads = 0

    # -- интерфейс, который зовёт адаптер ---------------------------------- #
    def script_load(self, _src: str) -> str:
        self.script_loads += 1
        return "emulated-sha"

    def evalsha(self, _sha: str, _nkeys: int, key: str, *args):
        if self.fail:
            raise ConnectionError("redis down")
        rate, burst, want, ttl_ms = float(args[0]), float(args[1]), float(args[2]), int(args[3])
        with self._lock:
            now = self._clock()
            bucket = self.hashes.get(key)
            if bucket is None:
                tokens, last = burst, now
            else:
                tokens, last = bucket["tokens"], bucket["ts"]
            elapsed = max(now - last, 0.0)
            tokens = min(burst, tokens + elapsed * rate / 1000.0)
            allowed, wait_ms = 0, 0
            if tokens >= want:
                tokens -= want
                allowed = 1
            else:
                wait_ms = max(int(-(-(want - tokens) * 1000.0 // rate)), 1)
            self.hashes[key] = {"tokens": tokens, "ts": now}
            self.expiry[key] = ttl_ms
            # строка, а не float: RESP усекает Lua-числа до целых
            return [allowed, f"{tokens!r}", wait_ms]

    def eval(self, _src: str, _nkeys: int, key: str):
        if self.fail:
            raise ConnectionError("redis down")
        bucket = self.hashes.get(key)
        if bucket is None:
            return [b"", b""]
        return [f"{bucket['tokens']!r}".encode(), f"{bucket['ts']!r}".encode()]

    def delete(self, key: str) -> bool:
        if self.fail:
            raise ConnectionError("redis down")
        self.expiry.pop(key, None)
        return self.hashes.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return key in self.hashes


class FlakyScriptRedis(FakeRedisWithLua):
    """Redis, который один раз отвечает NOSCRIPT (перезапуск с пустым кэшем скриптов)."""

    def __init__(self):
        super().__init__()
        self.noscript_left = 1

    def evalsha(self, sha, nkeys, *args):
        if self.noscript_left > 0:
            self.noscript_left -= 1
            raise RuntimeError("NOSCRIPT No matching script. Please use EVAL.")
        return super().evalsha(sha, nkeys, *args)


LIMITS = {"yfinance": {"rate": 5.0, "burst": 2}}


# ====================================================================== #
# 1. Ключи и словарь провайдеров
# ====================================================================== #
def test_rate_limit_keys_are_canonical():
    """``moex`` и ``moex_iss`` — одно ведро: иначе лимит одного провайдера делится на два."""
    assert rate_limit_key("moex") == rate_limit_key("moex_iss") == "gex:rl:moex_iss"
    assert rate_limit_key("YFINANCE") == "gex:rl:yfinance"
    assert rate_limit_key("cboe") == "gex:rl:cboe"


def test_non_provider_targets_are_allowed():
    """``cboe`` (CDN) и ``telegram`` (сокет уведомлений) — не источники кэша, но лимитировать их надо."""
    assert canonical_provider("telegram") == "telegram"
    assert canonical_provider("cboe") == "cboe"
    assert canonical_provider("moex") == "moex_iss"


def test_ip_keys_are_scoped_and_ipv6_safe():
    """Область разделяет лимиты, IPv6-двоеточия не ломают схему ключа."""
    assert ip_rate_limit_key("auth", "10.0.0.7") == "gex:rl:ip:auth:10.0.0.7"
    # В схеме ровно 4 разделителя (gex:rl:ip:scope:ip); IPv6 не должен добавить пятый,
    # иначе адрес разъедется по сегментам и ведро будет «своё» на каждый октет.
    v6 = ip_rate_limit_key("auth", "2001:db8::1")
    assert v6 == "gex:rl:ip:auth:2001_db8__1", v6
    assert v6.count(":") == 4, f"лишние разделители в ключе: {v6}"
    assert ip_rate_limit_key("auth", "1.1.1.1") != ip_rate_limit_key("telegram", "1.1.1.1")


def test_duplicate_provider_names_in_config_are_rejected():
    """Один провайдер под двумя именами = два ведра = двойной лимит: это ошибка конфига."""
    try:
        make_limits({"moex": {"rate": 1, "burst": 1}, "moex_iss": {"rate": 2, "burst": 2}})
    except ValueError as exc:
        assert "двойной лимит" in str(exc)
    else:
        raise AssertionError("дубли имён провайдера приняты молча")


def test_invalid_limits_rejected_at_build_time():
    """Некорректный лимит должен падать при сборке, а не делением на ноль в проде."""
    def rejected(limits) -> bool:
        try:
            make_limits(limits)
        except ValueError:
            return True
        return False

    assert rejected({"x": {"rate": 0, "burst": 5}}), "rate=0 принят"
    assert rejected({"x": {"rate": 5, "burst": 0}}), "burst=0 принят"
    assert rejected({"x": {"rate": float("nan"), "burst": 5}}), "rate=nan принят"
    assert not rejected({"x": {"rate": 5, "burst": 5}}), "корректный лимит отклонён"


def test_bucket_ttl_is_bounded():
    assert bucket_ttl_ms(5.0, 10) == 60_000            # нижняя граница 60 с
    assert bucket_ttl_ms(0.001, 100) == 3_600_000      # верхняя граница 1 ч
    assert bucket_ttl_ms(1.0, 10) == 60_000


# ====================================================================== #
# 2. Поведение ведра
# ====================================================================== #
def test_bucket_enforces_burst_then_refills():
    redis = FakeRedisWithLua()
    bucket = RedisTokenBucket(redis, "yfinance", 5.0, 2)
    assert bucket.acquire(tokens=1, blocking=False) is True
    assert bucket.acquire(tokens=1, blocking=False) is True
    assert bucket.acquire(tokens=1, blocking=False) is False, "burst не соблюдён"


def test_bucket_refills_over_time():
    """Пополнение считается по времени Redis, а не по локальным часам воркера."""
    now = [1000.0]
    redis = FakeRedisWithLua(clock=lambda: now[0])
    bucket = RedisTokenBucket(redis, "yfinance", 5.0, 5)
    for _ in range(5):
        bucket.acquire(tokens=1, blocking=False)
    assert bucket.acquire(tokens=1, blocking=False) is False
    now[0] += 1000.0  # прошла секунда при rate=5/с
    assert bucket.acquire(tokens=1, blocking=False) is True, "токены не пополнились"


def test_fractional_tokens_are_not_truncated():
    """Остаток возвращается строкой: RESP усекает Lua-числа до целых."""
    redis = FakeRedisWithLua()
    allowed, left, _ = RedisTokenBucket(redis, "yfinance", 3.0, 3)._try(0.5)
    assert allowed and left == 2.5, f"остаток усечён: {left!r}"


def test_unreachable_single_request_does_not_loop_forever():
    """Запрос больше burst не может быть удовлетворён никогда — не ждём его."""
    redis = FakeRedisWithLua()
    bucket = RedisTokenBucket(redis, "yfinance", 1.0, 2, sleeper=lambda s: None)
    start = time.monotonic()
    assert bucket.acquire(tokens=5.0, blocking=True) is False
    assert time.monotonic() - start < 0.5, "ждали недостижимый запрос"


def test_blocking_wait_is_bounded():
    """Лимитер не должен превращаться в бесконечную очередь."""
    slept: list[float] = []
    redis = FakeRedisWithLua()
    bucket = RedisTokenBucket(
        redis, "yfinance", 0.001, 1, sleeper=slept.append, max_wait_s=0.5
    )
    bucket.acquire(tokens=1, blocking=False)
    assert bucket.acquire(tokens=1, blocking=True) is False
    assert slept and all(s <= 0.5 for s in slept), f"ждали больше лимита: {slept}"


# ====================================================================== #
# 3. Единый авторитет: два воркера — один лимит
# ====================================================================== #
def test_two_workers_share_one_bucket():
    """Ключевая проверка итерации: два инстанса лимитера на одном Redis = ОДИН лимит."""
    now = [1000.0]
    redis = FakeRedisWithLua(clock=lambda: now[0])
    worker_a = RedisRateLimiter(redis, LIMITS)
    worker_b = RedisRateLimiter(redis, LIMITS)

    allowed = 0
    for i in range(10):  # поровну между «воркерами»
        limiter = worker_a if i % 2 == 0 else worker_b
        if limiter.acquire("yfinance", blocking=False):
            allowed += 1
    assert allowed == 2, f"при burst=2 пропущено {allowed} — ведро не разделяется"

    # прежнее поведение: два независимых локальных лимитера дали бы 4
    local_a, local_b = RateLimiter(LIMITS), RateLimiter(LIMITS)
    local_allowed = sum(
        1 for i in range(10)
        if (local_a if i % 2 == 0 else local_b).acquire("yfinance", blocking=False)
    )
    assert local_allowed == 4, "дефект «× число воркеров» не воспроизвёлся"


def test_lookup_canonicalises_provider_name():
    """``wait("moex")`` обязан найти ведро ``moex_iss``: иначе запрос уйдёт без лимита."""
    redis = FakeRedisWithLua()
    limiter = RedisRateLimiter(redis, {"moex_iss": {"rate": 1.0, "burst": 1}})
    assert limiter.get_bucket("moex") is not None, "алиас не нашёл ведро — тихая дыра в лимите"
    assert limiter.acquire("moex", blocking=False) is True
    assert limiter.acquire("moex_iss", blocking=False) is False, "алиас и канон — разные вёдра"


def test_unknown_provider_is_not_limited():
    """Неизвестный провайдер не ограничивается (поведение прежнего лимитера)."""
    redis = FakeRedisWithLua()
    assert RedisRateLimiter(redis, LIMITS).acquire("nope", blocking=False) is True


def test_reset_all_clears_buckets():
    redis = FakeRedisWithLua()
    limiter = RedisRateLimiter(redis, LIMITS)
    limiter.acquire("yfinance", blocking=False)
    limiter.acquire("yfinance", blocking=False)
    assert limiter.acquire("yfinance", blocking=False) is False
    limiter.reset_all()
    assert limiter.acquire("yfinance", blocking=False) is True


# ====================================================================== #
# 4. Деградация и перезагрузка скрипта
# ====================================================================== #
def test_redis_failure_falls_back_and_is_counted():
    """Недоступный Redis снимает распределённый лимит — и это обязано быть видно."""
    redis = FakeRedisWithLua()
    local = RateLimiter(LIMITS)
    limiter = RedisRateLimiter(redis, LIMITS, degraded_limiter=local)
    redis.fail = True

    assert limiter.acquire("yfinance", blocking=False) is True
    assert limiter.degraded_calls == 1, "деградация не учтена — она останется незаметной"
    assert limiter.acquire("yfinance", blocking=False) is True
    assert limiter.degraded_calls == 2


def test_noscript_is_recovered():
    """Redis перезапустили → кэш скриптов пуст: лимитер обязан перезагрузить скрипт."""
    redis = FlakyScriptRedis()
    limiter = RedisRateLimiter(redis, LIMITS)
    assert limiter.acquire("yfinance", blocking=False) is True, "NOSCRIPT не обработан"
    assert redis.script_loads >= 2, "скрипт не был перезагружен"


def test_script_loaded_once():
    """EVALSHA, а не EVAL: скрипт загружается один раз, а не на каждый запрос."""
    redis = FakeRedisWithLua()
    limiter = RedisRateLimiter(redis, LIMITS)
    for _ in range(5):
        limiter.acquire("yfinance", blocking=False)
    assert redis.script_loads == 1, f"скрипт загружался {redis.script_loads} раз"


# ====================================================================== #
# 5. IP-лимит (защита от перебора)
# ====================================================================== #
def test_ip_limit_is_shared_between_workers():
    """Порог перебора не должен умножаться на число воркеров."""
    redis = FakeRedisWithLua()
    limiters = [RedisIpLimiter(redis, "auth", rate=1.0, burst=3) for _ in range(3)]
    allowed = sum(
        1 for i in range(12) if limiters[i % 3].allow("10.0.0.7")
    )
    assert allowed == 3, f"при burst=3 пропущено {allowed} — порог умножен на воркеры"


def test_ip_limit_is_per_ip_and_per_scope():
    redis = FakeRedisWithLua()
    limiter = RedisIpLimiter(redis, "auth", rate=1.0, burst=1)
    assert limiter.allow("10.0.0.1") is True
    assert limiter.allow("10.0.0.1") is False, "второй запрос с того же IP прошёл"
    assert limiter.allow("10.0.0.2") is True, "чужой IP заблокирован"
    other_scope = RedisIpLimiter(redis, "telegram", rate=1.0, burst=1)
    assert other_scope.allow("10.0.0.1") is True, "области делят одно ведро"


def test_ip_limit_degrades_visibly():
    redis = FakeRedisWithLua()
    limiter = RedisIpLimiter(redis, "auth", rate=1.0, burst=1)
    redis.fail = True
    assert limiter.allow("10.0.0.7") is True
    assert limiter.degraded_calls == 1


# ====================================================================== #
# 6. Фасад: выбор авторитета и повышение
# ====================================================================== #
def test_authority_is_local_without_redis():
    facade = RateLimiterAuthority(redis_getter=lambda: None, clock=lambda: 0.0)
    assert facade.acquire("yfinance", blocking=False) is True
    assert facade.authority == "local"


def test_authority_upgrades_when_redis_appears():
    """Воркер мог стартовать раньше Redis: авторитет обязан повыситься, а не застыть."""
    state = {"redis": None}
    facade = RateLimiterAuthority(redis_getter=lambda: state["redis"], clock=lambda: 0.0)
    facade.acquire("yfinance", blocking=False)
    assert facade.authority == "local"

    state["redis"] = FakeRedisWithLua()
    facade._next_recheck = 0.0  # снимаем окно перепроверки
    facade.acquire("yfinance", blocking=False)
    assert facade.authority == "redis", "авторитет не повысился после появления Redis"


def test_authority_degrades_and_recovers():
    """Redis пропал — лимит per-process (и это учтено), вернулся — снова общий."""
    redis = FakeRedisWithLua()
    facade = RateLimiterAuthority(redis_getter=lambda: redis, clock=lambda: 0.0)
    facade.acquire("yfinance", blocking=False)
    assert facade.authority == "redis"

    redis.fail = True
    assert facade.acquire("yfinance", blocking=False) is True
    assert facade._remote.degraded_calls == 1


def test_facade_exposes_providers_for_diagnostics():
    facade = RateLimiterAuthority({"yfinance": {"rate": 1.0, "burst": 1}})
    assert facade.providers == ["yfinance"]
    assert facade.reset_all() is None


def test_local_limiter_still_works_for_callers():
    """Локальный лимитер остаётся рабочим: он фолбэк и часть публичного API."""
    limiter = RateLimiter(LIMITS)
    assert limiter.acquire("yfinance", blocking=False) is True
    assert limiter.acquire("yfinance", blocking=False) is True
    assert limiter.acquire("yfinance", blocking=False) is False
    assert isinstance(limiter.get_bucket("yfinance"), TokenBucket)



# ====================================================================== #
# 7. Контракт с RedisClient: лимитер опирается на Lua-операции
# ====================================================================== #
def test_redis_client_declares_lua_operations():
    """``SCRIPT LOAD``/``EVALSHA``/``EVAL`` обязаны быть у клиента.

    Дефект был реальным: адаптер звал ``redis.script_load(...)``, а ``RedisClient``
    таких методов не имел — то есть в проде Lua-лимитер молча уходил в деградацию
    «Redis недоступен» на **каждом** запросе, и лимит снова становился per-process.
    Проверка по AST: набор обязан работать без pandas.
    """
    import ast

    src = (ROOT / "gex" / "adapters" / "cache" / "redis_client.py").read_text(encoding="utf-8")
    methods: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "RedisClient":
            methods = {i.name for i in node.body if isinstance(i, ast.FunctionDef)}
    missing = {"script_load", "evalsha", "eval", "delete", "get"} - methods
    assert not missing, f"RedisClient не умеет {sorted(missing)} — лимитер уйдёт в деградацию"


def test_adapter_only_uses_declared_client_operations():
    """Адаптер не может звать методы, которых у клиента нет.

    Именно так и был пропущен дефект с ``script_load``: фейк в тестах умел всё,
    а продовый клиент — ничего.
    """
    import ast

    def methods_of(path: str, cls: str) -> set[str]:
        src = (ROOT / path).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                return {i.name for i in node.body if isinstance(i, ast.FunctionDef)}
        raise AssertionError(f"не найден класс {cls} в {path}")

    client_methods = methods_of("gex/adapters/cache/redis_client.py", "RedisClient")
    adapter_src = (ROOT / "gex" / "adapters" / "ratelimit" / "redis_lua.py").read_text(encoding="utf-8")
    called: set[str] = set()
    for node in ast.walk(ast.parse(adapter_src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_redis"
        ):
            called.add(node.func.attr)

    assert called, "не найдено обращений адаптера к Redis — проверка потеряла смысл"
    undeclared = called - client_methods
    assert not undeclared, f"адаптер зовёт методы, которых нет у RedisClient: {sorted(undeclared)}"


def test_none_reply_from_client_is_not_treated_as_allowed():
    """``None`` от клиента (его соглашение при ошибке) — это НЕ «разрешено»."""
    redis = FakeRedisWithLua()
    limiter = RedisRateLimiter(redis, LIMITS, degraded_limiter=RateLimiter(LIMITS))
    redis.evalsha = lambda *a, **k: None
    assert limiter.acquire("yfinance", blocking=False) is True
    assert limiter.degraded_calls == 1, "None принят за успешный ответ — лимит снят молча"


def test_garbage_reply_is_rejected():
    from gex.adapters.ratelimit.redis_lua import RateLimitBackendError

    redis = FakeRedisWithLua()
    redis.evalsha = lambda *a, **k: ["мусор"]
    try:
        TokenBucketScript(redis).run("k", 1.0, 1, 1.0, 1000)
    except RateLimitBackendError:
        return
    raise AssertionError("мусорный ответ принят за валидный")


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
    print(f"--- rate limiter: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
