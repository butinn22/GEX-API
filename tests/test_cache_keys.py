"""Провайдер-инклюзивные ключи кэша (итерация 25).

Проверяется то, ради чего итерация делалась — **коллизии между источниками**:

1. Разные провайдеры дают разные ключи для одного символа (критерий приёмки ROADMAP).
2. Один и тот же источник под разными именами/регистром/числовым форматом даёт ОДИН ключ:
   ``moex`` == ``moex_iss``, ``btc`` == ``BTC``, ``5`` == ``5.0`` == ``"5"``. Это не косметика:
   до итерации ``"rts"`` и ``"RTS"`` писали в разные ключи, то есть половина обращений
   промахивалась мимо кэша.
3. Неоднозначный ключ **невозможно построить**: нет провайдера → ошибка; ``cache_key``
   с провайдер-зависимым видом → ошибка.
4. Ключи, которые итерация не трогала (``res:*``, состояние), сохранили прежнюю форму:
   у них 16 живых мест вызова, и молчаливая смена формы дала бы «вечный промах» кэша,
   который ничем не сигнализируется.

Запуск (без зависимостей — только stdlib):
    python tests/test_cache_keys.py
    pytest tests/test_cache_keys.py -q
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.cache import keys as K  # noqa: E402
from gex.adapters.cache.keys import CacheKeyError  # noqa: E402


class Skipped(Exception):
    """Проверка требует pandas/redis (финальный этап с .venv-test) — не «зелёная», а пропущенная."""


def _require_cache_key():
    """Вернуть ``gex.redis_client.cache_key`` или явно пропустить проверку.

    Импорт ленивый: модуль ``redis_client`` тянет pandas, а провайдер-инварианты
    обязаны проверяться и без него (stdlib-only прогон).
    """
    try:
        from gex.adapters.cache.redis_client import cache_key
    except ImportError as exc:  # pandas (или redis) не установлены
        raise Skipped("нужен pandas (окружение .venv-test) — проверка не выполнена") from exc
    return cache_key


def _raises(fn, *args, exc: type[BaseException] = CacheKeyError, **kwargs) -> bool:
    """Упало ли ожидаемым исключением.

    Различаем два класса ошибок намеренно: отсутствие провайдера-аргумента — это
    ``TypeError`` (ошибка вызывающего, ловится ещё до выполнения), а недопустимое
    значение провайдера/части ключа — ``CacheKeyError`` (ошибка данных).
    """
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


# ====================================================================== #
# 1. Коллизии: провайдер различает ключи
# ====================================================================== #
def test_chain_keys_differ_by_provider():
    """Критерий приёмки: один символ, разные площадки → разные ключи.

    Именно эта коллизия была в коде: ``gex:chain:BTC:5`` писали и Bybit (свои страйки и
    экспирации), и yfinance (``BTC-USD``) — это разные инструменты.
    """
    keys = {
        K.chain_key("BTC", 5, provider=p)
        for p in (K.PROVIDER_BYBIT, K.PROVIDER_YFINANCE, K.PROVIDER_MOEX, K.PROVIDER_WEBULL)
    }
    assert len(keys) == 4, f"провайдеры не различаются: {keys}"
    assert K.chain_key("BTC", 5, provider="bybit") == "gex:chain:bybit:BTC:5"


def test_ohlcv_and_hv_and_spot_keys_differ_by_provider():
    """Та же проверка для остальных семейств (ohlcv на крипте пишут Bybit и yfinance-fallback)."""
    assert K.ohlcv_key("BTC", "1h", provider="bybit") != K.ohlcv_key("BTC", "1h", provider="yfinance")
    assert K.hv_key("RTS", provider="moex_iss") != K.hv_key("RTS", provider="yfinance")
    assert K.spot_key("SPY", provider="yfinance") != K.spot_key("SPY", provider="webull")


def test_commodity_keys_carry_provider():
    assert K.commodity_key("spot", "GOLD", provider="yfinance") == "gex:commodity:spot:yfinance:GOLD"
    assert K.commodity_key("chain", "GOLD", 5, provider="yfinance") != K.commodity_key(
        "chain", "SILVER", 5, provider="yfinance"
    )
    # Провайдер у товарного блока обязателен, хотя сегодня он всегда yfinance:
    # иначе новый источник молча унаследует чужой ключ.
    assert _raises(K.commodity_key, "spot", "GOLD", exc=TypeError)


# ====================================================================== #
# 2. Один источник → один ключ (нормализация)
# ====================================================================== #
def test_provider_aliases_collapse():
    """``moex`` и ``moex_iss`` — одно и то же; иначе один источник получил бы два ключа."""
    assert K.normalize_provider("moex") == K.normalize_provider("moex_iss") == "moex_iss"
    assert K.normalize_provider("YF") == K.normalize_provider("yfinance") == "yfinance"
    assert K.chain_key("BTC", 5, provider="moex") == K.chain_key("BTC", 5, provider="MOEX_ISS")


def test_symbol_case_and_whitespace_collapse():
    """Регистр и пробелы не создают второй ключ: ``" rts"``, ``"rts"`` и ``"RTS"`` — одно."""
    assert K.chain_key(" rts ", 5, provider="moex_iss") == K.chain_key("RTS", 5, provider="moex_iss")
    assert K.ohlcv_key("spy", "1H", provider="yfinance") == K.ohlcv_key("SPY", "1h", provider="yfinance")


def test_numeric_parts_canonicalize():
    """``5``, ``5.0`` и ``"5"`` — один ключ: иначе тип аргумента менял бы ключ."""
    a = K.chain_key("BTC", 5, provider="bybit")
    assert a == K.chain_key("BTC", 5.0, provider="bybit")
    assert a == K.chain_key("BTC", "5", provider="bybit")


# ====================================================================== #
# 3. Неоднозначный ключ невозможно построить
# ====================================================================== #
def test_provider_is_mandatory():
    assert _raises(K.chain_key, "BTC", 5, exc=TypeError), "chain_key без провайдера должен падать"
    assert _raises(K.ohlcv_key, "BTC", "1h", exc=TypeError), "ohlcv_key без провайдера должен падать"
    assert _raises(K.hv_key, "SPY", exc=TypeError), "hv_key без провайдера должен падать"


def test_unknown_provider_rejected():
    """Неизвестный источник не попадает в ключ молча — иначе он станет вторым bybit."""
    assert _raises(K.normalize_provider, "polygon")
    assert _raises(K.chain_key, "BTC", 5, provider="polygon_invented")


def test_key_parts_cannot_break_the_schema():
    """``:`` и пробелы в части склеили бы сегменты — ключ стал бы неоднозначным."""
    assert _raises(K.ohlcv_key, "BTC:USD", "1h", provider="bybit")
    assert _raises(K.ohlcv_key, "BTC USD", "1h", provider="bybit")
    assert _raises(K.commodity_key, "", "GOLD", provider="yfinance")


def test_empty_and_none_parts_rejected():
    assert _raises(K.chain_key, "", 5, provider="bybit")
    assert _raises(K.chain_key, None, 5, provider="bybit")


def test_non_integer_count_rejected():
    assert _raises(K.chain_key, "BTC", 5.5, provider="bybit")
    assert _raises(K.chain_key, "BTC", True, provider="bybit")


# ====================================================================== #
# 4. Рантайм-страж на старом построителе
# ====================================================================== #
def test_cache_key_rejects_provider_scoped_kinds():
    """``gex.redis_client.cache_key`` больше не строит такие ключи (даже динамически)."""
    cache_key = _require_cache_key()

    for kind in sorted(K.PROVIDER_SCOPED_KINDS) + ["commodity:spot", "commodity:ohlcv"]:
        assert _raises(cache_key, kind, "BTC", 5), f"{kind}: неоднозначный ключ построился"


def test_cache_key_keeps_providerless_shape():
    """Совместимость: форма ``res:*`` не изменилась — это 16 живых мест вызова."""
    cache_key = _require_cache_key()

    assert cache_key("res", "ta", "SPY", 1000) == "gex:res:TA:SPY:1000"
    assert cache_key("res", "livegex", "spy", 5, 8) == "gex:res:LIVEGEX:SPY:5:8"
    assert cache_key("sigstate3", "u-42") == "gex:sigstate3:u-42"
    assert cache_key("vol") == "gex:vol"


def test_legacy_shape_is_a_superset_check():
    """Сверка формы ``res`` с прежней реализацией (копия алгоритма до итер. 25).

    Проверяются все таблицы, которые реально встречаются в роутерах: если форма поедет,
    кэш ответов станет холодным на каждом рестарте и это не будет видно ни в логах,
    ни в метриках — поэтому сравнение с эталоном, а не «на глаз».
    """
    cache_key = _require_cache_key()

    def legacy(prefix: str, *parts) -> str:
        cleaned = []
        for p in parts:
            if p is None:
                continue
            s = str(p)
            if s.isalpha():
                s = s.upper()
            cleaned.append(s)
        return f"gex:{prefix}:" + ":".join(cleaned) if cleaned else f"gex:{prefix}"

    tables = [
        ("res", ("ta", "SPY", 10000)),
        ("res", ("livegex", "SPY", 5, 8)),
        ("res", ("sec-revenue", "AAPL", "annual")),
        ("res", ("breadth", "sector")),
        ("res", ("hybrid", "BTC", "4h", 500)),
        ("sigstate3", ("u-42",)),
        ("auto_scan", ("US_OPTIONS", "signals")),
        ("vol", ()),
    ]
    for prefix, parts in tables:
        assert cache_key(prefix, *parts) == legacy(prefix, *parts), f"форма изменилась: {prefix}"


# ====================================================================== #
# 5. Страж R6 (статический) — и его отсутствие дрейфа
# ====================================================================== #
def test_r6_gate_is_clean():
    """Ни один продакшн-модуль не строит провайдер-зависимый ключ через ``cache_key``."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "quality" / "ast_guard.py"), "--json"],
        capture_output=True, text=True, cwd=str(ROOT), check=False,
    )
    report = json.loads(out.stdout)
    assert report["provider_key_violations"] == [], report["provider_key_violations"]


def test_guard_kind_set_matches_module():
    """Множество видов в гейте не должно разъехаться с модулем ключей.

    В гейте оно продублировано строкой, чтобы гейт оставался stdlib-only; этот тест —
    страховка от дрейфа, иначе R6 тихо перестанет ловить новый вид ключа.
    """
    sys.path.insert(0, str(ROOT / "scripts" / "quality"))
    import ast_guard  # noqa: E402

    assert set(ast_guard.PROVIDER_SCOPED_KINDS) == set(K.PROVIDER_SCOPED_KINDS), (
        f"дрейф: гейт {set(ast_guard.PROVIDER_SCOPED_KINDS)} ≠ модуль {set(K.PROVIDER_SCOPED_KINDS)}"
    )


# ====================================================================== #
# 6. Расписка миграции: вызывающие перешли на builders
# ====================================================================== #
MIGRATED = {
    "gex/adapters/fetchers/yf_fetcher.py": "chain_key",
    "gex/adapters/fetchers/bybit_fetcher.py": "chain_key",
    "gex/adapters/fetchers/moex_fetcher.py": "chain_key",
    # Webull перешёл на chain_key_v2: область выборки (горизонт + страйк-окно)
    # входит в ключ, иначе цепочка «1 экспирация» и «N экспираций» делили бы
    # один ключ (аудит 2026-09-17).
    "gex/adapters/fetchers/webull_fetcher.py": "chain_key_v2",
    "gex/adapters/fetchers/ta_fetcher.py": "ohlcv_key",
    "gex/application/ohlcv_service.py": "ohlcv_key",
    "gex/adapters/fetchers/commodity_fetcher.py": "commodity_key",
    "gex/application/commodity_dynamics.py": "commodity_key",
    "gex/routers/gexcone_router.py": "hv_key",
}


def test_migrated_callers_use_builders():
    """Миграция не может «откатиться» молча: файл обязан импортировать и звать builder."""
    for rel, builder in MIGRATED.items():
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {
            (a.asname or a.name)
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and (n.module or "").endswith(("cache.keys", "cache_keys"))
            for a in n.names
        }
        called = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert builder in imported, f"{rel}: не импортирует {builder}"
        assert builder in called, f"{rel}: импортирует {builder}, но не вызывает"


def test_commodity_sites_pass_provider_explicitly():
    """У товарных ключей провайдер по умолчанию yfinance, но он обязан быть назван явно.

    Иначе при добавлении второго товарного источника ключ молча останется «yfinance».
    """
    for rel in ("gex/adapters/fetchers/commodity_fetcher.py", "gex/application/commodity_dynamics.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "commodity_key":
                assert any(kw.arg == "provider" for kw in node.keywords), (
                    f"{rel}:{node.lineno}: commodity_key без provider="
                )



# ====================================================================== #
# 7. Поведение среза крипты (реальный код, а не макет)
# ====================================================================== #
def test_crypto_slice_is_provider_scoped():
    """Чтение и запись среза крипты идут по ключу источника, который отдал бары.

    Функция ``_crypto_ohlcv`` вынимается из модуля по AST и исполняется с подставленными
    зависимостями: так проверяется **настоящий** код (а не его копия в тесте), без сети.

    Что именно фиксируется: Bybit-бары не должны оказаться под yfinance-ключом и наоборот —
    иначе потребитель получает свечи неизвестной площадки (разные границы интервала и объём).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise Skipped("нужен pandas (окружение .venv-test)") from exc

    import ast as _ast
    import logging
    import typing

    from gex.adapters.cache.redis_client import serialize_value

    fn_ast = next(
        n for n in _ast.parse((ROOT / "gex" / "application" / "ohlcv_service.py").read_text(encoding="utf-8")).body
        if isinstance(n, _ast.FunctionDef) and n.name == "_crypto_ohlcv"
    )
    code = compile(_ast.Module(body=[fn_ast], type_ignores=[]), "gex/application/ohlcv_service.py", "exec")

    class FakeRedis:
        def __init__(self):
            self.store: dict = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, ex=None):
            self.store[key] = serialize_value(value)
            return True

    bybit_df = pd.DataFrame({"Close": [1, 2, 3]})
    yf_df = pd.DataFrame({"Close": [9, 9, 9]})

    def build(bybit_fn):
        ns = {
            "__name__": "gex.application.ohlcv_service", "__package__": "gex.application",
            "pd": pd, "Optional": typing.Optional,
            "PROVIDER_BYBIT": K.PROVIDER_BYBIT, "PROVIDER_YFINANCE": K.PROVIDER_YFINANCE,
            "ohlcv_key": K.ohlcv_key, "logger": logging.getLogger("test"),
            "_YF_CRYPTO_TICKER": {"BTC": "BTC-USD"},
            "fetch_bybit_klines": bybit_fn,
            "_stock_fetcher_with_cache": lambda redis, redis_ok: type(
                # Подпись обязана совпадать с настоящей: после итер. 34 helper переехал
                # в модульную область и принимает redis (внутри _crypto_ohlcv замыкания
                # нет). Пока подстановка принимала ноль аргументов, тест падал сам —
                # и это ровно то, что он должен делать при смене контракта.
                "St", (), {"fetch_timeframe": lambda s, t, tf: yf_df}
            )(),
        }
        exec(code, ns)  # noqa: S102 — изолированный namespace, код из репозитория
        return ns["_crypto_ohlcv"]

    bybit_ok = lambda t, tf, limit=1000: bybit_df  # noqa: E731
    bybit_down = lambda t, tf, limit=1000: (_ for _ in ()).throw(RuntimeError("bybit down"))  # noqa: E731

    # промах: бары Bybit кладутся под ключ bybit, а не под «общий»
    store = FakeRedis()
    out = build(bybit_ok)("BTC", "1h", store, True)
    assert list(out["Close"]) == [1, 2, 3]
    assert list(store.store) == [K.ohlcv_key("BTC", "1h", provider=K.PROVIDER_BYBIT)]

    # в кэше только yfinance-срез → возвращается он, а не свежий Bybit
    store = FakeRedis()
    store.store[K.ohlcv_key("BTC", "1h", provider=K.PROVIDER_YFINANCE)] = serialize_value(
        pd.DataFrame({"Close": [7, 7]})
    )
    assert list(build(bybit_ok)("BTC", "1h", store, True)["Close"]) == [7, 7]

    # в кэше только bybit-срез
    store = FakeRedis()
    store.store[K.ohlcv_key("BTC", "1h", provider=K.PROVIDER_BYBIT)] = serialize_value(
        pd.DataFrame({"Close": [5]})
    )
    assert list(build(bybit_ok)("BTC", "1h", store, True)["Close"]) == [5]

    # Bybit упал → fallback пишется под ключ yfinance
    store = FakeRedis()
    out = build(bybit_down)("BTC", "1h", store, True)
    assert list(out["Close"]) == [9, 9, 9]
    assert list(store.store) == [K.ohlcv_key("BTC", "1h", provider=K.PROVIDER_YFINANCE)]

    # битый срез одного источника → берётся второй, а не падаем
    store = FakeRedis()
    store.store[K.ohlcv_key("BTC", "1h", provider=K.PROVIDER_BYBIT)] = b"broken"
    store.store[K.ohlcv_key("BTC", "1h", provider=K.PROVIDER_YFINANCE)] = serialize_value(
        pd.DataFrame({"Close": [4, 4]})
    )
    assert list(build(bybit_ok)("BTC", "1h", store, True)["Close"]) == [4, 4]

    # Redis недоступен → прямой путь к источнику, без обращения к кэшу
    assert list(build(bybit_ok)("BTC", "1h", None, False)["Close"]) == [1, 2, 3]


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Skipped as exc:
            skipped += 1
            print(f"SKIP {fn.__name__}: {exc}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- cache keys: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
