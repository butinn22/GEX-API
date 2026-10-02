"""Контракт портов: форма интерфейсов и их независимость от инфраструктуры.

Смысл портов — домен/application описывают внешний мир, не зная реализации. Значит:
  1) модули портов обязаны импортироваться **без** pandas/numpy/redis/fastapi/sqlalchemy;
  2) Protocol'ы обязаны быть ``runtime_checkable`` (иначе нельзя проверить подмену в тестах);
  3) любой объект с нужными методами обязан структурно подходить под порт.

    python tests/test_ports_contract.py
    pytest tests/test_ports_contract.py -q
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PORTS = (
    "gex.ports.market_data",
    "gex.ports.option_chain",
    "gex.ports.cache",
    "gex.ports.rate_limit",
    "gex.ports.job_queue",
    "gex.ports.notifier",
)

EXPECTED_METHODS = {
    "gex.ports.market_data": {"MarketDataPort": ("fetch_timeframes", "fetch_single", "fetch_spot")},
    "gex.ports.option_chain": {"OptionChainPort": ("fetch",)},
    "gex.ports.cache": {"CachePort": ("get", "invalidate")},
    "gex.ports.rate_limit": {"RateLimitPort": ("acquire", "wait")},
    "gex.ports.notifier": {"NotifierPort": ("send",)},
}


def test_ports_import_without_infrastructure():
    """Импорт портов не должен тянуть pandas/numpy/redis/fastapi/sqlalchemy/yfinance."""
    banned = {"pandas", "numpy", "redis", "fastapi", "sqlalchemy", "yfinance", "httpx", "requests"}
    for name in PORTS:
        before = set(sys.modules)
        importlib.import_module(name)
        newly = {m.split(".")[0] for m in set(sys.modules) - before}
        leaked = newly & banned
        assert not leaked, f"{name} импортировал инфраструктуру: {sorted(leaked)}"


def test_expected_protocols_and_methods_exist():
    for module_name, contracts in EXPECTED_METHODS.items():
        module = importlib.import_module(module_name)
        for cls_name, methods in contracts.items():
            cls = getattr(module, cls_name, None)
            assert cls is not None, f"{module_name}: нет {cls_name}"
            assert getattr(cls, "_is_runtime_protocol", False), f"{cls_name} не runtime_checkable"
            for method in methods:
                assert hasattr(cls, method), f"{cls_name}: нет метода {method}"


def test_structural_satisfaction_for_each_port():
    """Объект с нужными методами обязан подходить под порт (структурная типизация)."""
    from gex.ports import cache as cache_port
    from gex.ports import market_data as md_port
    from gex.ports import notifier as notifier_port
    from gex.ports import option_chain as chain_port
    from gex.ports import rate_limit as rate_port

    class FakeMarketData:
        def fetch_timeframes(self, ticker, *, limit=None):
            return {}

        def fetch_single(self, ticker, timeframe, *, limit=400):
            return None

        def fetch_spot(self, ticker):
            return None

    class FakeChain:
        def fetch(self, symbol, *, max_expiries=5):
            return None

    class FakeCache:
        def get(self, key, *, fresh, stale_max, compute, max_wait_ms=None):
            return None

        def invalidate(self, key):
            return None

    class FakeLimiter:
        def acquire(self, provider, *, endpoint="*", cost=1):
            return rate_port.Decision(allowed=True)

        def wait(self, provider, *, endpoint="*", cost=1):
            return rate_port.Decision(allowed=True)

    class FakeNotifier:
        def send(self, text, *, chat_id=None, parse_mode=None):
            return True

    assert isinstance(FakeMarketData(), md_port.MarketDataPort)
    assert isinstance(FakeChain(), chain_port.OptionChainPort)
    assert isinstance(FakeCache(), cache_port.CachePort)
    assert isinstance(FakeLimiter(), rate_port.RateLimitPort)
    assert isinstance(FakeNotifier(), notifier_port.NotifierPort)


def test_dtos_have_expected_shape():
    from gex.ports.cache import CacheStatus, CachedPayload
    from gex.ports.job_queue import Job, JobResult, Priority
    from gex.ports.rate_limit import Decision

    payload = CachedPayload(value={"a": 1}, status=CacheStatus.STALE, ts=1.0, fresh=10, stale_max=60)
    assert payload.version == 1 and payload.computing is False

    job = Job(task_type="ohlcv", idempotency_key="imoex_breadth:2026-09-16:23:00")
    assert job.priority is Priority.BACKGROUND and job.payload == {}
    assert JobResult(ok=False, error="timeout").error == "timeout"

    decision = Decision(allowed=False, retry_after_ms=1500, rule_name="yfinance:global:rps")
    assert decision.retry_after_seconds == 1.5


def test_ports_do_not_import_adapters_or_application():
    """Обратное направление зависимостей: порт не импортирует адаптеры и сценарии.

    Проверка идёт по AST, а не по подстроке: в docstring'ах ``gex/ports/__init__.py`` слово
    ``gex.adapters`` встречается как пояснение архитектуры, и текстовый поиск давал ложное
    срабатывание (тот же урок, что и с ``scanner_crypto`` в ROADMAP v2.1 §0.1).
    """
    import ast

    this_dir = Path(__file__).resolve().parents[1] / "gex" / "ports"
    forbidden = ("gex.adapters", "gex.application", "gex.deps", "main")
    for path in sorted(this_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden), f"{path.name}: импорт {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), f"{path.name}: импорт {alias.name}"


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
    print(f"--- ports contract: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
