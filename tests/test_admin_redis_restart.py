"""Перезапуск Redis из админки: команда выполняется на сервере, ошибки не роняют страницу.

Сам Redis в тестах НЕ перезапускается: ``subprocess.run`` подменён, поэтому проверяется
логика обработчика — коды ответов, тексты и переподключение, — а не внешний сервис.

    python tests/test_admin_redis_restart.py
    pytest tests/test_admin_redis_restart.py -q
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.auth.admin import system as system_mod  # noqa: E402
from gex.auth.config import settings  # noqa: E402

ENDPOINT = "/redis/restart"


class _FakeRedis:
    """Заглушка клиента: фиксируем, что переподключение действительно вызвано."""

    def __init__(self, connected: bool = True):
        self._connected = connected
        self.reconnect_calls = 0

    def reconnect(self) -> bool:
        self.reconnect_calls += 1
        return self._connected


@pytest.fixture
def client():
    """Минимальное приложение: только подроутер system, без БД и фоновых сервисов."""
    app = FastAPI()
    app.include_router(system_mod.router)
    app.dependency_overrides[system_mod._require_admin] = lambda: object()
    return TestClient(app)


def _patch_get_redis(monkeypatch, fake: _FakeRedis) -> None:
    """Подменить ``get_redis`` там, откуда обработчик его импортирует.

    Обработчик импортирует имя внутри вызова, то есть берёт модуль из ``sys.modules``.
    Соседний набор (``tests/test_result_cache.py``) подменяет и потом снимает этот модуль
    в ``sys.modules``, поэтому патч по «своей» ссылке на объект модуля виден не всегда —
    ошибка проявляется только при совместном прогоне в определённом порядке (в одиночном
    набор «зелёный»). Ставим в ``sys.modules`` собственный модуль-шим: разрешение имени
    становится предсказуемым независимо от порядка, а ``monkeypatch`` вернёт всё назад.
    """
    shim = types.ModuleType("gex.adapters.cache.redis_client")
    shim.get_redis = lambda: fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gex.adapters.cache.redis_client", shim)


@pytest.fixture
def configured(monkeypatch):
    """Команда перезапуска задана, Redis-клиент подменён."""
    def _apply(command: str = "redis-restart-stub", connected: bool = True):
        monkeypatch.setattr(settings, "REDIS_RESTART_COMMAND", command, raising=False)
        monkeypatch.setattr(settings, "REDIS_RESTART_CWD", "", raising=False)
        monkeypatch.setattr(settings, "REDIS_RESTART_TIMEOUT_SECONDS", 5.0, raising=False)
        fake = _FakeRedis(connected)
        _patch_get_redis(monkeypatch, fake)
        return fake
    return _apply


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["stub"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ====================================================================== #
#  Защита доступа
# ====================================================================== #
def test_endpoint_is_admin_protected():
    """Та же защита, что у остальных админских действий (``_require_admin``)."""
    route = next(r for r in system_mod.router.routes if r.path == ENDPOINT)
    guards = [d.call for d in route.dependant.dependencies]
    assert system_mod._require_admin in guards, "ручка перезапуска Redis без админ-гарда"


def test_anonymous_request_is_rejected():
    """Без авторизации ручка не выполняется вовсе."""
    app = FastAPI()
    app.include_router(system_mod.router)
    r = TestClient(app).post(ENDPOINT)
    assert r.status_code in (401, 403)


# ====================================================================== #
#  Отказы: понятный ответ вместо падения
# ====================================================================== #
def test_unconfigured_command_returns_501(client, monkeypatch):
    monkeypatch.setattr(settings, "REDIS_RESTART_COMMAND", "", raising=False)
    r = client.post(ENDPOINT)
    assert r.status_code == 501
    assert "REDIS_RESTART_COMMAND" in r.json()["detail"]


def test_nonzero_exit_returns_502(client, configured, monkeypatch):
    configured()
    monkeypatch.setattr(system_mod.subprocess, "run", lambda *a, **k: _completed(1, stderr="boom"))
    r = client.post(ENDPOINT)
    assert r.status_code == 502
    assert "boom" in r.json()["detail"]


def test_timeout_returns_504(client, configured, monkeypatch):
    configured()

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="stub", timeout=5.0)

    monkeypatch.setattr(system_mod.subprocess, "run", _boom)
    r = client.post(ENDPOINT)
    assert r.status_code == 504


def test_missing_binary_returns_502(client, configured, monkeypatch):
    """Команды нет в PATH (например, docker не установлен) — не 500 и не падение."""
    configured()

    def _boom(*a, **k):
        raise FileNotFoundError("no docker")

    monkeypatch.setattr(system_mod.subprocess, "run", _boom)
    r = client.post(ENDPOINT)
    assert r.status_code == 502
    assert "не найдена" in r.json()["detail"]


# ====================================================================== #
#  Успех и деградация
# ====================================================================== #
def test_success_reports_ok_and_reconnects(client, configured, monkeypatch):
    fake = configured(connected=True)
    monkeypatch.setattr(system_mod.subprocess, "run", lambda *a, **k: _completed(0, stdout="restarted"))
    r = client.post(ENDPOINT)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["redis_connected"] is True
    assert body["exit_code"] == 0
    assert fake.reconnect_calls == 1, "после перезапуска нужно переподключиться сразу"


def test_success_but_redis_down_reports_degraded(client, configured, monkeypatch):
    """Команда прошла, но сервис не поднялся — 200 со статусом degraded и подсказкой."""
    configured(connected=False)
    monkeypatch.setattr(system_mod.subprocess, "run", lambda *a, **k: _completed(0))
    r = client.post(ENDPOINT)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["redis_connected"] is False
    assert body["detail"], "при деградации нужна подсказка, что делать"


def test_command_is_taken_from_settings_not_request(client, configured, monkeypatch):
    """Строка команды приходит из конфигурации: тело запроса на неё не влияет."""
    configured(command="configured-command")
    seen: dict[str, object] = {}

    def _capture(args, **kwargs):
        seen["args"] = args
        return _completed(0)

    monkeypatch.setattr(system_mod.subprocess, "run", _capture)
    r = client.post(ENDPOINT, json={"command": "rm -rf /"})
    assert r.status_code == 200
    assert seen["args"] == ["configured-command"], f"выполнено не то: {seen['args']}"
    assert r.json()["command"] == "configured-command"


def test_shell_is_not_used(client, configured, monkeypatch):
    """``shell=False``: метасимволы не интерпретируются оболочкой."""
    configured(command="redis-restart-stub --force")
    captured: dict[str, object] = {}

    def _capture(args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return _completed(0)

    monkeypatch.setattr(system_mod.subprocess, "run", _capture)
    client.post(ENDPOINT)
    assert captured.get("shell") in (None, False), "shell=True открыл бы инъекцию команд"
    assert captured["args"] == ["redis-restart-stub", "--force"], "аргументы не разобраны"


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
        except Exception as exc:  # неожиданное — иначе «0 FAIL» врало бы
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- admin redis restart: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
