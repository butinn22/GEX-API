"""Единый HTTP-транспорт: политика повторов, классификация ошибок, таймауты.

Тесты идут **без сети и без установленных сетевых библиотек**: отправитель, часы и генератор
случайных чисел подставляются снаружи. Так проверяется именно логика транспорта, а не доступность
интернета (окружение без requests — штатная ситуация для этого репозитория, см. STATUS §4.5).

    python tests/test_http_transport.py
    pytest tests/test_http_transport.py -q
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Повторы — штатная ситуация для этих тестов, и каждый из них пишет warning. Молчим,
# чтобы вывод был читаемым; проверяется логика, а не текст сообщений.
logging.getLogger("gex.adapters.transport.http").setLevel(logging.CRITICAL)

from gex.adapters.transport.http import (  # noqa: E402
    HttpRequest,
    HttpResponse,
    HttpTransport,
    HttpTransportError,
    PermanentHttpError,
    RateLimitedError,
    RetryPolicy,
    TransientHttpError,
    backoff_seconds,
    classify_exception,
    classify_status,
)


# ── помощники ────────────────────────────────────────────────────────────────

class _Recorder:
    """Подставной отправитель: отдаёт ответы по списку, считает вызовы."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.calls.append(request)
        item = self.responses.pop(0) if self.responses else HttpResponse(200, "{}")
        if isinstance(item, Exception):
            raise item
        return item


def _transport(recorder, **kwargs) -> HttpTransport:
    kwargs.setdefault("sleeper", lambda _seconds: None)
    return HttpTransport(recorder, **kwargs)


def test_module_imports_without_network_libraries():
    """Импорт транспорта не тянет requests/httpx: ``requests`` импортируется только в sender'е."""
    for name in ("gex.adapters.transport.http", "requests", "httpx"):
        sys.modules.pop(name, None)

    before = set(sys.modules)
    importlib.import_module("gex.adapters.transport.http")
    leaked = {"requests", "httpx", "urllib3"} & {m.split(".")[0] for m in set(sys.modules) - before}
    assert not leaked, f"транспорт импортирует сеть на этапе импорта: {sorted(leaked)}"


def test_successful_request_does_not_retry():
    recorder = _Recorder([HttpResponse(200, '{"ok": true}')])
    events = []
    transport = _transport(recorder, on_attempt=lambda req, err: events.append(err))

    response = transport.get("https://provider/x", params={"a": 1}, expect_json=True)

    assert response.json() == {"ok": True}
    assert len(recorder.calls) == 1
    assert events == [None], "успешная попытка обязана попадать в метрику"


def test_retry_then_success_on_transient_status():
    recorder = _Recorder([HttpResponse(503, "busy"), HttpResponse(200, "{}")])
    delays = []
    transport = _transport(recorder, sleeper=delays.append)

    transport.get("https://provider/x")

    assert len(recorder.calls) == 2, "5xx обязателен к повтору"
    assert len(delays) == 1 and delays[0] > 0


def test_client_error_is_not_retried():
    recorder = _Recorder([HttpResponse(404, "missing")])
    transport = _transport(recorder)

    try:
        transport.get("https://provider/x")
    except PermanentHttpError as exc:
        assert exc.status == 404
    else:
        raise AssertionError("ожидалась PermanentHttpError")

    assert len(recorder.calls) == 1, "повтор 4xx — это потраченный лимит провайдера"


def test_all_attempts_exhausted_raises_last_error():
    recorder = _Recorder([HttpResponse(500, "boom")] * 5)
    transport = _transport(recorder, policy=RetryPolicy(max_attempts=3))

    try:
        transport.get("https://provider/x")
    except HttpTransportError as exc:
        assert exc.status == 500
    else:
        raise AssertionError("ожидалась ошибка транспорта")

    assert len(recorder.calls) == 3, "max_attempts считает попытки, а не повторы"


def test_retry_after_header_wins_over_backoff():
    recorder = _Recorder([
        HttpResponse(429, "slow down", headers={"Retry-After": "3"}),
        HttpResponse(200, "{}"),
    ])
    delays = []
    transport = _transport(recorder, sleeper=delays.append)

    transport.get("https://provider/x")

    assert delays == [3.0], "если провайдер назвал время — ждём его, а не своё"


def test_retry_after_is_capped():
    """«Приходите через час» не должно превращаться в часовой sleep в воркере."""
    recorder = _Recorder([
        HttpResponse(429, "slow down", headers={"Retry-After": "3600"}),
        HttpResponse(200, "{}"),
    ])
    delays = []
    transport = _transport(
        recorder, sleeper=delays.append, policy=RetryPolicy(max_delay=8.0)
    )

    transport.get("https://provider/x")

    assert delays == [8.0]


def test_unparsable_retry_after_falls_back_to_backoff():
    recorder = _Recorder([
        HttpResponse(429, "slow", headers={"Retry-After": "soon"}),
        HttpResponse(200, "{}"),
    ])
    delays = []
    transport = _transport(recorder, sleeper=delays.append, policy=RetryPolicy(jitter=0.0))

    transport.get("https://provider/x")

    assert delays == [0.5]


def test_elapsed_budget_stops_retrying():
    """Общий бюджет важнее числа попыток: медленный провайдер нельзя долбить бесконечно."""
    ticks = iter([0.0, 0.5, 40.0, 40.1, 40.2, 40.3])
    recorder = _Recorder([HttpResponse(500, "boom")] * 10)
    transport = _transport(
        recorder,
        policy=RetryPolicy(max_attempts=5, max_elapsed=1.0),
        clock=lambda: next(ticks),
    )

    try:
        transport.get("https://provider/x")
    except HttpTransportError:
        pass
    else:
        raise AssertionError("ожидалась ошибка транспорта")

    assert len(recorder.calls) == 2, (
        "бюджет важнее max_attempts: при max_attempts=5 было сделано 2 вызова, "
        "потому что к третьему бюджет уже исчерпан"
    )


def test_network_exception_is_retried_and_programming_error_is_not():
    """``OSError`` (в т. ч. requests.RequestException) — повторяем; ``ValueError`` — нет."""
    recorder = _Recorder([TimeoutError("timeout"), HttpResponse(200, "{}")])
    transport = _transport(recorder)
    transport.get("https://provider/x")
    assert len(recorder.calls) == 2

    recorder = _Recorder([ValueError("bad argument")])
    transport = _transport(recorder)
    try:
        transport.get("https://provider/x")
    except PermanentHttpError:
        pass
    else:
        raise AssertionError("ошибка программирования не должна повторяться")
    assert len(recorder.calls) == 1


def test_broken_json_is_permanent_but_checked_before_return():
    recorder = _Recorder([HttpResponse(200, "<html>not json</html>")])
    transport = _transport(recorder, policy=RetryPolicy(max_attempts=3))

    try:
        transport.get("https://provider/x", expect_json=True)
    except PermanentHttpError:
        pass
    else:
        raise AssertionError("битый JSON — не повод повторять запрос")

    assert len(recorder.calls) == 1


def test_json_is_parsed_once():
    response = HttpResponse(200, '{"v": 1}', url="https://provider/x")
    assert response.json() == {"v": 1}
    response.text = '{"v": 2}'  # после первого разбора тело уже не перечитывается
    assert response.json() == {"v": 1}


def test_request_defaults_do_not_mutate_source():
    original = HttpRequest("GET", "https://provider/x", headers={"X-Trace": "1"})
    transport = _transport(_Recorder([]), default_headers={"User-Agent": "gex"})

    prepared = original.with_defaults(headers=transport._default_headers, timeout=20.0)

    assert prepared.timeout == 20.0 and original.timeout is None
    assert prepared.headers == {"User-Agent": "gex", "X-Trace": "1"}
    assert original.headers == {"X-Trace": "1"}, "исходный запрос не должен меняться"


def test_backoff_grows_exponentially_and_is_capped():
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0, jitter=0.0)
    delays = [backoff_seconds(attempt, policy, rng=lambda: 0.5) for attempt in range(1, 6)]

    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0]
    assert delays[-1] == policy.max_delay, "ожидание обязано иметь потолок"


def test_backoff_jitter_stays_within_bounds():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=0.2)
    for attempt in (1, 2, 3):
        for draw in (0.0, 0.5, 1.0):
            delay = backoff_seconds(attempt, policy, rng=lambda d=draw: d)
            expected = min(policy.base_delay * 2 ** (attempt - 1), policy.max_delay)
            assert abs(delay - expected) <= expected * policy.jitter + 1e-9


def test_retry_policy_validates_its_arguments():
    for bad in ({"max_attempts": 0}, {"jitter": 1.5}):
        try:
            RetryPolicy(**bad)
        except ValueError:
            continue
        raise AssertionError(f"ожидалась ошибка валидации для {bad}")


def test_error_taxonomy_maps_status_and_exceptions():
    assert classify_status(204) is None
    assert isinstance(classify_status(429), RateLimitedError)
    assert isinstance(classify_status(503), TransientHttpError)
    assert isinstance(classify_status(404), PermanentHttpError)

    timeout = TimeoutError("x")
    assert isinstance(classify_exception(timeout), TransientHttpError)
    assert isinstance(classify_exception(OSError("x")), TransientHttpError)
    assert isinstance(classify_exception(KeyError("x")), PermanentHttpError)

    already = RateLimitedError("x", retry_after=2.0)
    assert classify_exception(already) is already, "свою ошибку не переклассифицируем"
    assert isinstance(classify_status(500), TransientHttpError)


def test_transport_errors_are_runtime_errors():
    """Совместимость: вызывающий код исторически ловил ``RuntimeError``."""
    assert issubclass(HttpTransportError, RuntimeError)
    assert issubclass(RateLimitedError, TransientHttpError)
    assert TransientHttpError.retryable and not PermanentHttpError.retryable


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
    print(f"--- http transport: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
