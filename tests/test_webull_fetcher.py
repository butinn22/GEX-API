"""Unit-тесты Webull fetcher: ретрай-логика ``_post_options`` (без сети).

Проверяем добавленную в Phase-3/Phase-4 логику ровно одного повтора на
транзиентных ошибках (429/5xx/сетевые) и диагностику ``_classify_drop``.
Транспорт (``_session.post``) подменяется заглушкой, ``_backoff_delay`` — нулевой,
поэтому внешние API не вызываются.
"""
from __future__ import annotations

import pandas as pd
import pytest
import requests

from gex.adapters.fetchers import webull_fetcher as wf


class _FakeResp:
    """Минимальный отклик: у ``_post_options`` читается только ``status_code``."""

    def __init__(self, status_code: int):
        self.status_code = status_code


def _fetcher() -> wf.WebullOptionsFetcher:
    return wf.WebullOptionsFetcher(max_expiries=2)


# --------------------------------------------------------------------- #
#  Ретрай _post_options (ровно один повтор)
# --------------------------------------------------------------------- #
def test_retry_once_after_429_then_success(monkeypatch):
    calls: list[int] = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResp(429)
        return _FakeResp(200)

    f = _fetcher()
    monkeypatch.setattr(f._session, "post", fake_post)
    monkeypatch.setattr(wf, "_backoff_delay", lambda attempt: 0.0)

    resp = f._post_options({"count": -1}, "AAPL")
    assert resp.status_code == 200
    assert len(calls) == 2


def test_retry_bounded_on_double_429(monkeypatch):
    """Двойной 429 не зацикливается: возвращается 429 после ровно 2 попыток."""
    calls: list[int] = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return _FakeResp(429)

    f = _fetcher()
    monkeypatch.setattr(f._session, "post", fake_post)
    monkeypatch.setattr(wf, "_backoff_delay", lambda attempt: 0.0)

    resp = f._post_options({}, "AAPL")
    assert resp.status_code == 429
    assert len(calls) == 2


def test_network_error_then_success(monkeypatch):
    calls: list[int] = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("boom")
        return _FakeResp(200)

    f = _fetcher()
    monkeypatch.setattr(f._session, "post", fake_post)
    monkeypatch.setattr(wf, "_backoff_delay", lambda attempt: 0.0)

    resp = f._post_options({}, "AAPL")
    assert resp.status_code == 200
    assert len(calls) == 2


def test_network_error_both_times_raises(monkeypatch):
    calls: list[int] = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError("boom")

    f = _fetcher()
    monkeypatch.setattr(f._session, "post", fake_post)
    monkeypatch.setattr(wf, "_backoff_delay", lambda attempt: 0.0)

    with pytest.raises(RuntimeError, match="network error"):
        f._post_options({}, "AAPL")
    assert len(calls) == 2


def test_fetch_raises_on_non_200_after_retries(monkeypatch):
    """Полный путь ``fetch``: после исчерпания ретрая не-200 → RuntimeError."""
    f = _fetcher()
    monkeypatch.setattr(
        wf, "_resolve_ticker_id", lambda ticker, session, timeout: 12345,
    )
    monkeypatch.setattr(
        f, "_post_options", lambda payload, ticker: _FakeResp(429),
    )

    with pytest.raises(RuntimeError, match="Webull HTTP 429"):
        f.fetch("AAPL")


# --------------------------------------------------------------------- #
#  Диагностика отбрасывания строк
# --------------------------------------------------------------------- #
def test_classify_drop_priority():
    # Отбрасываются только «пустые» контракты: без OI или с битым страйком.
    # Приоритет: OI → страйк.
    assert wf._classify_drop(0.0, 0.0) == "zero_oi"
    assert wf._classify_drop(0.0, 10.0) == "zero_oi"
    assert wf._classify_drop(10.0, 0.0) == "bad_strike"


def test_classify_drop_keeps_valid_row():
    assert wf._classify_drop(10.0, 10.0) is None


# --------------------------------------------------------------------- #
#  Фанаут по экспирациям
#
#  Регрессия, которую закрывает этот блок: базовый POST возвращает все
#  экспирации, но заполненной (OI > 0 / IV > 0) — только ближайшая.
#  Старая версия молча отбрасывала остальные фильтром ``oi > 0``, и GEX
#  считался по ОДНОЙ экспирации независимо от ``max_expiries``
#  (аудит 2026-09-17: SPY — 1 экспирация вместо 8).
# --------------------------------------------------------------------- #
def _json_resp(status_code: int, payload: dict) -> _FakeResp:
    """Отклик, у которого читают и ``status_code``, и ``json()``."""
    resp = _FakeResp(status_code)
    resp.json = lambda: payload  # type: ignore[method-assign]
    return resp


def _opt(strike: float, oi: float, direction: str = "call") -> dict:
    return {
        "strikePrice": strike,
        "openInterest": oi,
        "impVol": 0.2,
        "direction": direction,
        "unSymbol": "AAPL",
    }


def _expiry(days: int, options: list) -> dict:
    """Блок ``expireDateList`` для даты ``now + days``."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    date = (now + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    return {"from": {"date": date}, "data": options}


def _baseline() -> dict:
    """Типовой ответ Webull: OI только у ближайшей экспирации."""
    return {
        "close": 100.0,
        "expireDateList": [
            _expiry(7, [_opt(100.0, 10.0), _opt(100.0, 5.0, "put")]),
            _expiry(14, [_opt(100.0, 0.0), _opt(100.0, 0.0, "put")]),
            _expiry(21, [_opt(100.0, 0.0), _opt(100.0, 0.0, "put")]),
        ],
    }


class _FakeThreadSession:
    """Сессия фанаута: отвечает заполненной запрошенной экспирацией."""

    def __init__(self, sink: list):
        self._sink = sink

    def post(self, url, json=None, headers=None, timeout=None):
        self._sink.append(json)
        date = json["expireDate"]
        return _json_resp(200, {
            "expireDateList": [_expiry_from(date, [_opt(100.0, 20.0), _opt(100.0, 7.0, "put")])],
        })


def _expiry_from(date: str, options: list) -> dict:
    return {"from": {"date": date}, "data": options}


def _patch_transport(monkeypatch, fetcher, bodies: list, baseline=None):
    monkeypatch.setattr(wf, "_resolve_ticker_id", lambda t, s, to: 1)
    monkeypatch.setattr(wf, "_thread_session", lambda: _FakeThreadSession(bodies))
    monkeypatch.setattr(
        fetcher._session, "post",
        lambda *a, **k: _json_resp(200, baseline if baseline is not None else _baseline()),
    )


def test_fan_out_fills_expiries_without_baseline_oi(monkeypatch):
    """Две экспирации без OI в базовом ответе дозапрашиваются отдельными POST."""
    bodies: list[dict] = []
    f = wf.WebullOptionsFetcher(max_expiries=3)
    _patch_transport(monkeypatch, f, bodies)

    snap = f.fetch("AAPL")
    meta = snap.meta["webull"]

    assert meta["expiries_selected"] == 3
    assert meta["expiries_from_baseline"] == 1     # только ближайшая
    assert meta["expiries_fetched"] == 2           # остальные — фанаутом
    assert meta["expiries_failed"] == 0
    assert meta["requests"] == 3                   # базовый + 2

    # Цепочка содержит все три экспирации, а не одну.
    assert int((snap.chain["T"] * 365).round().nunique()) == 3

    # Полный OI: 15 из базового + 27 × 2 из дозапрошенных.
    total = float(snap.chain["oi"].sum())
    assert total > 15.0, "GEX должен считаться по всей выборке, а не по одной экспирации"
    assert total == pytest.approx(69.0)


def test_fan_out_payload_carries_expiredate_and_unsymbol(monkeypatch):
    """Каждый запрос экспирации несёт и ``expireDate``, и ``unSymbol``.

    Без ``unSymbol`` Webull отвечает HTTP 417
    «UnSymbol can't be null when expireDate is not null!» — именно это
    сообщение и подсказало рабочий формат запроса.
    """
    bodies: list[dict] = []
    f = wf.WebullOptionsFetcher(max_expiries=3)
    _patch_transport(monkeypatch, f, bodies)

    f.fetch("AAPL")

    assert len(bodies) == 2
    for body in bodies:
        assert body["expireDate"]
        assert body["unSymbol"] == "AAPL"
        assert body["tickerId"] == 1
        assert body["count"] == -1


def test_max_days_trims_expiries_and_requests(monkeypatch):
    """Горизонт режет экспирации до запросов: экономия без потери данных."""
    bodies: list[dict] = []
    f = wf.WebullOptionsFetcher(max_expiries=3, max_days=10)
    _patch_transport(monkeypatch, f, bodies)

    snap = f.fetch("AAPL")
    meta = snap.meta["webull"]

    # 14 и 21 день вне горизонта — их анализатор всё равно отбросил бы.
    assert meta["expiries_selected"] == 1
    assert bodies == []
    assert meta["requests"] == 1
    assert int((snap.chain["T"] * 365).round().nunique()) == 1


def test_failed_expiry_does_not_kill_the_load(monkeypatch):
    """Ошибка одной экспирации не роняет загрузку: остаётся базовая."""
    bodies: list[dict] = []

    class _BrokenSession:
        def post(self, url, json=None, headers=None, timeout=None):
            bodies.append(json)
            return _json_resp(500, {})

    f = wf.WebullOptionsFetcher(max_expiries=3)
    monkeypatch.setattr(wf, "_resolve_ticker_id", lambda t, s, to: 1)
    monkeypatch.setattr(wf, "_thread_session", lambda: _BrokenSession())
    monkeypatch.setattr(
        f._session, "post", lambda *a, **k: _json_resp(200, _baseline()),
    )

    snap = f.fetch("AAPL")          # не бросает
    meta = snap.meta["webull"]
    assert meta["expiries_failed"] == 2
    assert meta["expiries_fetched"] == 0
    assert int((snap.chain["T"] * 365).round().nunique()) == 1  # только базовая
