"""Tests for :mod:`gex.moex_candles_fetcher` and MOEX OHLCV routing.

Run with::

    python -m pytest tests/test_moex_candles.py -q

All tests are hermetic (no network): ``requests.get`` is monkeypatched to
return canned ISS responses. The explicitly network-marked test is skipped
automatically when offline.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from gex.adapters.fetchers.moex_candles_fetcher import (
    MOEXCandlesFetcher,
    _MOEX_ALIASES,
    _MOEX_INSTRUMENTS,
    _MOEX_OHLCV_ASSETS,
    _parse_msk_to_utc,
    TIMEFRAMES,
)


# ====================================================================== #
#  Canned ISS payloads
# ====================================================================== #
# FORTS securities.json — два активных RTS-контракта и один просроченный.
# Даты генерируются относительно «сегодня»: экспирация ближнего обязана быть
# в будущем, иначе тест ломался бы по календарю (прежняя завязка на RIU6 с
# LTD 2026-09-17 перестала работать в конце сентября 2026 — ближний стал
# просроченным, и правильно побеждал RIZ6).
# ====================================================================== #
def _ltd(days: int) -> str:
    """LASTTRADEDATE как строка YYYY-MM-DD: сегодня + days дней (UTC)."""
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


_SECS_NEAR_SECID = "RIN6"  # ближний (побеждает)
_SECS_FAR_SECID = "RIF6"   # дальний (должен проиграть)
_SECS_COLUMNS = ["SECID", "SHORTNAME", "ASSETCODE", "LASTTRADEDATE"]
_SECS_DATA = [
    [_SECS_FAR_SECID, "RTS-far", "RTS", _ltd(+120)],  # дальний (должен проиграть)
    [_SECS_NEAR_SECID, "RTS-near", "RTS", _ltd(+30)],  # ближний (побеждает)
    ["RIH6", "RTS-expired", "RTS", _ltd(-60)],          # просроченный (отбрасывается)
]
_SECURITIES_PAYLOAD = {
    "securities": {"columns": _SECS_COLUMNS, "data": _SECS_DATA},
}


def _candles_payload(rows: list[list[Any]]) -> dict:
    """Build a candles.json payload with the canonical ISS column order.

    ISS columns: open, close, high, low, value, volume, begin, end.
    """
    return {
        "candles": {
            "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
            "data": rows,
        }
    }


# 5 часовых свечей (MSK 10:00..14:00) → UTC 07:00..11:00.
_HOURLY_ROWS = [
    [100.0, 101.0, 102.0,  99.0, 0, 1000, "2026-07-17 10:00:00", "2026-07-17 10:59:59"],
    [101.0, 103.0, 104.0, 100.0, 0, 1500, "2026-07-17 11:00:00", "2026-07-17 11:59:59"],
    [103.0, 102.0, 105.0, 101.0, 0, 1200, "2026-07-17 12:00:00", "2026-07-17 12:59:59"],
    [102.0, 106.0, 107.0, 101.0, 0, 2000, "2026-07-17 13:00:00", "2026-07-17 13:59:59"],
    [106.0, 108.0, 109.0, 105.0, 0, 1800, "2026-07-17 14:00:00", "2026-07-17 14:59:59"],
]
_DAILY_ROWS = [
    [100.0, 108.0, 110.0,  98.0, 0, 50000, "2026-07-16 00:00:00", "2026-07-16 23:59:59"],
    [108.0, 112.0, 115.0, 107.0, 0, 60000, "2026-07-17 00:00:00", "2026-07-17 23:59:59"],
]


class _FakeResp:
    """Minimal stand-in for requests.Response — returns canned JSON."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _patch_requests(monkeypatch, candles_by_interval: dict[int, dict]):
    """Patch requests.get so securities + candles return canned payloads.

    `candles_by_interval` maps ISS interval code (60, 24) -> payload.
    The FORTS securities.json payload is returned for the securities URL.
    The candles URL returns payload keyed by the `interval` query param,
    regardless of page (`start`).
    """
    import requests

    def fake_get(url, params=None, timeout=None, headers=None, **kw):
        if "securities.json" in url and "candles" not in url:
            return _FakeResp(_SECURITIES_PAYLOAD)
        if "candles.json" in url:
            interval = int(params.get("interval"))
            payload = candles_by_interval.get(interval)
            if payload is None:
                return _FakeResp({"candles": {"columns": [], "data": []}})
            return _FakeResp(payload)
        return _FakeResp({})

    monkeypatch.setattr(requests, "get", fake_get)
    # Clean class-level caches so tests don't leak into each other.
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_frontmonth", {})
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_candles", {})


# ====================================================================== #
#  1. Конфиг инструментов
# ====================================================================== #
def test_assets_set_contains_all_families():
    """Currency / index / stock / alias tickers all in _MOEX_OHLCV_ASSETS."""
    for t in ("USDRUB", "CNYRUB", "RTS", "MIX", "SBER", "GAZP", "LKOH"):
        assert t in _MOEX_OHLCV_ASSETS
    # Legacy aliases (Si/CNY from options fetcher) still accepted.
    assert "SI" in _MOEX_OHLCV_ASSETS and "CNY" in _MOEX_OHLCV_ASSETS
    assert TIMEFRAMES == ("1h", "2h", "4h", "1d")


def test_aliases_map_to_currency_perps():
    """SI -> USDRUB, CNY -> CNYRUB (the perpetual-futures currency route)."""
    assert _MOEX_ALIASES["SI"] == "USDRUB"
    assert _MOEX_ALIASES["CNY"] == "CNYRUB"


def test_instrument_kinds_correct():
    """Each family resolves to the expected engine/board/kind."""
    assert _MOEX_INSTRUMENTS["USDRUB"]["kind"] == "currency_perp"
    assert _MOEX_INSTRUMENTS["USDRUB"]["secid"] == "USDRUBF"
    assert _MOEX_INSTRUMENTS["USDRUB"]["engine"] == "futures"
    assert _MOEX_INSTRUMENTS["CNYRUB"]["secid"] == "CNYRUBF"
    assert _MOEX_INSTRUMENTS["RTS"]["kind"] == "index_future"
    assert _MOEX_INSTRUMENTS["RTS"]["prefix"] == "RI"
    assert _MOEX_INSTRUMENTS["SBER"]["kind"] == "stock"
    assert _MOEX_INSTRUMENTS["SBER"]["engine"] == "stock"
    assert _MOEX_INSTRUMENTS["SBER"]["market"] == "shares"
    assert _MOEX_INSTRUMENTS["SBER"]["board"] == "TQBR"


# ====================================================================== #
#  2. MSK -> UTC conversion
# ====================================================================== #
def test_parse_msk_to_utc_shifts_minus_3h():
    ts = _parse_msk_to_utc("2026-07-17 10:00:00")
    assert ts is not None
    assert str(ts.tz) == "UTC"
    assert ts.hour == 7   # 10:00 MSK -> 07:00 UTC


def test_parse_msk_to_utc_invalid_returns_none():
    assert _parse_msk_to_utc(None) is None
    assert _parse_msk_to_utc("") is None
    assert _parse_msk_to_utc("not-a-date") is None


# ====================================================================== #
#  3. Резолв secid по типу инструмента
# ====================================================================== #
def test_resolve_secid_currency_perp_fixed(monkeypatch):
    """Currency perps have a fixed secid (USDRUBF), no ISS lookup."""
    _patch_requests(monkeypatch, {})
    f = MOEXCandlesFetcher()
    cfg = _MOEX_INSTRUMENTS["USDRUB"]
    assert f._resolve_secid("USDRUB", cfg) == "USDRUBF"


def test_resolve_secid_stock_fixed(monkeypatch):
    """Stocks use the ticker itself as secid (no ISS lookup)."""
    _patch_requests(monkeypatch, {})
    f = MOEXCandlesFetcher()
    cfg = _MOEX_INSTRUMENTS["SBER"]
    assert f._resolve_secid("SBER", cfg) == "SBER"


def test_resolve_secid_index_future_picks_nearest(monkeypatch):
    """index_future -> nearest-expiry contract (min LASTTRADEDATE in future).

    Canned securities list: near (today+30d), far (today+120d), expired
    (today-60d). The expired contract must be rejected, and among the two
    future contracts the nearer one wins.
    """
    _patch_requests(monkeypatch, {})
    f = MOEXCandlesFetcher()
    cfg = _MOEX_INSTRUMENTS["RTS"]
    secid = f._resolve_secid("RTS", cfg)
    assert secid == _SECS_NEAR_SECID  # ближайший, не дальний


# ====================================================================== #
#  4. fetch() — контракт DataFrame
# ====================================================================== #
def test_fetch_returns_4_timeframes_canonical_columns(monkeypatch):
    """fetch() -> dict 1h/2h/4h/1d, Open/High/Low/Close/Volume, tz=UTC index."""
    _patch_requests(monkeypatch, {
        60: _candles_payload(_HOURLY_ROWS),
        24: _candles_payload(_DAILY_ROWS),
    })
    f = MOEXCandlesFetcher()
    tfs = f.fetch("SBER")   # stock -> fixed secid, no securities lookup
    assert set(tfs.keys()) == {"1h", "2h", "4h", "1d"}
    for tf, df in tfs.items():
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert str(df.index.tz) == "UTC"
        assert df.index.is_monotonic_increasing
        assert not df[["Open", "High", "Low", "Close"]].isna().any().any()


def test_fetch_msk_to_utc_applied(monkeypatch):
    """Hourly begin (10:00 MSK) lands at 07:00 UTC in the index."""
    _patch_requests(monkeypatch, {
        60: _candles_payload(_HOURLY_ROWS),
        24: _candles_payload(_DAILY_ROWS),
    })
    f = MOEXCandlesFetcher()
    tfs = f.fetch("USDRUB")
    assert tfs["1h"].index[0].hour == 7


def test_fetch_resample_matches_canonical_agg(monkeypatch):
    """2h/4h aggregation matches a direct pandas resample of 1h (the contract)."""
    _patch_requests(monkeypatch, {
        60: _candles_payload(_HOURLY_ROWS),
        24: _candles_payload(_DAILY_ROWS),
    })
    f = MOEXCandlesFetcher()
    tfs = f.fetch("SBER")
    df1h, df2h, df4h = tfs["1h"], tfs["2h"], tfs["4h"]
    assert len(df2h) < len(df1h)
    assert len(df4h) <= len(df2h)

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ref2h = df1h.resample("2h", label="left", closed="left").agg(agg).dropna(
        subset=["Open", "High", "Low", "Close"])
    ref2h["Volume"] = ref2h["Volume"].fillna(0.0)
    pd.testing.assert_frame_equal(df2h.sort_index(), ref2h.sort_index())


def test_fetch_spot_returns_last_hourly_close(monkeypatch):
    """fetch_spot = last Close of 1h (= 108.0 from canned data)."""
    _patch_requests(monkeypatch, {
        60: _candles_payload(_HOURLY_ROWS),
        24: _candles_payload(_DAILY_ROWS),
    })
    f = MOEXCandlesFetcher()
    assert f.fetch_spot("USDRUB") == 108.0


def test_fetch_accepts_aliases(monkeypatch):
    """SI/CNY aliases route to USDRUB/CNYRUB (same data)."""
    _patch_requests(monkeypatch, {
        60: _candles_payload(_HOURLY_ROWS),
        24: _candles_payload(_DAILY_ROWS),
    })
    f = MOEXCandlesFetcher()
    assert f.fetch_spot("SI") == 108.0
    assert f.fetch_spot("CNY") == 108.0


def test_fetch_accepts_lowercase(monkeypatch):
    """Lowercase ticker is accepted and canonicalized."""
    _patch_requests(monkeypatch, {
        60: _candles_payload(_HOURLY_ROWS),
        24: _candles_payload(_DAILY_ROWS),
    })
    f = MOEXCandlesFetcher()
    assert f.fetch_spot("sber") == 108.0


# ====================================================================== #
#  5. Ошибки и валидация
# ====================================================================== #
def test_fetch_rejects_unsupported_ticker():
    f = MOEXCandlesFetcher()
    with pytest.raises(ValueError, match="Неподдерживаемый MOEX"):
        f.fetch("BRZ")


def test_fetch_raises_on_empty_hourly(monkeypatch):
    _patch_requests(monkeypatch, {
        60: {"candles": {"columns": ["open"], "data": []}},
        24: _candles_payload(_DAILY_ROWS),
    })
    f = MOEXCandlesFetcher()
    with pytest.raises(ValueError, match="часовую историю"):
        f.fetch("SBER")


def test_fetch_survives_missing_daily(monkeypatch):
    """Empty 1d is non-fatal — 1h/2h/4h still returned, 1d omitted."""
    _patch_requests(monkeypatch, {
        60: _candles_payload(_HOURLY_ROWS),
        24: {"candles": {"columns": ["open"], "data": []}},
    })
    f = MOEXCandlesFetcher()
    tfs = f.fetch("SBER")
    assert {"1h", "2h", "4h"} <= set(tfs.keys())
    assert "1d" not in tfs


# ====================================================================== #
#  6. Пагинация (iss.reverse=true + start offset)
# ====================================================================== #
def test_pagination_concatenates_pages(monkeypatch):
    """Multi-page responses (start=0, 500, ...) are concatenated into one DF.

    ISS caps at 500 rows/request; the fetcher pages via `start` until data
    ends. A full page (== _ISS_PAGE_SIZE) signals "more data may exist", so
    the fetcher requests the next page. Here page 0 returns exactly
    _ISS_PAGE_SIZE rows (only 3 distinct, rest are fillers to hit the cap),
    page 1 returns 2 more — the merged result keeps all 5 distinct bars.
    """
    import requests
    from gex.adapters.fetchers.moex_candles_fetcher import _ISS_PAGE_SIZE

    distinct_p0 = [
        [100.0, 101.0, 102.0, 99.0, 0, 1000, "2026-07-17 10:00:00", "2026-07-17 10:59:59"],
        [101.0, 102.0, 103.0, 100.0, 0, 1100, "2026-07-17 11:00:00", "2026-07-17 11:59:59"],
        [102.0, 103.0, 104.0, 101.0, 0, 1200, "2026-07-17 12:00:00", "2026-07-17 12:59:59"],
    ]
    # Pad page 0 to exactly _ISS_PAGE_SIZE rows so pagination continues.
    page0 = list(distinct_p0)
    for i in range(_ISS_PAGE_SIZE - len(distinct_p0)):
        page0.append(distinct_p0[-1])  # duplicate filler rows (dedup later)
    page1 = [
        [103.0, 104.0, 105.0, 102.0, 0, 1300, "2026-07-17 13:00:00", "2026-07-17 13:59:59"],
        [104.0, 105.0, 106.0, 103.0, 0, 1400, "2026-07-17 14:00:00", "2026-07-17 14:59:59"],
    ]
    page_payloads = {0: _candles_payload(page0), 1: _candles_payload(page1)}

    def fake_get(url, params=None, timeout=None, headers=None, **kw):
        if "candles.json" in url:
            interval = int(params.get("interval"))
            if interval == 60:
                start = int(params.get("start", 0))
                p = page_payloads.get(start // _ISS_PAGE_SIZE)
                return _FakeResp(p if p is not None else {"candles": {"columns": [], "data": []}})
            if interval == 24:
                return _FakeResp({"candles": {"columns": [], "data": []}})
        return _FakeResp({})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_frontmonth", {})
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_candles", {})

    f = MOEXCandlesFetcher()
    tfs = f.fetch("SBER")
    # 3 distinct page-0 + 2 page-1 = 5 bars, sorted, deduplicated.
    assert len(tfs["1h"]) == 5
    assert tfs["1h"].index.is_monotonic_increasing
    assert tfs["1h"].index.is_unique


def test_pagination_uses_reverse_param(monkeypatch):
    """The fetcher MUST send iss.reverse=true (newest-first) to avoid the
    'old data returned instead of fresh' bug."""
    import requests

    seen_reverse = []

    def fake_get(url, params=None, timeout=None, headers=None, **kw):
        if "candles.json" in url:
            seen_reverse.append(params.get("iss.reverse"))
            # Return one row then empty to stop pagination.
            if int(params.get("start", 0)) == 0:
                return _FakeResp(_candles_payload([_HOURLY_ROWS[0]]))
            return _FakeResp({"candles": {"columns": [], "data": []}})
        if "securities.json" in url:
            return _FakeResp(_SECURITIES_PAYLOAD)
        return _FakeResp({})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_frontmonth", {})
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_candles", {})

    f = MOEXCandlesFetcher()
    f.fetch("SBER")
    assert seen_reverse, "no candles request was made"
    assert all(v == "true" for v in seen_reverse), f"iss.reverse not 'true': {seen_reverse}"


# ====================================================================== #
#  7. Кэширование
# ====================================================================== #
def test_candles_cached_within_ttl(monkeypatch):
    """Repeated fetch() within TTL reuses the cached DataFrame (no 2nd network)."""
    call_count = {"n": 0}
    candles = {60: _candles_payload(_HOURLY_ROWS), 24: _candles_payload(_DAILY_ROWS)}

    import requests

    def counting_get(url, params=None, timeout=None, headers=None, **kw):
        call_count["n"] += 1
        if "candles.json" in url:
            return _FakeResp(candles.get(int(params.get("interval")),
                                         {"candles": {"columns": [], "data": []}}))
        return _FakeResp({})

    monkeypatch.setattr(requests, "get", counting_get)
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_frontmonth", {})
    monkeypatch.setattr(MOEXCandlesFetcher, "_cache_candles", {})

    MOEXCandlesFetcher().fetch("SBER")
    after_first = call_count["n"]
    MOEXCandlesFetcher().fetch("SBER")   # class-level cache hit
    assert call_count["n"] == after_first


# ====================================================================== #
#  8. Конструктор валидация
# ====================================================================== #
def test_constructor_rejects_bad_params():
    with pytest.raises(ValueError):
        MOEXCandlesFetcher(timeout=0)
    with pytest.raises(ValueError):
        MOEXCandlesFetcher(timeout=-1)
    with pytest.raises(ValueError):
        MOEXCandlesFetcher(max_bars=-5)


# ====================================================================== #
#  9. Live-смок (только онлайн)
# ====================================================================== #
def _online() -> bool:
    import socket
    try:
        socket.create_connection(("iss.moex.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _online(), reason="network unavailable — live MOEX smoke skipped")
def test_live_fetch_returns_fresh_data():
    """Live ISS fetch: last 1h bar is recent (within 3 days), not months-old."""
    from datetime import datetime, timezone
    f = MOEXCandlesFetcher()
    for ticker in ("MIX", "USDRUB", "SBER"):
        tfs = f.fetch(ticker)
        assert {"1h", "2h", "4h"} <= set(tfs.keys())
        last = tfs["1h"].index[-1]
        age_days = (datetime.now(timezone.utc) - last.to_pydatetime()).days
        assert age_days <= 3, f"{ticker}: last 1h bar is {age_days}d old (pagination bug?)"
        assert tfs["1h"]["Close"].iloc[-1] > 0
