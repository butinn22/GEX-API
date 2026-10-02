"""FIFO-кэш закрытых баров и догрузка в ``TATimeframesFetcher``.

Проверяемые инварианты (требования к слою):

1. **FIFO-500**: серия удерживает только ``max_bars`` последних баров; новые дописываются
   в хвост, старейшие вытесняются из головы.
2. **Запись только при новых барах**: ``update`` с барами не новее последнего сохранённого
   ничего не меняет в базах (и не пишет в Redis); ``added`` честно равен числу новых баров.
3. **Ярус памяти** работает без Redis и переживает его отказ; **ярус Redis** разделяем
   между экземплярами (другой процесс читает то, что записал первый).
4. **Расписание проверок** (``is_due``/``next_close_after``): провайдер опрашивается только
   после расчётного закрытия следующего бара — с учётом торговых сессий (сб/вс — нет баров)
   и круглосуточных рынков.
5. **Фетчер** больше не перекачивает историю: повторный вызов не ходит к провайдеру;
   догрузка идёт только за хвостом после последнего бара; 2h/4h — производные той же
   часовой серии.

Redis подменяется фейком — тесты проверяют поведение, а не сеть.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import types

import pandas as pd
import pytest

from gex.adapters.cache.bar_store import (
    BAR_SCHEMA_VERSION,
    DEFAULT_MAX_BARS,
    BarStore,
    CachedBars,
    bars_to_frame,
    frame_to_bars,
    next_close_after,
)
from gex.adapters.cache.keys import bars_key
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher

MSK = timezone(timedelta(hours=3))

#: Среда 2026-09-23 14:00 МСК — середина дня, до закрытия сессии США (16:30–23:00 МСК).
WED_14_MSK_TS = datetime(2026, 9, 23, 14, 0, tzinfo=MSK).timestamp()


class Clock:
    """Управляемые часы: тест двигает время, а не ждёт его."""

    def __init__(self, start: float = WED_14_MSK_TS):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRedis:
    """Минимальный совместимый с RedisClient фейк: строки + счётчик записей."""

    def __init__(self, *, fail: bool = False):
        self.kv: dict[str, bytes] = {}
        self.fail = fail
        self.connected = True
        self.set_calls = 0
        self.del_calls = 0

    def _check(self):
        if self.fail:
            raise ConnectionError("redis down")

    def get(self, key):
        self._check()
        return self.kv.get(key)

    def set(self, key, value, ex=None, *, px=None, nx=False):
        self._check()
        if nx and key in self.kv:
            return None
        self.kv[key] = value.encode("utf-8") if isinstance(value, str) else value
        self.set_calls += 1
        return True

    def delete(self, key):
        self._check()
        if key in self.kv:
            del self.kv[key]
            self.del_calls += 1
            return True
        return False


def make_bars(n: int, start_ts: float = WED_14_MSK_TS, step: int = 3600) -> list[dict]:
    """``n`` часовых баров, начиная с ``start_ts`` (закрытие каждого — ровно в ``start + i*step``)."""
    bars = []
    for i in range(n):
        ts = start_ts - (n - 1 - i) * step  # последний бар закрывается в start_ts
        bars.append(
            {
                "t": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "o": 100.0 + i, "h": 101.0 + i, "l": 99.0 + i, "c": 100.5 + i, "v": 1000.0 + i,
            }
        )
    return bars


def _store(redis=None, clock=None, **kwargs) -> BarStore:
    return BarStore(redis, clock=clock or Clock(), **kwargs)


# ====================================================================== #
# 1. FIFO-поведение серии
# ====================================================================== #
class TestFifoRetention:

    def test_over_limit_keeps_only_newest_bars(self):
        store = _store()
        entry, added = store.update("SPY", "1h", make_bars(520))

        assert added == 520
        assert len(entry) == DEFAULT_MAX_BARS
        assert entry.last_ts == pytest.approx(WED_14_MSK_TS)
        # Голова — самый старый из оставшихся: вытеснены ровно первые 20 баров.
        assert _first_ts(entry) == pytest.approx(WED_14_MSK_TS - (500 - 1) * 3600)

    def test_incremental_append_evicts_oldest_in_fifo_order(self):
        store = _store()
        store.update("SPY", "1h", make_bars(500))

        # Закрылись два новых бара (после 14:00).
        new_bars = [
            {"t": datetime.fromtimestamp(WED_14_MSK_TS + 3600, tz=timezone.utc).isoformat(),
             "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
            {"t": datetime.fromtimestamp(WED_14_MSK_TS + 7200, tz=timezone.utc).isoformat(),
             "o": 2, "h": 2, "l": 2, "c": 2, "v": 2},
        ]
        entry, added = store.update("SPY", "1h", new_bars)

        assert added == 2
        assert len(entry) == DEFAULT_MAX_BARS
        assert entry.last_ts == pytest.approx(WED_14_MSK_TS + 7200)
        # FIFO: ушли два старейших (голова сдвинулась на 2 часа).
        assert _first_ts(entry) == pytest.approx(WED_14_MSK_TS - (500 - 3) * 3600)

    def test_duplicate_bars_are_deduped(self):
        store = _store()
        store.update("SPY", "1h", make_bars(100))
        _, added = store.update("SPY", "1h", make_bars(100))  # те же бары ещё раз

        assert added == 0
        assert len(store.get("SPY", "1h")) == 100


def _first_ts(entry: CachedBars) -> float:
    return datetime.fromisoformat(entry.bars[0]["t"].replace("Z", "+00:00")).timestamp()


# ====================================================================== #
# 2. Запись только при новых барах
# ====================================================================== #
class TestWriteOnChangeOnly:

    def test_unchanged_update_does_not_touch_redis(self):
        redis = FakeRedis()
        store = _store(redis)
        store.update("SPY", "1d", make_bars(100, step=86400))
        writes_after_bootstrap = redis.set_calls

        entry, added = store.update("SPY", "1d", make_bars(100, step=86400))

        assert added == 0
        assert redis.set_calls == writes_after_bootstrap, "закрытые свечи не переписываются в Redis"
        assert len(entry) == 100
        # В памяти процесса сдвинулся только факт проверки.
        assert store.get("SPY", "1d").checked_at > entry.checked_at - 1  # свежая метка

    def test_checked_at_is_memory_only_until_bars_change(self):
        redis = FakeRedis()
        clock = Clock()
        store = _store(redis, clock=clock)
        store.update("SPY", "1h", make_bars(100))

        clock.advance(3600)
        _, added = store.update("SPY", "1h", make_bars(100))
        assert added == 0

        # Другой процесс читает Redis: там старая метка проверки — это честно:
        # бары не менялись, а «когда проверял» — локальный факт процесса.
        other = _store(redis, clock=clock)
        assert other.get("SPY", "1h").bars == store.get("SPY", "1h").bars


# ====================================================================== #
# 3. Два яруса
# ====================================================================== #
class TestTiers:

    def test_redis_payload_is_shared_between_instances(self):
        redis = FakeRedis()
        store_a = _store(redis)
        store_a.update("AAPL", "4h", make_bars(10, step=14400))

        store_b = _store(redis)  # «другой процесс»: пустая память, общий Redis
        entry = store_b.get("AAPL", "4h")
        assert entry is not None and len(entry) == 10
        assert entry.source == ""
        assert entry.bars[0]["o"] == 100.0

    def test_memory_tier_serves_when_redis_down(self):
        redis = FakeRedis()
        store = _store(redis)
        store.update("SPY", "1h", make_bars(50))

        redis.fail = True
        assert len(store.get("SPY", "1h")) == 50  # память продолжает отдавать

    def test_unknown_schema_is_treated_as_miss(self):
        redis = FakeRedis()
        key = bars_key("SPY", "1h", provider="yfinance")
        redis.kv[key] = b'{"v": 999, "bars": []}'
        assert _store(redis).get("SPY", "1h") is None

    def test_evict_oldest_removes_head_and_persists(self):
        redis = FakeRedis()
        store = _store(redis)
        store.update("SPY", "1h", make_bars(100))

        evicted = store.evict_oldest("SPY", "1h", count=7)
        assert evicted == 7
        assert len(store.get("SPY", "1h")) == 93
        # FIFO: голова сдвинулась на 7 баров; изменения видны другому процессу.
        other = _store(redis)
        assert len(other.get("SPY", "1h")) == 93
        assert _first_ts(other.get("SPY", "1h")) == pytest.approx(
            WED_14_MSK_TS - (100 - 8) * 3600
        )

    def test_memory_tier_bounds_number_of_series(self):
        store = _store(mem_series=8)
        for i in range(12):
            store.update(f"T{i}", "1h", make_bars(5))
        assert len(store._mem) <= 8, "память процесса не должна расти без границы"


# ====================================================================== #
# 4. Расписание проверок (когда бар «закрылся»)
# ====================================================================== #
class TestUpdateSchedule:

    def test_due_without_entry(self):
        assert _store().is_due("SPY", "1h") is True

    def test_not_due_until_next_bar_close(self):
        clock = Clock()
        store = _store(clock=clock)
        store.update("SPY", "1h", make_bars(10))

        assert store.is_due("SPY", "1h", sessions=("us",)) is False

    def test_due_after_close_plus_slack(self):
        clock = Clock()
        store = _store(clock=clock)
        store.update("SPY", "1h", make_bars(10))
        # 14:00 МСК + 1 час (закрытие 15:00 — вне сессии США!) → следующий бар — 17:30.
        next_close = next_close_after("1h", WED_14_MSK_TS, ("us",))
        clock.now = next_close + 59
        assert store.is_due("SPY", "1h", sessions=("us",)) is False
        clock.now = next_close + 61
        assert store.is_due("SPY", "1h", sessions=("us",)) is True

    def test_daily_due_only_after_session_close(self):
        clock = Clock()
        store = _store(clock=clock)
        store.update("SPY", "1d", make_bars(5, step=86400))
        # До 23:00 МСК — не проверяем.
        clock.now = datetime(2026, 9, 23, 22, 0, tzinfo=MSK).timestamp()
        assert store.is_due("SPY", "1d", sessions=("us",)) is False
        clock.now = datetime(2026, 9, 23, 23, 0, tzinfo=MSK).timestamp() + 61
        assert store.is_due("SPY", "1d", sessions=("us",)) is True

    def test_weekends_do_not_close_bars(self):
        clock = Clock()
        store = _store(clock=clock)
        # Серия актуальна до закрытия пятничной сессии: последний бар — пт 23:00 МСК.
        fri_close = datetime(2026, 9, 25, 23, 0, tzinfo=MSK)
        clock.now = fri_close.timestamp()
        store.update("SPY", "1h", make_bars(10, start_ts=fri_close.timestamp()))
        # Пятница 23:30 МСК: следующий бар — только понедельник 17:30.
        clock.now = fri_close.timestamp() + 1800
        assert store.is_due("SPY", "1h", sessions=("us",)) is False
        # Суббота: бары не закрываются — провайдер молчит.
        sat = datetime(2026, 9, 26, 12, 0, tzinfo=MSK)
        clock.now = sat.timestamp()
        assert store.is_due("SPY", "1h", sessions=("us",)) is False
        # Понедельник 17:30 + запас — можно проверять.
        mon = datetime(2026, 9, 28, 17, 30, tzinfo=MSK).timestamp() + 61
        clock.now = mon
        assert store.is_due("SPY", "1h", sessions=("us",)) is True

    def test_crypto_round_the_clock(self):
        clock = Clock()
        store = _store(clock=clock)
        store.update("BTC", "1h", make_bars(10))
        clock.now = WED_14_MSK_TS + 3600 + 61  # ровно час спустя
        assert store.is_due("BTC", "1h", sessions=None) is True
        clock.now = WED_14_MSK_TS + 3600 - 1
        assert store.is_due("BTC", "1h", sessions=None) is False

    def test_failed_check_reschedules_via_mark_checked(self):
        clock = Clock()
        store = _store(clock=clock)
        store.update("SPY", "1h", make_bars(10))
        clock.now = next_close_after("1h", WED_14_MSK_TS, ("us",)) + 61
        store.mark_checked("SPY", "1h")  # провайдер не ответил — проверку зафиксировали

        assert store.is_due("SPY", "1h", sessions=("us",)) is False
        # Следующая проверка — после закрытия следующего бара, не раньше.
        nxt = next_close_after("1h", store.get("SPY", "1h").checked_at, ("us",))
        assert nxt > clock.now


# ====================================================================== #
# 5. Фетчер: бутстрап, догрузка, производные
# ====================================================================== #
def _history_stub(frames: dict):
    """Заглушка ``_safe_history``: возвращает кадры по (interval, period/start)."""

    def fake(self, yf_ticker, interval, period=None, start=None):
        key = (interval, period, start)
        if key in frames:
            return frames[key]
        return None

    return fake


def _synth_hourly(n: int, last_ts: float) -> pd.DataFrame:
    idx = [pd.Timestamp(last_ts - (n - 1 - i) * 3600, unit="s", tz="UTC") for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1000.0 + i for i in range(n)],
        },
        index=idx,
    )


def _synth_daily(n: int, last_ts: float) -> pd.DataFrame:
    idx = [pd.Timestamp(last_ts - (n - 1 - i) * 86400, unit="s", tz="UTC") for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [200.0 + i for i in range(n)],
            "High": [201.0 + i for i in range(n)],
            "Low": [199.0 + i for i in range(n)],
            "Close": [200.5 + i for i in range(n)],
            "Volume": [2000.0 + i for i in range(n)],
        },
        index=idx,
    )


class TestFetcherIncremental:

    def test_bootstrap_fetches_each_interval_once_and_caps_fifo(self, monkeypatch):
        fetcher = TATimeframesFetcher()
        frames = {
            ("1h", "1y", None): _synth_hourly(2500, WED_14_MSK_TS),
            ("1d", "2y", None): _synth_daily(502, WED_14_MSK_TS),
        }
        monkeypatch.setattr(
            fetcher, "_safe_history", types.MethodType(_history_stub(frames), fetcher)
        )

        result = fetcher.fetch("SPY")

        assert set(result) == {"1h", "2h", "4h", "1d"}
        assert len(result["1h"]) == DEFAULT_MAX_BARS, "FIFO: не больше 500 баров"
        assert len(result["2h"]) == DEFAULT_MAX_BARS  # 2500/2 -> 1250 -> FIFO 500
        assert len(result["4h"]) == 500 <= DEFAULT_MAX_BARS
        assert len(result["1d"]) == DEFAULT_MAX_BARS
        # 2h/4h — производные часовой серии: последний 4h-бар агрегирует тот же хвост
        # (Close 4h == Close 1h на последнем часовом баре; сам бар 4h ещё не «закрылся»).
        assert result["4h"]["Close"].iloc[-1] == result["1h"]["Close"].iloc[-1]

    def test_warm_fetch_does_not_call_provider(self, monkeypatch):
        fetcher = TATimeframesFetcher()
        calls: list = []
        frames = {
            ("1h", "1y", None): _synth_hourly(2500, WED_14_MSK_TS),
            ("1d", "2y", None): _synth_daily(502, WED_14_MSK_TS),
        }

        def fake(self, yf_ticker, interval, period=None, start=None):
            calls.append((interval, period, start))
            return frames.get((interval, period, start))

        monkeypatch.setattr(fetcher, "_safe_history", types.MethodType(fake, fetcher))
        fetcher.fetch("SPY")
        assert len(calls) == 2, calls

        second = fetcher.fetch("SPY")
        assert len(calls) == 2, f"повторный вызов не должен ходить к провайдеру: {calls}"
        assert len(second["4h"]) == 500

    def test_due_fetch_pulls_only_tail(self, monkeypatch):
        clock = Clock()
        fetcher = TATimeframesFetcher()
        monkeypatch.setattr(fetcher._bars, "_clock", clock)

        calls: list = []

        def fake(self, yf_ticker, interval, period=None, start=None):
            calls.append((interval, period, start))
            if period == "1y":
                return _synth_hourly(2500, WED_14_MSK_TS)
            if period == "2y":
                return _synth_daily(502, WED_14_MSK_TS)
            # Инкремент: только хвост после последнего бара (2 новых часовых, 1 новый дневной).
            if interval == "1h":
                return _synth_hourly(50, WED_14_MSK_TS + 2 * 3600)
            return _synth_daily(10, WED_14_MSK_TS + 86400)

        monkeypatch.setattr(fetcher, "_safe_history", types.MethodType(fake, fetcher))
        fetcher.fetch("SPY")
        n_before = len(calls)

        # После закрытия следующих баров (для 1h — 17:30 МСК, для 1d — 23:00 МСК).
        clock.now = datetime(2026, 9, 23, 23, 0, tzinfo=MSK).timestamp() + 61
        fetcher.fetch("SPY")

        assert len(calls) == n_before + 2, calls[n_before:]
        # Оба новых запроса — инкрементальные (со start=), без period.
        for interval, period, start in calls[n_before:]:
            assert period is None and start is not None, (interval, period, start)

    def test_fetch_timeframe_uses_store_without_provider(self, monkeypatch):
        fetcher = TATimeframesFetcher()
        calls: list = []

        def fake(self, yf_ticker, interval, period=None, start=None):
            calls.append((interval, period, start))
            if period == "1y":
                return _synth_hourly(2500, WED_14_MSK_TS)
            return None

        monkeypatch.setattr(fetcher, "_safe_history", types.MethodType(fake, fetcher))

        df4h = fetcher.fetch_timeframe("SPY", "4h")
        assert len(df4h) == 500
        n_after_bootstrap = len(calls)

        again = fetcher.fetch_timeframe("SPY", "4h")
        assert len(calls) == n_after_bootstrap, "кэш тёплый — провайдер не опрашивается"
        assert again.index.equals(df4h.index)

    def test_provider_failure_keeps_last_good_series_and_schedules_next_check(self, monkeypatch):
        fetcher = TATimeframesFetcher()
        # Часы мокаются ДО первого fetch: бутстрап записывает ``checked_at`` этим
        # временем, а расписание проверок базируется на ``max(last_ts, checked_at)``.
        # Если бутстрап идёт на реальных часах (пятница), а потом время отматывается
        # к среде, «следующее закрытие» оказывается в будущем относительно мока —
        # провайдер не спрашивается, и ValueError не поднимается.
        clock = Clock(WED_14_MSK_TS + 3600)  # среда 15:00 МСК — сразу после первого часа
        monkeypatch.setattr(fetcher._bars, "_clock", clock)
        monkeypatch.setattr(
            fetcher, "_safe_history",
            lambda *a, **k: _synth_hourly(600, WED_14_MSK_TS) if k.get("period") else None,
        )
        first = fetcher.fetch("SPY")
        assert len(first["1h"]) == 500

        # Среда 23:05 МСК: сессия США закрылась, следующий бар «должен был закрыться»,
        # но провайдер лёг — fetch обязан честно упасть, а не отдать молча кэш.
        clock.now = datetime(2026, 9, 23, 23, 5, tzinfo=MSK).timestamp()
        monkeypatch.setattr(fetcher, "_safe_history", lambda *a, **k: None)  # провайдер лёг

        with pytest.raises(ValueError):
            fetcher.fetch("SPY")

        # Серия не потеряна, а следующая проверка отложена (mark_checked).
        entry = fetcher._bars.get("SPY", "1h")
        assert entry is not None and len(entry) == 500
        assert fetcher._bars.is_due("SPY", "1h", sessions=("us",)) is False


# ====================================================================== #
# 6. Преобразователи кадров и схема
# ====================================================================== #
class TestConverters:

    def test_frame_to_bars_roundtrip(self):
        frame = _synth_hourly(10, WED_14_MSK_TS)
        bars = frame_to_bars(frame)
        restored = bars_to_frame(bars)

        assert restored.index.equals(frame.index)
        assert restored["Close"].round(6).tolist() == frame["Close"].round(6).tolist()
        assert bars[0]["t"].endswith("+00:00")
        assert set(bars[0]) == {"t", "o", "h", "l", "c", "v"}, "ровно OHLC + объём + время"

    def test_payload_schema_version_is_declared(self):
        redis = FakeRedis()
        store = _store(redis)
        store.update("SPY", "1d", make_bars(3, step=86400))

        raw = redis.kv[bars_key("SPY", "1d", provider="yfinance")]
        import json

        payload = json.loads(raw.decode("utf-8"))
        assert payload["v"] == BAR_SCHEMA_VERSION
        assert len(payload["bars"]) == 3


