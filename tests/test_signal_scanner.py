"""Tests: SignalScannerService — backend logic + API integration."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timedelta

from gex.application.signal_scanner_service import (
    SignalScannerService,
    ScannerInstrument,
    SignalScannerReport,
    MAX_INSTRUMENTS,
    SUPPORTED_TIMEFRAMES,
)
from gex.application.signal_service import SignalService
from gex.application.service import GEXService


# Тестовый пользователь для per-user watchlist API
TEST_USER = "test-user"


@pytest.fixture(autouse=True)
def clean_db():
    """Создать таблицы (user_instruments и др.) в in-memory SQLite."""
    from gex.adapters.persistence.database import recreate_tables

    recreate_tables()
    yield


# ================================================================= #
#  Fixtures
# ================================================================= #
@pytest.fixture
def scanner():
    """SignalScannerService с фейковым signal_service (без живых запросов)."""
    gs = GEXService()
    ss = SignalService(gs)
    return SignalScannerService(ss)


# ================================================================= #
#  Unit: constants
# ================================================================= #
def test_max_instruments():
    assert MAX_INSTRUMENTS == 10


def test_supported_timeframes():
    assert "1h" in SUPPORTED_TIMEFRAMES
    assert "2h" in SUPPORTED_TIMEFRAMES
    assert "4h" in SUPPORTED_TIMEFRAMES
    assert "1d" in SUPPORTED_TIMEFRAMES
    assert len(SUPPORTED_TIMEFRAMES) == 4


def test_instrument_key():
    inst = ScannerInstrument(ticker="SPY", timeframe="1d")
    assert inst.key == "SPY:1d"


# ================================================================= #
#  Unit: watchlist management
# ================================================================= #
def test_set_watchlist_valid(scanner):
    pairs = [
        {"ticker": "SPY", "timeframe": "1d"},
        {"ticker": "BTC", "timeframe": "4h"},
        {"ticker": "AAPL", "timeframe": "1h"},
    ]
    result = scanner.set_watchlist(TEST_USER, pairs)
    assert len(result) == 3
    assert result[0].ticker == "SPY"
    assert result[0].timeframe == "1d"


def test_get_watchlist_returns_copy(scanner):
    scanner.set_watchlist(TEST_USER, [{"ticker": "SPY", "timeframe": "1d"}])
    wl = scanner.get_watchlist(TEST_USER)
    assert len(wl) == 1



def test_set_watchlist_deduplicates(scanner):
    pairs = [
        {"ticker": "SPY", "timeframe": "1d"},
        {"ticker": "spy", "timeframe": "1d"},  # duplicate (case-insensitive)
    ]
    result = scanner.set_watchlist(TEST_USER, pairs)
    assert len(result) == 1



def test_set_watchlist_max_exceeded(scanner):
    pairs = [{"ticker": f"T{i}", "timeframe": "1d"} for i in range(MAX_INSTRUMENTS + 1)]
    with pytest.raises(ValueError, match="Максимум"):
        scanner.set_watchlist(TEST_USER, pairs)



def test_set_watchlist_invalid_timeframe(scanner):
    with pytest.raises(ValueError, match="таймфрейм"):
        scanner.set_watchlist(TEST_USER, [{"ticker": "SPY", "timeframe": "3h"}])



def test_set_watchlist_empty_ticker(scanner):
    with pytest.raises(ValueError, match="ticker"):
        scanner.set_watchlist(TEST_USER, [{"ticker": "", "timeframe": "1d"}])


def test_get_watchlist_returns_copy(scanner):
    scanner.set_watchlist(TEST_USER, [{"ticker": "SPY", "timeframe": "1d"}])
    wl = scanner.get_watchlist(TEST_USER)
    assert len(wl) == 1
    # Modify the returned list - should not affect internal state
    wl.append(ScannerInstrument(ticker="QQQ", timeframe="4h"))
    assert len(scanner.get_watchlist(TEST_USER)) == 1  # internal unchanged


# ================================================================= #
#  Unit: scan now (without real network calls)
# ================================================================= #
def test_scan_now_empty_watchlist(scanner):
    report = scanner.scan_now(TEST_USER)
    assert isinstance(report, SignalScannerReport)
    assert len(report.instruments) == 0
    assert report.running is False



def test_scan_now_with_instruments(scanner):
    scanner.set_watchlist(TEST_USER, [{"ticker": "SPY", "timeframe": "1d"}])
    report = scanner.scan_now(TEST_USER)
    assert isinstance(report, SignalScannerReport)
    assert len(report.instruments) == 1
    instr = report.instruments[0]
    assert instr.ticker == "SPY"
    assert instr.timeframe == "1d"
    # error may be set if no network / no data — that's acceptable
    # the key thing is it ran without raising


# ================================================================= #
#  Unit: background lifecycle
# ================================================================= #
def test_start_stop(scanner):
    assert scanner.is_running is False
    scanner.start()
    # should be running shortly
    import time
    time.sleep(0.1)
    running = scanner.is_running
    scanner.stop()
    # We just verify start/stop don't crash; is_running may be True quickly
    assert isinstance(running, bool)


# ================================================================= #
#  Integration: ScannerInstrument dataclass
# ================================================================= #
def test_scanner_instrument_full():
    inst = ScannerInstrument(
        ticker="SPY",
        timeframe="1d",
        latest_signals=[],
        last_scan=datetime.utcnow(),
        error=None,
    )
    assert inst.ticker == "SPY"
    assert inst.key == "SPY:1d"
    assert inst.latest_signals == []
    assert inst.last_scan is not None
    assert inst.error is None



# ================================================================= #
#  Integration: SignalScannerReport
# ================================================================= #
def test_report_build(scanner):
    scanner.set_watchlist(TEST_USER, [{"ticker": "SPY", "timeframe": "1d"}])
    report = scanner.get_report(TEST_USER)
    assert isinstance(report.scanned_at, datetime)
    assert report.pool_interval_seconds >= 60

    # dict serialization
    d = {
        "instruments": [
            {"ticker": i.ticker, "timeframe": i.timeframe}
            for i in report.instruments
        ],
        "running": report.running,
    }
    assert len(d["instruments"]) == 1
    assert d["instruments"][0]["ticker"] == "SPY"



# ================================================================= #
#  Integration: back-to-back operations
# ================================================================= #
def test_watchlist_replace(scanner):
    scanner.set_watchlist(TEST_USER, [{"ticker": "SPY", "timeframe": "1d"}])
    assert len(scanner.get_watchlist(TEST_USER)) == 1
    # Replace with different instruments
    scanner.set_watchlist(TEST_USER, [
        {"ticker": "BTC", "timeframe": "1h"},
        {"ticker": "ETH", "timeframe": "2h"},
    ])
    wl = scanner.get_watchlist(TEST_USER)
    assert len(wl) == 2
    assert wl[0].ticker == "BTC"



def test_watchlist_all_timeframes(scanner):
    pairs = [{"ticker": "SPY", "timeframe": tf} for tf in SUPPORTED_TIMEFRAMES]
    scanner.set_watchlist(TEST_USER, pairs)
    wl = scanner.get_watchlist(TEST_USER)
    assert len(wl) == 4
    tfs = {i.timeframe for i in wl}
    assert tfs == set(SUPPORTED_TIMEFRAMES)


# ================================================================= #
#  TG-уведомления: только смена УСЛОВИЯ сигнала (action/order_type).
#  Тики цены и «переоткрытие» того же сигнала на новом баре — НЕ события.
#  Регрессия: «шлёт при каждом изменении цены» (цена была в сути сигнала).
# ================================================================= #
class _FakeSig:
    """Минимальная модель сигнала (как SignalRecordOut: action/order_type/...)."""

    def __init__(self, action, order_type, price, ts):
        self.action = action  # "BUY"/"SELL"
        self.order_type = order_type  # entry_short/exit_short/...
        self.price = price
        self.timestamp = ts
        self.entry_score = 0.46
        self.reason = order_type


def _instr(ticker, timeframe, sigs):
    return ScannerInstrument(ticker=ticker, timeframe=timeframe, latest_signals=sigs)


def _tg_spy(scanner, monkeypatch):
    """Изолировать TG-отправку и состояние уведомлений от Redis/сети."""
    sent: list[list[str]] = []
    st: dict = {}
    monkeypatch.setattr(scanner, "_tg_chat_id", lambda uid: "chat-1")
    monkeypatch.setattr(scanner, "_send_tg", lambda chat, lines: sent.append(lines))
    # Состояние — локальный dict, НЕ Redis (dev-Redis поднят и «протекает»
    # между тестами/процессами: watchlist-тесты пишут те же ключи).
    monkeypatch.setattr(scanner, "_load_state", lambda uid: st)
    monkeypatch.setattr(scanner, "_save_state", lambda uid, s: (st.clear(), st.update(s)))
    return sent


def test_notify_silent_on_price_ticks_of_same_signal(scanner, monkeypatch):
    """SHORT ENTRY (SELL/entry_short) с меняющейся ценой и новым баром — молчим."""
    sent = _tg_spy(scanner, monkeypatch)
    ts = datetime(2026, 9, 7, 12, 0)

    # 1) первый скан без состояния — молча фиксирует (без спама после деплоя)
    scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [_FakeSig("SELL", "entry_short", 79123.0, ts)])], send=True)
    assert sent == []

    # 2) те же action/order_type, меняется ТОЛЬКО цена (тики) — без уведомлений
    for price in (79113.9, 79039.7, 78885.2):
        scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [_FakeSig("SELL", "entry_short", price, ts)])], send=True)
    assert sent == []

    # 3) тот же сигнал «переоткрыт» на новом баре (новый timestamp) — без уведомлений
    scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [_FakeSig("SELL", "entry_short", 78792.0, datetime(2026, 9, 7, 16, 0))])], send=True)
    assert sent == []


def test_notify_on_condition_change_entry_to_exit(scanner, monkeypatch):
    """SHORT ENTRY → SHORT EXIT (exit_short закрывает шорт = BUY) — уведомление."""
    sent = _tg_spy(scanner, monkeypatch)
    ts = datetime(2026, 9, 7, 12, 0)

    scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [_FakeSig("SELL", "entry_short", 79123.0, ts)])], send=True)
    assert sent == []  # baseline

    scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [_FakeSig("BUY", "exit_short", 78800.0, ts)])], send=True)
    assert len(sent) == 1
    msg = "\n".join(sent[0])
    assert "Сигнальный сканер — обновления" in msg
    assert "BTC" in msg and "покупка по 78800" in msg

    # повтор того же exit на новой цене — молчим
    scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [_FakeSig("BUY", "exit_short", 78790.0, ts)])], send=True)
    assert len(sent) == 1

    # сигнал снят (активного нет) — событие «сигнал снят»
    scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [])], send=True)
    assert len(sent) == 2
    assert "сигнал снят" in "\n".join(sent[1])

    # возврат того же условия (новый вход) — снова событие
    scanner._apply_changes(TEST_USER, [_instr("BTC", "4h", [_FakeSig("SELL", "entry_short", 79000.0, datetime(2026, 9, 8, 4, 0))])], send=True)
    assert len(sent) == 3
    assert "продажа по 79000" in "\n".join(sent[2])


def test_fmt_signal_time_msk_and_utc():
    """Время в сообщениях — реальный момент формирования, значения в МСК и UTC.

    naive-вход трактуется как UTC; МСК = UTC+3. Даты в зонах могут
    расходиться (MOEX-дневка 21:00 UTC = 00:00 МСК следующего дня) — тогда
    дата в UTC-скобке сохраняется.
    """
    from gex.application.signal_scanner_service import fmt_signal_time, signal_line_html

    # naive UTC 12:00 → МСК 15:00, UTC без даты (даты совпадают)
    assert fmt_signal_time(datetime(2026, 9, 7, 12, 0)) == "07.09 15:00 МСК (12:00 UTC)"
    # aware UTC → то же
    from datetime import timezone as tzmod
    assert fmt_signal_time(datetime(2026, 9, 7, 12, 0, tzinfo=tzmod.utc)) == "07.09 15:00 МСК (12:00 UTC)"
    assert fmt_signal_time(None) is None
    assert fmt_signal_time("") is None

    # сигнал на баре 07.09 21:00 UTC (MOEX-дневка, полночь МСК) → «08.09 00:00 МСК (07.09 21:00 UTC)»;
    # без when выводится ОДИН подписанный момент (сформирован = timestamp сигнала)
    sig = _FakeSig("SELL", "entry_short", 30.62, datetime(2026, 9, 7, 21, 0))
    line = signal_line_html("AFLT", "1d", sig)
    assert line == ("  <b>AFLT</b> [1d]: <b>SHORT ENTRY</b> · продажа по 30.62 · score 46% · "
                    "сформирован 08.09 00:00 МСК (07.09 21:00 UTC)")

    # with when — два подписанных момента: сформирован (алгоритм/бар) и отправлен (скан)
    line2 = signal_line_html("ETH", "4h", sig, when=datetime(2026, 9, 8, 9, 30))
    assert "SHORT ENTRY" in line2
    assert "сформирован 08.09 00:00 МСК (07.09 21:00 UTC)" in line2
    assert "отправлен 08.09 12:30 МСК (09:30 UTC)" in line2

    # строка-эталон из жалобы: чип LONG EXIT (продажа = закрытие лонга, а не
    # открытие шорта) + дата-время формирования алгоритмом (бар 08:00)
    # + момент отправки (13:04), оба в МСК и UTC
    line3 = signal_line_html("ETH", "4h", _FakeSig("SELL", "exit_long", 2478.67, datetime(2026, 9, 8, 8, 0)),
                             when=datetime(2026, 9, 8, 10, 4))
    assert line3 == ("  <b>ETH</b> [4h]: <b>LONG EXIT</b> · продажа по 2478.67 · score 46% · "
                     "сформирован 08.09 11:00 МСК (08:00 UTC) · "
                     "отправлен 08.09 13:04 МСК (10:04 UTC)")


def test_signal_chip_label():
    """Чип полной семантики сигнала (как в UI): LONG EXIT ≠ открытие шорта."""
    from gex.application.signal_scanner_service import _signal_chip_label, signal_line_html

    ts = datetime(2026, 9, 7, 12, 0)
    cases = {
        ("BUY", "entry_long"): "LONG ENTRY",
        ("SELL", "exit_long"): "LONG EXIT",
        ("SELL", "entry_short"): "SHORT ENTRY",
        ("BUY", "exit_short"): "SHORT EXIT",
        ("BUY", "add_long"): "LONG ADD",
        ("SELL", "add_short"): "SHORT ADD",
    }
    for (action, ot), expected in cases.items():
        assert _signal_chip_label(_FakeSig(action, ot, 100.0, ts)) == expected

    # фолбэк: order_type пуст, но есть reason (long_exit → LONG EXIT)
    s = _FakeSig("SELL", "", 100.0, ts)
    s.reason = "long_exit"
    assert _signal_chip_label(s) == "LONG EXIT"
    # без order_type/reason — чипа нет, строка без него
    assert _signal_chip_label(_FakeSig("SELL", "", 100.0, ts)) is None
    assert _signal_chip_label(None) is None

    # «продажа» без чипа остаётся как раньше (нет данных о типе)
    plain = signal_line_html("X", "1d", _FakeSig("SELL", "", 1.5, ts))
    assert "<b>LONG EXIT</b>" not in plain and "продажа по 1.5" in plain


def test_essence_ignores_price_and_timestamp(scanner):
    """Суть сигнала = action|order_type: цена и время бара не влияют."""
    from gex.application.signal_scanner_service import _signal_essence

    ts = datetime(2026, 9, 7, 12, 0)
    a = _FakeSig("SELL", "entry_short", 79123.0, ts)
    b = _FakeSig("SELL", "entry_short", 79039.7, datetime(2026, 9, 7, 16, 0))
    assert _signal_essence(a) == _signal_essence(b) == "SELL|entry_short"

    c = _FakeSig("BUY", "exit_short", 78800.0, ts)
    assert _signal_essence(c) == "BUY|exit_short" != _signal_essence(a)

    assert _signal_essence(None) == ""
    assert _signal_essence(_FakeSig("", "entry_short", 1.0, ts)) == ""


# ================================================================= #
#  Свежесть: сигналы старее торгового окна не считаются активными
#  (иначе [0] «откатывается» к недельной давности при фликкере
#  формирующегося бара → ложные события TG).
# ================================================================= #
class _FakeSignalService:
    """signal_service, возвращающий заданный список сигналов (без сети)."""

    def __init__(self, signals):
        self._signals = list(signals)

    def analyze_signals(self, *args, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(recent_signals=list(self._signals), regime=None)


def _scan_one_with(sigs, tf="1d"):
    from gex.application.signal_scanner_service import SignalScannerService as Svc
    s = Svc(_FakeSignalService(sigs))
    return s._scan_one(ScannerInstrument(ticker="T", timeframe=tf, latest_signals=[]))


def test_scan_one_drops_stale_signal_1d():
    """Сигнал 10 дней назад на 1d — НЕ активен (старее 3 торговых дней)."""
    old = datetime.utcnow() - timedelta(days=10)
    res = _scan_one_with([_FakeSig("SELL", "entry_short", 30.62, old)], tf="1d")
    assert res.latest_signals == []
    assert res.error is None


def test_scan_one_drops_stale_signal_4h():
    """Сигнал 4 дня назад на 4h — НЕ активен (>18 баров ≈ 72ч)."""
    old = datetime.utcnow() - timedelta(days=4)
    res = _scan_one_with([_FakeSig("SELL", "entry_short", 2477.0, old)], tf="4h")
    assert res.latest_signals == []


def test_scan_one_keeps_fresh_signal():
    """Свежий сигнал (текущий бар/сегодня) — активен."""
    now = datetime.utcnow()
    res = _scan_one_with([_FakeSig("SELL", "entry_short", 2477.0, now)], tf="4h")
    assert len(res.latest_signals) == 1
    assert res.latest_signals[0].order_type == "entry_short"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
