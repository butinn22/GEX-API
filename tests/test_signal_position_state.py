# -*- coding: utf-8 -*-
"""Регрессии машины позиции: сигналы только при открытой позиции.

Покрывает баг автосканера «LONG EXIT сразу после SHORT EXIT без входов»:
выходы/добавления без открытой позиции не генерируются вовсе; сигналы идут
очерёдностью вход → (добавления) → выход; сканеры получают текущую позицию.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd  # noqa: F401  (используется в аннотации _frame)
import pytest

from gex.application.signal_service import SignalService, position_to_dict

_START = datetime(2026, 9, 1)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Кадр features: колонки сигналов (bool) + close; индекс — часовые бары."""
    base = {
        "long_entry_signal": False,
        "short_entry_signal": False,
        "long_exit_signal": False,
        "short_exit_signal": False,
        "combined_long_add": False,
        "combined_short_add": False,
    }
    data = []
    for r in rows:
        row = dict(base)
        row.update(r)
        data.append(row)
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex([_START + timedelta(hours=i) for i in range(len(df))])
    return df


def _rows(n: int, close: float = 100.0) -> list[dict]:
    return [{"close": close} for _ in range(n)]


# ── 1. Главная регрессия: выходы без входа не генерируются ──────────────


def test_exit_without_entry_never_emitted():
    """Кресты exit туда-сюда без входов — раньше давали «LONG EXIT/SHORT EXIT».

    Это в точности баг со страницы: выход идёт сразу после выхода, входа нет.
    """
    rows = _rows(24)
    for i in range(len(rows)):
        rows[i]["short_exit_signal"] = i % 2 == 0
        rows[i]["long_exit_signal"] = i % 2 == 1
        rows[i]["combined_long_add"] = True

    f = _frame(rows)
    events, pos = SignalService._simulate_position_events(f)

    assert events == []
    assert pos == {"side": "flat", "avg_price": None, "since_idx": None}


def test_exit_before_any_entry_suppressed_but_later_sequence_kept():
    """Выход до входа игнорируется; после реального входа — фиксируется."""
    rows = _rows(12)
    rows[0]["long_exit_signal"] = True   # выход без позиции — мусор
    rows[3]["long_entry_signal"] = True  # вход
    rows[3 + 1]["long_entry_signal"] = True  # повторный вход при позиции — не сигнал
    rows[7]["long_exit_signal"] = True   # легальный выход

    events, pos = SignalService._simulate_position_events(_frame(rows))

    assert events == [
        (3, "entry_long", "long_entry"),
        (7, "exit_long", "long_exit"),
    ]
    assert pos["side"] == "flat"


# ── 2. Очерёдность: повторный вход при позиции игнорируется ────────────


def test_no_duplicate_entry_while_in_position_and_side_switch():
    """Пока LONG открыт, short_entry не входит; смена стороны — только после выхода."""
    rows = _rows(12)
    for i in range(1, 6):
        rows[i]["long_entry_signal"] = True
    rows[3]["short_entry_signal"] = True   # при открытом лонге — игнор
    rows[7]["long_exit_signal"] = True     # закрыли лонг
    rows[8]["short_entry_signal"] = True   # теперь можно в шорт

    events, pos = SignalService._simulate_position_events(_frame(rows))

    assert events == [
        (1, "entry_long", "long_entry"),
        (7, "exit_long", "long_exit"),
        (8, "entry_short", "short_entry"),
    ]
    assert pos["side"] == "short"
    assert pos["since_idx"] == 8


# ── 3. Добавления: только при позиции, кулдаун 10 баров, средняя цена ──


def test_adds_require_position_cooldown_and_avg_price():
    rows = _rows(14)
    rows[0]["combined_long_add"] = True          # без позиции — не сигнал
    rows[1]["long_entry_signal"] = True          # вход @100
    for i in range(2, 14):
        rows[i]["combined_long_add"] = True      # спам добавлений
        rows[i]["close"] = 100.0 + (i - 1) * 10  # 2→110, 12→210 …

    events, pos = SignalService._simulate_position_events(_frame(rows))

    add_idx = [i for i, ot, _ in events if ot == "add_long"]
    assert add_idx == [2, 12], "кулдаун 10 баров: 2 → 12"
    assert pos["side"] == "long"
    # 1.0 @100 + 0.1 @110 + 0.1 @210 = 132 / 1.2 = 110.0
    assert pos["avg_price"] == pytest.approx(110.0)


def test_add_priority_over_exit_like_evaluate():
    """Одновременный add+exit: как в evaluate() — сначала добавление."""
    rows = _rows(6)
    rows[1]["long_entry_signal"] = True
    rows[3]["combined_long_add"] = True
    rows[3]["long_exit_signal"] = True

    events, _ = SignalService._simulate_position_events(_frame(rows))

    assert events == [
        (1, "entry_long", "long_entry"),
        (3, "add_long", "long_add"),
    ]


# ── 4. Извлечение записей: порядок, позиция, схема ──────────────────────


def _svc() -> SignalService:
    return SignalService(gex_service=None, ta_fetcher=object())


def test_extract_returns_records_newest_first_and_position():
    rows = _rows(30)
    rows[3]["close"] = 101.5
    rows[3]["long_entry_signal"] = True
    f = _frame(rows)

    records, pos = _svc()._extract_recent_signals(f, f, None, 5)

    assert [r.order_type for r in records] == ["entry_long"]
    assert records[0].price == 101.5
    assert pos["side"] == "long"
    assert pos["since_idx"] == 3

    schema = SignalService._position_to_schema(pos, f)
    assert schema.side == "long"
    assert schema.avg_price == 101.5
    assert schema.since == f.index[3].to_pydatetime()


def test_extract_limits_to_n_recent_newest_first():
    rows = _rows(40)
    rows[2]["long_entry_signal"] = True
    rows[10]["long_exit_signal"] = True
    rows[20]["short_entry_signal"] = True
    rows[30]["short_exit_signal"] = True

    records, pos = _svc()._extract_recent_signals(_frame(rows), _frame(rows), None, 2)

    assert [r.order_type for r in records] == ["exit_short", "entry_short"]
    assert pos["side"] == "flat"


def test_extract_never_starts_with_orphan_exit():
    """Любая последовательность записей, прочитанная хронологически, валидна."""
    rows = _rows(60)
    for i in range(0, 60, 4):
        rows[i]["short_exit_signal"] = True
    for i in range(2, 60, 4):
        rows[i]["long_exit_signal"] = True
    rows[7]["long_entry_signal"] = True
    rows[44]["short_entry_signal"] = True

    f = _frame(rows)
    records, pos = _svc()._extract_recent_signals(f, f, None, 50)

    side = None
    for r in reversed(records):  # хронологически
        if r.order_type == "entry_long":
            assert side is None
            side = "long"
        elif r.order_type == "entry_short":
            assert side is None
            side = "short"
        elif r.order_type == "exit_long":
            assert side == "long"
            side = None
        elif r.order_type == "exit_short":
            assert side == "short"
            side = None
        elif r.order_type in ("add_long", "add_short"):
            assert side is not None
    assert side is None  # хвост закрыт: позиций нет
    assert pos["side"] == "flat"


# ── 5. position_to_dict (сканеры): устойчивость к mock-объектам ─────────


def _ns(ot: str, h: int):
    return SimpleNamespace(order_type=ot, timestamp=_START + timedelta(hours=h))


def test_with_entry_context_adds_preceding_entry():
    """Окно свежести обрезало вход — якорь дотягивает его (пропуская добавления)."""
    from gex.application.signal_service import with_entry_context

    full = [
        _ns("exit_long", 10), _ns("add_long", 8), _ns("entry_long", 6),
        _ns("exit_short", 4), _ns("entry_short", 3),
    ]
    shown = full[:4]  # обрезали entry_short из-за окна свежести
    out = with_entry_context(full, shown)

    assert out == [*shown, full[4]]
    assert out[-1].order_type == "entry_short"


def test_with_entry_context_noop_when_oldest_is_entry():
    from gex.application.signal_service import with_entry_context

    full = [_ns("exit_long", 10), _ns("entry_long", 6)]
    shown = full[:2]
    assert with_entry_context(full, shown) == shown

    # Добавление при позиции: «add без входа» тоже не показываем — якорь добирает вход.
    full2 = [_ns("add_long", 10), _ns("entry_long", 6)]
    shown2 = full2[:1]
    assert with_entry_context(full2, shown2) == [*shown2, full2[1]]


def test_with_entry_context_noop_when_no_entry_available():
    from gex.application.signal_service import with_entry_context

    full = [_ns("exit_long", 10), _ns("exit_short", 4)]
    shown = full[:1]
    assert with_entry_context(full, shown) == shown

    assert with_entry_context([], []) == []


def test_drop_unanchored_signals_removes_orphans_after_block_filter():
    """Фильтр скрыл вход — осиротевший выход тоже не показываем."""
    from gex.routers._helpers import drop_unanchored_signals

    def s(ot: str) -> dict:
        return {"order_type": ot}

    # GOLD-кейс: блокированные входы отсечены, остались два выхода — уходят оба.
    assert drop_unanchored_signals([s("exit_short"), s("exit_long")]) == []

    # Если видимый вход есть — последовательность сохраняется целиком.
    full = [s("entry_short"), s("exit_short"), s("entry_long"), s("exit_long")]
    assert drop_unanchored_signals(full) == full

    # «add» без видимой позиции — убирается, при позиции — остаётся.
    assert drop_unanchored_signals([s("add_long"), s("exit_long")]) == []
    assert drop_unanchored_signals([s("entry_long"), s("add_long"), s("exit_long")]) == [
        s("entry_long"), s("add_long"), s("exit_long"),
    ]

    # Неизвестный тип строки не трогаем (обратная совместимость).
    assert drop_unanchored_signals([s("hold")]) == [s("hold")]
    assert drop_unanchored_signals([]) == []


# ── 7. Трейлинг-стоп и принудительный разворот ─────────────────────────


def test_trailing_stop_moves_long_exit():
    """Трейлинг 2%: выход в точке отката от пика (low ≤ пик×0.98)."""
    rows = _rows(7)
    rows[1].update(close=100.0, high=100.0, low=99.0, long_entry_signal=True)
    # Лоу держатся выше трейла (пик − 2%): 105→102.9, 112→109.8, 120→117.6
    for i, (h, l, c) in enumerate([(105, 103, 104), (112, 110, 111), (120, 118, 119)], start=2):
        rows[i].update(high=h, low=l, close=c)
    rows[5].update(high=119.0, low=117.0, close=118.0)  # стоп 120×0.98=117.6 → пробит
    rows[6].update(high=121.0, low=118.0, close=120.0)

    events, pos = SignalService._simulate_position_events(_frame(rows), trailing_pct=2.0)

    assert events == [
        (1, "entry_long", "long_entry"),
        (5, "exit_long", "trailing_stop"),
    ]
    assert pos["side"] == "flat"


def test_trailing_beats_column_exit_same_bar():
    rows = _rows(6)
    rows[1].update(close=100.0, high=100.0, low=99.0, long_entry_signal=True)
    rows[2].update(close=119.0, high=120.0, low=118.0)  # пик 120, лоу выше стопа 117.6
    rows[3].update(close=115.0, high=118.0, low=115.0, long_exit_signal=True)  # стоп пробит

    events, _ = SignalService._simulate_position_events(_frame(rows), trailing_pct=2.0)

    assert [(i, ot, r) for i, ot, r in events][-1] == (3, "exit_long", "trailing_stop")


def test_trailing_off_by_default_and_reason_kept():
    rows = _rows(7)
    rows[1].update(close=100.0, high=100.0, low=99.0, long_entry_signal=True)
    rows[5].update(close=118.0, high=119.0, low=110.0)
    rows[6].update(close=120.0, high=121.0, low=119.0, long_exit_signal=True)

    events, _ = SignalService._simulate_position_events(_frame(rows))

    assert [(i, ot) for i, ot, _ in events] == [(1, "entry_long"), (6, "exit_long")]
    assert events[-1][2] == "long_exit"


def test_reverse_signal_closes_and_flips():
    """Свежий противоположный вход: принудительное закрытие + разворот."""
    rows = _rows(6)
    rows[1].update(close=100.0, long_entry_signal=True)
    rows[4].update(close=98.0, short_entry_signal=True)

    events, pos = SignalService._simulate_position_events(_frame(rows), reverse=True)

    assert events == [
        (1, "entry_long", "long_entry"),
        (4, "exit_long", "reverse_signal"),
        (4, "entry_short", "short_entry"),
    ]
    assert pos["side"] == "short"
    assert pos["since_idx"] == 4
    assert pos["avg_price"] == 98.0


def test_reverse_requires_fresh_edge_not_condition():
    """Постоянно истинное условие противоположного входа разворот не дёргает."""
    rows = _rows(6)
    rows[0].update(close=100.0, short_entry_signal=True)
    for i in range(1, 6):
        rows[i].update(short_entry_signal=True)          # условие шорта не гаснет
    rows[2].update(close=101.0, long_entry_signal=True)  # свежий край лонга

    events, pos = SignalService._simulate_position_events(_frame(rows), reverse=True)

    assert events == [
        (0, "entry_short", "short_entry"),
        (2, "exit_short", "reverse_signal"),
        (2, "entry_long", "long_entry"),
    ]
    assert pos["side"] == "long"


def test_extract_with_trailing_flags_reason():
    rows = _rows(12)
    rows[1].update(close=100.0, high=100.0, low=99.0, long_entry_signal=True)
    rows[2].update(close=103.0, high=103.0, low=100.0)
    rows[3].update(close=97.0, high=103.0, low=97.0)     # −5.8% от пика 103

    f = _frame(rows)
    records, pos = _svc()._extract_recent_signals(f, f, None, 5, trailing_pct=2.0)

    assert [r.order_type for r in records] == ["exit_long", "entry_long"]
    assert records[0].reason == "trailing_stop"
    assert pos["side"] == "flat"


def test_snapshot_roundtrip_events_equal():
    """Машина из снапшота даёт те же события/позицию, что из полного кадра."""
    import numpy as np

    n = 40
    idx = pd.DatetimeIndex([_START + timedelta(hours=i) for i in range(n)])
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame(
        {"close": close, "high": close + 1.0, "low": close - 1.0}, index=idx,
    )
    for col in ("long_entry_signal", "short_entry_signal", "long_exit_signal",
                "short_exit_signal", "combined_long_add", "combined_short_add",
                "long_entry_a", "long_entry_b", "short_entry_a", "short_entry_b",
                "can_enter_long", "can_enter_short"):
        df[col] = False
    df.iloc[3, df.columns.get_loc("long_entry_signal")] = True
    df.iloc[12, df.columns.get_loc("long_exit_signal")] = True
    df.iloc[20, df.columns.get_loc("short_entry_signal")] = True
    for col in ("atr", "novelsrc", "ema10", "ema200", "my_vwap_state",
                "my_vwap_state_1", "my_vwap_state_5", "adline", "adl50",
                "adl_macd", "tp_f", "trend_coefficient"):
        df[col] = 1.0

    snap = SignalService._build_snapshot(df)
    df2 = pd.DataFrame(snap["arrays"], index=pd.DatetimeIndex(snap["index"]))

    ev1, pos1 = SignalService._simulate_position_events(df, trailing_pct=2.0, reverse=True)
    ev2, pos2 = SignalService._simulate_position_events(df2, trailing_pct=2.0, reverse=True)

    assert ev1 == ev2
    assert pos1 == pos2
    assert len(ev1) >= 3  # вход @3, выход @12, шорт @20 — как минимум


def test_position_to_dict_variants():
    assert position_to_dict(MagicMock()) is None  # mock без валидного side

    dt = datetime(2026, 9, 18, 8, 0)
    obj = SimpleNamespace(
        position=SimpleNamespace(side="short", avg_price=1301.3456789, since=dt)
    )
    assert position_to_dict(obj) == {
        "side": "short",
        "avg_price": 1301.345679,
        "since": dt.isoformat(),
    }

    dict_obj = SimpleNamespace(
        position={"side": "flat", "avg_price": None, "since": "2026-09-19T12:00:00"}
    )
    assert position_to_dict(dict_obj) == {
        "side": "flat",
        "avg_price": None,
        "since": "2026-09-19T12:00:00",
    }

    assert position_to_dict(SimpleNamespace(position=None)) is None
    assert position_to_dict(SimpleNamespace(position={"side": "weird"})) is None
