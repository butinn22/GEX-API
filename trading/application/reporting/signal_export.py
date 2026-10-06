"""Signal + position exports (CSV / XLSX) for the live signal engine.

The exporters take the persisted rows (``KeySignalRow`` / ``SignalPositionRow``)
and render them through the dependency-free table writers in
:mod:`trading.application.reporting.trade_export` (no ``openpyxl`` in this
project), so a downloaded workbook opens cleanly in Excel/LibreOffice and
parses with pandas.

Nothing is invented: an open position exports with an empty realised PnL
column rather than a zero, so a human reading the sheet can tell "no result
yet" from "break-even".
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from trading.adapters.persistence.models import KeySignalRow, SignalPositionRow
from trading.application.reporting.trade_export import table_to_csv, table_to_xlsx

__all__ = [
    "SIGNAL_COLUMNS",
    "POSITION_COLUMNS",
    "signal_rows",
    "position_rows",
    "signals_to_csv",
    "signals_to_xlsx",
    "positions_to_csv",
    "positions_to_xlsx",
]

#: Every field a live signal carries — the trade plan, not just a direction.
SIGNAL_COLUMNS: tuple[str, ...] = (
    "signal_id", "emitted_at", "bar_time", "symbol", "venue", "side", "state",
    "reason", "strength", "price", "entry_price", "stop_loss", "take_profit",
    "position_size", "risk_pct", "risk_amount", "timeframe", "strategy",
    "strategy_version", "preset", "source",
)

#: Every field a position carries — lifecycle, management levels and PnL.
POSITION_COLUMNS: tuple[str, ...] = (
    "position_id", "symbol", "side", "status", "strategy", "strategy_version",
    "preset", "timeframe", "source", "entry_time", "entry_price", "quantity",
    "initial_stop", "stop_price", "take_profit", "trail_price", "best_price",
    "worst_price", "mfe_r", "bars_held", "exit_time", "exit_price", "exit_reason",
    "risk_amount", "risk_pct", "gross_pnl", "net_pnl", "pnl_r", "pct_return",
    "unrealised_pnl", "mark_price", "created_at", "updated_at",
)


def _iso(value: Any) -> str:
    return value.isoformat() if value is not None else ""


def _num(value: Any) -> Any:
    return "" if value is None else value


def signal_rows(rows: Iterable[KeySignalRow], *, venue_by_symbol: Any = None) -> list[list[Any]]:
    """Render signal rows as a table (one inner list per signal)."""
    if venue_by_symbol is None:
        venue_by_symbol = venue_of
    out: list[list[Any]] = []
    for r in rows:
        venue = str(venue_by_symbol(r.symbol) or "")
        out.append([
            r.id, _iso(r.timestamp), _iso(r.bar_time), r.symbol, venue, r.side, r.state,
            r.reason, r.strength, r.price, _num(r.entry_price), _num(r.stop_loss),
            _num(r.take_profit), _num(r.position_size), _num(r.risk_pct),
            _num(r.risk_amount), r.timeframe or "", r.strategy, r.strategy_version,
            _preset_of(r), r.source,
        ])
    return out


def _preset_of(row: Any) -> str:
    """Preset name, recovered from the stored indicator snapshot."""
    import json

    try:
        data = json.loads(row.indicators_json or "{}")
    except (TypeError, ValueError):
        return ""
    return str(data.get("preset") or "")


def venue_of(symbol: str) -> str:
    """Venue that lists ``symbol`` (local instrument-universe lookup, no I/O).

    Derived rather than stored so no migration is needed for a column that is a
    pure function of the ticker.
    """
    try:
        from trading.application.instruments import resolve_symbol

        return str(resolve_symbol(symbol).get("exchange") or "")
    except Exception:  # never let an export fail on a lookup
        return ""


def position_rows(rows: Iterable[SignalPositionRow]) -> list[list[Any]]:
    """Render position rows as a table (one inner list per position)."""
    return [[
        r.id, r.symbol, r.side, r.status, r.strategy, r.strategy_version, r.preset,
        r.timeframe, r.source, _iso(r.entry_time), _num(r.entry_price), _num(r.quantity),
        _num(r.initial_stop), _num(r.stop_price), _num(r.take_profit), _num(r.trail_price),
        _num(r.best_price), _num(r.worst_price), _num(r.mfe_r), r.bars_held,
        _iso(r.exit_time), _num(r.exit_price), r.exit_reason, _num(r.risk_amount),
        _num(r.risk_pct), _num(r.gross_pnl), _num(r.net_pnl), _num(r.pnl_r),
        _num(r.pct_return), _num(r.unrealised_pnl), _num(r.mark_price),
        _iso(r.created_at), _iso(r.updated_at),
    ] for r in rows]


def signals_to_csv(rows: Sequence[KeySignalRow]) -> str:
    return table_to_csv(SIGNAL_COLUMNS, signal_rows(rows))


def signals_to_xlsx(rows: Sequence[KeySignalRow]) -> bytes:
    return table_to_xlsx(SIGNAL_COLUMNS, signal_rows(rows), sheet_name="signals")


def positions_to_csv(rows: Sequence[SignalPositionRow]) -> str:
    return table_to_csv(POSITION_COLUMNS, position_rows(rows))


def positions_to_xlsx(rows: Sequence[SignalPositionRow]) -> bytes:
    return table_to_xlsx(POSITION_COLUMNS, position_rows(rows), sheet_name="positions")
