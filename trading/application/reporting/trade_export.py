"""Trade export: CSV + XLSX for backtest and live trade events.

The XLSX writer is dependency-free (stdlib ``zipfile`` + a minimal
SpreadsheetML package) because ``openpyxl`` is not installed in this project;
it emits one ``trades`` worksheet with inline strings.

Live trades are reconstructed from the ``orders`` table: filled rows are
replayed per symbol in time order and classified with the same
``long_entry/add/exit`` state machine as the backtest ledger. Realized PnL is
only computable when prices are known (limit/stop orders); market-order rows
without a price export with zeroed PnL rather than a wrong number.
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from xml.sax.saxutils import escape

from trading.application.backtest.trade_log import TradeEvent, events_from_fill
from trading.domain import Fill, Position, PositionSide, Side

__all__ = [
    "TRADE_CSV_COLUMNS",
    "events_to_csv",
    "events_to_xlsx",
    "events_from_order_rows",
    "table_to_csv",
    "table_to_xlsx",
]

TRADE_CSV_COLUMNS = [
    "timestamp", "symbol", "state", "direction", "side",
    "price", "quantity", "realized_pnl", "pct_return",
    "strategy", "reason",
]


def _row(ev: TradeEvent) -> list[Any]:
    return [
        ev.timestamp.isoformat(), ev.symbol, ev.state.value, ev.direction,
        ev.side.value, ev.price, ev.quantity, ev.realized_pnl, ev.pct_return,
        ev.strategy or "", ev.reason or "",
    ]


def events_to_csv(events: Sequence[TradeEvent]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(TRADE_CSV_COLUMNS)
    for ev in events:
        writer.writerow(_row(ev))
    return buf.getvalue()


# ── minimal XLSX (SpreadsheetML) writer ──────────────────────────────

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"""

_WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{name}" sheetId="1" r:id="rId1"/></sheets></workbook>"""

_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"""


def _col(ref: int) -> str:
    """1-based column index → spreadsheet letter (1 → A, 27 → AA)."""
    letters = ""
    while ref > 0:
        ref, rem = divmod(ref - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _sheet_xml(rows: Iterable[list[Any]]) -> str:
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>',
    ]
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(row, start=1):
            ref = f"{_col(c)}{r}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        out.append(f"<row r=\"{r}\">{''.join(cells)}</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


def events_to_xlsx(events: Sequence[TradeEvent]) -> bytes:
    rows = [TRADE_CSV_COLUMNS] + [_row(ev) for ev in events]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("xl/workbook.xml", _WORKBOOK.format(name="trades"))
        z.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        z.writestr("xl/worksheets/sheet1.xml", _sheet_xml(rows))
    return buf.getvalue()


# ── live trades: classify rows from the orders table ─────────────────


def table_to_csv(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """CSV for an arbitrary table (headers + rows), machine-readable."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
    return buf.getvalue()


def table_to_xlsx(columns: Sequence[str], rows: Iterable[Sequence[Any]],
                  sheet_name: str = "Sheet1") -> bytes:
    """XLSX for an arbitrary table — same dependency-free writer as above."""
    data = [list(columns)] + [
        ["" if v is None else v for v in row] for row in rows
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("xl/workbook.xml", _WORKBOOK.format(name=sheet_name))
        z.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        z.writestr("xl/worksheets/sheet1.xml", _sheet_xml(data))
    return buf.getvalue()


def events_from_order_rows(rows: Sequence[Any]) -> list[TradeEvent]:
    """Replay filled order rows per symbol into classified trade events.

    ``rows`` are ``OrderRow``-shaped (attributes, not dicts). Rows without a
    known price are still exported (state/direction/quantity) but with zeroed
    PnL — an unknown price must not fabricate a gain or loss.
    """
    ordered = sorted(
        rows,
        key=lambda r: r.created_at.timestamp() if r.created_at else 0.0,
    )
    positions: dict[str, Position] = {}
    events: list[TradeEvent] = []
    for row in ordered:
        qty = float(row.filled_quantity or 0.0)
        if row.status != "filled" or qty <= 0:
            continue
        pos = positions.get(row.symbol, Position(row.symbol))
        known_price = row.limit_price or row.stop_price
        if known_price:
            price = float(known_price)
        elif pos.side is not PositionSide.FLAT:
            price = pos.average_entry_price  # keep avg cost stable
        else:
            price = 0.0
        fill = Fill(
            order_id=str(row.id), symbol=row.symbol, side=Side(row.side),
            price=price, quantity=qty,
            timestamp=row.created_at or datetime.now(UTC),
        )
        for ev in events_from_fill(pos, fill, strategy=row.strategy, reason=row.reason):
            if not known_price:
                ev = replace(ev, realized_pnl=0.0, pct_return=0.0)
            events.append(ev)
        positions[row.symbol] = pos.apply_fill(fill)
    return events
