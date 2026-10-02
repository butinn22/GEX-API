"""Bybit: свечи спотового рынка (кольцо ``adapters``).

Почему появился отдельный модуль
--------------------------------
Загрузка свечей Bybit была **пять раз скопирована** — в ``ohlcv_service``, ``novel_candles``,
``signal_service``, ``trendline_service`` и ``macd_trend_service``. Копии разошлись в мелочах:
в одной ошибка транспорта превращалась в ``RuntimeError``, в другой молча в ``None``; карта
интервалов и таблица «спот-символов» дублировались четырежды. Любая правка (лимит, формат
ответа, таймаут) требовала пяти синхронных изменений — и рано или поздно одной несделанной.

Заодно выяснилось, что ``_BYBIT_SPOT_SYMBOL`` была **пустой работой**: она строилась как
``{coin: f"{coin}USDT"}`` и читалась как ``.get(coin, f"{coin}USDT")`` — то есть для любого
аргумента давала ``f"{coin}USDT"``. Символ теперь строится напрямую.

Сеть ходит только через :mod:`gex.adapters.transport.http` — здесь нет ни ``requests``,
ни собственных таймаутов.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from gex.adapters.transport.http import HttpTransport, get_shared_transport
from gex.adapters.providers.catalog import BYBIT_INTERVAL

__all__ = ["BybitError", "KLINE_URL", "INTERVAL_TO_BYBIT", "spot_symbol", "fetch_kline_records", "fetch_ohlcv"]

log = logging.getLogger(__name__)

KLINE_URL = "https://api.bybit.com/v5/market/kline"

#: Таймфреймы приложения → интервал Bybit. Источник — каталог: раньше карта жила
#: здесь, а yfinance/MOEX держали свои — и «2h/4h нативные только у Bybit» было
#: видно лишь из комментариев.
INTERVAL_TO_BYBIT = dict(BYBIT_INTERVAL)

#: Bybit ограничивает выдачу 1000 барами за запрос.
MAX_LIMIT = 1000


class BybitError(RuntimeError):
    """Сбой получения свечей Bybit.

    Наследник ``RuntimeError``: вызывающий код исторически ловил именно его, и у всех пяти
    точек вызова есть fallback на yfinance (поведение не меняется).
    """


def spot_symbol(coin: str) -> str:
    """Символ спота: ``BTC`` → ``BTCUSDT``."""
    return f"{coin}USDT"


def _rows_to_records(rows: Any) -> list[dict]:
    """Разбор ответа Bybit в записи DataFrame.

    Bybit отдаёт свечи **новыми вперёд**; каждая строка —
    ``[startTime, open, high, low, close, volume, turnover]``. Битые строки пропускаются:
    одна некорректная свеча не должна ронять всю загрузку.
    """
    records: list[dict] = []
    for row in rows:
        try:
            records.append({
                "ts": pd.Timestamp(int(row[0]), unit="ms", tz="UTC"),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
            })
        except (IndexError, ValueError, TypeError):
            continue
    return records


def fetch_kline_records(
    coin: str,
    timeframe: str,
    limit: int = MAX_LIMIT,
    *,
    transport: Optional[HttpTransport] = None,
) -> list[dict]:
    """Сырые свечи Bybit в хронологическом порядке (старые первыми).

    Пустой ответ — не ошибка: биржа честно возвращает пустой список, вызывающий сам решает,
    идти ли в fallback. Всё остальное (сеть, HTTP-статус, ``retCode != 0``, битый JSON) —
    :class:`BybitError`.
    """
    interval = INTERVAL_TO_BYBIT.get(timeframe)
    if interval is None:
        raise BybitError(f"Неподдерживаемый Bybit-интервал для tf='{timeframe}'")

    symbol = spot_symbol(coin)
    client = transport or get_shared_transport()

    response = client.get(
        KLINE_URL,
        params={
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": min(max(limit, 1), MAX_LIMIT),
        },
        expect_json=True,
    )

    payload = response.json() or {}
    ret_code = payload.get("retCode")
    if ret_code != 0:
        raise BybitError(f"Bybit kline API error for {symbol}: {payload.get('retMsg', 'unknown')}")

    rows = (payload.get("result") or {}).get("list") or []
    return _rows_to_records(rows)


def fetch_ohlcv(
    coin: str,
    timeframe: str,
    limit: int = MAX_LIMIT,
    *,
    transport: Optional[HttpTransport] = None,
) -> Optional[pd.DataFrame]:
    """Свечи в виде DataFrame с индексом-временем (колонки ``Open/High/Low/Close/Volume``).

    ``None`` — когда данных нет (пустой ответ или все строки битые), чтобы вызывающий мог
    уйти в fallback. Дубликаты по времени схлопываются с сохранением последнего значения.
    """
    records = fetch_kline_records(coin, timeframe, limit, transport=transport)
    if not records:
        return None

    frame = pd.DataFrame(records).set_index("ts").sort_index()
    return frame[~frame.index.duplicated(keep="last")]
