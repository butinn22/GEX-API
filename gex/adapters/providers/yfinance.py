"""yfinance за портом — единственная точка обращения к SDK Yahoo (ring: adapters/providers).

Зачем
-----
Приложение звало yfinance напрямую в **11 файлах**: ``yf.Ticker(sym).history(...)``,
``fast_info``, ``options``, ``option_chain`` — и каждый раз с собственным ``try/except``,
своей проверкой на пустой ответ и своим приведением колонок. Пять копий одного и того же
кода означают пять мест, где поведение может разъехаться (например, где-то
``auto_adjust=True`` по умолчанию, где-то нет).

Здесь эти операции собраны один раз, поверх дедлайн-обёртки
:mod:`gex.adapters.transport.yf_transport` (она уже защищает от «источник висит вечно»).

Договорённости
--------------
* **Сбой источника — это ``None``, а не исключение.** Yahoo отвечает и ошибками, и пустыми
  кадрами, и зависаниями; вызывающий код в 11 местах всё равно ловил исключение, чтобы
  вернуть None. Здесь это сделано один раз, а причина остаётся в логе.
* **``None`` и «пустой кадр» различаются на уровне домена.** Пустой кадр — это «данных нет»
  (праздник, делистинг), и вызывающий обычно идёт в fallback; исключение — «источник сломался».
* **Колонки нормализуются** (``Close``/``Volume`` и т.д.): у Yahoo они могут приходить
  в разном регистре, а ``MultiIndex`` приходит из ``download``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from gex.adapters.transport.yf_transport import YfDeadlineError, get_shared_yf_transport

logger = logging.getLogger(__name__)

__all__ = [
    "YFinanceError",
    "close_series",
    "company_info",
    "closes_for",
    "download",
    "fast_info",
    "history",
    "history_or_raise",
    "normalize_columns",
    "option_chain",
    "option_expiries",
    "spot",
]


class YFinanceError(RuntimeError):
    """Ошибка уровня провайдера, когда вызывающему нужен именно сбой, а не ``None``."""


def _transport():
    """Общая дедлайн-обёртка: одна на процесс (создаёт пул и следит за временем)."""
    return get_shared_yf_transport()


# ====================================================================== #
#  Нормализация
# ====================================================================== #
#: Официальные имена колонок Yahoo → как их ждёт остальной код.
_FIELD_ALIASES: dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
}


def normalize_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Привести колонки Yahoo к ``Open/High/Low/Close/Volume``.

    * ``MultiIndex`` (приходит из ``download``) схлопывается по **первому** уровню —
      это уровень поля, а второй это тикер: ``('Close', 'SPY')`` → ``Close``.
      Взять последний уровень (тикер) значит получить колонки ``SPY`` вместо ``Close``,
      а при нескольких тикерах — ещё и одинаковые имена, то есть молча испорченные данные;
      если поля повторяются (несколько тикеров), имена склеиваются ``Close_SPY``;
    * известные поля переименовываются в канонический регистр, **остальные не трогаются**:
      ``capitalize()`` превратил бы ``Adj Close`` в ``Adj close``, а тикер ``SPY`` в ``Spy``.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        fields = [str(c[0]) for c in out.columns]
        if len(set(fields)) == len(fields):
            out.columns = fields
        else:
            out.columns = [f"{c[0]}_{c[1]}" for c in out.columns]
    out = out.rename(
        columns={c: _FIELD_ALIASES[str(c).strip().lower()] for c in out.columns
                 if str(c).strip().lower() in _FIELD_ALIASES}
    )
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except (TypeError, ValueError) as exc:
            # Индекс остаётся как есть: для части вызовов он и не нужен, но причина
            # должна быть видна (молчание здесь скрыло бы смену формата у провайдера).
            logger.debug("yfinance: индекс не приводится к датам (%s)", exc)
    return out.sort_index()


def _clean(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Нормализовать и вернуть ``None``, если данных нет.

    Пустой ответ Yahoo (``df.empty``) — это «данных нет», а не ошибка: вызывающий чаще
    всего идёт в fallback, и отличать это от сбоя ему не нужно. Различие сохраняется
    в логе: сбой пишется warning, пустой ответ — debug.
    """
    if df is None:
        return None
    normalized = normalize_columns(df)
    if len(normalized) == 0:
        logger.debug("yfinance не вернул строк данных")
        return None
    return normalized


# ====================================================================== #
#  Свечи
# ====================================================================== #
def history(
    symbol: str,
    *,
    period: str = "2y",
    interval: str = "1d",
    auto_adjust: bool = False,
    seconds: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Свечи Yahoo (``None`` при сбое или отсутствии данных).

    Parameters
    ----------
    symbol : str
        Тикер Yahoo (``SPY``, ``BTC-USD``, ``GC=F``).
    period : str
        Глубина: ``1d``, ``2y``, ``max``, ``5y``.
    interval : str
        Интервал Yahoo: ``1h``, ``1d``, ``1mo``.
    auto_adjust : bool
        Учитывать ли дивиденды/сплиты. По умолчанию ``False`` — как в существующем коде:
        смена этого флага задним числом изменила бы исторические уровни.
    """
    try:
        df = _transport().history(
            symbol, interval=interval, period=period, auto_adjust=auto_adjust, seconds=seconds
        )
    except YfDeadlineError as exc:
        # Отдельно: источник не ответил за отведённое время. Это деградация провайдера,
        # и она обязана быть видна (иначе «данных нет» и «источник повис» неразличимы).
        logger.warning("yfinance history(%s, %s) превысил дедлайн: %s", symbol, interval, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — yfinance бросает разные типы
        logger.warning("yfinance history(%s, %s) упал: %s", symbol, interval, exc)
        return None
    return _clean(df)


def history_or_raise(symbol: str, **kwargs: Any) -> pd.DataFrame:
    """Свечи или :class:`YFinanceError` — для путей, где пустой ответ обязан быть ошибкой."""
    df = history(symbol, **kwargs)
    if df is None:
        raise YFinanceError(f"yfinance не вернул свечи для {symbol}")
    return df


def download(
    symbols: str | Sequence[str],
    *,
    period: str = "2y",
    interval: str = "1d",
    normalize: bool = True,
    seconds: Optional[float] = None,
    **kwargs: Any,
) -> Optional[pd.DataFrame]:
    """Пакетная загрузка (``yf.download``).

    ``normalize=True`` (по умолчанию) приводит ``MultiIndex`` к полям
    (``('Close','SPY')`` → ``Close``) и регистр колонок: без этого ``df["Close"]`` вернул бы
    не Series, и дальше поехала бы вся арифметика.

    ``normalize=False`` — для вызывающих, которые разбирают ``MultiIndex`` сами
    (например, при ``group_by="ticker"`` первый уровень это тикер, а не поле, и
    «привести к полям» означало бы испортить кадр). Прочие ``kwargs`` уходят в yfinance
    как есть: адаптер не должен запрещать параметры провайдера, которых не знает.
    """
    try:
        df = _transport().download(
            symbols, period=period, interval=interval, progress=False, seconds=seconds, **kwargs
        )
    except Exception as exc:  # noqa: BLE001 — включая YfDeadlineError (наследник RuntimeError)
        logger.warning("yfinance download(%s) упал: %s", symbols, exc)
        return None
    return _clean(df) if normalize else df


# ====================================================================== #
#  Цена и опционы
# ====================================================================== #
def fast_info(symbol: str, *, seconds: Optional[float] = None) -> dict:
    """Быстрый снимок Yahoo как обычный dict (``{}`` при сбое).

    ``fast_info`` — объект SDK с разными именами полей в версиях, поэтому доступ идёт
    через :func:`spot`, а не напрямую из вызывающего кода.
    """
    try:
        info = _transport().fast_info(symbol, seconds=seconds)
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance fast_info(%s) недоступен: %s", symbol, exc)
        return {}
    try:
        return dict(info)
    except (TypeError, ValueError):
        # Не отображение — читаем известные поля по одному.
        out: dict = {}
        for key in ("last_price", "lastPrice", "regular_market_price", "previous_close"):
            value = getattr(info, key, None)
            if value is not None:
                out[key] = value
        return out


def company_info(symbol: str, *, seconds: Optional[float] = None) -> dict:
    """Расширенные поля эмитента (``Ticker.info``) как обычный dict; ``{}`` при сбое.

    Не то же, что :func:`fast_info`: ``info`` — отдельный, самый медленный запрос к тому же
    API (десятки полей), и он часто недоступен. Ошибка не должна ронять вызывающего: цена и
    валюта уже пришли из ``fast_info``, а сектор и название — необязательное дополнение.

    Функция появилась потому, что ``gex/orchestrator/adapters/yfinance_adapter.py`` обращался
    к ``yf.Ticker(...).info`` напрямую, минуя транспорт: это обходило и дедлайн, и правило
    «выход в сеть — только из ``gex/adapters``» (гейт egress это и поймал).
    """
    try:
        info = _transport().info(symbol, seconds=seconds)
    except Exception as exc:  # noqa: BLE001 — info необязателен
        logger.debug("yfinance info(%s) недоступен: %s", symbol, exc)
        return {}
    try:
        return dict(info)
    except (TypeError, ValueError):
        return {}


def spot(symbol: str, *, interval: str = "1d", period: str = "1d") -> Optional[float]:
    """Последняя цена: ``fast_info``, затем последний ``Close`` из истории.

    Два источника, потому что у индексов ``fast_info`` часто пуст, а у делистингованных
    бумаг пуста история. Порядок важен: ``fast_info`` дешевле и не тянет свечи.
    """
    info = fast_info(symbol)
    for key in ("lastPrice", "last_price", "regularMarketPrice", "regular_market_price", "previousClose"):
        value = info.get(key)
        if value:
            try:
                price = float(value)
            except (TypeError, ValueError) as exc:
                logger.debug("yfinance spot(%s): %r не число (%s)", symbol, value, exc)
                continue
            if price > 0:
                return price

    df = history(symbol, period=period, interval=interval)
    if df is not None and "Close" in df.columns:
        closes = df["Close"].dropna()
        if len(closes) > 0:
            try:
                price = float(closes.iloc[-1])
            except (TypeError, ValueError):
                return None
            return price if price > 0 else None
    return None


def option_expiries(symbol: str, *, seconds: Optional[float] = None) -> tuple[str, ...]:
    """Список дат экспирации (пустой кортеж, если опционов нет или источник недоступен)."""
    try:
        expiries = _transport().options(symbol, seconds=seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance options(%s) упал: %s", symbol, exc)
        return ()
    return tuple(str(e) for e in (expiries or ()))


def option_chain(
    symbol: str, expiry: str, *, seconds: Optional[float] = None
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """Цепочка на дату: ``(calls, puts)`` или ``None``.

    Возвращаются два кадра, а не объект SDK: вызывающему нужны колонки (``openInterest``,
    ``impliedVolatility``), а не детали конкретной версии yfinance.
    """
    try:
        chain = _transport().option_chain(symbol, expiry, seconds=seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance option_chain(%s, %s) упал: %s", symbol, expiry, exc)
        return None
    calls = getattr(chain, "calls", None)
    puts = getattr(chain, "puts", None)
    if calls is None or puts is None:
        logger.debug("yfinance option_chain(%s, %s) без calls/puts", symbol, expiry)
        return None
    return calls, puts


def close_series(symbol: str, *, period: str = "2y", interval: str = "1d") -> Optional[pd.Series]:
    """Ряд закрытий (частый случай: нужна только колонка ``Close``)."""
    df = history(symbol, period=period, interval=interval)
    if df is None or "Close" not in df.columns:
        return None
    series = df["Close"].astype(float).dropna()
    return series if len(series) > 0 else None


def closes_for(symbols: Iterable[str], *, period: str = "2y", interval: str = "1d") -> dict[str, pd.Series]:
    """Ряды закрытий по списку тикеров (тикеры без данных пропускаются)."""
    out: dict[str, pd.Series] = {}
    for symbol in symbols:
        series = close_series(symbol, period=period, interval=interval)
        if series is not None:
            out[symbol] = series
    return out
