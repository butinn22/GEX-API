"""Breadth v3 — «Структура движения рынка» для подписчиков (GET /breadth, BASIC).

Три расчётных блока (раздельные redis-кэши, ключи зависят от версии):

1. market — «слои» движения индекса:
   - ES=F  — весь рынок (кап-взвешенный S&P 500, фьючерс);
   - RSP   — равновзвешенный рынок (Invesco S&P 500 Equal Weight);
   - giants — ЛИДЕРЫ: либо ETF MAGS (7 мега-гигантов), либо динамическая
     равновзвешенная корзина Топ-10 акций S&P 500 по капитализации
     (query-параметр ?giants=mags|top10).
   rel_rsp = RSP/ES, rel_giants = giants/ES — кумулятивные отклонения слоёв
   от всего рынка (рисуются на клиенте как % от старта окна).

2. stocks — истинная ширина по всем акциям S&P 500 (~500):
   % бумаг выше EMA20/50/200; настоящий McClellan по A/D (osc + суммация);
   raw_full — полный ряд A/D-импульса (для корреляций).

3. corr — «хитрая» корреляция к движению лидеров (скользящее окно 20 дней):
   - rsp: corr(дневное rel-движение лидеров, дневное rel-движение RSP/ES) —
     согласованность «лидеры ↔ широкая база»;
   - mcc: corr(дневное rel-движение лидеров, A/D-импульс S&P 500) —
     подтверждение движения лидеров рыночной шириной.
   Значения в [-1..1]: положительные = движение лидеров «на широкой базе»,
   падающие/отрицательные = рост/падение держится на узкой группе.

Плюс current-сигналы (5-дневные движения) и диагноз state:
ES-направление (UP/DN/FL по ±0.5% за 5д) + структура
(GIANTS/BROAD/COHER по разнице относительных 5д-движений ±1.0%).
"""
from __future__ import annotations

import io
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from gex.adapters.providers.yfinance import download as yfinance_download
from gex.adapters.providers.yfinance import history as yfinance_history
import gex.adapters.providers.yfinance as _yf_provider

from gex.adapters.cache.redis_client import RedisClient, deserialize_value
from gex.adapters.providers.catalog import universe_path

logger = logging.getLogger(__name__)

# ── Вселенная S&P 500 ────────────────────────────────────────────────────
UNIVERSE_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)
SNAPSHOT_FILE = universe_path("sp500")
# Капитализации S&P 500 (для корзины Топ-10)
CAP_URL = "https://stockanalysis.com/list/sp-500-stocks/"
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
# Парные классы акций одной компании: оставляем представителя (GOOGL)
_CLASS_MERGE: dict[str, str] = {"GOOG": "GOOGL"}
TOP10_N = 10

# ── Кэши ─────────────────────────────────────────────────────────────────
_cache = RedisClient()
_KEY_UNIVERSE = "gex:breadth:universe:v2"        # список тикеров, TTL 7 дней
_KEY_CAPS = "gex:breadth:caps:v1"                # [(ticker, cap)], TTL 7 дней
_KEY_MARKET = "gex:breadth:market:v3:{mode}"     # ES/RSP/giants + rel, TTL 10 мин
_KEY_STOCKS = "gex:breadth:stocks:v3"            # EMA-широта + McClellan (+raw_full), TTL 6 ч
_KEY_PULSE = "gex:breadth:pulse:v1"              # робастный композит 6 инструментов, TTL 10 мин
_KEY_CORR = "gex:breadth:corr:v1:{mode}"         # (устаревшие корреляции; ключ сохранён для совместимости)
_TTL_UNIVERSE = 7 * 24 * 3600
_TTL_CAPS = 7 * 24 * 3600
_TTL_MARKET = 600
_TTL_STOCKS = 6 * 3600
_TTL_PULSE = 600
_TTL_CORR = 600

# Инструменты «пульса рынка»: широкие индексы/ETF США + корзина Топ-10
_PULSE_ETFS = ["IWM", "DIA", "RSP", "QQQ", "SPY"]

# Акции: 3 года (~756 баров); первые 250 (прогрев EMA200 и McClellan) режутся
# → на графике ~2 года чистой ширины.
_STOCKS_PERIOD = "3y"
_STOCKS_WARMUP = 250
_MIN_VALID = 300  # минимум баров у тикера, чтобы попасть в расчёт

# Пороги диагноза (откалиброваны на истории 2023-04..2026-09)
_ES_5D_TH = 0.5   # ±% движения ES за 5 дней → UP/DN, иначе FL
_DIFF_TH = 1.0    # разница 5д-движений лидеров и RSP/ES → структура

_CORR_WIN = 20    # скользящее окно корреляций (торговых дней)

_lock = threading.Lock()  # защита тяжёлого пересчёта stocks


# ═════════════════════════════════════════════════════════════════════════
# Универсум и капитализации
# ═════════════════════════════════════════════════════════════════════════
def fetch_universe(force_refresh: bool = False) -> list[str]:
    """Список тикеров S&P 500: redis (7д) → GitHub → локальный снапшот."""
    if not force_refresh:
        try:
            if _cache.connected:
                cached = _cache.get(_KEY_UNIVERSE)
                if cached is not None:
                    return deserialize_value(cached)
        except Exception:
            pass

    tickers: list[str] | None = None
    try:
        r = requests.get(UNIVERSE_URL, headers=_UA, timeout=25)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        tickers = [str(t).strip().replace(".", "-") for t in df["Symbol"].astype(str)]
        tickers = [t for t in tickers if t and t != "nan"]
        try:
            SNAPSHOT_FILE.write_text("\n".join(["Тикер"] + tickers), encoding="utf-8")
        except Exception:
            pass
    except Exception as e:
        logger.warning("breadth: universe fetch failed (%s), falling back to snapshot", e)

    if not tickers or len(tickers) < 400:
        try:
            rows = SNAPSHOT_FILE.read_text(encoding="utf-8").splitlines()
            tickers = [ln.strip() for ln in rows[1:] if ln.strip() and not ln.startswith("Тикер")]
        except Exception as e:
            logger.warning("breadth: snapshot unreadable: %s", e)
            tickers = []

    if len(tickers) >= 400:
        try:
            if _cache.connected:
                _cache.set(_KEY_UNIVERSE, tickers, ex=_TTL_UNIVERSE)
        except Exception:
            pass
    return tickers


def _cap_number(s) -> float:
    """'1.23T'/'456.78B'/'9.1M' → float."""
    m = re.match(r"([\d.]+)\s*([TBM]?)", str(s).replace("$", "").strip())
    if not m:
        return 0.0
    mult = {"T": 1e12, "B": 1e9, "M": 1e6}.get(m.group(2), 1.0)
    try:
        return float(m.group(1)) * mult
    except ValueError:
        return 0.0


def _fetch_caps() -> list[tuple[str, float]]:
    """(ticker, marketCap) по S&P 500 со stockanalysis.com (HTML-таблица).

    Парные классы (GOOGL/GOOG) объединяются в одну компанию.
    """
    r = requests.get(CAP_URL, headers=_UA, timeout=25)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    df = next((t for t in tables if "Market Cap" in [str(c) for c in t.columns]), None)
    if df is None:
        raise RuntimeError("caps: таблица Market Cap не найдена")
    df = df[["Symbol", "Market Cap"]].copy()
    df["_cap"] = df["Market Cap"].apply(_cap_number)
    merged: dict[str, float] = {}
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip()
        sym = _CLASS_MERGE.get(sym, sym)  # GOOG → GOOGL (сумма классов)
        merged[sym] = merged.get(sym, 0.0) + float(row["_cap"])
    out = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    return [(sym.replace(".", "-"), cap) for sym, cap in out]


def get_caps() -> list[tuple[str, float]]:
    """Ранжированные капитализации (redis 7д → stockanalysis)."""
    try:
        if _cache.connected:
            cached = _cache.get(_KEY_CAPS)
            if cached is not None:
                return deserialize_value(cached)
    except Exception:
        pass
    caps = _fetch_caps()
    if len(caps) >= 50:
        try:
            if _cache.connected:
                _cache.set(_KEY_CAPS, caps, ex=_TTL_CAPS)
        except Exception:
            pass
    return caps


def top10_tickers() -> list[str]:
    caps = get_caps()
    return [sym for sym, _ in caps[:TOP10_N]]


# ═════════════════════════════════════════════════════════════════════════
# Загрузка котировок
# ═════════════════════════════════════════════════════════════════════════
def _download_close(tickers: list[str], period: str) -> pd.DataFrame:
    """Массовая загрузка Close-рядов (колонки = тикеры, общий индекс)."""
    # Central orchestrator batch path (when enabled): one batched external request.
    try:
        from gex.orchestrator.sync_gateway import sync_fetch_bulk_close
        close = sync_fetch_bulk_close(tickers, period=period, max_wait_ms=60000)
        if close is not None and not close.empty:
            return close.ffill()
    except Exception:
        pass

    # normalize=False: при group_by="ticker" первый уровень MultiIndex — тикер, а не поле,
    # поэтому разбор остаётся здесь (адаптер даёт один путь к сети, не диктуя форму кадра).
    data = yfinance_download(
        tickers, period=period, auto_adjust=False,
        group_by="ticker", threads=True, normalize=False,
    )
    if data is None or len(data) == 0:
        logger.warning("batch download пуст для %d тикеров", len(tickers))
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=1)
    else:
        close = data["Close"] if "Close" in data.columns else data
    close = close.ffill()
    return close


def _download_small(tickers: list[str]) -> pd.DataFrame:
    """Индивидуальная загрузка нескольких тикеров (надёжнее батча для ES=F и пр.)."""
    series: dict[str, pd.Series] = {}

    # Central orchestrator path for small symbol sets (when enabled).
    try:
        from gex.orchestrator.sync_gateway import sync_fetch_ohlcv
        for t in tickers:
            try:
                df = sync_fetch_ohlcv(t, "yfinance", "1d", limit=1000)
                if df is None or df.empty:
                    continue
                s = df["Close"].astype(float).dropna()
                s.name = t
                if len(s) > 200:
                    series[t] = s
            except Exception as e:  # noqa: BLE001
                logger.warning("breadth orchestrator: %s history failed: %s", t, e)
        if len(series) >= 2:
            frame = pd.concat(series.values(), axis=1, join="inner").dropna()
            frame.columns = list(series.keys())
            return frame
        series.clear()
    except Exception as exc:  # noqa: BLE001
        logger.warning("breadth orchestrator small-download unavailable: %s", exc)

    for t in tickers:
        try:
            d = yfinance_history(t, period="max")
            if d is None:
                continue
            s = d["Close"].astype(float).dropna()
            s.name = t
            if len(s) > 200:
                series[t] = s
        except Exception as e:  # noqa: BLE001
            logger.warning("breadth: %s history failed: %s", t, e)
    if len(series) < 2:
        raise RuntimeError("Не удалось загрузить ES/RSP/giants")
    frame = pd.concat(series.values(), axis=1, join="inner").dropna()
    frame.columns = list(series.keys())
    return frame


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _mcclellan(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """McClellan-осциллятор (EMA19−EMA39) и суммация из raw-ряда."""
    osc = (_ema(raw, 19) - _ema(raw, 39)).dropna()
    ssum = osc.cumsum()
    return osc, ssum


def _iso(dt) -> str:
    return str(pd.Timestamp(dt).date())


def _clean(series: pd.Series, nd: int = 4) -> list[float | None]:
    out: list[float | None] = []
    for v in series.tolist():
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(round(float(v), nd))
    return out


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, 4)


# ═════════════════════════════════════════════════════════════════════════
# Market: ES / RSP / giants (MAGS или Топ-10)
# ═════════════════════════════════════════════════════════════════════════
def _ew_basket_level(closes: pd.DataFrame, name: str = "basket") -> pd.Series:
    """Равновзвешенный индекс уровня: cumprod(1 + средняя дневная доходность)."""
    ret = closes.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    level = (1.0 + ret.mean(axis=1)).cumprod() * 100.0
    level.name = name
    return level


def _giants_series(mode: str, frame: pd.DataFrame) -> tuple[pd.Series, str, int]:
    """Ряд «лидеров»: MAGS-close или равновзвешенная корзина Топ-10."""
    if mode == "mags":
        if "MAGS" not in frame.columns:
            raise RuntimeError("MAGS недоступен")
        return frame["MAGS"], "MAGS", 1
    members = [t for t in top10_tickers() if t in frame.columns]
    if len(members) < 6:
        raise RuntimeError(f"Топ-10 по капитализации недоступен (найдено {len(members)})")
    return _ew_basket_level(frame[members], "top10"), "TOP10", len(members)


def compute_market(mode: str) -> dict:
    """«Слои» рынка: ES, RSP, giants + отношения к ES (общий отрезок с 2023-04)."""
    tickers = ["ES=F", "RSP"]
    if mode == "mags":
        tickers.append("MAGS")
    else:
        tickers.extend(top10_tickers())
    frame = _download_small(list(dict.fromkeys(tickers)))
    for col in ("ES=F", "RSP"):
        if col not in frame.columns:
            raise RuntimeError(f"Нет данных {col}")
    es, rsp = frame["ES=F"], frame["RSP"]
    giants, g_label, g_n = _giants_series(mode, frame)

    idx = pd.concat([es, rsp, giants], axis=1, join="inner").dropna().index
    # История «китов»: MAGS с 2023-04; корзина Топ-10 доступна с 2012 — хватает
    # ~6 лет (1500 баров), иначе ответ раздувается до ~3600 точек.
    if len(idx) > 1500:
        idx = idx[-1500:]
    rel_rsp = (rsp / es).loc[idx]
    rel_giants = (giants / es).loc[idx]

    out = {
        "giants": g_label,          # "MAGS" | "TOP10"
        "n_giants": g_n,
        "dates": [_iso(d) for d in idx],
        "es": _clean(es.loc[idx], 2),
        "rsp": _clean(rsp.loc[idx], 2),
        "giants_close": _clean(giants.loc[idx], 2),
        "rel_rsp": _clean(rel_rsp, 6),
        "rel_giants": _clean(rel_giants, 6),
    }
    try:
        if _cache.connected:
            _cache.set(_KEY_MARKET.format(mode=mode), out, ex=_TTL_MARKET)
    except Exception:
        pass
    return out


def get_market(mode: str) -> dict:
    key = _KEY_MARKET.format(mode=mode)
    try:
        if _cache.connected:
            cached = _cache.get(key)
            if cached is not None:
                return deserialize_value(cached)
    except Exception:
        pass
    return compute_market(mode)


# ═════════════════════════════════════════════════════════════════════════
# Stocks: S&P 500 широта + настоящий McClellan
# ═════════════════════════════════════════════════════════════════════════
def compute_stocks() -> dict:
    universe = fetch_universe()
    if len(universe) < 400:
        raise RuntimeError("S&P 500 универсум недоступен")

    close = _download_close(universe, period=_STOCKS_PERIOD)
    good = [c for c in close.columns if int(close[c].count()) > _MIN_VALID]
    if len(good) < 300:
        raise RuntimeError(f"Мало валидных тикеров S&P500: {len(good)}")
    c = close[good]
    # НЕ dropna(how="any"): новые члены индекса имеют короткую историю.
    valid_n = c.notna().sum(axis=1).replace(0, np.nan)

    # % бумаг выше EMA20/50/200 (от доступных в день бумаг)
    above: dict[int, pd.Series] = {}
    for span in (20, 50, 200):
        e = _ema(c, span)
        above[span] = (c > e).sum(axis=1) / valid_n * 100.0

    # настоящий McClellan по A/D
    chg = c.pct_change()
    up = (chg > 0).sum(axis=1)
    dn = (chg < 0).sum(axis=1)
    denom = (up + dn).replace(0, np.nan)
    raw = ((up - dn) / denom * 1000.0).fillna(0.0)
    osc, ssum = _mcclellan(raw)
    ssum = ssum.reindex(osc.index).dropna()

    # выходной срез: от warmup (прогрев EMA200 + стабилизация McClellan)
    idx = osc.index[_STOCKS_WARMUP:]
    dates = [_iso(d) for d in idx]
    dates_full = [_iso(d) for d in osc.index]

    out = {
        "dates": dates,
        "n": len(good),
        "above20": _clean(above[20].loc[idx], 2),
        "above50": _clean(above[50].loc[idx], 2),
        "above200": _clean(above[200].loc[idx], 2),
        "mcc_osc": _clean(osc.loc[idx], 2),
        "mcc_sum": _clean(ssum.loc[idx], 1),
        "adv": _clean(up.loc[idx], 0),
        "dec": _clean(dn.loc[idx], 0),
        # полные ряды для корреляций (raw_full стартует на ~200 баров раньше)
        "dates_full": dates_full,
        "raw_full": _clean(raw.loc[osc.index], 1),
    }
    try:
        if _cache.connected:
            _cache.set(_KEY_STOCKS, out, ex=_TTL_STOCKS)
    except Exception:
        pass
    return out


def get_stocks() -> dict:
    try:
        if _cache.connected:
            cached = _cache.get(_KEY_STOCKS)
            if cached is not None:
                return deserialize_value(cached)
    except Exception:
        pass
    with _lock:  # повторная проверка после захвата — один тяжёлый расчёт
        try:
            if _cache.connected:
                cached = _cache.get(_KEY_STOCKS)
                if cached is not None:
                    return deserialize_value(cached)
        except Exception:
            pass
        return compute_stocks()


# ═════════════════════════════════════════════════════════════════════════
# Pulse: робастный композит «усреднённого движения рынка США»
# ═════════════════════════════════════════════════════════════════════════
def compute_pulse() -> dict:
    """«Пульс рынка США»: робастное среднее движение 6 инструментов.

    Инструменты: IWM (Russell-2000), DIA (Dow-30), RSP (S&P 500 равновзв.),
    QQQ (NASDAQ-100), SPY (S&P 500) + равновзвешенная корзина Топ-10 по кап.
    Дневной шаг композита = среднее трёх робастных статистик дневных
    доходностей инструментов: геометрическое среднее, медиана и усечённое
    среднее (без одного мин/макс). Выброс одного инструмента почти не влияет.
    Накопление шагов → уровень; поверх — сглаживание EMA10.
    """
    tickers = list(_PULSE_ETFS) + top10_tickers()
    frame = _download_small(list(dict.fromkeys(tickers)))

    cols: dict[str, pd.Series] = {}
    for t in _PULSE_ETFS:
        if t in frame.columns:
            cols[t] = frame[t]
    if len(cols) < 4:
        raise RuntimeError("Мало инструментов для пульса рынка")
    members = [t for t in top10_tickers() if t in frame.columns]
    if len(members) >= 6:
        cols["TOP10"] = _ew_basket_level(frame[members], "TOP10")

    base = pd.concat(cols.values(), axis=1, join="inner").dropna()
    base.columns = list(cols.keys())
    rets = base.pct_change().replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="any")
    if len(rets) < 100:
        raise RuntimeError("Недостаточно истории для пульса рынка")

    r = rets.values.astype(float)
    k = r.shape[1]
    # геометрическое среднее дневных доходностей
    with np.errstate(all="ignore"):
        gm = np.exp(np.mean(np.log1p(np.clip(r, -0.9, 5.0)), axis=1)) - 1.0
    med = np.median(r, axis=1)
    if k >= 4:  # усечённое среднее: без одного минимума и максимума
        trm = np.sort(r, axis=1)[:, 1:-1].mean(axis=1)
    else:
        trm = med
    robust = (gm + med + trm) / 3.0

    level = pd.Series(100.0 * np.cumprod(1.0 + robust), index=rets.index)
    smooth = level.ewm(span=10, adjust=False).mean()
    if len(level) > 1500:
        level = level.iloc[-1500:]
        smooth = smooth.iloc[-1500:]

    out = {
        "dates": [_iso(d) for d in level.index],
        "level": _clean(level, 3),
        "smooth": _clean(smooth, 3),
        "n": k,
        "instruments": list(cols.keys()),
    }
    try:
        if _cache.connected:
            _cache.set(_KEY_PULSE, out, ex=_TTL_PULSE)
    except Exception:
        pass
    return out


def get_pulse() -> dict:
    try:
        if _cache.connected:
            cached = _cache.get(_KEY_PULSE)
            if cached is not None:
                return deserialize_value(cached)
    except Exception:
        pass
    return compute_pulse()


# ═════════════════════════════════════════════════════════════════════════
# Corr: (устаревшие) скользящие корреляции — ключ сохранён, данные не нужны UI
# ═════════════════════════════════════════════════════════════════════════
def compute_corr(mode: str) -> dict:
    """Скользящие корреляции дневного rel-движения лидеров:
    rsp — с rel-движением RSP/ES (база); mcc — с A/D-импульсом S&P 500 (ширина).
    """
    market = get_market(mode)  # кэш
    stocks = get_stocks()      # кэш
    md = market.get("dates") or []
    sdf = stocks.get("dates_full") or []
    if len(md) < _CORR_WIN + 40 or len(sdf) < _CORR_WIN + 40:
        raise RuntimeError("Недостаточно данных для корреляций")

    mi = pd.to_datetime(md)
    gi = pd.Series(market["rel_giants"], index=mi, dtype=float)
    rsp = pd.Series(market["rel_rsp"], index=mi, dtype=float)
    si = pd.to_datetime(sdf)
    raw = pd.Series(stocks["raw_full"], index=si, dtype=float)

    dgi = gi.pct_change()
    drsp = rsp.pct_change()
    frame = pd.concat([dgi.rename("g"), drsp.rename("r"), raw.rename("m")], axis=1, join="inner")
    if len(frame) <= _CORR_WIN:
        raise RuntimeError("Мало общих баров для корреляций")

    c_rsp = frame["g"].rolling(_CORR_WIN).corr(frame["r"])
    c_mcc = frame["g"].rolling(_CORR_WIN).corr(frame["m"])

    # отдаём только срез, совпадающий с «чистым» окном широты (stocks.dates)
    start = pd.Timestamp(stocks["dates"][0])
    mask = (frame.index >= start) & c_rsp.notna() & c_mcc.notna()
    idx = frame.index[mask]

    out = {
        "win": _CORR_WIN,
        "dates": [_iso(d) for d in idx],
        "rsp": _clean(c_rsp.loc[idx], 3),
        "mcc": _clean(c_mcc.loc[idx], 3),
    }
    try:
        if _cache.connected:
            _cache.set(_KEY_CORR.format(mode=mode), out, ex=_TTL_CORR)
    except Exception:
        pass
    return out


def get_corr(mode: str) -> dict:
    key = _KEY_CORR.format(mode=mode)
    try:
        if _cache.connected:
            cached = _cache.get(key)
            if cached is not None:
                return deserialize_value(cached)
    except Exception:
        pass
    return compute_corr(mode)


# ═════════════════════════════════════════════════════════════════════════
# Текущие сигналы + диагноз
# ═════════════════════════════════════════════════════════════════════════
def _pct_5d(series: list[float | None]) -> float | None:
    if len(series) < 6:
        return None
    a, b = series[-1], series[-6]
    if a is None or b is None or b == 0:
        return None
    return round((a / b - 1) * 100, 2)


def diagnose(es5: float | None, r5: float | None, m5: float | None) -> str:
    """Код состояния: <ES-направление><структура> (UP/DN/FL + GIANTS/BROAD/COHER)."""
    if es5 is None or r5 is None or m5 is None:
        return "FLCOHER"
    es_dir = "UP" if es5 > _ES_5D_TH else ("DN" if es5 < -_ES_5D_TH else "FL")
    diff = m5 - r5
    struct = "GIANTS" if diff > _DIFF_TH else ("BROAD" if diff < -_DIFF_TH else "COHER")
    return es_dir + struct


def get_breadth_v2(mode: str = "mags") -> dict:
    """Полный ответ GET /breadth?giants=mags|top10."""
    market = get_market(mode)
    stocks = get_stocks()
    corr = get_corr(mode)
    pulse = get_pulse()

    md = market.get("dates") or []
    sd = stocks.get("dates") or []
    if len(md) < 60 or len(sd) < 100:
        raise RuntimeError("Недостаточно данных для рыночной ширины")

    es5 = _pct_5d(market["es"])
    rsp5 = _pct_5d(market["rsp"])
    giants5 = _pct_5d(market["giants_close"])
    r5 = _pct_5d(market["rel_rsp"])
    g5 = _pct_5d(market["rel_giants"])

    def last(arr, nd: int = 2) -> float | None:
        if not arr:
            return None
        v = arr[-1]
        return None if v is None else round(float(v), nd)

    def delta(arr, days: int) -> float | None:
        if len(arr) <= days or arr[-1] is None or arr[-1 - days] is None:
            return None
        return round(float(arr[-1]) - float(arr[-1 - days]), 2)

    current = {
        "day": md[-1],
        "es_5d_pct": es5,
        "rsp_5d_pct": rsp5,
        "giants_5d_pct": giants5,
        "rel_rsp_5d_pct": r5,
        "rel_giants_5d_pct": g5,
        "above20": last(stocks["above20"]),
        "above20_1d": delta(stocks["above20"], 1),
        "above50": last(stocks["above50"]),
        "above50_1d": delta(stocks["above50"], 1),
        "above200": last(stocks["above200"]),
        "above200_1d": delta(stocks["above200"], 1),
        "mcc_osc": last(stocks["mcc_osc"]),
        "mcc_sum": last(stocks["mcc_sum"], 0),
        "mcc_sum_5d": delta(stocks["mcc_sum"], 5),
        "mcc_osc_5d": delta(stocks["mcc_osc"], 5),
        "state": diagnose(es5, r5, g5),
        "n_stocks": stocks.get("n"),
    }

    return {
        "market": market,
        "stocks": {k: v for k, v in stocks.items() if not k.endswith("_full")},
        "corr": corr,
        "pulse": pulse,
        "current": current,
    }


def ensure_warm() -> bool:
    """Фоновый прогрев (scheduler): пересчитывает только протухшие блоки."""
    try:
        for mode in ("mags", "top10"):
            get_market(mode)   # 10-мин кэш — обычно мгновенно
        get_stocks()           # 6-часовой кэш; тяжёлый расчёт только при промахе
        get_pulse()            # 10-мин кэш (IWM/DIA/RSP/QQQ/SPY + Топ-10)
        return True
    except Exception as e:
        logger.warning("breadth ensure_warm failed: %s", e)
        return False


# ── Дедлайн-защищённые выборки для страниц композита ─────────────────────
def fetch_yf_history(symbol: str, **kwargs) -> Optional[pd.DataFrame]:
    """Одна история символа через провайдера под дедлайном (High/Low/Close и т.д.).

    Общая точка доступа роутеров к yfinance-истории: прямой
    ``yf.Ticker(...).history()`` не имеет таймаута, и при деградации источника
    запрос держался бы минутами. ``providers.yfinance.history`` поверх
    ``yf_transport`` ограничивает каждый вызов дедлайном.
    """
    try:
        df = yfinance_history(symbol, **kwargs)
    except Exception as exc:  # noqa: BLE001 — недоступность источника не должна ронять страницу
        logger.warning("yfinance %s: %s", symbol, exc)
        return None
    return df if df is not None and not df.empty else None


def fetch_yf_close_series(symbols: dict[str, str], *, period: str = "1y", **kwargs) -> dict[str, pd.Series]:
    """Close-серии по нескольким yfinance-символам (VIX/VVIX/DXY и т.п.).

    Каждый вызов идёт через провайдера под дедлайном (``providers.yfinance`` поверх
    ``yf_transport``). Индекс нормализуется к naive (CBOE/остальные источники отдают
    naive — в общем кадре tz не должен мешать).
    """
    out: dict[str, pd.Series] = {}
    for yf_symbol, name in symbols.items():
        df = fetch_yf_history(yf_symbol, period=period, **kwargs)
        if df is not None and "Close" in df.columns:
            s = df["Close"].astype(float).dropna()
            s.name = name
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            out[name] = s
    return out


def fetch_spy_put_call_ratio() -> Optional[float]:
    """Put/Call OI по ближайшей экспирации SPY — под дедлайном транспорта."""
    try:
        dates = _yf_provider.option_expiries("SPY")
        if not dates:
            return None
        chain = _yf_provider.option_chain("SPY", dates[0])
        if chain is None:
            return None
        calls, puts = chain
        total_call_oi = calls["openInterest"].sum()
        total_put_oi = puts["openInterest"].sum()
        if total_call_oi > 0:
            return float(total_put_oi / total_call_oi)
        return None
    except Exception as exc:  # noqa: BLE001 — PCR необязателен для формулы
        logger.warning("PCR: %s", exc)
        return None
