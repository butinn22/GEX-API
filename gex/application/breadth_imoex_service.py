"""Breadth IMOEX v1 — «Структура движения рынка MOEX» для /breadth-imoex.

Сервис получает сырые данные через оркестратор (``data_type=imoex_breadth``,
provider ``iss``) и считает те же блоки, что и ``breadth_service`` для США,
но без «Пульса» и без yfinance/RSP:

1. market — «слои» рынка: индекс IMOEX, равновзвешенная корзина состава
   IMOEX (замена отсутствующего RSP) и равновзвешенная корзина Топ-10 по
   капитализации; отклонения корзин от IMOEX.
2. stocks — истинная широта: % бумаг выше EMA20/50/200 и McClellan A/D.
3. current — 5-дневные сигналы и диагноз (UP/DN/FL + GIANTS/BROAD/COHER).

Страница НИКОГДА не вызывает ``ensure_warm`` — только читает ``get_latest``.
Полный парсинг ISS запускается планировщиком строго в 23:00 и 08:00 МСК.
Последний удачный расчёт хранится в Redis и файл-снапшоте.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gex.adapters.cache.redis_client import RedisClient, deserialize_value

logger = logging.getLogger(__name__)

# ── Кэш и снапшот ────────────────────────────────────────────────────────
_KEY_LATEST = "gex:breadth_imoex:latest:v2"
_TTL_LATEST = 7 * 24 * 3600
_SNAPSHOT_FILE = Path(__file__).parent / "imoex_breadth_snapshot.json"
_cache = RedisClient()
_lock = threading.Lock()  # защита от параллельных ensure_warm

# ── Параметры расчёта ─────────────────────────────────────────────────────
_HISTORY_WINDOW = 1500   # максимум точек в ответе market (~6 лет)
_WARMUP = 250            # прогрев EMA200 + стабилизация McClellan
_MIN_BARS = 300          # минимум баров у бумаги для попадания в расчёт
_MIN_STOCKS = 30         # минимум бумаг в проде (частичная загрузка)
_MIN_FRACTION = 0.70     # порог доли валидных бумаг состава в проде
_MIN_LEADERS = 6         # минимум доступных лидеров для корзины Топ-10

# Волатильность: ATR(14) Wilder в % от Close + RVI (дневной).
_VOL_ATR_PERIOD = 14
_VOL_WARMUP = 20
_TRADING_DAYS = 252

# Пороги диагноза — как у США (breadth_service).
_IMOEX_5D_TH = 0.5
_DIFF_TH = 1.0

_last_failure_reason: str | None = None


def last_failure_reason() -> str | None:
    """Причина последнего неудачного ensure_warm (для 503-ответа admin-refresh)."""
    return _last_failure_reason


# ═════════════════════════════════════════════════════════════════════════
# Хелперы
# ═════════════════════════════════════════════════════════════════════════
def _num(value: Any, nd: int = 4) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, nd)


def _clean(series: pd.Series, nd: int = 4) -> list[float | None]:
    out: list[float | None] = []
    for v in series.tolist():
        if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
            out.append(None)
        else:
            out.append(round(float(v), nd))
    return out


def _iso(dt) -> str:
    return str(pd.Timestamp(dt).date())


def _bars_to_series(bars: list[dict[str, Any]]) -> pd.Series:
    """Бары оркестратора (t,c) → Series Close с DatetimeIndex (UTC)."""
    if not bars:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([b.get("t") for b in bars], errors="coerce", utc=True)
    values = [float(b.get("c")) if b.get("c") is not None else np.nan for b in bars]
    s = pd.Series(values, index=idx, dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s


def _bars_to_frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """Бары оркестратора (t,o,h,l,c,v) → OHLCV DataFrame с DatetimeIndex (UTC)."""
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    idx = pd.to_datetime([b.get("t") for b in bars], errors="coerce", utc=True)
    rows = [
        {
            "Open": b.get("o"),
            "High": b.get("h"),
            "Low": b.get("l"),
            "Close": b.get("c"),
            "Volume": b.get("v"),
        }
        for b in bars
    ]
    df = pd.DataFrame(rows, index=idx, dtype=float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def _atr_pct(frame: pd.DataFrame, period: int = _VOL_ATR_PERIOD) -> pd.Series:
    """ATR(period) Wilder в % от Close."""
    if not {"High", "Low", "Close"}.issubset(frame.columns) or len(frame) < period + 1:
        return pd.Series(dtype=float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    close = frame["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return (atr / close * 100.0).replace([np.inf, -np.inf], np.nan)


def _compute_volatility(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Средний ATR(14)% по бумагам + RVI, приведённый к дневному (RVI/√252)."""
    atr_frames: list[pd.Series] = []
    for item in raw.get("stocks") or []:
        frame = _bars_to_frame(item.get("bars") or [])
        if len(frame) < 60:
            continue
        atr = _atr_pct(frame)
        if atr.notna().sum() >= 20:
            atr_frames.append(atr)
    if not atr_frames:
        return None

    atr_matrix = pd.concat(atr_frames, axis=1)
    atr_mean = atr_matrix.mean(axis=1, skipna=True)

    rvi = _bars_to_series((raw.get("rvi") or {}).get("bars") or [])
    if not rvi.empty:
        rvi_daily_all = (rvi / np.sqrt(_TRADING_DAYS)).replace([np.inf, -np.inf], np.nan)
        idx = atr_mean.index.intersection(rvi_daily_all.index)
    else:
        rvi_daily_all = pd.Series(dtype=float)
        idx = atr_mean.index
    if len(idx) <= _VOL_WARMUP:
        return None
    idx = idx[_VOL_WARMUP:]

    atr_out = atr_mean.reindex(idx)
    if not rvi_daily_all.empty:
        rvi_daily_out = rvi_daily_all.reindex(idx)
        rvi_annual_out = rvi.reindex(idx)
        ratio = atr_out / rvi_daily_out.replace(0, np.nan)
    else:
        rvi_daily_out = pd.Series([np.nan] * len(idx), index=idx)
        rvi_annual_out = pd.Series([np.nan] * len(idx), index=idx)
        ratio = pd.Series([np.nan] * len(idx), index=idx)

    return {
        "dates": [_iso(d) for d in idx],
        "atr_pct": _clean(atr_out, 3),
        "rvi_daily": _clean(rvi_daily_out, 3),
        "rvi_annual": _clean(rvi_annual_out, 2),
        "ratio": _clean(ratio, 3),
        "n": len(atr_frames),
        "atr_period": _VOL_ATR_PERIOD,
    }


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _mcclellan(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """McClellan-осциллятор (EMA19−EMA39) и суммация из raw-ряда."""
    osc = (_ema(raw, 19) - _ema(raw, 39)).dropna()
    ssum = osc.cumsum()
    return osc, ssum


def _ew_basket_level(closes: pd.DataFrame, name: str = "basket") -> pd.Series:
    """Равновзвешенный индекс уровня: cumprod(1 + средняя дневная доходность).

    Пропуски (NaN) игнорируются в каждый день — корзина живёт на доступных
    бумагах и не обрезается самой короткой историей.
    """
    ret = closes.pct_change().replace([np.inf, -np.inf], np.nan)
    step = ret.mean(axis=1, skipna=True).fillna(0.0)
    level = (1.0 + step).cumprod() * 100.0
    level.name = name
    return level


def _pct_5d(series: list[float | None]) -> float | None:
    if len(series) < 6:
        return None
    a, b = series[-1], series[-6]
    if a is None or b is None or b == 0:
        return None
    return round((a / b - 1) * 100, 2)


def diagnose(imoex5: float | None, rel_ew5: float | None, rel_leaders5: float | None) -> str:
    """Код состояния: <IMOEX-направление><структура> (UP/DN/FL + GIANTS/BROAD/COHER)."""
    if imoex5 is None or rel_ew5 is None or rel_leaders5 is None:
        return "FLCOHER"
    direction = "UP" if imoex5 > _IMOEX_5D_TH else ("DN" if imoex5 < -_IMOEX_5D_TH else "FL")
    diff = rel_leaders5 - rel_ew5
    structure = "GIANTS" if diff > _DIFF_TH else ("BROAD" if diff < -_DIFF_TH else "COHER")
    return direction + structure


def _last(arr: list[float | None], nd: int = 2) -> float | None:
    if not arr:
        return None
    v = arr[-1]
    return None if v is None else round(float(v), nd)


def _delta(arr: list[float | None], days: int) -> float | None:
    if len(arr) <= days or arr[-1] is None or arr[-1 - days] is None:
        return None
    return round(float(arr[-1]) - float(arr[-1 - days]), 2)


# ═════════════════════════════════════════════════════════════════════════
# Расчёт
# ═════════════════════════════════════════════════════════════════════════
def compute_result(raw: dict[str, Any], *, warmup: int = _WARMUP, min_stocks: int = 0) -> dict[str, Any]:
    """Из сырого payload оркестратора собрать ответ страницы /breadth-imoex.

    ``min_stocks=0`` — чистая функция без прод-ограничений; ``ensure_warm``
    передаёт прод-пороги (30 бумаг и ≥70% состава) до вызова.
    """
    imoex = _bars_to_series((raw.get("imoex") or {}).get("bars") or [])
    if len(imoex) < 100:
        raise ValueError("IMOEX index history is too short")

    series_map: dict[str, pd.Series] = {}
    for item in raw.get("stocks") or []:
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        s = _bars_to_series(item.get("bars") or [])
        if len(s) >= max(_MIN_BARS, warmup + 50):
            series_map[symbol] = s
    if not series_map:
        raise ValueError("No valid constituent close series")
    if min_stocks and len(series_map) < min_stocks:
        raise ValueError(f"Not enough valid constituents: {len(series_map)} < {min_stocks}")

    closes = pd.concat(series_map.values(), axis=1)
    closes.columns = list(series_map.keys())
    closes = closes.sort_index()

    # ── market: IMOEX / EW-корзина / Топ-10 ───────────────────────────────
    top10 = [str(s).upper() for s in (raw.get("top10") or [])]
    members = [s for s in top10 if s in closes.columns]
    if len(members) < _MIN_LEADERS:
        raise ValueError(f"Not enough leaders: {len(members)} < {_MIN_LEADERS}")
    ew = _ew_basket_level(closes, "EW")
    leaders = _ew_basket_level(closes[members], "TOP10")

    idx = pd.concat([imoex, ew, leaders], axis=1, join="inner").dropna().index
    if len(idx) < warmup + 20:
        raise ValueError("Not enough common history for market layers")
    if len(idx) > _HISTORY_WINDOW:
        idx = idx[-_HISTORY_WINDOW:]
    rel_ew = (ew / imoex).loc[idx]
    rel_leaders = (leaders / imoex).loc[idx]

    market = {
        "benchmark": "IMOEX",
        "leaders": "TOP10",
        "n_leaders": len(members),
        "dates": [_iso(d) for d in idx],
        "imoex": _clean(imoex.loc[idx], 2),
        "ew": _clean(ew.loc[idx], 2),
        "leaders_close": _clean(leaders.loc[idx], 2),
        "rel_ew": _clean(rel_ew, 6),
        "rel_leaders": _clean(rel_leaders, 6),
    }

    # ── stocks: широта EMA + McClellan A/D ─────────────────────────────────
    valid_n = closes.notna().sum(axis=1).replace(0, np.nan)
    above: dict[int, pd.Series] = {}
    for span in (20, 50, 200):
        e = _ema(closes, span)
        above[span] = (closes > e).sum(axis=1) / valid_n * 100.0

    chg = closes.pct_change()
    up = (chg > 0).sum(axis=1)
    dn = (chg < 0).sum(axis=1)
    denom = (up + dn).replace(0, np.nan)
    raw_osc = ((up - dn) / denom * 1000.0).fillna(0.0)
    osc, ssum = _mcclellan(raw_osc)

    cut = osc.index[warmup:]
    stocks = {
        "dates": [_iso(d) for d in cut],
        "n": len(closes.columns),
        "above20": _clean(above[20].loc[cut], 2),
        "above50": _clean(above[50].loc[cut], 2),
        "above200": _clean(above[200].loc[cut], 2),
        "mcc_osc": _clean(osc.loc[cut], 2),
        "mcc_sum": _clean(ssum.loc[cut], 1),
        "adv": _clean(up.loc[cut], 0),
        "dec": _clean(dn.loc[cut], 0),
    }

    # ── current: 5-дневные сигналы + диагноз ───────────────────────────────
    imoex5 = _pct_5d(market["imoex"])
    ew5 = _pct_5d(market["ew"])
    leaders5 = _pct_5d(market["leaders_close"])
    rel_ew5 = _pct_5d(market["rel_ew"])
    rel_leaders5 = _pct_5d(market["rel_leaders"])

    current = {
        "day": market["dates"][-1],
        "imoex_5d_pct": imoex5,
        "ew_5d_pct": ew5,
        "leaders_5d_pct": leaders5,
        "rel_ew_5d_pct": rel_ew5,
        "rel_leaders_5d_pct": rel_leaders5,
        "above20": _last(stocks["above20"]),
        "above20_1d": _delta(stocks["above20"], 1),
        "above50": _last(stocks["above50"]),
        "above50_1d": _delta(stocks["above50"], 1),
        "above200": _last(stocks["above200"]),
        "above200_1d": _delta(stocks["above200"], 1),
        "mcc_osc": _last(stocks["mcc_osc"]),
        "mcc_sum": _last(stocks["mcc_sum"], 0),
        "mcc_sum_5d": _delta(stocks["mcc_sum"], 5),
        "mcc_osc_5d": _delta(stocks["mcc_osc"], 5),
        "state": diagnose(imoex5, rel_ew5, rel_leaders5),
        "n_stocks": stocks["n"],
    }

    volatility = _compute_volatility(raw)

    return {
        "market": market,
        "stocks": stocks,
        "current": current,
        "volatility": volatility,
        "meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_completed_day": market["dates"][-1],
            "source": "iss",
        },
    }


# ═════════════════════════════════════════════════════════════════════════
# Хранение последнего удачного расчёта
# ═════════════════════════════════════════════════════════════════════════
def store_latest(payload: dict[str, Any]) -> bool:
    """Сохранить последний удачный расчёт в Redis + файл-снапшот."""
    ok = True
    try:
        if _cache.connected:
            _cache.set(_KEY_LATEST, payload, ex=_TTL_LATEST)
    except Exception:
        logger.warning("breadth_imoex: redis write failed", exc_info=True)
        ok = False
    try:
        _SNAPSHOT_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("breadth_imoex: snapshot write failed", exc_info=True)
        ok = False
    return ok


def get_latest() -> dict[str, Any] | None:
    """Последний удачный расчёт: Redis → файл-снапшот → None."""
    try:
        if _cache.connected:
            cached = _cache.get(_KEY_LATEST)
            if cached is not None:
                return deserialize_value(cached)
    except Exception:
        pass
    try:
        if _SNAPSHOT_FILE.exists():
            return json.loads(_SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("breadth_imoex: snapshot unreadable: %s", exc)
    return None


def ensure_warm(force_refresh: bool = True, *, bypass_rate_limit: bool = False) -> bool:
    """Плановый запуск (23:00/08:00 МСК) или admin-refresh: ISS через оркестратор.

    GET-обработчик страницы никогда не вызывает эту функцию. Плановые запуски
    идут под суточной квотой; admin-refresh передаёт ``bypass_rate_limit=True``.
    """
    global _last_failure_reason
    with _lock:
        try:
            from gex.orchestrator.sync_gateway import sync_fetch_imoex_breadth

            raw = sync_fetch_imoex_breadth(
                force_refresh=force_refresh,
                bypass_rate_limit=bypass_rate_limit,
            )
        except Exception as exc:
            logger.warning("breadth_imoex: orchestrator fetch failed: %s", exc)
            _last_failure_reason = f"оркестратор/ISS недоступны: {exc}"
            return False
        if not raw:
            logger.warning("breadth_imoex: orchestrator returned no data")
            _last_failure_reason = "оркестратор вернул пустые данные (ISS недоступен или пустой ответ)"
            return False

        n_universe = len(raw.get("universe") or [])
        n_stocks = len(raw.get("stocks") or [])
        # ≥70% состава и ≥30 бумаг, когда состав это позволяет; для маленького
        # fallback-универсума (10 голубых фишек) требуем все 10 доступных.
        required = max(int(np.ceil(n_universe * _MIN_FRACTION)), min(_MIN_STOCKS, n_universe))
        if n_stocks < required:
            logger.warning(
                "breadth_imoex: partial load too small (%d/%d stocks, need %d) — keeping previous data",
                n_stocks, n_universe, required,
            )
            _last_failure_reason = f"недостаточно данных: получено {n_stocks} бумаг из {n_universe} (нужно {required})"
            return False

        try:
            payload = compute_result(raw, min_stocks=required)
        except Exception as exc:
            logger.warning("breadth_imoex: compute failed: %s", exc)
            _last_failure_reason = f"ошибка расчёта: {exc}"
            return False
        ok = store_latest(payload)
        if ok:
            _last_failure_reason = None
        else:
            _last_failure_reason = "не удалось сохранить результат (Redis и файл-снапшот)"
        return ok
