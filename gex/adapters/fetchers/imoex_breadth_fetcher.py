"""Raw data fetching for /breadth-imoex from MOEX ISS.

The module provides two layers:

* pure parsers (``parse_universe_payload``, ``parse_candles_block``,
  ``drop_forming_session``, ``top10_secids``) that are unit-testable without
  network access;
* :class:`ImoexBreadthFetcher` — the network-facing fetcher used by the
  orchestrator's ISS adapter (``data_type=imoex_breadth``).

ISS endpoints used:

* IMOEX composition / capitalization:
  ``/iss/statistics/engines/stock/markets/index/analytics/IMOEX.json``
  (block ``analytics``, fallback block ``tickers``);
* IMOEX index daily candles:
  ``/iss/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json``
  (interval=24, reverse pagination);
* constituent daily candles: reused :class:`gex.moex_candles_fetcher.MOEXCandlesFetcher`
  (engine=stock, market=shares, board=TQBR).

Safety rules implemented here:

* every direct HTTP call goes through ``get_rate_limiter().wait("moex_iss")``;
* only ordinary shares are kept (preferred shares are filtered);
* bars from the current still-forming MOEX trading session are dropped.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from gex.adapters.ratelimit.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════
# Константы ISS
# ═════════════════════════════════════════════════════════════════════════
_MSK = timezone(timedelta(hours=3))  # Москва: фиксированный UTC+3, без DST
_UA = {"User-Agent": "gex-app/1.0"}

_ISS_ANALYTICS_URL = (
    "https://iss.moex.com/iss/statistics/engines/stock/markets/index"
    "/analytics/IMOEX.json"
)
_ISS_INDEX_CANDLES_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX"
    "/securities/{secid}/candles.json"
)
_ISS_TQBR_CANDLES_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR"
    "/securities/{secid}/candles.json"
)

_ISS_PAGE_SIZE = 500
_ISS_MAX_PAGES = 4          # 2 года дневок = ~500 баров, хватает 2 страниц
_HISTORY_DAYS = 730         # глубина истории для breadth
_MIN_BARS = 250             # минимум баров у бумаги (прогрев EMA200)
_FORMING_SESSION_CUTOFF = 19  # до 19:00 МСК дневная свеча текущего дня неполная

# Жёсткий fallback-универсум (голубые фишки), если ISS analytics недоступен.
FALLBACK_UNIVERSE: tuple[dict[str, Any], ...] = (
    {"secid": "SBER", "weight": 0.0, "capitalization": 0.0},
    {"secid": "LKOH", "weight": 0.0, "capitalization": 0.0},
    {"secid": "GAZP", "weight": 0.0, "capitalization": 0.0},
    {"secid": "ROSN", "weight": 0.0, "capitalization": 0.0},
    {"secid": "GMKN", "weight": 0.0, "capitalization": 0.0},
    {"secid": "NVTK", "weight": 0.0, "capitalization": 0.0},
    {"secid": "TATN", "weight": 0.0, "capitalization": 0.0},
    {"secid": "PLZL", "weight": 0.0, "capitalization": 0.0},
    {"secid": "YDEX", "weight": 0.0, "capitalization": 0.0},
    {"secid": "MOEX", "weight": 0.0, "capitalization": 0.0},
)

# Известные префы (страховка, если ISS-короткое имя неоднозначно).
_PREF_SECIDS = {
    "SBERP", "SNGSP", "TATNP", "TRNFP", "KZOSP", "KAZTP", "NKNCP", "RTKMP",
    "BANEP", "KROTP", "MSNGP", "RBCMP", "UPROV",
}


# ═════════════════════════════════════════════════════════════════════════
# Чистые парсеры (unit-test friendly)
# ═════════════════════════════════════════════════════════════════════════
def _num(value: Any) -> float:
    """Число из ISS-поля; 0.0 для пустых/невалидных значений."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.0
        return f if f == f and abs(f) != float("inf") else 0.0
    try:
        return float(str(value).strip().replace(" ", "").replace("$", ""))
    except (TypeError, ValueError):
        return 0.0


def _is_preferred(secid: str, shortname: str | None) -> bool:
    """True для привилегированных акций MOEX (SBERP/SNGSP/...)."""
    sec = (secid or "").strip().upper()
    if sec in _PREF_SECIDS:
        return True
    name = (shortname or "").strip().lower()
    if name:
        # «Сбербанк-п», «Татнефть 3 ап», «Транснефть ап» и т.п.
        if re.search(r"(?:^|[\s\-])(?:ап|пр)(?:\.|$|\s)", name) or name.endswith("-п"):
            return True
        return False
    # Без имени: суффикс P — признак префа на MOEX (исключений среди
    # ликвидных имён ИМОЕКС не выявлено).
    return sec.endswith("P")


def parse_universe_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Состав IMOEX из ISS analytics/tickers блока.

    Возвращает ``[{"secid", "weight", "capitalization"}]`` без префов.
    При пустом/невалидном payload — жёсткий fallback-список.
    """
    block = payload.get("analytics") or payload.get("tickers")
    if not isinstance(block, dict):
        return [dict(u) for u in FALLBACK_UNIVERSE]

    columns = [str(c).strip().lower() for c in (block.get("columns") or [])]
    rows = block.get("data") or []
    if not columns or not rows:
        return [dict(u) for u in FALLBACK_UNIVERSE]

    def col_index(*names: str) -> int | None:
        for name in names:
            if name in columns:
                return columns.index(name)
        return None

    i_secid = col_index("secid", "secids")
    if i_secid is None:
        i_secid = col_index("ticker")  # в некоторых ответах только ticker
    i_short = col_index("shortnames", "shortname", "secname")
    i_weight = col_index("weight")
    i_cap = col_index("capitalization", "marketcap", "issuercapitalization")

    out: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        secid = str(row[i_secid] if i_secid is not None and i_secid < len(row) else "").strip().upper()
        if not secid or secid == "NONE":
            continue
        shortname = None
        if i_short is not None and i_short < len(row):
            shortname = str(row[i_short]).strip()
        if _is_preferred(secid, shortname):
            continue
        weight = _num(row[i_weight]) if i_weight is not None and i_weight < len(row) else 0.0
        cap = _num(row[i_cap]) if i_cap is not None and i_cap < len(row) else 0.0
        out.append({"secid": secid, "weight": weight, "capitalization": cap})

    if not out:
        return [dict(u) for u in FALLBACK_UNIVERSE]
    return out


def top10_secids(universe: list[dict[str, Any]] | None, n: int = 10) -> list[str]:
    """Топ-N по капитализации (затем по весу); fallback на голубые фишки.

    Если ISS не отдал ни капитализацию, ни веса — используем фиксированный
    порядок голубых фишек, а не алфавитный.
    """
    if not universe:
        return [u["secid"] for u in FALLBACK_UNIVERSE][:n]
    has_cap = any(float(u.get("capitalization") or 0.0) > 0 for u in universe)
    has_weight = any(float(u.get("weight") or 0.0) > 0 for u in universe)
    if not has_cap and not has_weight:
        available = {u["secid"] for u in universe if u.get("secid")}
        ordered = [u["secid"] for u in FALLBACK_UNIVERSE if u["secid"] in available]
        ordered += [u["secid"] for u in universe if u.get("secid") and u["secid"] not in ordered]
        return ordered[:n]
    ordered = sorted(
        universe,
        key=lambda u: (-float(u.get("capitalization") or 0.0), -float(u.get("weight") or 0.0), u["secid"]),
    )
    secids = [u["secid"] for u in ordered if u.get("secid")]
    if len(secids) < n:
        for fallback in FALLBACK_UNIVERSE:
            if fallback["secid"] not in secids:
                secids.append(fallback["secid"])
            if len(secids) >= n:
                break
    return secids[:n]


def parse_candles_block(payload: dict[str, Any], block_name: str = "candles") -> pd.DataFrame:
    """ISS candles-блок → OHLCV DataFrame (колонки Open/High/Low/Close/Volume).

    Индекс — DatetimeIndex из колонки ``begin`` (MSK), нормализованный к UTC.
    """
    block = payload.get(block_name) or {}
    columns = [str(c).strip().lower() for c in (block.get("columns") or [])]
    rows = block.get("data") or []

    def col_index(*names: str) -> int | None:
        for name in names:
            if name in columns:
                return columns.index(name)
        return None

    i_begin = col_index("begin", "end")
    i_open = col_index("open")
    i_high = col_index("high")
    i_low = col_index("low")
    i_close = col_index("close", "legalcloseprice")
    i_volume = col_index("volume", "value")

    records: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        begin = row[i_begin] if i_begin is not None and i_begin < len(row) else None
        try:
            ts = pd.Timestamp(begin)
        except Exception:
            continue
        if pd.isna(ts):
            continue
        # ISS отдаёт begin в MSK; для единообразия с остальным стеком держим UTC.
        ts = ts - pd.Timedelta(hours=3)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        rec: dict[str, Any] = {"ts": ts}
        for key, idx in (
            ("Open", i_open), ("High", i_high), ("Low", i_low),
            ("Close", i_close), ("Volume", i_volume),
        ):
            value = row[idx] if idx is not None and idx < len(row) else None
            rec[key] = _num(value)
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = df.set_index("ts")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df.sort_index()


def drop_forming_session(
    frame: pd.DataFrame | pd.Series,
    *,
    now_msk: datetime | None = None,
) -> pd.DataFrame | pd.Series:
    """Убрать бар текущей незавершённой сессии MOEX.

    Сессия считается формирующейся до 19:00 МСК. В 23:00/08:00 МСК (наше
    расписание) фильтр ничего не удаляет: в 23:00 дневная свеча дня уже
    завершена, в 08:00 последняя свеча — вчерашняя.
    """
    now = now_msk or datetime.now(_MSK)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_MSK)
    if now.hour >= _FORMING_SESSION_CUTOFF:
        return frame
    today = now.date()
    mask = [pd.Timestamp(ts).date() != today for ts in frame.index]
    if isinstance(frame, pd.Series):
        return frame.loc[mask]
    return frame.loc[mask]


# ═════════════════════════════════════════════════════════════════════════
# Сетевой fetcher
# ═════════════════════════════════════════════════════════════════════════
def _df_to_bars(df: pd.DataFrame, *, close_only: bool = False) -> list[dict[str, Any]]:
    """DataFrame → список баров для JSON-обёртки оркестратора."""
    out: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        item: dict[str, Any] = {"t": ts.isoformat(), "c": float(row["Close"])}
        if not close_only:
            item.update({
                "o": float(row.get("Open", row["Close"])),
                "h": float(row.get("High", row["Close"])),
                "l": float(row.get("Low", row["Close"])),
                "v": float(row.get("Volume", 0.0) or 0.0),
            })
        out.append(item)
    return out


class ImoexBreadthFetcher:
    """Полный ISS-парсер для /breadth-imoex (один запуск = один payload)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = float(timeout)
        self._limiter = get_rate_limiter()

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #
    def _get_json(self, url: str, params: dict[str, Any] | None = None, *, retries: int = 2) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            self._limiter.wait("moex_iss")
            try:
                resp = requests.get(url, params=params, timeout=self.timeout, headers=_UA)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                logger.warning("IMOEX breadth ISS GET failed (attempt %d): %s", attempt + 1, exc)
                if attempt < retries:
                    time.sleep(0.8)
        raise RuntimeError(f"ISS request failed: {last_exc}")

    def fetch_universe(self) -> list[dict[str, Any]]:
        """Состав IMOEX: ISS analytics → fallback-список голубых фишек."""
        try:
            payload = self._get_json(
                _ISS_ANALYTICS_URL,
                params={"iss.meta": "off", "iss.only": "analytics", "limit": 100},
            )
            universe = parse_universe_payload(payload)
            if len(universe) >= 10:
                return universe
            logger.warning("IMOEX universe too small (%d) — fallback list used", len(universe))
        except Exception as exc:
            logger.warning("IMOEX universe fetch failed: %s", exc)
        return [dict(u) for u in FALLBACK_UNIVERSE]

    def _fetch_daily_candles(self, url_template: str, secid: str) -> pd.DataFrame:
        """Дневные свечи одного инструмента ISS (пагинация + завершённые дни)."""
        from_str = (datetime.now(timezone.utc) - timedelta(days=_HISTORY_DAYS)).strftime("%Y-%m-%d")
        frames: list[pd.DataFrame] = []
        for page in range(_ISS_MAX_PAGES):
            params = {
                "iss.meta": "off",
                "iss.only": "candles",
                "interval": 24,
                "from": from_str,
                "iss.reverse": "true",
                "start": page * _ISS_PAGE_SIZE,
            }
            payload = self._get_json(url_template.format(secid=secid), params=params)
            df = parse_candles_block(payload)
            if df.empty:
                break
            frames.append(df)
            if len(df) < _ISS_PAGE_SIZE:
                break
        if not frames:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="first")].sort_index()
        out = drop_forming_session(out)
        return out.tail(_HISTORY_DAYS + 5)

    def fetch_index_daily(self) -> pd.DataFrame:
        """Дневные свечи индекса IMOEX (board SNDX), только завершённые дни."""
        return self._fetch_daily_candles(_ISS_INDEX_CANDLES_URL, "IMOEX")

    def fetch_rvi_daily(self) -> pd.DataFrame:
        """Дневные значения индекса RVI (SNDX) — для сравнения с ATR."""
        return self._fetch_daily_candles(_ISS_INDEX_CANDLES_URL, "RVI")

    def fetch_constituent_ohlc(self, secids: list[str]) -> dict[str, pd.DataFrame]:
        """Дневные OHLCV бумаг состава (TQBR) — любой SECID, параллельно, ~2 rps."""
        frames: dict[str, pd.DataFrame] = {}
        lock = threading.Lock()

        def one(secid: str) -> tuple[str, pd.DataFrame | None]:
            try:
                df = self._fetch_daily_candles(_ISS_TQBR_CANDLES_URL, secid)
            except Exception as exc:
                logger.warning("IMOEX breadth %s daily candles failed: %s", secid, exc)
                return secid, None
            if df is None or df.empty:
                return secid, None
            required = ["Open", "High", "Low", "Close"]
            if any(col not in df.columns for col in required):
                return secid, None
            df = df[required + ([c for c in ("Volume",) if c in df.columns])]
            df = df.dropna(subset=["Close"])
            df = drop_forming_session(df)
            df = df.tail(_HISTORY_DAYS + 5)
            if len(df) < _MIN_BARS:
                return secid, None
            return secid, df

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(one, secid): secid for secid in dict.fromkeys(secids)}
            for future in as_completed(futures):
                secid, df = future.result()
                if df is None:
                    continue
                with lock:
                    frames[secid] = df
        return frames

    def fetch_constituent_closes(self, secids: list[str]) -> dict[str, pd.Series]:
        """Дневные Close бумаг состава (совместимый API поверх fetch_constituent_ohlc)."""
        frames = self.fetch_constituent_ohlc(secids)
        return {secid: df["Close"].astype(float).dropna() for secid, df in frames.items()}

    def fetch_payload(self) -> dict[str, Any]:
        """Собрать полный payload для data_type=imoex_breadth.

        Returns
        -------
        dict
            ``{"universe": [...], "top10": [...], "imoex": {"bars": [...]},
            "rvi": {"bars": [...]}, "stocks": [{"symbol", "bars": [...]}]}``.
        """
        universe = self.fetch_universe()
        top10 = top10_secids(universe)
        index_df = self.fetch_index_daily()
        if index_df.empty:
            raise RuntimeError("IMOEX index candles are empty")

        secids = [u["secid"] for u in universe]
        ohlc = self.fetch_constituent_ohlc(secids)
        rvi_df = self.fetch_rvi_daily()

        stocks = [
            {"symbol": secid, "bars": _df_to_bars(df)}
            for secid, df in sorted(ohlc.items())
        ]
        return {
            "universe": universe,
            "top10": top10,
            "imoex": {"bars": _df_to_bars(index_df)},
            "rvi": {"bars": _df_to_bars(rvi_df) if not rvi_df.empty else []},
            "stocks": stocks,
        }
