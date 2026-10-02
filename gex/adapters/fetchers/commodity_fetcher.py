"""Polling данных товарных рынков через Yahoo Finance (yfinance).

Два режима:
  * ETF-прокси (has_options=True) — опционная цепочка через ETF → полный GEX;
  * Price-action (has_options=False) — только OHLCV, без опционов.

Использует yfinance. Все данные кэшируются в Redis (Cache-Aside).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from gex.commodity_assets import COMMODITY_ASSETS
from gex.domain.data_loader import OptionSnapshot
from gex.adapters.cache.redis_client import RedisClient, deserialize_value
from gex.adapters.cache.keys import PROVIDER_YFINANCE, commodity_key
from gex.adapters.providers.yfinance import history as yfinance_history
from gex.adapters.providers.yfinance import spot as yfinance_spot

logger = logging.getLogger(__name__)

_OHLCV_PERIODS = {"1d": "1y", "1h": "90d"}


class CommodityFetcher:
    """Получение товарных котировок + опционных цепочек ETF через yfinance.

    Parameters
    ----------
    redis_client : Optional[RedisClient]
        Клиент Redis для кэширования.
    """

    def __init__(self, redis_client: Optional[RedisClient] = None):
        self._redis = redis_client

    # ------------------------------------------------------------------ #
    #  Spot
    # ------------------------------------------------------------------ #
    def fetch_spot(self, asset: str) -> dict:
        asset_upper = asset.strip().upper()
        cfg = self._get_cfg(asset_upper)
        yf_sym = cfg["yf_symbol"]
        cache_key_str = commodity_key("spot", asset_upper, provider=PROVIDER_YFINANCE)

        if self._redis is not None and self._redis.connected:
            cached = self._redis.get(cache_key_str)
            if cached is not None:
                try:
                    return deserialize_value(cached)
                except Exception:
                    pass

        # Central orchestrator path (when enabled).
        try:
            from gex.orchestrator.sync_gateway import sync_fetch_spot
            spot = sync_fetch_spot(yf_sym, "yfinance")
            if spot is not None:
                data = {
                    "symbol": asset_upper,
                    "spot": round(float(spot), 4),
                    "label": cfg["label"],
                    "unit": cfg["unit"],
                    "category": cfg["category"],
                    "has_options": cfg.get("has_options", False),
                    "etf_proxy": cfg.get("etf_proxy"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                if self._redis is not None and self._redis.connected:
                    try:
                        self._redis.set(cache_key_str, data, ex=300)
                    except Exception:
                        pass
                return data
        except Exception:
            pass

        logger.info("Fetching spot for %s (%s)", asset_upper, yf_sym)
        # Спот через адаптер: он сам перебирает fast_info и последний Close,
        # а «источник упал» и «цену не отдали» различимы в логе адаптера.
        spot = yfinance_spot(yf_sym)
        if not spot or spot <= 0:
            raise RuntimeError(f"Spot fetch failed for {yf_sym}: пустой ответ yfinance")

        data = {
            "symbol": asset_upper,
            "spot": round(float(spot), 4),
            "label": cfg["label"],
            "unit": cfg["unit"],
            "category": cfg["category"],
            "has_options": cfg.get("has_options", False),
            "etf_proxy": cfg.get("etf_proxy"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._redis is not None and self._redis.connected:
            try:
                self._redis.set(cache_key_str, data, ex=300)
            except Exception:
                pass

        return data

    # ------------------------------------------------------------------ #
    #  OHLCV
    # ------------------------------------------------------------------ #
    def fetch_ohlcv(
        self, asset: str, timeframe: str = "1d", limit: int = 200
    ) -> dict:
        asset_upper = asset.strip().upper()
        cfg = self._get_cfg(asset_upper)
        yf_sym = cfg["yf_symbol"]
        period = _OHLCV_PERIODS.get(timeframe, "1y")
        cache_key_str = commodity_key("ohlcv", asset_upper, timeframe, limit, provider=PROVIDER_YFINANCE)

        if self._redis is not None and self._redis.connected:
            cached = self._redis.get(cache_key_str)
            if cached is not None:
                try:
                    return deserialize_value(cached)
                except Exception:
                    pass

        # Central orchestrator path (when enabled).
        try:
            from gex.orchestrator.sync_gateway import sync_fetch_ohlcv
            df = sync_fetch_ohlcv(yf_sym, "yfinance", timeframe, limit=limit)
            if df is not None and not df.empty:
                df = df.tail(limit)
                spot = float(df["Close"].iloc[-1])
                bars = []
                for idx, row in df.iterrows():
                    bars.append({
                        "t": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                        "o": round(float(row["Open"]), 4),
                        "h": round(float(row["High"]), 4),
                        "l": round(float(row["Low"]), 4),
                        "c": round(float(row["Close"]), 4),
                        "v": int(row.get("Volume", 0)),
                    })
                data = {
                    "symbol": asset_upper,
                    "asset_type": "commodity",
                    "timeframe": timeframe,
                    "spot": spot,
                    "bars": bars,
                }
                if self._redis is not None and self._redis.connected:
                    try:
                        self._redis.set(cache_key_str, data, ex=600)
                    except Exception:
                        pass
                return data
        except Exception:
            pass

        logger.info("Fetching OHLCV for %s (%s) tf=%s", asset_upper, yf_sym, timeframe)
        try:
            interval = "1h" if timeframe == "1h" else "1d"
            df = yfinance_history(yf_sym, period=period, interval=interval)
            if df is None:
                raise RuntimeError(f"No OHLCV data for {yf_sym}")

            df = df.tail(limit)
            spot = float(df["Close"].iloc[-1])

            bars = []
            for idx, row in df.iterrows():
                bars.append({
                    "t": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "o": round(float(row["Open"]), 4),
                    "h": round(float(row["High"]), 4),
                    "l": round(float(row["Low"]), 4),
                    "c": round(float(row["Close"]), 4),
                    "v": int(row.get("Volume", 0)),
                })
        except Exception as e:
            logger.error("yfinance OHLCV failed for %s: %s", yf_sym, e)
            raise RuntimeError(f"OHLCV fetch failed for {yf_sym}: {e}")

        data = {
            "symbol": asset_upper,
            "asset_type": "commodity",
            "timeframe": timeframe,
            "spot": spot,
            "bars": bars,
        }

        if self._redis is not None and self._redis.connected:
            try:
                self._redis.set(cache_key_str, data, ex=600)
            except Exception:
                pass

        return data

    # ------------------------------------------------------------------ #
    #  Option Chain (ETF proxy) — с нормализацией цен к базовому активу
    # ------------------------------------------------------------------ #
    def fetch_option_chain(
        self, asset: str, max_expiries: int = 5, max_days: Optional[float] = None,
    ) -> Optional[OptionSnapshot]:
        """Получить опционную цепочку ETF-прокси, нормализованную к цене товара.

        ETF торгуется в своих единицах (GLD ~$398 при gold $4383/oz).
        Все страйки и spot пересчитываются в единицы базового актива
        через коэффициент ``ratio = futures_price / etf_price``.

        Аудит 2026-09-17: коэффициент проверяется по **полосе актива**
        (``ratio_band`` в :data:`gex.commodity_assets.COMMODITY_ASSETS`), а не
        по общей ``[0.2, 5.0]`` — та годилась только для фондов «1 акция ≈
        1 единица товара» и ложно бракововала золото (11.0), CPER (0.167),
        палладий (55.6) и платину (110.7). Вне полосы цепочка **не**
        масштабируется: лучше отдать профиль в единицах прокси с явным
        флагом в ``meta["commodity_proxy"]``, чем в неверных единицах товара.

        Returns None, если ETF-прокси недоступен или опционы отсутствуют.
        """
        asset_upper = asset.strip().upper()
        cfg = self._get_cfg(asset_upper)

        if not cfg.get("has_options") or not cfg.get("etf_proxy"):
            return None

        etf_ticker = cfg["etf_proxy"]
        cache_key_str = commodity_key("chain", asset_upper, max_expiries, provider=PROVIDER_YFINANCE)

        if self._redis is not None and self._redis.connected:
            cached = self._redis.get(cache_key_str)
            if cached is not None:
                try:
                    return deserialize_value(cached)
                except Exception:
                    pass

        logger.info(
            "Fetching option chain for %s via ETF proxy %s", asset_upper, etf_ticker
        )

        # ── Fetch ETF option chain + commodity spot параллельно ──
        commodity_spot = None
        try:
            commodity_data = self.fetch_spot(asset_upper)
            commodity_spot = commodity_data["spot"]
        except Exception as e:
            logger.warning("Commodity spot unavailable for %s: %s", asset_upper, e)

        try:
            from gex.adapters.fetchers.yf_fetcher import YFOptionsFetcher
            fetcher = YFOptionsFetcher(
                max_expiries=max_expiries, redis_client=self._redis, max_days=max_days,
            )
            snapshot = fetcher.fetch(etf_ticker)
            snapshot.symbol = asset_upper
        except Exception as e:
            logger.warning(
                "Option chain unavailable for %s (proxy %s): %s",
                asset_upper, etf_ticker, e,
            )
            return None

        # ── Нормализация цен ETF → базовый актив ──
        # Полоса берётся из конфигурации актива: общая [0.2, 5.0] была неверна
        # для трастов с дробной долей унции (см. docstring метода и аудит).
        band = cfg.get("ratio_band") or [0.2, 5.0]
        etf_spot = snapshot.spot
        proxy_meta: dict = {
            "asset": asset_upper,
            "proxy": etf_ticker,
            "base_symbol": cfg["yf_symbol"],
            "base_spot": commodity_spot,
            "etf_spot": etf_spot,
            "ratio": None,
            "ratio_band": band,
            "rescaled": False,
            "note": "",
        }

        if commodity_spot and commodity_spot > 0 and snapshot.spot > 0:
            ratio = commodity_spot / snapshot.spot
            proxy_meta["ratio"] = ratio
            logger.info(
                "Normalizing %s: base(%s)=%.2f, ETF(%s)=%.2f, ratio=%.4f, band=%s",
                asset_upper, cfg["yf_symbol"], commodity_spot, etf_ticker,
                snapshot.spot, ratio, band,
            )
            in_band = band[0] <= ratio <= band[1]
            if not in_band:
                # Рассинхронизация или изменившаяся структура фонда (сплит).
                # Масштабировать нельзя — вернём цепочку в единицах прокси и
                # пометим это: потребитель/UI обязаны это показать.
                logger.warning(
                    "Commodity %s: rescale ratio %.4f вне полосы %s — цепочка "
                    "оставлена в единицах прокси %s (base=%.2f, etf=%.2f)",
                    asset_upper, ratio, band, etf_ticker, commodity_spot, etf_spot,
                )
                proxy_meta["note"] = (
                    f"коэффициент {ratio:.4f} вне полосы {band} — "
                    f"страйки в единицах {etf_ticker}, не в единицах товара"
                )
            else:
                snapshot.chain["strike"] = snapshot.chain["strike"] * ratio
                snapshot.spot = commodity_spot
                proxy_meta["rescaled"] = True
        else:
            logger.warning(
                "Cannot normalize %s: commodity_spot=%s, etf_spot=%s",
                asset_upper, commodity_spot, snapshot.spot,
            )
            proxy_meta["note"] = "нет цены базового актива — цепочка в единицах прокси"

        snapshot.meta["commodity_proxy"] = proxy_meta

        if self._redis is not None and self._redis.connected:
            try:
                self._redis.set(cache_key_str, snapshot, ex=600)
            except Exception:
                pass

        return snapshot

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _get_cfg(self, asset: str) -> dict:
        cfg = COMMODITY_ASSETS.get(asset)
        if cfg is None:
            raise ValueError(
                f"Unsupported commodity '{asset}'. "
                f"Available: {list(COMMODITY_ASSETS)}."
            )
        return cfg
