"""Polling опционных данных через Yahoo Finance (yfinance).

Модуль реализует «живое» получение опционных цепочек с.yahoo Finance
и преобразование их в канонический :class:`OptionSnapshot`, совместимый
с существующим GEX pipeline.

Используемые столбцы yfinance option_chain:
  * ``strike``            — цена страйка;
  * ``openInterest``      — открытый интерес (контракты);
  * ``impliedVolatility`` — подразумеваемая волатильность (годовая);
  * ``lastPrice``         — последняя цена опциона.

Пример использования::

    fetcher = YFOptionsFetcher(max_expiries=5)
    snapshot = fetcher.fetch("AAPL")
    # snapshot.chain: DataFrame со столбцами strike, type, oi, iv, T
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from gex.domain.data_loader import GEXDataLoader, OptionSnapshot
from gex.adapters.cache.redis_client import RedisClient, serialize_value, deserialize_value
from gex.adapters.cache.keys import PROVIDER_YFINANCE, chain_key
from gex.adapters.transport.yf_transport import DeadlineTicker, get_shared_yf_transport

# ETF proxy mapping: user-facing ticker → yfinance symbol with options
_YF_TICKER_MAP: dict[str, str] = {
    "ES": "^SPX",
    "NQ": "^NDX",
}

logger = logging.getLogger(__name__)

# Короткая экспоненциальная задержка с джиттером для одной ретрай-попытки
# на транзиентных сетевых ошибках (yfinance/Webull — без SLA, один ретрай дешёв).
_BACKOFF_BASE_SECONDS: float = 0.5


def _backoff_delay(attempt: int) -> float:
    """Экспоненциальная задержка ``0.5·2^attempt`` + джиттер 0–0.3 с."""
    return _BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0.0, 0.3)


class YFOptionsFetcher:
    """Получение свежих опционных данных через yfinance.

    Parameters
    ----------
    max_expiries : int
        Максимальное количество ближайших экспираций для загрузки.
        По умолчанию 5 — оптимальный баланс между полнотой и скоростью.
    """

    def __init__(
        self,
        max_expiries: int = 5,
        redis_client: Optional[RedisClient] = None,
        max_days: Optional[float] = None,
    ):
        if max_expiries < 1:
            raise ValueError("max_expiries должен быть >= 1")
        self.max_expiries = int(max_expiries)
        self._redis = redis_client
        # Горизонт в днях: экспирации дальше него анализатор всё равно отбросит
        # фильтром по days, поэтому не тратим на них запросы (аудит 2026-09-17).
        self.max_days = float(max_days) if max_days else None

    def fetch(self, ticker: str) -> OptionSnapshot:
        """Polling: получить свежие опционные данные, вернуть OptionSnapshot.

        Каждый вызов делает HTTP-запрос к Yahoo Finance — данные всегда свежие.

        Parameters
        ----------
        ticker : str
            Тикер в формате yfinance (``SPY``, ``AAPL``, ``^SPX``, ``QQQ``, …).

        Returns
        -------
        OptionSnapshot
            Канонический снапшот, готовый для GEX pipeline.

        Raises
        ------
        ValueError
            Если тикер не найден, опционных данных нет или данные пустые
            после очистки.
        RuntimeError
            При ошибках сети или внутренних ошибках yfinance.
        """
        ticker_str = ticker.strip().upper()
        yf_ticker = _YF_TICKER_MAP.get(ticker_str, ticker_str)
        logger.info("yfinance polling: ticker=%s (yf=%s), max_expiries=%d",
                    ticker_str, yf_ticker, self.max_expiries)

        # ── Redis cache check ──
        # Провайдер в ключе: у Bybit под тем же тикером «BTC» лежит ДРУГОЙ инструмент
        # (свои страйки/экспирации), поэтому общий ключ был ошибкой (итер. 25).
        cache_key_str = chain_key(ticker_str, self.max_expiries, provider=PROVIDER_YFINANCE)
        if self._redis is not None and self._redis.connected:
            cached_data = self._redis.get(cache_key_str)
            if cached_data is not None:
                try:
                    result = deserialize_value(cached_data)
                    if isinstance(result, OptionSnapshot):
                        logger.info("  Chain CACHE HIT for %s", ticker_str)
                        return result
                except Exception:
                    logger.debug("Chain deserialize error for %s — refetching", ticker_str)

        # --- 1. Создаём Ticker и получаем spot (используем yf_ticker для yfinance) ---
        # Через транспорт: тот же интерфейс, но с дедлайном — yfinance не принимает timeout
        # и иначе может держать воркер неограниченно долго.
        yt = get_shared_yf_transport().ticker(yf_ticker)

        # spot: пробуем несколько источников (yfinance не всегда отдаёт все поля)
        spot = self._get_spot(yt, ticker_str)
        if spot is None or spot <= 0:
            raise ValueError(
                f"Не удалось получить текущую цену для тикера '{ticker_str}'. "
                "Убедитесь, что тикер корректный и рынок открыт/был открыт."
            )

        # --- 2. Получаем список дат экспирации ---
        expiries = self._get_expiries(yt, ticker_str)
        if not expiries:
            raise ValueError(f"Нет доступных экспираций для тикера '{ticker_str}'.")

        # «Сейчас» в UTC (даты экспирации tz-naive, см. комментарий ниже).
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

        # Ограничиваем N ближайшими с ненулевым OI.
        # Yahoo Finance может отдавать OI=0 для ближайших экспираций
        # (особенно для 0DTE/1DTE SPX/NDX). Сканируем все экспирации
        # и берём до max_expiries тех, где есть реальный OI.
        expiries_to_try = list(expiries)
        if self.max_days is not None:
            kept = []
            for e in expiries_to_try:
                try:
                    days = (pd.Timestamp(e) - now).total_seconds() / 86400.0
                except (ValueError, TypeError):
                    continue
                if days <= self.max_days:
                    kept.append(e)
            if kept:
                expiries_to_try = kept
            else:
                logger.warning(
                    "  горизонт max_days=%s не покрывает ни одной экспирации %s — "
                    "оставляю ближайшие", self.max_days, ticker_str,
                )
        expiries = []
        # Кэш предскана: цепочка уже скачана на шаге OI-прескана — не качаем её
        # второй раз в цикле загрузки ниже (N+1 → N вызовов option_chain).
        pre_scanned: dict[str, object] = {}
        for e in expiries_to_try:
            if len(expiries) >= self.max_expiries:
                break
            try:
                test_chain = yt.option_chain(e)
                pre_scanned[e] = test_chain
                has_oi = (
                    (test_chain.calls["openInterest"].fillna(0) > 0).any()
                    or (test_chain.puts["openInterest"].fillna(0) > 0).any()
                )
                if has_oi:
                    expiries.append(e)
                else:
                    logger.debug("  expiry %s: OI=0, skipping", e)
            except Exception:
                logger.debug("  expiry %s: fetch error, skipping", e)

        if not expiries:
            # fallback: use first N original expiries (old behaviour)
            expiries = list(expiries_to_try[: self.max_expiries])
            logger.info("  no expiries with OI>0, falling back to first %d", len(expiries))

        logger.info("  expiries: %s", [str(e) for e in expiries])

        # --- 3. Загружаем цепочки по каждой экспирации ---
        all_frames: list[pd.DataFrame] = []
        # expiry из yfinance — строки 'YYYY-MM-DD' (tz-naive), поэтому оба
        # операнда в tz-naive форме, чтобы вычитание не падало с
        # "Cannot subtract tz-naive and tz-aware" (``now`` определён above).

        for expiry in expiries:
            try:
                chain = pre_scanned.get(expiry)
                if chain is None:
                    chain = yt.option_chain(expiry)
                # chain — namedtuple с полями .calls и .puts (оба DataFrame)
                calls = chain.calls
                puts = chain.puts

                if calls.empty and puts.empty:
                    logger.debug("  expiry %s: пустая цепочка, пропускаем", expiry)
                    continue

                # --- Преобразование в каноничный формат ---
                # T — время до экспирации в годах
                T = max((pd.Timestamp(expiry) - now).total_seconds() / 86400.0, 0.0) / 365.0

                for frame, opt_type in ((calls, "C"), (puts, "P")):
                    if frame.empty:
                        continue
                    df = pd.DataFrame()
                    df["strike"] = frame["strike"].astype(float)
                    df["oi"] = frame["openInterest"].fillna(0).astype(float)
                    # NaN (а не 0): у части контрактов Yahoo не отдаёт IV, но они
                    # имеют открытый интерес и должны войти в GEX — волатильность
                    # восстановит _interpolate_iv по соседним страйкам группы.
                    df["iv"] = frame["impliedVolatility"].astype(float)
                    df["type"] = opt_type
                    df["T"] = T
                    all_frames.append(df)

            except Exception as exc:
                logger.warning("  expiry %s: ошибка загрузки — %s", expiry, exc)
                continue

        if not all_frames:
            raise ValueError(
                f"Не удалось загрузить опционные данные для '{ticker_str}'. "
                "Все экспирации вернули пустые цепочки или ошибки."
            )

        # --- 4. Объединяем все экспирации ---
        raw = pd.concat(all_frames, ignore_index=True)

        # --- 5. Фильтрация до очистки ---
        # Отбрасываем только строки без открытого интереса или с некорректным
        # страйком: такой контракт не даёт вклада в GEX. Отсутствующая IV —
        # НЕ причина отбрасывания, её восстановит _interpolate_iv.
        pre_filter_count = int(len(raw))
        kept_no_iv = int((raw["iv"].isna() | (raw["iv"] <= 0)).sum())
        dropped_oi = int((raw["oi"] <= 0).sum())
        dropped_strike = int((raw["strike"] <= 0).sum())
        # NaN IV → NaN (не 0), чтобы интерполяция в лоадере видела пропуск.
        raw.loc[raw["iv"] <= 0, "iv"] = float("nan")
        raw = raw[(raw["oi"] > 0) & (raw["strike"] > 0)].copy()

        if raw.empty:
            raise ValueError(
                f"Опционная цепочка '{ticker_str}' пуста после фильтрации "
                "(нет контрактов с OI > 0)."
            )

        logger.info(
            "  yfinance %s: %d строк до фильтра, %d осталось, отброшено %d "
            "(oi<=0: %d, strike<=0: %d), без IV (к интерполяции): %d",
            ticker_str, pre_filter_count, len(raw),
            pre_filter_count - len(raw),
            dropped_oi, dropped_strike, kept_no_iv,
        )

        logger.info(
            "  загружено %d строк (%d страйков) по %d экспирациям, spot=%.2f",
            len(raw),
            raw["strike"].nunique(),
            len(expiries),
            spot,
        )

        # --- 6. Очистка через GEXDataLoader ---
        loader = GEXDataLoader(spot=spot, symbol=ticker_str)
        snapshot = loader.load_dataframe(
            raw, as_of=datetime.now(timezone.utc), preserve_expiry=True,
        )

        logger.info(
            "  yfinance %s: очистка %d → %d строк (%d страйков); подано без IV: %d",
            ticker_str, len(raw), len(snapshot.chain),
            snapshot.chain["strike"].nunique(), kept_no_iv,
        )

        # ── Сохраняем в Redis ──
        if self._redis is not None and self._redis.connected:
            self._redis.set(cache_key_str, snapshot, ex=600)

        return snapshot

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_spot(yt: "DeadlineTicker", ticker: str) -> Optional[float]:
        """Получить текущую цену актива из нескольких источников."""
        # 1. fast_info.lastPrice (самый надёжный для ETF/акций)
        try:
            spot = yt.fast_info.get("lastPrice")
            if spot is not None and spot > 0:
                return float(spot)
        except (KeyError, AttributeError, TypeError):
            pass

        # 2. info.regularMarketPrice
        try:
            info = yt.info
            if info and "regularMarketPrice" in info:
                spot = info["regularMarketPrice"]
                if spot is not None and spot > 0:
                    return float(spot)
        except (KeyError, AttributeError, TypeError):
            pass

        # 3. history(last 1 bar) → close
        try:
            hist = yt.history(period="1d")
            if not hist.empty:
                spot = float(hist["Close"].iloc[-1])
                if spot > 0:
                    return spot
        except Exception:
            pass

        logger.warning("Не удалось получить spot для %s", ticker)
        return None

    @staticmethod
    def _get_expiries(yt: "DeadlineTicker", ticker: str) -> list[str]:
        """Получить отсортированный список дат экспирации.

        Важно: возвращаются **исходные строки** (``"YYYY-MM-DD"``), а не
        ``pd.Timestamp``. В yfinance >= 1.5 ``option_chain(expiry)`` ищет
        экспирацию точным сопоставлением строк по ``yt.options``; передача
        Timestamp (с компонентом времени ``00:00:00``) приводит к ошибке
        ``Expiration ... cannot be found``. Поэтому сортируем по дате, но
        отдаём строки — именно их ожидает ``option_chain``.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                expiries = yt.options
                if expiries is None or len(expiries) == 0:
                    return []
                # yfinance возвращает кортеж дат в формате YYYY-MM-DD.
                # Сортируем по дате, но сохраняем исходные строки.
                def _sort_key(e: str):
                    try:
                        return pd.Timestamp(e)
                    except (ValueError, TypeError):
                        return pd.Timestamp.max  # неразборчивые — в конец
                return sorted(expiries, key=_sort_key)
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Ошибка получения экспираций для %s (попытка %d): %s — "
                        "повтор через %.2fs",
                        ticker, attempt + 1, exc, delay,
                    )
                    time.sleep(delay)
                    continue
        logger.warning("Ошибка получения экспираций для %s: %s", ticker, last_exc)
        return []
