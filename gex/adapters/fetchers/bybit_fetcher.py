"""Polling опционных данных криптовалют через Bybit V5 API.

Модуль получает опционные цепочки на криптовалюты (**BTC**, **ETH**, **SOL**,
**XRP**, **DOGE**) через публичный endpoint::

    https://api.bybit.com/v5/market/tickers?category=option&baseCoin={coin}

и преобразует их в канонический :class:`OptionSnapshot`, совместимый с
существующим GEX pipeline (столбцы ``strike,type,oi,iv,T``).

Специфика Bybit
---------------
* Опционы — европейские, cash-settled, котируются в USDT. Один контракт
  соответствует **1 монете** базового актива → ``per_contract = 1`` (масштаб
  GEX; на положения стен/Gamma Flip/режим не влияет).
* API отдаёт **настоящую markIv** (годовая подразумеваемая волатильность,
  уже в долях единицы, напр. ``0.4608``) — в отличие от MOEX, где IV
  аппроксимируется через индекс RVI. Дополнительно отдаются готовые греки
  (``delta``, ``gamma``, ``vega``, ``theta``) — для GEX они не нужны, pipeline
  пересчитывает греки сам из цепочки.
* Базис для GEX — ``underlyingPrice`` (цена монеты в USD): единая для всех
  контрактов данной монеты на момент снапшота.
* ``openInterest`` — в единицах базового актива (монетах), а не в контрактах.
  Для 1 контракт = 1 монета значения совпадают.
* Символ контракта кодирует все параметры: ``COIN-DDMMMYY-STRIKE-TYPE-CURRENCY``
  (напр. ``BTC-12JUL26-58000-P-USDT``, ``XRP-13JUL26-1.12-C-USDT``). Парсится
  через split; strike — ``float`` (поддерживает дробные страйки для низкоценных
  монет вроде DOGE/XRP).

Покрытие монет
~~~~~~~~~~~~~~
Реальный ликвидный опционный рынок на Bybit существует для **BTC**, **ETH**,
**SOL**, **XRP**, **DOGE** (по состоянию на 2026). Прочие тикеры (BNB, TRX, HYPE,
LEO, ZEC) опционов не имеют — это факт рынка, эндпоинт вернёт пустой список
(``ValueError``). Ранее список жил в ``scanner_crypto.py``; модуль удалён как мёртвый
(аудит 2026-09-16), актуальный источник справочников — ``gex/adapters/providers/catalog.py``.

Конвенция знаков дилера
~~~~~~~~~~~~~~~~~~~~~~~
По умолчанию — equity-конвенция SqueezeMetrics (``call_sign=+1, put_sign=-1``),
сопоставимая с публичными крипто-GEX дашбордами (CoinGlass и др.) и с остальными
активами проекта. Знаки параметризованы в :data:`_CRYPTO_ASSETS` — при иной
гипотезе о позиционировании дилеров достаточно поменять значения в конфиге.

Пример::

    fetcher = BybitOptionsFetcher(max_expiries=5)
    snap = fetcher.fetch("BTC")      # OptionSnapshot с цепочкой BTC
    snap = fetcher.fetch("ETH")      # OptionSnapshot с цепочкой ETH
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from gex.assets_config import CRYPTO_ASSETS
from gex.domain.data_loader import OptionSnapshot
from gex.adapters.cache.redis_client import RedisClient, deserialize_value
from gex.adapters.cache.keys import PROVIDER_BYBIT, chain_key

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Константы инструментов
# ====================================================================== #
# Поддерживаемые базовые активы (baseCoin на Bybit) и их конфигурация.
#   * r — USD безрисковая ставка (крипто-опционы котируются в USDT, базис — USD).
#   * q — дивидендная доходность (у крипты нет дивидендов → 0).
#   * per_contract — масштаб GEX: 1 контракт = 1 монета базового актива.
#     На положения уровней (Call/Put Wall, Gamma Flip) и режим не влияет —
#     только на абсолютную величину Net GEX.
#   * call_sign / put_sign — знаки дилера. По умолчанию SqueezeMetrics
#     (как для equity/индексов), сопоставимо с публичными крипто-GEX дашбордами.
#: Псевдоним таблицы из :mod:`gex.assets_config` — **тот же объект**, не копия.
#: Копия разошлась бы: ``per_contract`` читают и конус, и расширенный профиль.
_CRYPTO_ASSETS: dict[str, dict] = CRYPTO_ASSETS

# Bybit V5 endpoint: опционы по базовой монете.
_BYBIT_OPTIONS_URL = "https://api.bybit.com/v5/market/tickers"

# Срок жизни кэша HTTP-ответов Bybit, секунды (5 минут).
# В течение этого окна повторные вызовы ручек /crypto/gex/{coin} отдают
# закэшированные данные — Bybit обновляет option tickers ~раз в секунду,
# 5 минут — разумный баланс свежести и нагрузки на API.
_BYBIT_CACHE_TTL_SECONDS = 5 * 60


def _bybit_expiry_to_datetime(exp_str: str) -> Optional[datetime]:
    """Распарсить дату экспирации Bybit формата ``DDMMMYY`` (напр. ``12JUL26``).

    Использует собственный словарь месяцев, **не** ``strptime`` с ``%b`` —
    последний зависит от системной локали (на русскоязычной Windows ``JUL``
    не парсится). Возвращает timezone-aware UTC datetime (экспирация Bybit —
    08:00 UTC).

    Returns
    -------
    datetime | None
        ``None`` при ошибке разбора строки.
    """
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    s = exp_str.strip().upper()
    if len(s) != 7:
        return None
    try:
        day = int(s[:2])
        mon = months.get(s[2:5])
        year = 2000 + int(s[5:7])
        if mon is None:
            return None
        # Bybit expiry = 08:00 UTC
        return datetime(year, mon, day, 8, 0, tzinfo=timezone.utc)
    except (ValueError, KeyError):
        return None


class BybitOptionsFetcher:
    """Получение свежих опционных данных криптовалют через Bybit V5 API.

    Parameters
    ----------
    max_expiries : int
        Ограничение на число ближайших экспираций (``0`` = без ограничения).
        По умолчанию 5 — баланс между полнотой ближнего профиля и скоростью.
    timeout : float
        Таймаут HTTP-запроса, секунды.

    Notes
    -----
    Кэш — **class-level** (разделяется между экземплярами, т.к. сервис
    создаёт новый фетчер на каждый запрос) и **per-coin**: Bybit делает
    отдельный запрос на каждую монету, в отличие от MOEX, где один ответ
    содержит все активы. TTL :data:`_BYBIT_CACHE_TTL_SECONDS` (5 минут).
    """

    # Кэш per-coin: {coin -> (monotonic_timestamp, raw_list)}.
    # Class-level: один кэш на все экземпляры фетчера.
    _cache_lock = threading.Lock()
    _cache: dict[str, tuple[float, list[dict]]] = {}

    def __init__(self, max_expiries: int = 5, timeout: float = 30.0, redis_client: Optional[RedisClient] = None):
        if max_expiries < 0:
            raise ValueError(f"max_expiries должен быть >= 0, получено {max_expiries}")
        if timeout <= 0:
            raise ValueError(f"timeout должен быть > 0, получено {timeout}")
        self.max_expiries = int(max_expiries)
        self.timeout = float(timeout)
        self._redis = redis_client

    # ------------------------------------------------------------------ #
    #  Публичная точка входа
    # ------------------------------------------------------------------ #
    def fetch(self, base_coin: str) -> OptionSnapshot:
        """Polling: получить свежие опционные данные по монете → OptionSnapshot.

        Parameters
        ----------
        base_coin : str
            Базовая монета: ``"BTC"``, ``"ETH"``, ``"SOL"``, ``"XRP"``,
            ``"DOGE"`` (регистронезависимо).

        Returns
        -------
        OptionSnapshot
            Цепочка с каноническими столбцами ``strike,type,oi,iv,T`` +
            aux-столбец ``symbol`` (для отладки).

        Raises
        ------
        ValueError
            Если монета не поддерживается или Bybit вернул пустую цепочку
            (опционов по этой монете нет в данный момент).
        RuntimeError
            При сетевых ошибках / некорректном JSON.
        """
        coin = base_coin.strip().upper()
        cfg = _CRYPTO_ASSETS.get(coin)
        if cfg is None:
            raise ValueError(
                f"Неподдерживаемая криптовалюта '{coin}'. Доступны: "
                f"{list(_CRYPTO_ASSETS)}."
            )

        # ── Redis cache check ──
        redis_key = chain_key(coin, self.max_expiries, provider=PROVIDER_BYBIT)
        if self._redis is not None and self._redis.connected:
            cached_data = self._redis.get(redis_key)
            if cached_data is not None:
                try:
                    result = deserialize_value(cached_data)
                    if isinstance(result, OptionSnapshot):
                        logger.info("  Bybit chain CACHE HIT for %s", coin)
                        return result
                except Exception:
                    logger.debug("Bybit deserialize error for %s — refetching", coin)

        raw_list = self._fetch(coin)
        now = datetime.now(timezone.utc)

        chain = self._build_chain(raw_list, now)
        if chain.empty:
            raise ValueError(
                f"Bybit вернул пустую цепочку опционов для '{coin}'. "
                "Возможно, биржа временно не отдаёт данные или опционов по "
                "этой монете нет."
            )

        # Spot = underlyingPrice (единый для всех контрактов монеты).
        spot = self._pick_spot(raw_list, coin)

        if self.max_expiries > 0:
            chain = self._limit_expiries(chain, self.max_expiries)

        snapshot = OptionSnapshot(
            symbol=coin,
            spot=spot,
            as_of=datetime.now(timezone.utc),
            chain=chain.reset_index(drop=True),
            meta={
                "source": "Bybit V5",
                "iv_source": "markIv",
                "per_contract": cfg["per_contract"],
                "max_expiries": self.max_expiries,
            },
        )

        # ── Сохраняем в Redis ──
        if self._redis is not None and self._redis.connected:
            self._redis.set(redis_key, snapshot, ex=600)

        return snapshot

    # ------------------------------------------------------------------ #
    #  Кэшированный фетч сырого ответа
    # ------------------------------------------------------------------ #
    def _fetch(self, coin: str) -> list[dict]:
        """Вернуть список option-ticker-ов по монете (с кэшированием per-coin).

        Double-checked locking по образцу MOEXOptionsFetcher, но кэш keyed
        по монете (Bybit делает отдельный запрос на каждую).
        """
        # Fast path без блокировки
        cached = BybitOptionsFetcher._cache.get(coin)
        if cached is not None:
            cached_at, payload = cached
            if time.monotonic() - cached_at < _BYBIT_CACHE_TTL_SECONDS:
                return payload

        with BybitOptionsFetcher._cache_lock:
            # Re-check внутри блокировки (другой поток мог обновить)
            cached = BybitOptionsFetcher._cache.get(coin)
            if cached is not None:
                cached_at, payload = cached
                if time.monotonic() - cached_at < _BYBIT_CACHE_TTL_SECONDS:
                    return payload
            payload = self._fetch_network(coin)
            BybitOptionsFetcher._cache[coin] = (time.monotonic(), payload)
            return payload

    @staticmethod
    def _fetch_network(coin: str) -> list[dict]:
        """Сетевой запрос к Bybit V5 (без кэша). Изолирует HTTP-логику.

        Raises
        ------
        RuntimeError
            При сетевой ошибке или некорректном JSON/ответе.
        """
        import requests

        try:
            resp = requests.get(
                _BYBIT_OPTIONS_URL,
                params={"category": "option", "baseCoin": coin},
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Сетевая ошибка Bybit для {coin}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Некорректный JSON от Bybit для {coin}: {exc}") from exc

        if data.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit API error для {coin}: retCode={data.get('retCode')}, "
                f"retMsg={data.get('retMsg')!r}"
            )

        result = data.get("result") or {}
        lst = result.get("list") or []
        if not lst:
            logger.warning("Bybit вернул пустой список опционов для %s", coin)
        return lst

    # ------------------------------------------------------------------ #
    #  Разбор сырых контрактов → каноническая цепочка
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_chain(raw_list: list[dict], now: datetime) -> pd.DataFrame:
        """Преобразовать сырой ответ Bybit в DataFrame цепочки.

        Парсит ``symbol`` вида ``BTC-12JUL26-58000-P-USDT`` (5 частей) и
        маппит поля ``openInterest``→oi, ``markIv``→iv. Отбрасывает строки с
        невалидными/нулевыми значениями.
        """
        rows: list[dict] = []
        for item in raw_list:
            symbol = item.get("symbol") or ""
            parsed = BybitOptionsFetcher._parse_symbol(symbol)
            if parsed is None:
                continue

            strike = parsed["strike"]
            opt_type = parsed["type"]
            exp_dt = _bybit_expiry_to_datetime(parsed["expiry"])
            if exp_dt is None:
                continue

            # T в годах (calendar). now — timezone-aware UTC.
            if exp_dt <= now:
                continue  # просроченный контракт
            T = (exp_dt - now).total_seconds() / (365.0 * 86400.0)

            oi = _to_float(item.get("openInterest"))
            iv = _to_float(item.get("markIv"))

            # Фильтр арбитражных/неверных значений
            if not (strike > 0 and oi > 0 and T > 0 and iv > 0):
                continue
            # markIv от Bybit — годовая в долях (напр. 0.46). Аномально большие
            # (>5.0 = 500%) — мусор/мало ликвидности, отбрасываем.
            if iv > 5.0:
                continue

            rows.append({
                "strike": strike,
                "type": opt_type,
                "oi": oi,
                "iv": iv,
                "T": T,
                "symbol": symbol,  # aux для отладки
            })

        return pd.DataFrame(rows, columns=["strike", "type", "oi", "iv", "T", "symbol"])

    @staticmethod
    def _parse_symbol(symbol: str) -> Optional[dict]:
        """Распарсить ``COIN-DDMMMYY-STRIKE-TYPE-CURRENCY`` → компоненты.

        Пример: ``BTC-12JUL26-58000-P-USDT`` →
        ``{coin: "BTC", expiry: "12JUL26", strike: 58000.0, type: "P", currency: "USDT"}``.

        Поддерживает дробные страйки (``XRP-13JUL26-1.12-C-USDT``).
        Возвращает ``None`` при несоответствии формату.
        """
        parts = symbol.split("-")
        if len(parts) != 5:
            return None
        coin, expiry, strike_str, type_str, currency = parts
        try:
            strike = float(strike_str)
        except ValueError:
            return None
        t = type_str.upper()
        if t not in ("C", "P"):
            return None
        return {
            "coin": coin.upper(),
            "expiry": expiry.upper(),
            "strike": strike,
            "type": t,
            "currency": currency.upper(),
        }

    # ------------------------------------------------------------------ #
    #  Spot из underlyingPrice
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_spot(raw_list: list[dict], coin: str) -> float:
        """Взять spot = ``underlyingPrice`` (единая цена монеты для всех контрактов).

        Берёт медиану по всем контрактам для робастности (значения должны
        совпадать, но при асинхронных обновлениях возможен разброс).
        """
        prices = [
            p for p in (_to_float(item.get("underlyingPrice")) for item in raw_list)
            if p and p > 0
        ]
        if not prices:
            raise ValueError(
                f"Не удалось получить underlyingPrice (spot) для {coin} от Bybit."
            )
        return float(np.median(prices))

    # ------------------------------------------------------------------ #
    #  Ограничение по числу экспираций
    # ------------------------------------------------------------------ #
    @staticmethod
    def _limit_expiries(chain: pd.DataFrame, max_expiries: int) -> pd.DataFrame:
        """Оставить только ``max_expiries`` ближайших экспираций (по T)."""
        if max_expiries <= 0 or chain.empty:
            return chain
        unique_T = sorted(chain["T"].unique())
        keep = set(unique_T[:max_expiries])
        return chain[chain["T"].isin(keep)].reset_index(drop=True)


# ====================================================================== #
#  Вспомогательное
# ====================================================================== #
def _to_float(value) -> Optional[float]:
    """Безопасно привести строку/число к float; ``None`` при неудаче.

    Bybit отдаёт все числовые поля как строки (напр. ``"0.4608"``, ``"0"``).
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
