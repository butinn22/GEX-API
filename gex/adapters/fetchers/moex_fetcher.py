"""Polling опционных данных с Московской биржи (MOEX ISS).

Модуль получает опционные цепочки на фьючерсы на индексы **RTS** и **MIX**,
а также на валютные пары **CNYRUB** (CNY) и **USDRUB** (Si) через публичный
ISS-API::

    https://iss.moex.com/iss/engines/futures/markets/options/securities.json

и преобразует их в канонический :class:`OptionSnapshot`, совместимый с
существующим GEX pipeline (столбцы ``strike,type,oi,iv,T``).

Специфика MOEX
--------------
* Опционы — маржируемые **американские** опционы на фьючерсные контракты
  (исполнение в пункт). Базис для GEX — цена **фьючерса** (поле
  ``UNDERLYINGSETTLEPRICE``), а не индекса: именно по ней клирингует вариационная
  маржа, именно её используют маркетмейкеры для хеджа.
* ISS **не отдаёт implied volatility**. По согласованной конвенции проекта
  всем контрактам присваивается плоская IV из индекса **RVI** (Russian
  Volatility Index, ближайший аналог VIX для рынка RTS), с поправкой на срок
  до экспирации через квадратный корень времени::

      iv_i = RVI/100 * sqrt(T_i / T_RVI)

  где ``T_RVI`` — горизонт RVI (30 календарных дней = 30/365 лет). Для MIX
  RVI используется как прокси (отдельного VIX-индекса нет).

Котировки
~~~~~~~~~
ISS отдаёт несколько ценовых полей; приоритет выбора цены (если когда-либо
потребуется считать IV из котировок, а не из RVI):

    LAST (последняя сделка) → SETTLEPRICE (клина) → PREVPRICE

Сейчас для GEX цена опциона не нужна — IV берётся из RVI.

Контрактные множители
~~~~~~~~~~~~~~~~~~~~~
Все поддерживаемые фьючерсы (RTS, MIX, CNY, Si) торгуются в пунктах.
``per_contract`` — масштабный коэффициент GEX: он не влияет на положения
стен, Gamma Flip и режим, только на абсолютную величину Net GEX.
Для RTS/MIX — индекс×100 (``100``), для CNYRUB — лот 1000 (``1000``),
для USDRUB/Si — лот в пунктах (``1``). Конкретные значения задаются в
:data:`_INSTRUMENTS`/:data:`_MOEX_ASSETS`.

Отбор контрактов
~~~~~~~~~~~~~~~~
Для валютных CNY/Si на ISS есть два рода опционов: маржируемые на фьючерс
(``UNDERLYINGTYPE='F'``, underlying ``CRU6``/``SiU6``) и премиальные на
спот-курс (``UNDERLYINGTYPE='C'``, underlying ``CNYRUB_TOM``/``USD000UTSTOM``).
Берутся **только** фьючерсные — у них есть ``UNDERLYINGSETTLEPRICE`` (цена
фьючерса = spot для pipeline), и конвенция GEX едина для всех инструментов.

Источники данных и ключи
------------------------
securities:   SECID, ASSETCODE, UNDERLYINGTYPE, OPTIONTYPE, STRIKE,
              LASTTRADEDATE, UNDERLYINGASSET, UNDERLYINGSETTLEPRICE
marketdata:   SECID, OPENPOSITION (OI), LAST, SETTLEPRICE

Пример::

    fetcher = MOEXOptionsFetcher()
    snap = fetcher.fetch("RTS")      # OptionSnapshot c цепочкой RTS
    snap = fetcher.fetch("MIX")      # OptionSnapshot c цепочкой MIX
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from gex.domain.data_loader import OptionSnapshot
from gex.adapters.cache.redis_client import RedisClient, deserialize_value
from gex.adapters.cache.keys import PROVIDER_MOEX, chain_key

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Константы инструментов
# ====================================================================== #
# Поддерживаемые базовые активы (ASSETCODE на MOEX) и их конфигурация.
#   * iv_index — тикер VIX-подобного индекса для плоской волатильности.
#     RTS использует RVI (native); MIX/CNY/Si не имеют собственного VIX,
#     поэтому RVI берётся как робастный прокси рублевой волатильности.
#     (MOEX ISS не отдаёт implied volatility ни по одному инструменту —
#     приложения option-calc/optionboard на ISS отсутствуют.)
#   * per_contract — масштаб GEX (на положения уровней не влияет).
#   * underlying_type — тип базиса для отбора контрактов. Для индексных RTS/MIX
#     это 'F' (фьючерс). Для валютных CNY/Si на ISS есть два рода опционов —
#     маржируемые на фьючерс ('F', underlying CRU6/SiU6) и премиальные на спот-курс
#     ('C', underlying CNYRUB_TOM/USD000UTSTOM). Берём только 'F' — у них есть
#     UNDERLYINGSETTLEPRICE (цена фьючерса как spot), и конвенция GEX едина для всех.
#   * quote_scale — делитель для приведения цепочки опционов на ФЬЮЧЕРС к шкале
#     цены инструмента в приложении (свечи/спот). Для SI (USDRUB) цена фьючерса
#     в пунктах = курс × 1000 (контракт 1000 USD): Si ≈ 86 500 при USDRUB 86.5 →
#     quote_scale=1000 (курс). CNY/CR (CNYRUB) котируется самим курсом (~12.9)
#     и совпадает со свечами CNYRUBF → quote_scale=1. RTS/MIX — индексные
#     фьючерсы, своя шкала, совпадающая со свечами → 1.
#   * assetcode — ASSETCODE опционов на ISS (по умолчанию = ключ; для акций,
#     чей фьючерс имеет иной код, задаётся явно: SBER→SBRF, GAZP→GAZR).
_INSTRUMENTS: dict[str, dict] = {
    "RTS": {"iv_index": "RVI", "iv_horizon_days": 30, "per_contract": 100, "underlying_type": "F", "quote_scale": 1.0, "assetcode": "RTS"},
    "MIX": {"iv_index": "RVI", "iv_horizon_days": 30, "per_contract": 100, "underlying_type": "F", "quote_scale": 1.0, "assetcode": "MIX"},
    "CNY": {"iv_index": "RVI", "iv_horizon_days": 30, "per_contract": 1000, "underlying_type": "F", "quote_scale": 1.0, "assetcode": "CNY"},
    "SI":  {"iv_index": "RVI", "iv_horizon_days": 30, "per_contract": 1,   "underlying_type": "F", "quote_scale": 1000.0, "assetcode": "Si"},
}

# ── Опционы FORTS на ФЬЮЧЕРСЫ российских АКЦИЙ (ISS: ASSETCODE 'F') ─────────
# Ключ = тикер акции, как его знает приложение (свечи /ohlcv, сканеры);
# assetcode = ASSETCODE опционов на фьючерс этой акции (проверено на ISS
# 2026-09-07); lot = число акций в контракте → цена фьючерса в пунктах =
# цена акции × lot, поэтому quote_scale = lot приводит страйки/спот к шкале
# акции (совпадает со свечами). per_contract — калибровочный масштаб GEX.
_STOCK_FUTURE_OPTIONS: dict[str, dict] = {
    "GAZP":  {"assetcode": "GAZR", "lot": 100},
    "GMKN":  {"assetcode": "GMKN", "lot": 10},
    "LKOH":  {"assetcode": "LKOH", "lot": 10},
    "MOEX":  {"assetcode": "MOEX", "lot": 100},
    "ROSN":  {"assetcode": "ROSN", "lot": 100},
    "SBER":  {"assetcode": "SBRF", "lot": 100},
    "SNGS":  {"assetcode": "SNGR", "lot": 1000},
    "SNGSP": {"assetcode": "SNGP", "lot": 1000},
    "TATN":  {"assetcode": "TATN", "lot": 100},
    "VKCO":  {"assetcode": "VKCO", "lot": 10},
    "VTBR":  {"assetcode": "VTBR", "lot": 100},
}
_INSTRUMENTS.update({
    code: {
        "iv_index": "RVI", "iv_horizon_days": 30, "underlying_type": "F",
        "assetcode": meta["assetcode"], "lot": meta["lot"],
        "quote_scale": float(meta["lot"]), "per_contract": 100,
        "note": f"опционы на фьючерс акции (ASSETCODE {meta['assetcode']}, лот {meta['lot']})",
    }
    for code, meta in _STOCK_FUTURE_OPTIONS.items()
})

# ── Еженедельные опционы MOEX на сами АКЦИИ (ASSETCODE 'S') ─────────────────
# Базис — акция: UNDERLYINGSETTLEPRICE = цена акции в рублях (шкала 1:1 со
# свечами /ohlcv) → quote_scale=1. Недельные серии — удобны для конуса
# вероятностей (/gexcone). Список проверен на ISS 2026-09-07 (OI>0, USP>0).
# Приоритет у F-опционов (глубже цепочка) — но AFLT/ALRS/NLMK переведены на S:
# их F-цепочки неликвидны в ближних сериях (OI на дальних), S даёт недельные
# серии с OI. PLZL исключён: S-опционы в «фьючерсной» шкале (страйки ≈ цена×10).
_STOCK_SPOT_OPTIONS: dict[str, dict] = {
    "AFKS": {"assetcode": "AFKS"},
    "AFLT": {"assetcode": "AFLT"},
    "ALRS": {"assetcode": "ALRS"},
    "CHMF": {"assetcode": "CHMF"},
    "IRAO": {"assetcode": "IRAO"},
    "MAGN": {"assetcode": "MAGN"},
    "MSNG": {"assetcode": "MSNG"},
    "NLMK": {"assetcode": "NLMK"},
    "OZON": {"assetcode": "OZON"},
    "POSI": {"assetcode": "POSI"},
    "RTKM": {"assetcode": "RTKM"},
    "RUAL": {"assetcode": "RUAL"},
    "SVCB": {"assetcode": "SVCB"},
    "YDEX": {"assetcode": "YDEX"},
}
_INSTRUMENTS.update({
    code: {
        "iv_index": "RVI", "iv_horizon_days": 30, "underlying_type": "S",
        "assetcode": meta["assetcode"],
        "quote_scale": 1.0, "per_contract": 100,
        "note": "еженедельные опционы MOEX на акцию (ASSETCODE 'S')",
    }
    for code, meta in _STOCK_SPOT_OPTIONS.items()
})

# ISS endpoint: опционный рынок FORTS.
_ISS_OPTIONS_URL = (
    "https://iss.moex.com/iss/engines/futures/markets/options/securities.json"
    "?iss.meta=off&iss.only=securities,marketdata"
)
# ISS endpoint: значение индекса волатильности RVI (stock/index engine).
_ISS_RVI_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/index/securities/RVI.json"
    "?iss.meta=off&iss.only=marketdata"
)

# Fallback-вола (годовых), если RVI недоступен (внебиржевое время / сбой ISS).
# Периодически сверять с реальным средним RVI по RTS.
_DEFAULT_IV_FALLBACK = 0.40

# Срок жизни кэша HTTP-ответов ISS, секунды (5 минут).
# В течение этого окна повторные вызовы ручек /moex/gex/{asset} отдают
# закэшированный в памяти ответ ISS, не делая нового сетевого запроса.
# По истечении TTL флаг _fresh сбрасывается, и следующий запрос снова идёт к ISS.
_ISS_CACHE_TTL_SECONDS = 5 * 60


class MOEXOptionsFetcher:
    """Получение свежих опционных данных RTS/MIX/CNY/Si через MOEX ISS.

    HTTP-ответы ISS кэшируются в памяти процесса на :data:`_ISS_CACHE_TTL_SECONDS`
    (5 минут): в пределах TTL повторные вызовы (в т.ч. из разных ручек
    ``/moex/gex/{asset}`` и ``/moex/gex/{asset}/profile`` и для разных активов)
    переиспользуют один и тот же сетевой ответ и не делают новый запрос к ISS.
    Один ответ опционного рынка содержит данные по всем активам сразу, поэтому
    кэш ведётся общий, а не по активу. По истечении TTL запись инвалидируется,
    и следующий запрос снова идёт к ISS.

    Кэш — атрибуты уровня класса, поэтому он разделяется между всеми экземплярами
    fetcher'а (сервис создаёт новый экземпляр на каждый вызов ручки). Доступ
    защищён блокировкой для безопасности при конкурентных запросах.

    Parameters
    ----------
    timeout : float
        Таймаут HTTP-запроса к ISS, секунды.
    max_expiries : int
        Ограничение на число ближайших экспираций (по дате LASTTRADEDATE),
        попадающих в цепочку. ``0`` = без ограничения (все экспирации).
    iv_fallback : float
        Плоская годовая волатильность, если RVI получить не удалось.
    """

    # --- In-memory TTL-кэш ISS-ответов (общий для всех экземпляров) --- #
    # Сетевой ответ опционного рынка нужен одновременно для всех активов
    # (RTS/MIX/CNY/Si), поэтому кэшируется целиком, а не по активу. RVI —
    # общий индекс, тоже кэшируется один.
    _cache_lock = threading.Lock()
    _cache_options: tuple[float, tuple[list[dict], dict[str, dict]]] | None = None
    _cache_rvi: tuple[float, float] | None = None

    def __init__(
        self,
        timeout: float = 60.0,
        max_expiries: int = 0,
        iv_fallback: float = _DEFAULT_IV_FALLBACK,
        redis_client: Optional[RedisClient] = None,
    ):
        if timeout <= 0:
            raise ValueError("timeout должен быть > 0")
        if max_expiries < 0:
            raise ValueError("max_expiries должен быть >= 0")
        if iv_fallback <= 0:
            raise ValueError("iv_fallback должен быть > 0")
        self.timeout = float(timeout)
        self.max_expiries = int(max_expiries)
        self.iv_fallback = float(iv_fallback)
        self._redis = redis_client

    # ------------------------------------------------------------------ #
    #  Главный API
    # ------------------------------------------------------------------ #
    def fetch(self, asset: str) -> OptionSnapshot:
        """Polling: получить свежую опционную цепочку RTS/MIX/CNY/Si → OptionSnapshot.

        Parameters
        ----------
        asset : str
            Код базового актива MOEX: ``"RTS"``, ``"MIX"``, ``"CNY"`` (CNYRUB),
            ``"Si"`` (USDRUB) или тикер российской акции с опционами FORTS на её
            фьючерс (``LKOH``/``ALRS``/``SBER``/``GAZP``/``TATN``/... — см.
            :data:`_INSTRUMENTS`). Регистронезависимо.

        Returns
        -------
        OptionSnapshot
            Канонический снапшот (``strike,type,oi,iv,T``), готовый для
            GEX pipeline. ``spot`` = цена фьючерса (``UNDERLYINGSETTLEPRICE``
            ближайшей серии), приведённая к шкале цены инструмента
            (``quote_scale``); ``symbol`` = внешний код актива.

        Raises
        ------
        ValueError
            Если ``asset`` не поддерживается, данных нет или цепочка пуста
            после фильтрации.
        RuntimeError
            При сетевых ошибках ISS.
        """
        asset_code = asset.strip().upper()
        if asset_code not in _INSTRUMENTS:
            raise ValueError(
                f"Неподдерживаемый актив '{asset_code}'. "
                f"Доступны: {list(_INSTRUMENTS)}."
            )
        cfg = _INSTRUMENTS[asset_code]

        logger.info("MOEX ISS polling: asset=%s, max_expiries=%d",
                    asset_code, self.max_expiries)

        # ── Redis cache check ──
        redis_key = chain_key(asset_code, self.max_expiries, provider=PROVIDER_MOEX)
        if self._redis is not None and self._redis.connected:
            cached_data = self._redis.get(redis_key)
            if cached_data is not None:
                try:
                    result = deserialize_value(cached_data)
                    if isinstance(result, OptionSnapshot):
                        logger.info("  MOEX chain CACHE HIT for %s", asset_code)
                        return result
                except Exception:
                    logger.debug("MOEX deserialize error for %s — refetching", asset_code)

        # --- 1. Грузим опционный рынок (один запрос, фильтр по ASSETCODE) ---
        securities, marketdata = self._fetch_options()
        chain = self._build_chain(
            cfg["assetcode"], securities, marketdata,
            underlying_type=cfg["underlying_type"],
        )

        if chain.empty:
            raise ValueError(
                f"Не удалось построить цепочку для '{asset_code}' — нет "
                "контрактов с OI > 0 и корректной ценой фьючерса."
            )

        # --- 2. Ограничиваем число ближайших экспираций ---
        if self.max_expiries > 0:
            chain = self._limit_expiries(chain, self.max_expiries)

        # --- 3. Плоская IV из RVI (с поправкой на срок) ---
        iv_base = self._fetch_rvi(cfg["iv_index"])
        if iv_base is None:
            logger.warning(
                "RVI недоступен — используется fallback IV=%.3f для %s",
                self.iv_fallback, asset_code,
            )
            iv_base = self.iv_fallback
        iv_horizon_years = cfg["iv_horizon_days"] / 365.0
        # Терм-структура через sqrt(T): короткие экспирации — выше вола.
        chain["iv"] = iv_base * np.sqrt(chain["T"] / iv_horizon_years)
        # Греки требуют 0 < iv < ... (фильтр pipeline: iv > 0).
        chain["iv"] = chain["iv"].clip(lower=1e-4)

        # --- 4. Spot = цена фьючерса (ближайшей по экспирации серии) ---
        spot = self._pick_futures_spot(
            cfg["assetcode"], securities, underlying_type=cfg["underlying_type"]
        )

        # --- 4b. Валютные пары SI/CNY: пункты фьючерса → курс валюты.
        # Цепочка опционов торгуется на фьючерс (цена = курс × контракт 1000),
        # а цена инструмента во всём приложении — это курс (USDRUBF/CNYRUBF).
        # Приводим страйки и spot к шкале курса, чтобы стены/GEX-уровни
        # совпадали со свечами и ценой инструмента (см. quote_scale выше).
        quote_scale = float(cfg.get("quote_scale", 1.0))
        if quote_scale != 1.0:
            chain = chain.assign(strike=chain["strike"] / quote_scale)
            spot = spot / quote_scale

        # --- 5. Сборка OptionSnapshot (в обход GEXDataLoader, чтобы
        # сохранить разбивку по экспирациям: loader схлопывает одинаковые
        # (strike, type) с разным T, теряя информацию о сроках). ---
        logger.info(
            "  собрано %d строк (%d страйков, %d экспираций), "
            "spot=%s, IV(RVI)=%.3f%s",
            len(chain), chain["strike"].nunique(),
            chain["T"].round(5).nunique(),
            f"{spot:,.2f}" if quote_scale != 1.0 else f"{spot:.0f}",
            iv_base,
            f" (÷{quote_scale:g} к курсу)" if quote_scale != 1.0 else "",
        )

        snapshot = OptionSnapshot(
            symbol=asset_code,
            spot=spot,
            as_of=datetime.now(timezone.utc),
            chain=chain.reset_index(drop=True),
            meta={
                "source": "MOEX ISS",
                "iv_source": cfg["iv_index"],
                "iv_base": float(iv_base),
                "per_contract": cfg["per_contract"],
                "quote_scale": quote_scale,
                "max_expiries": self.max_expiries,
            },
        )

        # ── Сохраняем в Redis ──
        if self._redis is not None and self._redis.connected:
            self._redis.set(redis_key, snapshot, ex=600)

        return snapshot

    # ------------------------------------------------------------------ #
    #  Загрузка ISS
    # ------------------------------------------------------------------ #
    def _fetch_options(self) -> tuple[list[dict], dict[str, dict]]:
        """Запросить опционный рынок и распарсить securities + marketdata.

        Результат кэшируется в памяти на :data:`_ISS_CACHE_TTL_SECONDS`
        (5 минут): в пределах TTL повторные вызовы отдают закэшированный ответ
        без сетевого запроса. Кэш общий (один ответ — для всех активов).

        Returns
        -------
        (securities_rows, marketdata_by_secid)
        """
        # 1. Сначала проверяем кэш (без блокировки — быстрый путь).
        cached = self._cache_options
        if cached is not None:
            cached_at, payload = cached
            if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                logger.info("  ISS: опционный рынок из кэша (TTL=%.0fs)",
                            _ISS_CACHE_TTL_SECONDS)
                return payload
            # TTL истёк — запись «протухла», нужен свежий запрос.

        # 2. Берём блокировку, чтобы избежать дублирующих параллельных
        #    запросов (thundering herd), и перепроверяем кэш внутри неё —
        #    другой поток мог уже обновить его, пока мы ждали.
        with self._cache_lock:
            cached = self._cache_options
            if cached is not None:
                cached_at, payload = cached
                if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                    logger.info("  ISS: опционный рынок из кэша (TTL=%.0fs)",
                                _ISS_CACHE_TTL_SECONDS)
                    return payload

            logger.info("  ISS: запрос опционного рынка (кэш пуст/истёк)")
            payload = self._fetch_options_network()
            # Пишем в атрибут КЛАССА (не экземпляра) — service создаёт новый
            # fetcher на каждый вызов ручки, поэтому кэш должен жить на классе,
            # чтобы повторные запросы переиспользовали ответ.
            MOEXOptionsFetcher._cache_options = (time.monotonic(), payload)
            return payload

    @staticmethod
    def _fetch_options_network() -> tuple[list[dict], dict[str, dict]]:
        """Живой сетевой запрос опционного рынка ISS → распарсенный payload."""
        import requests

        try:
            resp = requests.get(
                _ISS_OPTIONS_URL,
                timeout=60.0,
                headers={"User-Agent": "gex-app/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ошибка запроса опционов MOEX ISS: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Некорректный JSON MOEX ISS: {exc}") from exc

        sec = data.get("securities", {})
        md = data.get("marketdata", {})
        s_cols = sec.get("columns", [])
        s_data = sec.get("data", [])
        m_cols = md.get("columns", [])
        m_data = md.get("data", [])

        securities_rows = [dict(zip(s_cols, row)) for row in s_data]
        marketdata_by_secid = {
            row[0]: dict(zip(m_cols, row)) for row in m_data if row
        }
        logger.info("  ISS: %d контрактов опционного рынка загружено",
                    len(securities_rows))
        return securities_rows, marketdata_by_secid

    def _fetch_rvi(self, index_code: str) -> Optional[float]:
        """Получить значение индекса волатильности (RVI) → годовая доля.

        Результат кэшируется в памяти на :data:`_ISS_CACHE_TTL_SECONDS`
        (5 минут). Берётся поле ``CURRENTVALUE`` (в процентах годовых),
        переводится в доли (``/100``). При недоступности — ``None``.
        """
        # Быстрый путь — проверяем кэш без блокировки.
        cached = self._cache_rvi
        if cached is not None:
            cached_at, value = cached
            if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                logger.info("  RVI: из кэша (TTL=%.0fs)", _ISS_CACHE_TTL_SECONDS)
                return value

        with self._cache_lock:
            # Перепроверка внутри блокировки (двойная проверка с блокировкой).
            cached = self._cache_rvi
            if cached is not None:
                cached_at, value = cached
                if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                    logger.info("  RVI: из кэша (TTL=%.0fs)",
                                _ISS_CACHE_TTL_SECONDS)
                    return value

            logger.info("  RVI: запрос (кэш пуст/истёк)")
            value = self._fetch_rvi_network()
            # Кэшируем только успешный результат (не None), иначе при
            # временном сбое ISS на 5 минут остался бы «битый» None.
            # Пишем в атрибут КЛАССА — см. комментарий в _fetch_options.
            if value is not None:
                MOEXOptionsFetcher._cache_rvi = (time.monotonic(), value)
            return value

    @staticmethod
    def _fetch_rvi_network() -> Optional[float]:
        """Живой сетевой запрос RVI → годовая доль (или ``None`` при сбое)."""
        import requests

        try:
            resp = requests.get(
                _ISS_RVI_URL,
                timeout=60.0,
                headers={"User-Agent": "gex-app/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("RVI: запрос не удался — %s", exc)
            return None

        md = data.get("marketdata", {})
        cols = md.get("columns", [])
        rows = md.get("data", [])
        if not rows:
            return None
        rec = dict(zip(cols, rows[0]))
        # CURRENTVALUE — значение индекса на момент запроса (в процентах).
        val = rec.get("CURRENTVALUE")
        if val is None:
            # Клиринговое/предыдущее значение как fallback.
            val = rec.get("LASTCLOSE") or rec.get("CLOSE") or rec.get("LAST")
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val) or val <= 0:
            return None
        return val / 100.0

    # ------------------------------------------------------------------ #
    #  Сборка цепочки
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_chain(
        asset_code: str,
        securities: list[dict],
        marketdata: dict[str, dict],
        underlying_type: str = "F",
    ) -> pd.DataFrame:
        """Отфильтровать контракты по ASSETCODE и собрать каноническую цепочку.

        Критерии отбора строки:
          * ``ASSETCODE == asset_code``;
          * ``UNDERLYINGTYPE == underlying_type`` (по умолчанию ``'F'`` —
            только опционы на фьючерс; для валютных CNY/Si это отсекает
            премиальные опционы на спот-курс типа ``'C'``, у которых нет
            ``UNDERLYINGSETTLEPRICE`` иная конвенция расчёта);
          * ``OPTIONTYPE`` в {'C', 'P'};
          * ``STRIKE > 0`` и ``UNDERLYINGSETTLEPRICE > 0`` (нужен для spot);
          * OI (``OPENPOSITION``) > 0;
          * ``LASTTRADEDATE`` в будущем (T > 0).

        ``iv`` здесь НЕ заполняется — его проставляет :meth:`fetch` из RVI,
        потому что срок-поправка требует уже вычисленного T.
        """
        rows: list[dict] = []
        now = datetime.now(timezone.utc)
        asset_upper = asset_code.upper()
        for s in securities:
            # ASSETCODE на MOEX может быть смешанного регистра ("Si" для USDRUB),
            # поэтому сравниваем в верхнем регистре.
            if str(s.get("ASSETCODE", "")).upper() != asset_upper:
                continue
            # Отбор по типу базиса: 'F' = фьючерс, 'C' = валюта (спот-курс).
            if s.get("UNDERLYINGTYPE") != underlying_type:
                continue
            opt_type = s.get("OPTIONTYPE")
            if opt_type not in ("C", "P"):
                continue
            strike = s.get("STRIKE")
            fut_price = s.get("UNDERLYINGSETTLEPRICE")
            secid = s.get("SECID")
            if not secid:
                continue
            md = marketdata.get(secid, {})
            oi = md.get("OPENPOSITION")
            ltd = s.get("LASTTRADEDATE")
            try:
                strike = float(strike)
                fut_price = float(fut_price)
                oi = float(oi) if oi is not None else 0.0
            except (TypeError, ValueError):
                continue
            if not (strike > 0 and fut_price > 0 and oi > 0):
                continue
            T = _lasttrade_to_years(ltd, now)
            if T is None or T <= 0:
                continue
            rows.append({
                "strike": strike,
                "type": opt_type,
                "oi": oi,
                "iv": np.nan,  # проставляется из RVI в fetch()
                "T": T,
                # Метаданные для spot и отладки (в pipeline не используются).
                "futures_price": fut_price,
                "underlying": s.get("UNDERLYINGASSET"),
                "lasttrade": ltd,
            })
        if not rows:
            return pd.DataFrame(columns=["strike", "type", "oi", "iv", "T"])
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Ограничение числа экспираций
    # ------------------------------------------------------------------ #
    @staticmethod
    def _limit_expiries(chain: pd.DataFrame, max_expiries: int) -> pd.DataFrame:
        """Оставить ``max_expiries`` ближайших по дате экспираций."""
        # Дата экспирации = as_of + T. Берём T (он уже отсортирован по сроку).
        unique_T = sorted(chain["T"].unique())
        keep = set(unique_T[:max_expiries])
        return chain[chain["T"].isin(keep)].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Spot — цена фьючерса ближайшей серии
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_futures_spot(
        asset_code: str,
        securities: list[dict],
        underlying_type: str = "F",
    ) -> float:
        """Выбрать цену фьючерса для spot.

        Стратегия: среди контрактов ASSETCODE (с типом базиса ``underlying_type``)
        берём ближайшую по экспирации серию (минимальная LASTTRADEDATE) и её
        ``UNDERLYINGSETTLEPRICE``. Все опционы одной серии ссылаются на один
        фьючерс — цена едина.
        """
        candidates = []
        asset_upper = asset_code.upper()
        for s in securities:
            if str(s.get("ASSETCODE", "")).upper() != asset_upper:
                continue
            if s.get("UNDERLYINGTYPE") != underlying_type:
                continue
            ltd = s.get("LASTTRADEDATE")
            fut_price = s.get("UNDERLYINGSETTLEPRICE")
            try:
                fut_price = float(fut_price)
            except (TypeError, ValueError):
                continue
            if fut_price > 0 and ltd:
                candidates.append((ltd, fut_price))
        if not candidates:
            raise ValueError(
                f"Не удалось определить цену фьючерса (spot) для '{asset_code}'."
            )
        # Ближайшая экспирация (лексикографически 'YYYY-MM-DD' = по дате).
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]


# ====================================================================== #
#  Вспомогательные функции
# ====================================================================== #
def _lasttrade_to_years(lasttrade: object, now: datetime) -> Optional[float]:
    """Перевести LASTTRADEDATE ISS ('YYYY-MM-DD') во время до экспирации, лет.

    Возвращает ``None`` при неразборчивой дате. Т — календарное (для греков).
    """
    if not lasttrade:
        return None
    try:
        exp = datetime.strptime(str(lasttrade), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
    now_utc = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now
    days = (exp - now_utc).total_seconds() / 86400.0
    return days / 365.0
