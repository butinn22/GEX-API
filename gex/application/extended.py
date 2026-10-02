"""Расширенный GEX-анализ для крипты и акций США (yfinance + Bybit).

Модуль реализует **все** метрики Gamma Exposure, описанные в ТЗ-файле:

Раздел 1-2 — базовые расчёты
    * гамма опциона (модель Блэка-Шоулза);
    * GEX отдельного опциона ``= Γ × OI × per_contract × 100``;
    * агрегация по страйку: Call/Put/Net GEX, Aggregate Gamma (AG);
    * Zero Gamma Level (линейная интерполяция смены знака);
    * Call/Put Wall (макс |Net GEX| по знаку) + сила стены;
    * Power Zone (топ-10% по AG, группировка соседних страйков).

Раздел 3 — агрегация по экспирациям
    * временной вес ``e^(-дней/30)``;
    * взвешенное суммирование Call/Put/Net GEX.

Раздел 4 — метрики влияния на рынок
    * Gamma Dollar = ``Net GEX × Страйк × 0.01``;
    * Hedge Requirement (сценарии ±X%);
    * Directional Bias (знак Total Net GEX);
    * Put/Call Ratio по GEX.

Раздел 5 — Aggregate Gamma
    * Total AG, AG_normalized для area chart.

Раздел 6 — уровни поддержки/сопротивления
    * топ-5 страйков по |Net GEX| с типизацией.

Раздел 7 — дополнительные метрики
    * Gamma Exposure Score (0-100);
    * Delta Hedge Ratio (Call GEX / |Put GEX|);
    * Max Pain (теоретический).

Конвенция знаков дилера — SqueezeMetrics (equity): дилеры **покупают** коллы
(``call_sign=+1``) и **продают** путы (``put_sign=-1``). Это делает Put GEX
всегда отрицательным — как и зафиксировано в ТЗ. Для всех активов проекта
(US equity/ETF, крипта Bybit) используется единая конвенция.

Формула GEX строго следует ТЗ: ``GEX = Γ × OI × per_contract × 100``, где
``per_contract`` = 1 для крипты (1 контракт = 1 монета) и 100 для акций
(1 контракт = 100 акций), второе умножение на 100 — конвертация в доллары.

Пример::

    analyzer = ExtendedGEXAnalyzer()
    report = analyzer.analyze("BTC")        # автоопределение Bybit
    report = analyzer.analyze("AAPL")       # автоопределение yfinance
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from gex.domain.data_loader import OptionSnapshot
from gex.domain.greeks import bs_gamma, bs_delta
from gex.assets_config import DEFAULT_ASSETS, FUTURES_TICKERS
from gex.adapters.fetchers.bybit_fetcher import BybitOptionsFetcher, _CRYPTO_ASSETS

# ETF proxy mapping for indices without direct options (currently empty)
_YF_PROXY_MAP: dict[str, str] = {}

# Фьючерсы: параметры берутся из общего справочника (``assets_config.DEFAULT_ASSETS``),
# признак «это фьючерс» — из ``FUTURES_TICKERS``. Своя таблица здесь была третьей копией
# тех же чисел (``per_contract=100``, ``q=r``): значения совпадали, но только потому, что
# потребитель по умолчанию использовал ``r=0.045``. Правка в одном месте не доехала бы
# до остальных — и расхождение проявилось бы как разные конус и профиль по одному тикеру.

from gex.adapters.fetchers.yf_fetcher import YFOptionsFetcher
from gex.adapters.cache.redis_client import RedisClient

if TYPE_CHECKING:
    # Только для аннотаций: ``coverage`` в отчёте и ``_strikes_to_lite`` ссылаются на
    # типы AUTO-режима. Рантайм-импорт остаётся локальным в analyze_auto — здесь он нужен
    # лишь статическим анализаторам (pyflakes/r8), иначе forward-ref «undefined name».
    from gex.application.auto_scope import AutoCoverage, StrikeLite


# ====================================================================== #
#  Константы
# ====================================================================== #
# Поддерживаемые крипто-монеты берутся из bybit_fetcher._CRYPTO_ASSETS.
# Любой тикер вне этого множества трактуется как акция/ETF США (yfinance).
# Знаки дилера SqueezeMetrics (см. модульный docstring).
_CALL_SIGN: float = +1.0
_PUT_SIGN: float = -1.0

# Параметр временного затухания веса экспирации (ТЗ раздел 3.1).
# Чем меньше τ, тем сильнее ближние экспирации доминируют.
_EXPIRY_TAU_DAYS: float = 30.0

# Граница топ-% для Power Zone (ТЗ раздел 2.4).
_POWER_ZONE_TOP_QUANTILE: float = 0.10

# Максимальное расстояние между соседними страйками в зоне (ТЗ раздел 2.4).
# 5 «пунктов» — для акций/индексов это доллары, для крипты тоже абсолютные
# единицы цены. Делается относительным от spot (5% зоны), чтобы быть
# универсальным для BTC (~60000) и DOGE (~0.15).
_POWER_ZONE_MAX_GAP_PCT: float = 0.05

# Порог «нейтральности» Total Net GEX как доли от Total AG (ТЗ раздел 4.3).
# |Total Net GEX| / Total AG < порог → NEUTRAL.
_NEUTRAL_BIAS_THRESHOLD: float = 0.02


# ====================================================================== #
#  Dataclass-ы результата
# ====================================================================== #
@dataclass
class ExtendedStrike:
    """Aggregated GEX на одном страйке + производные метрики."""

    strike: float
    gex_call: float
    gex_put: float
    gex_net: float
    ag: float                      # Aggregate Gamma = |Call GEX| + |Put GEX|
    gamma_call: float              # сырая гамма (BSM), сумма по коллам
    gamma_put: float
    oi_call: float
    oi_put: float
    gamma_dollar: float            # Net GEX × strike × 0.01
    delta_hedge_ratio: Optional[float]   # Call GEX / |Put GEX|
    ag_normalized: float           # AG / max(AG) ∈ [0, 1]
    weight: float                  # суммарный временной вес экспираций страйка


@dataclass
class PowerZone:
    """Группа соседних страйков с экстремально высокой AG."""

    center: float                  # средневзвешенный страйк по AG
    width: float                   # max_strike - min_strike
    total_ag: float                # сумма AG страйков зоны
    dominant_type: str             # "CALL" | "PUT" — знак суммы Net GEX зоны
    n_strikes: int
    min_strike: float
    max_strike: float


@dataclass
class KeyLevel:
    """Уровень поддержки/сопротивления из топ-|Net GEX|."""

    strike: float
    type: str                      # "RESISTANCE" (Net GEX > 0) | "SUPPORT" (< 0)
    strength: float                # |Net GEX|
    distance_pct: float            # (strike - spot) / spot × 100


@dataclass
class HedgeScenario:
    """Сценарий движения цены → масштаб хеджирования."""

    scenario_pct: float            # +1.0, -1.0 и т.п.
    shares: float                  # сколько акций/монет дилерам нужно купить/продать
    dollar_value: float            # shares × spot (знак: +покупка / −продажа)


@dataclass
class VolumeZone:
    """Объём (OI) опционов в одной зоне денежности."""
    zone: str           # "ask_or_above" | "bid_or_below" | "between"
    label: str          # "Ask or above" | "Bid or below" | "Between market"
    call_oi: float
    put_oi: float
    call_pct: float     # % от общего OI
    put_pct: float


@dataclass
class VolumeProfile:
    """Профиль распределения OI по зонам денежности для donut chart."""
    spot: float
    atm_pct: float      # ширина ATM зоны в % от spot
    total_call_oi: float
    total_put_oi: float
    total_oi: float
    call_pct: float      # % calls от total
    put_pct: float       # % puts от total
    zones: list[VolumeZone]


# ====================================================================== #
#  ExtendedGEXReport
# ====================================================================== #
@dataclass
class ExtendedGEXReport:
    """Полный расширенный GEX-отчёт по одному активу."""

    # --- Идентификация ---
    symbol: str
    source: str                    # "crypto" (Bybit) | "stock" (yfinance)
    spot: float
    per_contract: int

    # --- Профиль по страйкам ---
    per_strike: list[ExtendedStrike]
    net_gex: float                 # Total Net GEX
    total_call_gex: float
    total_put_gex: float
    total_ag: float                # Σ AG
    regime: str                    # "POSITIVE" | "NEGATIVE"
    directional_bias: str          # "BULLISH" | "BEARISH" | "NEUTRAL"

    # --- Ключевые уровни ---
    zero_gamma: Optional[float]    # уровень нулевой гаммы (линейная интерполяция)
    call_wall: float
    call_wall_strength: float      # |Net GEX| / Σ|Net GEX|
    put_wall: float
    put_wall_strength: float
    power_zones: list[PowerZone]
    key_levels: list[KeyLevel]     # топ-5 по |Net GEX|

    # --- Метрики влияния ---
    gamma_dollar_total: float      # Σ |Net GEX × strike × 0.01| (масштаб хеджа)
    put_call_ratio: float          # |Total Put GEX| / Total Call GEX
    gamma_exposure_score: float    # |Net GEX| / Total AG × 100 ∈ [0, 100]
    max_pain: Optional[float]
    hedge_scenarios: list[HedgeScenario] = field(default_factory=list)
    # Динамическое описание GEX-профиля (зоны, HH/HL/LH/LL, EMA). None, если
    # GEX-данных недостаточно (build_narrative сам решает по доступности).
    narrative: Optional["object"] = None
    # Доверительный интервал цены по ключевым GEX-уровням (±10% индексы /
    # ±15-20% акции/крипта). None, если страйков слишком мало для оценки.
    price_band: Optional["object"] = None
    # Распределение объёмов OI по зонам денежности (donut chart)
    volume_profile: Optional["VolumeProfile"] = None
    # Метаданные AUTO-режима (охват/эскалация/разреженность). None для ручных
    # запросов и для путей, где AUTO не применялся.
    coverage: Optional["AutoCoverage"] = None

    def summary(self) -> str:  # pragma: no cover - человекочитаемый дамп
        return (
            f"=== Extended GEX: {self.symbol} ({self.source}) spot={self.spot:.2f} ===\n"
            f"  Net GEX      : {self.net_gex:+,.0f}\n"
            f"  Regime       : {self.regime}\n"
            f"  Bias         : {self.directional_bias}\n"
            f"  Zero Gamma   : {self.zero_gamma:.2f}" if self.zero_gamma else "—" + "\n"
            f"  Call Wall    : {self.call_wall:.2f} (strength={self.call_wall_strength:.2%})\n"
            f"  Put Wall     : {self.put_wall:.2f} (strength={self.put_wall_strength:.2%})\n"
            f"  PCR          : {self.put_call_ratio:.2f}\n"
            f"  GEX Score    : {self.gamma_exposure_score:.1f}\n"
            f"  Max Pain     : {self.max_pain:.2f}" if self.max_pain else "—" + "\n"
            f"  Power zones  : {len(self.power_zones)}\n"
            f"  Strikes      : {len(self.per_strike)}\n"
        )


# ====================================================================== #
#  Анализатор
# ====================================================================== #
class ExtendedGEXAnalyzer:
    """Расчёт расширенного GEX-профиля по всем метрикам ТЗ.

    Parameters
    ----------
    r : float
        Безрисковая ставка (годовых). Для крипты 0.045 (USDT-базис),
        для акций берётся дефолт сервиса.
    call_sign, put_sign : float
        Знаки дилера (SqueezeMetrics по умолчанию).
    """

    def __init__(
        self,
        r: float = 0.045,
        q: float = 0.0,
        call_sign: float = _CALL_SIGN,
        put_sign: float = _PUT_SIGN,
        redis_client: Optional[RedisClient] = None,
    ):
        self.r = float(r)
        self.q = float(q)
        self.call_sign = float(call_sign)
        self.put_sign = float(put_sign)
        self._redis = redis_client

    # ------------------------------------------------------------------ #
    #  Точка входа: получить данные → построить отчёт
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        ticker: str,
        days: float = 30.0,
        max_expiries: int = 5,
        hedge_scenarios_pct: Optional[list[float]] = None,
        snapshot: Optional[OptionSnapshot] = None,
        source: str = "auto",
    ) -> ExtendedGEXReport:
        """Построить расширенный GEX-отчёт по тикеру.

        Автоопределение источника: если ``ticker`` ∈ поддерживаемым крипто-монетам
        (BTC/ETH/SOL/XRP/DOGE) — Bybit, иначе — yfinance.

        Parameters
        ----------
        ticker : str
            Тикер (BTC, ETH, AAPL, SPY, …).
        days : float
            Горизонт анализа в днях (фильтр по сроку экспирации).
        max_expiries : int
            Сколько ближайших экспираций загрузить.
        hedge_scenarios_pct : list[float], optional
            Сценарии движения цены для Hedge Requirement в %, напр. ``[-1, 1]``.
            По умолчанию ``[-1.0, 1.0]``.
        snapshot : OptionSnapshot, optional
            Уже готовая цепочка (для тестов / оффлайн-режима). Если задан —
            сетевые вызовы не выполняются, ``ticker`` берётся из снапшота.
        """
        if hedge_scenarios_pct is None:
            hedge_scenarios_pct = [-1.0, 1.0]

        # --- 1. Получение данных ---
        if source == "aggregated":
            # Режим удалён: он вызывал несуществующие методы (`_average_strikes`,
            # `_merge_key_levels`, `_average_hedge_scenarios`) и падал с AttributeError на этой же
            # строке. Отвечаем понятной ошибкой вместо 500 (BUG-AGG; аудит 03: EC-1).
            raise ValueError(
                "source='aggregated' больше не поддерживается: используйте 'auto'/'webull'/'yfinance'"
            )

        if snapshot is None:
            snapshot, source, per_contract, effective_q = self._fetch(
                ticker, max_expiries,
                force_source=source if source != "auto" else None,
                max_days=days,
            )
        else:
            source, per_contract, effective_q = self._classify(ticker, snapshot)
            ticker = snapshot.symbol

        # Шаги 2–12 вынесены в ``_assemble`` без изменения поведения: AUTO-режим
        # вызывает ту же сборку повторно на объединённой цепочке.
        return self._assemble(
            ticker, snapshot, days, per_contract, effective_q, source,
            hedge_scenarios_pct,
        )

    # ------------------------------------------------------------------ #
    #  Сборка отчёта из готовой цепочки (шаги 2–12 исходного analyze)
    # ------------------------------------------------------------------ #
    def _assemble(
        self,
        ticker: str,
        snapshot: OptionSnapshot,
        days: float,
        per_contract: int,
        effective_q: float,
        source: str,
        hedge_scenarios_pct: Optional[list[float]] = None,
    ) -> ExtendedGEXReport:
        """Построить :class:`ExtendedGEXReport` из готового ``OptionSnapshot``.

        Это точный вынос шагов 2–12 исходного ``analyze`` (фильтр по сроку →
        профиль → уровни → рыночные итоги → narrative/price_band/volume_profile),
        поведение не менялось. ``analyze_auto`` вызывает его дважды: на первичной
        цепочке и, при эскалации, на объединённой.
        """
        if hedge_scenarios_pct is None:
            hedge_scenarios_pct = [-1.0, 1.0]

        # --- 2. Фильтр по сроку экспирации (как в GEXService) ---
        filtered = self._filter_by_days(snapshot, days)
        chain = filtered.chain
        if chain.empty:
            raise ValueError(
                f"Цепочка '{ticker}' пуста после фильтрации по days={days}."
            )

        spot = float(filtered.spot)

        # --- 3. Построение профиля по страйкам ---
        per_strike_df = self._build_per_strike(chain, spot, per_contract, effective_q)
        if per_strike_df.empty:
            raise ValueError(f"Профиль '{ticker}' пуст: нет валидных опционов.")

        # --- 4. Ключевые уровни ---
        zero_gamma = self._zero_gamma_level(per_strike_df)
        call_wall, call_strength = self._wall(per_strike_df, positive=True)
        put_wall, put_strength = self._wall(per_strike_df, positive=False)
        power_zones = self._power_zones(per_strike_df, spot)

        # --- 5. Рыночные итоги ---
        net_gex = float(per_strike_df["gex_net"].sum())
        total_call = float(per_strike_df["gex_call"].sum())
        total_put = float(per_strike_df["gex_put"].sum())   # < 0
        total_ag = float(per_strike_df["ag"].sum())
        regime = "POSITIVE" if net_gex >= 0 else "NEGATIVE"
        bias = self._directional_bias(net_gex, total_ag)
        pcr = self._put_call_ratio(total_call, total_put)
        score = self._gamma_exposure_score(net_gex, total_ag)
        max_pain = self._max_pain(chain, spot)

        # --- 6. Уровни S/R (топ-5 по |Net GEX|) ---
        key_levels = self._key_levels(per_strike_df, spot, top_n=5)

        # --- 7. Gamma Dollar total (масштаб хеджирования) ---
        gamma_dollar_total = float(per_strike_df["gamma_dollar"].abs().sum())

        # --- 8. Hedge Requirement по сценариям ---
        hedge_scns = [
            self._hedge_requirement(chain, per_strike_df, spot, per_contract, pct)
            for pct in hedge_scenarios_pct
        ]

        # --- 9. Сборка ExtendedStrike-списка ---
        strikes_out = self._strikes_to_dataclass(per_strike_df)

        report = ExtendedGEXReport(
            symbol=ticker,
            source=source,
            spot=spot,
            per_contract=per_contract,
            per_strike=strikes_out,
            net_gex=net_gex,
            total_call_gex=total_call,
            total_put_gex=total_put,
            total_ag=total_ag,
            regime=regime,
            directional_bias=bias,
            zero_gamma=zero_gamma,
            call_wall=call_wall,
            call_wall_strength=call_strength,
            put_wall=put_wall,
            put_wall_strength=put_strength,
            power_zones=power_zones,
            key_levels=key_levels,
            gamma_dollar_total=gamma_dollar_total,
            put_call_ratio=pcr,
            gamma_exposure_score=score,
            max_pain=max_pain,
            hedge_scenarios=hedge_scns,
        )

        # --- 10. Динамическое описание (narrative) — graceful: не роняет анализ.
        try:
            from gex.application.narrative import build_narrative
            report.narrative = build_narrative(report, ticker)
        except Exception as exc:  # noqa: BLE001
            logger.debug("narrative для %s не построен: %s", ticker, exc)
            report.narrative = None

        # --- 11. Доверительный интервал цены по ключевым GEX-уровням.
        try:
            from gex.domain.price_band import compute_price_band
            report.price_band = compute_price_band(report)
        except Exception as exc:  # noqa: BLE001
            logger.debug("price_band для %s не построен: %s", ticker, exc)
            report.price_band = None

        # --- 12. Volume Profile (OI distribution by moneyness zone).
        try:
            report.volume_profile = self._compute_volume_profile(chain, spot)
        except Exception as exc:  # noqa: BLE001
            logger.debug("volume_profile для %s не построен: %s", ticker, exc)
            report.volume_profile = None

        # --- 13. Метаданные охвата (ВСЕГДА, не только AUTO) ---
        # Аудит 2026-09-17: качество выборки должно быть видно и в ручном
        # режиме — профиль на 1 экспирации и на 8 выглядят одинаково
        # убедительно, если не показывать, из чего они собраны. AUTO позже
        # перезапишет блок полноценными метаданными (эскалация/резерв/время).
        report.coverage = _build_coverage(
            days=days, source=source, report=report, chain=chain,
        )

        return report

    # ------------------------------------------------------------------ #
    #  AUTO-режим: максимальный охват + эскалация + метаданные охвата
    # ------------------------------------------------------------------ #
    def analyze_auto(
        self,
        ticker: str,
        source: str = "auto",
        hedge_scenarios_pct: Optional[list[float]] = None,
        max_days: float = 90.0,
        max_expiries: int = 20,
    ) -> ExtendedGEXReport:
        """Построить отчёт в режиме AUTO (design §5.2).

        Логика:
        1. Первичная загрузка в **наибольшем** охвате (``max_days``/``max_expiries``).
           ``source`` остаётся первичным источником (``auto`` → существующая цепочка
           ``_fetch``: webull→yfinance).
        2. Сборка отчёта и оценка разреженности (:func:`detect_sparse`).
        3. Если профиль разрежен **и** есть fallback-источник —
           **ровно одна** дополнительная загрузка fallback'а, объединение цепочек
           без двойного счёта (:func:`merge_chains`) и повторная сборка.
        4. Возврат отчёта с ``coverage``.

        Ошибка fallback-загрузки **не фатальна**: возвращаем первичный отчёт с
        ``partial=True``. AUTO никогда не превращает рабочий результат в ошибку.
        """
        import time as _time
        from dataclasses import replace as _replace

        from gex.application.auto_scope import (
            AutoCoverage,
            StrikeLite,
            count_expiries,
            detect_sparse,
            fallback_source_for,
            merge_chains,
        )

        if hedge_scenarios_pct is None:
            hedge_scenarios_pct = [-1.0, 1.0]

        if source == "aggregated":
            raise ValueError(
                "source='aggregated' больше не поддерживается: используйте 'auto'/'webull'/'yfinance'"
            )

        t0 = _time.perf_counter()

        # --- 1. Первичная загрузка в максимальном охвате ---
        snapshot, primary_source, per_contract, effective_q = self._fetch(
            ticker, max_expiries,
            force_source=source if source != "auto" else None,
            max_days=max_days,
        )
        symbol = snapshot.symbol

        # --- 2. Сборка + оценка разреженности первичного профиля ---
        report = self._assemble(
            symbol, snapshot, max_days, per_contract, effective_q,
            primary_source, hedge_scenarios_pct,
        )
        final_reasons = detect_sparse(
            _strikes_to_lite(report.per_strike), report.spot,
            report.call_wall, report.put_wall,
            expiries=count_expiries(snapshot.chain),
        )

        # --- 3. Эскалация: ровно один fallback-фетч, без циклов ---
        fallback = fallback_source_for(primary_source, symbol) if final_reasons else None
        escalated = fallback is not None
        fallback_used = False
        partial = False
        final_chain = snapshot.chain

        if fallback is not None:
            try:
                fb_snapshot, _fb_source, _fb_pc, _fb_q = self._fetch(
                    symbol, max_expiries, force_source=fallback, max_days=max_days,
                )
                merged_chain = merge_chains(snapshot.chain, fb_snapshot.chain)
                # spot/symbol всегда первичные: replace сохраняет их.
                merged_snapshot = _replace(snapshot, chain=merged_chain)
                report = self._assemble(
                    symbol, merged_snapshot, max_days, per_contract, effective_q,
                    primary_source, hedge_scenarios_pct,
                )
                final_chain = merged_chain
                fallback_used = True
                # sparse должен отражать ФИНАЛЬНЫЙ профиль.
                final_reasons = detect_sparse(
                    _strikes_to_lite(report.per_strike), report.spot,
                    report.call_wall, report.put_wall,
                    expiries=count_expiries(merged_chain),
                )
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "AUTO: fallback-загрузка '%s' для %s не удалась (%s) — "
                    "возвращаю первичный результат (partial)",
                    fallback, symbol, exc,
                )
                partial = True

        # --- 4. Метаданные охвата ---
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)
        sources_used = [primary_source] + ([fallback] if fallback_used else [])
        strike_min, strike_max, nearest_expiry_days = _coverage_strike_meta(final_chain)
        furthest_expiry_days: Optional[float] = None
        if final_chain is not None and len(final_chain) > 0:
            try:
                furthest_expiry_days = float(
                    (final_chain["T"].astype(float) * 365.0).max()
                )
            except (TypeError, ValueError):
                furthest_expiry_days = None
        report.coverage = AutoCoverage(
            mode="auto",
            resolved_days=float(max_days),
            resolved_expiries=int(max_expiries),
            sources_used=sources_used,
            primary_source=primary_source,
            fallback_used=fallback_used,
            escalated=escalated,
            partial=partial,
            expirations_merged=count_expiries(final_chain),
            strike_count=len(report.per_strike),
            total_oi=float(sum(s.oi_call + s.oi_put for s in report.per_strike)),
            sparse=bool(final_reasons),
            sparse_reasons=list(final_reasons),
            elapsed_ms=elapsed_ms,
            strike_min=strike_min, strike_max=strike_max,
            nearest_expiry_days=nearest_expiry_days,
            furthest_expiry_days=furthest_expiry_days,
        )
        return report

    # ------------------------------------------------------------------ #
    #  Источник данных
    # ------------------------------------------------------------------ #
    def _classify(self, ticker: str, snapshot) -> tuple[str, int, float]:
        """
        Классифицировать источник/множитель по тикеру для готового снапшота.
        Возвращает (source, per_contract, effective_q) — согласовано с _fetch.
        """
        sym = ticker.strip().upper()
        if sym in FUTURES_TICKERS:
            cfg = DEFAULT_ASSETS[sym]
            return ("futures", int(cfg["per_contract"]), float(cfg["q"]))
        if sym in _CRYPTO_ASSETS:
            cfg = _CRYPTO_ASSETS[sym]
            return ("crypto", int(cfg["per_contract"]), 0.0)
        return ("stock", 100, self.q)

    def _fetch(
        self,
        ticker: str,
        max_expiries: int,
        force_source: Optional[str] = None,
        max_days: Optional[float] = None,
    ) -> tuple[OptionSnapshot, str, int, float]:
        """Получить OptionSnapshot.

        Args:
            ticker: Тикер.
            max_expiries: Число экспираций.
            force_source: "webull" | "yfinance" | None (авто).
            max_days: Горизонт в днях. Передаётся фетчеру, чтобы он не тратил
                запросы на экспирации, которые ``_filter_by_days`` всё равно
                отбросит (аудит 2026-09-17). Для Bybit не применяется
                (там свой набор экспираций).

        Returns:
            (snapshot, source, per_contract, q)
        """
        sym = ticker.strip().upper()

        # Futures: SPX/NDX index options as proxy
        if sym in FUTURES_TICKERS:
            cfg = DEFAULT_ASSETS[sym]
            fetcher = YFOptionsFetcher(
                max_expiries=max_expiries, redis_client=self._redis, max_days=max_days,
            )
            snap = fetcher.fetch(cfg["yf_ticker"])
            from dataclasses import replace
            snap = replace(snap, symbol=sym)
            return snap, "futures", int(cfg["per_contract"]), float(cfg["q"])

        if sym in _CRYPTO_ASSETS:
            cfg = _CRYPTO_ASSETS[sym]
            fetcher = BybitOptionsFetcher(max_expiries=max_expiries, redis_client=self._redis)
            snap = fetcher.fetch(sym)
            return snap, "crypto", int(cfg["per_contract"]), 0.0

        # US stocks: respect force_source or auto-detect
        # Дивидендная доходность q берётся из справочника (как в классическом
        # пути service.py:220), а не из дефолта конструктора 0.0 — иначе
        # SPY/QQQ/DIA/IWM получали заниженную гамму (аудит §4.6).
        stock_q = DEFAULT_ASSETS.get(sym, {}).get("q", 0.0)

        if force_source == "yfinance":
            fetcher = YFOptionsFetcher(
                max_expiries=max_expiries, redis_client=self._redis, max_days=max_days,
            )
            yf_sym = _YF_PROXY_MAP.get(sym, sym)
            snap = fetcher.fetch(yf_sym)
            if yf_sym != sym:
                from dataclasses import replace
                snap = replace(snap, symbol=sym)
            return snap, "yfinance", 100, stock_q

        if force_source == "webull":
            from gex.adapters.fetchers.webull_fetcher import WebullOptionsFetcher
            wb = WebullOptionsFetcher(
                max_expiries=max_expiries, redis_client=self._redis, max_days=max_days,
            )
            snap = wb.fetch(sym)
            return snap, "webull", 100, stock_q

        # Auto: Try Webull first for US stocks (real OI, pre-computed greeks, free)
        try:
            from gex.adapters.fetchers.webull_fetcher import WebullOptionsFetcher
            wb = WebullOptionsFetcher(
                max_expiries=max_expiries, redis_client=self._redis, max_days=max_days,
            )
            snap = wb.fetch(sym)
            return snap, "webull", 100, stock_q
        except Exception as exc:
            logger.info(
                "Webull unavailable for %s: %s — falling back to yfinance",
                sym, exc,
            )

        # Fallback: yfinance
        fetcher = YFOptionsFetcher(
            max_expiries=max_expiries, redis_client=self._redis, max_days=max_days,
        )
        yf_sym = _YF_PROXY_MAP.get(sym, sym)
        snap = fetcher.fetch(yf_sym)
        # Return original ticker as symbol (not proxy)
        if yf_sym != sym:
            from dataclasses import replace
            snap = replace(snap, symbol=sym)
        return snap, "yfinance", 100, stock_q


    # ------------------------------------------------------------------ #
    #  Фильтр цепочки по сроку экспирации (дни → годы)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _filter_by_days(snapshot: OptionSnapshot, days: float) -> OptionSnapshot:
        """Оставить опционы с T ≤ days/365.

        Если окно не покрывает ни одной экспирации — вернуть только ближайший
        экспирационный бакет (``min round(T*365)``) вместо полного снапшота:
        молчаливое расширение окна искажало бы временной вес и sparse-детекцию.
        Полный снапшот возвращается только когда сама цепочка пуста (дальше
        ``_assemble`` выбросит понятную ошибку). Не бросаем исключение: AUTO
        не должен превращать рабочий результат в 500.
        """
        from dataclasses import replace

        chain = snapshot.chain
        if chain.empty:
            return snapshot

        max_T = days / 365.0
        mask = chain["T"] <= max_T
        if mask.any():
            return replace(
                snapshot,
                chain=chain[mask].reset_index(drop=True),
            )

        tday = (chain["T"].astype(float) * 365.0).round()
        nearest = tday == tday.min()
        logger.warning(
            "_filter_by_days: окно days=%.1f не покрывает ни одной экспирации "
            "символа %s — возвращаю только ближайший бакет (~%d дн)",
            days, snapshot.symbol, int(tday.min()),
        )
        return replace(
            snapshot,
            chain=chain[nearest].reset_index(drop=True),
        )

    # ------------------------------------------------------------------ #
    #  Раздел 1-3: построение профиля по страйкам
    # ------------------------------------------------------------------ #
    def _build_per_strike(
        self,
        chain: pd.DataFrame,
        spot: float,
        per_contract: int,
        effective_q: Optional[float] = None,
    ) -> pd.DataFrame:
        """Построить агрегированный профиль по страйкам с временными весами.

        Шаги:
          1. Гамма BSM для каждого опциона;
          2. GEX = sign × Γ × OI × per_contract × 100 (ТЗ 1.2);
          3. Временной вес экспирации ``e^(-T_days/30)`` (ТЗ 3.1);
          4. Взвешенное суммирование Call/Put GEX по страйку (ТЗ 3.2);
          5. Net GEX, AG, Gamma Dollar, DHR.
        """
        q = effective_q if effective_q is not None else self.q
        df = chain.copy()

        # --- 1. Гамма BSM (векторизованно) ---
        gamma = bs_gamma(
            spot, df["strike"], df["T"], self.r, df["iv"], q,
        )
        # --- 2. Знак дилера и GEX ---
        is_call = (df["type"].values == "C")
        sign = np.where(is_call, self.call_sign, self.put_sign)
        # GEX = Γ × OI × per_contract × 100 (ТЗ 1.2: per_contract × dollar-scale=100)
        gex = sign * gamma * df["oi"].values * per_contract * 100.0

        # --- 3. Временной вес экспирации ---
        t_days = df["T"].values * 365.0
        weight = np.exp(-t_days / _EXPIRY_TAU_DAYS)

        # --- 4. Взвешенный GEX (ТЗ 3.2: суммируем с весом экспирации) ---
        gex_weighted = gex * weight

        df = df.assign(
            gamma=gamma,
            sign=sign,
            gex=gex,
            gex_weighted=gex_weighted,
            weight=weight,
        )

        # --- 5. Агрегация по страйку ---
        rows = []
        for k, sub in df.groupby("strike", sort=True):
            calls = sub[sub["type"] == "C"]
            puts = sub[sub["type"] == "P"]
            gex_call = float(calls["gex_weighted"].sum())
            gex_put = float(puts["gex_weighted"].sum())
            gex_net = gex_call + gex_put
            ag = abs(gex_call) + abs(gex_put)   # ТЗ 1.3 / 5.2
            rows.append({
                "strike": float(k),
                "gex_call": gex_call,
                "gex_put": gex_put,
                "gex_net": gex_net,
                "ag": ag,
                "gex_abs": abs(gex_net),
                "gamma_call": float(calls["gamma"].sum()),
                "gamma_put": float(puts["gamma"].sum()),
                "oi_call": float(calls["oi"].sum()),
                "oi_put": float(puts["oi"].sum()),
                "weight": float(sub["weight"].sum()),
            })
        result = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        if result.empty:
            return result

        # --- 6. Производные поля ---
        # Gamma Dollar = Net GEX × strike × 0.01 (ТЗ 4.1)
        result["gamma_dollar"] = result["gex_net"] * result["strike"] * 0.01
        # AG_normalized = AG / max(AG) (ТЗ 5.2)
        max_ag = result["ag"].max()
        result["ag_normalized"] = (
            result["ag"] / max_ag if max_ag > 0 else 0.0
        )
        # Delta Hedge Ratio = Call GEX / |Put GEX| (ТЗ 7.2)
        result["delta_hedge_ratio"] = result.apply(
            lambda r: (r["gex_call"] / abs(r["gex_put"]))
            if r["gex_put"] != 0 else None,
            axis=1,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Раздел 2.1: Zero Gamma Level
    # ------------------------------------------------------------------ #
    @staticmethod
    def _zero_gamma_level(per_strike: pd.DataFrame) -> Optional[float]:
        """Цена, при которой суммарный Net GEX = 0.

        Кумулятивная сумма Net GEX по страйкам (снизу вверх), поиск смены знака,
        линейная интерполяция между соседними страйками (ТЗ 2.1).

        Берётся **последняя** смена знака: сохранённые далёкие OTM-путы с малым,
        но ненулевым GEX могут дать ранний «шумовой» переход нуля ниже спота;
        настоящий гамма-флип — это устойчивый переход в положительную
        (call-доминирующую) зону выше пут-тяжёлой области.
        """
        if per_strike.empty:
            return None
        df = per_strike.sort_values("strike").reset_index(drop=True)
        cum = df["gex_net"].cumsum()
        # Индексы, где знак кумулятивы меняется.
        sign_change = np.sign(cum).diff().fillna(0) != 0
        idx = np.where(sign_change.values)[0]
        if len(idx) == 0:
            return None
        i = int(idx[-1])
        if i == 0:
            # Переход на самой первой строке — взять её как точку.
            return float(df.loc[0, "strike"])
        s0, s1 = float(df.loc[i - 1, "strike"]), float(df.loc[i, "strike"])
        g0, g1 = float(cum.iloc[i - 1]), float(cum.iloc[i])
        if np.isclose(g1 - g0, 0.0):
            return float(0.5 * (s0 + s1))
        s_star = s0 - g0 * (s1 - s0) / (g1 - g0)
        return float(s_star)

    # ------------------------------------------------------------------ #
    #  Раздел 2.2 / 2.3: Call/Put Wall + сила
    # ------------------------------------------------------------------ #
    def _wall(
        self,
        per_strike: pd.DataFrame,
        positive: bool,
    ) -> tuple[float, float]:
        """Найти стену (страйк с экстремальным Net GEX) и её силу.

        Parameters
        ----------
        positive : bool
            ``True`` → Call Wall (макс положительный Net GEX);
            ``False`` → Put Wall (мин отрицательный Net GEX).

        Returns
        -------
        (strike, strength)
            strength = |Net GEX стены| / Σ |Net GEX всех страйков| (ТЗ 2.2).
        """
        if per_strike.empty:
            return float("nan"), 0.0
        subset = per_strike[per_strike["gex_net"] > 0] if positive \
            else per_strike[per_strike["gex_net"] < 0]
        if subset.empty:
            return float("nan"), 0.0

        if positive:
            row = subset.loc[subset["gex_net"].idxmax()]
        else:
            row = subset.loc[subset["gex_net"].idxmin()]
        strike = float(row["strike"])
        total_abs = float(per_strike["gex_abs"].sum())
        strength = abs(float(row["gex_net"])) / total_abs if total_abs > 0 else 0.0
        return strike, float(strength)

    # ------------------------------------------------------------------ #
    #  Раздел 2.4: Power Zones
    # ------------------------------------------------------------------ #
    @staticmethod
    def _power_zones(per_strike: pd.DataFrame, spot: float) -> list[PowerZone]:
        """Группы соседних страйков с экстремально высокой AG (топ-10%).

        Алгоритм (ТЗ 2.4):
          1. отсортировать страйки по AG (убывание);
          2. взять топ-10% (не менее 1 страйка);
          3. объединить соседние (gap ≤ 5% от spot) в зоны;
          4. для каждой зоны: центр (AG-взвешенный), ширина, total_ag,
             доминирующий тип (знак Σ Net GEX).
        """
        if per_strike.empty:
            return []
        df = per_strike.sort_values("ag", ascending=False).reset_index(drop=True)
        # Топ-10% по AG, минимум 1 страйк.
        n_top = max(1, int(math.ceil(len(df) * _POWER_ZONE_TOP_QUANTILE)))
        top = df.head(n_top).sort_values("strike").reset_index(drop=True)

        max_gap = spot * _POWER_ZONE_MAX_GAP_PCT
        zones: list[PowerZone] = []
        current: list[pd.Series] = []

        for _, row in top.iterrows():
            if not current:
                current.append(row)
                continue
            prev_strike = float(current[-1]["strike"])
            if float(row["strike"]) - prev_strike <= max_gap:
                current.append(row)
            else:
                zones.append(ExtendedGEXAnalyzer._build_zone(current))
                current = [row]
        if current:
            zones.append(ExtendedGEXAnalyzer._build_zone(current))

        # Зоны сортируем по убыванию total_ag (сильнейшие первыми).
        zones.sort(key=lambda z: z.total_ag, reverse=True)
        return zones

    @staticmethod
    def _build_zone(strikes: list[pd.Series]) -> PowerZone:
        """Собрать PowerZone из списка строк per_strike."""
        ks = [float(s["strike"]) for s in strikes]
        ags = [float(s["ag"]) for s in strikes]
        nets = [float(s["gex_net"]) for s in strikes]
        total_ag = sum(ags)
        # AG-взвешенный центр.
        if total_ag > 0:
            center = sum(k * a for k, a in zip(ks, ags)) / total_ag
        else:
            center = sum(ks) / len(ks)
        net_sum = sum(nets)
        dominant = "CALL" if net_sum >= 0 else "PUT"
        return PowerZone(
            center=float(center),
            width=float(max(ks) - min(ks)),
            total_ag=float(total_ag),
            dominant_type=dominant,
            n_strikes=len(strikes),
            min_strike=float(min(ks)),
            max_strike=float(max(ks)),
        )

    # ------------------------------------------------------------------ #
    #  Раздел 4.1: Gamma Dollar (по страйкам) — считается в _build_per_strike
    #  Раздел 4.2: Hedge Requirement
    # ------------------------------------------------------------------ #
    def _hedge_requirement(
        self,
        chain: pd.DataFrame,
        per_strike: pd.DataFrame,
        spot: float,
        per_contract: int,
        scenario_pct: float,
    ) -> HedgeScenario:
        """Сколько акций/монет дилерам нужно купить/продать при движении цены.

        ТЗ 4.2: изменение цены на ``scenario_pct``% → изменение дельты каждого
        опциона ≈ Γ × Δprice. Хедж на страйке = Net GEX × Δdelta. Сумма по
        страйкам → общий долларовый хедж → перевод в количество акций через spot.

        Здесь используем агрегированный профиль по страйкам: на каждом страйке
        доля хеджа = ``gex_net × (Δprice/100)`` (т.к. gex_net уже в долларах
        на 1% движения спота, см. формулу GEX). Δprice в % = scenario_pct.
        """
        delta_price_pct = scenario_pct  # движение в %
        # gex_net = $ на 1% движения; при сценарии X% хедж = gex_net × X
        # (знак: + = дилеры покупают, − = продают).
        dollar_hedge = float((per_strike["gex_net"] * delta_price_pct).sum())
        shares = dollar_hedge / spot if spot > 0 else 0.0
        return HedgeScenario(
            scenario_pct=float(scenario_pct),
            shares=float(shares),
            dollar_value=float(dollar_hedge),
        )

    # ------------------------------------------------------------------ #
    #  Раздел 4.3: Directional Bias
    # ------------------------------------------------------------------ #
    @staticmethod
    def _directional_bias(net_gex: float, total_ag: float) -> str:
        """BULLISH / BEARISH / NEUTRAL по знаку Total Net GEX (ТЗ 4.3)."""
        if total_ag <= 0:
            return "NEUTRAL"
        if abs(net_gex) / total_ag < _NEUTRAL_BIAS_THRESHOLD:
            return "NEUTRAL"
        return "BULLISH" if net_gex > 0 else "BEARISH"

    # ------------------------------------------------------------------ #
    #  Раздел 4.4: Put/Call Ratio
    # ------------------------------------------------------------------ #
    @staticmethod
    def _put_call_ratio(total_call_gex: float, total_put_gex: float) -> float:
        """PCR = |Total Put GEX| / Total Call GEX (ТЗ 4.4).

        total_put_gex < 0 по конвенции; берём модуль. Если коллов нет — inf
        заменяем на 0 (робастность).
        """
        if total_call_gex == 0:
            return 0.0
        return float(abs(total_put_gex) / total_call_gex)

    # ------------------------------------------------------------------ #
    #  Раздел 6: Топ-N уровней S/R
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key_levels(
        per_strike: pd.DataFrame,
        spot: float,
        top_n: int = 5,
    ) -> list[KeyLevel]:
        """Топ-N страйков по |Net GEX| с типизацией SUPPORT/RESISTANCE (ТЗ 6.1)."""
        if per_strike.empty:
            return []
        df = per_strike.sort_values("gex_abs", ascending=False).head(top_n)
        levels: list[KeyLevel] = []
        for _, row in df.iterrows():
            net = float(row["gex_net"])
            strike = float(row["strike"])
            levels.append(KeyLevel(
                strike=strike,
                type="RESISTANCE" if net > 0 else "SUPPORT",
                strength=abs(net),
                distance_pct=((strike - spot) / spot * 100.0) if spot > 0 else 0.0,
            ))
        return levels

    # ------------------------------------------------------------------ #
    #  Раздел 7.1: Gamma Exposure Score (0-100)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gamma_exposure_score(net_gex: float, total_ag: float) -> float:
        """Score = |Total Net GEX| / Total AG × 100 (ТЗ 7.1).

        Если Total AG = 0 — возвращаем 0. Значение по построению ∈ [0, 100].
        """
        if total_ag <= 0:
            return 0.0
        return float(abs(net_gex) / total_ag * 100.0)

    # ------------------------------------------------------------------ #
    #  Раздел 7.3: Max Pain
    # ------------------------------------------------------------------ #
    @staticmethod
    def _max_pain(chain: pd.DataFrame, spot: Optional[float] = None) -> Optional[float]:
        """Страйк, где суммарные потери держателей опционов **минимальны** (ТЗ 7.3).

        Для каждого кандидатного страйка K_p:
          потери_коллов = Σ max(0, K_call - K_p) × OI_call
          потери_путов  = Σ max(0, K_p - K_put) × OI_put
        Max Pain = K_p с **минимумом** суммарных потерь: при экспирации на этом
        страйке держатели опционов теряют меньше всего. При равных потерях
        выбирается страйк, ближайший к ``spot`` (если передан), иначе — первый
        минимальный страйк (``candidate_strikes`` отсортирован по возрастанию).
        """
        if chain.empty:
            return None
        calls = chain[chain["type"] == "C"]
        puts = chain[chain["type"] == "P"]
        if calls.empty and puts.empty:
            return None

        # Кандидаты — все страйки в цепочке (дискретный поиск).
        candidate_strikes = sorted(chain["strike"].unique())
        if not candidate_strikes:
            return None

        call_K = calls["strike"].values
        call_OI = calls["oi"].values
        put_K = puts["strike"].values
        put_OI = puts["oi"].values

        best_strike: Optional[float] = None
        best_loss = float("inf")
        for kp in candidate_strikes:
            # Потери по коллам: в деньгах коллы, чей страйк выше K_p.
            if len(call_K):
                call_loss = float(np.sum(np.maximum(0.0, call_K - kp) * call_OI))
            else:
                call_loss = 0.0
            # Потери по путам: в деньгах путы, чей страйк ниже K_p.
            if len(put_K):
                put_loss = float(np.sum(np.maximum(0.0, kp - put_K) * put_OI))
            else:
                put_loss = 0.0
            total = call_loss + put_loss
            if total < best_loss:
                best_loss = total
                best_strike = float(kp)
            elif total == best_loss and spot is not None and best_strike is not None:
                # Ничья: предпочитаем страйк, ближайший к spot.
                if abs(float(kp) - float(spot)) < abs(best_strike - float(spot)):
                    best_strike = float(kp)
        return best_strike

    # ------------------------------------------------------------------ #
    #  Сборка ExtendedStrike-списка
    # ------------------------------------------------------------------ #
    @staticmethod
    def _strikes_to_dataclass(per_strike: pd.DataFrame) -> list[ExtendedStrike]:
        rows: list[ExtendedStrike] = []
        for _, r in per_strike.iterrows():
            rows.append(ExtendedStrike(
                strike=float(r["strike"]),
                gex_call=float(r["gex_call"]),
                gex_put=float(r["gex_put"]),
                gex_net=float(r["gex_net"]),
                ag=float(r["ag"]),
                gamma_call=float(r["gamma_call"]),
                gamma_put=float(r["gamma_put"]),
                oi_call=float(r["oi_call"]),
                oi_put=float(r["oi_put"]),
                gamma_dollar=float(r["gamma_dollar"]),
                delta_hedge_ratio=(
                    None if pd.isna(r["delta_hedge_ratio"])
                    else float(r["delta_hedge_ratio"])
                ),
                ag_normalized=float(r["ag_normalized"]),
                weight=float(r["weight"]),
            ))
        return rows

    # ------------------------------------------------------------------ #
    #  Volume Profile (OI distribution by moneyness zone)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_volume_profile(
        chain: pd.DataFrame, spot: float,
    ) -> "VolumeProfile":
        """Распределение OI по зонам: Ask-or-above / Between / Bid-or-below.

        Зоны:
          - "Bid or below":  strike ≤ spot × (1 - atm_pct)
          - "Between market": spot × (1 - atm_pct) < strike < spot × (1 + atm_pct)
          - "Ask or above":  strike ≥ spot × (1 + atm_pct)

        atm_pct ширина ATM зоны: 1% для ETF/акций, 2% для крипты.
        """
        if chain.empty or spot <= 0:
            return VolumeProfile(
                spot=spot, atm_pct=0.01, total_call_oi=0, total_put_oi=0,
                total_oi=0, call_pct=0, put_pct=0, zones=[],
            )

        # ATM zone width: narrower for high-priced stocks, wider for cheap/crypto
        atm_pct = 0.005 if spot > 500 else 0.01 if spot > 50 else 0.02
        lo = spot * (1.0 - atm_pct)
        hi = spot * (1.0 + atm_pct)

        def _zone(strike: float) -> str:
            if strike <= lo:
                return "bid_or_below"
            elif strike >= hi:
                return "ask_or_above"
            return "between"

        chain_copy = chain.copy()
        chain_copy["zone"] = chain_copy["strike"].apply(_zone)

        calls = chain_copy[chain_copy["type"] == "C"]
        puts = chain_copy[chain_copy["type"] == "P"]

        total_oi = float(chain_copy["oi"].sum())
        total_call = float(calls["oi"].sum())
        total_put = float(puts["oi"].sum())

        zones = []
        zone_labels = [
            ("ask_or_above", "Ask or above"),
            ("between", "Between market"),
            ("bid_or_below", "Bid or below"),
        ]

        for zkey, zlabel in zone_labels:
            call_oi = float(calls[calls["zone"] == zkey]["oi"].sum())
            put_oi = float(puts[puts["zone"] == zkey]["oi"].sum())
            zones.append(VolumeZone(
                zone=zkey,
                label=zlabel,
                call_oi=call_oi,
                put_oi=put_oi,
                call_pct=round(call_oi / total_oi * 100, 2) if total_oi > 0 else 0,
                put_pct=round(put_oi / total_oi * 100, 2) if total_oi > 0 else 0,
            ))

        return VolumeProfile(
            spot=spot,
            atm_pct=atm_pct,
            total_call_oi=total_call,
            total_put_oi=total_put,
            total_oi=total_oi,
            call_pct=round(total_call / total_oi * 100, 2) if total_oi > 0 else 0,
            put_pct=round(total_put / total_oi * 100, 2) if total_oi > 0 else 0,
            zones=zones,
        )


# ====================================================================== #
#  Адаптер per-strike → StrikeLite для detect_sparse
# ====================================================================== #
def _strikes_to_lite(strikes: list[ExtendedStrike]) -> list["StrikeLite"]:
    """Преобразовать ``ExtendedStrike`` в :class:`StrikeLite` для оценки разреженности."""
    from gex.application.auto_scope import StrikeLite

    return [
        StrikeLite(
            strike=float(s.strike),
            oi_call=float(s.oi_call),
            oi_put=float(s.oi_put),
            gex_net=float(s.gex_net),
        )
        for s in strikes
    ]


def _coverage_strike_meta(
    chain: pd.DataFrame,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Additive Phase-4 охват: (strike_min, strike_max, nearest_expiry_days).

    Пустая/``None`` цепочка → ``(None, None, None)`` (поля опциональны).
    """
    if chain is None or len(chain) == 0:
        return None, None, None
    return (
        float(chain["strike"].min()),
        float(chain["strike"].max()),
        float((chain["T"].astype(float) * 365.0).min()),
    )


def _build_coverage(
    days: float,
    source: str,
    report: "ExtendedGEXReport",
    chain: pd.DataFrame,
) -> "AutoCoverage":
    """Метаданные качества выборки для **любого** режима (аудит 2026-09-17).

    Отличается от блока AUTO только тем, что не знает про эскалацию:
    ``fallback_used``/``escalated``/``partial`` остаются ``False``, а
    ``mode`` = ``"manual"``. Все «фактические» поля (число экспираций,
    страйков, контрактов, диапазон страйков, ближняя/дальняя экспирация)
    считаются одинаково — их и показывает блок Data Coverage.
    """
    from gex.application.auto_scope import (
        AutoCoverage,
        count_expiries,
        detect_sparse,
    )

    n_expiries = count_expiries(chain)
    reasons = detect_sparse(
        _strikes_to_lite(report.per_strike), report.spot,
        report.call_wall, report.put_wall,
        expiries=n_expiries,
    )
    strike_min, strike_max, nearest = _coverage_strike_meta(chain)
    furthest: Optional[float] = None
    if chain is not None and len(chain) > 0:
        try:
            furthest = float((chain["T"].astype(float) * 365.0).max())
        except (TypeError, ValueError):
            furthest = None

    return AutoCoverage(
        mode="manual",
        resolved_days=float(days),
        resolved_expiries=int(n_expiries),
        sources_used=[source],
        primary_source=source,
        expirations_merged=int(n_expiries),
        strike_count=len(report.per_strike),
        total_oi=float(sum(s.oi_call + s.oi_put for s in report.per_strike)),
        sparse=bool(reasons),
        sparse_reasons=list(reasons),
        strike_min=strike_min,
        strike_max=strike_max,
        nearest_expiry_days=nearest,
        furthest_expiry_days=furthest,
    )


