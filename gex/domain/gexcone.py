"""GEX Cone — конус вероятностей цены на основе OI, GEX и Aggregate Gamma.

GEX-ядро — от единого движка
----------------------------
Стены (Call/Put Wall), Gamma Flip, режим, Net GEX и Aggregate Gamma конус
**не считает сам**: ``build_gex_cone`` получает готовый канонический
``GEXProfile`` (и адаптированный per-strike кадр ``stk_all``) от
:meth:`GEXPipelineRunner.run_gex_profile_domain` — ровно того же движка и по
той же цепочке (после ``_filter_by_days``), что и главная страница GEX.
Любое изменение движка автоматически отражается в конусе. Собственные
фильтры конуса (3σ OTM, 90% OI) влияют только на визуализацию: экспирации,
уровни, путь конуса и ось Y.

Идея
----
Классический «конус вероятностей» (probability cone) показывает, как
неопределённость цены растёт со временем до экспирации: из текущего спота
расходятся границы 1σ/2σ/3σ, вычисленные из ATM-волатильности опционной
цепочки по логнормальному распределению:

.. math::
    \\ln S_T \\sim \\ln S_0 + (r - q - \\sigma^2/2)\\,T + \\sigma\\sqrt{T}\\,Z

Расчёт конуса (OI + GEX + Aggregate Gamma)
------------------------------------------
База — **OI-взвешенная ATM-IV** каждой экспирации (:func:`_atm_iv`): страйки
в ±10% от спота, вес = OI. Это «OI-составляющая» конуса.

Затем волатильность корректируется на **GEX-структуру экспирации**
(«GEX/AG-составляющая»): считается дилерская гамма-экспозиция по опционам
данной даты (BSM-гамма × знак дилера × OI × контрактный множитель):

.. math::
    GEX_i = sign_i \\cdot \\Gamma_i \\cdot OI_i \\cdot 100 \\cdot S^2 \\cdot 0.01,
    \\quad AG = \\sum_i |GEX_i|, \\quad score = \\min\\!\\left(100,\\ \\frac{|\\sum GEX_i|}{AG}\\,100\\right)

Режим экспирации задаёт множитель волатильности:

* **POSITIVE** (дилеры long gamma, net GEX ≥ 0) — рынок «приклеен» к страйкам,
  волатильность подавляется: :math:`\\sigma_{gex} = \\sigma_{ATM}(1 - 0.05\\cdot score/100)`;
* **NEGATIVE** (дилеры short gamma, net GEX < 0) — тренды усиливаются,
  хвосты шире: :math:`\\sigma_{gex} = \\sigma_{ATM}(1 + 0.20\\cdot score/100)`.

Границы σ и квантили строятся по :math:`\\sigma_{gex}` — конус отражает
не только «цену» волатильности, но и гамма-структуру рынка.

Уровни конуса (OI + GEX + AG)
-----------------------------
Каждый уровень — это страйк, вокруг которого сосредоточены объёмы и
дилерская гамма. Ранжирование идёт по **композитной силе**:

.. math::
    strength = 0.5\\cdot\\frac{OI}{\\max OI} + 0.5\\cdot\\frac{AG}{\\max AG}

* ``composite`` — топ-N страйков по композитной силе (сопротивления выше
  спота, поддержки ниже);
* ``call_wall`` / ``put_wall`` — GEX-стены экспирации/цепочки: страйк с
  максимальной положительной (отрицательной) дилерской гаммой;
* ``gamma_flip`` (глобально) — цена, при которой суммарный GEX = 0.

Ступеньки вероятностей (даты + объёмы + уровни)
-----------------------------------------------
Для каждой экспирации T базовая логнормальная вероятность P(S_T > K)
умножается на затухающие факторы **всех стен этой экспирации**, лежащих между
спотом и K:

.. math::
    P_{adj}(S_T > K) = P_{base}(S_T > K) \\cdot
    \\prod_{R_i \\le K} \\exp\\left(-\\lambda\\, w_i\\,
    \\frac{K - R_i + S_0}{S_0}\\right)

На самом уровне (K = R_i) фактор = ``exp(-λ·w_i)`` — **дискретная ступенька**;
дальше затухание продолжается экспоненциально. Чем сильнее уровень (больше
OI/AG), тем сильнее ступенька. Зеркально для поддержек (P(S_T < K)).

Модуль — **чистая математика** без HTTP/I/O: на вход ``OptionSnapshot``
(цепочка + спот), на выход — dataclass, готовый к Pydantic-сериализации.
"""

# --------------------------------------------------------------------------- #
#  Разбор файла (итерация 40)
# --------------------------------------------------------------------------- #
# Расчёт разложен на `gex/domain/options/cone/*`: model (константы, структуры, общие
# примитивы), expiries (отбор цепочки и уровни), probabilities (логнормальная база,
# поправка на стены, лестница), path (путь конуса и глобальные уровни).
#
# Здесь остаются точка входа `build_gex_cone`, публичные `atr_14`/`historical_vol` и
# реэкспорт всего вынесенного — поэтому `from gex.domain.gexcone import ...` у вызывающих не
# меняется. Числа закреплены эталоном `tests/test_cone_golden.py`, снятым с исходного
# файла ДО разбора (три конфигурации, лестница вероятностей по каждому уровню).

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional
import numpy as np
import pandas as pd

from gex.domain.data_loader import (
    OptionSnapshot,
)
from gex.domain.metrics import GEXProfile
from gex.domain.options.cone.expiries import (
    _atm_iv,
    _build_expirations,
    _expiry_levels,
    _filter_extreme_otm,
    _filter_oi_quantile,
    expirations_levels_iter,
)
from gex.domain.options.cone.model import (
    ConeExpiry,
    ConeExpiryLevel,
    DEFAULT_GLOBAL_TOP,
    DEFAULT_OI_QUANTILE,
    DEFAULT_TOP_OI_PER_EXPIRY,
    GexConeData,
    GlobalLevel,
    _ATM_BAND,
    _FLIP_STRENGTH,
    _HV_PERIOD_DAYS,
    _IV_CEIL,
    _IV_FLOOR,
    _MIN_STICK_STRENGTH,
    _NEGATIVE_GAMMA_MULT,
    _OTM_SIGMAS,
    _POSITIVE_GAMMA_MULT,
    _PULL_AMP,
    _WALL_PULL,
    _W_OI,
    _composite_strength,
    _finite_or_none,
    _gamma_score,
    _strike_gex,
    _vol_mult,
)
from gex.domain.options.cone.path import (
    _build_cone_path,
    _global_levels,
    _smooth_pull,
)
from gex.domain.options.cone.probabilities import (
    _append_prob,
    _lognorm_quantile,
    _lognorm_z,
    _prob_above,
    _prob_below,
    _wall_factor,
)


__all__ = [
    "ConeExpiry",
    "ConeExpiryLevel",
    "DEFAULT_GLOBAL_TOP",
    "DEFAULT_OI_QUANTILE",
    "DEFAULT_TOP_OI_PER_EXPIRY",
    "GexConeData",
    "GlobalLevel",
    "_ATM_BAND",
    "_FLIP_STRENGTH",
    "_HV_PERIOD_DAYS",
    "_IV_CEIL",
    "_IV_FLOOR",
    "_MIN_STICK_STRENGTH",
    "_NEGATIVE_GAMMA_MULT",
    "_OTM_SIGMAS",
    "_POSITIVE_GAMMA_MULT",
    "_PULL_AMP",
    "_WALL_PULL",
    "_W_OI",
    "_append_prob",
    "_atm_iv",
    "_build_cone_path",
    "_build_expirations",
    "_composite_strength",
    "_expiry_levels",
    "_filter_extreme_otm",
    "_filter_oi_quantile",
    "_finite_or_none",
    "_gamma_score",
    "_global_levels",
    "_lognorm_quantile",
    "_lognorm_z",
    "_prob_above",
    "_prob_below",
    "_smooth_pull",
    "_strike_gex",
    "_vol_mult",
    "_wall_factor",
    "atr_14",
    "build_gex_cone",
    "expirations_levels_iter",
    "historical_vol",
]


def build_gex_cone(
    snapshot: OptionSnapshot,
    *,
    profile: GEXProfile,
    stk_all: pd.DataFrame,
    r: float = 0.045,
    q: float = 0.0,
    atm_vol: Optional[float] = None,
    wall_decay: float = 4.0,
    max_expiries: int = 8,
    horizon_days: int = 14,
    top_oi_per_expiry: int = DEFAULT_TOP_OI_PER_EXPIRY,
    global_top: int = DEFAULT_GLOBAL_TOP,
    oi_quantile: float = DEFAULT_OI_QUANTILE,
    hv: Optional[float] = None,
    atr: Optional[float] = None,
) -> GexConeData:
    """Построить GEX-конус вероятностей поверх КАНОНИЧЕСКОГО GEX-профиля.

    Parameters
    ----------
    snapshot : OptionSnapshot
        Цепочка (``chain``: strike/type/oi/iv/T, ``spot``, ``as_of``) — та же,
        что передана движку (до days-фильтра) или уже отфильтрованная по days:
        конус использует её ТОЛЬКО для визуализации (экспирации, уровни, путь).
    profile : GEXProfile
        Канонический профиль, посчитанный единым движком
        (:meth:`GEXPipelineRunner.run_gex_profile_domain`) по цепочке,
        отфильтрованной по days — ровно как у главной страницы GEX. Стены,
        Gamma Flip, режим, Net GEX берутся отсюда, а не пересчитываются.
    stk_all : pd.DataFrame
        Адаптер ``profile.per_strike`` в колонки ``strike``/``gex_net``/``ag``
        (см. :func:`gex.application.gex_engine.stk_all_from_profile`). Весь
        GEX/AG конуса — из канонического профиля.
    r : float
        Безрисковая ставка (годовая, непрерывная) — параметр движка.
    q : float
        Дивидендная доходность (годовая, непрерывная) — параметр движка.
    atm_vol : float, optional
        OI-взвешенная ATM-вола движка (та же σ, что у главной страницы).
        Если None — локальная оценка ``_atm_iv`` по цепочке.
    wall_decay : float
        Параметр λ затухания вероятности за объёмными страйками. 0 — стены
        выключены (чистый логнормальный конус). Рекомендуемый диапазон 1..10.
    max_expiries : int
        Максимум экспираций в конусе (после группировки по датам).
    horizon_days : int
        Горизонт прогноза, дней: в конус попадают экспирации с dte ≤ horizon
        (минимум 2 ближайшие, если в окне меньше). Оно же — ``days`` движка.
    top_oi_per_expiry : int
        Сколько топ-уровней (по OI+AG) включать для каждой экспирации.
    global_top : int
        Сколько глобальных топ-уровней (по OI+AG цепочки) включать с каждой
        стороны для горизонтальных линий и таблицы.
    oi_quantile : float
        Доля суммарного OI, покрываемая страйками цепочки (0.9 = 90%).
        Влияет только на визуализацию (уровни/ось/путь) — профиль уже
        посчитан по полной цепочке движком.
    hv : float, optional
        Историческая волатильность (годовая, 0..1) — для диапазона оси Y и
        фильтра «явно OTM».
    atr : float, optional
        Average True Range (14 дней, в ценах) — фильтр «явно OTM».

    Returns
    -------
    GexConeData
        Экспирации (GEX-метрики, границы σ, квантили, уровни), глобальные
        уровни (вкл. Call/Put Wall и Gamma Flip) и агрегаты AG/γ-score.
        Все headline-метрики совпадают с главной страницей GEX при том же
        горизонте (``horizon_days`` = ``days``).

    Raises
    ------
    ValueError
        Если цепочка пуста или спот некорректен.
    """
    chain = snapshot.chain
    if chain is None or len(chain) == 0:
        raise ValueError("Пустая опционная цепочка — конус не построить.")

    spot = float(snapshot.spot)
    if spot <= 0:
        raise ValueError("spot должен быть > 0")

    # --- 0. Фильтры страйков ТОЛЬКО для визуализации: явно OTM (HV/IV/ATR) + доля OI ---
    # Профиль (стены, Gamma Flip, режим, Net GEX, AG) уже посчитан движком по
    # полной цепочке (как на главной странице GEX) — фильтры на него не влияют.
    # Они лишь выбирают «значимые» страйки для экспираций/уровней/оси конуса.
    iv_atm_pre = atm_vol if atm_vol and atm_vol > 0 else _atm_iv(chain, spot)
    chain = _filter_extreme_otm(chain, spot, hv, iv_atm_pre, atr)
    chain = _filter_oi_quantile(chain, oi_quantile)
    snapshot = replace(snapshot, chain=chain)

    # --- 1. Агрегаты из канонического профиля (никакого пересчёта GEX) ---
    total_ag = float(stk_all["ag"].sum()) if not stk_all.empty else 0.0
    gamma_score = _gamma_score(profile.net_gex, total_ag)
    vol_mult = _vol_mult(profile.regime, gamma_score)

    # --- 2. Экспирации: IV → vol_gex → границы σ + уровни каждой даты ---
    # GEX/AG дат берутся из канонического stk_all (подмножество страйков даты),
    # а не пересчитываются отдельным движком.
    expirations = _build_expirations(
        chain, snapshot, spot, r, q, top_oi_per_expiry, stk_all,
    )

    # --- 2.1 Горизонт: оставляем экспирации не дальше horizon_days (минимум 2) ---
    within = [e for e in expirations if e.dte <= horizon_days]
    if len(within) >= 2:
        expirations = within
    else:
        # Мало дат в окне — берём 2 ближайшие (иначе конус вырождается)
        expirations = expirations[:2]

    # --- 2.2 Лимит числа экспираций ---
    if len(expirations) > max_expiries:
        expirations = expirations[:max_expiries]

    # --- 3. Глобальные уровни: топ по OI+AG + GEX-стены + Gamma Flip ---
    globals_ = _global_levels(chain, spot, global_top, profile, stk_all)

    # --- 3.1 Плавные линии конуса: логнормальная граница + притяжение к стенам ---
    # Верхняя линия идёт к 2σ, нижняя — к 3σ (put-skew: вниз дальше). Каждая
    # стена (объёмный страйк / Call·Put Wall) плавно «тянет» линию к себе:
    # чем больше OI/AG (strength), тем сильнее и шире зона притяжения —
    # получается кривая с плавными изгибами вместо жёсткой лестницы.
    all_walls = list(expirations_levels_iter(expirations)) + globals_
    for exp in expirations:
        exp.upper_cone = _smooth_pull(exp.upper_2sd, spot, all_walls, "resistance")
        exp.lower_cone = _smooth_pull(exp.lower_3sd, spot, all_walls, "support")

    # --- 3.2 Плотная полилиния конуса (по дням) для гладкой отрисовки ---
    iv_atm = atm_vol if atm_vol and atm_vol > 0 else _atm_iv(chain, spot)
    cone_path = _build_cone_path(expirations, spot, snapshot.as_of, all_walls, iv_atm)

    # --- 4. Вероятности со ступеньками (каскад стен СВОЕЙ экспирации) ---
    all_levels = list(expirations_levels_iter(expirations)) + globals_
    for lvl in all_levels:
        lvl.probs = []
    for exp in expirations:
        walls = exp.levels
        for lvl in all_levels:
            _append_prob(lvl, walls, exp, spot, r, q, wall_decay)

    # --- 5. Лимит ближайших экспираций уже применён (2.1/2.2) —
    #     синхронизируем probs уровней с финальным списком экспираций ---
    final_dtes = [e.dte for e in expirations]
    for lvl in list(expirations_levels_iter(expirations)) + globals_:
        lvl.probs = [p for p in lvl.probs if p["dte"] in final_dtes]

    # --- 6. (ATM-вола посчитана в 3.2 для пути конуса) ---

    # --- 7. Ось Y: диапазон цен = конус ∪ HV-коррекция ∪ значимые страйки ---
    # Верх: max(верхняя линия конуса, спот + HV·√T, самый высокий значимый
    # страйк). Низ: зеркально. Значимые страйки (из 90% OI-набора) учитываются
    # только если не дальше 1.5× ширины конуса от спота — дальние объёмные
    # страйки остаются в уровнях/стенах, но не раздувают ось Y.
    strikes_90 = sorted(set(chain["strike"].tolist()))
    t_max = max((e.dte for e in expirations), default=14) / 365.0
    hv_move = spot * hv * math.sqrt(t_max) if hv and hv > 0 else 0.0
    upper_log = max(p["upper"] for p in cone_path) if cone_path else max(e.upper_cone for e in expirations)
    lower_log = min(p["lower"] for p in cone_path) if cone_path else min(e.lower_cone for e in expirations)
    up_zone = spot + 1.5 * max(upper_log - spot, 1e-9)
    lo_zone = spot - 1.5 * max(spot - lower_log, 1e-9)
    sig_up = [s for s in strikes_90 if spot < s <= up_zone]
    sig_lo = [s for s in strikes_90 if lo_zone <= s < spot]
    upper_bound = max([upper_log, spot + hv_move] + ([max(sig_up)] if sig_up else []))
    lower_bound = min([lower_log, spot - hv_move] + ([min(sig_lo)] if sig_lo else []))

    as_of = snapshot.as_of
    return GexConeData(
        ticker=snapshot.symbol,
        spot=spot,
        as_of=as_of.isoformat() if as_of is not None else None,
        r=r,
        q=q,
        iv_atm=iv_atm,
        wall_decay=wall_decay,
        regime=profile.regime,
        net_gex=profile.net_gex,
        total_ag=total_ag,
        gamma_score=gamma_score,
        vol_mult=vol_mult,
        call_wall=_finite_or_none(profile.call_wall),
        put_wall=_finite_or_none(profile.put_wall),
        gamma_flip=_finite_or_none(profile.gamma_flip),
        oi_quantile=oi_quantile,
        hv=_finite_or_none(hv),
        atr=_finite_or_none(atr),
        axis_min=float(lower_bound),
        axis_max=float(upper_bound),
        cone_path=cone_path,
        expirations=expirations,
        levels=globals_,
    )


def atr_14(
    highs,
    lows,
    closes,
    period: int = 14,
) -> Optional[float]:
    """Average True Range (Wilder, простым средним) из дневных свечей.

    True Range = max(H−L, |H−C_prev|, |L−C_prev|); ATR = среднее TR за
    ``period`` дней. Возвращает ``None`` при недостатке данных.
    """
    try:
        h = pd.Series([float(x) for x in highs]).dropna()
        l = pd.Series([float(x) for x in lows]).dropna()
        c = pd.Series([float(x) for x in closes]).dropna()
    except (TypeError, ValueError):
        return None
    n = min(len(h), len(l), len(c))
    if n < period + 2:
        return None
    h, l, c = h.tail(n), l.tail(n), c.tail(n)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    tr = tr.dropna()
    if len(tr) < period:
        return None
    atr = float(tr.tail(period).mean())
    return atr if math.isfinite(atr) and atr > 0 else None


def historical_vol(closes, period_days: int = _HV_PERIOD_DAYS) -> Optional[float]:
    """Историческая волатильность из дневных цен закрытия (годовая).

    ``closes`` — итерируемый ряд цен закрытия (старые → новые). Берутся
    последние ``period_days + 1`` значений, считаются лог-доходности,
    стандартное отклонение × √252. Если данных меньше 10 точек — ``None``.
    """
    try:
        s = pd.Series([float(c) for c in closes])
    except (TypeError, ValueError):
        return None
    s = s.dropna()
    if len(s) < 11:
        return None
    s = s.tail(period_days + 1)
    rets = np.log(s / s.shift(1)).dropna()
    if len(rets) < 10:
        return None
    vol = float(rets.std(ddof=1) * math.sqrt(252.0))
    return vol if math.isfinite(vol) and vol > 0 else None
