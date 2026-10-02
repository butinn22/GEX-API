"""Расчёт GEX-профиля и ключевых уровней.

Модель дельта-хеджирования дилеров
----------------------------------
Классическая модель (SqueezeMetrics / Kownatzki) предполагает, что маркетмейкеры
(«дилеры») дельта-хеджируют свои позиции. Знак вклада опциона в суммарную гамму
дилера зависит от того, *купил* он опцион (положительная гамма) или *продал*
(отрицательная гамма). Гамма опциона всегда положительна, знак даёт позиция.

Отраслевое соглашение о потоке (flow assumption) для US equity/index:

  * Инвесторы **продают** коллы (covered calls / overlays) → дилеры **покупают**
    коллы → коллы дают **положительную** гамму дилеру.
  * Инвесторы **покупают** путы (защитные позиции) → дилеры **продают** путы
    → путы дают **отрицательную** гамму дилеру.

Итого: ``call_sign = +1``, ``put_sign = -1``. Это конвенция SqueezeMetrics,
общепринятая в GEX-аналитике (SpotGamma, Tier1Alpha и др.).

Для **индексов волатильности (VIX/VVIX)** конвенция обратная: институционалы
**покупают** и коллы, и путы (ставки на рост/падение волатильности, хедж
дельта-хеджируемых портфелей) → дилеры **продают** оба типа. В этом случае
в :class:`GEXMetrics` передаётся ``call_sign=-1, put_sign=+1``. Симметричная
инверсия сохраняет структуру модели (стены, Gamma Flip, смена режима) —
меняется лишь то, какие страйки оказываются Call/Put Wall.

Net gamma exposure одного опциона (в долларах на 1% движения цены):

.. math::
    \\mathrm{GEX}_i = \\mathrm{sign}_i \\cdot \\Gamma_i \\cdot OI_i \\cdot 100 \\cdot S^2 \\cdot 0.01

где :math:`\\Gamma_i` — BSM-гамма, :math:`OI_i` — открытый интерес, 100 —
контрактный множитель, :math:`S^2 \\cdot 0.01` переводит «per-1% спота» в доллары.

Экономический смысл
~~~~~~~~~~~~~~~~~~~
* **Положительная суммарная гамма** (Net GEX > 0): при росте цены дилеры продают
  базис (дельта растёт → её сбрасывают), при падении — покупают. Это **подавляет
  волатильность** (mean-reversion, рынок «приклеивается» к страйкам).
* **Отрицательная суммарная гамма** (Net GEX < 0): при росте цены дилеры докупают,
  при падении — распродают. Это **усиливает тренды** (trend-boosting, «гамма-пин»
  ломается, волатильность растёт).

Определения уровней
~~~~~~~~~~~~~~~~~~~
* **Call Wall** — страйк с максимальным **положительным** net GEX (сопротивление
  сверху, спот обычно ниже него). На нём dealers long gamma, движения гасятся.
* **Put Wall** — страйк с максимальным **отрицательным** net GEX (поддержка
  снизу, спот обычно выше него).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from gex.domain.greeks import bs_gamma, bs_delta
from gex.domain.data_loader import OptionSnapshot


@dataclass
class GEXProfile:
    """Полный GEX-профиль рыночной структуры на одну дату.

    Attributes
    ----------
    per_strike : pd.DataFrame
        Таблица по страйкам: strike, gex_call, gex_put, gex_net, gamma_call,
        gamma_put, oi_call, oi_put, gex_abs.
    net_gex : float
        Суммарный GEX (в $/%spot), знак задаёт рыночный режим.
    gamma_flip : Optional[float]
        Цена базового актива, при которой net_gex(S) = 0. ``None`` если перехода нет.
    call_wall : float
        Страйк с максимальным положительным GEX (сопротивление).
    put_wall : float
        Страйк с максимальным отрицательным GEX (поддержка).
    call_wall_oi : float
        Страйк с максимальным OI по коллам (альтернативное определение стены).
    put_wall_oi : float
        Страйк с максимальным OI по путам.
    secondary_call_walls : list[float]
        Дополнительные Call Walls: страйки с наибольшим положительным GEX
        (без учёта primary), упорядочены по убыванию |GEX|.
    secondary_put_walls : list[float]
        Дополнительные Put Walls: страйки с наибольшим отрицательным GEX
        (без учёта primary), упорядочены по убыванию |GEX|.
    regime : str
        'POSITIVE' | 'NEGATIVE'.
    z_score : Optional[float]
        Z-score отклонения спота от Gamma Flip (нормировка на ATM-волу за горизонт).
    """

    per_strike: pd.DataFrame
    net_gex: float
    gamma_flip: Optional[float]
    call_wall: float
    put_wall: float
    call_wall_oi: float
    put_wall_oi: float
    regime: str
    z_score: Optional[float]
    # Поля с дефолтами — обязательно после полей без дефолта (правило dataclass).
    secondary_call_walls: list[float] = field(default_factory=list)
    secondary_put_walls: list[float] = field(default_factory=list)

    def summary(self) -> str:
        gf = f"{self.gamma_flip:.2f}" if self.gamma_flip is not None else "—"
        zs = f"{self.z_score:+.2f}" if self.z_score is not None else "—"
        return (
            f"GEX Profile\n"
            f"  Net GEX     : {self.net_gex:+,.0f} $/%spot\n"
            f"  Regime      : {self.regime}\n"
            f"  Gamma Flip  : {gf}\n"
            f"  Call Wall   : {self.call_wall:.2f}  (OI-wall: {self.call_wall_oi:.2f})\n"
            f"  Put Wall    : {self.put_wall:.2f}  (OI-wall: {self.put_wall_oi:.2f})\n"
            f"  Z(vs Flip)  : {zs}\n"
        )


class GEXMetrics:
    """Расчётник GEX-профиля и структурных уровней.

    Parameters
    ----------
    spot, r, q : float
        Рыночные параметры для греков (q>0 для индексов с дивидендами).
    per_contract : int
        Множитель контракта (100 для большинства US equity/индексов).
    pct_move : float
        На какой процентный ход базиса переводить GEX (по умолчанию 1%).
    call_sign, put_sign : float
        Знак вклада колла/пута в суммарную дилерскую гамму (поток-конвенция).
        По умолчанию SqueezeMetrics для equity/index: дилеры **покупают** коллы
        (``+1``) и **продают** путы (``-1``). Для VIX конвенция обратная
        (институционалы покупают коллы как хедж → дилеры шортят коллы), там
        передаётся ``call_sign=-1, put_sign=+1``.
    """

    # Дефолтные знаки SqueezeMetrics (для equity/индексов/фьючерсов MOEX).
    # Для индексов волатильности (VIX) передавайте инвертированную пару в __init__.
    CALL_SIGN = +1.0   # дилер long calls  → положительная гамма (сопротивление сверху)
    PUT_SIGN = -1.0    # дилер short puts  → отрицательная гамма (поддержка снизу)

    def __init__(
        self,
        spot: float,
        r: float = 0.045,
        q: float = 0.0,
        per_contract: int = 100,
        pct_move: float = 0.01,
        call_sign: float = CALL_SIGN,
        put_sign: float = PUT_SIGN,
    ):
        if spot <= 0:
            raise ValueError("spot должен быть > 0")
        self.S = float(spot)
        self.r = float(r)
        self.q = float(q)
        self.per_contract = int(per_contract)
        self.pct_move = float(pct_move)
        self.call_sign = float(call_sign)
        self.put_sign = float(put_sign)

    # ------------------------------------------------------------------ #
    #  Главный API
    # ------------------------------------------------------------------ #
    def compute(self, snapshot: OptionSnapshot, horizon_years: Optional[float] = None) -> GEXProfile:
        """Построить полный GEX-профиль для снапшота.

        Parameters
        ----------
        snapshot : OptionSnapshot
            Очищенная опционная цепочка.
        horizon_years : float, optional
            Горизонт для расчёта Z-score (например, 1 торговый день = 1/252).
            Если ``None``, берётся медианное T по цепочке.
        """
        chain = snapshot.chain
        spot = self.S

        # --- Греки векторизованно ---
        gamma = bs_gamma(spot, chain["strike"], chain["T"], self.r, chain["iv"], self.q)
        delta = bs_delta(spot, chain["strike"], chain["T"], self.r, chain["iv"], self.q,
                         is_call=(chain["type"] == "C").values)

        # --- Знак дилера по типу опциона ---
        sign = np.where(chain["type"].values == "C", self.call_sign, self.put_sign)

        # --- GEX в $ на 1% движения спота ---
        gex_per_contract = (
            sign * gamma * self.per_contract * (spot ** 2) * self.pct_move
        )
        gex_total = gex_per_contract * chain["oi"].values

        # --- Группировка по страйкам ---
        chain = chain.assign(
            gamma=gamma,
            delta=delta,
            gex=gex_total,
            sign=sign,
        )
        per_strike = self._aggregate_by_strike(chain, snapshot.as_of)

        net_gex = float(per_strike["gex_net"].sum())
        regime = "POSITIVE" if net_gex >= 0 else "NEGATIVE"

        # --- Ключевые уровни ---
        call_wall, put_wall = self._walls_by_gex(per_strike)
        call_wall_oi, put_wall_oi = self._walls_by_oi(per_strike)
        # Gamma Flip: метод Брента на GEX_total(S) (chain = исходная цепочка
        # с колонками strike/type/oi/iv/T — пересчёт гаммы при гипотетическом S).
        gamma_flip = self._find_gamma_flip(per_strike, raw_chain=chain)
        secondary_call_walls, secondary_put_walls = self._secondary_walls(
            per_strike, call_wall, put_wall
        )

        # --- Z-score спота относительно Gamma Flip ---
        T = horizon_years if horizon_years is not None else float(np.median(chain["T"]))
        z = self._z_score_vs_flip(spot, gamma_flip, chain, T)

        return GEXProfile(
            per_strike=per_strike,
            net_gex=net_gex,
            gamma_flip=gamma_flip,
            call_wall=call_wall,
            put_wall=put_wall,
            call_wall_oi=call_wall_oi,
            put_wall_oi=put_wall_oi,
            secondary_call_walls=secondary_call_walls,
            secondary_put_walls=secondary_put_walls,
            regime=regime,
            z_score=z,
        )

    # ------------------------------------------------------------------ #
    #  Агрегация по страйкам
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate_by_strike(
        chain: pd.DataFrame,
        as_of: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Свернуть (call+put) на каждом страйке в одну строку GEX-профиля.

        Для каждого страйка считается диапазон экспираций:
          * ``t_min`` / ``t_max`` — мин/макс время до экспирации, **дни**;
          * ``expiry_from`` / ``expiry_to`` — соответствующие календарные даты
            (``as_of + t_* дней``). Если ``as_of`` не задан — ``None``.
        """
        g = chain.groupby("strike", sort=True)
        rows = []
        for k, sub in g:
            calls = sub[sub["type"] == "C"]
            puts = sub[sub["type"] == "P"]
            # Диапазон экспираций по всем опционам на этом strike
            t_days = sub["T"].values * 365.0  # T в годах → дни
            t_min = float(t_days.min()) if len(t_days) > 0 else 0.0
            t_max = float(t_days.max()) if len(t_days) > 0 else 0.0
            if as_of is not None:
                # Нормализуем as_of к datetime (мог прийти pd.Timestamp).
                base = pd.Timestamp(as_of).to_pydatetime()
                expiry_from = (base + timedelta(days=t_min)).date()
                expiry_to = (base + timedelta(days=t_max)).date()
            else:
                expiry_from = None
                expiry_to = None
            rows.append({
                "strike": k,
                "gex_call": float(calls["gex"].sum()),
                "gex_put": float(puts["gex"].sum()),
                "gamma_call": float(calls["gamma"].sum()),
                "gamma_put": float(puts["gamma"].sum()),
                "oi_call": float(calls["oi"].sum()),
                "oi_put": float(puts["oi"].sum()),
                "t_min": t_min,
                "t_max": t_max,
                "expiry_from": expiry_from,
                "expiry_to": expiry_to,
            })
        df = pd.DataFrame(rows)
        # Net GEX по страйку и абсолютное значение для поиска стен
        df["gex_net"] = df["gex_call"] + df["gex_put"]
        df["gex_abs"] = df["gex_net"].abs()
        df["oi_total"] = df["oi_call"] + df["oi_put"]
        return df

    # ------------------------------------------------------------------ #
    #  Call/Put Walls
    # ------------------------------------------------------------------ #
    @staticmethod
    def _walls_by_gex(per_strike: pd.DataFrame) -> tuple[float, float]:
        """Call Wall = страйк с max положительной гаммой (сопротивление сверху);
        Put Wall  = страйк с max отрицательной гаммой (поддержка снизу)."""
        if per_strike.empty:
            return np.nan, np.nan
        # Call Wall: максимум положительного net GEX (защита от отбора отрицательных)
        pos = per_strike[per_strike["gex_net"] > 0]
        neg = per_strike[per_strike["gex_net"] < 0]
        call_wall = pos.loc[pos["gex_net"].idxmax(), "strike"] if not pos.empty else np.nan
        put_wall = neg.loc[neg["gex_net"].idxmin(), "strike"] if not neg.empty else np.nan
        return float(call_wall), float(put_wall)

    @staticmethod
    def _walls_by_oi(per_strike: pd.DataFrame) -> tuple[float, float]:
        """Стены по открытому интересу (макс OI по коллам / путам)."""
        if per_strike.empty:
            return np.nan, np.nan
        call_wall = per_strike.loc[per_strike["oi_call"].idxmax(), "strike"]
        put_wall = per_strike.loc[per_strike["oi_put"].idxmax(), "strike"]
        return float(call_wall), float(float(put_wall))

    @staticmethod
    def ranked_walls(per_strike: pd.DataFrame, top_n: int = 3) -> tuple[list[float], list[float]]:
        """До ``top_n`` сильнейших call/put стен, ранжированных по |GEX| по убыванию.

        Индекс 0 = primary wall, далее — secondary. Страйки с положительным
        net GEX → call walls, с отрицательным → put walls (как в
        :meth:`_walls_by_gex`), упорядочены по убыванию абсолютного GEX.

        Удобнее, чем раздельные ``_walls_by_gex`` + ``_secondary_walls``: один
        вызов сразу даёт и primary, и secondary стены нужной длины.
        """
        if per_strike.empty:
            return [], []
        pos = per_strike[per_strike["gex_net"] > 0].sort_values("gex_abs", ascending=False)
        neg = per_strike[per_strike["gex_net"] < 0].sort_values("gex_abs", ascending=False)
        call_walls = pos["strike"].head(top_n).astype(float).tolist()
        put_walls = neg["strike"].head(top_n).astype(float).tolist()
        return call_walls, put_walls

    @staticmethod
    def _secondary_walls(
        per_strike: pd.DataFrame,
        primary_call: float,
        primary_put: float,
        n: int = 2,
    ) -> tuple[list[float], list[float]]:
        """Дополнительные стены: n сильнейших по |GEX| с каждой стороны
        (исключая primary). Реализация через :meth:`ranked_walls` — единое
        ранжирование, primary отбрасывается.
        """
        if per_strike.empty:
            return [], []
        # Берём с запасом (top_n+1) и отбрасываем primary — получаем n secondary.
        call_ranked, put_ranked = GEXMetrics.ranked_walls(per_strike, top_n=n + 1)
        sec_call = [s for s in call_ranked if not np.isclose(float(s), float(primary_call))][:n]
        sec_put = [s for s in put_ranked if not np.isclose(float(s), float(primary_put))][:n]
        return sec_call, sec_put

    # ------------------------------------------------------------------ #
    #  Gamma Flip
    # ------------------------------------------------------------------ #
    def _gex_total_at(self, spot_hypo: float, raw_chain: pd.DataFrame) -> float:
        """Суммарная дилерская GEX при гипотетическом споте ``spot_hypo``.

        В отличие от агрегированного ``per_strike`` (где гамма считается один
        раз при текущем споте), здесь гамма **пересчитывается** для каждого
        опциона при гипотетической цене S — это «истинная» поверхность
        GEX_total(S), корень которой и есть Gamma Flip.

        Векторизовано через ``bs_gamma`` (принимает массивы страйков) — быстро
        даже при многократных вызовах из ``brentq``.
        """
        gamma = bs_gamma(
            spot_hypo, raw_chain["strike"], raw_chain["T"],
            self.r, raw_chain["iv"], self.q,
        )
        sign = np.where(raw_chain["type"].values == "C", self.call_sign, self.put_sign)
        gex = (
            sign * gamma * self.per_contract
            * (spot_hypo ** 2) * self.pct_move * raw_chain["oi"].values
        )
        return float(gex.sum())

    def _find_gamma_flip(
        self,
        per_strike: pd.DataFrame,
        raw_chain: Optional[pd.DataFrame] = None,
        search_pct: float = 0.15,
    ) -> Optional[float]:
        """Найти уровень цены, где суммарная дилерская GEX = 0.

        Основной метод — метод Брента (``brentq``) на «истинной» функции
        ``GEX_total(S)`` (пересчёт гаммы при гипотетическом споте). Это точнее
        дискретной интерполяции по кумулятиве, т.к. учитывает зависимость
        гаммы каждого опциона от цены базиса.

        Fallback (если нет смены знака на сетке поиска или недоступен scipy):
        старый кумулятивно-интерполяционный метод ``_find_gamma_flip_cumulative``.

        Parameters
        ----------
        per_strike : pd.DataFrame
            Агрегированный профиль по страйкам (для fallback).
        raw_chain : pd.DataFrame, optional
            Исходная цепочка опционов (для пересчёта гаммы). Если ``None`` —
            сразу используется кумулятивный fallback.
        search_pct : float
            Полуширина диапазона поиска вокруг текущего спота, в долях (0.15 = ±15%).
        """
        if per_strike.empty:
            return None
        if raw_chain is None or raw_chain.empty:
            return self._find_gamma_flip_cumulative(per_strike)

        spot = self.S
        lo, hi = spot * (1.0 - search_pct), spot * (1.0 + search_pct)
        f_lo = self._gex_total_at(lo, raw_chain)
        f_hi = self._gex_total_at(hi, raw_chain)

        # Нет смены знака на сетке поиска → рынок глубоко в одном режиме,
        # Gamma Flip вне диапазона. Падаем на кумулятивный метод (он ищет
        # переход по всему набору страйков).
        if f_lo * f_hi > 0:
            return self._find_gamma_flip_cumulative(per_strike)

        try:
            from scipy.optimize import brentq
            s_star = brentq(
                lambda s: self._gex_total_at(s, raw_chain),
                lo, hi, xtol=1e-4,
            )
            return float(s_star)
        except (ImportError, ValueError, RuntimeError):
            # scipy недоступен / не сошёлся → кумулятивный fallback.
            return self._find_gamma_flip_cumulative(per_strike)

    @staticmethod
    def _find_gamma_flip_cumulative(per_strike: pd.DataFrame) -> Optional[float]:
        """Fallback: Gamma Flip через кумулятиву GEX по страйкам.

        Стратегия:
          1. Строим кумулятивную сумму GEX снизу вверх по страйкам.
          2. Ищем смежные страйки, где знак cumulative меняется.
          3. Линейно интерполируем цену перехода между ними.

        Эквивалентно корню :math:`G(S^*)=0` на дискретном профиле при
        фиксированном (текущем) споте. Менее точно, чем Брент, но устойчиво
        работает, когда на сетке поиска нет смены знака.
        """
        if per_strike.empty:
            return None
        df = per_strike.sort_values("strike").reset_index(drop=True)
        cum = df["gex_net"].cumsum()
        # Сдвигаем для поиска переходов через ноль
        sign_change = np.sign(cum).diff().fillna(0) != 0
        idx = np.where(sign_change.values)[0]
        if len(idx) == 0:
            # Нет перехода: рынок глубоко в одном режиме → flip вне диапазона страйков
            return None
        i = idx[0]
        # Линейная интерполяция между страйками (i-1) и i
        s0, s1 = df.loc[i - 1, "strike"], df.loc[i, "strike"]
        g0, g1 = cum.iloc[i - 1], cum.iloc[i]
        if np.isclose(g1 - g0, 0.0):
            return float(0.5 * (s0 + s1))
        # s* = s0 - g0 * (s1 - s0) / (g1 - g0)
        s_star = s0 - g0 * (s1 - s0) / (g1 - g0)
        return float(s_star)

    # ------------------------------------------------------------------ #
    #  Z-score: отклонение спота от Gamma Flip
    # ------------------------------------------------------------------ #
    @staticmethod
    def _z_score_vs_flip(
        spot: float,
        gamma_flip: Optional[float],
        chain: pd.DataFrame,
        horizon_years: float,
    ) -> Optional[float]:
        """Z-score отклонения спота от Gamma Flip, нормированный на ожидаемое
        стандартное отклонение доходности за горизонт:

        .. math::
            Z = \\frac{\\ln(S / S^*)}{\\sigma_{ATM}\\sqrt T}

        :math:`\\sigma_{ATM}` — ATM implied волатильность (медианная по цепочке
        как робастный прокси, чтобы не зависеть от единственного страйка).
        """
        if gamma_flip is None or gamma_flip <= 0:
            return None
        # Робастная ATM-вола: медиана IV, взвешенная по OI (стабильна к крыльям)
        w = chain["oi"].values
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        atm_vol = float(np.sum(chain["iv"].values * w))
        vol = atm_vol * np.sqrt(max(horizon_years, 1e-9))
        if vol < 1e-9:
            return None
        return float(np.log(spot / gamma_flip) / vol)
