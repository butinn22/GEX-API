"""Expected Value и риск-менеджмент торговых сетапов.

Постановка EV
-------------
Для дискретного набора исходов сделки :math:`\\{(p_i, \\Pi_i)\\}` (вероятность,
PnL в долларах) математическое ожидание:

.. math::
    \\mathrm{EV} = \\sum_i p_i\\,\\Pi_i

Торговый сетап описывается:
  * ценой входа :math:`S_0`, целевым уровнем (Take Profit) и уровнем стопа
    (Stop Loss), выраженными в цене базового актива;
  * вероятностями :math:`p_{tp}` касания TP и :math:`p_{sl}` касания SL,
    вычисляемыми через :class:`~gex.stochastic.StochasticEngine`;
  * размером позиции :math:`N` (в акциях/контрактах).

EV на сделку (долл.): :math:`\\mathrm{EV} = p_{tp}\\Pi_{tp} + p_{sl}\\Pi_{sl} + (1-p_{tp}-p_{sl})\\Pi_0`

где :math:`\\Pi_0` — PnL при не-срабатывании ни одного уровня (обычно закрываем
по текущей в момент экспирации, либо 0 для упрощённой модели).

Специальный сценарий: Put Wall bounce в positive-gamma
------------------------------------------------------
Классический сетап: цена достигает Put Wall (макс OI по путам / max отрицательной
гаммы) в режиме POSITIVE GAMMA. Поскольку в этом режиме дилеры давят цену вверх
(подавление волатильности, mean-reversion), ожидается отскок. Берём длинную
позицию, TP — Call Wall (или Gamma Flip), SL — ниже Put Wall.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gex.domain.metrics import GEXProfile
from gex.domain.stochastic import StochasticEngine, LevelProbability


@dataclass
class TradeSetup:
    """Описание торгового сетапа.

    Attributes
    ----------
    direction : 'long' | 'short'
        Направление сделки.
    entry : float
        Цена входа.
    take_profit : float
        Уровень фиксации прибыли (TP).
    stop_loss : float
        Уровень стоп-лосса (SL).
    size : float
        Размер позиции (количество акций / единиц базиса).
    label : str
        Человекочитаемое имя сетапа.
    """

    direction: str
    entry: float
    take_profit: float
    stop_loss: float
    size: float = 1.0
    label: str = "setup"

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short"):
            raise ValueError("direction должен быть 'long' или 'short'")
        if self.direction == "long" and not (self.stop_loss < self.entry < self.take_profit):
            raise ValueError(
                f"long-сетап требует SL({self.stop_loss}) < entry({self.entry}) < TP({self.take_profit})"
            )
        if self.direction == "short" and not (self.take_profit < self.entry < self.stop_loss):
            raise ValueError(
                f"short-сетап требует TP({self.take_profit}) < entry({self.entry}) < SL({self.stop_loss})"
            )


@dataclass
class EVResult:
    """Результат расчёта EV.

    Attributes
    ----------
    ev_dollars : float
        Мат. ожидание PnL в долларах на одну сделку (с учётом size).
    ev_per_share : float
        EV на единицу базиса.
    p_tp : float
        Вероятность касания TP.
    p_sl : float
        Вероятность касания SL.
    rr : float
        Risk/Reward соотношение :math:`|\\Pi_{tp}/\\Pi_{sl}|`.
    kelly : float
        Доля капитала по критерию Келли (упрощённая, биноминальная).
    edge : float
        Преимущество: :math:`p_{tp} - p_{tp}^{\\,*}` где :math:`p_{tp}^{\\,*}`
        — безубыточная вероятность :math:`=|\\Pi_{sl}|/(|\\Pi_{tp}|+|\\Pi_{sl}|)`.
    """

    ev_dollars: float
    ev_per_share: float
    p_tp: float
    p_sl: float
    rr: float
    kelly: float
    edge: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EV({self.ev_dollars:+,.2f}$ | per_share {self.ev_per_share:+.4f}) "
            f"P(TP)={self.p_tp:.3f} P(SL)={self.p_sl:.3f} RR=1:{self.rr:.2f} "
            f"Kelly={self.kelly:.3f} Edge={self.edge:+.3f}"
        )


class EVCalculator:
    """Калькулятор Expected Value для торговых сетапов.

    Parameters
    ----------
    engine : StochasticEngine
        Стохастический движок для расчёта вероятностей касания.
    """

    def __init__(self, engine: StochasticEngine):
        self.engine = engine

    # ------------------------------------------------------------------ #
    #  Базовый расчёт EV
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        setup: TradeSetup,
        profile: GEXProfile,
        T: float,
        sigma: float,
        mu: float = 0.0,
        p_close: float = 0.0,
        verbose: bool = False,
    ) -> EVResult:
        """Посчитать EV сетапа.

        Parameters
        ----------
        setup : TradeSetup
            Описание сделки.
        profile : GEXProfile
            Текущий GEX-профиль (для режима и параметров SDE).
        T : float
            Горизонт, лет.
        sigma : float
            Базовая волатильность, годовых.
        mu : float
            Дрейф (для GBM-режима).
        p_close : float
            PnL при «ничейном» исходе (ни TP, ни SL не достигнуты). В долларах
            на единицу базиса. По умолчанию 0.
        verbose : bool
            Печатать промежуточные вероятности.
        """
        entry, tp, sl = setup.entry, setup.take_profit, setup.stop_loss

        # --- PnL на единицу базиса ---
        if setup.direction == "long":
            pnl_tp = tp - entry   # >0
            pnl_sl = sl - entry   # <0
        else:
            pnl_tp = entry - tp   # >0
            pnl_sl = entry - sl   # <0

        # --- Вероятности касания TP и SL через стохастический движок ---
        p_tp_obj: LevelProbability = self.engine.touch_probability(
            entry, tp, T, sigma, profile, mu=mu
        )
        p_sl_obj: LevelProbability = self.engine.touch_probability(
            entry, sl, T, sigma, profile, mu=mu
        )
        # Корректируем совместную вероятность: нельзя допустить, чтобы их сумма >1.
        # Берём min с 1 и нормируем «нейтральный» исход на остаток.
        p_tp, p_sl = p_tp_obj.p_hit, p_sl_obj.p_hit
        if p_tp + p_sl > 1.0:
            s = p_tp + p_sl
            p_tp, p_sl = p_tp / s, p_sl / s
        p_neutral = max(0.0, 1.0 - p_tp - p_sl)

        if verbose:
            print(f"  [TP {tp:.2f}] {p_tp_obj}")
            print(f"  [SL {sl:.2f}] {p_sl_obj}")
            print(f"  P(neutral) = {p_neutral:.3f}")

        # --- EV ---
        ev_per_share = p_tp * pnl_tp + p_sl * pnl_sl + p_neutral * p_close
        ev_dollars = ev_per_share * setup.size

        # --- Метрики ---
        rr = abs(pnl_tp / pnl_sl) if pnl_sl != 0 else np.inf
        # Безубыточная вероятность
        p_be = abs(pnl_sl) / (abs(pnl_tp) + abs(pnl_sl)) if (abs(pnl_tp) + abs(pnl_sl)) > 0 else 0.0
        edge = p_tp - p_be
        # Упрощённый Kelly (биномиальная аппроксимация)
        kelly = self._kelly_fraction(p_tp, pnl_tp, pnl_sl)

        return EVResult(
            ev_dollars=ev_dollars,
            ev_per_share=ev_per_share,
            p_tp=p_tp,
            p_sl=p_sl,
            rr=rr,
            kelly=kelly,
            edge=edge,
        )

    # ------------------------------------------------------------------ #
    #  Келли
    # ------------------------------------------------------------------ #
    @staticmethod
    def _kelly_fraction(p: float, gain: float, loss: float) -> float:
        r"""Доля капитала по критерию Келли для биноминальной модели.

        .. math::
            f^* = \\frac{p\\,b - (1-p)}{b}, \\qquad b = \\frac{\\text{gain}}{|\text{loss}|}
        """
        b = abs(gain / loss) if loss != 0 else 0.0
        if b <= 0:
            return 0.0
        f = (p * b - (1.0 - p)) / b
        return float(np.clip(f, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Готовый сетап: Put Wall bounce в positive gamma
    # ------------------------------------------------------------------ #
    def put_wall_bounce(
        self,
        profile: GEXProfile,
        spot: float,
        T: float,
        sigma: float,
        sl_buffer: float = 0.002,
        tp_target: Optional[float] = None,
        size: float = 1.0,
        verbose: bool = False,
    ) -> tuple[TradeSetup, EVResult]:
        """Сетап «отскок от Put Wall в режиме POSITIVE GAMMA».

        Логика:
          * Вход — у Put Wall (ожидается поддержка).
          * TP — Gamma Flip (или Call Wall, если Flip близко к входу).
          * SL — чуть ниже Put Wall на ``sl_buffer`` (в долях от спота).

        Returns
        -------
        (TradeSetup, EVResult)
        """
        if profile.regime != "POSITIVE":
            if verbose:
                print(f"  WARN: режим {profile.regime}, сетап рассчитан для POSITIVE gamma.")
        put_wall = profile.put_wall
        if not np.isfinite(put_wall):
            raise ValueError("Put Wall не определён — профиль не содержит отрицательной гаммы.")

        entry = spot if np.isclose(spot, put_wall, rtol=0.01) else put_wall
        sl = entry * (1.0 - sl_buffer)
        tp = tp_target if tp_target is not None else (
            profile.gamma_flip if (profile.gamma_flip and profile.gamma_flip > entry)
            else profile.call_wall
        )
        if tp is None or not np.isfinite(tp) or tp <= entry:
            raise ValueError("Не удалось определить целевой уровень TP для отскока.")

        setup = TradeSetup(
            direction="long",
            entry=entry,
            take_profit=float(tp),
            stop_loss=float(sl),
            size=size,
            label="PutWall bounce (pos-gamma)",
        )
        result = self.evaluate(setup, profile, T, sigma, mu=0.0, verbose=verbose)
        return setup, result
