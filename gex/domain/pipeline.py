"""Оркестрация полного GEX-анализа: данные → метрики → вероятности → EV.

Этот модуль связывает все компоненты пакета в единый конвейер ежедневного
анализа. Используется как точка входа в продакшен-системе и в демонстрации.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from gex.domain.data_loader import GEXDataLoader, OptionSnapshot
from gex.domain.metrics import GEXMetrics, GEXProfile
from gex.domain.stochastic import StochasticEngine, LevelProbability
from gex.domain.ev import EVCalculator, TradeSetup, EVResult


@dataclass
class GEXReport:
    """Итоговый отчёт по одному активу на одну дату."""

    symbol: str
    as_of: datetime
    spot: float
    profile: GEXProfile
    p_flip: Optional[LevelProbability] = None
    p_call_wall: Optional[LevelProbability] = None
    p_put_wall: Optional[LevelProbability] = None
    setup: Optional[TradeSetup] = None
    ev: Optional[EVResult] = None

    def __repr__(self) -> str:  # pragma: no cover
        head = (f"=== GEX Report: {self.symbol} @ {self.as_of:%Y-%m-%d} "
                f"(spot={self.spot:.2f}) ===\n")
        head += self.profile.summary()
        for lbl, obj in (("Flip", self.p_flip),
                         ("CallWall", self.p_call_wall),
                         ("PutWall", self.p_put_wall)):
            if obj is not None:
                head += f"  P(touch {lbl:8s}) = {obj.p_hit:.3f}  [{obj.method}]\n"
        if self.setup is not None and self.ev is not None:
            head += (f"  Setup '{self.setup.label}': entry={self.setup.entry:.2f}, "
                     f"TP={self.setup.take_profit:.2f}, SL={self.setup.stop_loss:.2f}\n")
            head += f"  {self.ev}\n"
        return head


class GEXPipeline:
    """Полный конвейер: загрузка → GEX-профиль → вероятности → EV.

    Parameters
    ----------
    spot, symbol, r, q : float/str
        Параметры передаются в :class:`GEXDataLoader` и :class:`GEXMetrics`.
    call_sign, put_sign : float
        Знаки дилера (по умолчанию SqueezeMetrics). Для VIX передавайте
        ``call_sign=-1, put_sign=+1`` (см. :class:`GEXMetrics`).
    """

    def __init__(
        self,
        spot: float,
        symbol: str = "SPX",
        r: float = 0.045,
        q: float = 0.0,
        per_contract: int = 100,
        call_sign: float = GEXMetrics.CALL_SIGN,
        put_sign: float = GEXMetrics.PUT_SIGN,
    ):
        self.loader = GEXDataLoader(spot=spot, symbol=symbol, contract_multiplier=per_contract)
        self.metrics = GEXMetrics(
            spot=spot, r=r, q=q, per_contract=per_contract,
            call_sign=call_sign, put_sign=put_sign,
        )
        self.engine = StochasticEngine()
        self.ev_calc = EVCalculator(self.engine)
        self.symbol = symbol
        self.spot = spot

    # ------------------------------------------------------------------ #
    #  Главный запуск
    # ------------------------------------------------------------------ #
    def run(
        self,
        snapshot: OptionSnapshot,
        sigma: Optional[float] = None,
        T: Optional[float] = None,
        mu: float = 0.0,
        run_put_wall_setup: bool = True,
        verbose: bool = False,
    ) -> GEXReport:
        """Прогнать полный анализ.

        Parameters
        ----------
        snapshot : OptionSnapshot
            Очищенная цепочка.
        sigma : float, optional
            ATM-волатильность. Если None — берётся OI-взвешенная медиана IV.
        T : float, optional
            Горизонт прогноза, лет. Если None — медиана T по цепочке.
        mu : float
            Дрейф для GBM-режима.
        run_put_wall_setup : bool
            Запустить готовый сетап «отскок от Put Wall» (только для POSITIVE).
        """
        # --- 1. GEX-профиль ---
        if T is None:
            T = float(np.median(snapshot.chain["T"]))
        profile = self.metrics.compute(snapshot, horizon_years=T)
        if sigma is None:
            w = snapshot.chain["oi"].values
            w = w / w.sum() if w.sum() > 0 else None
            sigma = float(np.average(snapshot.chain["iv"].values, weights=w))

        # --- 2. Вероятности касания ключевых уровней ---
        p_flip, p_cw, p_pw = None, None, None
        if profile.gamma_flip is not None:
            p_flip = self.engine.touch_probability(
                self.spot, profile.gamma_flip, T, sigma, profile, mu=mu)
        if np.isfinite(profile.call_wall):
            p_cw = self.engine.touch_probability(
                self.spot, profile.call_wall, T, sigma, profile, mu=mu)
        if np.isfinite(profile.put_wall):
            p_pw = self.engine.touch_probability(
                self.spot, profile.put_wall, T, sigma, profile, mu=mu)

        # --- 3. Торговый сетап ---
        setup, ev = None, None
        if run_put_wall_setup and profile.regime == "POSITIVE":
            try:
                setup, ev = self.ev_calc.put_wall_bounce(
                    profile, self.spot, T, sigma, verbose=verbose)
            except ValueError as e:
                if verbose:
                    print(f"  Setup skipped: {e}")

        return GEXReport(
            symbol=self.symbol,
            as_of=snapshot.as_of,
            spot=self.spot,
            profile=profile,
            p_flip=p_flip,
            p_call_wall=p_cw,
            p_put_wall=p_pw,
            setup=setup,
            ev=ev,
        )
