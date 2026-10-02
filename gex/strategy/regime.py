"""regime: Режим рынка: параметры, слайдеры, метрики и вердикт, а также допуск сигналов.

Вынесено из ``gex/trading_algorithm.py`` (итерация 37). Методы перенесены дословно: разбиение god-класса не должно менять числа, а доказательство — golden-эталон
``tests/test_strategy_golden.py``, сверяющий все колонки кадра, решения ``evaluate``,
режим, оценку входа и риск до и после выноса.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from gex.domain.trend_regime import (
    RegimeParams,
    RegimeSliders,
    compute_regime_metrics,
    evaluate_regime,
    signal_allowed,
)


class _RegimeMixin:
    def regime_params(self) -> RegimeParams:
        """Параметры детектора тренда/флэта из настроек стратегии."""
        s = self.settings
        return RegimeParams(
            window=int(s.regime_window),
            recent=int(s.regime_recent),
            atr_period=int(s.regime_atr_period),
            bb_period=int(s.regime_bb_period),
            bb_mult=float(s.regime_bb_mult),
            w_price_recent=float(s.regime_price_recent_weight),
        )


    def regime_sliders(self) -> RegimeSliders:
        """Слайдеры флэта из настроек стратегии (дефолт для анонима)."""
        s = self.settings
        return RegimeSliders(
            flat=float(s.flat_slider),
            atr=s.flat_slider_atr,
            bbw=s.flat_slider_bbw,
            pct=s.flat_slider_pct,
            flat_score_threshold=float(s.flat_score_threshold),
            trend_strength_low=float(s.trend_strength_low),
        )


    def trend_regime_metrics(self, ohlc: Any, i: int = -1) -> dict[str, Any] | None:
        """Слайдер-НЕЗАВИСИМЫЕ метрики режима на баре ``i``.

        ``None`` — если истории не хватает (нужно ≥ ``window + 5`` баров):
        тогда верифицировать нечем и сигналы НЕ режутся.
        """
        return compute_regime_metrics(ohlc, i=i, params=self.regime_params())


    def trend_regime(
        self,
        ohlc: Any,
        i: int = -1,
        sliders: RegimeSliders | None = None,
    ) -> dict[str, Any] | None:
        """Метрики + вердикт (``state``/``trend_strength``/``flat_score``/``is_flat``)."""
        metrics = self.trend_regime_metrics(ohlc, i=i)
        verdict = evaluate_regime(metrics, sliders or self.regime_sliders())
        if verdict is None:
            return None
        return {**verdict, "metrics": metrics}


    def regime_signal_allowed(
        self,
        order_type: str | None,
        verdict: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        """Пропускать ли сигнал при текущем вердикте режима.

        При ``flat_filter_signals=False`` (дефолт) всегда ``(True, None)`` —
        поведение остальных страниц не меняется.
        """
        if not self.settings.flat_filter_signals:
            return True, None
        return signal_allowed(
            order_type, verdict, gate_exits=bool(self.settings.flat_gate_exits)
        )
