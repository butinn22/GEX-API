"""Порты стратегии: что приходит в неё снаружи (ring: ports).

``GEXContext`` — контракт на вход GEX-данных в решение. Он специально **не** знает ни о
``ExtendedGEXReport``, ни о сервисах: ``from_gex_analysis`` читает атрибуты и возвращает
``None``, если их нет, поэтому источник (живой расчёт, кэш, тест) значения не имеет.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from dataclasses import dataclass


@dataclass(frozen=True)
class GEXContext:
    regime: str
    gamma_flip: float | None = None
    net_gex: float = 0.0
    z_score: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    direction: str | None = None
    confidence: float | None = None

    @classmethod
    def from_gex_analysis(cls, analysis: Any) -> "GEXContext | None":
        try:
            regime = getattr(analysis, "regime", None)
            if regime is None:
                return None
            return cls(
                regime=str(regime),
                gamma_flip=getattr(analysis, "gamma_flip", None),
                net_gex=float(getattr(analysis, "profile", analysis).net_gex)
                if hasattr(getattr(analysis, "profile", analysis), "net_gex") else 0.0,
                z_score=getattr(getattr(analysis, "profile", analysis), "z_score", None),
                call_wall=getattr(getattr(analysis, "profile", analysis), "call_wall", None),
                put_wall=getattr(getattr(analysis, "profile", analysis), "put_wall", None),
                direction=getattr(analysis, "direction", None),
                confidence=getattr(analysis, "confidence", None),
            )
        except (AttributeError, TypeError, ValueError):
            return None
