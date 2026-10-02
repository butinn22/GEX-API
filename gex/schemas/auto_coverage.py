"""Pydantic schema for AUTO-mode coverage metadata (design doc §5).

Lives in its own module so it can be imported by **both** ``extended_schemas``
(the ``/ext/gex`` response) and ``gex_analysis`` (the ``/moex/gex`` response)
without creating a schema import cycle.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AutoCoverageOut(BaseModel):
    """Serialisable dataset-quality block for a GEX report.

    Аудит 2026-09-17: блок больше **не** только AUTO. Метаданные охвата
    считаются и в ручном режиме (``mode="manual"``) — иначе качество выборки
    видно лишь при AUTO, а «красивый» профиль на скудных данных выглядит
    одинаково убедительно в любом режиме. Все поля ``Optional``/с дефолтами,
    поэтому для старых клиентов блок по-прежнему обратно совместим.
    """

    model_config = {"from_attributes": True}

    mode: Literal["auto", "manual"] = Field(
        "auto", description="Режим, в котором посчитан охват: auto | manual"
    )
    resolved_days: float = Field(..., description="Горизонт анализа, дней")
    resolved_expiries: int = Field(
        ..., description="Число экспираций (запрошенный кап; 0 = ALL на MOEX-пути)"
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Real providers contributing rows, in priority order",
    )
    primary_source: str = Field(..., description="Same as the top-level `source`")
    fallback_used: bool = Field(False, description="A fallback fetch contributed rows")
    escalated: bool = Field(
        False, description="Escalation was attempted (sparse + fallback exists)"
    )
    partial: bool = Field(
        False, description="Escalation attempted but the fallback fetch failed"
    )
    expirations_merged: int = Field(
        0, description="Distinct expiry buckets (rounded T-days) in the final chain"
    )
    strike_count: int = Field(0, description="len(report.per_strike)")
    total_oi: float = Field(0.0, description="Σ(oi_call + oi_put) over the final profile")
    sparse: bool = Field(False, description="Final profile is still thin after escalation")
    sparse_reasons: list[str] = Field(
        default_factory=list, description="Machine-readable reason codes"
    )
    elapsed_ms: int = Field(0, description="Wall time of the AUTO computation, ms")
    strike_min: Optional[float] = Field(
        None, description="Lowest strike in the final chain (additive Phase-4 field)"
    )
    strike_max: Optional[float] = Field(
        None, description="Highest strike in the final chain (additive Phase-4 field)"
    )
    nearest_expiry_days: Optional[float] = Field(
        None, description="Days to the nearest expiry bucket in the final chain"
    )
    furthest_expiry_days: Optional[float] = Field(
        None,
        description="Days to the furthest expiry bucket — верхняя граница "
                    "использованного диапазона данных (additive, аудит 2026-09-17)",
    )

    @model_validator(mode="after")
    def _round(self) -> "AutoCoverageOut":
        self.resolved_days = round(self.resolved_days, 1)
        self.total_oi = round(self.total_oi)
        if self.nearest_expiry_days is not None:
            self.nearest_expiry_days = round(self.nearest_expiry_days, 1)
        if self.furthest_expiry_days is not None:
            self.furthest_expiry_days = round(self.furthest_expiry_days, 1)
        if self.strike_min is not None:
            self.strike_min = round(self.strike_min, 2)
        if self.strike_max is not None:
            self.strike_max = round(self.strike_max, 2)
        return self


def auto_coverage_to_schema(coverage: Any) -> AutoCoverageOut:
    """Adapt an :class:`~gex.application.auto_scope.AutoCoverage` into the API schema.

    Reads attributes duck-typed, so it also accepts an already-built schema.
    """
    source = coverage
    return AutoCoverageOut(
        mode=getattr(source, "mode", "auto"),
        resolved_days=float(getattr(source, "resolved_days", 0.0)),
        resolved_expiries=int(getattr(source, "resolved_expiries", 0)),
        sources_used=list(getattr(source, "sources_used", []) or []),
        primary_source=str(getattr(source, "primary_source", "")),
        fallback_used=bool(getattr(source, "fallback_used", False)),
        escalated=bool(getattr(source, "escalated", False)),
        partial=bool(getattr(source, "partial", False)),
        expirations_merged=int(getattr(source, "expirations_merged", 0)),
        strike_count=int(getattr(source, "strike_count", 0)),
        total_oi=float(getattr(source, "total_oi", 0.0)),
        sparse=bool(getattr(source, "sparse", False)),
        sparse_reasons=list(getattr(source, "sparse_reasons", []) or []),
        elapsed_ms=int(getattr(source, "elapsed_ms", 0)),
        strike_min=_optional_float(getattr(source, "strike_min", None)),
        strike_max=_optional_float(getattr(source, "strike_max", None)),
        nearest_expiry_days=_optional_float(getattr(source, "nearest_expiry_days", None)),
        furthest_expiry_days=_optional_float(getattr(source, "furthest_expiry_days", None)),
    )


def _optional_float(value: object) -> Optional[float]:
    """Coerce a duck-typed value to ``float | None`` (None/NaN stay None)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN → None
