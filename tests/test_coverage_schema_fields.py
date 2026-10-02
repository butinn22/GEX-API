"""Схемные тесты аддитивных Phase-4 полей ``AutoCoverageOut``.

Проверяем обратную совместимость: старый словарь (без новых полей) валидируется,
новые поля дефолтятся в ``None``; при наличии — округляются (страйки → 2 знака,
``nearest_expiry_days`` → 1 знак); ``NaN`` коэрсится в ``None`` адаптером.
"""
from __future__ import annotations

from gex.schemas.auto_coverage import AutoCoverageOut, auto_coverage_to_schema


def _old_shape() -> dict:
    """Форма блока ``auto`` до появления Phase-4 полей (14 базовых)."""
    return {
        "mode": "auto",
        "resolved_days": 90.0,
        "resolved_expiries": 20,
        "sources_used": ["webull"],
        "primary_source": "webull",
        "fallback_used": False,
        "escalated": False,
        "partial": False,
        "expirations_merged": 3,
        "strike_count": 11,
        "total_oi": 12345.0,
        "sparse": False,
        "sparse_reasons": [],
        "elapsed_ms": 42,
    }


def test_old_shape_dict_validates_with_none_fields():
    out = AutoCoverageOut.model_validate(_old_shape())
    assert out.strike_min is None
    assert out.strike_max is None
    assert out.nearest_expiry_days is None


def test_fields_round_to_expected_precision():
    out = AutoCoverageOut.model_validate({
        **_old_shape(),
        "strike_min": 80.123456,
        "strike_max": 120.987654,
        "nearest_expiry_days": 7.456789,
    })
    assert out.strike_min == 80.12
    assert out.strike_max == 120.99
    assert out.nearest_expiry_days == 7.5


def test_nan_coerced_to_none_via_adapter():
    """``auto_coverage_to_schema``: NaN/None → None (дукт-типизированный источник)."""

    class _Obj:
        mode = "auto"
        resolved_days = 90.0
        resolved_expiries = 20
        sources_used = ["webull"]
        primary_source = "webull"
        fallback_used = False
        escalated = False
        partial = False
        expirations_merged = 3
        strike_count = 11
        total_oi = 12345.0
        sparse = False
        sparse_reasons = []
        elapsed_ms = 42
        strike_min = float("nan")
        strike_max = None
        nearest_expiry_days = float("nan")

    out = auto_coverage_to_schema(_Obj())
    assert out.strike_min is None
    assert out.strike_max is None
    assert out.nearest_expiry_days is None
