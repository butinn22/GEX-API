"""Value objects: Price (tick-rounded), Quantity (lot-rounded), Money.

The whole point of these types is to round at the boundary: a broker rejects
any price/quantity that is not a multiple of the instrument's tick/lot size, and
floating-point drift otherwise produces untradeable values. Rounding lives here
so upstream code can pass raw floats and get a tradable value out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Union

__all__ = ["Price", "Quantity", "Money"]

_Number = Union[int, float]


def _round_to_step(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step`` (ties away from zero).

    Uses ``Decimal`` (exact) rather than float division: ``0.25 / 0.1`` is not
    exactly 2.5 in binary float, which would silently round the wrong way.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    ratio = (Decimal(str(value)) / Decimal(str(step))).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return float(ratio * Decimal(str(step)))


@dataclass(frozen=True, order=True)
class Price:
    """A price, optionally rounded to the instrument's tick size (0 = no rounding)."""

    value: float
    tick_size: float = 0.0

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("price must be >= 0")
        if self.tick_size < 0:
            raise ValueError("tick_size must be >= 0")
        if self.tick_size > 0:
            object.__setattr__(self, "value", _round_to_step(self.value, self.tick_size))

    def __float__(self) -> float:
        return self.value

    def __add__(self, other: _Number | "Price") -> "Price":
        o = float(other)
        return Price(self.value + o, self.tick_size)

    def __sub__(self, other: _Number | "Price") -> "Price":
        o = float(other)
        return Price(self.value - o, self.tick_size)

    def __mul__(self, other: _Number) -> "Price":
        return Price(self.value * float(other), self.tick_size)

    __rmul__ = __mul__


@dataclass(frozen=True, order=True)
class Quantity:
    """A quantity (units of an instrument), rounded to the lot size (0 = free)."""

    value: float
    lot_size: float = 0.0

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("quantity must be >= 0")
        if self.lot_size < 0:
            raise ValueError("lot_size must be >= 0")
        if self.lot_size > 0:
            object.__setattr__(self, "value", _round_to_step(self.value, self.lot_size))

    def __float__(self) -> float:
        return self.value

    def __add__(self, other: _Number | "Quantity") -> "Quantity":
        o = float(other)
        return Quantity(self.value + o, self.lot_size)

    def __sub__(self, other: _Number | "Quantity") -> "Quantity":
        o = float(other)
        return Quantity(self.value - o, self.lot_size)

    def __mul__(self, other: _Number) -> "Quantity":
        return Quantity(self.value * float(other), self.lot_size)

    __rmul__ = __mul__


@dataclass(frozen=True)
class Money:
    """An amount in a currency. Arithmetic enforces a single currency."""

    amount: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", round(self.amount, 12))

    def _check(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"currency mismatch: {self.currency} vs {other.currency}"
            )

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: _Number) -> "Money":
        return Money(self.amount * float(factor), self.currency)

    __rmul__ = __mul__

    def __neg__(self) -> "Money":
        return Money(-self.amount, self.currency)

    def __float__(self) -> float:
        return self.amount

    @staticmethod
    def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
