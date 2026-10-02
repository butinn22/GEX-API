"""Структура рынка HH/HL/LH/LL — канон (ring: domain, чистая арифметика).

Что здесь дублировалось
-----------------------
Подсчёт свингов был скопирован один-в-один между ``ta.detect_trend`` (``ta.py:760-777``) и
``trendlines.classify_fractals`` (``trendlines.py:529-550``), но **разрешение направления
различалось**, из-за чего один и тот же набор свингов давал разные ответы на разных страницах:

| Правило | Требование для BULLISH | Пример расхождения |
|---|---|---|
| ``ta`` | ``bull_score > bear_score`` **и** ``HH ≥ 1`` **и** ``HL ≥ 1`` | ``HH=1, HL=1, LH=3, LL=0`` → **RANGE** |
| ``trendlines`` | ``HH ≥ 1`` и ``HL ≥ 1`` (без сравнения сумм, если ``is_bear`` ложно) | тот же вход → **BULLISH** |

Канон сохраняет **оба** правила явным параметром ``rule`` (``"ta"`` / ``"trendlines"``), поэтому
миграция не меняет поведение страниц, а расхождение становится видимым и проверяемым тестом.

Отдельно вынесен momentum-veto из ``ta`` (``ta.py:791-801``) — он применяется поверх направления
свингов и не зависит от них, поэтому живёт отдельной чистой функцией.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "SwingCounts",
    "count_swings",
    "ta_swing_direction",
    "trendlines_direction",
    "trendlines_strength",
    "apply_momentum_veto",
    "RULES",
]

RULES = ("ta", "trendlines")


@dataclass(frozen=True)
class SwingCounts:
    """Счётчики структуры: HH/LH по свинг-хаям, HL/LL по свинг-лоям."""

    higher_highs: int = 0
    lower_highs: int = 0
    higher_lows: int = 0
    lower_lows: int = 0

    @property
    def bull_score(self) -> int:
        return self.higher_highs + self.higher_lows

    @property
    def bear_score(self) -> int:
        return self.lower_highs + self.lower_lows

    @property
    def total(self) -> int:
        return self.bull_score + self.bear_score


def count_swings(
    swing_high_prices: Sequence[float],
    swing_low_prices: Sequence[float],
) -> SwingCounts:
    """Посчитать HH/LH по последовательным свинг-хаям и HL/LL по свинг-лоям.

    Идентично обеим прежним реализациям: сравнение идёт с **предыдущим** свингом того же типа,
    равные значения не считаются ни ростом, ни падением.
    """
    higher_highs = lower_highs = 0
    for j in range(1, len(swing_high_prices)):
        prev, cur = swing_high_prices[j - 1], swing_high_prices[j]
        if cur > prev:
            higher_highs += 1
        elif cur < prev:
            lower_highs += 1

    higher_lows = lower_lows = 0
    for j in range(1, len(swing_low_prices)):
        prev, cur = swing_low_prices[j - 1], swing_low_prices[j]
        if cur > prev:
            higher_lows += 1
        elif cur < prev:
            lower_lows += 1

    return SwingCounts(
        higher_highs=higher_highs,
        lower_highs=lower_highs,
        higher_lows=higher_lows,
        lower_lows=lower_lows,
    )


def ta_swing_direction(counts: SwingCounts) -> str:
    """Направление **только по свингам** в правиле ``ta`` (``ta.py:782-787``).

    ``BULLISH`` требует перевеса суммы свингов И хотя бы одного HH и одного HL;
    ``BEARISH`` — зеркально. Иначе ``RANGE``.
    """
    if counts.total > 0 and counts.bull_score > counts.bear_score \
            and counts.higher_highs >= 1 and counts.higher_lows >= 1:
        return "BULLISH"
    if counts.total > 0 and counts.bear_score > counts.bull_score \
            and counts.lower_highs >= 1 and counts.lower_lows >= 1:
        return "BEARISH"
    return "RANGE"


def trendlines_direction(counts: SwingCounts) -> str:
    """Направление в правиле ``trendlines`` (``trendlines.py:553-583``).

    Приоритет у «чистой» структуры (``is_bull`` без ``is_bear``); при смешанной — большинство сумм;
    при равенстве — ``RANGE``.
    """
    total = counts.total
    if total == 0:
        return "RANGE"

    is_bull = counts.higher_highs >= 1 and counts.higher_lows >= 1
    is_bear = counts.lower_highs >= 1 and counts.lower_lows >= 1

    if is_bull and not is_bear:
        return "BULLISH"
    if is_bear and not is_bull:
        return "BEARISH"
    bull, bear = counts.bull_score, counts.bear_score
    if bull > bear:
        return "BULLISH"
    if bear > bull:
        return "BEARISH"
    return "RANGE"


def trendlines_strength(counts: SwingCounts) -> float:
    """Сила 0..100 как доля перевеса, с округлением до 1 знака (как ``trendlines``)."""
    total = counts.total
    if total == 0:
        return 0.0
    return round(max(counts.bull_score, counts.bear_score) / total * 100.0, 1)


def apply_momentum_veto(swing_direction: str, momentum_direction: str | None) -> str:
    """Momentum-veto из ``ta.detect_trend`` (``ta.py:791-801``).

    * согласие или нейтральный momentum → направление свингов;
    * свинги ``RANGE``, но momentum направленный → берём momentum;
    * прямое расхождение → ``RANGE`` (тренд под вопросом).
    """
    mom = momentum_direction or "NEUTRAL"
    if swing_direction != "RANGE" and (mom == swing_direction or mom == "NEUTRAL"):
        return swing_direction
    if mom in ("BULLISH", "BEARISH") and swing_direction == "RANGE":
        return mom
    if swing_direction != "RANGE" and mom in ("BULLISH", "BEARISH") and mom != swing_direction:
        return "RANGE"
    return swing_direction
