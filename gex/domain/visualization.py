"""Визуализация GEX-уровней: ASCII-схема и HTML-версия для Telegram.

Две функции-рендерера:
  * :func:`render_text_visualization` — широкая ASCII-схема для текстового вывода;
  * :func:`render_telegram_html` — компактная HTML-версия под Telegram
    Bot API (``parse_mode=HTML``).

Оба рендерера строятся из одного набора уровней (:func:`_build_rows`),
поэтому их вывод всегда согласован. Архитектура позаимствована из
эталонной реализации: каждый уровень — :class:`LevelRow`, а тип линии и
суффикс задаются словарями (расширяемо без if/elif).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from gex.domain.metrics import GEXProfile


# ====================================================================== #
#  Подписи режима (человекочитаемые, по-русски)
# ====================================================================== #
#: Полная подпись режима с расшифровкой режима волатильности.
REGIME_LABELS_RU: dict[str, str] = {
    "POSITIVE": "Положительная гамма (Низкая волатильность)",
    "NEGATIVE": "Отрицательная гамма (Высокая волатильность)",
}

#: Краткая подпись режима (для заголовков/Telegram).
REGIME_SHORT_RU: dict[str, str] = {
    "POSITIVE": "Положительная гамма",
    "NEGATIVE": "Отрицательная гамма",
}

#: 💡-интерпретация режима для нижней части сообщения.
REGIME_INSIGHT_RU: dict[str, str] = {
    "POSITIVE": (
        "Дилеры гасят движения цены — рынок склонен к диапазону между стенами."
    ),
    "NEGATIVE": (
        "Дилеры усиливают движения цены — риск резких выносов за уровни."
    ),
}


def regime_label(regime: str) -> str:
    """Полная подпись режима (POSITIVE/NEGATIVE → по-русски)."""
    return REGIME_LABELS_RU.get(regime, regime)


def regime_short(regime: str) -> str:
    """Краткая подпись режима."""
    return REGIME_SHORT_RU.get(regime, regime)


def regime_insight(regime: str) -> str:
    """Текст-интерпретация режима для 💡-строки."""
    return REGIME_INSIGHT_RU.get(regime, "")


# ====================================================================== #
#  Модель уровня
# ====================================================================== #
Kind = Literal[
    "call_secondary", "call_primary", "flip", "spot", "put_primary", "put_secondary"
]


@dataclass
class LevelRow:
    """Один уровень на схеме.

    Attributes
    ----------
    price : float
        Цена уровня.
    label : str
        Подпись ('CALL WALL', 'GAMMA FLIP', 'SPOT', ...).
    kind : Kind
        Тип уровня — задаёт символ линии и суффикс через словари.
    """

    price: float
    label: str
    kind: Kind


def _build_rows(
    spot: float,
    gamma_flip: Optional[float],
    call_walls: list[float],
    put_walls: list[float],
) -> list[LevelRow]:
    """Собрать уровни для отрисовки, отсортированные по убыванию цены.

    Call walls подаются от primary к secondary (index 0 = primary),
    аналогично put walls. Неконечные значения (NaN) и ``None`` отбрасываются.
    """
    rows: list[LevelRow] = []

    for i, cw in enumerate(call_walls):
        if cw is None or not np.isfinite(cw):
            continue
        kind: Kind = "call_primary" if i == 0 else "call_secondary"
        label = "CALL WALL" if i == 0 else "Call Wall"
        rows.append(LevelRow(float(cw), label, kind))

    if gamma_flip is not None and np.isfinite(gamma_flip):
        rows.append(LevelRow(float(gamma_flip), "GAMMA FLIP", "flip"))

    if spot is not None and np.isfinite(spot):
        rows.append(LevelRow(float(spot), "SPOT", "spot"))

    for i, pw in enumerate(put_walls):
        if pw is None or not np.isfinite(pw):
            continue
        kind = "put_primary" if i == 0 else "put_secondary"
        label = "PUT WALL" if i == 0 else "Put Wall"
        rows.append(LevelRow(float(pw), label, kind))

    # Сверху — дороже.
    rows.sort(key=lambda r: r.price, reverse=True)
    return rows


# ====================================================================== #
#  Оформление: символ линии и суффикс (расширяемо через словари)
# ====================================================================== #
#: Символ линии для каждого типа уровня.
_LINE_CHARS: dict[Kind, str] = {
    "call_secondary": "━",
    "call_primary": "━",
    "flip": "╌",
    "spot": "─",
    "put_primary": "━",
    "put_secondary": "━",
}

#: Текстовый суффикс уровня в широкой ASCII-схеме.
_SUFFIX: dict[Kind, str] = {
    "call_primary": " ▲ Сопротивление",
    "put_primary": " ▼ Поддержка",
    "spot": " ● Текущая цена",
    "flip": "",
    "call_secondary": "",
    "put_secondary": "",
}

#: Маркер уровня в компактной Telegram-версии.
_TG_MARKER: dict[Kind, str] = {
    "call_primary": " ▲",
    "put_primary": " ▼",
    "spot": " ●",
}


def _fmt_price(price: float) -> str:
    """Цена без незначащих нулей: 748.0 → '748', 740.52 → '740.52'."""
    if price is None or not np.isfinite(price):
        return "—"
    return f"{price:.2f}".rstrip("0").rstrip(".")


def levels_from_profile(
    spot: float,
    profile: GEXProfile,
    top_n: int = 3,
) -> tuple[list[float], list[float], Optional[float]]:
    """Извлечь call/put стены и gamma_flip из :class:`GEXProfile`.

    Стены собираются как primary + secondary до ``top_n`` (через
    :meth:`GEXMetrics.ranked_walls`). Primary гарантированно первый.
    """
    from gex.domain.metrics import GEXMetrics

    call_walls: list[float] = []
    put_walls: list[float] = []

    # Primary из профиля (с защитой от NaN), затем дополняем secondary.
    primary_call = float(profile.call_wall) if np.isfinite(profile.call_wall) else None
    primary_put = float(profile.put_wall) if np.isfinite(profile.put_wall) else None

    if primary_call is not None:
        call_walls.append(primary_call)
    if primary_put is not None:
        put_walls.append(primary_put)

    # Дополняем secondary стенами (исключая уже добавленный primary).
    for w in profile.secondary_call_walls:
        if w not in call_walls and np.isfinite(w):
            call_walls.append(float(w))
    for w in profile.secondary_put_walls:
        if w not in put_walls and np.isfinite(w):
            put_walls.append(float(w))

    # Если primary не нашли напрямую — берём ранжирование целиком.
    if primary_call is None or primary_put is None:
        rc, rp = GEXMetrics.ranked_walls(profile.per_strike, top_n=top_n)
        if primary_call is None and rc:
            call_walls = rc
        if primary_put is None and rp:
            put_walls = rp

    # Обрезаем до top_n.
    return call_walls[:top_n], put_walls[:top_n], profile.gamma_flip


# ====================================================================== #
#  Рендереры
# ====================================================================== #
def render_text_visualization(
    spot: float,
    gamma_flip: Optional[float],
    call_walls: list[float],
    put_walls: list[float],
    width: int = 45,
) -> str:
    """Широкая ASCII-схема уровней (поле ``text_visualization`` ответа).

    Стены — сплошная ``━``, Gamma Flip — пунктир ``╌``, спот — тонкая ``─``.
    """
    rows = _build_rows(spot, gamma_flip, call_walls, put_walls)
    lines = []
    for r in rows:
        bar = _LINE_CHARS[r.kind] * width
        lines.append(f"{bar} {_fmt_price(r.price)} ({r.label}){_SUFFIX[r.kind]}")
    return "\n\n".join(lines)


def render_telegram_html(
    symbol: str,
    spot: float,
    regime: str,
    gamma_flip: Optional[float],
    call_walls: list[float],
    put_walls: list[float],
    width: int = 3,
) -> str:
    """Компактная HTML-версия уровней для Telegram (``parse_mode='HTML'``).

    Это тело сообщения (без дублирующей шапки): шапку добавляет вызывающий
    форматтер (тикер/спот/режим). Эмодзи не используются.
    """
    rows = _build_rows(spot, gamma_flip, call_walls, put_walls)
    lines = []
    for r in rows:
        bar = _LINE_CHARS[r.kind] * width
        marker = _TG_MARKER.get(r.kind, "")
        lines.append(f"{bar} {_fmt_price(r.price)} ({r.label}){marker}")
    ascii_block = "\n".join(lines)

    out = [f"<b>{symbol}: GEX-уровни</b>", f"<pre>{ascii_block}</pre>"]
    insight = regime_insight(regime)
    if insight:
        out.append(insight)
    return "\n".join(out)
