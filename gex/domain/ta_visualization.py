"""Сводка и визуализация технического анализа: JSON-блок + Telegram HTML.

Параллельно :mod:`gex.visualization` (которая делает то же для GEX), здесь:

  * :func:`build_ta_summary` — собирает «логичный объёмный summarize» в виде
    структуры, сериализуемой в ``TASummaryOut`` (тренд, сила, поддержка/
    сопротивление, разворот, осцилляторы, дивергенции, вердикт);
  * :func:`render_ta_telegram_html` — рендерит то же содержание красивым
    читаемым HTML-сообщением для Telegram (``parse_mode=HTML``).

Оба построены из списка :class:`gex.ta.TimeframeAnalysis` (после multi-TF
подтверждения) и общего консенсуса.
"""
from __future__ import annotations

from typing import Any, Optional

from gex.domain.ta import TimeframeAnalysis, TrendInfo, ReversalProb, MomentumStrength


# ====================================================================== #
#  Человекочитаемые подписи (по-русски)
# ====================================================================== #
TREND_EMOJI: dict[str, str] = {
    "BULLISH": "🟢",
    "BEARISH": "🔴",
    "RANGE": "🟡",
    "NEUTRAL": "⚪",
}

TREND_LABEL_RU: dict[str, str] = {
    "BULLISH": "Восходящий",
    "BEARISH": "Нисходящий",
    "RANGE": "Боковик",
    "NEUTRAL": "Нейтральный",
}

DIVERGENCE_LABEL_RU: dict[str, str] = {
    "BULLISH": "Бычья 🔼",
    "BEARISH": "Медвежья 🔻",
}


# ====================================================================== #
#  Вспомогательное форматирование
# ====================================================================== #
def _strength_bar(strength: float, width: int = 10) -> str:
    """Текстовый бар силы 0..100: '█████░░░░░'."""
    filled = int(round(strength / 100.0 * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _strength_word(strength: float) -> str:
    """Словесная оценка силы тренда."""
    if strength >= 75:
        return "Очень сильный"
    if strength >= 55:
        return "Сильный"
    if strength >= 35:
        return "Умеренный"
    if strength >= 15:
        return "Слабый"
    return "Очень слабый"


def _rsi_zone(rsi: float) -> str:
    """Зона RSI с эмодзи."""
    if rsi >= 70:
        return "Перекупленность 🔥"
    if rsi >= 55:
        return "Бычья зона"
    if rsi >= 45:
        return "Нейтральная"
    if rsi >= 30:
        return "Медвежья зона"
    return "Перепроданность ❄️"


def _rsi_zone_text(rsi: float) -> str:
    """Зона RSI текстом (для Telegram-сообщений, без эмодзи)."""
    if rsi >= 70:
        return "перекупленность"
    if rsi >= 55:
        return "бычья зона"
    if rsi >= 45:
        return "нейтральная"
    if rsi >= 30:
        return "медвежья зона"
    return "перепроданность"


def _reversal_word(p: float) -> str:
    """Словесная оценка вероятности разворота."""
    if p >= 0.65:
        return "Высокая"
    if p >= 0.45:
        return "Умеренная"
    if p >= 0.25:
        return "Низкая"
    return "Очень низкая"


def _pct_str(x: float, decimals: int = 2) -> str:
    """Число со знаком и '%', NaN-safe."""
    import math
    if x is None or math.isnan(x):
        return "—"
    return f"{x:+.{decimals}f}%"


# ====================================================================== #
#  Сборка summarize-структуры
# ====================================================================== #
def build_ta_summary(
    symbol: str,
    spot: float,
    analyses: list[TimeframeAnalysis],
    consensus_trend: str,
    consensus_p_reversal: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    """Собрать «объёмный» summarize-блок тех. анализа.

    Возвращает ``dict``, напрямую маппящийся в :class:`TASummaryOut`.

    Структура (всё на русском, для человека):

      * ``headline``    — однострочный вердикт (эмодзи + тренд + сила);
      * ``trend``       — направление, словесная сила, средняя сила по TF;
      * ``support_resistance`` — ближайшие поддержка/сопротивление (из свингов
        и EMA) по каждому TF + общая;
      * ``reversal``    — вероятность разворота + словесная оценка + метод;
      * ``oscillators`` — RSI/MACD по каждому TF + зоны + сигналы пересечений;
      * ``divergences`` — наличие и список дивергенций по TF + чистый сигнал;
      * ``multi_tf``    — подтверждение младшими TF (по старшему TF);
      * ``verdict``     — итоговый текстовый вердикт с рекомендацией.
    """
    # --- Средняя сила тренда по TF (взвешенно) ---
    w_sum = sum(weights.get(a.timeframe, 0.0) for a in analyses) or 1.0
    avg_strength = sum(
        a.trend.strength * weights.get(a.timeframe, 0.0) for a in analyses
    ) / w_sum

    # --- Поддержка/сопротивление ---
    support_resistance = _build_support_resistance(analyses, spot)

    # --- Разворот (по консенсусу) ---
    reversal_block = {
        "consensus_p_reversal": round(consensus_p_reversal, 3),
        "assessment": _reversal_word(consensus_p_reversal),
        "by_timeframe": {
            a.timeframe: {
                "p_reversal": round(a.reversal.p_reversal, 3),
                "p_markov": round(a.reversal.p_markov, 3),
                "p_mc": round(a.reversal.p_mc, 3),
                "method": a.reversal.method,
            }
            for a in analyses
        },
    }

    # --- Осцилляторы по TF ---
    oscillators = _build_oscillators(analyses)

    # --- Дивергенции по TF ---
    divergences_block = _build_divergences(analyses)

    # --- Multi-TF подтверждение (по старшему TF, обычно 1d) ---
    multi_tf = _build_multi_tf(analyses)

    # --- Headline + verdict ---
    headline = (
        f"{TREND_EMOJI.get(consensus_trend, '⚪')} {symbol}: "
        f"{TREND_LABEL_RU.get(consensus_trend, consensus_trend)} тренд, "
        f"сила {_strength_word(avg_strength).lower()} ({avg_strength:.0f}/100)"
    )
    verdict = _build_verdict(
        consensus_trend, avg_strength, consensus_p_reversal,
        divergences_block["net_signal"], oscillators,
    )

    return {
        "headline": headline,
        "trend": {
            "consensus_direction": consensus_trend,
            "consensus_direction_label": TREND_LABEL_RU.get(consensus_trend, consensus_trend),
            "avg_strength": round(avg_strength, 1),
            "strength_word": _strength_word(avg_strength),
            "by_timeframe": {
                a.timeframe: {
                    "direction": a.trend.direction,
                    "strength": round(a.trend.strength, 1),
                }
                for a in analyses
            },
        },
        "support_resistance": support_resistance,
        "reversal": reversal_block,
        "oscillators": oscillators,
        "divergences": divergences_block,
        "multi_tf": multi_tf,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------- #
#  Поддержки/сопротивления: из свингов и EMA
# ---------------------------------------------------------------------- #
def _build_support_resistance(
    analyses: list[TimeframeAnalysis], spot: float
) -> dict[str, Any]:
    """Ближайшие поддержка/сопротивление.

    Для каждого TF:
      * сопротивление — ближайший свинг-хай выше спота (иначе EMA200 если выше);
      * поддержка — ближайший свинг-лоу ниже спота (иначе EMA200 если ниже).
    Общий уровень — медиана по TF.
    """
    per_tf: dict[str, dict[str, float]] = {}
    resistances: list[float] = []
    supports: list[float] = []

    for a in analyses:
        swing_highs = [s for s in a.trend.swing_highs if s > spot]
        swing_lows = [s for s in a.trend.swing_lows if s < spot]
        resistance = min(swing_highs) if swing_highs else (
            a.indicators.ema200 if a.indicators.ema200 > spot else a.trend.recent_high
        )
        support = max(swing_lows) if swing_lows else (
            a.indicators.ema200 if a.indicators.ema200 < spot else a.trend.recent_low
        )
        per_tf[a.timeframe] = {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
        }
        if resistance > spot:
            resistances.append(resistance)
        if support < spot:
            supports.append(support)

    import statistics
    overall = {
        "support": round(statistics.median(supports), 2) if supports else round(spot * 0.97, 2),
        "resistance": round(statistics.median(resistances), 2) if resistances else round(spot * 1.03, 2),
    }
    return {"overall": overall, "by_timeframe": per_tf}


# ---------------------------------------------------------------------- #
#  Осцилляторы
# ---------------------------------------------------------------------- #
def _build_oscillators(analyses: list[TimeframeAnalysis]) -> dict[str, Any]:
    """Состояние RSI и MACD по каждому TF + общие сигналы."""
    per_tf: dict[str, dict[str, Any]] = {}
    for a in analyses:
        i = a.indicators
        per_tf[a.timeframe] = {
            "rsi": round(i.rsi, 1),
            "rsi_zone": _rsi_zone(i.rsi),
            "macd_hist": round(i.macd_hist, 4),
            "macd_bull_cross": i.macd_bull_cross,
            "macd_bear_cross": i.macd_bear_cross,
        }
    return {"by_timeframe": per_tf}


# ---------------------------------------------------------------------- #
#  Дивергенции
# ---------------------------------------------------------------------- #
def _build_divergences(analyses: list[TimeframeAnalysis]) -> dict[str, Any]:
    """Сводка дивергенций осцилляторов по TF + чистый сигнал."""
    import math
    per_tf: dict[str, Any] = {}
    bull_count = 0
    bear_count = 0
    for a in analyses:
        d = a.divergence
        if d is None or not d.divergences:
            per_tf[a.timeframe] = {"has_divergence": False, "items": []}
            continue
        items = [
            {
                "oscillator": div.oscillator,
                "type": div.type,
                "type_label": DIVERGENCE_LABEL_RU.get(div.type, div.type),
                "strength": round(div.strength, 1),
                "bars_ago": div.bars_ago,
            }
            for div in d.divergences
        ]
        per_tf[a.timeframe] = {"has_divergence": True, "items": items}
        if d.has_bullish:
            bull_count += 1
        if d.has_bearish:
            bear_count += 1

    net = "NEUTRAL"
    if bull_count > bear_count:
        net = "BULLISH"
    elif bear_count > bull_count:
        net = "BEARISH"
    return {
        "by_timeframe": per_tf,
        "has_any_divergence": bull_count + bear_count > 0,
        "net_signal": net,
        "net_signal_label": DIVERGENCE_LABEL_RU.get(net, TREND_LABEL_RU.get(net, net)),
    }


# ---------------------------------------------------------------------- #
#  Multi-TF подтверждение
# ---------------------------------------------------------------------- #
def _build_multi_tf(analyses: list[TimeframeAnalysis]) -> dict[str, Any]:
    """Подтверждение по старшему TF (1d, иначе последний)."""
    # Старший TF — последний в порядке
    senior = None
    for a in analyses:
        senior = a
    if senior is None or senior.confirmation is None:
        return {"available": False}
    c = senior.confirmation
    return {
        "available": True,
        "base_timeframe": c.base_timeframe,
        "base_direction": c.base_direction,
        "agreeing": list(c.agreeing),
        "disagreeing": list(c.disagreeing),
        "neutral": list(c.neutral),
        "confirmation_ratio": round(c.confirmation_ratio, 3),
        "assessment": (
            "Полное подтверждение" if c.confirmation_ratio >= 0.75
            else "Частичное подтверждение" if c.confirmation_ratio >= 0.5
            else "Расхождение" if c.confirmation_ratio < 0.35
            else "Смешанное"
        ),
    }


# ---------------------------------------------------------------------- #
#  Итоговый вердикт
# ---------------------------------------------------------------------- #
def _verdict_divergence_parts(div_signal: str) -> list[str]:
    """Фрагменты вердикта о дивергенциях (без эмодзи)."""
    if div_signal == "BEARISH":
        return ["медвежьи дивергенции осцилляторов ослабляют импульс"]
    if div_signal == "BULLISH":
        return ["бычьи дивергенции поддерживают возможный отскок"]
    return []


def _build_verdict(
    trend: str,
    strength: float,
    p_reversal: float,
    div_signal: str,
    oscillators: dict[str, Any],
) -> str:
    """Текстовый вердикт с рекомендацией (без обещания гарантий)."""
    parts: list[str] = []

    # Базовая направленность
    if trend == "BULLISH":
        parts.append("преобладает восходящий тренд")
    elif trend == "BEARISH":
        parts.append("преобладает нисходящий тренд")
    else:
        parts.append("рынок в боковике, чёткого тренда нет")

    # Сила
    parts.append(f"силой {strength:.0f}/100 ({_strength_word(strength).lower()})")

    # Риск разворота
    if p_reversal >= 0.55:
        parts.append(f"высокий риск разворота ({p_reversal:.0%})")
    elif p_reversal <= 0.25:
        parts.append(f"низкий риск разворота ({p_reversal:.0%})")
    else:
        parts.append(f"умеренный риск разворота ({p_reversal:.0%})")

    # Дивергенции как предупреждение
    parts.extend(_verdict_divergence_parts(div_signal))

    verdict = "На данный момент " + ", ".join(parts) + "."
    verdict += " Оценка вероятностная и не является торговой рекомендацией."
    return verdict


# ====================================================================== #
#  Рендер Telegram HTML
# ====================================================================== #
def render_ta_telegram_html(
    symbol: str,
    spot: float,
    analyses: list[TimeframeAnalysis],
    consensus_trend: str,
    consensus_p_reversal: float,
    weights: dict[str, float],
) -> str:
    """Красивое читаемое HTML-сообщение для Telegram (``parse_mode=HTML``).

    Содержит разделы:
      1. Заголовок (тикер, спот, консенсус-тренд, сила);
      2. Тренды по таймфреймам (с барами силы);
      3. Поддержка / сопротивление;
      4. Потенциал разворота (вероятность по TF);
      5. Осцилляторы (RSI с зонами, MACD);
      6. Дивергенции;
      7. Multi-timeframe подтверждение;
      8. Итоговый вердикт.
    """
    summary = build_ta_summary(
        symbol, spot, analyses, consensus_trend, consensus_p_reversal, weights
    )
    w_sum = sum(weights.get(a.timeframe, 0.0) for a in analyses) or 1.0
    avg_strength = sum(
        a.trend.strength * weights.get(a.timeframe, 0.0) for a in analyses
    ) / w_sum

    arrow = "▲" if consensus_trend == "BULLISH" else ("▼" if consensus_trend == "BEARISH" else "◆")
    lines: list[str] = []

    def _dir_arrow(d: str) -> str:
        return "▲" if d == "BULLISH" else ("▼" if d == "BEARISH" else "◆")

    def _dir_word(d: str) -> str:
        return TREND_LABEL_RU.get(d, d)

    # --- 1. Заголовок ---
    lines.append(f"<b>{symbol} · Технический анализ</b>")
    lines.append(f"Спот: {spot:.2f}")
    lines.append(
        f"Тренд: {arrow} {_dir_word(consensus_trend)} · "
        f"Сила: {_strength_word(avg_strength)} "
        f"<code>{_strength_bar(avg_strength)} {avg_strength:.0f}/100</code>"
    )
    lines.append("")

    # --- 2. Тренды по таймфреймам ---
    lines.append("<b>Тренды по таймфреймам</b>")
    for a in analyses:
        t = a.trend
        lines.append(
            f"  <code>{a.timeframe:>2}</code> {_dir_arrow(t.direction)} "
            f"{_dir_word(t.direction):<10} "
            f"<code>{_strength_bar(t.strength, 8)}</code> {t.strength:.0f}"
        )
    lines.append("")

    # --- 3. Поддержка / сопротивление ---
    sr = summary["support_resistance"]
    lines.append("<b>Поддержка / Сопротивление</b>")
    lines.append(
        f"  Поддержка: {sr['overall']['support']:.2f} · "
        f"Сопротивление: {sr['overall']['resistance']:.2f}"
    )
    for tf, lvl in sr["by_timeframe"].items():
        lines.append(
            f"  <code>{tf:>2}</code> поддержка {lvl['support']:.2f} · "
            f"сопротивление {lvl['resistance']:.2f}"
        )
    lines.append("")

    # --- 4. Разворот ---
    lines.append("<b>Потенциал разворота</b>")
    lines.append(
        f"  Вероятность: {consensus_p_reversal:.0%} "
        f"({_reversal_word(consensus_p_reversal).lower()})"
    )
    for a in analyses:
        r = a.reversal
        lines.append(
            f"  <code>{a.timeframe:>2}</code> P={r.p_reversal:.0%} "
            f"(Markov {r.p_markov:.0%} / MC {r.p_mc:.0%})"
        )
    lines.append("")

    # --- 5. Осцилляторы ---
    lines.append("<b>Осцилляторы</b>")
    for a in analyses:
        i = a.indicators
        cross = ""
        if i.macd_bull_cross:
            cross = " · бычий кросс"
        elif i.macd_bear_cross:
            cross = " · медвежий кросс"
        lines.append(
            f"  <code>{a.timeframe:>2}</code> RSI {i.rsi:.0f} "
            f"({_rsi_zone_text(i.rsi)}) · MACD h={i.macd_hist:+.3f}{cross}"
        )
    lines.append("")

    # --- 6. Дивергенции ---
    div = summary["divergences"]
    lines.append("<b>Дивергенции осцилляторов</b>")
    if not div["has_any_divergence"]:
        lines.append("  Явных дивергенций не обнаружено.")
    else:
        net = div["net_signal"]
        net_word = {"BULLISH": "бычья", "BEARISH": "медвежья"}.get(net, "нейтральный")
        lines.append(f"  Чистый сигнал: {net_word}")
        for tf, info in div["by_timeframe"].items():
            if not info["has_divergence"]:
                continue
            for item in info["items"]:
                type_word = {"BULLISH": "бычья", "BEARISH": "медвежья"}.get(item["type"], item["type"])
                lines.append(
                    f"  <code>{tf:>2}</code> {item['oscillator']} {type_word} "
                    f"(сила {item['strength']:.0f}, {item['bars_ago']} бар. назад)"
                )
    lines.append("")

    # --- 7. Multi-TF подтверждение ---
    mtf = summary["multi_tf"]
    lines.append("<b>Multi-timeframe подтверждение</b>")
    if mtf.get("available"):
        ratio = mtf["confirmation_ratio"]
        lines.append(
            f"  <b>{mtf['base_timeframe']}</b>: {_dir_arrow(mtf['base_direction'])} "
            f"{_dir_word(mtf['base_direction'])} · {mtf['assessment']} "
            f"(доля {ratio:.0%})"
        )
        if mtf["agreeing"]:
            lines.append(f"  Подтверждают: {', '.join(mtf['agreeing'])}")
        if mtf["disagreeing"]:
            lines.append(f"  Противоречат: {', '.join(mtf['disagreeing'])}")
    else:
        lines.append("  Недостаточно таймфреймов для подтверждения.")
    lines.append("")

    # --- 8. Вердикт ---
    lines.append("<b>Итог</b>")
    lines.append(f"<i>{summary['verdict']}</i>")

    return "\n".join(lines)
