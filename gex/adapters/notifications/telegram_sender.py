"""
Telegram message sender module.

Sends messages to a Telegram chat via the Bot API using the `requests` library
(synchronous, no async). Credentials берутся из settings (env):
``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``; пусто = уведомления отключены.

Handles:
  - 4096-character message limit (splits oversized content into batches)
  - Small configurable delay between batches
  - Basic HTML or MarkdownV2 parse mode
  - Structured error handling and logging
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, TYPE_CHECKING

import requests

from gex.auth.config import settings

if TYPE_CHECKING:
    # Аннотации типов схем — только для стат. анализа. При выполнении sender
    # не зависит от schemas (форматеры работают через duck-typing по атрибутам),
    # поэтому модуль остаётся изолированно импортируемым без циклических связей.
    from gex.schemas import (
        GEXAnalysisOut,
        GEXProfileOut,
        TAAnalysisOut,
        TimeframeOut,
        ScanReportOut,
        ScanRecordOut,
        SignalAnalysisOut,
        TrendlineAnalysisOut,
        MacdTrendAnalysisOut,
    )
    from gex.schemas.extended_schemas import ExtendedGEXAnalysisOut

logger = logging.getLogger(__name__)

_TELEGRAM_BASE_URL: str = "https://api.telegram.org"

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
_TELEGRAM_BATCH_DELAY: float = 0.2
_VALID_PARSE_MODES: set[str] = {"HTML", "MarkdownV2"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_telegram_message(
    text: str,
    *,
    parse_mode: str | None = None,
    chat_id: str | None = None,
    polish: bool = True,
) -> dict[str, Any]:
    """
    Send *text* to a Telegram chat.

    Parameters
    ----------
    text : str
        Message content (plain text, MarkdownV2, or HTML).
    parse_mode : str, optional
        ``"MarkdownV2"``, ``"HTML"``, or ``None`` for plain text.
    chat_id : str, optional
        Target chat ID. If None, uses the default hardcoded chat.
    polish : bool, optional
        True (default) — убрать декоративные эмодзи/лишние пробелы, сжать
        пустые строки; если чат привязан к пользователю — добавить в шапку
        его Telegram-ник, а в подпись — «Отчёт сформирован в GEXANALYTICS».
        False — отправить текст как есть (внутренние алерты и пр.).

    Returns
    -------
    dict with keys ``success``, ``batches_sent``, ``errors``, ``raw_responses``.
    """
    # Central orchestrator path: applies Redis rate limits and prevents Telegram spam.
    try:
        from gex.orchestrator.sync_gateway import is_orchestrator_ready, sync_send_telegram_message
        if is_orchestrator_ready():
            result = sync_send_telegram_message(
                text,
                chat_id=chat_id,
                parse_mode=parse_mode,
                polish=polish,
            )
            if result is not None:
                return result
            return {
                "success": False,
                "batches_sent": 0,
                "errors": ["Orchestrator telegram send failed; message not delivered to avoid duplicate sends"],
                "raw_responses": [],
            }
    except Exception:  # noqa: BLE001
        pass

    resolved_parse_mode = _normalize_parse_mode(parse_mode)
    target_chat = chat_id or settings.TELEGRAM_CHAT_ID

    bot_token = settings.TELEGRAM_BOT_TOKEN
    try:
        from gex.auth.runtime_config import get_telegram_config
        rt = get_telegram_config()
        if rt.get("bot_token"):
            bot_token = rt["bot_token"]
        if rt.get("chat_id") and not chat_id:
            target_chat = rt["chat_id"]
    except Exception:  # noqa: BLE001
        pass

    if not bot_token:
        logger.warning("Telegram-уведомления отключены: TELEGRAM_BOT_TOKEN не задан в .env")
        return {"success": False, "batches_sent": 0, "errors": ["TELEGRAM_BOT_TOKEN not configured"], "raw_responses": []}

    if not text or not text.strip():
        return {"success": False, "batches_sent": 0, "errors": ["Message text is empty"], "raw_responses": []}

    payload = text.strip()
    if polish:
        payload = _polish_message(text)
        nick = _resolve_nickname(target_chat)
        if resolved_parse_mode == "HTML":
            footer = _BRAND_FOOTER_HTML
        else:
            footer = _BRAND_FOOTER_PLAIN
        if nick:
            nick = _esc(nick)
            if resolved_parse_mode == "HTML":
                payload = f"<b>@{nick}</b>\n\n" + payload + "\n\n" + footer
            else:
                payload = f"@{nick}\n\n" + payload + "\n\n" + footer
        else:
            payload = payload + "\n\n" + footer
        if not payload:
            payload = text.strip()

    chunks = _split_message(payload)
    logger.info("Sending %d chunk(s) to chat %s (parse_mode=%s)", len(chunks), target_chat, resolved_parse_mode)

    raw_responses: list[dict[str, Any]] = []
    send_errors: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        try:
            resp = _post_telegram_message(token=bot_token, chat_id=target_chat, text=chunk, parse_mode=resolved_parse_mode)
            raw_responses.append(resp)
            ok = resp.get("ok", False)
            if not ok:
                err_msg = f"Chunk {i}/{len(chunks)} failed (error_code={resp.get('error_code','?')}): {resp.get('description','unknown')}"
                send_errors.append(err_msg)
                logger.error(err_msg)
            else:
                logger.info("Chunk %d/%d sent successfully", i, len(chunks))
        except Exception as exc:
            err_msg = f"Chunk {i}/{len(chunks)} raised an exception: {exc}"
            send_errors.append(err_msg)
            logger.exception(err_msg)
            raw_responses.append({"ok": False, "exception": str(exc)})
        if i < len(chunks):
            time.sleep(_TELEGRAM_BATCH_DELAY)

    logger.info("Telegram send result: success=%s, chunks=%d, errors=%s", len(send_errors) == 0, len(chunks), send_errors)
    return {"success": len(send_errors) == 0, "batches_sent": len(chunks), "errors": send_errors if send_errors else None, "raw_responses": raw_responses}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_parse_mode(parse_mode: str | None) -> str | None:
    if parse_mode is None:
        return None
    stripped = parse_mode.strip()
    if not stripped:
        return None
    if stripped not in _VALID_PARSE_MODES:
        logger.warning("Unknown parse_mode=%r -> falling back to plain text", stripped)
        return None
    return stripped


def _split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _post_telegram_message(token: str, chat_id: str, text: str, parse_mode: str | None = None) -> dict[str, Any]:
    url = f"{_TELEGRAM_BASE_URL}/bot{token}/sendMessage"
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if parse_mode is not None:
        payload["parse_mode"] = parse_mode
    logger.debug("POST telegram sendMessage (len=%d)", len(text))  # токен не логируем
    try:
        response = requests.post(url, json=payload, timeout=15)
        try:
            data = response.json()
        except Exception:
            data = {"ok": False, "error_code": response.status_code, "description": response.text or "Unknown Telegram API error"}
        if not response.ok:
            data["ok"] = False
            data["error_code"] = response.status_code
        return data
    except requests.RequestException as exc:
        return {"ok": False, "error_code": 0, "description": f"Network error: {exc}"}


# =========================================================================== #
#  «Полировка» сообщений: без эмодзи-мусора, единый структурный стиль.
# =========================================================================== #
# Убираются декоративные эмодзи (цветные кружки/смайлы/иконки-лейблы) —
# смысл всегда продублирован текстом рядом (подпись + значение). Остаются
# типографские символы: ▲/▼/◆/● (направления и уровни), █░ (бары силы),
# •/· (списки и разделители). Пробелы схлопываются ТОЛЬКО вне <pre>/<code>,
# чтобы не сломать ASCII-схемы уровней и моноширинные таблицы.
_EMOJI_STRIP_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # эмодзи и пиктограммы (вкл. флаги, цвета)
    "\U00002600-\U000027BF"   # прочие символы + дингбаты (✅⚠️🔥💡…)
    "\U00002B00-\U00002BFF"   # стрелки-эмодзи и звёзды (⬆️⭐…)
    "\U000023E9-\U000023FA"   # эмодзи-кнопки/часы
    "\U0000FE0F"              # variation selector (превращает символ в эмодзи)
    "\U0000200D"              # zero-width joiner
    "\U000020E3"              # combining keycap
    "]+",
    re.UNICODE,
)
_BRAND_FOOTER_HTML = "<i>Отчёт сформирован в GEXANALYTICS</i>"
_BRAND_FOOTER_PLAIN = "Отчёт сформирован в GEXANALYTICS"


def _polish_message(text: str) -> str:
    """Эмодзи-мусор долой, структура сохранена: <pre>/<code> не трогаем."""
    text = _EMOJI_STRIP_RE.sub("", text)
    # Разбиваем на «защищённые» фрагменты (<pre>/<code>) и остальной текст.
    parts = re.split(r"(<pre>.*?</pre>|<code>.*?</code>)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        if not chunk:
            continue
        lines: list[str] = []
        prev_blank = False
        for raw in chunk.split("\n"):
            ln = re.sub(r"[ \t]{2,}", " ", raw)
            if not ln.strip():
                if prev_blank:
                    continue
                prev_blank = True
            else:
                prev_blank = False
            lines.append(ln)
        parts[i] = "\n".join(lines)
    return "".join(parts).strip()


def _resolve_nickname(chat_id: str | None) -> str | None:
    """Telegram-ник пользователя по chat_id (для шапки отчёта). None — не нашли/ошибка."""
    if not chat_id:
        return None
    try:
        from gex.auth.models import User
        from gex.adapters.persistence.database import SessionLocal
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.telegram_chat_id == str(chat_id)).first()
        finally:
            db.close()
    except Exception:  # noqa: BLE001 — подпись не должна ронять отправку
        logger.debug("Telegram nickname lookup failed for chat %s", chat_id, exc_info=True)
        return None
    if user is None:
        return None
    nick = (getattr(user, "telegram_username", "") or "").strip().lstrip("@")
    return nick or None


# =========================================================================== #
#  GEX / TA форматеры: типизированная схема → HTML для Telegram
# =========================================================================== #
# Принцип: форматеры — чистые функции (без отправки). Берут готовые данные из
# Pydantic-схем (duck-typing по атрибутам) и возвращают HTML-строку под
# ``parse_mode='HTML'``. Полные анализы переиспользуют уже сгенерированные в
# visualization.py / ta_visualization.py поля ``telegram_html_message``; для
# лёгких ручек (без summarize) рендерят компактную карточку сами.
#
# Импорты схем вынесены в TYPE_CHECKING (см. выше) — выполнение не зависит от
# модуля schemas, циклических зависимостей нет.

_REGIME_SHORT_RU: dict[str, str] = {
    "POSITIVE": "Положительная гамма",
    "NEGATIVE": "Отрицательная гамма",
}
_TREND_LABEL_RU: dict[str, str] = {
    "BULLISH": "Восходящий",
    "BEARISH": "Нисходящий",
    "RANGE": "Боковик",
    "NEUTRAL": "Нейтральный",
}
_ARROW: dict[str, str] = {
    "BULLISH": "▲",
    "BEARISH": "▼",
    "RANGE": "◆",
    "NEUTRAL": "◇",
}


def _esc(text: Any) -> str:
    """Экранировать спецсимволы HTML (&, <, >) для parse_mode=HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_price(price: float) -> str:
    """Цена без незначащих нулей: 748.0 → '748', 740.52 → '740.52'."""
    if price is None:
        return "—"
    return f"{price:.2f}".rstrip("0").rstrip(".")


def _strength_bar(strength: float, width: int = 10) -> str:
    """Текстовый бар силы 0..100: '█████░░░░░'."""
    filled = max(0, min(width, int(round(strength / 100.0 * width))))
    return "█" * filled + "░" * (width - filled)


# --------------------------------------------------------------------------- #
#  GEX
# --------------------------------------------------------------------------- #
def format_gex_analysis_html(a: "GEXAnalysisOut", chat_id: str | None = None) -> str:
    """HTML-сообщение полного GEX-анализа.

    Использует готовый блок из ``a.summarize.telegram_html_message`` и добавляет
    шапку с направлением, уверенностью, вероятностями и режимом.
    """
    arrow = _ARROW.get(a.direction, "◆")
    trend_ru = _TREND_LABEL_RU.get(a.direction, a.direction)
    regime_ru = _REGIME_SHORT_RU.get(a.regime, a.regime)
    body = ""
    if getattr(a, "summarize", None) is not None:
        body = a.summarize.telegram_html_message or ""
    lines: list[str] = [
        f"<b>{_esc(a.symbol)} · GEX-анализ</b>",
        f"Спот: {_fmt_price(a.spot)} · Горизонт: {_esc(a.days)} дн.",
        f"Направление: {arrow} {_esc(trend_ru)} (уверенность {_esc(a.confidence)}%)",
        f"P(рост): {_esc(round(a.p_up * 100))}% · P(снижение): {_esc(round(a.p_down * 100))}%",
        f"Режим: {_esc(regime_ru)}",
    ]
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def format_gex_profile_html(p: "GEXProfileOut", ticker: str, chat_id: str | None = None) -> str:
    """Компактная карточка GEX-профиля (для лёгкой ручки без summarize)."""
    regime_ru = _REGIME_SHORT_RU.get(p.regime, p.regime)
    flip = "—" if p.gamma_flip is None else _fmt_price(p.gamma_flip)
    return (
        f"<b>{_esc(ticker)} · GEX-профиль</b>\n\n"
        f"Call Wall: {_fmt_price(p.call_wall)} (OI {_esc(p.call_wall_oi)})\n"
        f"Put Wall: {_fmt_price(p.put_wall)} (OI {_esc(p.put_wall_oi)})\n"
        f"Gamma Flip: {flip}\n"
        f"Net GEX: {_esc(p.net_gex)}\n"
        f"Режим: {_esc(regime_ru)}"
    )


# --------------------------------------------------------------------------- #
#  TA
# --------------------------------------------------------------------------- #
def format_extended_gex_html(a: "ExtendedGEXAnalysisOut", chat_id: str | None = None) -> str:
    """HTML-сообщение расширенного GEX-анализа (крипта/акции, все метрики ТЗ).

    Карточка: источник, спот, режим, направление, ключевые уровни (Zero Gamma,
    Call/Put Wall с силой), PCR, Gamma Exposure Score, Max Pain, Power Zones,
    Hedge Requirement по сценариям.
    """
    src_ru = "Крипта (Bybit)" if a.source == "crypto" else "Акция США (yfinance)"
    regime_ru = _REGIME_SHORT_RU.get(a.regime, a.regime)
    bias_ru = _TREND_LABEL_RU.get(a.directional_bias, a.directional_bias)
    bias_arrow = _ARROW.get(a.directional_bias, "◆")

    flip = "—" if a.zero_gamma is None else _fmt_price(a.zero_gamma)
    mp = "—" if a.max_pain is None else _fmt_price(a.max_pain)

    lines: list[str] = [
        f"<b>{_esc(a.symbol)} · Расширенный GEX</b>",
        f"Источник: {_esc(src_ru)}",
        f"Спот: {_fmt_price(a.spot)} · Горизонт: {_esc(a.days)} дн.",
        f"Режим: {_esc(regime_ru)} · Уклон: {bias_arrow} {_esc(bias_ru)}",
        f"Net GEX: {_esc(a.net_gex)} (Call {_esc(a.total_call_gex)} / Put {_esc(a.total_put_gex)})",
        "",
        "<b>Ключевые уровни</b>",
        f"  Zero Gamma: {flip}",
        f"  Call Wall: {_fmt_price(a.call_wall)} (сила {_esc(round(a.call_wall_strength * 100))}%)",
        f"  Put Wall: {_fmt_price(a.put_wall)} (сила {_esc(round(a.put_wall_strength * 100))}%)",
        f"  Max Pain: {mp}",
        "",
        "<b>Метрики влияния</b>",
        f"  PCR (GEX): {_esc(a.put_call_ratio)} · GEX Score: {_esc(a.gamma_exposure_score)}/100",
        f"  Σ Gamma$: {_esc(a.gamma_dollar_total)}",
    ]

    # Power Zones (до 3 сильнейших).
    if a.power_zones:
        lines.append("")
        lines.append("<b>Power Zones</b>")
        for z in a.power_zones[:3]:
            side = "Call" if z.dominant_type == "CALL" else "Put"
            lines.append(
                f"  {_fmt_price(z.center)} (±{_esc(round(z.width / 2, 2))}, "
                f"{side}, {_esc(z.n_strikes)} стр.)"
            )

    # Hedge Requirement по сценариям.
    if a.hedge_scenarios:
        lines.append("")
        lines.append("<b>Hedge Requirement</b>")
        for h in a.hedge_scenarios:
            action = "покупка" if h.shares >= 0 else "продажа"
            lines.append(
                f"  при {_esc(h.scenario_pct)}%: {_esc(abs(round(h.shares)))} акций ({action})"
            )

    return "\n".join(lines)


def format_ta_analysis_html(a: "TAAnalysisOut", chat_id: str | None = None) -> str:
    """HTML-сообщение полного TA-анализа — готовый блок из summarize."""
    return a.summarize.telegram_html_message or (
        f"<b>{_esc(a.symbol)} · Технический анализ</b>\n"
        f"Спот: {_fmt_price(a.spot)}\n"
        f"Консенсус-тренд: {_ARROW.get(a.consensus_trend, '◆')} "
        f"{_TREND_LABEL_RU.get(a.consensus_trend, a.consensus_trend)}\n"
        f"Вероятность разворота: {_esc(round(a.consensus_p_reversal * 100))}%"
    )


def format_ta_timeframe_html(tf: "TimeframeOut", ticker: str, chat_id: str | None = None) -> str:
    """Компактная карточка TA по одному таймфрейму (без summarize)."""
    t = tf.trend
    i = tf.indicators
    return (
        f"<b>{_esc(ticker)} · TA [{_esc(tf.timeframe)}]</b>\n"
        f"Цена: {_fmt_price(tf.last_close)}\n\n"
        f"Тренд: {_ARROW.get(t.direction, '◆')} "
        f"{_TREND_LABEL_RU.get(t.direction, t.direction)} "
        f"<code>{_strength_bar(t.strength)}</code> {_esc(t.strength)}/100\n"
        f"P разворота: {_esc(round(tf.reversal.p_reversal * 100))}%\n\n"
        f"Индикаторы\n"
        f"  EMA20 {_esc(i.ema20)} · EMA50 {_esc(i.ema50)} · EMA200 {_esc(i.ema200)}\n"
        f"  RSI {_esc(i.rsi)} · MACD h={_esc(i.macd_hist)}"
    )


# --------------------------------------------------------------------------- #
#  Сканер
# --------------------------------------------------------------------------- #
def format_scan_report_html(report: "ScanReportOut", chat_id: str | None = None) -> str:
    """Сводный отчёт автосканера: счётчики + статус/консенсус по тикерам."""
    scanned = getattr(report, "scanned_at", None)
    scanned_str = _esc(scanned.strftime("%Y-%m-%d %H:%M UTC")) if scanned else "—"
    lines: list[str] = [
        "<b>Автосканер TA + GEX</b>",
        f"Прогон: {scanned_str} · Интервал: {_esc(report.interval_hours)} ч",
        f"Успешно: {_esc(report.ok)} · Ошибок: {_esc(report.failed)} · Всего: {_esc(report.total)}",
        "",
    ]
    for r in report.records:
        # Консенсус-тренд из TA-отчёта (если есть), иначе статус.
        if r.ta is not None:
            trend = getattr(r.ta, "consensus_trend", None)
            if trend:
                tag = f"{_ARROW.get(trend, '◆')} {_TREND_LABEL_RU.get(trend, trend)}"
            else:
                tag = "ok"
        else:
            tag = "ошибка" if r.status == "error" else "—"
        lines.append(f"  • <b>{_esc(r.ticker)}</b> — {tag}")
    return "\n".join(lines)


def format_scan_record_html(record: "ScanRecordOut", chat_id: str | None = None) -> str:
    """Отчёт по одному тикеру: TA + GEX вместе."""
    lines: list[str] = [f"<b>{_esc(record.ticker)} · Прогон сканера</b>"]
    status_ok = record.status == "ok"
    lines.append(f"Статус: {'успешно' if status_ok else _esc(record.status)}\n")
    if record.error:
        lines.append(f"Ошибка: <i>{_esc(record.error)}</i>\n")

    ta_html = getattr(getattr(record, "ta", None), "summarize", None)
    if ta_html is not None and getattr(ta_html, "telegram_html_message", None):
        lines.append(ta_html.telegram_html_message)
    elif record.ta is not None:
        trend = getattr(record.ta, "consensus_trend", None)
        lines.append(
            f"TA: {_ARROW.get(trend, '◆')} {_TREND_LABEL_RU.get(trend, trend)}"
        )

    gex_sum = getattr(getattr(record, "gex", None), "summarize", None)
    if gex_sum is not None and getattr(gex_sum, "telegram_html_message", None):
        lines.append("")
        lines.append(gex_sum.telegram_html_message)
    elif record.gex is not None:
        d = getattr(record.gex, "direction", None)
        lines.append(
            f"\nGEX: {_ARROW.get(d, '◆')} {_TREND_LABEL_RU.get(d, d)}"
        )

    return "\n".join(lines)


# =========================================================================== #
#  Notify-функции: форматер → транспорт (ошибки логируются, не выбрасываются)
# =========================================================================== #
def _send(html: str, chat_id: str | None = None) -> dict[str, Any]:
    """Безопасная обёртка: форматнуть и отправить, не ронять вызывающего."""
    try:
        return send_telegram_message(html, parse_mode="HTML", chat_id=chat_id)
    except Exception:  # noqa: BLE001 — транспорт не должен ронять ручку/сканер
        logger.exception("Ошибка отправки в Telegram")
        return {"success": False, "errors": ["notify exception"], "batches_sent": 0}


def notify_gex_analysis(a: "GEXAnalysisOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить полный GEX-анализ в Telegram."""
    return _send(format_gex_analysis_html(a, chat_id=chat_id), chat_id=chat_id)


def notify_gex_profile(p: "GEXProfileOut", ticker: str, chat_id: str | None = None) -> dict[str, Any]:
    """Отправить GEX-профиль в Telegram."""
    return _send(format_gex_profile_html(p, ticker, chat_id=chat_id), chat_id=chat_id)


def notify_extended_gex(a: "ExtendedGEXAnalysisOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить расширенный GEX-анализ (крипта/акции) в Telegram."""
    return _send(format_extended_gex_html(a, chat_id=chat_id), chat_id=chat_id)


def notify_ta_analysis(a: "TAAnalysisOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить полный TA-анализ в Telegram."""
    return _send(format_ta_analysis_html(a, chat_id=chat_id), chat_id=chat_id)


def notify_ta_timeframe(tf: "TimeframeOut", ticker: str, chat_id: str | None = None) -> dict[str, Any]:
    """Отправить TA по одному таймфрейму в Telegram."""
    return _send(format_ta_timeframe_html(tf, ticker, chat_id=chat_id), chat_id=chat_id)


def notify_scan_report(report: "ScanReportOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить сводный отчёт сканера в Telegram."""
    return _send(format_scan_report_html(report, chat_id=chat_id), chat_id=chat_id)


def notify_scan_record(record: "ScanRecordOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить отчёт сканера по одному тикеру в Telegram."""
    return _send(format_scan_record_html(record, chat_id=chat_id), chat_id=chat_id)




# =========================================================================== #
#  Торговый сигнал (EMA-стратегия + GEX + verification) — только current_signal
# =========================================================================== #
_CONFIDENCE_LABEL_RU: dict[str, str] = {
    "high": "Высокая",
    "medium": "Средняя",
    "low": "Низкая",
    "none": "Нет сигнала",
}
_EMA_ALIGNMENT_LABEL_RU: dict[str, str] = {
    "bullish": "бычье (fast › mid › slow)",
    "bearish": "медвежье (fast ‹ mid ‹ slow)",
    "mixed": "смешанное",
}
_VWAP_POSITION_LABEL_RU: dict[str, str] = {
    "above": "выше VWAP",
    "below": "ниже VWAP",
    "at": "на VWAP",
}
_LEVEL_POSITION_LABEL_RU: dict[str, str] = {
    "above": "выше",
    "below": "ниже",
    "at": "на уровне",
}
_REGIME_INSIGHT_SIGNAL_RU: dict[str, str] = {
    "POSITIVE": (
        "Положительная гамма — дилеры гасят волатильность, цена тяготеет к "
        "Gamma Flip (mean-reversion). Сигналы по направлению к Flip усиливаются, "
        "против — ослабляются."
    ),
    "NEGATIVE": (
        "Отрицательная гамма — дилеры усиливают тренды (trend-boosting). "
        "Сигналы по направлению тренда от Flip усиливаются, контр-трендовые — "
        "ослабляются."
    ),
}
# Режимы рынка из verification.py → русские подписи.
_VERIFICATION_REGIME_RU: dict[str, str] = {
    "strong_trend": "сильный тренд",
    "weak_trend": "слабый тренд",
    "flat": "флэт",
    "impulse_volatility": "импульсная волатильность",
}


def format_signal_analysis_html(a: "SignalAnalysisOut", chat_id: str | None = None) -> str:
    """Подробное HTML-сообщение по актуальному сигналу с GEX-позиционированием и TA.

    Отправляется **только** текущий (на последнем баре) сигнал — исторические
    сигналы из ``recent_signals`` в Telegram не уходят.
    """
    s = a.current_signal
    g = a.gex_context
    action_arrow = "▲" if s.action == "buy" else ("▼" if s.action == "sell" else "◆")

    lines: list[str] = [
        f"<b>{_esc(a.symbol)} · Торговый сигнал [{_esc(a.timeframe.upper())}]</b>",
        f"Спот: {_fmt_price(a.spot)} · Обновлено: {_esc(a.generated_at.strftime('%Y-%m-%d %H:%M UTC'))}",
        "",
    ]

    # --- Вердикт сигнала ---
    is_entry = s.order_type in ("entry_long", "entry_short")
    direction_word = "LONG" if s.action == "buy" else ("SHORT" if s.action == "sell" else "HOLD")
    lines.append(f"Сигнал: {action_arrow} <b>{_esc(direction_word)}</b>"
                 + (f" ({_esc(s.reason)})" if s.reason and s.reason != "no_entry_signal" else ""))
    lines.append(f"Уверенность: {_esc(_CONFIDENCE_LABEL_RU.get(s.confidence_class, s.confidence_class))}"
                 + (f" · score входа {_esc(round(s.entry_score * 100))}/100" if is_entry else ""))

    # --- TP/SL/RR (только для входа) ---
    if is_entry and s.tp_price is not None and s.sl_price is not None:
        rr_str = f" · R/R 1:{_esc(s.rr_ratio)}" if s.rr_ratio is not None else ""
        lines.append(
            f"TP: {_fmt_price(s.tp_price)} · SL: {_fmt_price(s.sl_price)}{rr_str}"
        )
    if s.atr is not None:
        lines.append(f"ATR: {_fmt_price(s.atr)} (адаптивные TP/SL)")

    lines.append("")

    # --- Технический анализ ---
    lines.append("<b>Технический анализ</b>")
    ta_parts: list[str] = []
    if s.rsi is not None:
        rsi_state = "перекупленность" if s.rsi >= 70 else ("перепроданность" if s.rsi <= 30 else "нейтрально")
        ta_parts.append(f"RSI {_esc(s.rsi)} ({rsi_state})")
    if s.ema_alignment is not None:
        ta_parts.append(f"EMA: {_esc(_EMA_ALIGNMENT_LABEL_RU.get(s.ema_alignment, s.ema_alignment))}")
    if s.vwap_position is not None:
        ta_parts.append(_esc(_VWAP_POSITION_LABEL_RU.get(s.vwap_position, s.vwap_position)))
    if s.trend_coefficient is not None:
        ta_parts.append(f"тренд-коэф. {_esc(round(s.trend_coefficient, 2))}")
    if s.is_flat:
        ta_parts.append("флэт-зона")
    if ta_parts:
        lines.append("  " + " · ".join(ta_parts))

    # --- Verification ---
    if s.verification_score is not None:
        regime_ru = _VERIFICATION_REGIME_RU.get(s.verification_regime or "", s.verification_regime or "—")
        lines.append(
            f"  Verification: {_esc(round(s.verification_score))}/100 · "
            f"режим: {_esc(regime_ru)}"
        )
    lines.append("")

    # --- GEX-позиционирование ---
    if g is not None:
        lines.append("<b>GEX-позиционирование</b>")
        lines.append(f"  Режим: {_esc(_REGIME_SHORT_RU.get(g.regime, g.regime))}")
        if g.gamma_flip is not None:
            pos_label = _LEVEL_POSITION_LABEL_RU.get(s.spot_vs_gamma_flip or "at", "—")
            lines.append(
                f"  Gamma Flip: {_fmt_price(g.gamma_flip)} "
                f"(спот {pos_label} Flip)"
            )
        if g.z_score is not None:
            lines.append(f"  Z-score vs Flip: {_esc(round(g.z_score, 2))}")
        if g.net_gex is not None:
            lines.append(f"  Net GEX: {_esc(f'{g.net_gex:+,.0f}')} $/%spot")
        if g.call_wall is not None or g.put_wall is not None:
            cw = _fmt_price(g.call_wall) if g.call_wall is not None else "—"
            pw = _fmt_price(g.put_wall) if g.put_wall is not None else "—"
            lines.append(f"  Call Wall: {cw} · Put Wall: {pw}")
        if g.direction is not None:
            lines.append(
                f"  GEX-направление: {_ARROW.get(g.direction, '◆')} "
                f"{_TREND_LABEL_RU.get(g.direction, g.direction)}"
                + (f" · уверенность {_esc(round(g.confidence))}%" if g.confidence is not None else "")
            )
        # Влияние GEX на сигнал.
        lines.append(f"  Влияние на сигнал: ×{_esc(s.gex_multiplier)} — {_esc(s.gex_reason)}")
        insight = _REGIME_INSIGHT_SIGNAL_RU.get(g.regime)
        if insight:
            lines.append(f"  <i>{_esc(insight)}</i>")
    else:
        lines.append("<b>GEX:</b> <i>недоступен — сигнал без GEX-контекста</i>")

    lines.append("")
    lines.append(f"Баров в анализе: {_esc(a.bars_analyzed)} · "
                 f"тип: {_esc('крипта' if a.asset_type == 'crypto' else 'акция')}")

    return "\n".join(lines)


def notify_signal_analysis(a: "SignalAnalysisOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить актуальный торговый сигнал в Telegram (только current_signal).

    Исторические сигналы из ``recent_signals`` НЕ отправляются — только самый
    последний/актуальный сигнал с подробным описанием позиционирования.
    """
    return _send(format_signal_analysis_html(a, chat_id=chat_id), chat_id=chat_id)


#  Трендовые линии + фракталы HH/HL/LH/LL
# =========================================================================== #
# Для трендовых линий полный HTML уже собран сервисом в поле
# ``summarize.telegram_html_message`` (см. trendline_service). Notify-функция
# переиспользует его; если summarize нет — рендерит компактную карточку.

# Подписи направлений тренда (RU) — локальная копия, чтобы не плодить импорты.
_TL_TREND_LABEL_RU: dict[str, str] = {
    "BULLISH": "Восходящий",
    "BEARISH": "Нисходящий",
    "RANGE": "Боковик",
}


def format_trendline_analysis_html(a: "TrendlineAnalysisOut", chat_id: str | None = None) -> str:
    """HTML-сообщение анализа трендовых линий.

    Использует готовый блок из ``a.summarize.telegram_html_message`` (рендерится
    сервисом ``TrendlineService``); если summarize-блока нет — собирает
    компактную карточку из полей схемы.
    """
    summary = getattr(a, "summarize", None)
    if summary is not None and getattr(summary, "telegram_html_message", None):
        return summary.telegram_html_message

    arrow = "▲" if a.consensus_trend == "BULLISH" else ("▼" if a.consensus_trend == "BEARISH" else "◆")
    strength = getattr(getattr(a, "summarize", None), "consensus_strength", None)
    strength_str = f" · сила {_esc(round(strength))}/100" if strength is not None else ""
    asset_word = "крипта" if a.asset_type == "crypto" else "акция"
    tf_lines: list[str] = []
    for t in a.timeframes:
        tf_arrow = "▲" if t.combined_trend == "BULLISH" else ("▼" if t.combined_trend == "BEARISH" else "◆")
        tf_lines.append(
            f"  [{_esc(t.timeframe)}] {tf_arrow} "
            f"{_esc(_TL_TREND_LABEL_RU.get(t.combined_trend, t.combined_trend))} "
            f"({_esc(round(t.combined_strength))}/100) · угол {_esc(round(t.line_angle_deg, 1))}°"
        )
    body = "\n".join(tf_lines) if tf_lines else ""
    return (
        f"<b>{_esc(a.symbol)} · Трендовые линии + фракталы</b>\n"
        f"Спот: {_fmt_price(a.spot)} · тип: {_esc(asset_word)}\n"
        f"Консенсус: {arrow} "
        f"{_esc(_TL_TREND_LABEL_RU.get(a.consensus_trend, a.consensus_trend))}"
        f"{strength_str}\n"
        + body
    )


def notify_trendline_analysis(a: "TrendlineAnalysisOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить анализ трендовых линий + фракталов в Telegram."""
    return _send(format_trendline_analysis_html(a, chat_id=chat_id), chat_id=chat_id)


# =========================================================================== #
#  MACD-тренд (линии MACD/Signal + вероятность + теория игр)
# =========================================================================== #


def format_macd_trend_html(a: "MacdTrendAnalysisOut", chat_id: str | None = None) -> str:
    """HTML-сообщение MACD-тренд-анализа.

    Использует готовый блок из ``a.summarize.telegram_html_message`` (рендерится
    сервисом ``MacdTrendService``); если summarize-блока нет — собирает
    компактную карточку из полей схемы.
    """
    summary = getattr(a, "summarize", None)
    if summary is not None and getattr(summary, "telegram_html_message", None):
        return summary.telegram_html_message

    arrow = "▲" if a.consensus_trend == "BULLISH" else ("▼" if a.consensus_trend == "BEARISH" else "◆")
    strength = getattr(getattr(a, "summarize", None), "consensus_strength", None)
    strength_str = f" · сила {_esc(round(strength))}/100" if strength is not None else ""
    asset_word = "крипта" if a.asset_type == "crypto" else "акция"
    tf_lines: list[str] = []
    for t in a.timeframes:
        tf_arrow = "▲" if t.trend == "BULLISH" else ("▼" if t.trend == "BEARISH" else "◆")
        fs = "—" if t.final_trend_score is None else f"{t.final_trend_score:+.2f}"
        ang = "—" if t.angle_degrees is None else f"{t.angle_degrees:+.1f}°"
        quad = t.quadrant or "—"
        tf_lines.append(
            f"  [{_esc(t.timeframe)}] {tf_arrow} {_esc(_TREND_LABEL_RU.get(t.trend, t.trend))} · final {fs} · "
            f"{_esc(quad)} угол {ang}"
        )
    body = "\n".join(tf_lines) if tf_lines else ""
    return (
        f"<b>{_esc(a.symbol)} · Тренд по MACD</b>\n"
        f"Спот: {_fmt_price(a.spot)} · тип: {_esc(asset_word)}\n"
        f"Консенсус: {arrow} {_esc(_TREND_LABEL_RU.get(a.consensus_trend, a.consensus_trend))}"
        f"{strength_str}\n"
        + body
    )


def notify_macd_trend_analysis(a: "MacdTrendAnalysisOut", chat_id: str | None = None) -> dict[str, Any]:
    """Отправить MACD-тренд-анализ в Telegram."""
    return _send(format_macd_trend_html(a, chat_id=chat_id), chat_id=chat_id)