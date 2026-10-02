"""Форматирование сигналов: цены, время, суть, подпись и строка Telegram.

Вынесено из ``gex/signal_scanner_service.py`` (итерация 43). Функции перенесены дословно;
сервис реэкспортирует их, потому что роутер импортирует ``_esc_html`` и ``signal_line_html``
именно из сервиса — наружный контракт не менялся.

Три вещи, которые стоит знать, читая этот модуль:

* ``_signal_essence`` **намеренно не включает цену и время бара**: это ключ сравнения
  «тот же сигнал или новый». Включи их — и уведомление будет уходить на каждый тик цены
  и каждый бар. Свойство закреплено тестом ``test_essence_ignores_price_and_bar_time``;
* ``_signal_chip_label`` различает ENTRY/EXIT/ADD: «продажа» может быть и закрытием лонга,
  и открытием шорта, поэтому голые LONG/SHORT вводят в заблуждение;
* ``_esc_html(None)`` возвращает строку ``None`` (а не пустую) — зафиксировано тестом как
  наблюдение: если пропущенное поле попадёт в сообщение, пользователь увидит слово None.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from gex.assets_config import MOEX_ASSETS, DEFAULT_ASSETS

logger = logging.getLogger(__name__)

def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Достать поле из объекта-модели или dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
def _esc_html(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def fmt_signal_price(value: Any) -> str:
    """Цена сигнала без float-мусора (326.6600036621094 → 326.66).

    ≥ 1 — два знака; меньше — до 8 знаков (крипта-мелочь не теряется).
    Незначащие нули отбрасываются: 87.00 → '87'.
    """
    if value is None or value == "":
        return "?"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "?"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    # Малые цены (крипта): значащие разряды по порядку величины.
    try:
        d = int(math.floor(-math.log10(a))) + 3
    except ValueError:
        d = 8
    d = max(2, min(d, 8))
    s = f"{v:.{d}f}"
    if float(s) == 0.0:
        return f"{v:.6g}"
    return s.rstrip("0").rstrip(".")
def _to_utc_naive(value: Any) -> Optional[datetime]:
    """Привести метку времени к naive UTC (str / aware / naive) или None."""
    if value is None or value == "":
        return None
    ts = value
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts
def fmt_signal_time(value: Any) -> Optional[str]:
    """Момент формирования сигнала → '08.09 12:04 МСК (09:04 UTC)'.

    Вход — метка UTC (naive считается UTC; aware приводится к UTC). МСК =
    UTC+3 (фиксированный сдвиг, РФ без DST). Оба значения подписаны зоной,
    чтобы получатель не гадал о часовом поясе: дата в UTC-скобке опускается,
    только если она совпадает с датой МСК (иначе '07.09 21:00 UTC' рядом с
    '08.09 00:00 МСК' — случай MOEX-дневок, чья метка = полночь МСК).
    Возвращает None, если value пустое/непарсится.
    """
    ts = _to_utc_naive(value)
    if ts is None:
        return None
    msk = ts + timedelta(hours=3)
    msk_s = msk.strftime("%d.%m %H:%M")
    if msk.date() == ts.date():
        utc_s = ts.strftime("%H:%M UTC")
    else:
        utc_s = ts.strftime("%d.%m %H:%M UTC")
    return f"{msk_s} МСК ({utc_s})"
def _signal_essence(sig: Any) -> str:
    """«Суть» активного сигнала для сравнения — БЕЗ цены и времени бара.

    Суть = само условие сигнала: action (BUY/SELL) + order_type
    (entry_long/exit_long/add_long/...). Смена цены или «переоткрытие» того же
    сигнала на новом баре (новый timestamp) НЕ считаются изменением — иначе
    уведомления уходят на каждый тик цены/каждый бар. Пустой сигнал → ''.
    """
    if sig is None:
        return ""
    action = str(_field(sig, "action", "") or "").upper()
    if not action:
        return ""
    ot = str(_field(sig, "order_type", "") or "")
    return f"{action}|{ot}"
_CHIP_LABELS: dict[str, str] = {
    "entry_long": "LONG ENTRY",
    "exit_long": "LONG EXIT",
    "entry_short": "SHORT ENTRY",
    "exit_short": "SHORT EXIT",
    "add_long": "LONG ADD",
    "add_short": "SHORT ADD",
}
_REASON_TO_OT: dict[str, str] = {
    "long_entry": "entry_long",
    "long_exit": "exit_long",
    "short_entry": "entry_short",
    "short_exit": "exit_short",
    "long_add": "add_long",
    "short_add": "add_short",
}


def _signal_chip_label(sig: Any) -> Optional[str]:
    """Чип ПОЛНОЙ семантики сигнала как в UI: exit_long → 'LONG EXIT'.

    По order_type (entry_long/exit_long/entry_short/exit_short/add_long/
    add_short), фолбэк — reason. Голые LONG/SHORT (или «продажа») вводят в
    заблуждение: «продажа» может быть и закрытием лонга, и открытием шорта.
    """
    if sig is None:
        return None
    ot = str(_field(sig, "order_type", "") or "").lower()
    if ot in _CHIP_LABELS:
        return _CHIP_LABELS[ot]
    reason = str(_field(sig, "reason", "") or "").lower()
    ot_from_reason = _REASON_TO_OT.get(reason)
    if ot_from_reason and ot_from_reason in _CHIP_LABELS:
        return _CHIP_LABELS[ot_from_reason]
    return None
def signal_line_html(ticker: str, timeframe: str, sig: Any, when: Optional[datetime] = None) -> Optional[str]:
    """Одна строка сводки TG.

    '<b>NVDA</b> [4h]: <b>LONG EXIT</b> · продажа по 229.25 · score 20% ·
    сформирован 08.09 15:00 МСК (12:00 UTC) · отправлен 08.09 17:17 МСК (14:17 UTC)'.

    После тикера/ТФ — чип полной семантики (LONG/SHORT ENTRY/EXIT/ADD) как в
    UI, затем RU-слово действия (покупка/продажа). Два момента, каждый в
    fmt_signal_time (значения в МСК и UTC): «сформирован» = signal.timestamp —
    дата/время, когда сигнал сформировал алгоритм (бар записи сигнала);
    «отправлен» = ``when`` (момент скана/сборки сообщения; по умолчанию —
    timestamp сигнала, тогда выводится одно время).

    Возвращает None, если в сигнале нет действия (форматировать нечего).
    """
    if sig is None:
        return None
    action = str(_field(sig, "action", "") or "").upper()
    action_word = {"BUY": "покупка", "SELL": "продажа"}.get(action, "нейтрально")
    score_v = _field(sig, "entry_score", None)
    score = "?"
    if score_v is not None:
        try:
            score = str(int(float(score_v) * 100))
        except (TypeError, ValueError):
            pass
    chip = _signal_chip_label(sig)
    head = f"  <b>{_esc_html(ticker)}</b> [{_esc_html(timeframe)}]: "
    if chip:
        head += f"<b>{chip}</b> · "
    parts = [
        head
        + f"{action_word} по {fmt_signal_price(_field(sig, 'price', None))} · score {score}%",
    ]
    # Два момента в строке, каждый с подписями зон (fmt_signal_time):
    #  - «сформирован» — signal.timestamp: когда алгоритм сформировал сигнал
    #    (бар/дата-время записи сигнала, см. _extract_recent_signals);
    #  - «отправлен» — when (момент скана/сборки сообщения; по умолчанию —
    #    тот же timestamp сигнала).
    formed = fmt_signal_time(_field(sig, "timestamp", None))
    sent = fmt_signal_time(when if when is not None else _field(sig, "timestamp", None))
    time_parts: list[str] = []
    if formed:
        time_parts.append(f"сформирован {formed}")
    if sent and (not formed or sent != formed):
        time_parts.append(f"отправлен {sent}")
    if time_parts:
        parts.append(" · ".join(time_parts))
    return " · ".join(parts)
