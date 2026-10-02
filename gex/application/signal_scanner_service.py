"""Signal scanner service — per-user multi-ticker + multi-timeframe scanner.

Каждый пользователь настраивает до 10 инструментов (тикер + таймфрейм).
Инструменты хранятся в БД (user_instruments). Доступ только по подписке EXTENDED.
Фоновый поток опрашивает рынок для ВСЕХ пользователей и отправляет
персональные уведомления в Telegram.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from gex.adapters.persistence.database import SessionLocal
from gex.auth.user_instrument import UserInstrument
from gex.application.signal_service import SignalService, position_to_dict, with_entry_context
from gex.adapters.providers.catalog import CANONICAL_TIMEFRAMES

logger = logging.getLogger(__name__)

# Форматтеры живут в gex/formatters/signals.py; здесь — реэкспорт, потому что роутер
# импортирует _esc_html и signal_line_html именно из сервиса (контракт не меняем).
from gex.formatters.signals import (  # noqa: E402
    _esc_html,
    _field,
    _signal_chip_label,
    _signal_essence,
    fmt_signal_price,
    fmt_signal_time,
    signal_line_html,
)

MAX_INSTRUMENTS = 10
POLL_INTERVAL_SECONDS = 300  # 5 минут
SUPPORTED_TIMEFRAMES = CANONICAL_TIMEFRAMES


# ── Форматирование строк сводки для Telegram ─────────────────────────────
# Общее для фонового цикла (SignalScannerService._notify_user) и ручного
# прогона (routers/signal_scanner_router.py), чтобы текст не разъезжался.














_CHIP_LABELS: dict[str, str] = {
    "entry_long": "LONG ENTRY",
    "exit_long": "LONG EXIT",
    "entry_short": "SHORT ENTRY",
    "exit_short": "SHORT EXIT",
    "add_long": "LONG ADD",
    "add_short": "SHORT ADD",
}

# reason (long_entry/…) → order_type (entry_long/…) — запасной источник.
_REASON_TO_OT: dict[str, str] = {
    "long_entry": "entry_long",
    "long_exit": "exit_long",
    "short_entry": "entry_short",
    "short_exit": "exit_short",
    "long_add": "add_long",
    "short_add": "add_short",
}






@dataclass
class ScannerInstrument:
    """Один инструмент сканера."""

    ticker: str
    timeframe: str
    latest_signals: list = field(default_factory=list)
    last_scan: Optional[datetime] = None
    error: Optional[str] = None
    #: Сырые метрики режима (тренд/флэт на 200 барах), СЛАЙДЕР-НЕЗАВИСИМЫЕ.
    #: Вердикт (FLAT/UP/DOWN) считается на чтении по личному слайдеру
    #: пользователя (см. gex/trend_regime.py).
    regime: Optional[dict] = None
    #: Текущая позиция по машине состояний стратегии
    #: (``{"side": flat|long|short, "avg_price": float|None, "since": iso|None}``).
    position: Optional[dict] = None

    @property
    def key(self) -> str:
        return f"{self.ticker.upper()}:{self.timeframe}"


@dataclass
class SignalScannerReport:
    """Сводный отчёт."""

    scanned_at: datetime
    instruments: list[ScannerInstrument]
    running: bool
    pool_interval_seconds: int


class SignalScannerService:
    """Персональный сканер сигналов.

    Тредобезопасность: threading.Lock вокруг кэша результатов.
    Результаты кэшируются в памяти (instruments + last_signals).
    """

    def __init__(self, signal_service: SignalService):
        self._signal_service = signal_service
        # Кэш: user_id → list[ScannerInstrument]
        self._cache: dict[str, list[ScannerInstrument]] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        # Состояние уведомлений: user_id → {инструмент-key: «суть» активного
        # сигнала, "__init": "1"}. Дублируется в Redis (переживает рестарты):
        # уведомления шлются ТОЛЬКО при изменении условия сигнала
        # (action/order_type — без цены и времени бара!) или при
        # добавлении/удалении инструмента; повтор того же сигнала на новом
        # баре/с новой ценой не уведомляется.
        self._sig_state: dict[str, dict] = {}

    # ── Per-user watchlist management ───────────────────────────────

    def set_watchlist(self, user_id: str, pairs: list[dict]) -> list[ScannerInstrument]:
        """Установить инструменты пользователя (макс 10).

        Сохраняет в БД + обновляет кэш.
        """
        if len(pairs) > MAX_INSTRUMENTS:
            raise ValueError(f"Максимум {MAX_INSTRUMENTS} инструментов, получено {len(pairs)}")

        instruments: list[ScannerInstrument] = []
        seen: set[str] = set()

        for p in pairs:
            ticker = str(p.get("ticker", "")).strip().upper()
            tf = str(p.get("timeframe", "")).strip().lower()
            if not ticker:
                raise ValueError("ticker обязателен")
            if tf not in SUPPORTED_TIMEFRAMES:
                raise ValueError(f"Неподдерживаемый таймфрейм '{tf}'")
            key = f"{ticker}:{tf}"
            if key in seen:
                continue
            seen.add(key)
            instruments.append(ScannerInstrument(ticker=ticker, timeframe=tf))

        # Старые ключи — для событий «добавлен/удалён» (из кэша или БД)
        with self._lock:
            cached_old = self._cache.get(user_id)
        old_keys = {i.key for i in cached_old} if cached_old is not None else None

        # Сохранить в БД
        db = SessionLocal()
        try:
            if old_keys is None:
                rows = db.query(UserInstrument).filter(UserInstrument.user_id == user_id).all()
                old_keys = {f"{r.ticker.strip().upper()}:{r.timeframe.strip().lower()}" for r in rows}
            # Удалить старые
            db.query(UserInstrument).filter(UserInstrument.user_id == user_id).delete()
            # Вставить новые
            for instr in instruments:
                db.add(UserInstrument(user_id=user_id, ticker=instr.ticker, timeframe=instr.timeframe))
            db.commit()
        finally:
            db.close()

        # Обновить кэш (сохраняем существующие сигналы если были)
        with self._lock:
            old = self._cache.get(user_id, [])
            old_map = {i.key: i for i in old}
            merged = []
            for instr in instruments:
                old_instr = old_map.get(instr.key)
                if old_instr:
                    instr.position = old_instr.position
                if old_instr and old_instr.latest_signals:
                    instr.latest_signals = old_instr.latest_signals
                    instr.last_scan = old_instr.last_scan
                merged.append(instr)
            self._cache[user_id] = merged

        # События изменения списка: добавление/удаление инструмента → TG
        new_keys = {i.key for i in instruments}
        added_keys = sorted(new_keys - old_keys)
        removed_keys = sorted(old_keys - new_keys)
        self._sync_watchlist_state(user_id, added_keys, removed_keys)
        self._notify_watchlist_changes(user_id, added_keys, removed_keys)

        logger.info("SignalScanner: user=%s watchlist=%d инструментов (+%d, −%d)",
                    user_id, len(instruments), len(added_keys), len(removed_keys))
        return [ScannerInstrument(ticker=i.ticker, timeframe=i.timeframe, latest_signals=list(i.latest_signals), last_scan=i.last_scan, error=i.error) for i in merged]

    def get_watchlist(self, user_id: str) -> list[ScannerInstrument]:
        """Вернуть инструменты пользователя из кэша или БД."""
        with self._lock:
            if user_id in self._cache:
                return [self._copy(i) for i in self._cache[user_id]]

        # Загрузить из БД
        db = SessionLocal()
        try:
            rows = db.query(UserInstrument).filter(UserInstrument.user_id == user_id).all()
            instruments = [ScannerInstrument(ticker=r.ticker, timeframe=r.timeframe) for r in rows]
        finally:
            db.close()

        with self._lock:
            self._cache[user_id] = instruments
        return [self._copy(i) for i in instruments]

    def get_all_active_users(self) -> list[str]:
        """Вернуть user_id всех пользователей с инструментами + активной подпиской EXTENDED.

        Подписка считается активной, если статус EXTENDED/ADMIN и срок
        ``subscription_expires_at`` не наступил (ADMIN — без срока, всегда активен).
        """
        from gex.auth.models import User
        from sqlalchemy import or_

        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            rows = (
                db.query(UserInstrument.user_id)
                .distinct()
                .join(User, User.id == UserInstrument.user_id)
                .filter(
                    User.subscription_status.in_(["EXTENDED", "ADMIN"]),
                    or_(
                        User.subscription_status == "ADMIN",
                        User.subscription_expires_at.is_(None),
                        User.subscription_expires_at > now,
                    ),
                )
                .all()
            )
            return [r[0] for r in rows]
        finally:
            db.close()

    # ── Сканирование ────────────────────────────────────────────────

    def scan_now(self, user_id: str | None = None, notify: bool = True) -> SignalScannerReport:
        """Прогон инструментов: конкретного пользователя или всех активных.

        ``notify=False`` — сканировать без отправки сводки в Telegram
        (ручной прогон из UI: сводку шлёт сама ручка при notify=true,
        чтобы не дублировать сообщение).
        """
        if user_id:
            return self._scan_user(user_id, notify=notify)

        # Сканировать всех активных пользователей
        user_ids = self.get_all_active_users()
        for uid in user_ids:
            try:
                self._scan_user(uid)
            except Exception as exc:
                logger.error("SignalScanner: ошибка сканирования user=%s: %s", uid, exc)

        # Для отчёта берём первого или возвращаем пустой
        if user_ids:
            return self.get_report(user_ids[0])
        return self._empty_report()

    def _scan_user(self, user_id: str, notify: bool = True) -> SignalScannerReport:
        instruments = self.get_watchlist(user_id)
        if not instruments:
            return self._empty_report()

        # Персональные параметры позиционной машины (трейлинг-стоп/разворот):
        # применяются на скане пользователя — его сигналы и уведомления.
        trail_pct, reverse = self._user_scan_params(user_id)

        scanned: list[ScannerInstrument] = []
        for instr in instruments:
            result = self._scan_one(instr, trailing_pct=trail_pct, reverse=reverse)
            scanned.append(result)

        with self._lock:
            self._cache[user_id] = scanned

        # Уведомления об изменениях сигналов: фон (notify=True) шлёт только
        # реальные изменения; ручной прогон (notify=False) лишь синхронизирует
        # состояние — полную сводку при notify=true шлёт роутер.
        self._apply_changes(user_id, scanned, send=notify)

        logger.info("SignalScanner: user=%s — %d инструментов", user_id, len(scanned))
        return self._build_report(scanned)

    def get_report(self, user_id: str) -> SignalScannerReport:
        instruments = self.get_watchlist(user_id)
        return self._build_report(instruments)

    # ── Telegram-уведомления (только события, без повторов) ─────────

    def _tg_chat_id(self, user_id: str) -> Optional[str]:
        """chat_id, если юзер подключил Telegram, подписка активна и telegram_notify=on."""
        from gex.auth.models import User, subscription_is_active

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.telegram_chat_id:
                return None
            if not subscription_is_active(user):
                return None
            if not user.telegram_notify:
                return None
            return user.telegram_chat_id
        finally:
            db.close()

    @staticmethod
    def _send_tg(chat_id: str, lines: list[str]) -> None:
        from gex.adapters.notifications.telegram_sender import send_telegram_message

        try:
            send_telegram_message("\n".join(lines), parse_mode="HTML", chat_id=chat_id)
        except Exception as exc:
            logger.error("SignalScanner: ошибка отправки chat=%s: %s", chat_id, exc)

    def _load_state(self, user_id: str) -> dict:
        """Состояние «сутей» сигналов пользователя (память → Redis при старте)."""
        with self._lock:
            st = self._sig_state.get(user_id)
        if st is not None:
            return st
        st = {}
        try:
            from gex.adapters.cache.redis_client import cache_key, deserialize_value, get_redis
            redis = get_redis()
            if redis is not None and redis.connected:
                raw = redis.get(cache_key("sigstate3", user_id))
                if raw:
                    v = deserialize_value(raw)
                    if isinstance(v, dict):
                        st = {str(k): ("" if val is None else str(val)) for k, val in v.items()}
        except Exception:
            st = {}
        with self._lock:
            self._sig_state.setdefault(user_id, st)
            return self._sig_state[user_id]

    def _save_state(self, user_id: str, state: dict) -> None:
        with self._lock:
            self._sig_state[user_id] = dict(state)
        try:
            from gex.adapters.cache.redis_client import cache_key, get_redis
            redis = get_redis()
            if redis is not None and redis.connected:
                redis.set(cache_key("sigstate3", user_id), dict(state), ex=30 * 24 * 3600)
        except Exception:
            pass

    def _sync_watchlist_state(self, user_id: str, added_keys: list[str], removed_keys: list[str]) -> None:
        """Добавленные инструменты фиксируем как «сигнала нет», удалённые — убираем."""
        if not added_keys and not removed_keys:
            return
        state = self._load_state(user_id)
        state["__init"] = "1"
        for key in added_keys:
            state.setdefault(key, "")
        for key in removed_keys:
            state.pop(key, None)
        self._save_state(user_id, state)

    def _notify_watchlist_changes(self, user_id: str, added_keys: list[str], removed_keys: list[str]) -> None:
        """TG при добавлении/удалении инструмента (событие списка)."""
        if not added_keys and not removed_keys:
            return
        chat_id = self._tg_chat_id(user_id)
        if not chat_id:
            return
        lines = ["<b>Сигнальный сканер — список инструментов</b>", ""]
        for key in added_keys:
            tk, tf = key.rsplit(":", 1)
            lines.append(f"  + <b>{_esc_html(tk)}</b> [{_esc_html(tf)}]: инструмент добавлен")
        for key in removed_keys:
            tk, tf = key.rsplit(":", 1)
            lines.append(f"  − <b>{_esc_html(tk)}</b> [{_esc_html(tf)}]: инструмент удалён")
        self._send_tg(chat_id, lines)

    def _apply_changes(self, user_id: str, instruments: list[ScannerInstrument], send: bool) -> None:
        """Сравнить сигналы со состоянием и уведомить только об ИЗМЕНЕНИЯХ.

        Изменением считается смена «сути» активного сигнала инструмента
        (action/order_type — цена и время бара НЕ входят в суть, тики цены
        не уведомляются) или появление сигнала; повтор того же сигнала
        на новом баре НЕ уведомляется. Первый скан после
        старта (нет сохранённого состояния) молча фиксирует текущее состояние.
        """
        state = self._load_state(user_id)
        initialized = state.get("__init") == "1"
        new_state: dict = {"__init": "1"}
        changed: list[str] = []

        for instr in instruments:
            key = instr.key
            s0 = (instr.latest_signals or [None])[0]
            essence = _signal_essence(s0)
            new_state[key] = essence
            if not initialized:
                continue
            prev = state.get(key)
            if prev is None:
                prev = ""  # инструмента не было в состоянии — считаем «сигнала не было»
            if essence == prev:
                continue
            if essence:
                # when = момент скана инструмента (время срабатывания, не метка бара)
                when = instr.last_scan or datetime.now(timezone.utc)
                line = signal_line_html(instr.ticker, instr.timeframe, s0, when=when)
                changed.append(line or f"  <b>{_esc_html(instr.ticker)}</b> [{_esc_html(instr.timeframe)}]: обновление сигнала")
            else:
                changed.append(f"  <b>{_esc_html(instr.ticker)}</b> [{_esc_html(instr.timeframe)}]: сигнал снят (нет активного)")

        self._save_state(user_id, new_state)

        if not send or not changed:
            return
        chat_id = self._tg_chat_id(user_id)
        if not chat_id:
            return
        lines = ["<b>Сигнальный сканер — обновления</b>", ""] + changed
        self._send_tg(chat_id, lines)
        logger.info("SignalScanner: изменения отправлены user=%s (%d)", user_id, len(changed))

    # ── Фоновый цикл ────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="signal-scanner-loop")
        self._thread.start()
        self._running = True
        logger.info("SignalScanner: фоновый поток запущен (интервал %dс)", POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        logger.info("SignalScanner: остановлен")

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self) -> None:
        """Фоновый цикл: сканирует ВСЕХ активных пользователей каждые 5 минут."""
        self._running = True
        while not self._stop_event.is_set():
            try:
                self.scan_now()  # сканирует всех
            except Exception as exc:
                logger.error("SignalScanner: ошибка в цикле: %s", exc)
            self._stop_event.wait(POLL_INTERVAL_SECONDS)

    # ── Параметры пользователя / сканирование одного инструмента ─────

    @staticmethod
    def _user_scan_params(user_id: str) -> tuple[float, bool]:
        """Персональные параметры скана: (трейлинг-стоп %, разворот).

        Из настроек сканера пользователя (одна строка на аккаунт). Ошибки и
        отсутствие настроек → дефолты схемы: 0 (трейлинг выключен) и разворот
        включён.
        """
        pct, rev = 0.0, True
        db = SessionLocal()
        try:
            from gex.auth.models import User
            from gex.auth.settings_router import load_scanner_settings
            user = db.query(User).filter(User.id == user_id).first()
            settings = load_scanner_settings(db, user)
            try:
                pct = float(settings.get("trailing_pct_personal") or 0.0)
            except (TypeError, ValueError):
                pct = 0.0
            rev = bool(settings.get("reverse_close", True))
        except Exception as exc:  # noqa: BLE001 — настройки вспомогательные
            logger.warning("SignalScanner: настройки user=%s недоступны: %s", user_id, exc)
        finally:
            db.close()
        return pct, rev

    def _scan_one(
        self,
        instr: ScannerInstrument,
        trailing_pct: float = 0.0,
        reverse: bool = False,
    ) -> ScannerInstrument:
        stored = list(instr.latest_signals)
        try:
            analysis = self._signal_service.analyze_signals(
                instr.ticker.upper(),
                timeframe=instr.timeframe,
                n_recent=5,
                bars=200,
                trailing_pct=(trailing_pct or None),
                reverse=reverse,
            )
            raw_signals = list(analysis.recent_signals) if analysis.recent_signals else []
        except (ValueError, RuntimeError) as exc:
            return ScannerInstrument(ticker=instr.ticker, timeframe=instr.timeframe, latest_signals=stored, last_scan=instr.last_scan, error=str(exc), position=instr.position)
        except Exception as exc:
            return ScannerInstrument(ticker=instr.ticker, timeframe=instr.timeframe, latest_signals=stored, last_scan=instr.last_scan, error=f"Ошибка: {exc}", position=instr.position)

        # Сырые метрики режима (тренд/флэт) — вердикт по личному слайдеру на чтении.
        regime_metrics = self._extract_regime(analysis)
        # Текущая позиция по машине состояний — обновляется на каждом скане.
        position = position_to_dict(analysis)

        # Свежесть: сигналы старее торгового окна (3 торговых дня / 18 баров 4h)
        # НЕ считаются активными. Иначе «последним сигналом» всплывает запись
        # недельной давности: формирующийся бар то даёт сигнал, то нет, и при
        # его исчезновении [0] «откатывается» к старой записи → ложные события
        # в TG («продажа по 30.62 · 31.08 21:00» вместо свежего состояния).
        if raw_signals:
            from gex.application.auto_scanner_service import AutoScannerService
            fresh = [s for s in raw_signals if AutoScannerService._is_fresh_signal(s, instr.timeframe)]
            # Якорь очерёдности: «выход без входа перед ним» не показываем —
            # если самый старый из свежих выход/добавление, доводим до входа.
            fresh = with_entry_context(raw_signals, fresh)
        else:
            fresh = []

        if not fresh:
            return ScannerInstrument(ticker=instr.ticker, timeframe=instr.timeframe, latest_signals=[], last_scan=datetime.now(timezone.utc), error=None, regime=regime_metrics, position=position)

        if stored and self._signals_equal(fresh[0], stored[0]):
            return ScannerInstrument(ticker=instr.ticker, timeframe=instr.timeframe, latest_signals=stored, last_scan=datetime.now(timezone.utc), error=None, regime=regime_metrics, position=position)

        return ScannerInstrument(ticker=instr.ticker, timeframe=instr.timeframe, latest_signals=fresh, last_scan=datetime.now(timezone.utc), error=None, regime=regime_metrics, position=position)

    @staticmethod
    def _extract_regime(analysis: Any) -> Optional[dict]:
        """Вытащить сырые (слайдер-независимые) метрики режима из ответа анализа.

        ``None`` — детектор не смог посчитать (мало истории): сигналы НЕ режем.
        """
        regime = getattr(analysis, "regime", None)
        if regime is None:
            return None
        metrics = getattr(regime, "metrics", None)
        if isinstance(metrics, dict) and metrics:
            return dict(metrics)
        if isinstance(regime, dict):
            inner = regime.get("metrics")
            if isinstance(inner, dict) and inner:
                return dict(inner)
        return None

    @staticmethod
    def _signals_equal(a: Any, b: Any) -> bool:
        if type(a) != type(b):
            return False
        try:
            if a.action != b.action: return False
            if a.price != b.price: return False
            ts_a = getattr(a, "timestamp", None)
            ts_b = getattr(b, "timestamp", None)
            if ts_a and ts_b:
                from datetime import timedelta
                if abs(ts_a - ts_b) > timedelta(minutes=1):
                    return False
            return True
        except (AttributeError, TypeError):
            return False

    # ── Вспомогательные ─────────────────────────────────────────────

    def _build_report(self, instruments: list[ScannerInstrument]) -> SignalScannerReport:
        timestamps = [i.last_scan for i in instruments if i.last_scan]
        latest = max(timestamps) if timestamps else datetime.now(timezone.utc)
        return SignalScannerReport(scanned_at=latest, instruments=instruments, running=self._running, pool_interval_seconds=POLL_INTERVAL_SECONDS)

    def _empty_report(self) -> SignalScannerReport:
        return SignalScannerReport(scanned_at=datetime.now(timezone.utc), instruments=[], running=self._running, pool_interval_seconds=POLL_INTERVAL_SECONDS)

    @staticmethod
    def _copy(instr: ScannerInstrument) -> ScannerInstrument:
        return ScannerInstrument(ticker=instr.ticker, timeframe=instr.timeframe, latest_signals=list(instr.latest_signals), last_scan=instr.last_scan, error=instr.error, regime=dict(instr.regime) if instr.regime else None, position=dict(instr.position) if instr.position else None)
