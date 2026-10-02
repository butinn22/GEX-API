"""FIFO-кэш закрытых баров: память процесса + Redis (ring: adapters).

Зачем
-----
До этого слоя каждый потребитель свечей перекачивал историю целиком (до 730 дней часовых
баров — ~5076 строк на тикер), даже если нужны были последние 200–500 баров, а между двумя
запросами закрылась одна свеча. Разбор — ``docs/TA-CANDLE-FLOW.md``.

Здесь хранится **одна серия на (тикер × таймфрейм × провайдер)** в виде списка баров
``(t, o, h, l, c, v)``, отсортированного по времени, и удерживается только хвост из
``max_bars`` последних баров: новые бары дописываются в конец, самые старые вытесняются из
головы — **FIFO**.

Два яруса, и это не дублирование:

* **память процесса** — ``OrderedDict`` с ограничением числа серий: мгновенные повторные
  чтения и гарантия «кэш живёт, даже когда Redis недоступен»;
* **Redis** — версионированный JSON-конверт: серии видны всем процессам и переживают
  рестарт.

Инвариант «бары обновляются только когда закрылись новые»
----------------------------------------------------------
:meth:`BarStore.update` пишет бары в Redis **только** если появился хотя бы один бар новее
последнего сохранённого; в противном случае сдвигается только ``checked_at`` в памяти
процесса (факт «мы проверяли провайдера», чтобы не ходить к нему чаще, чем нужно). Так
закрытые свечи никогда не переписываются заново, а Redis не получает запись на каждый
промах.

Расписание проверок
-------------------
:meth:`BarStore.is_due` отвечает «пора ли спросить провайдера о новом баре»: проверка
назначается на момент **предполагаемого закрытия следующего бара** (последний бар + длина
таймфрейма, с учётом торговых сессий из ``gex.domain.freshness.SESSIONS``). Следствие:
дневные серии проверяются ~раз в день после закрытия сессии, часовые — раз в час во время
сессии, а «вне сессии» провайдер не дёргается вовсе. Для круглосуточных рынков
(``sessions=None``, крипта) следующий бар закрывается через ровно одну длину таймфрейма.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd

from gex.adapters.cache.keys import PROVIDER_YFINANCE, bars_key
from gex.domain.freshness import MSK, SESSIONS

logger = logging.getLogger(__name__)

__all__ = [
    "BAR_SCHEMA_VERSION",
    "BAR_CLOSE_SLACK_S",
    "DEFAULT_MAX_BARS",
    "TF_SECONDS",
    "BarStore",
    "CachedBars",
    "bars_to_frame",
    "frame_to_bars",
    "next_close_after",
]

#: Версия схемы значения в Redis. Меняется при несовместимом изменении полей баров.
BAR_SCHEMA_VERSION = 1

#: Предел серии по умолчанию: 500 баров достаточно страницам (окно линий тренда — 300,
#: EMA200, сканер — 500). Каждая серия (тикер × ТФ) удерживает свой хвост.
DEFAULT_MAX_BARS = 500

#: Срок жизни ключа в Redis. Намного больше цикла «закрылся бар» (день) — серия обязана
#: пережить выходные и каникулы, иначе после паузы начнётся заново с бутстрапа.
DEFAULT_REDIS_TTL_S = 7 * 24 * 3600

#: Потолок серий в памяти процесса (тикеры × 4 ТФ): защита от неограниченного роста.
DEFAULT_MEM_SERIES = 1024

#: Длина одного бара по таймфрейму (для расписания проверок).
TF_SECONDS: dict[str, int] = {"1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}

#: Запас после расчётного времени закрытия бара: сетевые задержки и «бар ещё не виден».
BAR_CLOSE_SLACK_S = 60

#: Сколько дней назад начинать инкрементальную догрузку (перекрытие с последним баром).
_INCREMENT_BACKOFF_DAYS = 3

#: Сколько дней вперёд искать следующую торговую сессию (каникулы/выходные).
_SESSION_SCAN_DAYS = 15


def _bar_ts(bar: dict) -> float:
    """Epoch-секунды бара из его ISO-метки ``t``."""
    raw = str(bar.get("t") or "")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw).timestamp()


def _iso(dt: datetime) -> str:
    """ISO-метка бара: всегда UTC, явный суффикс."""
    moment = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def frame_to_bars(df: pd.DataFrame) -> list[dict]:
    """OHLCV-кадр → список баров ``(t, o, h, l, c, v)`` с UTC-метками.

    «Список баров», а не pickle-кадр, — осознанный формат: он читается глазами в
    ``redis-cli``, переживает смену версий pandas и хранит ровно то, что обещано
    (OHLC, объём, дата и время каждого бара), без служебных колонок.
    """
    bars: list[dict] = []
    for ts, row in df.iterrows():
        moment = pd.Timestamp(ts)
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")
        bars.append(
            {
                "t": _iso(moment.to_pydatetime()),
                "o": round(float(row.get("Open") or 0.0), 6),
                "h": round(float(row.get("High") or 0.0), 6),
                "l": round(float(row.get("Low") or 0.0), 6),
                "c": round(float(row.get("Close") or 0.0), 6),
                "v": round(float(row.get("Volume") or 0.0), 2),
            }
        )
    return bars


def bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    """Список баров → OHLCV-кадр с UTC DatetimeIndex (контракт ``TATimeframesFetcher``)."""
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    idx = [pd.Timestamp(_bar_ts(b), unit="s", tz="UTC") for b in bars]
    frame = pd.DataFrame(
        {
            "Open": [b["o"] for b in bars],
            "High": [b["h"] for b in bars],
            "Low": [b["l"] for b in bars],
            "Close": [b["c"] for b in bars],
            "Volume": [b.get("v", 0.0) for b in bars],
        },
        index=idx,
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def next_close_after(tf: str, baseline_ts: float, sessions: Optional[tuple[str, ...]]) -> float:
    """Момент предполагаемого закрытия следующего бара таймфрейма ``tf`` после ``baseline_ts``.

    ``sessions=None`` — круглосуточный рынок: бар закрывается ровно через длину таймфрейма.
    Иначе — торговые сессии из ``gex.domain.freshness.SESSIONS`` (MSK): дневной бар
    закрывается вместе с сессией, интрадей-бар — на границе сессии с шагом таймфрейма
    (последний, неполный, бар — в момент закрытия сессии).

    Чистая функция: тестируется без сети и Redis.
    """
    step = TF_SECONDS.get(tf, 3600)
    if sessions is None:
        return baseline_ts + step

    known = [SESSIONS[s] for s in sessions if s in SESSIONS]
    if not known:
        return baseline_ts + step

    t0 = datetime.fromtimestamp(baseline_ts, tz=timezone.utc).astimezone(MSK)
    for day_offset in range(_SESSION_SCAN_DAYS):
        day: date = (t0.date() + timedelta(days=day_offset))
        # ``is_open`` проверяет и день недели, и окно сессии, поэтому время — момент
        # внутри сессии (её старт): здесь нам нужен только признак «торговый день».
        if not any(
            sess.is_open(datetime.combine(day, dtime(sess.start_min // 60, sess.start_min % 60), tzinfo=MSK))
            for sess in known
        ):
            continue  # выходной (сб/вс) — бары не закрываются
        for sess in known:
            start = datetime.combine(day, dtime(sess.start_min // 60, sess.start_min % 60), tzinfo=MSK)
            end = datetime.combine(day, dtime(sess.end_min // 60, sess.end_min % 60), tzinfo=MSK)
            if end.timestamp() <= baseline_ts:
                continue  # эта сессия уже закрылась
            if tf == "1d":
                return end.timestamp()  # дневной бар закрывается вместе с сессией
            if baseline_ts < start.timestamp():
                return start.timestamp() + step  # первый бар новой сессии
            k = int((baseline_ts - start.timestamp()) // step) + 1
            return min(start.timestamp() + k * step, end.timestamp())
    return baseline_ts + step  # страховка: каникулы дольше горизонта сканирования


class CachedBars:
    """Серия баров из кэша с метаданными проверки."""

    __slots__ = ("bars", "last_ts", "stored_at", "checked_at", "source")

    def __init__(
        self,
        bars: list[dict],
        *,
        stored_at: float,
        checked_at: float,
        source: str = "",
    ) -> None:
        self.bars = bars
        self.last_ts = _bar_ts(bars[-1]) if bars else 0.0
        self.stored_at = float(stored_at)
        self.checked_at = float(checked_at)
        self.source = source or ""

    def __len__(self) -> int:
        return len(self.bars)

    def to_frame(self) -> pd.DataFrame:
        return bars_to_frame(self.bars)

    def as_payload(self) -> dict:
        return {
            "v": BAR_SCHEMA_VERSION,
            "src": self.source,
            "stored_at": self.stored_at,
            "checked_at": self.checked_at,
            "bars": self.bars,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> Optional["CachedBars"]:
        if not isinstance(payload, dict) or payload.get("v") != BAR_SCHEMA_VERSION:
            return None
        bars = payload.get("bars")
        if not isinstance(bars, list):
            return None
        stored = payload.get("stored_at")
        checked = payload.get("checked_at")
        if not isinstance(stored, (int, float)) or not isinstance(checked, (int, float)):
            return None
        return cls(
            bars, stored_at=float(stored), checked_at=float(checked),
            source=str(payload.get("src") or ""),
        )


class BarStore:
    """FIFO-кэш закрытых баров: память процесса → Redis (запись только при новых барах).

    Потокобезопасен: методы чтения/записи защищены одним замком (сериализация маленькая,
    а состояние обязано быть консистентным между проверкой ``get`` и записью ``update``).
    """

    def __init__(
        self,
        redis: Optional[Any] = None,
        *,
        max_bars: int = DEFAULT_MAX_BARS,
        clock: Optional[Callable[[], float]] = None,
        ttl: int = DEFAULT_REDIS_TTL_S,
        mem_series: int = DEFAULT_MEM_SERIES,
    ) -> None:
        self._redis = redis
        self.max_bars = max(1, int(max_bars))
        self._ttl = int(ttl)
        self._mem_series = max(4, int(mem_series))
        self._clock = clock or time.time
        self._lock = threading.RLock()
        # key -> (CachedBars, inserted_at); OrderedDict — FIFO серий: переполнение
        # вытесняет серию, к которой дольше всех не обращались.
        self._mem: "OrderedDict[str, CachedBars]" = OrderedDict()
        self.stats: dict[str, int] = {
            "get_hit_mem": 0, "get_hit_redis": 0, "get_miss": 0,
            "update_added": 0, "update_unchanged": 0, "evicted_bars": 0,
            "evict_calls": 0, "redis_write": 0,
        }

    # ------------------------------------------------------------------ #
    #  Чтение
    # ------------------------------------------------------------------ #
    def get(self, ticker: str, tf: str, *, provider: str = PROVIDER_YFINANCE) -> Optional[CachedBars]:
        """Последние ``max_bars`` баров серии (память → Redis). ``None`` — серии нет."""
        key = bars_key(ticker, tf, provider=provider)
        with self._lock:
            entry = self._mem.get(key)
            if entry is not None:
                self._mem.move_to_end(key)
                self.stats["get_hit_mem"] += 1
                return entry

            entry = self._redis_read(key)
            if entry is not None:
                self._mem_set(key, entry)
                self.stats["get_hit_redis"] += 1
                return entry

        self.stats["get_miss"] += 1
        return None

    def last_bar_ts(self, ticker: str, tf: str, *, provider: str = PROVIDER_YFINANCE) -> Optional[float]:
        entry = self.get(ticker, tf, provider=provider)
        return entry.last_ts if entry is not None else None

    def is_due(
        self,
        ticker: str,
        tf: str,
        *,
        provider: str = PROVIDER_YFINANCE,
        sessions: Optional[tuple[str, ...]] = None,
        now: Optional[float] = None,
        slack: int = BAR_CLOSE_SLACK_S,
    ) -> bool:
        """Пора ли спрашивать провайдера о новых барах.

        ``True``, когда серии нет или прошло расчётное время закрытия следующего бара
        (последний бар + длина таймфрейма, с учётом сессий). Вне сессий и до закрытия
        следующего бара проверок не происходит — в этом и состоит «обновляем только
        когда закрылись новые бары».
        """
        entry = self.get(ticker, tf, provider=provider)
        if entry is None:
            return True
        moment = self._clock() if now is None else now
        # Базой расписания служит более позднее из: последний бар / последняя проверка.
        # После «пустой» проверки (новых баров нет) это переносит следующий запрос
        # на момент закрытия следующего бара, а не в прошлое.
        baseline = max(entry.last_ts, entry.checked_at)
        return moment >= next_close_after(tf, baseline, sessions) + slack

    # ------------------------------------------------------------------ #
    #  Запись
    # ------------------------------------------------------------------ #
    def update(
        self,
        ticker: str,
        tf: str,
        bars: list[dict],
        *,
        provider: str = PROVIDER_YFINANCE,
        checked_at: Optional[float] = None,
        source: str = "",
    ) -> tuple[CachedBars, int]:
        """Дописать **новые** бары к серии и удержать только хвост из ``max_bars`` (FIFO).

        Returns
        -------
        (entry, added)
            Серия после слияния и число добавленных баров. При ``added == 0`` бары в Redis
            **не переписываются** — только ``checked_at`` в памяти процесса. В Redis бары
            попадают исключительно когда закрылись новые (это инвариант всего модуля).
        """
        key = bars_key(ticker, tf, provider=provider)
        now = self._clock()
        with self._lock:
            existing = self._mem.get(key)
            if existing is None:
                existing = self._redis_read(key)

            last_ts = existing.last_ts if existing is not None else 0.0
            # Только бары строго новее последнего сохранённого; дедупликация по времени.
            seen: set[float] = set()
            fresh: list[dict] = []
            for bar in bars or []:
                ts = _bar_ts(bar)
                if ts > last_ts and ts not in seen:
                    seen.add(ts)
                    fresh.append(bar)
            fresh.sort(key=_bar_ts)

            merged = (existing.bars if existing is not None else []) + fresh
            if len(merged) > self.max_bars:
                self.stats["evicted_bars"] += len(merged) - self.max_bars
                merged = merged[-self.max_bars:]  # FIFO: старейшие из головы

            entry = CachedBars(
                merged,
                stored_at=now,
                checked_at=now if checked_at is None else checked_at,
                source=source or (existing.source if existing is not None else ""),
            )

            if fresh:
                self._mem_set(key, entry)
                self._redis_write(key, entry)
                self.stats["update_added"] += len(fresh)
            else:
                # Бары не изменились: сдвигаем только факт проверки (память процесса).
                self._mem_set(key, entry)
                self.stats["update_unchanged"] += 1
            return entry, len(fresh)

    def mark_checked(
        self,
        ticker: str,
        tf: str,
        *,
        provider: str = PROVIDER_YFINANCE,
        checked_at: Optional[float] = None,
    ) -> None:
        """Отметить неудачную/холостую проверку, не трогая бары и Redis.

        Нужно, когда провайдер не ответил: без этого ``is_due`` останется ``True`` и
        следующий запрос снова полезет к провайдеру немедленно.
        """
        key = bars_key(ticker, tf, provider=provider)
        with self._lock:
            existing = self._mem.get(key)
            if existing is None:
                existing = self._redis_read(key)
            if existing is None:
                return
            self._mem_set(
                key,
                CachedBars(
                    existing.bars,
                    stored_at=existing.stored_at,
                    checked_at=self._clock() if checked_at is None else checked_at,
                    source=existing.source,
                ),
            )

    def evict_oldest(
        self,
        ticker: str,
        tf: str,
        *,
        provider: str = PROVIDER_YFINANCE,
        count: int = 1,
    ) -> int:
        """Вытеснить ``count`` самых старых баров серии (голова FIFO), записав обе яруса.

        Явный механизм FIFO-вытеснения для админки/диагностики; в штатном режиме то же
        самое делает :meth:`update` при переполнении лимита.
        """
        key = bars_key(ticker, tf, provider=provider)
        with self._lock:
            existing = self._mem.get(key)
            if existing is None:
                existing = self._redis_read(key)
            if existing is None or not existing.bars:
                return 0
            n = min(max(1, int(count)), len(existing.bars))
            entry = CachedBars(
                existing.bars[n:],
                stored_at=self._clock(),
                checked_at=existing.checked_at,
                source=existing.source,
            )
            self._mem_set(key, entry)
            self._redis_write(key, entry)
            self.stats["evicted_bars"] += n
            self.stats["evict_calls"] += 1
            return n

    def drop(self, ticker: str, tf: str, *, provider: str = PROVIDER_YFINANCE) -> None:
        """Забыть серию целиком (ручной сброс)."""
        key = bars_key(ticker, tf, provider=provider)
        with self._lock:
            self._mem.pop(key, None)
            self._redis_delete(key)

    def clear(self) -> None:
        """Полный сброс (тесты)."""
        with self._lock:
            self._mem.clear()

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _mem_set(self, key: str, entry: CachedBars) -> None:
        self._mem[key] = entry
        self._mem.move_to_end(key)
        while len(self._mem) > self._mem_series:
            self._mem.popitem(last=False)  # FIFO серий: вытесняем самую старую

    def _redis_read(self, key: str) -> Optional[CachedBars]:
        redis = self._redis
        if redis is None or not getattr(redis, "connected", True):
            return None
        try:
            raw = redis.get(key)
        except Exception as exc:  # noqa: BLE001 — Redis необязателен
            logger.debug("BarStore GET %s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (ValueError, UnicodeDecodeError, AttributeError) as exc:
            logger.warning("BarStore %s: не JSON (%s) — игнорирую", key, exc)
            return None
        entry = CachedBars.from_payload(payload)
        if entry is None:
            logger.warning("BarStore %s: неизвестная схема — считаю серию пустой", key)
        return entry

    def _redis_write(self, key: str, entry: CachedBars) -> None:
        redis = self._redis
        if redis is None or not getattr(redis, "connected", True):
            return
        try:
            redis.set(key, json.dumps(entry.as_payload(), separators=(",", ":")), ex=self._ttl)
            self.stats["redis_write"] += 1
        except Exception as exc:  # noqa: BLE001 — запись кэша не критична
            logger.warning("BarStore SET %s: %s", key, exc)

    def _redis_delete(self, key: str) -> None:
        redis = self._redis
        if redis is None or not getattr(redis, "connected", True):
            return
        try:
            redis.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("BarStore DEL %s: %s", key, exc)
