"""Auto Signal Scanner — автоматический сканер по фиксированному списку тикеров.

Страница-клон сигнального сканера, но с автоматическим анализом ВСЕХ тикеров
из CSV-файла на таймфреймах 4H и 1D. Результаты накапливаются: только НОВЫЕ
сигналы (свежее предыдущего сканирования) добавляются в таблицу.

Глубина анализа: 500 баров (для корректных EMA). Глубина свежести сигнала:
3 торговых дня или 18 баров для 4H (6 баров/день × 3 дня).

Rate-limiting: yfinance 4 запроса/сек, запросы размазываются во времени
(~0.3с между тикерами). Сканирование прогрессивное: по N тикеров за вызов.

Торговые дни: если сегодня суббота — сигнал четверга считается свежим.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from gex.application.signal_service import SignalService, position_to_dict, with_entry_context
from gex.adapters.ratelimit.rate_limiter import get_rate_limiter, RateLimiter
from gex.adapters.cache.redis_client import RedisClient, cache_key, serialize_value, deserialize_value
from gex.adapters.providers.catalog import (
    SIGNAL_TIMEFRAMES,
    CatalogError,
    load_universe_file,
    universe_path,
)

logger = logging.getLogger(__name__)

# ── Константы ─────────────────────────────────────────────────────────
# Пути к вселенным берутся из каталога: раньше здесь были четыре независимых
# `Path(__file__).parent / "..."`, и те же файлы читались вторым разбором в breadth.
TICKERS_FILE = universe_path("us_options")
#: Российские акции (MOEX) — отдельный универсум сканера.
RU_TICKERS_FILE = universe_path("ru")
#: Криптовалюты (Bybit spot kline + yfinance fallback) — универсум сканера.
CRYPTO_TICKERS_FILE = universe_path("crypto")
#: Валюты и металлы (DXY, EUR/USD, USD/CNY, USD/JPY, GOLD, SILVER) — универсум сканера.
FX_TICKERS_FILE = universe_path("fx")
#: Секторальные ETF США (12 шт + RSP) — универсум сканера; тот же набор, что
#: у страницы «Композит секторов» (``sector_service.SECTOR_TICKERS``).
SECTOR_TICKERS_FILE = universe_path("sectors")
TIMEFRAMES: tuple[str, ...] = SIGNAL_TIMEFRAMES  # «медленные» ТФ (см. каталог)
BARS = 500  # глубина истории для корректных EMA
TRADING_DAY_LOOKBACK = 3  # торговых дня
BAR_LOOKBACK_4H = 18  # 6 баров/день × 3 дня
BARS_LOOKBACK_1D = 3   # 1 бар/день × 3 дня

# Rate limiting
FETCH_DELAY = 0.3  # сек между запросами (yfinance 4/сек → 0.25, запас 0.3)
BATCH_SIZE_DEFAULT = 10  # тикеров за один вызов run/next

# Фоновый интервал (должен быть ≥ времени сканирования всех)
# 113 тикеров × 2 ТФ × 0.3с ≈ 68 сек → 15 минут запас
POLL_INTERVAL_SECONDS = 900  # 15 минут

# Redis TTL
REDIS_SCAN_TTL = 3600  # 1 час для результатов сканирования
REDIS_OHLCV_TTL = 3600  # 1 час для OHLCV (авто-сканер)


@dataclass
class AutoScanInstrument:
    """Один инструмент с накопленными сигналами."""
    ticker: str
    timeframe: str
    signals: list[dict] = field(default_factory=list)  # накопленные сигналы
    last_scan: Optional[datetime] = None
    last_signal_ts: Optional[datetime] = None  # время последнего сигнала для дедупа
    error: Optional[str] = None
    #: Сырые метрики режима (тренд/флэт на 200 барах), СЛАЙДЕР-НЕЗАВИСИМЫЕ.
    #: Вердикт (FLAT/UP/DOWN) считается на чтении по личному слайдеру
    #: пользователя — пересканирование не требуется (см. gex/trend_regime.py).
    regime: Optional[dict] = None
    #: Текущая позиция по машине состояний стратегии
    #: (``{"side": flat|long|short, "avg_price": float|None, "since": iso|None}``).
    position: Optional[dict] = None
    #: Компактный снапшот истории (см. ``SignalService._build_snapshot``) —
    #: для переигровки машины позиции на чтении с персональным трейлингом.
    #: В Redis не сериализуется; восстанавливается первым сканом после старта.
    snapshot: Optional[dict] = None

    @property
    def key(self) -> str:
        return f"{self.ticker.upper()}:{self.timeframe}"


@dataclass
class AutoScannerState:
    """Состояние сканера."""
    scanned_at: Optional[datetime] = None
    instruments: dict[str, AutoScanInstrument] = field(default_factory=dict)
    running: bool = False
    total_tickers: int = 0
    scanned_count: int = 0
    total_fetches: int = 0
    completed_fetches: int = 0
    last_error: Optional[str] = None


class AutoScannerService:
    """Автоматический сканер сигналов по фиксированному списку тикеров.

    Тредобезопасность: threading.Lock вокруг состояния.
    Сигналы НАКАПЛИВАЮТСЯ — старые не удаляются при новом сканировании.
    Добавляются только сигналы, которых не было в предыдущем прогоне.

    Один инстанс = один универсум (US-акции или российские акции MOEX):
    собственный CSV, собственное состояние и Redis-ключ. Для MOEX-акций
    OHLCV берётся ТОЛЬКО через :class:`gex.moex_candles_fetcher.MOEXCandlesFetcher`
    (ISS Московской биржи, board TQBR) — SignalService сам определяет тип
    актива по ``_MOEX_OHLCV_ASSETS``; GEX-контекст пропускается (опционов на
    акции в этом сценарии нет — ``skip_gex=True`` экономит запрос).
    """

    def __init__(
        self,
        signal_service: SignalService,
        redis_client: Optional[RedisClient] = None,
        tickers_file: Path | str = TICKERS_FILE,
        universe: str = "us",
        skip_gex: bool = False,
    ):
        self._signal_service = signal_service
        self._redis = redis_client
        self._tickers_file = Path(tickers_file)
        self._universe = universe
        self._skip_gex = skip_gex
        self._state = AutoScannerState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._rate_limiter: RateLimiter = get_rate_limiter()
        # Названия компаний (для UI): ticker → human-readable name.
        self._names: dict[str, str] = {}
        # Переигровка машины позиции на чтении (персональный трейлинг):
        # кэш ((pct, reverse, last_scan) → результат) + свой лёгкий лок.
        self._replay_lock = threading.Lock()
        self._replay_cache: dict[str, tuple] = {}

        # Загружаем тикеры при инициализации
        self._load_tickers()
        # Пробуем восстановить сигналы из Redis
        self._load_from_redis()

    # ── Загрузка тикеров из CSV ───────────────────────────────────────

    def _load_tickers(self) -> None:
        """Загрузить список тикеров через каталог вселенных.

        Раньше здесь был свой `csv.reader` с `row[0]`: он молча пропускал файл
        с другим заголовком (у части вселенных `ticker,name`, у основной — `Тикер`),
        а дубликаты тикеров попадали в состояние по нескольку раз.
        """
        try:
            universe = load_universe_file(self._tickers_file)
        except CatalogError as exc:
            logger.error("AutoScanner[%s]: не удалось загрузить вселенную %s: %s",
                         self._universe, self._tickers_file, exc)
            return
        tickers = list(universe.tickers)
        names = dict(universe.names)

        with self._lock:
            self._names = names
            self._state.total_tickers = len(tickers)
            self._state.total_fetches = len(tickers) * len(TIMEFRAMES)
            # Инициализируем инструменты если ещё нет
            for ticker in tickers:
                for tf in TIMEFRAMES:
                    instr = AutoScanInstrument(ticker=ticker, timeframe=tf)
                    key = instr.key
                    if key not in self._state.instruments:
                        self._state.instruments[key] = instr

        logger.info("AutoScanner[%s]: загружено %d тикеров → %d инструментов (×2 ТФ)",
                     self._universe, len(tickers), len(tickers) * len(TIMEFRAMES))

    def get_tickers(self) -> list[str]:
        """Вернуть список тикеров."""
        with self._lock:
            seen: set[str] = set()
            tickers: list[str] = []
            for instr in self._state.instruments.values():
                t = instr.ticker.upper()
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
            return sorted(tickers)

    def get_names(self) -> dict[str, str]:
        """Названия компаний: {ticker: name} (для UI)."""
        with self._lock:
            return dict(self._names)

    @property
    def universe(self) -> str:
        """Универсум сканера: ``us``, ``ru``, ``crypto`` или ``fx``."""
        return self._universe

    # ── Статус ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Текущий статус сканера."""
        with self._lock:
            s = self._state
            instruments_with_signals = sum(
                1 for i in s.instruments.values() if i.signals
            )
            instruments_with_errors = sum(
                1 for i in s.instruments.values() if i.error
            )
            return {
                "running": s.running,
                "total_tickers": s.total_tickers,
                "total_instruments": len(s.instruments),
                "scanned_count": s.scanned_count,
                "total_fetches": s.total_fetches,
                "completed_fetches": s.completed_fetches,
                "instruments_with_signals": instruments_with_signals,
                "instruments_with_errors": instruments_with_errors,
                "scanned_at": s.scanned_at.isoformat() if s.scanned_at else None,
                "last_error": s.last_error,
            }

    # ── Получить накопленные сигналы ─────────────────────────────────

    def get_signals(
        self,
        ticker: Optional[str] = None,
        timeframe: Optional[str] = None,
        trailing_pct: Optional[float] = None,
        reverse: Optional[bool] = None,
    ) -> list[dict]:
        """Вернуть накопленные сигналы, опционально фильтруя по тикеру/ТФ.

        Если заданы ``trailing_pct``/``reverse`` (персональные настройки),
        сигналы и позиция ПЕРЕИГРЫВАЮТСЯ из снапшота истории машиной позиции
        с этими параметрами — выходы встают в места, определённые личным
        трейлинг-стопом (и принудительным разворотом). Без снапшота или без
        параметров — отдаётся накопленное фоновым сканом состояние.

        Чтение состояния — под локом; переигровка считается БЕЗ лока (тяжёлая
        операция не должна блокировать фоновые сканы). In-memory первичен.
        """
        use_replay = trailing_pct is not None or reverse is not None
        pct = float(trailing_pct or 0.0)
        rev = bool(reverse) if reverse is not None else False

        with self._lock:
            items = sorted(self._state.instruments.items())

        result: list[dict] = []
        for _key, instr in items:
            if ticker and instr.ticker.upper() != ticker.upper():
                continue
            if timeframe and instr.timeframe != timeframe:
                continue

            signals = list(instr.signals)
            position = dict(instr.position) if instr.position else None
            if use_replay:
                replayed = self._replay_instrument(instr, pct, rev)
                if replayed is not None:
                    signals = replayed["signals"]
                    position = replayed["position"]

            result.append({
                "ticker": instr.ticker,
                "timeframe": instr.timeframe,
                "signals": signals,
                "last_scan": instr.last_scan.isoformat() if instr.last_scan else None,
                "last_signal_ts": instr.last_signal_ts.isoformat() if instr.last_signal_ts else None,
                "error": instr.error,
                "regime": dict(instr.regime) if instr.regime else None,
                "position": position,
            })
        return result

    def _replay_instrument(
        self,
        instr: AutoScanInstrument,
        trailing_pct: float,
        reverse: bool,
    ) -> Optional[dict]:
        """Переиграть машину позиции из снапшота с персональными настройками.

        Возвращает ``{"signals": [dict…] в хронологическом порядке,
        "position": dict}`` или ``None`` (нет снапшота — инструмент ещё не
        сканировался после старта: вызывающий отдаёт накопленное состояние).
        Результат кэшируется до следующего скана инструмента.
        """
        snap = instr.snapshot
        if not isinstance(snap, dict) or not snap.get("arrays"):
            return None

        pct = round(float(trailing_pct or 0.0), 2)
        rev = bool(reverse)
        params = (pct, rev, instr.last_scan)
        with self._replay_lock:
            cached = self._replay_cache.get(instr.key)
        if cached is not None and cached[0] == params:
            return cached[1]

        try:
            df = pd.DataFrame(snap["arrays"])
            index = snap.get("index")
            if index is not None:
                df.index = pd.DatetimeIndex(index)
            # verify=False: снапшот не содержит колонок verification-движка.
            records, position = self._signal_service._extract_recent_signals(
                df, None, None, 5,
                trailing_pct=(pct or None), reverse=rev, verify=False,
            )
        except Exception:  # noqa: BLE001 — переигровка вспомогательная
            logger.debug("AutoScanner: replay %s не удался", instr.key, exc_info=True)
            return None

        fresh = [r for r in records if self._is_fresh_signal(r, instr.timeframe)]
        fresh = with_entry_context(records, fresh)
        out = {
            "signals": [self._signal_to_dict(r) for r in reversed(fresh)],
            "position": self._signal_service.position_to_payload(position, df.index),
        }
        with self._replay_lock:
            self._replay_cache[instr.key] = (params, out)
        return out

    def _save_to_redis(self) -> None:
        """Сохранить текущее состояние сигналов в Redis (для восстановления после рестарта)."""
        if self._redis is None or not self._redis.connected:
            return
        try:
            signals = self.get_signals()
            # auto_scan2 — второе поколение ключа: при переходе на позиционную
            # модель старые накопленные сигналы (с сиротами «выход без входа»)
            # не восстанавливаются — чистый старт (см. README решения).
            ck = cache_key("auto_scan2", self._universe, "signals")
            self._redis.set(ck, signals, ex=REDIS_SCAN_TTL)
        except Exception:
            pass

    def _load_from_redis(self) -> bool:
        """Загрузить сигналы из Redis (при старте). Возвращает True если загружено."""
        if self._redis is None or not self._redis.connected:
            return False
        try:
            ck = cache_key("auto_scan", self._universe, "signals")
            cached = self._redis.get(ck)
            if cached is None:
                return False
            result = deserialize_value(cached)
            if not isinstance(result, list):
                return False
            with self._lock:
                for item in result:
                    key = f"{item['ticker'].upper()}:{item['timeframe']}"
                    instr = self._state.instruments.get(key)
                    if instr is None:
                        instr = AutoScanInstrument(
                            ticker=item["ticker"],
                            timeframe=item["timeframe"],
                        )
                        self._state.instruments[key] = instr
                    # Восстанавливаем сигналы (не дублируем)
                    existing_ts = {s.get("timestamp") for s in instr.signals}
                    for sig in item.get("signals", []):
                        ts = sig.get("timestamp")
                        if ts and ts not in existing_ts:
                            instr.signals.append(sig)
                            existing_ts.add(ts)
                    if item.get("last_scan"):
                        try:
                            instr.last_scan = datetime.fromisoformat(item["last_scan"])
                        except (ValueError, TypeError):
                            pass
                    if isinstance(item.get("regime"), dict):
                        instr.regime = item["regime"]
                    if isinstance(item.get("position"), dict):
                        instr.position = item["position"]
                    if item.get("error"):
                        instr.error = item["error"]
            logger.info("AutoScanner[%s]: восстановлено %d инструментов из Redis",
                         self._universe, len(result))
            return True
        except Exception:
            logger.debug("AutoScanner[%s]: ошибка загрузки из Redis", self._universe, exc_info=True)
            return False

    # ── Сканирование (прогрессивное) ──────────────────────────────────

    def run_scan(self, batch_size: int = BATCH_SIZE_DEFAULT) -> dict:
        """Запустить полное сканирование всех тикеров.

        Parameters
        ----------
        batch_size : int
            Сколько тикеров обработать за этот вызов (для прогрессивного UI).

        Returns
        -------
        dict
            Статус после сканирования + число найденных новых сигналов.
        """
        all_tickers = self.get_tickers()
        if not all_tickers:
            return {**self.get_status(), "new_signals": 0}

        with self._lock:
            # Сбрасываем счётчики для нового полного прогона
            self._state.scanned_count = 0
            self._state.completed_fetches = 0
            self._state.last_error = None
            self._state.running = True

        new_total = 0

        try:
            # Определяем с каких тикеров начинать (несканированные)
            for i, ticker in enumerate(all_tickers):
                if self._stop_event.is_set():
                    break

                for tf in TIMEFRAMES:
                    if self._stop_event.is_set():
                        break
                    key = f"{ticker}:{tf}"
                    self._scan_one(ticker, tf)
                    with self._lock:
                        self._state.completed_fetches += 1

                with self._lock:
                    self._state.scanned_count += 1

                # Rate limiting: пауза между тикерами
                time.sleep(FETCH_DELAY)

        except Exception as exc:
            logger.error("AutoScanner: ошибка сканирования: %s", exc)
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            with self._lock:
                self._state.scanned_at = datetime.now(timezone.utc)
                self._state.running = False

        # Подсчитываем новые сигналы
        signals_after = self.get_signals()
        for s in signals_after:
            if s["signals"]:
                new_total += 1

        logger.info("AutoScanner: прогон завершён — %d/%d тикеров, %d с сигналами",
                     self._state.scanned_count, self._state.total_tickers, new_total)

        # Сохраняем в Redis для персистентности
        self._save_to_redis()

        return {**self.get_status(), "new_signals": new_total}

    def run_next_batch(self, batch_size: int = BATCH_SIZE_DEFAULT) -> dict:
        """Прогрессивное сканирование: следующая пачка тикеров.

        Продолжает с того места, где остановился предыдущий вызов.
        Для UI с кнопкой «Загрузить ещё» или авто-догрузкой.
        """
        all_tickers = self.get_tickers()

        with self._lock:
            current = self._state.scanned_count
            self._state.running = True

        if current >= len(all_tickers):
            with self._lock:
                self._state.running = False
            return {**self.get_status(), "new_signals": 0, "done": True}

        batch_tickers = all_tickers[current:current + batch_size]
        new_in_batch = 0

        try:
            for ticker in batch_tickers:
                if self._stop_event.is_set():
                    break
                for tf in TIMEFRAMES:
                    key = f"{ticker}:{tf}"
                    self._scan_one(ticker, tf)
                    with self._lock:
                        self._state.completed_fetches += 1
                with self._lock:
                    self._state.scanned_count += 1
                time.sleep(FETCH_DELAY)
        except Exception as exc:
            logger.error("AutoScanner: ошибка в batch: %s", exc)
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            with self._lock:
                self._state.scanned_at = datetime.now(timezone.utc)
                self._state.running = False

        # Подсчёт новых сигналов в этой пачке
        for ticker in batch_tickers:
            for tf in TIMEFRAMES:
                key = f"{ticker}:{tf}"
                with self._lock:
                    instr = self._state.instruments.get(key)
                    if instr and instr.signals:
                        new_in_batch += 1

        done = self._state.scanned_count >= len(all_tickers)
        result = {**self.get_status(), "new_signals": new_in_batch, "done": done}

        # Сохраняем в Redis после каждой пачки
        self._save_to_redis()

        return result

    # ── Сканирование одного тикер+ТФ ──────────────────────────────────

    def _scan_one(self, ticker: str, tf: str) -> None:
        """Просканировать один инструмент и обновить накопленные сигналы."""
        key = f"{ticker}:{tf}"

        # Rate limit перед запросом: RU — ISS Мосбиржи, crypto — Bybit,
        # US/FX — yfinance (валюты и металлы тоже идут через yfinance).
        rate_limit_name = {
            "ru": "moex_iss",
            "crypto": "bybit",
        }.get(self._universe, "yfinance")
        self._rate_limiter.wait(rate_limit_name)

        try:
            analysis = self._signal_service.analyze_signals(
                ticker,
                timeframe=tf,
                n_recent=5,
                bars=BARS,
                skip_gex=self._skip_gex,
                snapshot=True,
            )
            raw_signals = list(analysis.recent_signals) if analysis.recent_signals else []
        except (ValueError, RuntimeError) as exc:
            msg = str(exc)
            with self._lock:
                instr = self._state.instruments.get(key)
                if instr:
                    instr.error = msg
                    instr.last_scan = datetime.now(timezone.utc)
            return
        except Exception as exc:
            msg = f"Ошибка: {exc}"
            with self._lock:
                instr = self._state.instruments.get(key)
                if instr:
                    instr.error = msg
                    instr.last_scan = datetime.now(timezone.utc)
            return

        # Сырые метрики режима (тренд/флэт) — считаются один раз на скан,
        # вердикт по личному слайдеру пользователя — уже на чтении (роутер).
        regime_metrics = self._extract_regime(analysis)
        # Текущая позиция по машине состояний — обновляется на КАЖДОМ скане,
        # даже когда новых сигналов нет (выход из позиции мог случиться без
        # свежего сигнала в окне отображения).
        position = position_to_dict(analysis)
        # Снапшот истории — для переигровки машины на чтении (персональный
        # трейлинг-стоп): обновляем вместе с инструментом.
        snapshot = getattr(analysis, "snapshot", None)

        if not raw_signals:
            with self._lock:
                instr = self._state.instruments.get(key)
                if instr:
                    instr.error = None  # очищаем ошибку при успехе
                    instr.last_scan = datetime.now(timezone.utc)
                    if regime_metrics is not None:
                        instr.regime = regime_metrics
                    if position is not None:
                        instr.position = position
                    if snapshot is not None:
                        instr.snapshot = snapshot
            return

        # Проверка свежести: сигнал должен быть в пределах торгового окна
        fresh_signals = [
            s for s in raw_signals
            if self._is_fresh_signal(s, tf)
        ]
        # Якорь очерёдности: если самый старый из свежих — выход/добавление,
        # добавить предшествующий вход, даже если он чуть старше окна свежести
        # («выход без входа перед ним» на странице не показываем).
        fresh_signals = with_entry_context(raw_signals, fresh_signals)

        if not fresh_signals:
            with self._lock:
                instr = self._state.instruments.get(key)
                if instr:
                    instr.error = None
                    instr.last_scan = datetime.now(timezone.utc)
                    if regime_metrics is not None:
                        instr.regime = regime_metrics
                    if position is not None:
                        instr.position = position
                    if snapshot is not None:
                        instr.snapshot = snapshot
            return

        # Конвертируем сигналы в словари
        new_signals_dicts = [self._signal_to_dict(s) for s in fresh_signals]

        with self._lock:
            instr = self._state.instruments.get(key)
            if instr is None:
                instr = AutoScanInstrument(ticker=ticker, timeframe=tf)
                self._state.instruments[key] = instr

            instr.error = None
            instr.last_scan = datetime.now(timezone.utc)
            if regime_metrics is not None:
                instr.regime = regime_metrics
            if position is not None:
                instr.position = position
            if snapshot is not None:
                instr.snapshot = snapshot

            # Дедупликация: добавляем только сигналы, которых ещё нет.
            # Порядок накопления — ХРОНОЛОГИЧЕСКИЙ (старые → новые): список
            # приходит «новые первыми», поэтому разворачиваем при добавлении —
            # так UI (slice → reverse) читается сверху вниз как новые первыми.
            existing_ts = {s.get("timestamp") for s in instr.signals}
            added = 0
            for sig_dict in reversed(new_signals_dicts):
                ts = sig_dict.get("timestamp")
                if ts not in existing_ts:
                    instr.signals.append(sig_dict)
                    existing_ts.add(ts)
                    added += 1

            # Обновляем last_signal_ts
            if instr.signals:
                latest = max(
                    (s.get("timestamp") for s in instr.signals if s.get("timestamp")),
                    default=None
                )
                if latest:
                    try:
                        instr.last_signal_ts = datetime.fromisoformat(str(latest))
                    except (ValueError, TypeError):
                        pass

            if added > 0:
                logger.info("AutoScanner: %s — добавлено %d новых сигналов", key, added)

    # ── Проверка свежести сигнала (торговые дни) ─────────────────────

    @staticmethod
    def _is_fresh_signal(signal: Any, tf: str = "4h", now: Optional[datetime] = None) -> bool:
        """Проверить что сигнал попадает в окно свежести (торговые дни).

        Для 4H: сигнал в пределах последних 18 баров (~3 торговых дня).
        Для 1D: сигнал в пределах последних 3 баров (торговых дней).

        Суббота → четверг считается свежим (1 торговый день назад).

        ВАЖНО: используются ТОЛЬКО naive UTC datetime во избежание
        TypeError при вычитании aware/naive.
        """
        try:
            sig_ts = getattr(signal, "timestamp", None)
            if sig_ts is None:
                return False  # без timestamp — НЕ считаем свежим (не можем проверить)

            # Нормализуем в naive UTC datetime
            if isinstance(sig_ts, datetime):
                sig_dt = sig_ts
            elif isinstance(sig_ts, str):
                sig_dt = datetime.fromisoformat(sig_ts.replace("Z", "+00:00"))
            elif hasattr(sig_ts, "isoformat"):
                sig_dt = datetime.fromisoformat(str(sig_ts).replace("Z", "+00:00"))
            else:
                return False  # неизвестный формат — пропускаем

            # Приводим к naive UTC (критично: _index_to_datetime возвращает naive,
            # а datetime.now(timezone.utc) — aware → TypeError при вычитании)
            if sig_dt.tzinfo is not None:
                sig_dt = sig_dt.replace(tzinfo=None)

            # naive UTC now (консистентно с naive sig_dt); инъекция для тестов
            if now is None:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
            if now.tzinfo is not None:
                now = now.replace(tzinfo=None)

            # Считаем торговые дни назад
            trading_days_back = AutoScannerService._count_trading_days_back(sig_dt, now)

            if tf == "1d":
                return trading_days_back <= BARS_LOOKBACK_1D
            else:
                # 4H: проверяем и по торговым дням и по барам
                hours_diff = (now - sig_dt).total_seconds() / 3600.0
                bars_ago = hours_diff / 4.0
                return bars_ago <= BAR_LOOKBACK_4H and trading_days_back <= TRADING_DAY_LOOKBACK

        except Exception:
            logger.debug("AutoScanner: ошибка проверки свежести сигнала", exc_info=True)
            return False  # при ошибке — НЕ показываем (безопасный default)

    @staticmethod
    def _count_trading_days_back(sig_dt: datetime, now: datetime) -> int:
        """Посчитать сколько торговых дней прошло между sig_dt и now.

        Торговые дни: понедельник-пятница. Суббота/воскресенье не считаются.
        """
        if sig_dt.tzinfo is not None:
            sig_dt = sig_dt.replace(tzinfo=None)
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        sig_date = sig_dt.date()
        now_date = now.date()

        trading_days = 0
        current = sig_date
        while current <= now_date:
            if current.weekday() < 5:  # Пн-Пт
                trading_days += 1
            current += timedelta(days=1)

        # Возвращаем количество торговых дней МЕЖДУ (исключая сам день сигнала)
        return max(0, trading_days - 1)

    # ── Метрики режима (тренд/флэт) ─────────────────────────

    @staticmethod
    def _extract_regime(analysis: Any) -> Optional[dict]:
        """Вытащить сырые (слайдер-независимые) метрики режима из ответа анализа.

        Возвращает ``None``, если детектор не смог посчитать режим (мало
        истории) — тогда сигналы не будут отсекаться (нечем верифицировать).
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

    # ── Конвертация сигнала в словарь ────────────────────────────────

    @staticmethod
    def _signal_to_dict(sig: Any) -> dict:
        """Конвертировать объект сигнала в словарь для JSON."""
        return {
            "action": getattr(sig, "action", None),
            "price": getattr(sig, "price", None),
            "entry_score": getattr(sig, "entry_score", None),
            "confidence_class": getattr(sig, "confidence_class", None),
            "timestamp": getattr(sig, "timestamp", None),
            "tp_price": getattr(sig, "tp_price", None),
            "sl_price": getattr(sig, "sl_price", None),
            "reason": getattr(sig, "reason", None),
            "order_type": getattr(sig, "order_type", None),
            "gex_reason": getattr(sig, "gex_reason", None),
            "gex_multiplier": getattr(sig, "gex_multiplier", None),
            "verification_score": getattr(sig, "verification_score", None),
        }

    # ── Фоновый цикл ──────────────────────────────────────────────────

    def start(self) -> None:
        """Запустить фоновый цикл сканирования."""
        if self._state.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-scanner-loop")
        self._thread.start()
        logger.info("AutoScanner: фоновый поток запущен (интервал %dс)", POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        """Остановить фоновый цикл."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        with self._lock:
            self._state.running = False
        logger.info("AutoScanner: остановлен")

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._state.running

    def _loop(self) -> None:
        """Фоновый цикл: полное сканирование всех тикеров."""
        with self._lock:
            self._state.running = True

        while not self._stop_event.is_set():
            try:
                self.run_scan()
            except Exception as exc:
                logger.error("AutoScanner: ошибка в фоновом цикле: %s", exc)
            self._stop_event.wait(POLL_INTERVAL_SECONDS)

    # ── Сброс ─────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Сбросить все накопленные сигналы и ошибки."""
        with self._lock:
            for instr in self._state.instruments.values():
                instr.signals.clear()
                instr.error = None
                instr.last_signal_ts = None
                instr.regime = None
                instr.position = None
                instr.snapshot = None
            with self._replay_lock:
                self._replay_cache.clear()
            self._state.scanned_count = 0
            self._state.completed_fetches = 0
            self._state.last_error = None
            self._state.scanned_at = None
        logger.info("AutoScanner: состояние сброшено")
        return self.get_status()
