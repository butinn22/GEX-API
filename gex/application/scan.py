"""Сканер под арендой владельца: один сканер на кластер, состояние — в Redis (ring: application).

Два дефекта, которые здесь закрываются
--------------------------------------
1. **N воркеров = N сканеров.** В рабочей роли стартуют шесть фоновых циклов (сканер TA+GEX,
   4 авто-сканера по вселенным, сканер сигналов). Каждая реплика запускала свои — то есть
   провайдер получал нагрузку ×N, а Telegram — ×N одинаковых уведомлений. Пользователю это
   видно буквально: дубли сообщений. Владелец теперь один: аренда
   (:class:`gex.adapters.cache.lease.RedisLease`) берётся перед прогоном, продлевается на
   время работы, а если процесс умер — истекает, и её забирает живая реплика.
2. **Состояние жило в памяти процесса.** ``ScanService._results``/``_last_report`` — поля
   объекта, поэтому в роли ``web`` (API не запускает сканеры) чтение всегда давало «ещё не
   сканировали», хотя воркер работал. Здесь состояние публикуется в Redis, и любой процесс
   читает его оттуда: это и есть «роутер читает Redis» из плана.

Почему продление аренды не «на всякий случай»
---------------------------------------------
Полный прогон сканера идёт минутами и переживает исходный TTL. Без продления владелец
потерял бы аренду посреди работы, её забрала бы другая реплика — и два сканера пошли бы
параллельно, то есть дефект №1 вернулся бы в худшем виде (ни один из них не знает о другом).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

#: Срок аренды владельца сканера. Короче интервала прогона намеренно: продление —
#: признак жизни, поэтому умерший процесс теряет владение за секунды, а не за часы.
DEFAULT_LEASE_TTL_S = 120

#: Как часто обновлять «пульс» состояния в Redis (чтобы читатель видел живой прогресс).
DEFAULT_HEARTBEAT_S = 30.0

#: Маркер владения, когда аренды нет вовсе (нет Redis): процесс — единственный.
_NO_LEASE = "no-lease"


class LeaseLike(Protocol):
    """Аренда (реализуется :class:`gex.adapters.cache.lease.RedisLease`)."""

    def acquire(self, name: str, ttl_s: int) -> Optional[str]: ...
    def renew(self, name: str, token: Optional[str], ttl_s: int) -> bool: ...
    def release(self, name: str, token: Optional[str]) -> bool: ...


@dataclass(frozen=True)
class ScanState:
    """Опубликованное состояние сканера (то, что читают роутеры и админка)."""

    name: str
    owner: bool = False
    running: bool = False
    scanned: int = 0
    total: int = 0
    failed: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    updated_at: float = 0.0
    note: str = ""
    report: dict = field(default_factory=dict)

    def as_payload(self) -> dict:
        return {
            "name": self.name,
            "owner": self.owner,
            "running": self.running,
            "scanned": self.scanned,
            "total": self.total,
            "failed": self.failed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "note": self.note,
            "report": self.report,
            "version": 1,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> Optional["ScanState"]:
        """Разобрать состояние; ``None`` — если это не состояние или чужая версия схемы."""
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return None
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            return None
        return cls(
            name=name,
            owner=bool(payload.get("owner")),
            running=bool(payload.get("running")),
            scanned=int(payload.get("scanned") or 0),
            total=int(payload.get("total") or 0),
            failed=int(payload.get("failed") or 0),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            updated_at=float(payload.get("updated_at") or 0.0),
            note=str(payload.get("note") or ""),
            report=payload.get("report") if isinstance(payload.get("report"), dict) else {},
        )


class ScanEngine:
    """Владелец одного сканера: аренда, прогон, публикация состояния.

    Параметры ``read_state``/``write_state`` инжектируются: слой application не имеет права
    импортировать adapters (правило R3), поэтому доступ к Redis приходит функциями —
    композиционный корень подставляет ключ и сериализацию.
    """

    def __init__(
        self,
        name: str,
        lease: Optional[LeaseLike],
        *,
        read_state: Optional[Callable[[], Any]] = None,
        write_state: Optional[Callable[[dict], Any]] = None,
        lease_ttl_s: int = DEFAULT_LEASE_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.name = name
        self._lease = lease
        self._read_state = read_state
        self._write_state = write_state
        self._lease_ttl_s = max(int(lease_ttl_s), 1)
        self._clock = clock
        self._token: Optional[str] = None
        self._lost = False
        self.takeovers = 0
        self.skipped_not_owner = 0

    # ------------------------------------------------------------------ #
    #  Владение
    # ------------------------------------------------------------------ #
    @property
    def is_owner(self) -> bool:
        return self._token is not None

    def try_become_owner(self) -> bool:
        """Взять аренду, если свободна (или если она уже наша — продлить).

        Продление здесь, а не только в прогоне: между прогонами владелец обязан подтверждать
        жизнь, иначе аренда истечёт в простое и её заберёт другая реплика — а прежний процесс
        продолжит считать себя владельцем.
        """
        if self._lease is None:
            # Аренда не настроена (нет Redis): считаем себя владельцем, иначе движок
            # отказывался бы сканировать вообще. Владение помечаем явным маркером,
            # чтобы is_owner не зависел от наличия Redis.
            self._token = self._token or _NO_LEASE
            return True
        if self._token is not None and self._lease.renew(self.name, self._token, self._lease_ttl_s):
            return True
        token = self._lease.acquire(self.name, self._lease_ttl_s)
        if token is None:
            self._token = None
            return False
        if self._token is None:
            self.takeovers += 1
        self._token = token
        logger.info("Сканер %s: владение получено (реплика будет сканировать)", self.name)
        return True

    def release(self) -> None:
        """Отдать владение (остановка по требованию — чтобы реплика не ждала истечения TTL)."""
        if self._lease is not None and self._token is not None:
            self._lease.release(self.name, self._token)
        self._token = None

    # ------------------------------------------------------------------ #
    #  Прогон
    # ------------------------------------------------------------------ #
    def run_once(self, scan: Callable[[], Any]) -> Optional[Any]:
        """Выполнить прогон, если мы владелец. Иначе — ``None`` (и это не ошибка).

        Ключевое: **не-владелец не сканирует**. Это и отличает N воркеров от N сканеров.
        """
        if not self.try_become_owner():
            self.skipped_not_owner += 1
            logger.debug("Сканер %s: владеет другая реплика — пропускаю прогон", self.name)
            return None

        started = self._clock()
        self._publish(running=True, started_at=started, note="прогон начался")
        self._lost = False
        heartbeat = self._start_heartbeat()
        try:
            result = scan()
        except Exception as exc:  # noqa: BLE001 — прогон упал: состояние обязано это показать
            self._publish(
                running=False, started_at=started, finished_at=self._clock(),
                note=f"прогон упал: {exc}",
            )
            raise
        finally:
            heartbeat.set()
        self._settle_ownership()
        note = "владение потеряно: аренда не продлилась" if self._lost else ""
        self._publish(running=False, started_at=started, finished_at=self._clock(),
                      note=note, result=result)
        return result

    def _start_heartbeat(self) -> threading.Event:
        """Продлевать аренду, пока идёт прогон.

        Прогон сканера идёт минутами и переживает TTL аренды. Продлить «задним числом»
        нельзя: к моменту завершения аренда уже истекла и её мог забрать другой процесс,
        поэтому продление работает во время прогона, а не после.
        """
        stop = threading.Event()
        if self._lease is None or self._token is None or self._lease_ttl_s <= 1:
            return stop
        interval = max(self._lease_ttl_s / 3.0, 0.5)

        def beat() -> None:
            while not stop.wait(interval):
                if not self._lease.renew(self.name, self._token, self._lease_ttl_s):
                    self._lost = True
                    logger.warning(
                        "Сканер %s: аренда не продлилась — владение теряется", self.name
                    )
                    return

        threading.Thread(target=beat, daemon=True, name=f"scan-lease-{self.name}").start()
        return stop

    def _settle_ownership(self) -> None:
        """После прогона: либо подтвердить владение, либо честно его отпустить.

        Молчаливое «аренды нет, но процесс считает себя владельцем» — это и есть худший
        сценарий: два сканера работают параллельно, и ни один об этом не знает.
        """
        if self._lease is None:
            return
        if self._lost or not self._lease.renew(self.name, self._token, self._lease_ttl_s):
            self._token = None

    def loop(
        self,
        stop_event: threading.Event,
        scan: Callable[[], Any],
        *,
        interval_s: float,
        startup_delay_s: float = 0.0,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
    ) -> None:
        """Цикл: владелец сканирует по интервалу, остальные ждут и пробуют перехватить.

        Не-владелец не молчит совсем: он периодически проверяет аренду, поэтому если владелец
        умрёт, работа продолжится у него — без ручного вмешательства.
        """
        if stop_event.wait(startup_delay_s):
            return
        next_run = 0.0
        while not stop_event.is_set():
            if not self.is_owner:
                # Не владелец: пробуем перехватить, но **не сканируем**. Проверка идёт каждый
                # пульс, поэтому смерть владельца подхватывается без ручного вмешательства.
                self.try_become_owner()
                if not self.is_owner:
                    if stop_event.wait(heartbeat_s):
                        break
                    continue
            if self._clock() >= next_run:
                try:
                    self.run_once(scan)
                except Exception as exc:  # noqa: BLE001 — цикл обязан выживать
                    logger.error("Сканер %s: прогон завершился ошибкой: %s", self.name, exc)
                next_run = self._clock() + interval_s
            if stop_event.wait(heartbeat_s):
                break
        self.release()

    # ------------------------------------------------------------------ #
    #  Состояние
    # ------------------------------------------------------------------ #
    def read_state(self) -> Optional[ScanState]:
        """Прочитать состояние из Redis (работает и в процессе без сканера)."""
        if self._read_state is None:
            return None
        try:
            return ScanState.from_payload(self._read_state())
        except Exception as exc:  # noqa: BLE001 — чтение состояния не роняет запрос
            logger.debug("Состояние сканера %s недоступно: %s", self.name, exc)
            return None

    def _publish(self, *, running: bool, started_at: Optional[float] = None,
                 finished_at: Optional[float] = None, note: str = "",
                 result: Any = None) -> None:
        if self._write_state is None:
            return
        state = ScanState(
            name=self.name,
            owner=self.is_owner,
            running=running,
            started_at=started_at,
            finished_at=finished_at,
            updated_at=self._clock(),
            note=note,
            report=_report_fields(result),
            **(_counts(result) if result is not None else {}),
        )
        try:
            self._write_state(state.as_payload())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Состояние сканера %s не записано: %s", self.name, exc)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "owner": self.is_owner,
            "takeovers": self.takeovers,
            "skipped_not_owner": self.skipped_not_owner,
        }


def supervise(
    target: Any,
    engine: ScanEngine,
    stop_event: threading.Event,
    *,
    check_interval_s: float = 30.0,
) -> None:
    """Держать сервис запущенным **только у владельца аренды**.

    Почему обёртка, а не переписывание циклов: у шести сервисов свои циклы со своими
    интервалами и поведением (первый прогон, паузы между тикерами, уведомления). Менять их
    внутренности ради владения — риск сломать то, что работает; обёртка добавляет ровно одно
    свойство: сканирует только владелец, остальные реплики держат сервис остановленным.

    При потере аренды сервис останавливается: иначе прежний владелец продолжил бы сканировать
    параллельно с новым — то есть дефект «N сканеров» вернулся бы в худшем виде.
    """

    def _running() -> bool:
        state = getattr(target, "is_running", False)
        return bool(state() if callable(state) else state)

    while not stop_event.is_set():
        if engine.is_owner:
            if not _running():
                target.start()
        elif engine.try_become_owner():
            logger.info("Сервис %s: аренда получена — запускаю", engine.name)
            target.start()
        elif _running():
            logger.warning("Сервис %s: аренда потеряна — останавливаю, чтобы не сканировать дважды",
                           engine.name)
            target.stop()
        if stop_event.wait(check_interval_s):
            break

    if _running():
        target.stop()
    engine.release()


def _counts(result: Any) -> dict:
    """Числа из отчёта сканера (разные сервисы называют их по-своему)."""
    out: dict[str, int] = {}
    for src, dst in (("ok", "scanned"), ("total", "total"), ("failed", "failed"),
                     ("scanned", "scanned"), ("scanned_count", "scanned")):
        value = getattr(result, src, None)
        if isinstance(value, int):
            out.setdefault(dst, value)
    return out


def _report_fields(result: Any) -> dict:
    """Плоская сводка отчёта: что именно смотреть в админке, решает сервис."""
    if result is None:
        return {}
    for attr in ("as_dict", "to_dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception:  # noqa: BLE001
                break
    return {}


__all__ = [
    "DEFAULT_HEARTBEAT_S",
    "DEFAULT_LEASE_TTL_S",
    "LeaseLike",
    "ScanEngine",
    "ScanState",
    "supervise",
]
