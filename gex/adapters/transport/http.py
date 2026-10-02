"""Единый HTTP-транспорт приложения (кольцо ``adapters``).

Зачем он нужен (аудит 05: F-02, F-03, B-02; итерация 22 плана)
-------------------------------------------------------------
Внешние вызовы были разбросаны по ~30 местам, и каждый сайт сам решал четыре вопроса:

* какой библиотекой идти — ``requests``, ``httpx`` или ``urllib``;
* сколько ждать — в коде зафиксированы значения 3, 10, 15, 20, 25 и 200 секунд, то есть
  **шесть разных мнений** о допустимой задержке;
* повторять ли неудачный вызов — ретраи были в двух местах из тридцати, с разной политикой;
* как интерпретировать ответ — «500» где-то означало ``None``, где-то исключение.

Итог: лимиты провайдеров не соблюдались (общего бюджета на повторы не существовало),
таймауты расходились, а зависший источник мог держать поток без ограничения.

Что даёт модуль
---------------
* **одно** место, где задаются таймауты, ретраи и пул соединений;
* **одну** классификацию ошибок: повторяем только то, что имеет смысл повторять;
* **инъекцию** отправителя, часов и генератора случайных чисел — поэтому политика повторов
  проверяется тестами без сети и без установленного ``requests``.

Чего модуль сознательно **не** делает: не знает про провайдеров, кэш и лимиты. Лимитер
приходит снаружи (итерация 27), кэш — снаружи (итерация 26). Здесь только «дойти и получить
ответ».
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

__all__ = [
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_READ_TIMEOUT",
    "RETRYABLE_STATUSES",
    "HttpTransportError",
    "TransientHttpError",
    "PermanentHttpError",
    "RateLimitedError",
    "RetryPolicy",
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "backoff_seconds",
    "classify_exception",
    "classify_status",
    "requests_sender",
    "get_shared_transport",
    "reset_shared_transport",
]

log = logging.getLogger(__name__)

#: Соединение должно устанавливаться быстро; долгими бывают только чтение ответа.
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 20.0

#: Статусы, которые имеет смысл повторить. Всё остальное из 4xx — ошибка запроса,
#: повтор даст тот же результат и только потратит лимит провайдера.
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


# ─────────────────────────────────────────────────────────────────────────────
#  Ошибки
# ─────────────────────────────────────────────────────────────────────────────

class HttpTransportError(RuntimeError):
    """Базовая ошибка транспорта.

    Наследуется от ``RuntimeError`` потому, что вызывающий код исторически ловил именно его;
    так замена «голых» вызовов на транспорт не меняет поведение обработчиков.
    """

    retryable = False

    def __init__(self, message: str, *, url: str = "", status: Optional[int] = None):
        super().__init__(message)
        self.url = url
        self.status = status


class TransientHttpError(HttpTransportError):
    """Сбой, который может исчезнуть сам: таймаут, обрыв, 5xx."""

    retryable = True


class PermanentHttpError(HttpTransportError):
    """Сбой, который повтор не исправит: 4xx (кроме 408/425/429), некорректный JSON."""

    retryable = False


class RateLimitedError(TransientHttpError):
    """Провайдер ответил «слишком часто» (429) и, возможно, сказал когда вернуться."""

    def __init__(self, message: str, *, url: str = "", status: int = 429,
                 retry_after: Optional[float] = None):
        super().__init__(message, url=url, status=status)
        self.retry_after = retry_after


def classify_status(status: int, *, url: str = "") -> Optional[HttpTransportError]:
    """Превращает код ответа в ошибку либо ``None``, если ответ успешный."""
    if 200 <= status < 300:
        return None
    if status == 429:
        return RateLimitedError(f"HTTP 429 (rate limit) от {url or '?'}", url=url, status=status)
    if status in RETRYABLE_STATUSES:
        return TransientHttpError(f"HTTP {status} от {url or '?'}", url=url, status=status)
    return PermanentHttpError(f"HTTP {status} от {url or '?'}", status=status)


def classify_exception(exc: BaseException) -> HttpTransportError:
    """Классифицирует исключение отправителя: повторять или нет.

    ``requests.RequestException`` наследуется от ``IOError`` (== ``OSError``), поэтому сетевые
    сбои любой библиотеки попадают в transient. Всё прочее — ошибка программирования, и
    повторять её бессмысленно.
    """
    if isinstance(exc, HttpTransportError):
        return exc
    if isinstance(exc, (TimeoutError, OSError)):
        return TransientHttpError(f"сетевой сбой: {type(exc).__name__}: {exc}")
    return PermanentHttpError(f"неожиданный сбой транспорта: {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Политика повторов
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetryPolicy:
    """Сколько раз и как долго ждать перед повтором.

    ``max_attempts`` считает **попытки**, а не повторы: 3 = один вызов и до двух повторов.
    """

    max_attempts: int = 3
    base_delay: float = 0.5      # первый повтор ждёт ~0.5 с
    max_delay: float = 8.0       # потолок одного ожидания
    jitter: float = 0.2          # ±20 % — чтобы N воркеров не стучались синхронно
    # Общий бюджет на все попытки. Держится равным таймауту чтения: повтор — это шанс на успех
    # при *быстром* отказе, а не способ ждать дольше одного таймаута.
    max_elapsed: float = 20.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts должен быть >= 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter должен лежать в [0, 1]")


def backoff_seconds(attempt: int, policy: RetryPolicy, *, rng: Optional[Callable[[], float]] = None) -> float:
    """Задержка перед следующей попыткой (``attempt`` — номер **неудавшейся**, с единицы).

    Экспонента с потолком и симметричным джиттером: ``base * 2 ** (attempt - 1)``, усечённое
    ``max_delay``. Джиттер обязателен — без него все воркеры повторяют синхронно и дружно
    получают второй отказ (ровно то, что усиливает thundering herd вместо лимитера).
    """
    draw = rng or random.random
    raw = policy.base_delay * (2 ** max(0, attempt - 1))
    capped = min(raw, policy.max_delay)
    spread = capped * policy.jitter
    return max(0.0, capped - spread + 2 * spread * draw())


# ─────────────────────────────────────────────────────────────────────────────
#  Запрос и ответ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HttpRequest:
    """Описание одного вызова. Никакой сети — только данные."""

    method: str
    url: str
    params: Optional[Mapping[str, Any]] = None
    headers: Optional[Mapping[str, str]] = None
    json_body: Optional[Any] = None
    data: Optional[Any] = None
    timeout: Optional[float] = None

    def with_defaults(self, *, headers: Mapping[str, str], timeout: float) -> "HttpRequest":
        """Дополняет запрос заголовками и таймаутом транспорта (исходник не мутирует)."""
        merged = dict(headers)
        if self.headers:
            merged.update(self.headers)
        return HttpRequest(
            method=self.method,
            url=self.url,
            params=self.params,
            headers=merged,
            json_body=self.json_body,
            data=self.data,
            timeout=self.timeout if self.timeout is not None else timeout,
        )


@dataclass
class HttpResponse:
    """Ответ провайдера. Тело разбирается лениво и один раз."""

    status_code: int
    text: str = ""
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    _payload: Any = field(default=None, repr=False)
    _parsed: bool = field(default=False, repr=False)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        """Разобранный JSON. Некорректное тело — ``PermanentHttpError``: повтор не поможет."""
        if not self._parsed:
            import json

            self._parsed = True
            try:
                self._payload = json.loads(self.text) if self.text else None
            except ValueError as exc:
                raise PermanentHttpError(
                    f"некорректный JSON от {self.url or '?'}: {exc}", url=self.url
                ) from exc
        return self._payload

    def error(self) -> Optional["HttpTransportError"]:
        """Ошибка по коду ответа или ``None``, если ответ успешный.

        Метод, а не ``raise_for_status``: транспорту нужен **объект** ошибки, чтобы решить,
        повторять запрос или нет, а не немедленный выброс.
        """
        return classify_status(self.status_code, url=self.url)


#: Отправитель: получает запрос, возвращает ответ. Единственная точка, где есть сеть.
Sender = Callable[[HttpRequest], HttpResponse]


# ─────────────────────────────────────────────────────────────────────────────
#  Транспорт
# ─────────────────────────────────────────────────────────────────────────────

class HttpTransport:
    """Выполняет HTTP-запросы с таймаутом, ретраями и общим пулом соединений.

    Зависимости (``sender``, ``sleeper``, ``clock``) передаются снаружи, поэтому класс
    проверяется без сети: тест подставляет отправителя, который «падает» заданное число раз.

    Таймаут **соединения** — не здесь, а в отправителе (:func:`requests_sender`): он живёт
    на уровне пула соединений. Здесь задаётся только таймаут чтения ответа.
    """

    def __init__(
        self,
        sender: Sender,
        *,
        policy: Optional[RetryPolicy] = None,
        timeout: float = DEFAULT_READ_TIMEOUT,
        default_headers: Optional[Mapping[str, str]] = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        on_attempt: Optional[Callable[[HttpRequest, Optional[HttpTransportError]], None]] = None,
        name: str = "http",
    ):
        self._sender = sender
        self._policy = policy or RetryPolicy()
        self._timeout = timeout
        self._default_headers = dict(default_headers or {})
        self._sleeper = sleeper
        self._clock = clock
        self._on_attempt = on_attempt
        self.name = name

    @property
    def policy(self) -> RetryPolicy:
        return self._policy

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("POST", url, **kwargs)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        json_body: Optional[Any] = None,
        data: Optional[Any] = None,
        timeout: Optional[float] = None,
        expect_json: bool = False,
    ) -> HttpResponse:
        """Один запрос с политикой повторов.

        ``expect_json=True`` разбирает тело до выхода из метода, чтобы ошибка парсинга тоже
        попадала под политику повторов (битый ответ от перегруженного провайдера — обычное дело).
        """
        request = HttpRequest(
            method=method.upper(),
            url=url,
            params=params,
            headers=headers,
            json_body=json_body,
            data=data,
            timeout=timeout,
        ).with_defaults(headers=self._default_headers, timeout=self._timeout)

        started = self._clock()
        last_error: Optional[HttpTransportError] = None

        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                response = self._sender(request)
            except Exception as exc:  # noqa: BLE001 — классифицируем ниже, сеть непредсказуема
                # BaseException (KeyboardInterrupt, CancelledError) не перехватываем: это не сбой сети.
                last_error = classify_exception(exc)
            else:
                last_error = response.error()
                if last_error is None and expect_json:
                    # Разбор тела — часть обработки ответа: битый JSON от перегруженного
                    # провайдера не должен выглядеть как успешная попытка в метриках.
                    try:
                        response.json()
                    except HttpTransportError as exc:
                        last_error = exc
                if isinstance(last_error, RateLimitedError):
                    last_error.retry_after = self._retry_after(response)
                if last_error is None:
                    if self._on_attempt:
                        self._on_attempt(request, None)
                    return response

            if self._on_attempt:
                self._on_attempt(request, last_error)

            if not self._should_retry(attempt, last_error, started):
                raise last_error

            delay = self._next_delay(attempt, last_error)
            log.warning(
                "%s: %s %s — %s; повтор %d/%d через %.2f с",
                self.name, request.method, url, last_error, attempt,
                self._policy.max_attempts - 1, delay,
            )
            self._sleeper(delay)

        raise last_error or PermanentHttpError(f"транспорт не получил ответ от {url}", url=url)

    # ── внутренности ────────────────────────────────────────────────────────

    def _should_retry(self, attempt: int, error: Optional[HttpTransportError], started: float) -> bool:
        if attempt >= self._policy.max_attempts:
            return False
        if error is None or not error.retryable:
            return False
        return (self._clock() - started) < self._policy.max_elapsed

    def _next_delay(self, attempt: int, error: Optional[HttpTransportError]) -> float:
        delay_policy = backoff_seconds(attempt, self._policy)
        retry_after = getattr(error, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            # Провайдер назвал время сам — спорить с ним нельзя, иначе получим ещё один 429.
            # Потолок нужен, чтобы «приходите через час» не превращалось в часовой sleep в воркере.
            return min(float(retry_after), self._policy.max_delay)
        return delay_policy

    @staticmethod
    def _retry_after(response: Optional[HttpResponse]) -> Optional[float]:
        if response is None:
            return None
        raw = None
        for key, value in (response.headers or {}).items():
            if key.lower() == "retry-after":
                raw = value
                break
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  Отправитель на requests (единственное место импорта библиотеки)
# ─────────────────────────────────────────────────────────────────────────────

def requests_sender(
    *,
    pool_connections: int = 20,
    pool_maxsize: int = 20,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> Sender:
    """Создаёт отправителя на ``requests.Session`` с пулом соединений.

    ``requests`` импортируется здесь, а не наверху модуля: транспорт обязан импортироваться
    (и тестироваться) без установленных сетевых библиотек.

    ``max_retries=0`` на адаптере — принципиально: ретраями управляет :class:`HttpTransport`,
    иначе получим два независимых слоя повторов и суммарное число попыток nobody-knows.
    """
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=0,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})

    def _send(request: HttpRequest) -> HttpResponse:
        timeout = (connect_timeout, request.timeout or DEFAULT_READ_TIMEOUT)
        raw = session.request(
            method=request.method,
            url=request.url,
            params=request.params,
            headers=request.headers,
            json=request.json_body,
            data=request.data,
            timeout=timeout,
        )
        return HttpResponse(
            status_code=raw.status_code,
            text=raw.text,
            url=str(raw.url),
            headers=dict(raw.headers),
        )

    return _send


# ─────────────────────────────────────────────────────────────────────────────
#  Общий экземпляр процесса
# ─────────────────────────────────────────────────────────────────────────────

def transport_from_settings() -> HttpTransport:
    """Собирает транспорт из фасада конфигурации — единственного источника значений.

    Импорт ``gex.settings`` сделан внутри функции: модуль транспорта обязан импортироваться без
    pydantic-стека (так его можно проверять в изолированном окружении без зависимостей).
    """
    from gex.settings import load

    cfg = load().http
    return HttpTransport(
        requests_sender(pool_maxsize=cfg.pool_maxsize, connect_timeout=cfg.connect_timeout),
        policy=RetryPolicy(
            max_attempts=cfg.max_attempts,
            base_delay=cfg.backoff_base,
            max_delay=cfg.backoff_max,
            max_elapsed=cfg.max_elapsed,
        ),
        timeout=cfg.read_timeout,
        name="shared",
    )


_lock = threading.Lock()
_shared: Optional[HttpTransport] = None


def get_shared_transport() -> HttpTransport:
    """Ленивый общий транспорт для legacy-сайтов, куда пока не дошёл DI.

    Один экземпляр на процесс = один пул соединений и одна политика на всех. Как только модуль
    получает транспорт через конструктор, он перестаёт пользоваться этой функцией.
    """
    global _shared
    if _shared is None:
        with _lock:
            if _shared is None:
                _shared = transport_from_settings()
    return _shared


def reset_shared_transport() -> None:
    """Сбрасывает общий экземпляр (используется в тестах и при смене конфигурации)."""
    global _shared
    with _lock:
        _shared = None
