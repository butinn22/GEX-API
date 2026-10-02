"""Прогноз фундаментальных метрик: WMA + линейная регрессия + аномалии.

Сервис рассчитывает будущие значения ключевых фундаментальных показателей
компании (revenue, FCF, Net Debt/EBITDA, EPS, Operating Margin, ROE, PEG)
комбинацией взвешенной скользящей средней (WMA) и линейной регрессии.

Модели
------
* **WMA** — веса убывают от текущего года к прошлым (N..1):
  ``WMA = Σ(вес·значение) / Σ(весов)``. Трендовый прирост — среднее
  приращение скользящего WMA-ряда за последние N баров. Прогноз:
  ``WMA_last + trend·h``.
* **Линейная регрессия** — трендовая прямая по истории ``y = a + b·t``
  (t = 0..n−1). Прогноз: ``a + b·(n−1+h)``.
* **Комбинированный** — ``α·WMA + (1−α)·LinReg`` (α по умолчанию 0.5).

Аномалии
--------
Скачок считается аномалией, если ``|Δ| = |v_t/v_{t-1} − 1| ≥ threshold``
(порог настраивается от 0.2 до ∞, по умолчанию 0.6) **и** отклонение выходит
за пределы исторического тренда (``|Δ − median| > 3·MAD`` по «чистым»
темпам). При сглаживании аномальное значение заменяется на
``prev·(1 + median_growth)`` и прогноз пересчитывается — сервис всегда
отдаёт оба сценария: «сглаживать» и «учитывать полностью».

Ряды с неположительными значениями (убыточные годы, отрицательный net debt)
не поддерживают темпы роста — growth/anomaly для них возвращают None,
но линейные прогнозы считаются.
"""
from __future__ import annotations

import logging
from statistics import median

from gex.adapters.cache.redis_client import RedisClient, cache_key

logger = logging.getLogger(__name__)

#: Годовые горизонты прогноза: 1/2/3 года (по ТЗ — не 5 лет)
ANNUAL_HORIZONS = (1, 2, 3)

#: Квартальные горизонты прогноза: 1/2/3 квартала (по ТЗ — не 4/12/20)
QUARTERLY_HORIZONS = (1, 2, 3)

#: Базовое влияние последнего отчёта на изменение прогноза (60%, до 100%)
DEFAULT_LAST_REPORT_WEIGHT = 0.6

#: Метрики прогноза: (ключ в annual-ряду, ключ в ответе, название, тип)
#: flow — потоковые (аномалии детектятся), ratio — доли/отношения (%-скачки
#: бессмысленны: переход через ноль даёт огромный change)
FORECAST_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("revenue", "revenue", "Выручка", "flow"),
    ("free_cash_flow", "fcf", "Free Cash Flow", "flow"),
    ("net_debt_to_ebitda", "net_debt_to_ebitda", "Net Debt / EBITDA", "ratio"),
    ("eps_basic", "eps", "EPS", "flow"),
    ("operating_margin", "operating_margin", "Operating Margin", "ratio"),
    ("roe", "roe", "ROE", "ratio"),
)


# ══════════════════════════════════════════════════════════════════════ #
#  WMA — взвешенная скользящая средняя
# ══════════════════════════════════════════════════════════════════════ #
def wma_last(values: list[float], window: int = 5) -> float | None:
    """WMA последних ``window`` значений: веса window..1 (новое — больший вес).

    ``WMA = Σ(вес·значение) / Σ(весов)``. Если значений меньше окна —
    берётся всё доступное.
    """
    if not values:
        return None
    n = min(window, len(values))
    weights = list(range(1, n + 1))
    last_n = values[-n:]
    return sum(w * v for w, v in zip(weights, last_n, strict=False)) / sum(weights)


def wma_series(values: list[float], window: int = 5) -> list[float | None]:
    """Скользящий WMA-ряд по всей истории (None до прогрева окна)."""
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(wma_last(values[: i + 1], window))
    return out


def wma_trend(values: list[float], window: int = 5) -> float | None:
    """Средний прирост скользящего WMA-ряда за последние N баров (в год).

    Если скользящий ряд не успел прогреться (точек < 2) — fallback на
    среднее приращение исходных значений (короткая история).
    """
    series = [v for v in wma_series(values, window) if v is not None]
    if len(series) >= 2:
        n = min(window, len(series))
        return (series[-1] - series[-n]) / (n - 1)
    if len(values) >= 2:
        n = min(window, len(values))
        diffs = [values[i] - values[i - 1] for i in range(len(values) - n + 1, len(values))]
        return sum(diffs) / len(diffs) if diffs else 0.0
    return None


def wma_forecast(values: list[float], horizon: int = 1, window: int = 5) -> float | None:
    """WMA-прогноз на ``horizon`` лет: ``WMA_last + trend·h``."""
    base = wma_last(values, window)
    trend = wma_trend(values, window)
    if base is None or trend is None:
        return None
    return base + trend * horizon


# ══════════════════════════════════════════════════════════════════════ #
#  Линейная регрессия
# ══════════════════════════════════════════════════════════════════════ #
def linreg(values: list[float]) -> tuple[float, float]:
    """Коэффициенты (a, b) прямой ``y = a + b·t`` по точкам t = 0..n−1."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mx = (n - 1) / 2.0
    my = sum(values) / n
    sxx = sum((i - mx) ** 2 for i in range(n))
    sxy = sum((i - mx) * (v - my) for i, v in enumerate(values))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    return a, b


def linreg_forecast(values: list[float], horizon: int = 1) -> float | None:
    """Прогноз по линии тренда: ``a + b·(n−1+h)``."""
    if not values:
        return None
    a, b = linreg(values)
    return a + b * (len(values) - 1 + horizon)


# ══════════════════════════════════════════════════════════════════════ #
#  Комбинированный прогноз
# ══════════════════════════════════════════════════════════════════════ #
def recency_blend(
    values: list[float],
    model_h: float | None,
    horizon: int,
    last_weight: float,
) -> float | None:
    """Смешать прогноз модели с траекторией ПОСЛЕДНЕГО отчёта.

    ``recency_h = last · (1 + growth_last)^h``, где ``growth_last`` — темп
    изменения последнего отчёта (last/prev − 1). Итоговый прогноз:

    ``final_h = last_weight·recency_h + (1 − last_weight)·model_h``

    ``last_weight = 0.6`` (базово) — последний отчёт задаёт 60% изменения
    прогноза; ``1.0`` — прогноз полностью повторяет рост последнего отчёта.
    Если последний отчёт не даёт темпа (prev ≤ 0, last ≤ 0, меньше 2 точек)
    — возвращается исходный прогноз модели.
    """
    if last_weight is None or last_weight <= 0 or model_h is None:
        return model_h
    if len(values) < 2:
        return model_h
    last, prev = values[-1], values[-2]
    if last <= 0 or prev <= 0:
        return model_h
    growth = last / prev - 1.0
    recency = last * (1.0 + growth) ** horizon
    return last_weight * recency + (1.0 - last_weight) * model_h


def combined_forecast(
    values: list[float],
    horizon: int = 1,
    alpha: float = 0.5,
    window: int = 5,
) -> float | None:
    """``α·WMA + (1−α)·LinReg``. Если одна модель недоступна — другая."""
    w = wma_forecast(values, horizon, window)
    lr = linreg_forecast(values, horizon)
    if w is None and lr is None:
        return None
    if w is None:
        return lr
    if lr is None:
        return w
    return alpha * w + (1 - alpha) * lr


# ══════════════════════════════════════════════════════════════════════ #
#  Темпы роста
# ══════════════════════════════════════════════════════════════════════ #
def yoy_growth(values: list[float], gap: int = 1) -> float | None:
    """Рост год к году (последний переход): ``v_last/v_prev − 1``.

    ``gap`` — разница лет между точками (не соседние годы → None).
    """
    if len(values) < 2 or values[-2] <= 0 or values[-1] <= 0 or gap != 1:
        return None
    return values[-1] / values[-2] - 1.0


def cagr(values: list[float], years: int = 3) -> float | None:
    """Среднегодовой темп роста за ``years`` лет: ``(last/first)^(1/y) − 1``.

    Требует положительности ВСЕХ значений в окне (нули/убытки ломают геометрию).
    """
    if years <= 0 or len(values) < years + 1:
        return None
    window = values[-(years + 1):]
    if any(v <= 0 for v in window):
        return None
    return (window[-1] / window[0]) ** (1.0 / years) - 1.0


def median_growth(values: list[float], gaps: list[int] | None = None) -> float | None:
    """Медиана темпов роста по переходам (только положительные, соседние годы)."""
    changes = []
    for i in range(1, len(values)):
        if gaps is not None and gaps[i - 1] != 1:
            continue
        if values[i - 1] > 0 and values[i] > 0:
            changes.append(values[i] / values[i - 1] - 1.0)
    return median(changes) if changes else None


# ══════════════════════════════════════════════════════════════════════ #
#  Аномалии
# ══════════════════════════════════════════════════════════════════════ #
def detect_anomalies(
    values: list[float], threshold: float = 0.6, gaps: list[int] | None = None
) -> list[int]:
    """Индексы аномальных переходов (t, где v_t резко отличается от v_{t-1}).

    Условия:
    1. ``|Δ| ≥ threshold`` (порог настраивается, 0.2..∞);
    2. отклонение выходит за пределы исторического тренда:
       ``|Δ − median(чистые темпы)| > 3·MAD(чистые темпы)``.
    Переходы через пропуск лет (``gaps[i] != 1``) НЕ считаются аномалиями.
    Если «чистой» истории нет (MAD = 0) — достаточно порога.
    """
    if len(values) < 2:
        return []

    changes: list[float | None] = []
    for i in range(1, len(values)):
        if gaps is not None and gaps[i - 1] != 1:
            changes.append(None)
            continue
        prev, cur = values[i - 1], values[i]
        if prev <= 0 or cur <= 0:
            changes.append(None)
        else:
            changes.append(cur / prev - 1.0)

    clean = [c for c in changes if c is not None]
    med = median(clean) if clean else 0.0
    mad = median(abs(c - med) for c in clean) if clean else 0.0

    anomalies: list[int] = []
    for i, c in enumerate(changes, start=1):
        if c is None or abs(c) < threshold:
            continue
        # Выход за пределы предыдущего тренда: скачок заметно сильнее
        # исторического темпа. Стабильный ряд из одинаково больших скачков
        # (компания живёт на +70% каждый год) — НЕ аномалия.
        if mad > 1e-12:
            if abs(c - med) > 3 * mad and abs(c) > abs(med):
                anomalies.append(i)
        elif abs(c) > abs(med):
            anomalies.append(i)
    return anomalies


def smooth_anomalies(
    values: list[float], anomaly_indexes: list[int], gaps: list[int] | None = None
) -> list[float]:
    """Сгладить аномальные значения: ``v_t = v_{t-1}·(1 + median_growth)``.

    Медианный темп берётся по «чистым» переходам (соседние годы, где ни
    текущий, ни предыдущий период не аномален). Если темп недоступен
    (prev ≤ 0 или нет чистой истории) — значение заменяется на линию
    линейного тренда.
    """
    if not anomaly_indexes:
        return list(values)

    anomaly_set = set(anomaly_indexes)
    clean_changes = []
    for i in range(1, len(values)):
        if i in anomaly_set or (i - 1) in anomaly_set:
            continue
        if gaps is not None and gaps[i - 1] != 1:
            continue
        if values[i - 1] > 0 and values[i] > 0:
            clean_changes.append(values[i] / values[i - 1] - 1.0)
    med = median(clean_changes) if clean_changes else None

    a, b = linreg(values)
    out = list(values)
    for i in anomaly_set:
        if i <= 0 or i >= len(values):
            continue
        prev = out[i - 1]
        if med is not None and prev > 0:
            out[i] = prev * (1.0 + med)
        else:
            # Fallback на трендовую линию
            out[i] = a + b * i
    return out


# ══════════════════════════════════════════════════════════════════════ #
#  Трендовый режим, ошибка прогноза, доверительные интервалы
# ══════════════════════════════════════════════════════════════════════ #
def trend_regime(
    values: list[float], gap_last: int = 1, anomaly_detected: bool = False
) -> dict:
    """Классифицировать характер движения метрики по последним периодам.

    Сравниваются последний (c2) и предыдущий (c1) темпы роста:

    * рост / падение — направление движения;
    * ускорение / замедление — рост (падение) ускоряется или тормозит;
    * разворот — направление сменилось;
    * стабилизация — темп близок к нулю;
    * аномалия — детект аномального скачка на последнем периоде.

    Returns
    -------
    dict
        ``{regime, label, direction, delta}`` — machine-ключ, RU-метка,
        направление (up/down/flat), последний темп.
    """
    if len(values) < 3 or gap_last != 1:
        return {"regime": "unknown", "label": "недостаточно данных", "direction": "flat", "delta": None}

    c1 = values[-2] / values[-3] - 1.0 if values[-3] > 0 else None
    c2 = values[-1] / values[-2] - 1.0 if values[-2] > 0 else None

    if anomaly_detected:
        return {"regime": "anomaly", "label": "аномальное отклонение", "direction": "flat", "delta": c2}
    if c1 is None or c2 is None:
        return {"regime": "unknown", "label": "недостаточно данных", "direction": "flat", "delta": c2}
    if abs(c2) < 0.02:
        return {"regime": "stabilization", "label": "стабилизация", "direction": "flat", "delta": c2}

    if c2 > 0 and c1 > 0:
        if c2 > c1 * 1.15:
            return {"regime": "acceleration", "label": "ускорение роста", "direction": "up", "delta": c2}
        if c2 < c1 * 0.85:
            return {"regime": "deceleration", "label": "замедление роста", "direction": "up", "delta": c2}
        return {"regime": "growth", "label": "рост", "direction": "up", "delta": c2}
    if c2 < 0 and c1 < 0:
        if c2 < c1 * 1.15:
            return {"regime": "acceleration", "label": "ускорение падения", "direction": "down", "delta": c2}
        if c2 > c1 * 0.85:
            return {"regime": "deceleration", "label": "замедление падения", "direction": "down", "delta": c2}
        return {"regime": "decline", "label": "падение", "direction": "down", "delta": c2}

    return {"regime": "reversal", "label": "разворот тренда", "direction": "up" if c2 > 0 else "down", "delta": c2}


def forecast_error(values: list[float], window: int = 5, alpha: float = 0.5) -> dict | None:
    """Ошибка one-step-ahead комбинированного прогноза на исторических данных.

    Для каждой точки i (с прогретого окна) считаем прогноз по ряду до i−1
    и сравниваем с фактом: ``{rmse, mae, std}`` (стандартное отклонение
    остатков — основа доверительных интервалов).
    """
    if len(values) < window + 2:
        return None
    residuals = []
    for i in range(window, len(values)):
        pred = combined_forecast(values[:i], 1, alpha=alpha, window=window)
        if pred is None:
            continue
        residuals.append(values[i] - pred)
    if len(residuals) < 2:
        return None
    n = len(residuals)
    mae = sum(abs(r) for r in residuals) / n
    rmse = (sum(r * r for r in residuals) / n) ** 0.5
    mean_r = sum(residuals) / n
    std = (sum((r - mean_r) ** 2 for r in residuals) / n) ** 0.5
    return {"rmse": rmse, "mae": mae, "std": std}


def confidence_interval(point: float, std: float, horizon: int = 1, z: float = 1.28) -> dict:
    """Доверительный интервал точки прогноза: ``point ± z·std·√h`` (z=1.28 → 80%)."""
    margin = z * std * (horizon ** 0.5)
    return {"lower": point - margin, "upper": point + margin}


def scenario_forecast(point: float, std: float, horizon: int = 1) -> dict:
    """Сценарии будущего значения: пессимистичный/базовый/оптимистичный (±1σ·√h)."""
    margin = std * (horizon ** 0.5)
    return {
        "pessimistic": point - margin,
        "base": point,
        "optimistic": point + margin,
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Сборка прогноза одной метрики (чистая функция)
# ══════════════════════════════════════════════════════════════════════ #
def build_metric_forecast(
    history: list[dict],
    *,
    horizon_years: int = 5,
    window: int = 5,
    alpha: float = 0.5,
    anomaly_threshold: float = 0.6,
    anomaly_enabled: bool = True,
    last_weight: float = DEFAULT_LAST_REPORT_WEIGHT,
) -> dict:
    """Полный прогноз метрики: рост, WMA/LinReg/combined, аномалии, сценарии.

    Parameters
    ----------
    history : list[dict]
        ``[{end, fy, value}, ...]`` по возрастанию времени (годовые точки).
    anomaly_enabled : bool
        False для ratio-метрик (margin/ROE/NetDebt-EBITDA): процентные скачки
        долей бессмысленны, сценарии не строятся.
    last_weight : float
        Влияние последнего отчёта на изменение прогноза (0.6..1.0): смесь
        траектории последнего отчёта и прогноза модели в combined.

    Returns
    -------
    dict
        ``{history, growth, forecast, anomaly, scenarios}`` — см. схему
        ``MetricForecast``. Значения аномалий НЕ удаляются из history —
        сценарии показывают оба режима учёта.
    """
    values = [float(p["value"]) for p in history]
    if not values:
        return {
            "history": [], "growth": None, "regime": trend_regime(values),
            "forecast": None, "anomaly": None, "scenarios": None,
        }

    # Пропуски лет (переход через дыру — не «скачок», а отсутствие данных)
    gaps = [
        history[i]["end"].year - history[i - 1]["end"].year
        for i in range(1, len(history))
    ]
    continuous = all(g == 1 for g in gaps)

    positive = all(v > 0 for v in values)
    growth = None
    if positive:
        growth = {
            "yoy": yoy_growth(values, gaps[-1] if gaps else 1),
            "cagr_3y": cagr(values, 3) if continuous else None,
            "cagr_5y": cagr(values, 5) if continuous else None,
        }

    anomalies = (
        detect_anomalies(values, anomaly_threshold, gaps)
        if anomaly_enabled and positive and len(values) >= 3
        else []
    )
    anomaly = None
    if anomalies:
        idx = anomalies[-1]  # показываем последнюю (самую свежую) аномалию
        change = values[idx] / values[idx - 1] - 1.0 if values[idx - 1] > 0 else None
        anomaly = {
            "detected": True,
            "period": history[idx]["end"],
            "change": change,
            "note": (
                f"Резкое изменение показателя ({change:+.0%} за период до "
                f"{history[idx]['end']}), выходящее за пределы исторического тренда. "
                "Выберите режим учёта: сглаживать или учитывать полностью."
            ),
        }

    # Характер движения (для всех метрик; аномалия — только flow-детект на последнем периоде)
    anomaly_at_last = bool(anomalies) and anomalies[-1] == len(values) - 1
    regime = trend_regime(values, gaps[-1] if gaps else 1, anomaly_at_last)

    def _horizons(vals: list[float]) -> dict[str, float | None]:
        return {
            "h1": recency_blend(vals, combined_forecast(vals, 1, alpha, window), 1, last_weight),
            "h2": recency_blend(vals, combined_forecast(vals, 2, alpha, window), 2, last_weight),
            "h3": recency_blend(vals, combined_forecast(vals, 3, alpha, window), 3, last_weight),
        }

    forecast = None
    if len(values) >= 2:
        err = forecast_error(values, window, alpha)
        interval = None
        scenarios = None
        if err and err["std"] > 0:
            interval = {}
            scenarios = {}
            for h in ANNUAL_HORIZONS:
                key = f"h{h}"
                point = recency_blend(values, combined_forecast(values, h, alpha, window), h, last_weight)
                if point is None:
                    continue
                interval[key] = confidence_interval(point, err["std"], h)
                scenarios[key] = scenario_forecast(point, err["std"], h)
        forecast = {
            "wma": {f"h{h}": wma_forecast(values, h, window) for h in ANNUAL_HORIZONS},
            "linreg": {f"h{h}": linreg_forecast(values, h) for h in ANNUAL_HORIZONS},
            "combined": _horizons(values),
            "alpha": alpha,
            "error": err,
            "interval": interval,
            "scenarios": scenarios,
        }

    scenarios = None
    if anomalies:
        smoothed = smooth_anomalies(values, anomalies, gaps)
        scenarios = {
            "smooth": _horizons(smoothed),
            "keep": _horizons(values),
        }

    return {
        "history": history,
        "growth": growth,
        "regime": regime,
        "forecast": forecast,
        "anomaly": anomaly,
        "scenarios": scenarios,
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Калькулятор оценки: Price ↔ P/E ↔ Прибыль / EPS
# ══════════════════════════════════════════════════════════════════════ #
# ══════════════════════════════════════════════════════════════════════ #
#  Калькулятор оценки: Цена ↔ P/E ↔ Прибыль / EPS
# ══════════════════════════════════════════════════════════════════════ #
#: Режимы калькулятора (треугольник сценариев): что задаём → что находим
VALUATION_MODES = (
    "auto",                    # совместимость: любые 2 из price/pe/eps
    "price_pe_to_earnings",    # задаём цену и P/E → требуемая прибыль
    "earnings_pe_to_price",    # задаём прибыль и P/E → справедливая цена
    "price_earnings_to_pe",    # задаём цену и прибыль → подразумеваемый P/E
)


def calculate_valuation(req: dict) -> dict:
    """Сценарный калькулятор: достраивает недостающие показатели.

    Вход: комбинации из ``price``, ``pe``, ``eps``, ``earnings``, ``shares``,
    ``growth`` (десятичная доля, 0.15 = 15%) и ``mode``.

    Основная связь: ``P = (Прибыль / S) × P/E`` — три переменные связаны
    одной формулой, независимо менять их нельзя. Режимы (треугольник
    сценариев):

    * ``price_pe_to_earnings`` — задаём цену и P/E → требуемая прибыль;
    * ``earnings_pe_to_price`` — задаём прибыль и P/E → справедливая цена;
    * ``price_earnings_to_pe`` — задаём цену и прибыль → подразумеваемый P/E;
      (в явных режимах ``shares`` обязателен);
    * ``auto`` — любые два из price/pe/eps (обратная совместимость).

    Дополнительно: earnings = eps·shares, market cap = price·shares,
    PEG = P/E ÷ (growth·100), при growth ≤ 0 — предупреждение.
    """
    price, pe, eps = req.get("price"), req.get("pe"), req.get("eps")
    earnings, shares = req.get("earnings"), req.get("shares")
    growth = req.get("growth")
    mode = req.get("mode", "auto")
    solved: str | None = None

    # 1. Явные режимы (треугольник сценариев)
    if mode == "price_pe_to_earnings":
        eps = price / pe
        if shares:
            earnings = price * shares / pe
        solved = "earnings"
    elif mode == "earnings_pe_to_price":
        eps = earnings / shares if shares else None
        if eps:
            price = pe * eps
        solved = "price"
    elif mode == "price_earnings_to_pe":
        eps = earnings / shares if shares else None
        if eps:
            pe = price / eps
        solved = "pe"
    else:
        # 2. auto: price ↔ pe ↔ eps (нужно минимум два из трёх — проверено валидатором)
        if price is None and pe is not None and eps is not None:
            price = pe * eps
        elif pe is None and price is not None and eps is not None:
            pe = price / eps
        elif eps is None and price is not None and pe is not None:
            eps = price / pe

    # 3. earnings ↔ eps ↔ shares
    if earnings is None and eps is not None and shares is not None:
        earnings = eps * shares
    elif shares is None and eps is not None and earnings is not None:
        shares = earnings / eps
    elif eps is None and earnings is not None and shares is not None:
        eps = earnings / shares

    market_cap = price * shares if price is not None and shares is not None else None

    # 4. PEG
    peg = None
    peg_warning = None
    if growth is not None and pe is not None:
        if growth > 0:
            peg = pe / (growth * 100)
        else:
            peg_warning = "Рост прибыли ≤ 0 — PEG не определён"

    return {
        "mode": mode,
        "solved": solved,
        "price": price,
        "pe": pe,
        "eps": eps,
        "earnings": earnings,
        "shares": shares,
        "market_cap": market_cap,
        "growth": growth,
        "peg": peg,
        "peg_warning": peg_warning,
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Квартальный прогноз (переключатель «Год / Квартал»)
# ══════════════════════════════════════════════════════════════════════ #
def build_quarterly_forecast(
    history: list[dict],
    *,
    window: int = 5,
    alpha: float = 0.5,
    last_weight: float = DEFAULT_LAST_REPORT_WEIGHT,
) -> dict:
    """Прогноз по квартальному ряду (3M-строки): горизонты q1/q2/q3.

    ``last_weight`` — влияние последнего отчёта на изменение прогноза
    (0.6..1.0), применяется к combined-прогнозу через :func:`recency_blend`.

    Returns
    -------
    dict
        ``{history, forecast}`` — ``forecast = {wma, linreg, combined,
        interval, error}`` с ключами q1/q2/q3 (кварталы).
    """
    values = [float(p["value"]) for p in history]
    if not values:
        return {"history": [], "forecast": None}

    def _h(vals: list[float], h: int) -> float | None:
        return recency_blend(vals, combined_forecast(vals, h, alpha, window), h, last_weight)

    err = forecast_error(values, window, alpha)
    interval = None
    if err and err["std"] > 0:
        interval = {}
        for h in QUARTERLY_HORIZONS:
            point = _h(values, h)
            if point is None:
                continue
            interval[f"q{h}"] = confidence_interval(point, err["std"], h)

    forecast = None
    if len(values) >= 2:
        forecast = {
            "wma": {f"q{h}": wma_forecast(values, h, window) for h in QUARTERLY_HORIZONS},
            "linreg": {f"q{h}": linreg_forecast(values, h) for h in QUARTERLY_HORIZONS},
            "combined": {f"q{h}": _h(values, h) for h in QUARTERLY_HORIZONS},
            "interval": interval,
            "error": err,
            "alpha": alpha,
        }

    return {"history": history, "forecast": forecast}


# ══════════════════════════════════════════════════════════════════════ #
#  Прогноз цены акции: EPS-прогноз × P/E × мультипликатор качества
# ══════════════════════════════════════════════════════════════════════ #
def _quality_multiplier(metrics: dict, horizon: str, period: str = "annual") -> float:
    """Мультипликатор качества по тенденциям Revenue, EPS и FCF.

    Растущие выручка, прибыль на акцию и свободный денежный поток повышают
    оправданный P/E. По ТЗ влияние оценок на прогноз СНИЖЕНО: степени
    уменьшены вдвое (0.15/0.10/0.15 вместо 0.3/0.2/0.3) и диапазон
    сужен с 0.5..1.5 до 0.85..1.15.

    ``period='quarterly'`` — тот же расчёт по квартальным рядам метрик
    (``metrics[k]['quarterly']``) и квартальным горизонтам q1/q2/q3.
    """
    q = 1.0

    def _ratio(metric_key: str) -> float | None:
        m = metrics.get(metric_key)
        if not m:
            return None
        src = (m.get("quarterly") or {}) if period == "quarterly" else m
        if not src.get("forecast") or not src["forecast"].get("combined"):
            return None
        fc = src["forecast"]["combined"].get(horizon)
        hist = src.get("history") or []
        now = hist[-1]["value"] if hist else None
        if fc is None or not now or now <= 0:
            return None
        return fc / now

    revenue = _ratio("revenue")
    eps = _ratio("eps")
    fcf = _ratio("fcf")
    if revenue and revenue > 0:
        q *= revenue ** 0.15
    if eps and eps > 0:
        q *= eps ** 0.10
    if fcf and fcf > 0:
        q *= fcf ** 0.15
    return max(0.85, min(1.15, q))


def build_price_forecast(
    price: float,
    metrics: dict,
    eps_last: float,
    price_growth: float | None = None,
) -> dict:
    """Прогноз цены: EPS-прогноз × P/E × quality + поправка на 5-летний рост цены.

    Формула (смесь 50/50):

    ``price_h = 0.5 · max(0, EPS_h · P/E) · quality_h + 0.5 · price · (1 + g)^h``

    где ``g`` — среднегодовой рост цены акции за 5 лет (CAGR, поправочный
    коэффициент из исторических цен). Если EPS-прогноз отрицателен или
    недоступен — цена идёт по историческому росту. Итог ВСЕГДА ≥ 0
    (отрицательная цена невозможна).

    Parameters
    ----------
    price_growth : float | None
        CAGR цены за 5 лет (0.12 = +12%/год). None — без поправки.
    """
    eps = metrics.get("eps") or {}
    fc = (eps.get("forecast") or {}).get("combined") or {}
    if not price or price <= 0:
        return {
            "h1": None, "h2": None, "h3": None,
            "pe_base": None, "quality": None,
            "price_growth": price_growth, "method": None,
        }
    pe_base = price / eps_last if eps_last and eps_last > 0 else None
    quality: dict[str, float | None] = {}
    out: dict[str, float | None] = {}
    for key, years in (("h1", 1), ("h2", 2), ("h3", 3)):
        eps_h = fc.get(key)
        growth_factor = (1.0 + price_growth) ** years if price_growth is not None else None
        # Фундаментальная цель: EPS-прогноз обязан быть положительным
        fundamental = eps_h * pe_base if (pe_base and eps_h is not None and eps_h > 0) else None
        if fundamental is not None:
            q = _quality_multiplier(metrics, key)
            quality[key] = q
            if growth_factor is not None:
                # Фундаментальная цель + поправка на исторический рост цены
                out[key] = max(0.0, 0.5 * fundamental * q + 0.5 * price * growth_factor)
            else:
                out[key] = max(0.0, fundamental * q)
        elif growth_factor is not None:
            # EPS-драйвер недоступен/убыточен — только исторический рост цены
            quality[key] = None
            out[key] = max(0.0, price * growth_factor)
        else:
            quality[key] = None
            out[key] = None
    return {
        "h1": out.get("h1"),
        "h2": out.get("h2"),
        "h3": out.get("h3"),
        "pe_base": pe_base,
        # None-значения не отдаём: схема ответа — dict[str, float]
        "quality": {k: v for k, v in quality.items() if v is not None} or None,
        "price_growth": price_growth,
        "method": (
            "0.5·EPS(combined WMA+LinReg)×P/E×quality(revenue/EPS/FCF) + "
            "0.5·цена×(1+рост цены 5л)^h, итог ≥ 0"
        ),
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Квартальный прогноз цены + кросс-валидация с годовым
# ══════════════════════════════════════════════════════════════════════ #
#: Насколько квартальный темп может опережать годовую траекторию (25%)
_PACE_TOLERANCE = 1.25

#: Человеческие пояснения к кодам кросс-валидации
_CV_NOTES = {
    "direction": (
        "Квартальный тренд противоречил годовому прогнозу — "
        "сглажен к годовой траектории"
    ),
    "pace": (
        "Квартальный темп был быстрее годового — "
        "ограничен годовой траекторией"
    ),
}


def cross_validate_quarterly_price(
    price: float,
    quarterly: dict[str, float | None],
    annual: dict | None,
) -> dict:
    """Согласовать квартальный путь цены с годовым прогнозом.

    Годовая траектория интерполируется геометрически:
    ``implied_q = price · (price_1y / price)^(h/4)``. Правила:

    * **направление** — если квартальный прогноз идёт против годового
      (падение при годовом росте и наоборот), он смешивается 50/50 с
      годовой траекторией и прижимается в коридор ``[price, implied]``
      (противоречие в крайнем случае становится «без изменения»);
    * **темп** — квартальное изменение может опережать годовую траекторию не больше
      чем на ``_PACE_TOLERANCE`` (25%) и никогда — сильнее полного годового
      изменения (кварталы не могут «перерасти» год); иначе урезается.

    Returns
    -------
    dict
        ``{values, annual_implied, adjusted, codes, notes}`` — ``values``
        уже согласованы; ``codes`` — направления правок (direction/pace).
    """
    annual_h1 = (annual or {}).get("h1")
    if not price or price <= 0 or annual_h1 is None or annual_h1 <= 0:
        return {
            "values": dict(quarterly),
            "annual_implied": None,
            "adjusted": False,
            "codes": [],
            "notes": [],
        }

    implied: dict[str, float] = {}
    out: dict[str, float | None] = {}
    codes: list[str] = []
    year_ret = abs(annual_h1 / price - 1.0)
    for h in QUARTERLY_HORIZONS:
        key = f"q{h}"
        imp = price * (annual_h1 / price) ** (h / 4.0)
        implied[key] = imp
        value = quarterly.get(key)
        if value is None:
            out[key] = None
            continue
        imp_ret = imp / price - 1.0
        ret = value / price - 1.0
        if ret * imp_ret < 0:  # разные направления — тянем к годовой траектории
            value = 0.5 * value + 0.5 * imp
            lo, hi = (price, imp) if imp >= price else (imp, price)
            value = min(max(value, lo), hi)
            ret = value / price - 1.0
            if "direction" not in codes:
                codes.append("direction")
        # допустимый квартальный темп: годовая траектория + допуск, но не больше года
        limit = min(abs(imp_ret) * _PACE_TOLERANCE, year_ret)
        if abs(ret) > limit + 1e-12:  # быстрее годового пути
            value = price * (1.0 + (1.0 if ret >= 0 else -1.0) * limit)
            if "pace" not in codes:
                codes.append("pace")
        out[key] = max(0.0, value)

    return {
        "values": out,
        "annual_implied": implied,
        "adjusted": bool(codes),
        "codes": codes,
        "notes": [_CV_NOTES[c] for c in codes],
    }


def build_quarterly_price_forecast(
    price: float,
    metrics: dict,
    price_growth: float | None = None,
    annual: dict | None = None,
) -> dict | None:
    """Прогноз цены на 1/2/3 квартала теми же моделями, что и годовой.

    Драйвер — квартальный EPS-прогноз (WMA + LinReg + влияние последнего
    отчёта, :func:`build_quarterly_forecast`). Прибыль приводится к TTM:
    ``TTM_h = Σ прогноз(q1..qh) + Σ факт последних (4−h) кварталов``, цена
    считается той же смесью 50/50, что и годовая:

    ``price_qh = 0.5 · TTM_h · P/E(TTM) · quality_qh + 0.5 · price · (1+g)^(h/4)``

    Результат проходит кросс-валидацию с годовым прогнозом
    (:func:`cross_validate_quarterly_price`). Возвращает ``None``, если
    квартальных данных недостаточно (меньше 4 кварталов факта).
    """
    eps_q = ((metrics.get("eps") or {}).get("quarterly")) or {}
    hist = [
        float(p["value"]) for p in (eps_q.get("history") or [])
        if p.get("value") is not None
    ]
    fc = ((eps_q.get("forecast") or {}).get("combined")) or {}
    if not price or price <= 0 or len(hist) < 4 or not fc:
        return None

    eps_ttm = sum(hist[-4:])
    pe_base = price / eps_ttm if eps_ttm > 0 else None

    quality: dict[str, float] = {}
    raw: dict[str, float | None] = {}
    for h in QUARTERLY_HORIZONS:
        key = f"q{h}"
        forecasts = [fc.get(f"q{i}") for i in range(1, h + 1)]
        growth_factor = (
            (1.0 + price_growth) ** (h / 4.0) if price_growth is not None else None
        )
        fundamental = None
        if pe_base and all(v is not None for v in forecasts):
            ttm_h = sum(forecasts) + sum(hist[-(4 - h):])
            if ttm_h > 0:
                fundamental = ttm_h * pe_base
        if fundamental is not None:
            q = _quality_multiplier(metrics, key, period="quarterly")
            quality[key] = q
            raw[key] = max(
                0.0,
                0.5 * fundamental * q + 0.5 * price * growth_factor
                if growth_factor is not None else fundamental * q,
            )
        elif growth_factor is not None:
            raw[key] = max(0.0, price * growth_factor)
        else:
            raw[key] = None

    cv = cross_validate_quarterly_price(price, raw, annual)
    values = cv["values"]
    return {
        "q1": values.get("q1"),
        "q2": values.get("q2"),
        "q3": values.get("q3"),
        "pe_base": pe_base,
        "eps_ttm": eps_ttm,
        "quality": quality or None,
        "price_growth": price_growth,
        "cross_validation": {
            "adjusted": cv["adjusted"],
            "codes": cv["codes"],
            "notes": cv["notes"],
            "annual_implied": cv["annual_implied"],
        },
        "method": (
            "0.5·TTM EPS(прогноз кварталов)×P/E(TTM)×quality + "
            "0.5·цена×(1+рост цены 5л)^(h/4), кросс-валидация с годовым прогнозом"
        ),
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Сервис: оркестрация прогноза по тикеру
# ══════════════════════════════════════════════════════════════════════ #
class SecForecastService:
    """Прогноз фундаментальных метрик компании (поверх core fundamentals).

    Источник: :class:`~gex.sec_fundamentals.SecFundamentalsService`
    (annual-ряды + баланс из PostgreSQL/Redis). Цена — yfinance через
    ``TATimeframesFetcher.fetch_spot`` (Redis-кэш 300с).
    """

    def __init__(self, redis_client: RedisClient | None = None):
        self.redis = redis_client

    # ── Public API ────────────────────────────────────────────────────
    def get_forecast_core(
        self,
        ticker: str,
        *,
        horizon_years: int = 5,
        window: int = 5,
        alpha: float = 0.5,
        anomaly_threshold: float = 0.6,
        anomaly_mode: str = "both",
        last_report_weight: float = DEFAULT_LAST_REPORT_WEIGHT,
    ) -> dict:
        """Прогноз всех метрик БЕЗ цены/PEG — кэшируется SWR.

        Цена подмешивается отдельно (:meth:`with_price`), чтобы PEG и
        valuation не замораживались на TTL кэша core-данных.
        """
        from gex.application.sec.sec_fundamentals import SecFundamentalsService

        core = SecFundamentalsService(redis_client=self.redis).get_fundamentals_core(ticker)
        metrics = self._build_metrics(
            core, horizon_years, window, alpha, anomaly_threshold, last_report_weight
        )
        quarterly = self._build_quarterly(ticker, window, alpha, last_report_weight)
        # Квартальные ряды вкладываем в КАЖДУЮ метрику (схема MetricForecast.quarterly) —
        # фронт рисует переключатель «Год/Квартал» по наличию metrics[k].quarterly
        for key, qf in quarterly.items():
            if key in metrics:
                metrics[key]["quarterly"] = qf

        return {
            "ticker": core["ticker"],
            "cik": core["cik"],
            "parameters": {
                "horizon_years": horizon_years,
                "window": window,
                "alpha": alpha,
                "anomaly_threshold": anomaly_threshold,
                "anomaly_mode": anomaly_mode,
                "last_report_weight": last_report_weight,
            },
            "metrics": metrics,
            "quarterly": quarterly,
            "shares": (core.get("balance") or {}).get("shares_outstanding"),
            "eps_history": [p["value"] for p in self._annual_series(core, "eps_basic")],
        }

    def with_price(self, core: dict, price: float | None) -> dict:
        """Добавить цену, PEG, базовый сценарий и прогноз цены (год + кварталы)."""
        peg, valuation = self._peg_and_valuation(core, price)
        eps_last = (core.get("eps_history") or [None])[-1]
        price_forecast = None
        if price is not None:
            # Поправка на 5-летний рост цены (CAGR из yfinance, кэш 12ч)
            price_growth = self._fetch_price_growth(core.get("ticker") or "")
            metrics = core.get("metrics") or {}
            price_forecast = build_price_forecast(price, metrics, eps_last, price_growth)
            # Квартальный прогноз цены (переключатель «Год / Квартал» на фронте)
            price_forecast["quarterly"] = build_quarterly_price_forecast(
                price, metrics, price_growth, price_forecast
            )
        # Внутренние ключи (shares/eps_history) не входят в API-схему
        result = {k: v for k, v in core.items() if k not in ("shares", "eps_history")}
        result.update(
            price={"value": price, "source": "yfinance", "currency": "USD"}
            if price is not None else None,
            peg=peg,
            valuation=valuation,
            price_forecast=price_forecast,
        )
        return result

    def get_forecast(
        self,
        ticker: str,
        *,
        horizon_years: int = 5,
        window: int = 5,
        alpha: float = 0.5,
        anomaly_threshold: float = 0.6,
        anomaly_mode: str = "both",
        last_report_weight: float = DEFAULT_LAST_REPORT_WEIGHT,
    ) -> dict:
        """Удобная обёртка: core + цена (для тестов/live-скриптов)."""
        core = self.get_forecast_core(
            ticker,
            horizon_years=horizon_years,
            window=window,
            alpha=alpha,
            anomaly_threshold=anomaly_threshold,
            anomaly_mode=anomaly_mode,
            last_report_weight=last_report_weight,
        )
        return self.with_price(core, self.fetch_price(ticker))

    def get_shares_reconciliation(self, ticker: str) -> dict:
        """Количество акций: SEC + сверка/фолбэк Finnhub (profile2).

        Если SEC не дал акции — берём Finnhub; если дал — проверяем, что
        Finnhub сходится по размеру (масштабы ×1/×10³/×10⁶/×10⁹
        подбираются автоматически).
        """
        from gex.adapters.persistence.database import SessionLocal
        from gex.adapters.providers.finnhub_client import get_company_profile2, reconcile_shares
        from gex.adapters.persistence.sec_models import CompanyMetric

        sec_shares = None
        try:
            with SessionLocal() as db:
                row = (
                    db.query(CompanyMetric)
                    .filter(
                        CompanyMetric.ticker == ticker,
                        CompanyMetric.metric == "shares_outstanding",
                    )
                    .order_by(CompanyMetric.end.desc())
                    .first()
                )
                sec_shares = float(row.val) if row else None
        except Exception as exc:
            logger.debug("SEC shares недоступны для %s: %s", ticker, exc)

        fh_raw = None
        try:
            profile = get_company_profile2(ticker)
            fh_raw = profile.get("shareOutstanding")
        except Exception as exc:
            logger.warning("Finnhub shares недоступен для %s: %s", ticker, exc)

        result = reconcile_shares(sec_shares, fh_raw)
        result["ticker"] = ticker
        if result["source"] is None:
            raise ValueError(f"Тикер {ticker} не найден (нет данных об акциях в SEC и Finnhub)")
        return result

    def fetch_price(self, ticker: str) -> float | None:
        """Текущая цена из yfinance (Redis-кэш 300с). None при недоступности."""
        try:
            from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher

            return float(create_timeframes_fetcher(redis_client=self.redis).fetch_spot(ticker))
        except Exception as exc:
            logger.warning("Не удалось получить цену %s: %s", ticker, exc)
            return None

    def _fetch_price_growth(self, ticker: str) -> float | None:
        """Среднегодовой рост цены за 5 лет (CAGR) из месячных close yfinance.

        Поправочный коэффициент для прогноза цены акции. Кэшируется в Redis
        12ч (свежесть сопоставима с SEC-данными). При любой ошибке — None
        (прогноз строится без поправки, только по EPS×P/E×quality).
        """
        try:
            if self.redis is not None and self.redis.connected:
                key = cache_key("res", "sec-price-growth", ticker.strip().upper())
                raw = self.redis.get(key)
                if raw is not None:
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        pass

            from gex.adapters.providers.yfinance import history as yfinance_history

            hist = yfinance_history(ticker.strip().upper(), period="5y", interval="1mo")
            if hist is None:
                return None
            closes = hist["Close"].dropna()
            if closes is None or len(closes) < 2:
                return None
            first, last = float(closes.iloc[0]), float(closes.iloc[-1])
            if first <= 0 or last <= 0:
                return None
            years = max(len(closes) / 12.0, 1.0)
            growth = (last / first) ** (1.0 / years) - 1.0

            if self.redis is not None and self.redis.connected:
                self.redis.set(key, growth, ex=12 * 3600)
            return growth
        except Exception as exc:
            logger.debug("Price growth (5y) недоступен для %s: %s", ticker, exc)
            return None

    # ── Внутреннее ────────────────────────────────────────────────────
    @staticmethod
    def _annual_series(core: dict, key: str) -> list[dict]:
        """Годовой ряд метрики: [{end, fy, value}] (пропуская пустые годы)."""
        out = []
        for row in core.get("annual") or []:
            value = row.get(key)
            if value is None:
                continue
            out.append({"end": row["end"], "fy": row.get("fy"), "value": float(value)})
        return out

    def _build_metrics(
        self,
        core: dict,
        horizon_years: int,
        window: int,
        alpha: float,
        anomaly_threshold: float,
        last_report_weight: float,
    ) -> dict[str, dict]:
        """Прогноз по каждой метрике FORECAST_METRICS."""
        by_key = {
            "revenue": lambda r: r.get("revenue"),
            "free_cash_flow": lambda r: r.get("free_cash_flow"),
            "net_debt_to_ebitda": lambda r: (
                r.get("net_debt") / r["ebitda"]
                if r.get("net_debt") is not None and r.get("ebitda")
                else None
            ),
            "eps_basic": lambda r: r.get("eps_basic"),
            "operating_margin": lambda r: r.get("operating_margin"),
            "roe": lambda r: r.get("roe"),
        }

        metrics: dict[str, dict] = {}
        for key, out_key, _label, kind in FORECAST_METRICS:
            series = [
                {"end": r["end"], "fy": r.get("fy"), "value": by_key[key](r)}
                for r in (core.get("annual") or [])
                if by_key[key](r) is not None
            ]
            forecast = build_metric_forecast(
                series,
                horizon_years=horizon_years,
                window=window,
                alpha=alpha,
                anomaly_threshold=anomaly_threshold,
                anomaly_enabled=(kind == "flow"),
                last_weight=last_report_weight,
            )
            if out_key == "revenue":
                forecast["growth"]["quarterly"] = self._quarterly_revenue_growth(core["ticker"])
            metrics[out_key] = forecast
        return metrics

    def _build_quarterly(self, ticker: str, window: int, alpha: float, last_report_weight: float) -> dict:
        """Квартальные ряды (3M-строки 10-Q/10-K) + прогноз q1/q2/q3.

        Метрики с квартальным смыслом: revenue, EPS, FCF (CFO − |CapEx|),
        Operating Margin (EBIT/Revenue). Балансовые ratio (NetDebt/EBITDA,
        ROE) квартально не строятся — фронт оставит для них годовой режим.
        """
        try:
            from gex.adapters.persistence.database import SessionLocal
            from gex.adapters.persistence.sec_models import CompanyMetric

            with SessionLocal() as db:
                rows = db.query(CompanyMetric).filter(
                    CompanyMetric.ticker == ticker,
                    CompanyMetric.metric.in_(["revenue", "cfo", "capex", "eps_basic", "operating_income"]),
                ).all()
        except Exception as exc:
            logger.debug("Quarterly series недоступны для %s: %s", ticker, exc)
            return {}

        # 3M-строки (duration 70..115 дней) по метрикам
        by_metric: dict[str, dict[object, float]] = {}
        for r in rows:
            if not r.start or not r.end:
                continue
            days = (r.end - r.start).days
            if not 70 <= days <= 115:
                continue
            by_metric.setdefault(r.metric, {})[r.end] = r.val

        def _series(metric: str) -> list[dict]:
            return [
                {"end": end, "value": float(val)}
                for end, val in sorted(by_metric.get(metric, {}).items())
            ][-20:]  # последние 20 кварталов (~5 лет)

        def _merged(primary: str, secondary: str, fn) -> list[dict]:
            p = by_metric.get(primary, {})
            s = by_metric.get(secondary, {})
            out = []
            for end in sorted(p):
                if end in s:
                    out.append({"end": end, "value": float(fn(p[end], s[end]))})
            return out[-20:]

        quarterly: dict[str, dict] = {}
        for metric, series in (
            ("revenue", _series("revenue")),
            ("eps", _series("eps_basic")),
            ("fcf", _merged("cfo", "capex", lambda c, x: c - abs(x))),
            ("operating_margin", _merged("operating_income", "revenue", lambda oi, rev: oi / rev if rev else None)),
        ):
            series = [p for p in series if p["value"] is not None]
            quarterly[metric] = build_quarterly_forecast(
                series, window=window, alpha=alpha, last_weight=last_report_weight
            )
        return quarterly

    def _quarterly_revenue_growth(self, ticker: str) -> float | None:
        """Квартальный рост выручки: последняя 3M-строка / предыдущая 3M − 1.

        Берутся 10-Q строки revenue с duration ≈ 3 месяца (start ≈ end − 90д)
        — нарастающие 9M-строки отбрасываются.
        """
        try:
            from gex.adapters.persistence.database import SessionLocal
            from gex.application.sec.sec_fundamentals import read_metrics

            with SessionLocal() as db:
                rows = read_metrics(db, ticker, "revenue")
        except Exception as exc:
            # BUG-STATIC-01: раньше здесь глушился ImportError несуществующего имени
            # (`_read_metrics` — метод класса, а не функция модуля), поэтому квартальный рост
            # выручки молча возвращал None всегда. Теперь сбой виден в логах.
            logger.warning("Quarterly revenue недоступен для %s: %s", ticker, exc)
            return None

        three_month = []
        for r in rows:
            if not r.get("start") or not r.get("end"):
                continue
            days = (r["end"] - r["start"]).days
            if 70 <= days <= 115:  # 3M строки (~90 дней), любая форма (10-Q и 10-K)
                three_month.append(r)
        if len(three_month) < 2:
            return None
        three_month.sort(key=lambda r: r["end"])
        prev, last = three_month[-2]["val"], three_month[-1]["val"]
        if prev <= 0 or last <= 0:
            return None
        return last / prev - 1.0

    def _peg_and_valuation(
        self, core: dict, price: float | None
    ) -> tuple[dict | None, dict | None]:
        """PEG (текущий) и базовый сценарий для калькулятора."""
        eps_values = core.get("eps_history") or []
        eps_last = eps_values[-1] if eps_values else None
        shares = core.get("shares")

        valuation = None
        if price is not None and eps_last is not None and eps_last > 0:
            pe = price / eps_last
            earnings = eps_last * shares if shares else None
            valuation = {
                "price": price,
                "pe": pe,
                "eps": eps_last,
                "earnings": earnings,
                "shares": shares,
                "market_cap": price * shares if shares else None,
            }

        peg = None
        if price is not None and eps_values:
            growth = cagr(eps_values, 3) or cagr(eps_values, 5) or yoy_growth(eps_values)
            pe = price / eps_last if eps_last and eps_last > 0 else None
            if growth is not None and growth > 0 and pe is not None:
                peg = {
                    "growth": growth,
                    "pe": pe,
                    "peg": pe / (growth * 100),
                    "warning": None,
                }
            elif eps_last is not None:
                peg = {
                    "growth": growth,
                    "pe": pe,
                    "peg": None,
                    "warning": "Темп роста прибыли ≤ 0 — PEG не определён",
                }
        return peg, valuation
