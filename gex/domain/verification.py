from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE LONG"
    CLOSE_SHORT = "CLOSE SHORT"


class MarketRegime(str, Enum):
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    FLAT = "flat"
    IMPULSE_VOLATILITY = "impulse_volatility"


@dataclass(frozen=True)
class VerificationInput:
    signal_type: SignalType
    timeframe: str
    candles: list[dict[str, Any]]
    indicators_current: dict[str, Any]
    indicators_higher: dict[str, Any] | None = None
    volume_data: list[float] | None = None


@dataclass(frozen=True)
class VerificationResult:
    score: float
    market_regime: MarketRegime
    consensus_score: float
    divergence_score: float
    trend_strength_score: float
    breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass
class _IndicatorSeries:
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open_: np.ndarray
    volume: np.ndarray | None
    macd_line: np.ndarray
    macd_signal: np.ndarray
    macd_hist: np.ndarray
    ao: np.ndarray
    rsi: np.ndarray
    stoch_k: np.ndarray
    stoch_d: np.ndarray
    aroon_up: np.ndarray
    aroon_down: np.ndarray
    adx: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray
    atr: np.ndarray | None = None


class VerificationEngine:
    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)

    @staticmethod
    def _target_for_signal(sig: SignalType) -> float:
        if sig in (SignalType.LONG, SignalType.CLOSE_SHORT):
            return 1.0
        return -1.0

    def verify(self, data: VerificationInput) -> VerificationResult:
        n = len(data.candles)
        if n < 50:
            raise ValueError(f"Need at least 50 candles, got {n}")

        ind = self._to_arrays(data)
        regime = self._detect_regime(ind)
        weights = self._weights_for_regime(regime)

        group_a = self._score_group_a(ind, data.signal_type)
        group_b = self._score_group_b(ind, data.signal_type)
        consensus_raw = 0.55 * group_a["score"] + 0.45 * group_b["score"]
        consensus = float(np.clip(consensus_raw, 0.0, 1.0))

        divergences = self._detect_divergences(ind, data.signal_type)
        div_score = float(np.clip(0.5 + divergences["net"], 0.0, 1.0))

        trend_score = self._score_trend(ind, data.signal_type, data.indicators_higher)

        score = 100.0 * (
            weights["w_mode"] * consensus
            + weights["w_div"] * div_score
            + weights["w_trend"] * trend_score
        ) / (weights["w_mode"] + weights["w_div"] + weights["w_trend"])

        score = self._apply_punishments(score, ind, data.signal_type, divergences, data)
        score = float(np.clip(score, 0.0, 100.0))

        breakdown = {
            "market_regime": regime.value,
            "weights": weights,
            "group_a": group_a,
            "group_b": group_b,
            "consensus": consensus,
            "divergences": divergences,
            "divergence_score": div_score,
            "trend_strength_score": trend_score,
            "volume_factor": self._volume_factor(data, data.signal_type),
        }

        return VerificationResult(
            score=round(score, 2),
            market_regime=regime,
            consensus_score=round(consensus, 4),
            divergence_score=round(div_score, 4),
            trend_strength_score=round(trend_score, 4),
            breakdown=breakdown,
        )

    def _to_arrays(self, data: VerificationInput) -> _IndicatorSeries:
        c = np.array([bar.get("close", 0.0) for bar in data.candles], dtype=float)
        h = np.array([bar.get("high", 0.0) for bar in data.candles], dtype=float)
        l = np.array([bar.get("low", 0.0) for bar in data.candles], dtype=float)
        o = np.array([bar.get("open", 0.0) for bar in data.candles], dtype=float)
        v = np.array([bar.get("volume", 0.0) for bar in data.candles], dtype=float) if any(
            "volume" in bar for bar in data.candles
        ) else None
        ic = data.indicators_current
        return _IndicatorSeries(
            close=c,
            high=h,
            low=l,
            open_=o,
            volume=v,
            macd_line=np.array(ic.get("macd_line", []), dtype=float),
            macd_signal=np.array(ic.get("macd_signal", []), dtype=float),
            macd_hist=np.array(ic.get("macd_histogram", []), dtype=float),
            ao=np.array(ic.get("ao", []), dtype=float),
            rsi=np.array(ic.get("rsi", []), dtype=float),
            stoch_k=np.array(ic.get("stoch_k", []), dtype=float),
            stoch_d=np.array(ic.get("stoch_d", []), dtype=float),
            aroon_up=np.array(ic.get("aroon_up", []), dtype=float),
            aroon_down=np.array(ic.get("aroon_down", []), dtype=float),
            adx=np.array(ic.get("adx", []), dtype=float),
            plus_di=np.array(ic.get("plus_di", []), dtype=float),
            minus_di=np.array(ic.get("minus_di", []), dtype=float),
            atr=np.array(ic.get("atr", []), dtype=float) if "atr" in ic else None,
        )

    def _detect_regime(self, ind: _IndicatorSeries) -> MarketRegime:
        n = len(ind.adx)
        if n < 5:
            return MarketRegime.FLAT

        adx_now = ind.adx[-1]
        adx_prev5 = ind.adx[-5:]
        adx_rising = all(adx_prev5[i] <= adx_prev5[i + 1] for i in range(len(adx_prev5) - 1))

        aroon_up_now = ind.aroon_up[-1]
        aroon_down_now = ind.aroon_down[-1]

        if adx_now > 25 and (aroon_up_now > 70 or aroon_down_now > 70) and adx_rising:
            if self._is_impulse(ind):
                return MarketRegime.IMPULSE_VOLATILITY
            return MarketRegime.STRONG_TREND

        if 20 <= adx_now <= 25 or abs(aroon_up_now - aroon_down_now) < 30:
            return MarketRegime.WEAK_TREND

        return MarketRegime.FLAT

    def _is_impulse(self, ind: _IndicatorSeries) -> bool:
        if ind.atr is None or len(ind.atr) < 28:
            return False
        atr_now = ind.atr[-1]
        atr_sma = np.mean(ind.atr[-14:]) if len(ind.atr) >= 14 else atr_now
        return bool(atr_sma > 0 and atr_now > 1.5 * atr_sma)

    def _weights_for_regime(self, regime: MarketRegime) -> dict[str, float]:
        if regime == MarketRegime.STRONG_TREND:
            return {"w_mode": 0.15, "w_div": 0.20, "w_trend": 0.65}
        if regime == MarketRegime.WEAK_TREND:
            return {"w_mode": 0.35, "w_div": 0.25, "w_trend": 0.40}
        if regime == MarketRegime.FLAT:
            return {"w_mode": 0.60, "w_div": 0.25, "w_trend": 0.15}
        return {"w_mode": 0.30, "w_div": 0.40, "w_trend": 0.30}

    def _score_group_a(self, ind: _IndicatorSeries, sig: SignalType) -> dict[str, Any]:
        target = self._target_for_signal(sig)

        macd_pts, macd_details = self._score_macd(ind, target)
        ao_pts, ao_details = self._score_ao(ind, target)
        aroon_pts, aroon_details = self._score_aroon(ind, target)
        dmi_pts, dmi_details = self._score_dmi(ind, target)

        raw = 0.30 * macd_pts + 0.25 * ao_pts + 0.25 * aroon_pts + 0.20 * dmi_pts
        score = float(np.clip(raw, 0.0, 1.0))
        return {
            "score": score,
            "macd": macd_details,
            "ao": ao_details,
            "aroon": aroon_details,
            "dmi": dmi_details,
        }

    def _score_macd(self, ind: _IndicatorSeries, target: float) -> tuple[float, dict[str, Any]]:
        if len(ind.macd_hist) < 5:
            return 0.5, {}
        hist = ind.macd_hist
        line = ind.macd_line
        signal = ind.macd_signal

        conds = {
            "hist_positive": bool(target * hist[-1] > 0),
            "main_above_signal": bool(line[-1] > signal[-1]) if target > 0 else bool(line[-1] < signal[-1]),
            "hist_growing": bool(target * (hist[-1] - hist[-2]) > 0),
            "hist_growing_2of3": bool(target * (hist[-1] - hist[-2]) > 0 and target * (hist[-2] - hist[-3]) > 0),
            "ma_slope_positive_5": bool(np.all(np.diff(line[-5:]) * target > 0)) if len(line) >= 5 else False,
        }
        pts = sum(1 for v in conds.values() if v) / len(conds)
        return pts, conds

    def _score_ao(self, ind: _IndicatorSeries, target: float) -> tuple[float, dict[str, Any]]:
        if len(ind.ao) < 5:
            return 0.5, {}
        ao = ind.ao
        recent = ao[-5:]
        zero_cross_up = bool(np.any((recent[:-1] < 0) & (recent[1:] >= 0)))
        zero_cross_down = bool(np.any((recent[:-1] > 0) & (recent[1:] <= 0)))
        conds = {
            "ao_positive": bool(target * ao[-1] > 0),
            "ao_growing_2bars": bool(target * (ao[-1] - ao[-2]) > 0),
            "zero_cross_up_recent": zero_cross_up,
            "zero_cross_down_recent": zero_cross_down,
        }
        pts = sum(1 for v in conds.values() if v) / len(conds)
        if conds["ao_positive"] and conds["zero_cross_up_recent"] and target > 0:
            pts = max(pts, 0.9)
        if conds["ao_positive"] and conds["zero_cross_down_recent"] and target < 0:
            pts = max(pts, 0.9)
        return pts, conds

    def _score_aroon(self, ind: _IndicatorSeries, target: float) -> tuple[float, dict[str, Any]]:
        up = ind.aroon_up[-1]
        down = ind.aroon_down[-1]
        up_rising = ind.aroon_up[-1] > ind.aroon_up[-2] if len(ind.aroon_up) >= 2 else False
        conds = {
            "trend_favorable": bool(up > down) if target > 0 else bool(down > up),
            "extreme_favorable": bool(up > 70) if target > 0 else bool(down > 70),
            "up_rising": bool(up_rising) if target > 0 else bool(not up_rising),
            "counter_weak": bool(down < 30) if target > 0 else bool(up < 30),
        }
        pts = sum(1 for v in conds.values() if v) / len(conds)
        return pts, conds

    def _score_dmi(self, ind: _IndicatorSeries, target: float) -> tuple[float, dict[str, Any]]:
        pdm = ind.plus_di[-1]
        mdm = ind.minus_di[-1]
        adx_now = ind.adx[-1]
        adx_prev = ind.adx[-2] if len(ind.adx) >= 2 else adx_now
        conds = {
            "pdm_above_mdm": bool(pdm > mdm),
            "adx_rising": bool(adx_now > adx_prev),
        }
        pts = sum(1 for v in conds.values() if v) / len(conds)
        if target > 0 and not conds["pdm_above_mdm"]:
            pts -= 0.3
        if target < 0 and conds["pdm_above_mdm"]:
            pts -= 0.3
        return float(np.clip(pts, 0.0, 1.0)), conds

    def _score_group_b(self, ind: _IndicatorSeries, sig: SignalType) -> dict[str, Any]:
        target = self._target_for_signal(sig)

        rsi_pts, rsi_det = self._score_rsi(ind, target)
        stoch_pts, stoch_det = self._score_stoch(ind, target)
        sync_pts, sync_det = self._score_sync(ind, target)

        raw = 0.40 * rsi_pts + 0.35 * stoch_pts + 0.25 * sync_pts
        score = float(np.clip(raw, 0.0, 1.0))
        return {"score": score, "rsi": rsi_det, "stoch": stoch_det, "sync": sync_det}

    def _score_rsi(self, ind: _IndicatorSeries, target: float) -> tuple[float, dict[str, Any]]:
        if len(ind.rsi) < 3:
            return 0.5, {}
        r = ind.rsi
        growth = np.diff(r[-3:])
        conds = {
            "in_zone": bool(50 < r[-1] < 70) if target > 0 else bool(30 < r[-1] < 50),
            "growing_3bars": bool(np.all(growth > 0) if target > 0 else np.all(growth < 0)),
            "exit_oversold": bool(r[-1] > 30 and r[-2] <= 30) if target > 0 else bool(r[-1] < 70 and r[-2] >= 70),
        }
        pts = sum(1 for v in conds.values() if v) / len(conds)
        if conds["exit_oversold"]:
            pts = max(pts, 0.9)
        return pts, conds

    def _score_stoch(self, ind: _IndicatorSeries, target: float) -> tuple[float, dict[str, Any]]:
        if len(ind.stoch_k) < 3:
            return 0.5, {}
        k = ind.stoch_k
        d = ind.stoch_d
        conds = {
            "k_above_d": bool(k[-1] > d[-1]) if target > 0 else bool(k[-1] < d[-1]),
            "both_moving_favorable": bool(target * (k[-1] - k[-2]) > 0 and target * (d[-1] - d[-2]) > 0),
            "not_extreme": bool(k[-1] < 85) if target > 0 else bool(k[-1] > 15),
            "exit_oversold": bool(k[-1] > 20 and k[-2] <= 20) if target > 0 else bool(k[-1] < 80 and k[-2] >= 80),
        }
        pts = sum(1 for v in conds.values() if v) / len(conds)
        if conds["exit_oversold"]:
            pts = max(pts, 0.9)
        return pts, conds

    def _score_sync(self, ind: _IndicatorSeries, target: float) -> tuple[float, dict[str, Any]]:
        if len(ind.rsi) < 5 or len(ind.stoch_k) < 5:
            return 0.5, {}
        rsi_changes = np.diff(ind.rsi[-5:])
        stoch_changes = np.diff(ind.stoch_k[-5:])
        if len(rsi_changes) == 0 or len(stoch_changes) == 0:
            return 0.5, {}
        corr = np.corrcoef(rsi_changes, stoch_changes)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        pts = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
        return pts, {"correlation": round(float(corr), 4)}

    def _detect_divergences(self, ind: _IndicatorSeries, sig: SignalType) -> dict[str, Any]:
        highs = self._find_local_extrema(ind.high, 5, find_high=True)
        lows = self._find_local_extrema(ind.low, 5, find_high=False)
        target = self._target_for_signal(sig)
        divs: list[dict[str, Any]] = []

        for osc_name, osc in [
            ("rsi", ind.rsi),
            ("macd_hist", ind.macd_hist),
            ("ao", ind.ao),
            ("stoch_k", ind.stoch_k),
        ]:
            if len(osc) < 20:
                continue
            divs.extend(self._classic_divergences(osc, ind.high, ind.low, highs, lows, osc_name, target))

        net = sum(d["weight"] for d in divs)
        return {"count": len(divs), "items": divs, "net": float(np.clip(net, -1.0, 1.0))}

    def _find_local_extrema(self, series: np.ndarray, window: int, find_high: bool) -> list[int]:
        indices = []
        for i in range(window, len(series) - window):
            segment = series[i - window : i + window + 1]
            if find_high:
                if series[i] == np.max(segment):
                    indices.append(i)
            else:
                if series[i] == np.min(segment):
                    indices.append(i)
        return indices

    def _classic_divergences(
        self,
        osc: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        highs: list[int],
        lows: list[int],
        name: str,
        target: float,
    ) -> list[dict[str, Any]]:
        results = []
        if len(highs) >= 2:
            i2, i1 = highs[-2], highs[-1]
            if high[i2] < high[i1] and osc[i2] > osc[i1]:
                weight = -0.3 if target > 0 else 0.3
                results.append({"type": "classic_bearish", "oscillator": name, "weight": weight})
            elif high[i2] > high[i1] and osc[i2] < osc[i1] and target > 0:
                results.append({"type": "hidden_bullish", "oscillator": name, "weight": 0.2})
        if len(lows) >= 2:
            i2, i1 = lows[-2], lows[-1]
            if low[i2] > low[i1] and osc[i2] < osc[i1]:
                weight = 0.3 if target > 0 else -0.3
                results.append({"type": "classic_bullish", "oscillator": name, "weight": weight})
            elif low[i2] < low[i1] and osc[i2] > osc[i1] and target < 0:
                results.append({"type": "hidden_bearish", "oscillator": name, "weight": 0.2})
        return results

    def _score_trend(self, ind: _IndicatorSeries, sig: SignalType, higher: dict[str, Any] | None) -> float:
        if higher is not None:
            try:
                higher_ind = self._to_arrays(VerificationInput(
                    signal_type=sig,
                    timeframe="higher",
                    candles=[{} for _ in range(len(higher.get("rsi", [])))],
                    indicators_current=higher,
                ))
                regime = self._detect_regime(higher_ind)
                h_adx = higher_ind.adx[-1] if len(higher_ind.adx) > 0 else 0.0
                h_up = higher_ind.aroon_up[-1] if len(higher_ind.aroon_up) > 0 else 0.0
                h_down = higher_ind.aroon_down[-1] if len(higher_ind.aroon_down) > 0 else 0.0
                higher_score = self._higher_tf_alignment_score(h_adx, h_up, h_down, sig)
                return higher_score
            except Exception:
                pass

        adx_now = ind.adx[-1] if len(ind.adx) > 0 else 0.0
        aroon_up = ind.aroon_up[-1] if len(ind.aroon_up) > 0 else 0.0
        aroon_down = ind.aroon_down[-1] if len(ind.aroon_down) > 0 else 0.0
        return self._higher_tf_alignment_score(adx_now, aroon_up, aroon_down, sig)

    def _higher_tf_alignment_score(self, adx: float, aroon_up: float, aroon_down: float, sig: SignalType) -> float:
        if adx < 20:
            base = 0.5
        elif self._target_for_signal(sig) > 0:
            base = min(1.0, adx / 50.0) * (aroon_up / 100.0)
        else:
            base = min(1.0, adx / 50.0) * (aroon_down / 100.0)
        return float(np.clip(base, 0.0, 1.0))

    def _volume_factor(self, data: VerificationInput, sig: SignalType) -> dict[str, Any]:
        if not data.volume_data or len(data.volume_data) < 5:
            return {"score": 0.5, "details": {}}
        vol = np.array(data.volume_data, dtype=float)
        vol_sma = np.mean(vol[-5:])
        current = vol[-1]
        target = self._target_for_signal(sig)

        details = {
            "current_volume": float(current),
            "avg_volume_5": float(vol_sma),
            "ratio": float(current / vol_sma) if vol_sma > 0 else 1.0,
        }

        n = len(data.candles)
        if n >= 2:
            current_bar = data.candles[-1]
            prev_bar = data.candles[-2]
            bullish = current_bar.get("close", 0) > current_bar.get("open", 0)
            bearish = current_bar.get("close", 0) < current_bar.get("open", 0)
            if target > 0 and bullish and current > vol_sma:
                details["direction_confirm"] = True
            elif target < 0 and bearish and current > vol_sma:
                details["direction_confirm"] = True
            else:
                details["direction_confirm"] = False

        base_score = float(np.clip(details["ratio"] / 2.0, 0.0, 1.0))
        if details.get("direction_confirm", False):
            base_score = min(1.0, base_score + 0.15)

        return {"score": base_score, "details": details}

    def _apply_punishments(
        self,
        score: float,
        ind: _IndicatorSeries,
        sig: SignalType,
        divergences: dict[str, Any],
        data: VerificationInput | None = None,
    ) -> float:
        score = score * self._confluence_bonus(ind, sig)
        score = score * self._temporal_synergy_penalty(ind, sig)

        if data is not None:
            vf = self._volume_factor(data, sig)
            if vf["score"] > 0.5:
                score = score * (0.9 + 0.2 * vf["score"])

        target = self._target_for_signal(sig)
        net_div = divergences.get("net", 0.0)

        if abs(net_div) > 0.5:
            if target > 0 and net_div < -0.5:
                score *= 0.5
            elif target < 0 and net_div > 0.5:
                score *= 0.5

        if len(ind.macd_hist) >= 2:
            macd_against = (
                (target > 0 and ind.macd_hist[-1] < 0 and ind.macd_hist[-2] < 0) or
                (target < 0 and ind.macd_hist[-1] > 0 and ind.macd_hist[-2] > 0)
            )
            if macd_against and abs(net_div) > 0.5:
                score *= 0.5
            elif macd_against:
                score *= 0.7

        return score

    def _confluence_bonus(self, ind: _IndicatorSeries, sig: SignalType) -> float:
        """Бонус за согласованность осцилляторов (confluence).

        Проверяет, насколько 4 осциллятора (MACD-гистограмма, AO, Aroon, DMI)
        сонаправлены с сигналом. Когда ≥3 из 4 согласованы — бонус до +8%
        (множитель 1.08); когда ≥2 против сигнала — штраф −8% (множитель 0.92).
        Иначе нейтрально (1.0).

        Это повышает дискриминацию: сетап, подтверждённый несколькими
        независимыми индикаторами, получает премию; разогласованный — штраф.
        """
        target = self._target_for_signal(sig)
        aligned = 0
        opposed = 0

        # MACD-гистограмма: знак согласован с направлением сигнала.
        if len(ind.macd_hist) >= 1 and not np.isnan(ind.macd_hist[-1]):
            if target * ind.macd_hist[-1] > 0:
                aligned += 1
            elif target * ind.macd_hist[-1] < 0:
                opposed += 1

        # Awesome Oscillator: знак.
        if len(ind.ao) >= 1 and not np.isnan(ind.ao[-1]):
            if target * ind.ao[-1] > 0:
                aligned += 1
            elif target * ind.ao[-1] < 0:
                opposed += 1

        # Aroon: up>down для long, down>up для short.
        if len(ind.aroon_up) >= 1 and len(ind.aroon_down) >= 1:
            up, down = ind.aroon_up[-1], ind.aroon_down[-1]
            if not (np.isnan(up) or np.isnan(down)):
                if (target > 0 and up > down) or (target < 0 and down > up):
                    aligned += 1
                else:
                    opposed += 1

        # DMI: +DI > -DI для long, -DI > +DI для short.
        if len(ind.plus_di) >= 1 and len(ind.minus_di) >= 1:
            pdm, mdm = ind.plus_di[-1], ind.minus_di[-1]
            if not (np.isnan(pdm) or np.isnan(mdm)):
                if (target > 0 and pdm > mdm) or (target < 0 and mdm > pdm):
                    aligned += 1
                else:
                    opposed += 1

        if aligned >= 3:
            return 1.08
        if opposed >= 2:
            return 0.92
        return 1.0

    def _temporal_synergy_penalty(self, ind: _IndicatorSeries, sig: SignalType) -> float:
        """Штраф за «опоздавший» вход (temporal synergy).

        Когда тренд-индикаторы уже на экстремуме и разворачиваются против
        сигнала, вход запаздывает — вероятна коррекция. Конкретно:
          * Aroon на экстремуме (>70 по направлению сигнала), но падает —
            тренд стареет;
          * ADX > 40 (очень сильный тренд) и при этом падает — импульс угасает.

        В таких случаях применяется штраф ×0.90. Сигнал «свежий» (индикаторы
        растут или не на экстремуме) — нейтрально (1.0).
        """
        target = self._target_for_signal(sig)
        ageing_signals = 0

        # Aroon на экстремуме и разворачивается.
        if len(ind.aroon_up) >= 2 and len(ind.aroon_down) >= 2:
            up_now, up_prev = ind.aroon_up[-1], ind.aroon_up[-2]
            down_now, down_prev = ind.aroon_down[-1], ind.aroon_down[-2]
            if not any(np.isnan(x) for x in (up_now, up_prev, down_now, down_prev)):
                if target > 0 and up_now > 70 and up_now < up_prev:
                    ageing_signals += 1
                if target < 0 and down_now > 70 and down_now < down_prev:
                    ageing_signals += 1

        # ADX очень сильный, но падает — импульс угасает.
        if len(ind.adx) >= 2:
            adx_now, adx_prev = ind.adx[-1], ind.adx[-2]
            if not (np.isnan(adx_now) or np.isnan(adx_prev)):
                if adx_now > 40 and adx_now < adx_prev:
                    ageing_signals += 1

        if ageing_signals >= 1:
            return 0.90
        return 1.0


def verify_from_dataframe(
    signal_action: str,
    order_type: str,
    timeframe: str,
    dataframe: Any,
) -> dict[str, Any] | None:
    """Run verification on a signal using OHLCV data.

    Returns a dict with the verification result or None on failure.
    """
    if dataframe is None or len(dataframe) < 50:
        logger.warning("Verification skipped for %s %s: dataframe is None or has %d rows (need >= 50)", order_type, timeframe, 0 if dataframe is None else len(dataframe))
        return None

    try:
        import pandas as _pd
        from ta.trend import MACD, ADXIndicator, AroonIndicator
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import AverageTrueRange
    except ImportError as exc:
        logger.error("Verification import error for %s %s: %s", order_type, timeframe, exc)
        return None

    try:
        volume = dataframe.get("volume", _pd.Series([0.0] * len(dataframe)))

        candles: list[dict[str, Any]] = []
        for row in dataframe.to_dict("records"):
            candles.append({
                "open": float(row.get("open", 0.0)),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("close", 0.0)),
                "volume": float(row.get("volume", 0.0)),
            })

        close = dataframe["close"]
        high = dataframe["high"]
        low = dataframe["low"]
        median_price = (high + low) / 2.0
        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()

        macd_obj = MACD(close=close)
        macd_line = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_hist = macd_obj.macd_diff()

        rsi_obj = RSIIndicator(close=close)
        rsi = rsi_obj.rsi()

        stoch_obj = StochasticOscillator(high=high, low=low, close=close)
        stoch_k = stoch_obj.stoch()
        stoch_d = stoch_obj.stoch_signal()

        aroon_obj = AroonIndicator(high=high, low=low)
        aroon_up = aroon_obj.aroon_up()
        aroon_down = aroon_obj.aroon_down()

        adx_obj = ADXIndicator(high=high, low=low, close=close)
        adx = adx_obj.adx()
        plus_di = adx_obj.adx_pos()
        minus_di = adx_obj.adx_neg()

        atr_obj = AverageTrueRange(high=high, low=low, close=close)
        atr = atr_obj.average_true_range()

        def _to_list(series: Any) -> list[float]:
            return [float(x) if not _pd.isna(x) else 0.0 for x in series]

        indicators_current = {
            "macd_line": _to_list(macd_line),
            "macd_signal": _to_list(macd_signal),
            "macd_histogram": _to_list(macd_hist),
            "ao": _to_list(ao),
            "rsi": _to_list(rsi),
            "stoch_k": _to_list(stoch_k),
            "stoch_d": _to_list(stoch_d),
            "aroon_up": _to_list(aroon_up),
            "aroon_down": _to_list(aroon_down),
            "adx": _to_list(adx),
            "plus_di": _to_list(plus_di),
            "minus_di": _to_list(minus_di),
            "atr": _to_list(atr),
        }

        vol = volume.iloc[:] if hasattr(volume, "iloc") else volume
        volume_data = [float(v) if not _pd.isna(v) else 0.0 for v in vol]

        sig_type = _signal_type_from_scanner(order_type)
        if sig_type is None:
            logger.warning("Verification mapping failed for order_type=%r", order_type)
            return None

        engine = VerificationEngine()
        result = engine.verify(VerificationInput(
            signal_type=sig_type,
            timeframe=timeframe,
            candles=candles,
            indicators_current=indicators_current,
            volume_data=volume_data,
        ))

        return {
            "score": float(np.round(result.score, 2)),
            "market_regime": result.market_regime.value,
            "consensus_score": float(np.round(result.consensus_score, 4)),
            "divergence_score": float(np.round(result.divergence_score, 4)),
            "trend_strength_score": float(np.round(result.trend_strength_score, 4)),
            "breakdown": _json_safe(result.breakdown),
        }
    except Exception as exc:
        logger.error("Verification error for %s %s: %s", order_type, timeframe, exc)
        return None


def _signal_type_from_scanner(order_type: str) -> SignalType | None:
    mapping = {
        "entry_long": SignalType.LONG,
        "add_long": SignalType.LONG,
        "entry_short": SignalType.SHORT,
        "add_short": SignalType.SHORT,
        "exit_long": SignalType.CLOSE_LONG,
        "exit_short": SignalType.CLOSE_SHORT,
    }
    return mapping.get(order_type)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float, str, bool)):
        return value
    if value is None:
        return None
    return str(value)
