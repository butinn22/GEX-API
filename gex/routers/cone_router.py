"""Volatility Cone: /cone/{ticker} — quarterly volatility cone + VWAP Price Channel."""
from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from gex.application.ohlcv_service import fetch_ohlcv
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.domain.volatility_cone import DEFAULT_LOOKBACK_DAYS, compute_all

from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["volatility-cone"])


# ── Response schema ───────────────────────────────────────────────────
class ConeDataPoint(BaseModel):
    """Single bar of cone data (for lightweight array-based transport)."""

    timestamp: str = Field(..., description="ISO-8601 timestamp")
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_new_quarter: bool
    median_price: float | None = None
    vwap: float | None = None
    qema21: float | None = None
    upper_1sd: float | None = None
    lower_1sd: float | None = None
    upper_2sd: float | None = None
    lower_2sd: float | None = None
    upper_1sd_mr: float | None = None
    lower_1sd_mr: float | None = None
    upper_2sd_mr: float | None = None
    lower_2sd_mr: float | None = None
    upper_2sd_corr: float | None = None
    lower_2sd_corr: float | None = None
    upper_2sd_mr_corr: float | None = None
    lower_2sd_mr_corr: float | None = None
    corr_deviation_pct: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    vpc_upper: float | None = None
    vpc_lower: float | None = None
    vpc_dir: int = 0


class QuarterInfo(BaseModel):
    """Metadata for one detected quarter."""

    index: int = Field(..., description="0-based quarter number")
    start_bar: int = Field(..., description="First bar index in data array")
    end_bar: int = Field(..., description="Last bar index in data array")
    start_date: str = Field(..., description="ISO date of first bar")
    end_date: str = Field(..., description="ISO date of last bar")
    bar_count: int = Field(..., description="Number of bars in this quarter")
    start_price: float = Field(..., description="Open price at quarter start")
    end_price: float = Field(..., description="Close price at quarter end")


class ConeResponse(BaseModel):
    """Full volatility cone response for a ticker."""

    ticker: str
    timeframe: str
    bars_count: int
    quarters_detected: int
    quarters: list[QuarterInfo] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)
    data: list[ConeDataPoint]


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get("/cone/{ticker}", response_model=ConeResponse)
def get_volatility_cone(
    ticker: str,
    timeframe: str = Query("1d", description="Таймфрейм: 1h, 2h, 4h, 1d"),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS, ge=100, le=2000),
    limit: int = Query(1000, ge=50, le=5000, description="Максимум баров"),
    sd1_mult: float = Query(1.0, ge=0.1, le=5.0),
    sd2_mult: float = Query(2.0, ge=0.1, le=5.0),
    vwap_influence: float = Query(0.2, ge=0.0, le=1.0),
    rsi_influence: float = Query(0.15, ge=0.0, le=1.0),
    ema_len: int = Query(21, ge=5, le=80),
    bb_mult: float = Query(2.0, ge=0.5, le=4.0),
    carry_weight: float = Query(0.62, ge=0.0, le=1.0),
    rsi_1sd_boost: float = Query(1.2, ge=0.5, le=2.0),
    use_correction: bool = Query(True),
    correction_pct: float = Query(33.0, ge=0.0, le=100.0),
    vpc_length: int = Query(20, ge=5, le=100),
) -> ConeResponse:
    """Рассчитать квартальный конус волатильности + VWAP Price Channel.

    Возвращает покадровый массив со всеми границами конуса, BB, VWAP, VPC.
    Данные готовы для отрисовки на фронтенде без дополнительных расчётов.
    """

    def _compute() -> ConeResponse:
        ticker_clean = ticker.strip().upper()

        # Fetch OHLCV
        import pandas as pd
        ohlcv_out = fetch_ohlcv(ticker_clean, timeframe=timeframe, limit=limit)

        # Convert OHLCVOut → DataFrame
        records = []
        timestamps = pd.to_datetime([bar.t for bar in ohlcv_out.bars], utc=True)
        for bar in ohlcv_out.bars:
            records.append({
                "open": bar.o,
                "high": bar.h,
                "low": bar.l,
                "close": bar.c,
                "volume": bar.v,
            })
        df = pd.DataFrame(records, index=pd.DatetimeIndex(timestamps))

        if df is None or len(df) == 0:
            raise ValueError(f"Нет данных для '{ticker_clean}' на таймфрейме '{timeframe}'")

        # Compute cone
        cone_params = {
            "lookback_days": lookback_days,
            "rsi_length": 14,
            "sd1_mult": sd1_mult,
            "sd2_mult": sd2_mult,
            "vwap_influence": vwap_influence,
            "rsi_influence": rsi_influence,
            "ema_len": ema_len,
            "bb_mult": bb_mult,
            "carry_weight": carry_weight,
            "rsi_1sd_boost": rsi_1sd_boost,
            "use_correction": use_correction,
            "correction_pct": correction_pct,
        }
        result = compute_all(df, cone_params=cone_params, vpc_length=vpc_length)

        # Build response points
        data_points: list[ConeDataPoint] = []
        for idx, row in result.iterrows():
            dp = ConeDataPoint(
                timestamp=idx.isoformat(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                is_new_quarter=bool(row.get('is_new_quarter', False)),
                median_price=_opt_float(row.get('median_price')),
                vwap=_opt_float(row.get('vwap')),
                qema21=_opt_float(row.get('qema21')),
                upper_1sd=_opt_float(row.get('upper_1sd')),
                lower_1sd=_opt_float(row.get('lower_1sd')),
                upper_2sd=_opt_float(row.get('upper_2sd')),
                lower_2sd=_opt_float(row.get('lower_2sd')),
                upper_1sd_mr=_opt_float(row.get('upper_1sd_mr')),
                lower_1sd_mr=_opt_float(row.get('lower_1sd_mr')),
                upper_2sd_mr=_opt_float(row.get('upper_2sd_mr')),
                lower_2sd_mr=_opt_float(row.get('lower_2sd_mr')),
                upper_2sd_corr=_opt_float(row.get('upper_2sd_corr')),
                lower_2sd_corr=_opt_float(row.get('lower_2sd_corr')),
                upper_2sd_mr_corr=_opt_float(row.get('upper_2sd_mr_corr')),
                lower_2sd_mr_corr=_opt_float(row.get('lower_2sd_mr_corr')),
                corr_deviation_pct=_opt_float(row.get('corr_deviation_pct')),
                bb_upper=_opt_float(row.get('bb_upper')),
                bb_lower=_opt_float(row.get('bb_lower')),
                vpc_upper=_opt_float(row.get('vpc_upper')),
                vpc_lower=_opt_float(row.get('vpc_lower')),
                vpc_dir=int(row.get('vpc_dir', 0)),
            )
            data_points.append(dp)

        quarters_count = int(result['is_new_quarter'].sum())

        # Build quarter metadata for frontend slider
        quarter_boundaries: list[QuarterInfo] = []
        q_start: int | None = None
        q_idx = 0
        for i, (idx, row) in enumerate(result.iterrows()):
            if bool(row.get('is_new_quarter', False)):
                if q_start is not None and i > q_start:
                    end_row = result.iloc[i - 1]
                    quarter_boundaries.append(QuarterInfo(
                        index=q_idx,
                        start_bar=q_start,
                        end_bar=i - 1,
                        start_date=result.index[q_start].isoformat(),
                        end_date=result.index[i - 1].isoformat(),
                        bar_count=i - q_start,
                        start_price=float(result.iloc[q_start]['open']),
                        end_price=float(end_row['close']),
                    ))
                    q_idx += 1
                q_start = i
        # Last quarter (from last new_quarter to end)
        if q_start is not None and q_start < len(result):
            last_row = result.iloc[-1]
            quarter_boundaries.append(QuarterInfo(
                index=q_idx,
                start_bar=q_start,
                end_bar=len(result) - 1,
                start_date=result.index[q_start].isoformat(),
                end_date=result.index[-1].isoformat(),
                bar_count=len(result) - q_start,
                start_price=float(result.iloc[q_start]['open']),
                end_price=float(last_row['close']),
            ))

        return ConeResponse(
            ticker=ticker_clean,
            timeframe=timeframe,
            bars_count=len(data_points),
            quarters_detected=quarters_count,
            quarters=quarter_boundaries,
            params={
                "lookback_days": lookback_days,
                "sd1_mult": sd1_mult,
                "sd2_mult": sd2_mult,
                "vwap_influence": vwap_influence,
                "rsi_influence": rsi_influence,
                "ema_len": ema_len,
                "bb_mult": bb_mult,
                "use_correction": use_correction,
                "correction_pct": correction_pct,
            },
            data=data_points,
        )

    _key = cache_key("res", "cone", ticker, timeframe, lookback_days, limit, sd1_mult, sd2_mult,
                 vwap_influence, rsi_influence, ema_len, bb_mult, carry_weight, rsi_1sd_boost,
                 use_correction, correction_pct, vpc_length)
    return result_cache.get(_key, 600, lambda: handle(_compute, error_src="yfinance/Bybit"))


def _opt_float(val: object) -> float | None:
    """Convert to Optional[float], mapping NaN to None."""
    import math

    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
