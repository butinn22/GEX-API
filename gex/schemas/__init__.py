"""Pydantic v2 — схемы сериализации для API.

Разбит на доменные подмодули для масштабируемости.
Все имена реэкспортируются для обратной совместимости.
"""

from ._base import _Base, OptionRowIn, ChainIn
from .gex_analysis import (
    StrikeProfileOut, GEXProfileOut, ResistanceLevelOut, SupportLevelOut,
    GEXKeyLevelsOut, GEXSummaryOut, GEXAnalysisOut, profile_to_schema,
)
from .ta_analysis import (
    TAIndicatorsOut, MomentumOut, TrendOut, ReversalProbOut,
    MultiTFConfirmationOut, DivergenceItemOut, DivergenceSummaryOut,
    TimeframeOut, TASRLevelOut, TAReversalSummaryOut, TASummaryOut,
    TAAnalysisOut, timeframe_to_schema,
)
from .scanner import ScanRecordOut, ScanReportOut
from .signals import GEXContextOut, SignalRecordOut, CurrentSignalOut, SignalAnalysisOut, TrendRegimeOut, PositionOut
from .trendlines import (
    TrendlineOut, FractalStructureOut, TrendlineTimeframeOut,
    TrendlineSummaryOut, TrendlineAnalysisOut, trendline_to_schema,
)
from .macd_trend import (
    MacdTrendTimeframeOut, MacdTrendSummaryOut, MacdTrendAnalysisOut,
    macd_trend_to_schema,
)
from .hybrid_trend import (
    HybridStructureResponse, StructurePoint, StructureEvent, ZigzagPoint,
)
from .ohlcv import OHLCVBarOut, OHLCVOut
from .extended_schemas import (
    ExtendedGEXAnalysisOut,
    ExtendedStrikeOut,
    extended_report_to_schema,
)
from .auto_coverage import AutoCoverageOut, auto_coverage_to_schema
