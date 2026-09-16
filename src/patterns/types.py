from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timezone

@dataclass
class FVG:
    direction: int  # +1 Bullish (Demand Inbalance), -1 Bearish (Supply Inbalance)
    top: float
    bottom: float
    mid: float      # Consequent Encroachment (50%)
    candle_idx: int
    timestamp: Optional[datetime] = None
    mitigated: bool = False
    mitigation_idx: Optional[int] = None

@dataclass
class OrderBlock:
    direction: int  # +1 Bullish (Demand OB), -1 Bearish (Supply OB)
    top: float
    bottom: float
    mid: float      # Mean Threshold (50%)
    candle_idx: int
    timestamp: Optional[datetime] = None
    mitigated: bool = False
    mitigation_idx: Optional[int] = None
    invalidated: bool = False

@dataclass
class LiquiditySweep:
    direction: int  # +1 Bullish (Sweep de Mínimos / SSL), -1 Bearish (Sweep de Máximos / BSL)
    swept_level: float
    wick_extreme: float
    candle_idx: int
    timestamp: Optional[datetime] = None

@dataclass
class TradeSetup:
    symbol: str
    direction: str  # 'LONG' | 'SHORT'
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk_distance: float
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float
    confidence_score: float
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    chart_path: Optional[str] = None
