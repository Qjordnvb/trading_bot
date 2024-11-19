from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from .enums import TradingSignal, SignalStrength, EntryTiming

@dataclass
class TradeRecommendation:
    signal: TradingSignal
    strength: SignalStrength
    reasons: List[str]
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None  # Mantener para compatibilidad
    take_profits: Optional[List[dict]] = None

@dataclass
class TimingWindow:
    timing: EntryTiming
    timeframe: str
    target_price: float = None
    confidence: float = 0.0
    conditions: List[str] = field(default_factory=list)


