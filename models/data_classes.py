from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from .enums import TradingSignal, SignalStrength, EntryTiming

@dataclass
class TradeRecommendation:
    signal: TradingSignal
    strength: SignalStrength
    reasons: List[str]
    entry_price: float = None
    # lastPrice: float = None
    stop_loss: float = None
    take_profit: float = None

@dataclass
class TimingWindow:
    timing: EntryTiming
    timeframe: str
    target_price: float = None
    confidence: float = 0.0
    conditions: List[str] = field(default_factory=list)


