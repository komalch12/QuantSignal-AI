"""
Core Domain Types, Dataclasses, and Enums for QuantSignal AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
import pandas as pd


class SignalType(str, Enum):
    """Trading signal classification."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class OrderType(str, Enum):
    """Broker order execution types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Order direction."""
    BUY = "BUY"
    SELL = "SELL"


class TimeFrame(str, Enum):
    """Candlestick timeframe intervals."""
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"


@dataclass(frozen=True)
class OHLCVData:
    """Container for validated Open-High-Low-Close-Volume price history."""
    symbol: str
    timeframe: TimeFrame
    data: pd.DataFrame  # Expected columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    def validate(self) -> bool:
        """Validates dataframe schema and non-emptiness."""
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        if self.data.empty:
            return False
        return required_cols.issubset(set(self.data.columns))


@dataclass(frozen=True)
class Signal:
    """Quantitative trading signal object produced by strategies/models."""
    symbol: str
    signal_type: SignalType
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # Range [0.0, 1.0]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Broker order structure."""
    order_id: str
    symbol: str
    quantity: int
    side: OrderSide
    order_type: OrderType
    price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    status: str = "PENDING"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Trading position state."""
    symbol: str
    quantity: int
    average_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class StockSnapshot:
    """Live NSE Stock Market tick representation."""
    symbol: str
    company_name: str
    exchange: str = "NSE"
    ltp: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DepthLevel:
    """Represents a single price level in the order book."""
    price: float = 0.0
    volume: int = 0
    orders: int = 0


@dataclass
class MarketDepthSnapshot:
    """Live Level 2 Market Depth (DOM)."""
    symbol: str
    ltp: float = 0.0
    bids: list[DepthLevel] = field(default_factory=lambda: [DepthLevel() for _ in range(5)])
    asks: list[DepthLevel] = field(default_factory=lambda: [DepthLevel() for _ in range(5)])
    timestamp: datetime = field(default_factory=datetime.now)


class LiquidityStatus(str, Enum):
    """Liquidity classification for NSE stocks based on bid/ask depth volume."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class LiquiditySnapshot:
    """Liquidity state for a single NSE stock symbol."""
    symbol: str
    company_name: str
    ltp: float
    total_bid_qty: int
    total_ask_qty: int
    status: LiquidityStatus
    timestamp: datetime = field(default_factory=datetime.now)
