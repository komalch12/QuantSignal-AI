"""
Abstract Interfaces (Base Contracts) for QuantSignal AI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

from quant_signal.core.types import OHLCVData, Order, OrderSide, OrderType, Position, Signal, TimeFrame


class IBrokerAdapter(ABC):
    """Abstract Interface for Broker integrations (e.g. Fyers API v3)."""

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the broker API."""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: str,
        end_date: str,
    ) -> OHLCVData:
        """Fetch historical candlestick data."""
        pass

    @abstractmethod
    def subscribe_live_ticks(self, symbols: list[str], callback: Any) -> None:
        """Connect WebSocket and stream real-time price feeds."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType,
        price: float = 0.0,
    ) -> Order:
        """Place an order with the broker."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Fetch current open positions."""
        pass


class IIndicatorEngine(ABC):
    """Abstract Interface for Technical Analysis indicator calculations."""

    @abstractmethod
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes and appends technical indicators to the candlestick DataFrame."""
        pass


class IModelEngine(ABC):
    """Abstract Interface for Machine Learning models using Scikit-Learn & Joblib."""

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Loads a pre-trained model file from disk."""
        pass

    @abstractmethod
    def save_model(self, model_path: str) -> bool:
        """Saves current trained model to disk."""
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generates ML model predictions/probabilities from feature matrix."""
        pass


class IStrategyEngine(ABC):
    """Abstract Interface for Quantitative Trading Strategy signal generators."""

    @abstractmethod
    def generate_signal(self, data: OHLCVData) -> Signal:
        """Evaluates price data & indicators to produce a trading signal."""
        pass
