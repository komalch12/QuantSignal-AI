"""
Abstract MarketDataProvider interface for QuantSignal AI.

All broker adapters and data sources must implement this interface.
This ensures the scanner, liquidity filter, and all other services
can work with any data provider without modification.

Implementors:
    - DemoMarketDataProvider  (development/demo mode)
    - FyersMarketDataProvider (production Fyers broker)
    - AngelOneMarketDataProvider (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quant_signal.core.types import MarketDepthSnapshot, StockSnapshot


class MarketDataProvider(ABC):
    """
    Abstract base class for all market data sources.

    Design Principles:
    - All providers must be interchangeable at the service layer.
    - Business logic (scanner, LTP filter, liquidity) must NEVER import
      broker-specific classes directly.
    - Demo and live providers share identical method signatures.
    """

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of the data provider (e.g. 'Demo', 'Fyers')."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """
        True if this provider is connected to real-time broker data.
        False for demo/simulation providers.
        Never return True from DemoMarketDataProvider.
        """

    # ── Symbol Universe ───────────────────────────────────────────────────────

    @abstractmethod
    def get_symbols(self) -> list[str]:
        """
        Returns the list of all available NSE equity symbols.

        Returns:
            list[str]: Symbol list (e.g. ['SBIN-EQ', 'TATASTEEL-EQ', ...])
        """

    # ── Quotes ────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_quotes(self, symbols: list[str] | None = None) -> list[StockSnapshot]:
        """
        Returns live (or demo) price quotes for the given symbols.

        Args:
            symbols: Optional list of symbols. If None, fetches all symbols.

        Returns:
            list[StockSnapshot]: Snapshot for each symbol.
        """

    # ── Market Depth (Level 2) ────────────────────────────────────────────────

    @abstractmethod
    def get_market_depth(self, symbol: str) -> MarketDepthSnapshot:
        """
        Returns Level 2 order book depth for a specific symbol.

        Args:
            symbol: NSE equity symbol (e.g. 'SBIN-EQ').

        Returns:
            MarketDepthSnapshot: 5-level bid/ask order book snapshot.
        """

    # ── Historical Data ───────────────────────────────────────────────────────

    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """
        Returns OHLCV historical candlestick data.

        Args:
            symbol: NSE equity symbol.
            timeframe: Candle timeframe string (e.g. '1m', '5m', '1d').
            days: Number of historical days to fetch.

        Returns:
            pd.DataFrame: Columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_provider_info(self) -> dict[str, Any]:
        """Returns a safe metadata dict for UI diagnostics (no credentials)."""
        return {
            "provider_name": self.provider_name,
            "is_live": self.is_live,
            "data_mode": "LIVE" if self.is_live else "DEVELOPMENT / DEMO",
        }
