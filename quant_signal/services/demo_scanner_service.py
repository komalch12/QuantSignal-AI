"""
Demo Stock Scanner Service for QuantSignal AI.

Wraps DemoMarketDataProvider to provide the same interface as
StockScannerService, but without any broker authentication, WebSocket,
or network calls.

IMPORTANT:
- This service returns FIXED SAMPLE DATA only.
- It is NOT connected to any live market feed.
- Clearly labelled DEVELOPMENT / DEMO DATA MODE in all UI.
- The DemoMarketDataProvider can later be replaced by FyersMarketDataProvider
  or AngelOneMarketDataProvider without changing any scanner/business logic.
"""

from __future__ import annotations

from typing import Any
import pandas as pd

from quant_signal.logger import get_logger
from quant_signal.providers.demo_provider import DemoMarketDataProvider, _DEMO_TIMESTAMP

logger = get_logger(__name__)


class DemoScannerService:
    """
    Scanner service backed by DemoMarketDataProvider.

    Provides the same public interface as StockScannerService so that
    all UI views can work with either service via duck typing.

    Phase 4 Filters:
    - LTP Filter: 30 <= LTP <= 500 (assignment requirement).
    - Liquidity Filter: Bid Quantity > 1,000,000 AND Ask Quantity > 1,000,000.
    """

    DATA_MODE: str = "DEVELOPMENT / DEMO"
    IS_LIVE: bool = False

    def __init__(
        self,
        min_price: float = 30.0,
        max_price: float = 500.0,
        min_bid_qty: int = 1_000_000,
        min_ask_qty: int = 1_000_000,
    ) -> None:
        self.min_price = min_price
        self.max_price = max_price
        self.min_bid_qty = min_bid_qty
        self.min_ask_qty = min_ask_qty
        self.provider = DemoMarketDataProvider()

        # Pre-build DataFrames once (deterministic — no refresh needed)
        self._all_df: pd.DataFrame = self._build_all_dataframe()
        self._ltp_filtered_df: pd.DataFrame = self._apply_ltp_filter(self._all_df)
        self._liquidity_filtered_df: pd.DataFrame = self._apply_liquidity_filter(self._ltp_filtered_df)

        logger.info(
            f"DemoScannerService initialised: "
            f"{len(self._all_df)} total stocks, "
            f"{len(self._ltp_filtered_df)} pass ₹{min_price:.0f}–₹{max_price:.0f} LTP filter, "
            f"{len(self._liquidity_filtered_df)} pass Bid/Ask > 1M Liquidity filter."
        )

    # ── DataFrame Builders ─────────────────────────────────────────────────────

    def _build_all_dataframe(self) -> pd.DataFrame:
        """Builds a Pandas DataFrame of all 39 demo stocks with all required fields."""
        rows = []
        ts_str = _DEMO_TIMESTAMP.strftime("%H:%M:%S") + " (Simulated Demo Time)"
        for symbol, row in self.provider._data.items():
            rows.append({
                "symbol":       symbol,
                "company_name": row["company"],
                "exchange":     "NSE",
                "ltp":          float(row["ltp"]),
                "timestamp":    ts_str,
                "bid_price":    float(row["bid"]),
                "bid_quantity": int(row["bid_qty"]),
                "ask_price":    float(row["ask"]),
                "ask_quantity": int(row["ask_qty"]),
                "volume":       int(row["vol"]),
            })

        if not rows:
            return pd.DataFrame(columns=[
                "symbol", "company_name", "exchange", "ltp", "timestamp",
                "bid_price", "bid_quantity", "ask_price", "ask_quantity", "volume",
            ])

        df = pd.DataFrame(rows)
        df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
        return df.sort_values("symbol").reset_index(drop=True)

    def _apply_ltp_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the assignment LTP filter: 30 <= LTP <= 500."""
        if df.empty:
            return df
        filtered = df[(df["ltp"] >= self.min_price) & (df["ltp"] <= self.max_price)]
        return filtered.sort_values("ltp").reset_index(drop=True)

    def _apply_liquidity_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the Phase 4 liquidity filter: Bid Qty > 1,000,000 AND Ask Qty > 1,000,000."""
        if df.empty:
            return df
        filtered = df[
            (df["bid_quantity"] > self.min_bid_qty) &
            (df["ask_quantity"] > self.min_ask_qty)
        ]
        return filtered.sort_values("ltp").reset_index(drop=True)

    # ── Summary Metrics ────────────────────────────────────────────────────────

    def get_summary_metrics(self) -> dict[str, int]:
        """
        Returns Phase 4 summary metrics across the universe:
        - Total Stocks
        - LTP Matching (30 <= LTP <= 500)
        - Bid Quantity Matching (Bid Qty > 1,000,000)
        - Ask Quantity Matching (Ask Qty > 1,000,000)
        - Liquidity Matching (LTP filter AND Bid Qty > 1M AND Ask Qty > 1M)
        """
        total_stocks = len(self._all_df)
        ltp_matching = len(self._ltp_filtered_df)
        bid_qty_matching = len(self._all_df[self._all_df["bid_quantity"] > self.min_bid_qty])
        ask_qty_matching = len(self._all_df[self._all_df["ask_quantity"] > self.min_ask_qty])
        liquidity_matching = len(self._liquidity_filtered_df)

        return {
            "total_stocks":        total_stocks,
            "ltp_matching":        ltp_matching,
            "bid_qty_matching":    bid_qty_matching,
            "ask_qty_matching":    ask_qty_matching,
            "liquidity_matching":  liquidity_matching,
        }

    # ── Public Interface ───────────────────────────────────────────────────────

    def get_all_stocks_dataframe(self) -> pd.DataFrame:
        """Returns all demo stocks as a Pandas DataFrame."""
        return self._all_df.copy()

    def get_filtered_stocks_dataframe(self) -> pd.DataFrame:
        """
        Returns demo stocks filtered by LTP: ₹30 <= LTP <= ₹500.

        Columns: symbol, company_name, exchange, ltp, timestamp
        """
        return self._ltp_filtered_df.copy()

    def get_market_depth_dataframe(self) -> pd.DataFrame:
        """
        Returns Market Depth summary DataFrame for all stocks.

        Required Phase 4 Columns:
        Symbol, Company Name, LTP, Bid Price, Bid Quantity, Ask Price, Ask Quantity, Timestamp
        """
        cols = ["symbol", "company_name", "ltp", "bid_price", "bid_quantity", "ask_price", "ask_quantity", "timestamp"]
        return self._all_df[cols].copy()

    def get_liquidity_filtered_dataframe(self) -> pd.DataFrame:
        """
        Returns demo stocks satisfying both LTP filter (30-500) AND:
        Bid Quantity > 1,000,000 AND Ask Quantity > 1,000,000.

        Required Columns:
        Symbol, Company Name, LTP, Bid Price, Bid Quantity, Ask Price, Ask Quantity, Timestamp
        """
        cols = ["symbol", "company_name", "ltp", "bid_price", "bid_quantity", "ask_price", "ask_quantity", "timestamp"]
        return self._liquidity_filtered_df[cols].copy()

    def get_status_summary(self) -> dict[str, Any]:
        """Returns status summary dict compatible with UI."""
        metrics = self.get_summary_metrics()
        return {
            "data_mode":          self.DATA_MODE,
            "is_live":            self.IS_LIVE,
            "connection_state":   "DEMO",
            "provider_name":      self.provider.provider_name,
            "universe_count":     metrics["total_stocks"],
            "live_received":      0,
            "valid_ltp":          metrics["total_stocks"],
            "filtered_count":     metrics["ltp_matching"],
            "liquidity_count":    metrics["liquidity_matching"],
            "bid_qty_matching":   metrics["bid_qty_matching"],
            "ask_qty_matching":   metrics["ask_qty_matching"],
            "min_price":          self.min_price,
            "max_price":          self.max_price,
            "min_bid_qty":        self.min_bid_qty,
            "min_ask_qty":        self.min_ask_qty,
            "last_error":         None,
            "fyers_status":       "Not Configured",
            "data_provider":      "Demo",
            "live_market_data":   "Unavailable",
        }

    def get_market_depth(self, symbol: str):
        """Delegates to DemoMarketDataProvider for 5-level order book depth data."""
        return self.provider.get_market_depth(symbol)

    def get_historical_data(
        self, symbol: str, timeframe: str = "1d", days: int = 180
    ) -> pd.DataFrame:
        """Delegates to DemoMarketDataProvider for deterministic OHLCV history."""
        return self.provider.get_historical_data(symbol, timeframe, days)
