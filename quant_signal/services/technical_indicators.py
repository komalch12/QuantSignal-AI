"""
Technical Indicators Service for QuantSignal AI.

Phase 5 Implementation: SMMA (Smoothed Moving Average) 20 & 120 Calculation Engine.

Calculates SMMA 20 and SMMA 120 from historical price series in a fully
broker-independent manner using pure pandas and numpy.

Formula:
  Initial value (at index period - 1): Simple Moving Average (SMA) over the first 'period' elements.
  Subsequent values (for i >= period):
      SMMA[i] = (SMMA[i-1] * (period - 1) + Price[i]) / period
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

from quant_signal.logger import get_logger
from quant_signal.providers.base import MarketDataProvider
from quant_signal.providers.demo_provider import DemoMarketDataProvider, _DEMO_TIMESTAMP

logger = get_logger(__name__)


def calculate_smma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Smoothed Moving Average (SMMA / Wilder's Smoothing) for a price series.

    Formula:
      First valid SMMA at index (period - 1) = SMA(series[:period])
      SMMA[i] = (SMMA[i-1] * (period - 1) + series[i]) / period for i >= period

    Args:
        series: Pandas Series of numeric price data (e.g. 'close' or 'ltp').
        period: Lookback window period (e.g. 20, 120). Must be > 0.

    Returns:
        pd.Series: Calculated SMMA series matching original index.
                  Indices < period - 1 will contain NaN.

    Raises:
        ValueError: If period <= 0.
    """
    if period <= 0:
        raise ValueError(f"Period must be a positive integer, got {period}.")

    if series is None or series.empty:
        return pd.Series(dtype=float)

    # Convert to numeric float Series (handling any bad string inputs)
    clean_series = pd.to_numeric(series, errors="coerce")
    n = len(clean_series)

    if n < period:
        # Insufficient historical data
        return pd.Series([np.nan] * n, index=series.index)

    values = clean_series.to_numpy(dtype=float)
    smma_values = np.full(n, np.nan, dtype=float)

    # 1. First SMMA value is simple average over first 'period' bars
    initial_sma = np.mean(values[:period])
    smma_values[period - 1] = initial_sma

    # 2. Recursive calculation for remaining bars
    prev = initial_sma
    factor = period - 1
    for i in range(period, n):
        if np.isnan(values[i]):
            curr = prev
        else:
            curr = (prev * factor + values[i]) / period
        smma_values[i] = curr
        prev = curr

    return pd.Series(smma_values, index=series.index)


def determine_crossover_signal(
    prev_20: float,
    prev_120: float,
    curr_20: float,
    curr_120: float,
) -> str:
    """
    Determines SMMA 20 / SMMA 120 crossover signal status.

    Allowed Return Values:
      - 'BUY_CROSSOVER': if prev_20 <= prev_120 and curr_20 > curr_120
      - 'SELL_CROSSOVER': if prev_20 >= prev_120 and curr_20 < curr_120
      - 'NONE': otherwise (no crossover, equal values, or NaN data)
    """
    if np.isnan(prev_20) or np.isnan(prev_120) or np.isnan(curr_20) or np.isnan(curr_120):
        return "NONE"

    if prev_20 <= prev_120 and curr_20 > curr_120:
        return "BUY_CROSSOVER"
    elif prev_20 >= prev_120 and curr_20 < curr_120:
        return "SELL_CROSSOVER"
    return "NONE"


class TechnicalIndicatorService:
    """
    Broker-independent technical indicator service.

    Calculates SMMA 20, SMMA 120, Trend classification, Distance %, and Crossover status
    from underlying market data providers.
    """

    def __init__(self, provider: MarketDataProvider | None = None) -> None:
        self.provider: MarketDataProvider = provider or DemoMarketDataProvider()

    def calculate_stock_indicators(
        self,
        symbol: str,
        timeframe: str = "1d",
        days: int = 250,
    ) -> dict[str, Any]:
        """
        Calculates SMMA 20 and SMMA 120 for a given stock symbol.

        Args:
            symbol: Ticker symbol (e.g. 'SUZLON-EQ').
            timeframe: Candle timeframe string (default '1d').
            days: Historical days to fetch (default 250 days to ensure > 120 trading bars).

        Returns:
            dict with fields:
                - symbol: Ticker symbol
                - ltp: Latest Price
                - smma_20: Calculated SMMA 20 value
                - smma_120: Calculated SMMA 120 value
                - trend: 'Bullish 🟢' if SMMA20 > SMMA120 else 'Bearish 🔴'
                - distance_pct: Percentage distance between SMMA20 and SMMA120
                - crossover: 'BUY_CROSSOVER', 'SELL_CROSSOVER', or 'NONE'
                - history_df: Full historical DataFrame with columns ['timestamp', 'close', 'smma_20', 'smma_120']
                - has_sufficient_data: True if data length >= 120
        """
        df = self.provider.get_historical_data(symbol=symbol, timeframe=timeframe, days=days)

        if df.empty or "close" not in df.columns or len(df) < 120:
            return {
                "symbol": symbol,
                "ltp": 0.0,
                "smma_20": np.nan,
                "smma_120": np.nan,
                "trend": "Insufficient Data",
                "distance_pct": 0.0,
                "crossover": "NONE",
                "history_df": pd.DataFrame(),
                "has_sufficient_data": False,
            }

        # Calculate SMMA 20 & 120
        df = df.copy()
        df["smma_20"] = calculate_smma(df["close"], period=20)
        df["smma_120"] = calculate_smma(df["close"], period=120)

        latest_row = df.iloc[-1]
        ltp = float(latest_row["close"])
        smma_20 = float(latest_row["smma_20"])
        smma_120 = float(latest_row["smma_120"])

        if np.isnan(smma_20) or np.isnan(smma_120):
            trend = "Insufficient Data"
            dist_pct = 0.0
            crossover = "NONE"
        else:
            # Trend logic: SMMA20 > SMMA120 = Bullish, SMMA20 < SMMA120 = Bearish
            trend = "Bullish 🟢" if smma_20 > smma_120 else "Bearish 🔴"
            dist_pct = ((smma_20 - smma_120) / smma_120) * 100.0 if smma_120 != 0 else 0.0

            valid_rows = df.dropna(subset=["smma_20", "smma_120"])
            if len(valid_rows) >= 2:
                prev_row = valid_rows.iloc[-2]
                curr_row = valid_rows.iloc[-1]
                crossover = determine_crossover_signal(
                    prev_20=float(prev_row["smma_20"]),
                    prev_120=float(prev_row["smma_120"]),
                    curr_20=float(curr_row["smma_20"]),
                    curr_120=float(curr_row["smma_120"]),
                )
            else:
                crossover = "NONE"

        return {
            "symbol": symbol,
            "ltp": ltp,
            "smma_20": smma_20,
            "smma_120": smma_120,
            "trend": trend,
            "distance_pct": dist_pct,
            "crossover": crossover,
            "history_df": df,
            "has_sufficient_data": True,
        }

    def get_indicators_dataframe(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """
        Calculates SMMA 20 and SMMA 120 across multiple symbols and returns a summary DataFrame.

        Required Columns:
        symbol, company_name, ltp, smma_20, smma_120, trend, distance_pct, crossover, timestamp
        """
        target_symbols = symbols or self.provider.get_symbols()
        rows = []
        ts_str = _DEMO_TIMESTAMP.strftime("%H:%M:%S") + " (Simulated Demo Time)"

        # Get metadata mapping from provider if available
        company_map = {}
        if hasattr(self.provider, "_data"):
            company_map = {k: v.get("company", k) for k, v in self.provider._data.items()}

        for sym in target_symbols:
            res = self.calculate_stock_indicators(symbol=sym, days=250)
            comp_name = company_map.get(sym, sym)

            rows.append({
                "symbol": sym,
                "company_name": comp_name,
                "ltp": res["ltp"],
                "smma_20": res["smma_20"],
                "smma_120": res["smma_120"],
                "trend": res["trend"],
                "distance_pct": res["distance_pct"],
                "crossover": res["crossover"],
                "timestamp": ts_str,
            })

        if not rows:
            return pd.DataFrame(columns=[
                "symbol", "company_name", "ltp", "smma_20", "smma_120", "trend", "distance_pct", "crossover", "timestamp"
            ])

        df = pd.DataFrame(rows)
        return df.sort_values("symbol").reset_index(drop=True)

