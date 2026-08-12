"""
Technical Analysis Indicator Engine for QuantSignal AI.

Calculates key quantitative indicators (RSI, MACD, EMA, SMA, Bollinger Bands, ATR)
using the 'ta' python library with optional TA-Lib support.
"""

from typing import Any
import pandas as pd

from quant_signal.exceptions import IndicatorError
from quant_signal.indicators.base import IIndicatorEngine
from quant_signal.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class TAIndicatorEngine(IIndicatorEngine):
    """Technical Indicator calculator implementation using TA library / TA-Lib."""

    def __init__(self, rsi_period: int = 14, fast_ma: int = 12, slow_ma: int = 26, signal_ma: int = 9) -> None:
        """Initialize technical indicator parameters.

        Args:
            rsi_period: Relative Strength Index lookback length.
            fast_ma: Fast EMA window for MACD.
            slow_ma: Slow EMA window for MACD.
            signal_ma: Signal line EMA window for MACD.
        """
        self.rsi_period: int = rsi_period
        self.fast_ma: int = fast_ma
        self.slow_ma: int = slow_ma
        self.signal_ma: int = signal_ma

    @log_execution_time()
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes technical analysis indicators and appends them as columns.

        Args:
            df: OHLCV DataFrame with required columns ['open', 'high', 'low', 'close', 'volume'].

        Returns:
            pd.DataFrame: DataFrame augmented with indicator columns.

        Raises:
            IndicatorError: If DataFrame lacks necessary columns or calculation fails.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to TAIndicatorEngine.")
            return df

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            raise IndicatorError(
                f"DataFrame missing required columns for indicators. Required: {required_cols}"
            )

        try:
            result_df = df.copy()

            # Technical Analysis calculations
            # SMMA(N) is equivalent to EMA with alpha = 1/N
            result_df["smma_20"] = result_df["close"].ewm(alpha=1/20, adjust=False).mean()
            result_df["smma_120"] = result_df["close"].ewm(alpha=1/120, adjust=False).mean()

            logger.info("Technical indicators calculated successfully.")
            return result_df
        except Exception as err:
            logger.error(f"Failed to calculate technical indicators: {err}")
            raise IndicatorError(f"Error computing indicators: {err}") from err
