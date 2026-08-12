"""
Quantitative Signal Strategy Engine for QuantSignal AI.

Combines technical indicator signals and machine learning model scores to output
high-confidence BUY / SELL / HOLD trade signals.
"""

from __future__ import annotations

from datetime import datetime

from quant_signal.core.base import IIndicatorEngine, IModelEngine, IStrategyEngine
from quant_signal.core.types import OHLCVData, Signal, SignalType
from quant_signal.exceptions import StrategyError
from quant_signal.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class QuantSignalStrategy(IStrategyEngine):
    """Production Quantitative Trading Strategy combining rule-based & ML logic."""

    def __init__(
        self,
        indicator_engine: IIndicatorEngine,
        model_engine: IModelEngine | None = None,
        min_confidence_threshold: float = 0.65,
    ) -> None:
        """Initialize strategy dependencies.

        Args:
            indicator_engine: Technical indicators engine implementation.
            model_engine: Machine learning scoring engine implementation (optional).
            min_confidence_threshold: Minimum confidence score required to trigger actionable signal.
        """
        self.indicator_engine: IIndicatorEngine = indicator_engine
        self.model_engine: IModelEngine | None = model_engine
        self.min_confidence_threshold: float = min_confidence_threshold

    @log_execution_time()
    def generate_signal(self, data: OHLCVData) -> Signal:
        """Evaluates price data, technical indicators, and ML model outputs to generate a signal.

        Args:
            data: OHLCV candlestick container.

        Returns:
            Signal: Structured trading signal object.

        Raises:
            StrategyError: If market data validation fails.
        """
        if not data.validate():
            logger.error(f"Invalid OHLCV market data supplied for symbol: {data.symbol}")
            raise StrategyError(f"Market data validation failed for symbol '{data.symbol}'")

        logger.info(f"Evaluating quantitative signal logic for symbol '{data.symbol}'...")

        try:
            # 1. Add technical indicators
            enriched_df = self.indicator_engine.add_indicators(data.data)
            latest_price = float(enriched_df["close"].iloc[-1]) if not enriched_df.empty else 0.0

            # 2. Crossover signal evaluation
            crossover_type = "NONE"
            if len(enriched_df) >= 2 and "smma_20" in enriched_df.columns and "smma_120" in enriched_df.columns:
                prev_20 = enriched_df["smma_20"].iloc[-2]
                prev_120 = enriched_df["smma_120"].iloc[-2]
                curr_20 = enriched_df["smma_20"].iloc[-1]
                curr_120 = enriched_df["smma_120"].iloc[-1]
                if pd.notna(prev_20) and pd.notna(prev_120) and pd.notna(curr_20) and pd.notna(curr_120):
                    if prev_20 <= prev_120 and curr_20 > curr_120:
                        crossover_type = "BUY_CROSSOVER"
                    elif prev_20 >= prev_120 and curr_20 < curr_120:
                        crossover_type = "SELL_CROSSOVER"

            # 3. AI/ML confidence score evaluation
            confidence = 0.5
            if self.model_engine and hasattr(self.model_engine, "predict_proba"):
                try:
                    from quant_signal.services.ml_signal_service import extract_features_from_history
                    feat_df = extract_features_from_history(enriched_df)
                    if not feat_df.empty:
                        prob_series = self.model_engine.predict_proba(feat_df.iloc[[-1]])
                        if not prob_series.empty:
                            confidence = float(prob_series.iloc[-1])
                except Exception as ml_err:
                    logger.warning(f"Strategy ML confidence evaluation fallback: {ml_err}")

            # 4. Map signal classification
            sig_type = SignalType.HOLD
            if confidence >= self.min_confidence_threshold or crossover_type == "BUY_CROSSOVER":
                sig_type = SignalType.BUY
            elif confidence <= (1.0 - self.min_confidence_threshold) or crossover_type == "SELL_CROSSOVER":
                sig_type = SignalType.SELL

            signal = Signal(
                symbol=data.symbol,
                signal_type=sig_type,
                price=latest_price,
                timestamp=datetime.now(),
                confidence=round(confidence, 4),
                metadata={
                    "timeframe": data.timeframe.value,
                    "strategy_name": "QuantSignal_Hybrid_v1",
                    "crossover": crossover_type,
                },
            )
            logger.info(f"Signal generated: {signal.signal_type.value} @ {signal.price} (Confidence: {signal.confidence:.2f})")
            return signal

        except StrategyError:
            raise
        except Exception as err:
            logger.error(f"Strategy signal generation failure for '{data.symbol}': {err}")
            raise StrategyError(f"Error evaluating strategy for '{data.symbol}': {err}") from err
