"""
Unit Tests for SMMA Crossover Detection Service (Phase 6).

Tests:
1. BUY signal detection (prev_20 <= prev_120 and curr_20 > curr_120).
2. SELL signal detection (prev_20 >= prev_120 and curr_20 < curr_120).
3. NO SIGNAL behavior.
4. CrossoverService integration with TechnicalIndicatorService and DemoMarketDataProvider.
"""

from datetime import datetime, timedelta
import unittest
import pandas as pd
import numpy as np

from quant_signal.services.crossover_service import (
    CrossoverService,
    detect_crossovers_in_dataframe,
)


class TestCrossoverDetection(unittest.TestCase):
    """Test suite for crossover detection logic."""

    def test_buy_crossover_detection(self):
        """Test BUY signal is correctly triggered when SMMA20 crosses above SMMA120."""
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(3)]
        df = pd.DataFrame({
            "timestamp": dates,
            "close": [100.0, 105.0, 110.0],
            "smma_20": [98.0, 99.5, 102.0],
            "smma_120": [100.0, 100.0, 100.0],
        })
        # At index 0: SMMA20 (98.0) <= SMMA120 (100.0)
        # At index 1: SMMA20 (99.5) <= SMMA120 (100.0) -> No crossover yet
        # At index 2: SMMA20 (102.0) > SMMA120 (100.0) -> BUY Crossover!

        crossovers = detect_crossovers_in_dataframe("TEST-EQ", "Test Company", df)

        self.assertEqual(len(crossovers), 1)
        self.assertEqual(crossovers[0]["signal"], "BUY")
        self.assertEqual(crossovers[0]["symbol"], "TEST-EQ")
        self.assertEqual(crossovers[0]["ltp"], 110.0)
        self.assertEqual(crossovers[0]["smma_20"], 102.0)
        self.assertEqual(crossovers[0]["smma_120"], 100.0)
        self.assertEqual(crossovers[0]["prev_smma_20"], 99.5)
        self.assertEqual(crossovers[0]["prev_smma_120"], 100.0)

    def test_sell_crossover_detection(self):
        """Test SELL signal is correctly triggered when SMMA20 crosses below SMMA120."""
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(3)]
        df = pd.DataFrame({
            "timestamp": dates,
            "close": [100.0, 95.0, 90.0],
            "smma_20": [102.0, 100.5, 98.0],
            "smma_120": [100.0, 100.0, 100.0],
        })
        # At index 0: SMMA20 (102.0) >= SMMA120 (100.0)
        # At index 1: SMMA20 (100.5) >= SMMA120 (100.0) -> No crossover yet
        # At index 2: SMMA20 (98.0) < SMMA120 (100.0) -> SELL Crossover!

        crossovers = detect_crossovers_in_dataframe("TEST-EQ", "Test Company", df)

        self.assertEqual(len(crossovers), 1)
        self.assertEqual(crossovers[0]["signal"], "SELL")
        self.assertEqual(crossovers[0]["symbol"], "TEST-EQ")
        self.assertEqual(crossovers[0]["ltp"], 90.0)
        self.assertEqual(crossovers[0]["smma_20"], 98.0)
        self.assertEqual(crossovers[0]["smma_120"], 100.0)
        self.assertEqual(crossovers[0]["prev_smma_20"], 100.5)
        self.assertEqual(crossovers[0]["prev_smma_120"], 100.0)

    def test_touch_equals_crossover(self):
        """Test exact equality (touch) on previous bar triggers BUY if current is strictly greater."""
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(2)]
        df = pd.DataFrame({
            "timestamp": dates,
            "close": [100.0, 105.0],
            "smma_20": [100.0, 101.0],
            "smma_120": [100.0, 100.0],
        })
        # Prev: 100.0 <= 100.0 (Equal)
        # Curr: 101.0 > 100.0 -> BUY

        crossovers = detect_crossovers_in_dataframe("TEST-EQ", "Test Company", df)
        self.assertEqual(len(crossovers), 1)
        self.assertEqual(crossovers[0]["signal"], "BUY")

    def test_crossover_service_integration(self):
        """Test CrossoverService scans demo universe and produces metrics."""
        service = CrossoverService()
        df_crossovers = service.get_all_crossovers_dataframe()

        self.assertIn("symbol", df_crossovers.columns)
        self.assertIn("signal", df_crossovers.columns)
        self.assertIn("smma_20", df_crossovers.columns)
        self.assertIn("smma_120", df_crossovers.columns)

        metrics = service.get_summary_metrics(df_crossovers)
        self.assertGreaterEqual(metrics["total_crossovers"], 0)
        self.assertGreaterEqual(metrics["buy_count"], 0)
        self.assertGreaterEqual(metrics["sell_count"], 0)

    def test_signal_mapping_and_summary_cards(self):
        """Test STEP 2 signal mapping (BUY_CROSSOVER -> BUY, SELL_CROSSOVER -> SELL, NONE -> WATCH) and summary cards."""
        service = CrossoverService()
        self.assertEqual(service.map_crossover_to_signal("BUY_CROSSOVER"), "BUY")
        self.assertEqual(service.map_crossover_to_signal("SELL_CROSSOVER"), "SELL")
        self.assertEqual(service.map_crossover_to_signal("NONE"), "WATCH")

        df_signals = service.get_crossover_signals_dataframe()
        self.assertFalse(df_signals.empty)
        self.assertIn("symbol", df_signals.columns)
        self.assertIn("crossover", df_signals.columns)
        self.assertIn("signal", df_signals.columns)

        # Check all signals are strictly inside ('BUY', 'SELL', 'WATCH')
        for sig in df_signals["signal"].unique():
            self.assertIn(sig, ["BUY", "SELL", "WATCH"])

        cards = service.get_crossover_summary_cards(df_signals)
        self.assertEqual(cards["total_stocks"], len(df_signals))
        self.assertEqual(
            cards["total_stocks"],
            cards["buy_crossovers"] + cards["sell_crossovers"] + cards["no_crossover"]
        )


if __name__ == "__main__":
    unittest.main()

