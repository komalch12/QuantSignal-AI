"""
Unit Tests for Technical Indicator Service & SMMA Calculation Engine.

Tests:
1. calculate_smma mathematical correctness against known hand-calculated values.
2. Handling of insufficient data length (< period).
3. Handling of invalid period values (<= 0).
4. TechnicalIndicatorService integration with DemoMarketDataProvider.
"""

import unittest
import pandas as pd
import numpy as np

from quant_signal.services.technical_indicators import (
    TechnicalIndicatorService,
    calculate_smma,
)


class TestSMMACalculation(unittest.TestCase):
    """Test suite for calculate_smma function."""

    def test_smma_basic_math(self):
        """Test SMMA calculation on a known numerical sequence."""
        # Test series: [10, 12, 14, 16, 18], period = 3
        # Index 0: 10
        # Index 1: 12
        # Index 2: SMA(10, 12, 14) = 36 / 3 = 12.0
        # Index 3 (val 16): SMMA = (12.0 * (3-1) + 16) / 3 = (24 + 16) / 3 = 40 / 3 = 13.333333333333334
        # Index 4 (val 18): SMMA = (13.333333333333334 * 2 + 18) / 3 = (26.666666666666668 + 18) / 3 = 14.88888888888889
        series = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        result = calculate_smma(series, period=3)

        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[2], 12.0, places=5)
        self.assertAlmostEqual(result.iloc[3], 40.0 / 3.0, places=5)
        self.assertAlmostEqual(result.iloc[4], 134.0 / 9.0, places=5)

    def test_smma_insufficient_data(self):
        """Test that series shorter than period returns NaNs."""
        series = pd.Series([10.0, 12.0, 14.0])
        result = calculate_smma(series, period=5)

        self.assertEqual(len(result), 3)
        self.assertTrue(result.isna().all())

    def test_smma_invalid_period(self):
        """Test that period <= 0 raises ValueError."""
        series = pd.Series([10.0, 12.0, 14.0, 16.0])

        with self.assertRaises(ValueError):
            calculate_smma(series, period=0)

        with self.assertRaises(ValueError):
            calculate_smma(series, period=-5)

    def test_smma_20_manual_calculation(self):
        """Test SMMA(20) manual verification against Wilder's formula."""
        # 22 data points: 1.0 to 22.0
        prices = [float(x) for x in range(1, 23)]
        series = pd.Series(prices)
        res = calculate_smma(series, period=20)

        # Indices 0..18 should be NaN
        for i in range(19):
            self.assertTrue(np.isnan(res.iloc[i]), f"Index {i} should be NaN")

        # Index 19: SMA of first 20 numbers (1 to 20): sum = 210, mean = 210/20 = 10.5
        expected_idx_19 = sum(range(1, 21)) / 20.0  # 10.5
        self.assertAlmostEqual(res.iloc[19], expected_idx_19, places=6)

        # Index 20 (price = 21.0): (10.5 * 19 + 21.0) / 20 = (199.5 + 21.0) / 20 = 220.5 / 20 = 11.025
        expected_idx_20 = (expected_idx_19 * 19 + 21.0) / 20.0
        self.assertAlmostEqual(res.iloc[20], expected_idx_20, places=6)

        # Index 21 (price = 22.0): (11.025 * 19 + 22.0) / 20 = (209.475 + 22.0) / 20 = 231.475 / 20 = 11.57375
        expected_idx_21 = (expected_idx_20 * 19 + 22.0) / 20.0
        self.assertAlmostEqual(res.iloc[21], expected_idx_21, places=6)

    def test_smma_120_manual_calculation(self):
        """Test SMMA(120) manual verification against Wilder's formula."""
        # 122 data points: 100.0 for first 120, then 150.0, 200.0
        prices = [100.0] * 120 + [150.0, 200.0]
        series = pd.Series(prices)
        res = calculate_smma(series, period=120)

        # Indices 0..118 should be NaN
        for i in range(119):
            self.assertTrue(np.isnan(res.iloc[i]), f"Index {i} should be NaN")

        # Index 119: SMA of 120 items of 100.0 = 100.0
        self.assertAlmostEqual(res.iloc[119], 100.0, places=6)

        # Index 120 (price = 150.0): (100.0 * 119 + 150.0) / 120 = 12050 / 120 = 100.416666...
        expected_idx_120 = (100.0 * 119 + 150.0) / 120.0
        self.assertAlmostEqual(res.iloc[120], expected_idx_120, places=6)

        # Index 121 (price = 200.0): (expected_idx_120 * 119 + 200.0) / 120
        expected_idx_121 = (expected_idx_120 * 119 + 200.0) / 120.0
        self.assertAlmostEqual(res.iloc[121], expected_idx_121, places=6)

    def test_look_ahead_leakage(self):
        """Verify modifying future prices does NOT alter past SMMA values (no look-ahead leakage)."""
        prices_original = [10.0 + i for i in range(30)]
        prices_modified = list(prices_original)
        # Modify future prices at index 25..29
        for i in range(25, 30):
            prices_modified[i] = 9999.0

        res_orig = calculate_smma(pd.Series(prices_original), period=20)
        res_mod = calculate_smma(pd.Series(prices_modified), period=20)

        # Up to index 24, both results must be mathematically identical
        for i in range(25):
            if np.isnan(res_orig.iloc[i]):
                self.assertTrue(np.isnan(res_mod.iloc[i]))
            else:
                self.assertAlmostEqual(res_orig.iloc[i], res_mod.iloc[i], places=9)

    def test_historical_data_requirement_120(self):
        """Verify that SMMA(120) requires at least 120 observations."""
        series_119 = pd.Series([100.0] * 119)
        res_119 = calculate_smma(series_119, period=120)
        self.assertEqual(len(res_119), 119)
        self.assertTrue(res_119.isna().all())

        series_120 = pd.Series([100.0] * 120)
        res_120 = calculate_smma(series_120, period=120)
        self.assertEqual(len(res_120), 120)
        self.assertFalse(np.isnan(res_120.iloc[119]))
        self.assertAlmostEqual(res_120.iloc[119], 100.0, places=6)

    def test_technical_indicator_service_demo(self):
        """Test TechnicalIndicatorService returns valid SMMA_20 & SMMA_120 values for demo stock."""
        service = TechnicalIndicatorService()
        result = service.calculate_stock_indicators("SUZLON-EQ", days=250)

        self.assertEqual(result["symbol"], "SUZLON-EQ")
        self.assertTrue(result["has_sufficient_data"])
        self.assertFalse(np.isnan(result["smma_20"]))
        self.assertFalse(np.isnan(result["smma_120"]))
        self.assertIn(result["trend"], ["Bullish 🟢", "Bearish 🔴"])

    def test_indicators_dataframe(self):
        """Test get_indicators_dataframe generates full summary DataFrame."""
        service = TechnicalIndicatorService()
        df = service.get_indicators_dataframe()

        self.assertFalse(df.empty)
        self.assertIn("symbol", df.columns)
        self.assertIn("smma_20", df.columns)
        self.assertIn("smma_120", df.columns)
        self.assertIn("trend", df.columns)
        self.assertIn("distance_pct", df.columns)
        self.assertIn("crossover", df.columns)

    def test_crossover_signal_rules(self):
        """Test determine_crossover_signal for all 6 required scenarios."""
        from quant_signal.services.technical_indicators import determine_crossover_signal

        # 1. BUY crossover (prev_20 <= prev_120 and curr_20 > curr_120)
        self.assertEqual(
            determine_crossover_signal(prev_20=99.0, prev_120=100.0, curr_20=101.0, curr_120=100.0),
            "BUY_CROSSOVER"
        )
        self.assertEqual(
            determine_crossover_signal(prev_20=100.0, prev_120=100.0, curr_20=101.0, curr_120=100.0),
            "BUY_CROSSOVER"
        )

        # 2. SELL crossover (prev_20 >= prev_120 and curr_20 < curr_120)
        self.assertEqual(
            determine_crossover_signal(prev_20=101.0, prev_120=100.0, curr_20=99.0, curr_120=100.0),
            "SELL_CROSSOVER"
        )
        self.assertEqual(
            determine_crossover_signal(prev_20=100.0, prev_120=100.0, curr_20=99.0, curr_120=100.0),
            "SELL_CROSSOVER"
        )

        # 3. No crossover (curr_20 > curr_120 but prev_20 was already > prev_120)
        self.assertEqual(
            determine_crossover_signal(prev_20=102.0, prev_120=100.0, curr_20=103.0, curr_120=100.0),
            "NONE"
        )

        # 4. Equal SMMA values
        self.assertEqual(
            determine_crossover_signal(prev_20=100.0, prev_120=100.0, curr_20=100.0, curr_120=100.0),
            "NONE"
        )

        # 5. Missing historical values
        self.assertEqual(
            determine_crossover_signal(prev_20=np.nan, prev_120=100.0, curr_20=101.0, curr_120=100.0),
            "NONE"
        )

        # 6. Insufficient data
        service = TechnicalIndicatorService()
        result_short = service.calculate_stock_indicators("NON_EXISTENT", days=10)
        self.assertEqual(result_short["crossover"], "NONE")
        self.assertFalse(result_short["has_sufficient_data"])


if __name__ == "__main__":
    unittest.main()


