"""
Unit Tests for Step 4B — AI/ML Decision Layer & Recommendation Rule Alignment.

Tests:
1. Confidence calculation from ML model predict_proba.
2. Recommendation threshold mapping:
   - STRONG BUY threshold
   - BUY threshold
   - HOLD threshold
   - SELL threshold
3. Deterministic repeated inference (identical features -> identical confidence and recommendation).
4. NaN handling in confidence and crossover values.
5. Missing data handling (empty DataFrame or missing model).
6. Single-symbol inference returning complete dictionary.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from quant_signal.services.ml_signal_service import (
    MLSignalService,
    extract_features_from_history,
    map_confidence_to_recommendation,
)


class TestStep4BAIRecommendationRules(unittest.TestCase):
    """Test suite for Step 4B Recommendation Rules & Deterministic Decision Engine."""

    def test_recommendation_thresholds_strong_buy(self):
        """Test conf_pct >= 75.0 or (BUY_CROSSOVER and conf_pct >= 60.0) returns STRONG BUY 🟢."""
        self.assertEqual(map_confidence_to_recommendation(80.0, "NONE"), "STRONG BUY 🟢")
        self.assertEqual(map_confidence_to_recommendation(65.0, "BUY_CROSSOVER"), "STRONG BUY 🟢")

    def test_recommendation_thresholds_buy(self):
        """Test conf_pct >= 60.0 or BUY_CROSSOVER returns BUY 🟢."""
        self.assertEqual(map_confidence_to_recommendation(65.0, "NONE"), "BUY 🟢")
        self.assertEqual(map_confidence_to_recommendation(55.0, "BUY_CROSSOVER"), "BUY 🟢")

    def test_recommendation_thresholds_sell(self):
        """Test conf_pct <= 35.0 or SELL_CROSSOVER returns SELL 🔴."""
        self.assertEqual(map_confidence_to_recommendation(30.0, "NONE"), "SELL 🔴")
        self.assertEqual(map_confidence_to_recommendation(50.0, "SELL_CROSSOVER"), "SELL 🔴")

    def test_recommendation_thresholds_hold(self):
        """Test neutral confidence (e.g. 50.0%) without crossover returns HOLD 🟡."""
        self.assertEqual(map_confidence_to_recommendation(50.0, "NONE"), "HOLD 🟡")
        self.assertEqual(map_confidence_to_recommendation(45.0, "NONE"), "HOLD 🟡")

    def test_deterministic_repeated_inference(self):
        """Test identical inputs repeatedly produce identical confidence and recommendation."""
        service = MLSignalService()
        res1 = service.calculate_stock_ml_signal("SUZLON-EQ")
        res2 = service.calculate_stock_ml_signal("SUZLON-EQ")

        self.assertEqual(res1["confidence_pct"], res2["confidence_pct"])
        self.assertEqual(res1["recommendation"], res2["recommendation"])

    def test_nan_and_invalid_confidence_handling(self):
        """Test NaN and infinite confidence values are handled safely.

        NaN  -> not finite -> defaults to 50.0 -> HOLD.
        inf  -> not finite -> defaults to 50.0 -> HOLD.
        Both must NOT raise an exception.
        """
        self.assertEqual(map_confidence_to_recommendation(np.nan, "NONE"), "HOLD 🟡")
        self.assertEqual(map_confidence_to_recommendation(np.inf, "NONE"), "HOLD 🟡")
        self.assertEqual(map_confidence_to_recommendation(-np.inf, "NONE"), "HOLD 🟡")

    def test_missing_data_inference_safety(self):
        """Test inference for an unknown symbol is safe: confidence in [0, 100] range and valid recommendation."""
        service = MLSignalService()
        # An unknown symbol may still produce a result from fallback data; all we assert is safety.
        empty_res = service.calculate_stock_ml_signal("NONEXISTENT_SYMBOL_XYZ")
        self.assertGreaterEqual(empty_res["confidence_pct"], 0.0)
        self.assertLessEqual(empty_res["confidence_pct"], 100.0)
        valid_recs = {"STRONG BUY 🟢", "BUY 🟢", "HOLD 🟡", "SELL 🔴"}
        self.assertIn(empty_res["recommendation"], valid_recs)

    def test_single_symbol_inference_dict_schema(self):
        """Test single-symbol inference output dictionary contains all required keys."""
        service = MLSignalService()
        res = service.calculate_stock_ml_signal("SUZLON-EQ")

        required_keys = {
            "symbol", "ltp", "smma_20", "smma_120", "trend",
            "distance_pct", "crossover", "confidence_pct", "recommendation", "latest_features"
        }
        self.assertTrue(required_keys.issubset(set(res.keys())))


class TestStep4AFeatures(unittest.TestCase):
    """Test suite for Step 4A Feature Extraction (Regression Prevention)."""

    def test_crossover_val_buy_feature(self):
        """Test BUY crossover event generates positive crossover_val code (1)."""
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(10)]
        smma_20 = [98.0, 98.5, 99.0, 99.5, 99.8, 101.5, 102.0, 103.0, 104.0, 105.0]
        smma_120 = [100.0] * 10
        close = [100.0 + i for i in range(10)]

        history_df = pd.DataFrame({
            "timestamp": dates,
            "close": close,
            "smma_20": smma_20,
            "smma_120": smma_120,
        })

        feat_df = extract_features_from_history(history_df)
        self.assertFalse(feat_df.empty)
        crossover_codes = feat_df["crossover_val"].to_list()
        self.assertIn(1, crossover_codes)

    def test_crossover_val_sell_feature(self):
        """Test SELL crossover event generates negative crossover_val code (-1)."""
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(10)]
        smma_20 = [102.0, 101.5, 101.0, 100.5, 100.2, 98.5, 98.0, 97.0, 96.0, 95.0]
        smma_120 = [100.0] * 10
        close = [100.0 - i for i in range(10)]

        history_df = pd.DataFrame({
            "timestamp": dates,
            "close": close,
            "smma_20": smma_20,
            "smma_120": smma_120,
        })

        feat_df = extract_features_from_history(history_df)
        self.assertFalse(feat_df.empty)
        crossover_codes = feat_df["crossover_val"].to_list()
        self.assertIn(-1, crossover_codes)


class TestStep4CMSDataframeAndMetrics(unittest.TestCase):
    """Test suite for Step 4C AI/ML Signals DataFrame, Summary Metrics & Feature Importances."""

    def test_get_ml_signals_dataframe_schema_and_sorting(self):
        """Test get_ml_signals_dataframe returns valid schema and is sorted by confidence_pct descending."""
        service = MLSignalService()
        df = service.get_ml_signals_dataframe()

        self.assertFalse(df.empty)
        expected_cols = {
            "symbol", "company_name", "ltp", "smma_20", "smma_120", "trend",
            "distance_pct", "crossover", "crossover_raw", "confidence_pct", "recommendation", "timestamp"
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

        # Verify confidence_pct is sorted descending
        conf_list = df["confidence_pct"].to_list()
        self.assertEqual(conf_list, sorted(conf_list, reverse=True))

    def test_get_summary_metrics_calculation(self):
        """Test get_summary_metrics returns accurate dict structure and correct metrics."""
        service = MLSignalService()
        df = service.get_ml_signals_dataframe()
        metrics = service.get_summary_metrics(df)

        self.assertEqual(metrics["total_evaluated"], len(df))
        self.assertGreaterEqual(metrics["high_confidence_count"], 0)
        self.assertGreaterEqual(metrics["bullish_recommendations"], 0)
        self.assertGreaterEqual(metrics["avg_confidence"], 0.0)
        self.assertLessEqual(metrics["avg_confidence"], 100.0)

    def test_get_feature_importances_dict(self):
        """Test get_feature_importances returns a dict with 5 keys matching FEATURE_NAMES."""
        service = MLSignalService()
        importances = service.get_feature_importances()

        self.assertEqual(len(importances), 5)
        for feat in MLSignalService.FEATURE_NAMES:
            self.assertIn(feat, importances)
            self.assertIsInstance(importances[feat], float)
            self.assertGreaterEqual(importances[feat], 0.0)

    def test_ml_signals_dataframe_empty_universe(self):
        """Test get_ml_signals_dataframe with empty symbol list returns empty DataFrame with schema."""
        service = MLSignalService()
        df = service.get_ml_signals_dataframe(symbols=[])

        self.assertTrue(df.empty)
        expected_cols = {
            "symbol", "company_name", "ltp", "smma_20", "smma_120", "trend",
            "distance_pct", "crossover", "crossover_raw", "confidence_pct", "recommendation", "timestamp"
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_summary_metrics_empty_dataframe(self):
        """Test get_summary_metrics with empty DataFrame returns zeroed metrics."""
        service = MLSignalService()
        metrics = service.get_summary_metrics(pd.DataFrame())

        self.assertEqual(metrics["total_evaluated"], 0)
        self.assertEqual(metrics["high_confidence_count"], 0)
        self.assertEqual(metrics["bullish_recommendations"], 0)
        self.assertEqual(metrics["avg_confidence"], 0.0)

    def test_ml_signals_dataframe_subset_symbols(self):
        """Test get_ml_signals_dataframe with specific subset of symbols returns matching rows."""
        service = MLSignalService()
        subset = ["SUZLON-EQ", "RELIANCE-EQ"]
        df = service.get_ml_signals_dataframe(symbols=subset)

        self.assertEqual(len(df), len(subset))
        self.assertEqual(set(df["symbol"]), set(subset))


class TestStep4DIndividualStockExplainability(unittest.TestCase):
    """Test suite for Step 4D Individual Stock AI Decision Explainability & Feature Contribution."""

    def test_get_stock_explainability_schema(self):
        """Test get_stock_explainability returns dictionary with all required schema keys."""
        service = MLSignalService()
        explain = service.get_stock_explainability("SUZLON-EQ")

        required_keys = {
            "symbol", "company_name", "ltp", "trend", "crossover",
            "confidence_pct", "recommendation", "explanation", "features"
        }
        self.assertTrue(required_keys.issubset(set(explain.keys())))
        self.assertEqual(explain["symbol"], "SUZLON-EQ")
        self.assertIsInstance(explain["explanation"], str)
        self.assertGreater(len(explain["explanation"]), 10)

    def test_get_stock_explainability_features_list(self):
        """Test features key contains list of 5 feature dicts with feature, value, and importance."""
        service = MLSignalService()
        explain = service.get_stock_explainability("SUZLON-EQ")
        features = explain["features"]

        self.assertEqual(len(features), 5)
        for item in features:
            self.assertIn("feature", item)
            self.assertIn("value", item)
            self.assertIn("importance", item)
            self.assertIsInstance(item["feature"], str)
            self.assertIsInstance(item["value"], float)
            self.assertIsInstance(item["importance"], float)

    def test_get_stock_explainability_explanations(self):
        """Test explanation text contains confidence percentage and recommendation context."""
        service = MLSignalService()
        explain = service.get_stock_explainability("SUZLON-EQ")
        explanation = explain["explanation"]

        self.assertIn(f"{explain['confidence_pct']:.1f}%", explanation)
        self.assertIn(explain["crossover"], explanation)

    def test_get_stock_explainability_unknown_symbol(self):
        """Test passing unknown symbol handles safely and returns valid explainability schema."""
        service = MLSignalService()
        explain = service.get_stock_explainability("UNKNOWN_TICKER_XYZ")

        self.assertEqual(explain["symbol"], "UNKNOWN_TICKER_XYZ")
        self.assertIn("confidence_pct", explain)
        self.assertIn("recommendation", explain)
        self.assertIn("explanation", explain)
        self.assertIsInstance(explain["features"], list)


if __name__ == "__main__":
    unittest.main()


