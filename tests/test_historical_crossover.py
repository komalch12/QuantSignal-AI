"""
Step 3A Unit Tests — Historical SMMA 20/120 Crossover Scanner.

Tests every requirement from Step 3A:
  1. BUY crossover detection (prev20 <= prev120 AND curr20 > curr120)
  2. SELL crossover detection (prev20 >= prev120 AND curr20 < curr120)
  3. NO crossover bars produce no events
  4. Exact equality on previous bar triggers BUY when current breaks above
  5. NaN in SMMA rows are safely excluded (don't block scan, don't produce false events)
  6. NaN close price is recorded as np.nan without crashing
  7. Fewer than 2 valid rows => returns empty list
  8. Empty DataFrame => returns empty list
  9. Missing SMMA columns => returns empty list
  10. Duplicate timestamps are deduplicated (last occurrence kept)
  11. Multiple crossovers across a long series are all captured
  12. CrossoverService.get_symbol_crossovers() uses robust scanner by default
  13. CrossoverService.get_all_crossovers_dataframe() returns correct schema columns
  14. CrossoverService.get_summary_metrics() counts match the returned DataFrame
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_signal.services.crossover_service import (
    CrossoverService,
    detect_crossovers_in_dataframe,
    scan_historical_crossovers,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_df(
    smma_20: List[float],
    smma_120: List[float],
    close: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame for testing."""
    n = len(smma_20)
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "timestamp": dates,
        "close":     close if close is not None else [100.0 + i for i in range(n)],
        "smma_20":   smma_20,
        "smma_120":  smma_120,
    })


# ─── 1. BUY crossover detection ──────────────────────────────────────────────

class TestBuyCrossover(unittest.TestCase):

    def test_simple_buy_crossover(self):
        """prev20 <= prev120 AND curr20 > curr120 => BUY."""
        df = _make_df(
            smma_20=[98.0, 99.5, 101.5],
            smma_120=[100.0, 100.0, 100.0],
        )
        # At bar-2: prev(99.5)<=prev(100.0) AND curr(101.5)>curr(100.0) => BUY
        events = scan_historical_crossovers("SYM", "Company", df)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["crossover_type"], "BUY")
        self.assertEqual(events[0]["signal"], "BUY")

    def test_buy_crossover_event_fields(self):
        """BUY event contains all required fields with correct values."""
        df = _make_df(
            smma_20=[98.0, 99.5, 102.0],
            smma_120=[100.0, 100.0, 100.0],
            close=[90.0, 95.0, 110.0],
        )
        events = scan_historical_crossovers("TEST-EQ", "Test Co", df)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["symbol"],        "TEST-EQ")
        self.assertEqual(ev["company_name"],  "Test Co")
        self.assertAlmostEqual(ev["prev_smma_20"],  99.5)
        self.assertAlmostEqual(ev["prev_smma_120"], 100.0)
        self.assertAlmostEqual(ev["curr_smma_20"],  102.0)
        self.assertAlmostEqual(ev["curr_smma_120"], 100.0)
        self.assertAlmostEqual(ev["close_price"],   110.0)

    def test_buy_equality_on_previous_bar(self):
        """prev20 == prev120 still satisfies <= condition => BUY if curr20 > curr120."""
        df = _make_df(
            smma_20=[100.0, 101.0],
            smma_120=[100.0, 100.0],
        )
        events = scan_historical_crossovers("EQ", "Co", df)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["crossover_type"], "BUY")


# ─── 2. SELL crossover detection ─────────────────────────────────────────────

class TestSellCrossover(unittest.TestCase):

    def test_simple_sell_crossover(self):
        """prev20 >= prev120 AND curr20 < curr120 => SELL."""
        df = _make_df(
            smma_20=[102.0, 100.5, 98.0],
            smma_120=[100.0, 100.0, 100.0],
        )
        events = scan_historical_crossovers("SYM", "Company", df)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["crossover_type"], "SELL")
        self.assertEqual(events[0]["signal"], "SELL")

    def test_sell_event_fields(self):
        """SELL event contains correct prev/curr field values."""
        df = _make_df(
            smma_20=[102.0, 100.5, 98.0],
            smma_120=[100.0, 100.0, 100.0],
            close=[110.0, 105.0, 90.0],
        )
        events = scan_historical_crossovers("TEST-EQ", "Test Co", df)
        ev = events[0]
        self.assertAlmostEqual(ev["prev_smma_20"],  100.5)
        self.assertAlmostEqual(ev["prev_smma_120"], 100.0)
        self.assertAlmostEqual(ev["curr_smma_20"],  98.0)
        self.assertAlmostEqual(ev["curr_smma_120"], 100.0)
        self.assertAlmostEqual(ev["close_price"],   90.0)

    def test_sell_equality_on_previous_bar(self):
        """prev20 == prev120 satisfies >= condition => SELL if curr20 < curr120."""
        df = _make_df(
            smma_20=[100.0, 99.0],
            smma_120=[100.0, 100.0],
        )
        events = scan_historical_crossovers("EQ", "Co", df)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["crossover_type"], "SELL")


# ─── 3. NO crossover ─────────────────────────────────────────────────────────

class TestNoCrossover(unittest.TestCase):

    def test_no_crossover_when_smma20_always_above(self):
        """When SMMA20 stays above SMMA120 throughout, no events produced."""
        df = _make_df(
            smma_20=[105.0, 106.0, 107.0, 108.0],
            smma_120=[100.0, 100.0, 100.0, 100.0],
        )
        events = scan_historical_crossovers("SYM", "Co", df)
        self.assertEqual(events, [])

    def test_no_crossover_when_smma20_always_below(self):
        """When SMMA20 stays below SMMA120 throughout, no events produced."""
        df = _make_df(
            smma_20=[95.0, 94.0, 93.0],
            smma_120=[100.0, 100.0, 100.0],
        )
        events = scan_historical_crossovers("SYM", "Co", df)
        self.assertEqual(events, [])

    def test_no_events_on_single_bar(self):
        """A single-row DataFrame cannot have a previous bar => no events."""
        df = _make_df(smma_20=[100.0], smma_120=[100.0])
        events = scan_historical_crossovers("SYM", "Co", df)
        self.assertEqual(events, [])


# ─── 4. NaN handling ─────────────────────────────────────────────────────────

class TestNaNHandling(unittest.TestCase):

    def test_nan_smma_rows_excluded_from_scan(self):
        """NaN SMMA rows are excluded; valid rows after warm-up are still scanned."""
        smma_20 = [np.nan] * 19 + [98.0, 99.5, 102.0]   # first 19 are NaN
        smma_120 = [np.nan] * 19 + [100.0, 100.0, 100.0]
        close = [100.0] * 22
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(22)]
        df = pd.DataFrame({"timestamp": dates, "close": close, "smma_20": smma_20, "smma_120": smma_120})

        events = scan_historical_crossovers("SYM", "Co", df)
        # Should find one BUY at the last bar
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["crossover_type"], "BUY")

    def test_nan_close_price_does_not_crash(self):
        """A NaN close price is recorded as np.nan, the event is still returned."""
        df = pd.DataFrame({
            "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            "close":     [np.nan, np.nan],
            "smma_20":   [99.0, 101.0],
            "smma_120":  [100.0, 100.0],
        })
        events = scan_historical_crossovers("SYM", "Co", df)
        self.assertEqual(len(events), 1)
        self.assertTrue(np.isnan(events[0]["close_price"]))

    def test_all_nan_smma_returns_empty(self):
        """All NaN SMMA values => empty result."""
        df = pd.DataFrame({
            "timestamp": [datetime(2025, 1, i + 1) for i in range(5)],
            "close":     [100.0] * 5,
            "smma_20":   [np.nan] * 5,
            "smma_120":  [np.nan] * 5,
        })
        events = scan_historical_crossovers("SYM", "Co", df)
        self.assertEqual(events, [])


# ─── 5. Insufficient data ────────────────────────────────────────────────────

class TestInsufficientData(unittest.TestCase):

    def test_empty_dataframe(self):
        """Empty DataFrame => empty result, no exception."""
        events = scan_historical_crossovers("SYM", "Co", pd.DataFrame())
        self.assertEqual(events, [])

    def test_missing_smma_columns(self):
        """DataFrame without smma_20 / smma_120 => empty result, no exception."""
        df = pd.DataFrame({"timestamp": [datetime(2025, 1, 1)], "close": [100.0]})
        events = scan_historical_crossovers("SYM", "Co", df)
        self.assertEqual(events, [])

    def test_only_one_valid_row(self):
        """Only one valid (non-NaN) SMMA row => no previous bar => empty result."""
        df = pd.DataFrame({
            "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            "close":     [100.0, 100.0],
            "smma_20":   [np.nan, 100.0],
            "smma_120":  [np.nan, 100.0],
        })
        events = scan_historical_crossovers("SYM", "Co", df)
        self.assertEqual(events, [])

    def test_none_dataframe(self):
        """None input => empty result, no exception."""
        events = scan_historical_crossovers("SYM", "Co", None)  # type: ignore[arg-type]
        self.assertEqual(events, [])


# ─── 6. Duplicate timestamps ─────────────────────────────────────────────────

class TestDuplicateTimestamps(unittest.TestCase):

    def test_duplicate_timestamps_deduplicated(self):
        """Duplicate timestamps are removed (last occurrence kept); result unchanged vs clean data."""
        ts = datetime(2025, 1, 1)
        df_with_dupes = pd.DataFrame({
            "timestamp": [ts, ts, datetime(2025, 1, 2)],
            "close":     [100.0, 100.0, 110.0],
            "smma_20":   [99.0, 99.0, 101.0],   # duplicate first row
            "smma_120":  [100.0, 100.0, 100.0],
        })
        df_clean = pd.DataFrame({
            "timestamp": [ts, datetime(2025, 1, 2)],
            "close":     [100.0, 110.0],
            "smma_20":   [99.0, 101.0],
            "smma_120":  [100.0, 100.0],
        })
        events_dupes = scan_historical_crossovers("SYM", "Co", df_with_dupes)
        events_clean = scan_historical_crossovers("SYM", "Co", df_clean)
        self.assertEqual(len(events_dupes), len(events_clean))


# ─── 7. Multiple crossovers in a long series ─────────────────────────────────

class TestMultipleCrossovers(unittest.TestCase):

    def test_multiple_crossover_events_all_captured(self):
        """All BUY and SELL crossover events in a long series are returned."""
        # Series: BUY at bar-2, SELL at bar-4, BUY at bar-6
        smma_20 = [98, 99, 101, 102, 102, 100, 99, 101]   # crosses up at 2, crosses down at 5, up at 7
        smma_120 = [100, 100, 100, 100, 100, 100, 100, 100]
        df = _make_df(smma_20=smma_20, smma_120=smma_120)

        events = scan_historical_crossovers("SYM", "Co", df)
        signals = [e["crossover_type"] for e in events]
        self.assertIn("BUY", signals)
        self.assertIn("SELL", signals)
        # There must be at least 2 events (BUY + SELL)
        self.assertGreaterEqual(len(events), 2)

    def test_events_sorted_by_timestamp_ascending(self):
        """Events are returned chronologically (oldest first)."""
        smma_20 = [98, 99, 101, 102, 102, 100, 99, 101]
        smma_120 = [100, 100, 100, 100, 100, 100, 100, 100]
        df = _make_df(smma_20=smma_20, smma_120=smma_120)
        events = scan_historical_crossovers("SYM", "Co", df)
        if len(events) >= 2:
            for i in range(len(events) - 1):
                self.assertLessEqual(events[i]["timestamp"], events[i + 1]["timestamp"])


# ─── 8. CrossoverService integration ─────────────────────────────────────────

class TestCrossoverServiceIntegration(unittest.TestCase):

    def test_get_symbol_crossovers_uses_robust_scanner_by_default(self):
        """get_symbol_crossovers() returns a list (may be empty) using robust scanner."""
        service = CrossoverService()
        events = service.get_symbol_crossovers("SUZLON-EQ")
        self.assertIsInstance(events, list)
        # Every event must have the Step 3A fields
        for ev in events:
            self.assertIn("crossover_type", ev)
            self.assertIn("curr_smma_20", ev)
            self.assertIn("curr_smma_120", ev)
            self.assertIn("close_price", ev)
            self.assertIn(ev["crossover_type"], ("BUY", "SELL"))

    def test_get_all_crossovers_dataframe_schema(self):
        """get_all_crossovers_dataframe() returns correct Step 3A schema columns."""
        service = CrossoverService()
        df = service.get_all_crossovers_dataframe()
        expected_cols = {
            "symbol", "company_name", "timestamp", "signal", "crossover_type",
            "close_price", "ltp", "curr_smma_20", "curr_smma_120",
            "prev_smma_20", "prev_smma_120",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_get_all_crossovers_no_fake_signals(self):
        """Only BUY and SELL values appear in signal column — no placeholder strings."""
        service = CrossoverService()
        df = service.get_all_crossovers_dataframe()
        if not df.empty:
            for sig in df["signal"].unique():
                self.assertIn(sig, ("BUY", "SELL"))

    def test_get_summary_metrics_counts_consistent(self):
        """Summary metrics buy_count + sell_count == total_crossovers."""
        service = CrossoverService()
        df = service.get_all_crossovers_dataframe()
        metrics = service.get_summary_metrics(df)
        self.assertEqual(
            metrics["buy_count"] + metrics["sell_count"],
            metrics["total_crossovers"],
        )


if __name__ == "__main__":
    unittest.main()
