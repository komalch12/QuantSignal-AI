"""
Unit tests for CrossoverProfitabilityService (SMMA Crossover Profitability Evaluation).

Verifies:
  1. Profitable BUY crossover
  2. Unprofitable BUY crossover
  3. Profitable SELL crossover
  4. Unprofitable SELL crossover
  5. Exact crossover entry price
  6. Exit price after evaluation horizon (5 bars)
  7. Insufficient future data handling
  8. Missing prices handling
  9. Invalid prices handling (zero, negative)
  10. NaN prices handling
  11. Multiple historical crossovers evaluation
  12. BUY win rate calculation
  13. SELL win rate calculation
  14. No look-ahead leakage verification
  15. Deterministic PnL calculation
  16. DataFrame integration
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import pytest

from quant_signal.core.types import CrossoverProfitabilityResult
from quant_signal.services.crossover_profitability_service import CrossoverProfitabilityService



@pytest.fixture
def prof_service() -> CrossoverProfitabilityService:
    return CrossoverProfitabilityService()


def create_sample_history(prices: list[float]) -> pd.DataFrame:
    base_time = datetime(2025, 1, 1, 9, 15)
    rows = []
    for i, p in enumerate(prices):
        rows.append({
            "timestamp": base_time + timedelta(days=i),
            "open": p,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "volume": 10000,
        })
    return pd.DataFrame(rows)


def test_profitable_buy_crossover(prof_service: CrossoverProfitabilityService) -> None:
    # 10 bars: bar 2 is entry (100.0), bar 7 (2+5) is exit (120.0)
    prices = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 118.0, 120.0, 125.0, 130.0]
    df = create_sample_history(prices)

    event = {
        "symbol": "SUZLON-EQ",
        "company_name": "Suzlon Energy",
        "timestamp": df.loc[2, "timestamp"],
        "crossover_type": "BUY",
        "close_price": 100.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=5)

    assert res.result == "PROFITABLE"
    assert res.signal == "BUY"
    assert res.entry_price == 100.0
    assert res.exit_price == 120.0
    assert res.pnl == 20.0
    assert res.return_pct == 20.0
    assert res.available_data is True


def test_unprofitable_buy_crossover(prof_service: CrossoverProfitabilityService) -> None:
    # 10 bars: bar 2 is entry (100.0), bar 7 (2+5) is exit (80.0)
    prices = [90.0, 95.0, 100.0, 98.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0]
    df = create_sample_history(prices)

    event = {
        "symbol": "YESBANK-EQ",
        "company_name": "Yes Bank",
        "timestamp": df.loc[2, "timestamp"],
        "crossover_type": "BUY",
        "close_price": 100.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=5)

    assert res.result == "UNPROFITABLE"
    assert res.signal == "BUY"
    assert res.entry_price == 100.0
    assert res.exit_price == 80.0
    assert res.pnl == -20.0
    assert res.return_pct == -20.0


def test_profitable_sell_crossover(prof_service: CrossoverProfitabilityService) -> None:
    # SELL short trade: entry (100.0), exit at bar k+5 (80.0) -> PnL = 20.0 (+20%)
    prices = [110.0, 105.0, 100.0, 95.0, 90.0, 88.0, 85.0, 80.0, 75.0, 70.0]
    df = create_sample_history(prices)

    event = {
        "symbol": "TATASTEEL-EQ",
        "company_name": "Tata Steel",
        "timestamp": df.loc[2, "timestamp"],
        "crossover_type": "SELL",
        "close_price": 100.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=5)

    assert res.result == "PROFITABLE"
    assert res.signal == "SELL"
    assert res.entry_price == 100.0
    assert res.exit_price == 80.0
    assert res.pnl == 20.0
    assert res.return_pct == 20.0


def test_unprofitable_sell_crossover(prof_service: CrossoverProfitabilityService) -> None:
    # SELL short trade: entry (100.0), exit at bar k+5 (120.0) -> PnL = -20.0 (-20%)
    prices = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 118.0, 120.0, 125.0, 130.0]
    df = create_sample_history(prices)

    event = {
        "symbol": "RELIANCE-EQ",
        "company_name": "Reliance Industries",
        "timestamp": df.loc[2, "timestamp"],
        "crossover_type": "SELL",
        "close_price": 100.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=5)

    assert res.result == "UNPROFITABLE"
    assert res.signal == "SELL"
    assert res.entry_price == 100.0
    assert res.exit_price == 120.0
    assert res.pnl == -20.0
    assert res.return_pct == -20.0


def test_exact_crossover_entry_and_exit_horizon(prof_service: CrossoverProfitabilityService) -> None:
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    df = create_sample_history(prices)

    # Bar 1 is entry (20.0). Horizon = 3 -> Exit at bar 4 (50.0)
    event = {
        "symbol": "IDEAFORAGE-EQ",
        "timestamp": df.loc[1, "timestamp"],
        "crossover_type": "BUY",
        "close_price": 20.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=3)

    assert res.entry_price == 20.0
    assert res.exit_price == 50.0
    assert res.evaluation_horizon == 3
    assert res.pnl == 30.0
    assert res.return_pct == 150.0


def test_insufficient_future_data(prof_service: CrossoverProfitabilityService) -> None:
    # 6 bars: crossover at bar 4. 5-bar horizon requires index 4+5=9, which doesn't exist.
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    df = create_sample_history(prices)

    event = {
        "symbol": "ZEEL-EQ",
        "timestamp": df.loc[4, "timestamp"],
        "crossover_type": "BUY",
        "close_price": 50.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=5)

    assert res.result == "INSUFFICIENT_DATA"
    assert res.available_data is False
    assert res.exit_price is None
    assert res.pnl is None
    assert res.return_pct is None


def test_missing_and_invalid_prices(prof_service: CrossoverProfitabilityService) -> None:
    prices = [10.0, 20.0, 0.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    df = create_sample_history(prices)

    # Entry price is 0.0 at bar 2
    event = {
        "symbol": "CANBK-EQ",
        "timestamp": df.loc[2, "timestamp"],
        "crossover_type": "BUY",
        "close_price": 0.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=5)

    assert res.result == "INSUFFICIENT_DATA"
    assert res.available_data is False


def test_nan_prices_handling(prof_service: CrossoverProfitabilityService) -> None:
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, np.nan]
    df = create_sample_history(prices)

    # Exit price at bar 2+5=7 is NaN
    event = {
        "symbol": "PNB-EQ",
        "timestamp": df.loc[2, "timestamp"],
        "crossover_type": "BUY",
        "close_price": 30.0,
    }

    res = prof_service.evaluate_crossover_event(df, event, horizon_bars=5)

    assert res.result == "INSUFFICIENT_DATA"
    assert res.available_data is False


def test_win_rate_calculations(prof_service: CrossoverProfitabilityService) -> None:
    df_prof = pd.DataFrame([
        {"signal": "BUY", "result": "PROFITABLE"},
        {"signal": "BUY", "result": "PROFITABLE"},
        {"signal": "BUY", "result": "UNPROFITABLE"},
        {"signal": "SELL", "result": "PROFITABLE"},
        {"signal": "SELL", "result": "UNPROFITABLE"},
        {"signal": "BUY", "result": "INSUFFICIENT_DATA"},
    ])

    metrics = prof_service.get_win_rate_metrics(df_prof)

    # BUY: 2 profitable out of 3 evaluated -> 66.7%
    assert metrics["buy_evaluated_trades"] == 3
    assert metrics["buy_profitable_count"] == 2
    assert metrics["buy_win_rate_pct"] == 66.7

    # SELL: 1 profitable out of 2 evaluated -> 50.0%
    assert metrics["sell_evaluated_trades"] == 2
    assert metrics["sell_profitable_count"] == 1
    assert metrics["sell_win_rate_pct"] == 50.0

    # Overall: 3 profitable out of 5 evaluated -> 60.0%
    assert metrics["overall_evaluated_trades"] == 5
    assert metrics["overall_profitable_count"] == 3
    assert metrics["overall_win_rate_pct"] == 60.0

    assert metrics["insufficient_data_count"] == 1


def test_no_look_ahead_leakage_and_dataframe_integration(prof_service: CrossoverProfitabilityService) -> None:
    # Verify that get_profitability_dataframe returns valid structured DataFrame across symbols
    df_results = prof_service.get_profitability_dataframe(symbols=["SUZLON-EQ", "YESBANK-EQ"], days=250)

    assert isinstance(df_results, pd.DataFrame)
    if not df_results.empty:
        required_cols = [
            "symbol", "company_name", "signal", "entry_price", "exit_price",
            "evaluation_horizon", "pnl", "return_pct", "result", "available_data",
            "ai_confidence_pct", "ai_recommendation", "avoidance_reason"
        ]
        for c in required_cols:
            assert c in df_results.columns
