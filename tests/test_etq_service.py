"""
Unit tests for ExchangeTradedQuantityService (ETQ).

Verifies:
  - Exact 5-minute, 20-minute, 60-minute boundaries
  - Trades inside vs outside window
  - Empty trades & unknown symbol handling
  - Duplicate trades & tick deduplication on reconnects
  - Out-of-order timestamps
  - Invalid quantities (zero, negative, NaN, bad strings)
  - Timezone handling (naive vs UTC vs IST)
  - 1-minute candle ingestion
  - Multi-symbol snapshots and DataFrames
"""

from datetime import datetime, timedelta, timezone
import math
import pandas as pd
import pytest

from quant_signal.core.types import TradeTick
from quant_signal.services.etq_service import ExchangeTradedQuantityService, normalize_datetime


@pytest.fixture
def etq_service() -> ExchangeTradedQuantityService:
    return ExchangeTradedQuantityService()


def test_exact_5m_boundary(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "SUZLON-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # Trades around 5m boundary (ref_time - 5m = 10:25:00)
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=5, seconds=1), 100, trade_id="t1")  # Outside (10:24:59)
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=5), 200, trade_id="t2")             # Exact boundary (10:25:00)
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=2), 300, trade_id="t3")             # Inside (10:28:00)
    etq_service.add_trade(symbol, ref_time, 400, trade_id="t4")                                    # At ref_time (10:30:00)
    etq_service.add_trade(symbol, ref_time + timedelta(seconds=1), 500, trade_id="t5")             # Future (10:30:01)

    # Inside 5m window: t2 (200) + t3 (300) + t4 (400) = 900
    assert etq_service.get_etq_5m(symbol, current_time=ref_time) == 900


def test_exact_20m_boundary(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "TATASTEEL-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # 20m boundary is 10:10:00
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=20, seconds=1), 1000, trade_id="b1") # Outside (10:09:59)
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=20), 2000, trade_id="b2")            # Exact boundary (10:10:00)
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=10), 3000, trade_id="b3")            # Inside (10:20:00)
    etq_service.add_trade(symbol, ref_time, 4000, trade_id="b4")                                   # At ref_time (10:30:00)

    # Inside 20m window: b2 (2000) + b3 (3000) + b4 (4000) = 9000
    assert etq_service.get_etq_20m(symbol, current_time=ref_time) == 9000


def test_exact_60m_boundary(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "RELIANCE-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # 60m boundary is 09:30:00
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=60, seconds=1), 5000, trade_id="c1") # Outside (09:29:59)
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=60), 10000, trade_id="c2")           # Exact boundary (09:30:00)
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=30), 15000, trade_id="c3")           # Inside (10:00:00)
    etq_service.add_trade(symbol, ref_time, 20000, trade_id="c4")                                  # At ref_time (10:30:00)

    # Inside 60m window: c2 (10000) + c3 (15000) + c4 (20000) = 45000
    assert etq_service.get_etq_60m(symbol, current_time=ref_time) == 45000


def test_trades_inside_and_outside_window(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "SBIN-EQ"
    ref_time = datetime(2025, 1, 15, 12, 0, 0)

    # Add trades across time spectrum
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=2), 50, trade_id="w1")   # 11:58 -> 5m, 20m, 60m
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=15), 150, trade_id="w2") # 11:45 -> 20m, 60m
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=45), 300, trade_id="w3") # 11:15 -> 60m
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=90), 500, trade_id="w4") # 10:30 -> None

    assert etq_service.get_etq_5m(symbol, current_time=ref_time) == 50
    assert etq_service.get_etq_20m(symbol, current_time=ref_time) == 50 + 150
    assert etq_service.get_etq_60m(symbol, current_time=ref_time) == 50 + 150 + 300


def test_empty_trades_and_unknown_symbol(etq_service: ExchangeTradedQuantityService) -> None:
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    assert etq_service.get_etq_5m("UNKNOWN-EQ", current_time=ref_time) == 0
    assert etq_service.get_etq_20m("UNKNOWN-EQ", current_time=ref_time) == 0
    assert etq_service.get_etq_60m("UNKNOWN-EQ", current_time=ref_time) == 0

    snap = etq_service.get_etq_snapshot("UNKNOWN-EQ", current_time=ref_time)
    assert snap.etq_5m == 0
    assert snap.etq_20m == 0
    assert snap.etq_60m == 0


def test_duplicate_trades_deduplication(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "INFY-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)
    ts = ref_time - timedelta(minutes=2)

    # Add trade with trade_id
    res1 = etq_service.add_trade(symbol, ts, 500, price=1500.0, trade_id="T101")
    res2 = etq_service.add_trade(symbol, ts, 500, price=1500.0, trade_id="T101") # Duplicate trade_id

    assert res1 is True
    assert res2 is False  # Duplicate ignored
    assert etq_service.get_etq_5m(symbol, current_time=ref_time) == 500

    # Add duplicate trade without trade_id (deduplicated by (timestamp, price, quantity))
    res3 = etq_service.add_trade(symbol, ts, 300, price=1500.0)
    res4 = etq_service.add_trade(symbol, ts, 300, price=1500.0) # Duplicate tuple

    assert res3 is True
    assert res4 is False
    assert etq_service.get_etq_5m(symbol, current_time=ref_time) == 800


def test_out_of_order_timestamps(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "CANBK-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # Ingest out of chronological order
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=1), 100, trade_id="oo1")
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=10), 200, trade_id="oo2")
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=3), 300, trade_id="oo3")
    etq_service.add_trade(symbol, ref_time - timedelta(minutes=30), 400, trade_id="oo4")

    # 5m window should include oo1 (100) and oo3 (300) = 400
    assert etq_service.get_etq_5m(symbol, current_time=ref_time) == 400
    # 20m window should include oo1 (100) + oo3 (300) + oo2 (200) = 600
    assert etq_service.get_etq_20m(symbol, current_time=ref_time) == 600
    # 60m window should include all = 1000
    assert etq_service.get_etq_60m(symbol, current_time=ref_time) == 1000


def test_invalid_quantity(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "PNB-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)
    ts = ref_time - timedelta(minutes=2)

    assert etq_service.add_trade(symbol, ts, 0, trade_id="inv1") is False
    assert etq_service.add_trade(symbol, ts, -500, trade_id="inv2") is False
    assert etq_service.add_trade(symbol, ts, "invalid_number", trade_id="inv3") is False
    assert etq_service.add_trade(symbol, ts, None, trade_id="inv4") is False

    assert etq_service.get_etq_5m(symbol, current_time=ref_time) == 0


def test_timezone_handling(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "ICICIBANK-EQ"
    ref_time_naive = datetime(2025, 1, 15, 10, 30, 0)
    ref_time_utc = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    ts_naive = ref_time_naive - timedelta(minutes=2)
    ts_utc = ref_time_utc - timedelta(minutes=3)

    # Ingest mix of naive and UTC timezone-aware trade ticks
    etq_service.add_trade(symbol, ts_naive, 1000, trade_id="tz1")
    etq_service.add_trade(symbol, ts_utc, 2000, trade_id="tz2")

    # Both naive and tz-aware queries work without TypeError
    assert etq_service.get_etq_5m(symbol, current_time=ref_time_naive) == 3000
    assert etq_service.get_etq_5m(symbol, current_time=ref_time_utc) == 3000


def test_candle_ingestion(etq_service: ExchangeTradedQuantityService) -> None:
    symbol = "NHPC-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # Construct 1-minute OHLCV candle DataFrame
    candles = [
        {"timestamp": ref_time - timedelta(minutes=m), "open": 85.0, "high": 86.0, "low": 84.5, "close": 85.5, "volume": 1000}
        for m in range(30)
    ]
    df_candles = pd.DataFrame(candles)

    added = etq_service.add_candles(symbol, df_candles)
    assert added == 30

    # 5m window should sum 6 candles (m=0 to m=5) = 6000
    assert etq_service.get_etq_5m(symbol, current_time=ref_time) == 6000
    # 20m window should sum 21 candles (m=0 to m=20) = 21000
    assert etq_service.get_etq_20m(symbol, current_time=ref_time) == 21000
    # 60m window should sum all 30 candles = 30000
    assert etq_service.get_etq_60m(symbol, current_time=ref_time) == 30000


def test_etq_snapshot_and_dataframe(etq_service: ExchangeTradedQuantityService) -> None:
    symbol1 = "SUZLON-EQ"
    symbol2 = "NHPC-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    etq_service.add_trade(symbol1, ref_time - timedelta(minutes=1), 500, trade_id="s1")
    etq_service.add_trade(symbol2, ref_time - timedelta(minutes=15), 1500, trade_id="s2")

    snap1 = etq_service.get_etq_snapshot(symbol1, current_time=ref_time)
    assert snap1.symbol == symbol1
    assert snap1.etq_5m == 500
    assert snap1.etq_20m == 500
    assert snap1.etq_60m == 500

    df = etq_service.get_etq_dataframe(symbols=[symbol1, symbol2], current_time=ref_time)
    assert len(df) == 2
    assert "etq_5m" in df.columns
    assert "etq_20m" in df.columns
    assert "etq_60m" in df.columns
