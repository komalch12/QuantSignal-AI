"""
Unit tests for AveragePriceService (Average LTP 20m & 60m).

Verifies:
  1. Exact 20-minute boundary
  2. Exact 60-minute boundary
  3. Values inside the window
  4. Values outside the window
  5. Empty data & unknown symbol
  6. Missing values / null inputs
  7. NaN / Inf values
  8. Invalid prices (zero, negative)
  9. Duplicate observations deduplication
  10. Out-of-order timestamps
  11. Timezone handling (naive vs UTC vs IST)
  12. Known price series with deterministic expected average
  13. Snapshot & DataFrame integration
"""

from datetime import datetime, timedelta, timezone
import math
import pandas as pd
import pytest

from quant_signal.core.types import PriceObservation
from quant_signal.services.average_price_service import AveragePriceService


@pytest.fixture
def avg_service() -> AveragePriceService:
    return AveragePriceService()


def test_exact_20m_boundary(avg_service: AveragePriceService) -> None:
    symbol = "SUZLON-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # 20m boundary is 10:10:00
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=20, seconds=1), 10.0, observation_id="out1") # 10:09:59 (Outside)
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=20), 20.0, observation_id="b1")             # 10:10:00 (Exact boundary)
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=10), 40.0, observation_id="b2")             # 10:20:00 (Inside)
    avg_service.add_price_observation(symbol, ref_time, 60.0, observation_id="b3")                                    # 10:30:00 (Ref time)

    # Valid in 20m window: b1 (20.0), b2 (40.0), b3 (60.0) -> sum=120.0, count=3 -> avg = 40.0
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 40.0


def test_exact_60m_boundary(avg_service: AveragePriceService) -> None:
    symbol = "TATASTEEL-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # 60m boundary is 09:30:00
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=60, seconds=1), 10.0, observation_id="out1") # 09:29:59 (Outside)
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=60), 100.0, observation_id="c1")             # 09:30:00 (Exact boundary)
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=30), 200.0, observation_id="c2")             # 10:00:00 (Inside)
    avg_service.add_price_observation(symbol, ref_time, 300.0, observation_id="c3")                                    # 10:30:00 (Ref time)

    # Valid in 60m window: c1 (100), c2 (200), c3 (300) -> sum=600, count=3 -> avg = 200.0
    assert avg_service.get_average_ltp_60m(symbol, current_time=ref_time) == 200.0


def test_values_inside_and_outside_window(avg_service: AveragePriceService) -> None:
    symbol = "RELIANCE-EQ"
    ref_time = datetime(2025, 1, 15, 12, 0, 0)

    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=5), 100.0, observation_id="v1")  # 11:55 -> 20m & 60m
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=15), 200.0, observation_id="v2") # 11:45 -> 20m & 60m
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=45), 300.0, observation_id="v3") # 11:15 -> 60m only
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=90), 500.0, observation_id="v4") # 10:30 -> Outside both

    # 20m window: v1 (100.0) + v2 (200.0) -> avg = 150.0
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 150.0
    # 60m window: v1 (100.0) + v2 (200.0) + v3 (300.0) -> avg = 200.0
    assert avg_service.get_average_ltp_60m(symbol, current_time=ref_time) == 200.0


def test_empty_data_and_unknown_symbol(avg_service: AveragePriceService) -> None:
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    assert avg_service.get_average_ltp_20m("UNKNOWN-EQ", current_time=ref_time) == 0.0
    assert avg_service.get_average_ltp_60m("UNKNOWN-EQ", current_time=ref_time) == 0.0

    snap = avg_service.get_average_price_snapshot("UNKNOWN-EQ", current_time=ref_time)
    assert snap.avg_ltp_20m == 0.0
    assert snap.avg_ltp_60m == 0.0
    assert snap.sample_count_20m == 0
    assert snap.sample_count_60m == 0


def test_missing_values_and_null_inputs(avg_service: AveragePriceService) -> None:
    symbol = "SBIN-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)
    ts = ref_time - timedelta(minutes=2)

    assert avg_service.add_price_observation(symbol, ts, None) is False
    assert avg_service.add_price_observation(symbol, ts, "invalid_price") is False
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 0.0


def test_nan_and_inf_values(avg_service: AveragePriceService) -> None:
    symbol = "INFY-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)
    ts = ref_time - timedelta(minutes=2)

    assert avg_service.add_price_observation(symbol, ts, float("nan")) is False
    assert avg_service.add_price_observation(symbol, ts, float("inf")) is False
    assert avg_service.add_price_observation(symbol, ts, float("-inf")) is False

    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 0.0


def test_invalid_prices_zero_negative(avg_service: AveragePriceService) -> None:
    symbol = "PNB-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)
    ts = ref_time - timedelta(minutes=2)

    assert avg_service.add_price_observation(symbol, ts, 0.0) is False
    assert avg_service.add_price_observation(symbol, ts, -100.5) is False

    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 0.0


def test_duplicate_observations_deduplication(avg_service: AveragePriceService) -> None:
    symbol = "CANBK-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)
    ts = ref_time - timedelta(minutes=5)

    res1 = avg_service.add_price_observation(symbol, ts, 100.0, observation_id="OBS1")
    res2 = avg_service.add_price_observation(symbol, ts, 100.0, observation_id="OBS1") # Duplicate ID

    assert res1 is True
    assert res2 is False  # Duplicate ignored
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 100.0


def test_out_of_order_timestamps(avg_service: AveragePriceService) -> None:
    symbol = "NHPC-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    # Ingest out of chronological order
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=1), 100.0, observation_id="o1")
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=15), 200.0, observation_id="o2")
    avg_service.add_price_observation(symbol, ref_time - timedelta(minutes=5), 300.0, observation_id="o3")

    # In 20m window: o1 (100.0), o3 (300.0), o2 (200.0) -> sum = 600.0, count = 3 -> avg = 200.0
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 200.0


def test_timezone_handling(avg_service: AveragePriceService) -> None:
    symbol = "ICICIBANK-EQ"
    ref_time_naive = datetime(2025, 1, 15, 10, 30, 0)
    ref_time_utc = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    ts_naive = ref_time_naive - timedelta(minutes=2)
    ts_utc = ref_time_utc - timedelta(minutes=3)

    avg_service.add_price_observation(symbol, ts_naive, 100.0, observation_id="tz1")
    avg_service.add_price_observation(symbol, ts_utc, 200.0, observation_id="tz2")

    # Both naive and UTC queries return expected 150.0 without throwing TypeError
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time_naive) == 150.0
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time_utc) == 150.0


def test_known_price_series_deterministic_average(avg_service: AveragePriceService) -> None:
    symbol = "AXISBANK-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    prices = [100.0, 110.0, 120.0, 130.0, 140.0]
    for idx, p in enumerate(prices):
        avg_service.add_price_observation(
            symbol, ref_time - timedelta(minutes=idx * 2), p, observation_id=f"det_{idx}"
        )

    # Known sum = 100 + 110 + 120 + 130 + 140 = 600 / 5 = 120.0
    assert avg_service.get_average_ltp_20m(symbol, current_time=ref_time) == 120.0


def test_snapshot_and_dataframe_integration(avg_service: AveragePriceService) -> None:
    symbol1 = "SUZLON-EQ"
    symbol2 = "NHPC-EQ"
    ref_time = datetime(2025, 1, 15, 10, 30, 0)

    avg_service.add_price_observation(symbol1, ref_time - timedelta(minutes=1), 50.0, observation_id="s1")
    avg_service.add_price_observation(symbol2, ref_time - timedelta(minutes=10), 80.0, observation_id="s2")

    snap1 = avg_service.get_average_price_snapshot(symbol1, current_time=ref_time)
    assert snap1.symbol == symbol1
    assert snap1.avg_ltp_20m == 50.0
    assert snap1.avg_ltp_60m == 50.0
    assert snap1.sample_count_20m == 1

    df = avg_service.get_average_price_dataframe(symbols=[symbol1, symbol2], current_time=ref_time)
    assert len(df) == 2
    assert "avg_ltp_20m" in df.columns
    assert "avg_ltp_60m" in df.columns
