"""
Average LTP (Price) Service for QuantSignal AI.

Calculates rolling Average LTP (Last Traded Price) over 20-minute and 60-minute lookback windows.

Averaging Methodology:
  Average LTP = sum(valid LTP observations in window) / number of valid observations

Requirements handled:
  - Exact 20m and 60m time window boundaries.
  - Safe timezone normalization (handles naive, UTC, and IST datetimes seamlessly).
  - Price observation deduplication to avoid duplicate counting on WebSocket reconnects.
  - Out-of-order price observation sorting.
  - Filtering out non-numeric, negative, zero, NaN, or Inf price values.
  - Integration with 1-minute OHLCV candles and live tick feeds.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import threading
from typing import Any, Sequence

import pandas as pd

from quant_signal.core.types import AveragePriceSnapshot, PriceObservation
from quant_signal.logger import get_logger
from quant_signal.services.etq_service import normalize_datetime

logger = get_logger(__name__)


class AveragePriceService:
    """
    Thread-safe Average LTP tracking & calculation service.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Symbol -> list of PriceObservation
        self._observations: dict[str, list[PriceObservation]] = {}
        # Symbol -> set of seen observation keys
        self._seen_keys: dict[str, set[Any]] = {}

    def clear(self, symbol: str | None = None) -> None:
        """Clears accumulated price observations for a specific symbol or all symbols."""
        with self._lock:
            if symbol:
                self._observations.pop(symbol, None)
                self._seen_keys.pop(symbol, None)
            else:
                self._observations.clear()
                self._seen_keys.clear()

    # ── Observation Ingestion ─────────────────────────────────────────────────

    def add_price_observation(
        self,
        symbol: str,
        timestamp: datetime | str | Any,
        price: float | int | Any,
        observation_id: str | None = None,
    ) -> bool:
        """
        Adds a single price observation to the tracking cache.
        Returns True if observation was ingested, False if duplicate or invalid.
        """
        if not symbol:
            return False

        # 1. Validate price
        try:
            prc = float(price)
        except (ValueError, TypeError):
            return False

        if prc <= 0.0 or math.isnan(prc) or math.isinf(prc):
            return False

        # 2. Normalize timestamp
        norm_ts = normalize_datetime(timestamp)

        # 3. Deduplication key
        key: Any = observation_id if observation_id else (norm_ts.isoformat(), round(prc, 4))

        with self._lock:
            if symbol not in self._observations:
                self._observations[symbol] = []
                self._seen_keys[symbol] = set()

            if key in self._seen_keys[symbol]:
                return False  # Duplicate observation ignored

            self._seen_keys[symbol].add(key)
            obs = PriceObservation(
                symbol=symbol,
                timestamp=norm_ts,
                price=prc,
                observation_id=observation_id,
            )
            self._observations[symbol].append(obs)
            return True

    def add_price_observations(
        self,
        symbol: str,
        observations: Sequence[PriceObservation | dict[str, Any] | tuple[Any, ...]],
    ) -> int:
        """
        Bulk ingests multiple price observations for a symbol. Returns count of new observations added.
        """
        added_count = 0
        for item in observations:
            if isinstance(item, PriceObservation):
                if self.add_price_observation(
                    symbol=item.symbol or symbol,
                    timestamp=item.timestamp,
                    price=item.price,
                    observation_id=item.observation_id,
                ):
                    added_count += 1
            elif isinstance(item, dict):
                sym = item.get("symbol", symbol)
                ts = item.get("timestamp") or item.get("time") or item.get("datetime")
                prc = item.get("price") or item.get("ltp") or item.get("close")
                oid = item.get("observation_id") or item.get("id")
                if self.add_price_observation(symbol=sym, timestamp=ts, price=prc, observation_id=oid):
                    added_count += 1
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                ts, prc = item[0], item[1]
                oid = str(item[2]) if len(item) > 2 else None
                if self.add_price_observation(symbol=symbol, timestamp=ts, price=prc, observation_id=oid):
                    added_count += 1
        return added_count

    def add_candles(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Ingests 1-minute OHLCV candles using candle 'close' as minute price observation.
        Returns count of new minute bars added.
        """
        if df is None or df.empty or "close" not in df.columns:
            return 0

        ts_col = "timestamp" if "timestamp" in df.columns else ("date" if "date" in df.columns else None)
        if not ts_col:
            return 0

        added_count = 0
        for _, row in df.iterrows():
            ts = row[ts_col]
            prc = row["close"]
            norm_ts = normalize_datetime(ts)
            cid = f"candle_1m_ltp_{symbol}_{norm_ts.isoformat()}"
            if self.add_price_observation(symbol=symbol, timestamp=ts, price=prc, observation_id=cid):
                added_count += 1

        return added_count

    # ── Core Window Calculation ───────────────────────────────────────────────

    def calculate_average_ltp_for_window(
        self,
        symbol: str,
        window_minutes: int,
        current_time: datetime | str | Any | None = None,
    ) -> tuple[float, int]:
        """
        Calculates Average LTP and observation sample count for a given window (in minutes).

        Formula:
          Average LTP = sum(valid LTP observations in window) / number of valid observations

        Window rule:
          cutoff_time = current_time - timedelta(minutes=window_minutes)
          Observations included: cutoff_time <= obs.timestamp <= current_time
        """
        if window_minutes <= 0:
            return 0.0, 0

        ref_time = normalize_datetime(current_time) if current_time is not None else None

        with self._lock:
            obs_list = self._observations.get(symbol, [])
            if not obs_list:
                return 0.0, 0

            # Default to latest observation timestamp if reference time not provided
            if ref_time is None:
                ref_time = max(o.timestamp for o in obs_list)

            cutoff_time = ref_time - timedelta(minutes=window_minutes)

            # Gather valid observations within [cutoff_time, ref_time]
            valid_prices = [
                o.price for o in obs_list
                if cutoff_time <= o.timestamp <= ref_time and o.price > 0.0 and not math.isnan(o.price)
            ]

            if not valid_prices:
                return 0.0, 0

            avg_price = round(sum(valid_prices) / len(valid_prices), 2)
            return avg_price, len(valid_prices)

    def get_average_ltp_20m(
        self, symbol: str, current_time: datetime | str | Any | None = None
    ) -> float:
        """Returns Average LTP for the last 20 minutes."""
        avg_price, _ = self.calculate_average_ltp_for_window(
            symbol=symbol, window_minutes=20, current_time=current_time
        )
        return avg_price

    def get_average_ltp_60m(
        self, symbol: str, current_time: datetime | str | Any | None = None
    ) -> float:
        """Returns Average LTP for the last 60 minutes."""
        avg_price, _ = self.calculate_average_ltp_for_window(
            symbol=symbol, window_minutes=60, current_time=current_time
        )
        return avg_price

    # ── Snapshot & DataFrame Interfaces ──────────────────────────────────────

    def get_average_price_snapshot(
        self,
        symbol: str,
        current_time: datetime | str | Any | None = None,
    ) -> AveragePriceSnapshot:
        """Returns an AveragePriceSnapshot object for a specific symbol."""
        ref_time = normalize_datetime(current_time) if current_time is not None else datetime.utcnow()
        avg_20, count_20 = self.calculate_average_ltp_for_window(symbol=symbol, window_minutes=20, current_time=ref_time)
        avg_60, count_60 = self.calculate_average_ltp_for_window(symbol=symbol, window_minutes=60, current_time=ref_time)

        return AveragePriceSnapshot(
            symbol=symbol,
            avg_ltp_20m=avg_20,
            avg_ltp_60m=avg_60,
            sample_count_20m=count_20,
            sample_count_60m=count_60,
            timestamp=ref_time,
        )

    def get_average_price_dataframe(
        self,
        symbols: list[str] | None = None,
        current_time: datetime | str | Any | None = None,
    ) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame containing Average LTP metrics (avg_ltp_20m, avg_ltp_60m)
        across specified symbols.
        """
        with self._lock:
            target_symbols = symbols if symbols is not None else list(self._observations.keys())

        rows = []
        for sym in target_symbols:
            snap = self.get_average_price_snapshot(symbol=sym, current_time=current_time)
            rows.append({
                "symbol": sym,
                "avg_ltp_20m": snap.avg_ltp_20m,
                "avg_ltp_60m": snap.avg_ltp_60m,
                "sample_count_20m": snap.sample_count_20m,
                "sample_count_60m": snap.sample_count_60m,
                "timestamp": snap.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            })

        if not rows:
            return pd.DataFrame(columns=[
                "symbol", "avg_ltp_20m", "avg_ltp_60m", "sample_count_20m", "sample_count_60m", "timestamp"
            ])

        return pd.DataFrame(rows)
