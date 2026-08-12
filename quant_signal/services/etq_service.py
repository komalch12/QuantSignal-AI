"""
Exchange Traded Quantity (ETQ) Service for QuantSignal AI.

Calculates total executed exchange trade volume over 5-minute, 20-minute,
and 60-minute lookback windows.

Requirements handled:
  - Exact 5m, 20m, 60m time window boundaries.
  - Safe timezone normalization (handles naive, UTC, and IST datetimes seamlessly).
  - Trade tick deduplication to avoid double-counting on WebSocket reconnects.
  - Out-of-order trade tick sorting.
  - Sanitization of non-numeric, negative, or zero quantities.
  - Integration with 1-minute OHLCV candles and live tick feeds.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import threading
from typing import Any, Sequence

import pandas as pd

from quant_signal.core.types import ETQSnapshot, TradeTick
from quant_signal.logger import get_logger

logger = get_logger(__name__)


def normalize_datetime(dt: datetime | Any) -> datetime:
    """
    Normalizes a datetime object into a naive UTC datetime object for consistent comparisons.
    
    If given a string or Pandas Timestamp, parses it safely into a Python datetime first.
    If timezone-aware, converts to UTC and drops timezone info (making it naive UTC).
    If naive, assumes local/UTC naive datetime as-is.
    """
    if dt is None:
        return datetime.utcnow()

    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt).to_pydatetime()
        except Exception:
            return datetime.utcnow()
    elif hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()

    if not isinstance(dt, datetime):
        return datetime.utcnow()

    if dt.tzinfo is not None:
        # Convert to UTC and strip tzinfo for safe, offset-naive comparison
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.replace(tzinfo=None)

    return dt


class ExchangeTradedQuantityService:
    """
    Thread-safe Exchange Traded Quantity (ETQ) tracking & calculation service.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Symbol -> list of TradeTick
        self._trades: dict[str, list[TradeTick]] = {}
        # Symbol -> set of seen trade keys (trade_id or (timestamp, price, quantity))
        self._seen_keys: dict[str, set[Any]] = {}

    def clear(self, symbol: str | None = None) -> None:
        """Clears accumulated trade cache for a specific symbol or all symbols."""
        with self._lock:
            if symbol:
                self._trades.pop(symbol, None)
                self._seen_keys.pop(symbol, None)
            else:
                self._trades.clear()
                self._seen_keys.clear()

    # ── Trade Ingestion ───────────────────────────────────────────────────────

    def add_trade(
        self,
        symbol: str,
        timestamp: datetime | str | Any,
        quantity: int | float | Any,
        price: float = 0.0,
        trade_id: str | None = None,
    ) -> bool:
        """
        Adds a single trade tick to the ETQ tracking cache.
        Returns True if trade was ingested, False if duplicate or invalid.
        """
        if not symbol:
            return False

        # 1. Validate quantity
        try:
            qty = int(quantity)
        except (ValueError, TypeError):
            return False

        if qty <= 0 or math.isnan(qty):
            return False

        # 2. Normalize timestamp
        norm_ts = normalize_datetime(timestamp)

        # 3. Deduplication key
        key: Any = trade_id if trade_id else (norm_ts.isoformat(), float(price), qty)

        with self._lock:
            if symbol not in self._trades:
                self._trades[symbol] = []
                self._seen_keys[symbol] = set()

            if key in self._seen_keys[symbol]:
                return False  # Duplicate tick ignored to prevent double-counting

            self._seen_keys[symbol].add(key)
            tick = TradeTick(
                symbol=symbol,
                timestamp=norm_ts,
                quantity=qty,
                price=float(price),
                trade_id=trade_id,
            )
            self._trades[symbol].append(tick)
            return True

    def add_trades(
        self,
        symbol: str,
        trades: Sequence[TradeTick | dict[str, Any] | tuple[Any, ...]],
    ) -> int:
        """
        Bulk ingests multiple trades for a symbol. Returns count of new trades added.
        """
        added_count = 0
        for item in trades:
            if isinstance(item, TradeTick):
                if self.add_trade(
                    symbol=item.symbol or symbol,
                    timestamp=item.timestamp,
                    quantity=item.quantity,
                    price=item.price,
                    trade_id=item.trade_id,
                ):
                    added_count += 1
            elif isinstance(item, dict):
                sym = item.get("symbol", symbol)
                ts = item.get("timestamp") or item.get("time") or item.get("datetime")
                qty = item.get("quantity") or item.get("qty") or item.get("volume") or item.get("vol")
                prc = item.get("price", 0.0)
                tid = item.get("trade_id") or item.get("id")
                if self.add_trade(symbol=sym, timestamp=ts, quantity=qty, price=prc, trade_id=tid):
                    added_count += 1
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                ts, qty = item[0], item[1]
                prc = item[2] if len(item) > 2 else 0.0
                tid = str(item[3]) if len(item) > 3 else None
                if self.add_trade(symbol=symbol, timestamp=ts, quantity=qty, price=prc, trade_id=tid):
                    added_count += 1
        return added_count

    def add_candles(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Ingests 1-minute OHLCV candles where candle 'volume' represents executed trade quantity.
        Returns count of new minute bars added.
        """
        if df is None or df.empty or "volume" not in df.columns:
            return 0

        ts_col = "timestamp" if "timestamp" in df.columns else ("date" if "date" in df.columns else None)
        if not ts_col:
            return 0

        added_count = 0
        for _, row in df.iterrows():
            ts = row[ts_col]
            vol = row["volume"]
            prc = row["close"] if "close" in row.index else 0.0
            # Use candle timestamp as unique trade_id key for candle bar deduplication
            norm_ts = normalize_datetime(ts)
            cid = f"candle_1m_{symbol}_{norm_ts.isoformat()}"
            if self.add_trade(symbol=symbol, timestamp=ts, quantity=vol, price=prc, trade_id=cid):
                added_count += 1

        return added_count

    # ── Core Window Calculation ───────────────────────────────────────────────

    def calculate_etq_for_window(
        self,
        symbol: str,
        window_minutes: int,
        current_time: datetime | str | Any | None = None,
    ) -> int:
        """
        Calculates total executed trade quantity for a given window (in minutes).

        Window rule:
          cutoff_time = current_time - timedelta(minutes=window_minutes)
          Trades included: cutoff_time <= trade.timestamp <= current_time
        """
        if window_minutes <= 0:
            return 0

        ref_time = normalize_datetime(current_time) if current_time is not None else None

        with self._lock:
            ticks = self._trades.get(symbol, [])
            if not ticks:
                return 0

            # If ref_time not provided, default to the latest trade timestamp available for this symbol
            if ref_time is None:
                ref_time = max(t.timestamp for t in ticks)

            cutoff_time = ref_time - timedelta(minutes=window_minutes)

            # Sum quantity for trades falling within the [cutoff_time, ref_time] window
            total_qty = 0
            for t in ticks:
                if cutoff_time <= t.timestamp <= ref_time:
                    total_qty += t.quantity

            return total_qty

    def get_etq_5m(self, symbol: str, current_time: datetime | str | Any | None = None) -> int:
        """Returns total Exchange Traded Quantity in the last 5 minutes."""
        return self.calculate_etq_for_window(symbol=symbol, window_minutes=5, current_time=current_time)

    def get_etq_20m(self, symbol: str, current_time: datetime | str | Any | None = None) -> int:
        """Returns total Exchange Traded Quantity in the last 20 minutes."""
        return self.calculate_etq_for_window(symbol=symbol, window_minutes=20, current_time=current_time)

    def get_etq_60m(self, symbol: str, current_time: datetime | str | Any | None = None) -> int:
        """Returns total Exchange Traded Quantity in the last 60 minutes."""
        return self.calculate_etq_for_window(symbol=symbol, window_minutes=60, current_time=current_time)

    # ── Snapshot & DataFrame Interfaces ──────────────────────────────────────

    def get_etq_snapshot(
        self,
        symbol: str,
        current_time: datetime | str | Any | None = None,
    ) -> ETQSnapshot:
        """Returns an ETQSnapshot object for a specific symbol."""
        ref_time = normalize_datetime(current_time) if current_time is not None else datetime.utcnow()
        e5 = self.get_etq_5m(symbol=symbol, current_time=ref_time)
        e20 = self.get_etq_20m(symbol=symbol, current_time=ref_time)
        e60 = self.get_etq_60m(symbol=symbol, current_time=ref_time)

        return ETQSnapshot(
            symbol=symbol,
            etq_5m=e5,
            etq_20m=e20,
            etq_60m=e60,
            timestamp=ref_time,
        )

    def get_etq_dataframe(
        self,
        symbols: list[str] | None = None,
        current_time: datetime | str | Any | None = None,
    ) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame containing ETQ metrics (etq_5m, etq_20m, etq_60m)
        across specified symbols.
        """
        with self._lock:
            target_symbols = symbols if symbols is not None else list(self._trades.keys())

        rows = []
        for sym in target_symbols:
            snap = self.get_etq_snapshot(symbol=sym, current_time=current_time)
            rows.append({
                "symbol": sym,
                "etq_5m": snap.etq_5m,
                "etq_20m": snap.etq_20m,
                "etq_60m": snap.etq_60m,
                "timestamp": snap.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            })

        if not rows:
            return pd.DataFrame(columns=["symbol", "etq_5m", "etq_20m", "etq_60m", "timestamp"])

        return pd.DataFrame(rows)
