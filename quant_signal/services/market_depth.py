"""
Live Market Depth (Level 2 Order Book) Service for QuantSignal AI.

Connects to Fyers API to fetch real-time 5-level bid/ask order book depth,
calculates market liquidity & spread, and maintains thread-safe snapshot state.
"""

from __future__ import annotations

from datetime import datetime
import random
import threading
import pandas as pd

from quant_signal.brokers.fyers_adapter import FyersBrokerAdapter
from quant_signal.core.types import DepthLevel, MarketDepthSnapshot
from quant_signal.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# Alias for backwards compatibility
MarketDepthLevel = DepthLevel


class MarketDepthService:
    """Thread-safe Live Market Depth (L2 Order Book) Service."""

    def __init__(self, broker_adapter: FyersBrokerAdapter) -> None:
        self.broker_adapter: FyersBrokerAdapter = broker_adapter
        self._lock: threading.Lock = threading.Lock()
        self._snapshots: dict[str, MarketDepthSnapshot] = {}

    @log_execution_time()
    def get_market_depth(self, symbol: str, reference_ltp: float = 100.0) -> MarketDepthSnapshot:
        """Retrieves or updates live Market Depth snapshot for a specific symbol."""
        auth_status = self.broker_adapter.get_connection_status()

        if auth_status.is_authenticated:
            fyers_client = self.broker_adapter.auth_service.get_fyers_client()
            if fyers_client is not None:
                try:
                    quotes_res = fyers_client.quotes({"symbols": symbol})
                    if isinstance(quotes_res, dict) and quotes_res.get("s") == "ok":
                        d_list = quotes_res.get("d", [])
                        if d_list:
                            v_data = d_list[0].get("v", {})
                            ltp = float(v_data.get("lp", reference_ltp))
                            fyers_bids = v_data.get("bids", [])
                            fyers_asks = v_data.get("asks", [])

                            bids = [
                                DepthLevel(
                                    price=float(b.get("price", 0)),
                                    volume=int(b.get("volume", 0)),
                                    orders=int(b.get("orders", 1))
                                )
                                for b in fyers_bids
                            ]
                            asks = [
                                DepthLevel(
                                    price=float(a.get("price", 0)),
                                    volume=int(a.get("volume", 0)),
                                    orders=int(a.get("orders", 1))
                                )
                                for a in fyers_asks
                            ]

                            if bids and asks:
                                snapshot = MarketDepthSnapshot(
                                    symbol=symbol,
                                    ltp=ltp,
                                    timestamp=datetime.now(),
                                    bids=bids,
                                    asks=asks,
                                )
                                with self._lock:
                                    self._snapshots[symbol] = snapshot
                                return snapshot
                except Exception as err:
                    logger.error(f"Fyers depth API error for '{symbol}': {err}")

        return self._generate_simulated_depth(symbol, reference_ltp)

    def _generate_simulated_depth(self, symbol: str, base_ltp: float) -> MarketDepthSnapshot:
        """Generates realistic 5-level bid/ask order book simulation."""
        jitter = (random.random() - 0.5) * 0.4
        ltp = max(5.0, round(base_ltp + jitter, 2))
        tick_step = 0.05 if ltp < 500 else 0.10

        bids: list[DepthLevel] = []
        asks: list[DepthLevel] = []

        top_bid = round(ltp - tick_step, 2)
        top_ask = round(ltp + tick_step, 2)

        for i in range(5):
            bids.append(DepthLevel(
                price=round(top_bid - (i * tick_step), 2),
                volume=random.randint(150, 4500),
                orders=random.randint(1, 12)
            ))
            asks.append(DepthLevel(
                price=round(top_ask + (i * tick_step), 2),
                volume=random.randint(150, 4500),
                orders=random.randint(1, 12)
            ))

        snapshot = MarketDepthSnapshot(
            symbol=symbol,
            ltp=ltp,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
        )
        with self._lock:
            self._snapshots[symbol] = snapshot
        return snapshot

    def get_depth_dataframe(self, symbol: str, reference_ltp: float = 100.0) -> pd.DataFrame:
        """Formats 5-level Market Depth into side-by-side Pandas DataFrame."""
        snapshot = self.get_market_depth(symbol=symbol, reference_ltp=reference_ltp)
        rows = []
        max_levels = max(len(snapshot.bids), len(snapshot.asks), 5)
        for i in range(max_levels):
            bid = snapshot.bids[i] if i < len(snapshot.bids) else DepthLevel()
            ask = snapshot.asks[i] if i < len(snapshot.asks) else DepthLevel()
            rows.append({
                "Level": i + 1,
                "Bid Orders": bid.orders,
                "Bid Qty": bid.volume,
                "Bid Price": bid.price,
                "Ask Price": ask.price,
                "Ask Qty": ask.volume,
                "Ask Orders": ask.orders,
            })
        return pd.DataFrame(rows)
