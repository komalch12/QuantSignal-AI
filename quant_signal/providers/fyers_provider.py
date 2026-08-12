"""
Fyers Market Data Provider for QuantSignal AI.

Thin wrapper around FyersBrokerAdapter that implements MarketDataProvider.
Use this in production when valid Fyers API credentials are configured.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_signal.brokers.fyers_adapter import FyersBrokerAdapter
from quant_signal.core.types import MarketDepthSnapshot, DepthLevel, StockSnapshot
from quant_signal.logger import get_logger
from quant_signal.providers.base import MarketDataProvider

logger = get_logger(__name__)


class FyersMarketDataProvider(MarketDataProvider):
    """
    Production MarketDataProvider backed by Fyers API v3.
    Requires valid FYERS_CLIENT_ID and FYERS_SECRET_KEY in .env.
    """

    def __init__(self, broker_adapter: FyersBrokerAdapter) -> None:
        self._adapter = broker_adapter

    @property
    def provider_name(self) -> str:
        return "Fyers"

    @property
    def is_live(self) -> bool:
        status = self._adapter.get_connection_status()
        return status.is_authenticated

    def get_symbols(self) -> list[str]:
        # Symbol universe is loaded by StockScannerService from NSE_CM.csv
        return []

    def get_quotes(self, symbols: list[str] | None = None) -> list[StockSnapshot]:
        if not symbols:
            return []
        client = self._adapter.auth_service.get_fyers_client()
        if not client:
            return []
        try:
            res = client.quotes({"symbols": ",".join(symbols)})
            if isinstance(res, dict) and res.get("s") == "ok":
                result = []
                for item in res.get("d", []):
                    v = item.get("v", {})
                    sym = item.get("n", "").replace("NSE:", "")
                    result.append(StockSnapshot(
                        symbol=sym,
                        company_name=v.get("description", ""),
                        exchange="NSE",
                        ltp=float(v.get("lp", 0.0)),
                        timestamp=datetime.now(),
                    ))
                return result
        except Exception as e:
            logger.error(f"FyersMarketDataProvider.get_quotes error: {e}")
        return []

    def get_market_depth(self, symbol: str) -> MarketDepthSnapshot:
        client = self._adapter.auth_service.get_fyers_client()
        if not client:
            return MarketDepthSnapshot(symbol=symbol)
        try:
            res = client.quotes({"symbols": f"NSE:{symbol}"})
            if isinstance(res, dict) and res.get("s") == "ok":
                d_list = res.get("d", [])
                if d_list:
                    v = d_list[0].get("v", {})
                    ltp = float(v.get("lp", 0.0))
                    bids = [
                        DepthLevel(
                            price=float(b.get("price", 0)),
                            volume=int(b.get("volume", 0)),
                            orders=int(b.get("orders", 0)),
                        )
                        for b in v.get("bids", [])
                    ]
                    asks = [
                        DepthLevel(
                            price=float(a.get("price", 0)),
                            volume=int(a.get("volume", 0)),
                            orders=int(a.get("orders", 0)),
                        )
                        for a in v.get("asks", [])
                    ]
                    return MarketDepthSnapshot(
                        symbol=symbol, ltp=ltp, bids=bids or [DepthLevel() for _ in range(5)],
                        asks=asks or [DepthLevel() for _ in range(5)], timestamp=datetime.now(),
                    )
        except Exception as e:
            logger.error(f"FyersMarketDataProvider.get_market_depth error: {e}")
        return MarketDepthSnapshot(symbol=symbol)

    def get_historical_data(self, symbol: str, timeframe: str, days: int = 60) -> pd.DataFrame:
        # Placeholder — implement via Fyers history API when needed
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
