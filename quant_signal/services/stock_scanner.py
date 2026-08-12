"""
Live NSE Stock Scanner Service for QuantSignal AI.
"""

from __future__ import annotations

import csv
import io
import threading
import time
import urllib.request
from datetime import datetime
from typing import Any
import pandas as pd

from quant_signal.brokers.fyers_adapter import FyersBrokerAdapter
from quant_signal.core.types import StockSnapshot
from quant_signal.logger import get_logger
from quant_signal.services.fyers_auth_service import ConnectionState

logger = get_logger(__name__)


class StockScannerService:
    def __init__(
        self,
        broker_adapter: FyersBrokerAdapter,
        min_price: float = 30.0,
        max_price: float = 500.0,
    ) -> None:
        self.broker_adapter = broker_adapter
        self.min_price = min_price
        self.max_price = max_price

        self._lock = threading.Lock()
        self._snapshots: dict[str, StockSnapshot] = {}
        self._last_error: str | None = None

        self.universe_loaded = False
        self.universe_count = 0
        self.live_symbols_received: set[str] = set()
        self.fyers_ws = None

        # Start background worker for WebSocket connection
        self._worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._worker_thread.start()

    def _load_universe(self) -> bool:
        if self.universe_loaded:
            return True
        try:
            logger.info("Downloading Fyers NSE_CM symbol master...")
            url = "https://public.fyers.in/sym_details/NSE_CM.csv"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                decoded = response.read().decode('utf-8')

            reader = csv.reader(io.StringIO(decoded))
            loaded_count = 0
            with self._lock:
                for row in reader:
                    if len(row) > 13:
                        symbol = str(row[9]).strip()
                        company = str(row[1]).strip()
                        if symbol.endswith("-EQ"):
                            self._snapshots[symbol] = StockSnapshot(
                                symbol=symbol,
                                company_name=company or "Name unavailable",
                                exchange="NSE",
                                ltp=0.0
                            )
                            loaded_count += 1
            self.universe_count = loaded_count
            self.universe_loaded = True
            logger.info(f"Loaded {loaded_count} NSE Equity symbols.")
            return True
        except Exception as e:
            self._last_error = f"Universe Load Error: {str(e)}"
            logger.error(self._last_error)
            return False

    def _background_worker(self) -> None:
        """Manages the Fyers WebSocket connection and universe loading."""
        while True:
            try:
                auth_status = self.broker_adapter.get_connection_status()
                if not auth_status.is_authenticated:
                    self.broker_adapter.auth_service.set_state(ConnectionState.AUTH_REQUIRED)
                    self._last_error = "Access token missing or expired."
                    time.sleep(2.0)
                    continue

                if not self.universe_loaded:
                    self._load_universe()
                    if not self.universe_loaded:
                        time.sleep(2.0)
                        continue

                # Prepare token string
                client_id = self.broker_adapter.auth_service.config.client_id
                access_token = self.broker_adapter.auth_service.access_token
                ws_token = f"{client_id}:{access_token}"

                # Stable copy of symbols
                with self._lock:
                    all_symbols = list(self._snapshots.keys())

                def on_message(message):
                    with self._lock:
                        if isinstance(message, list):
                            for item in message:
                                sym = item.get("symbol")
                                ltp = item.get("ltp")
                                if sym and ltp is not None and sym in self._snapshots:
                                    self._snapshots[sym].ltp = float(ltp)
                                    self._snapshots[sym].timestamp = datetime.now()
                                    self.live_symbols_received.add(sym)
                        elif isinstance(message, dict):
                            sym = message.get("symbol")
                            ltp = message.get("ltp")
                            if sym and ltp is not None and sym in self._snapshots:
                                self._snapshots[sym].ltp = float(ltp)
                                self._snapshots[sym].timestamp = datetime.now()
                                self.live_symbols_received.add(sym)

                def on_error(message):
                    logger.error(f"Fyers WS Error: {message}")
                    self.broker_adapter.auth_service.set_state(ConnectionState.AUTH_ERROR)

                def on_close(message):
                    logger.warning("Fyers WS Closed.")
                    self.broker_adapter.auth_service.set_state(ConnectionState.FEED_DISCONNECTED)

                def on_open():
                    logger.info("Fyers WS Opened. Subscribing to universe...")
                    self.broker_adapter.auth_service.set_state(ConnectionState.FEED_CONNECTED)
                    # Subscribe in chunks to avoid WebSocket size limits
                    chunk_size = 500
                    for i in range(0, len(all_symbols), chunk_size):
                        chunk = all_symbols[i:i+chunk_size]
                        try:
                            if self.fyers_ws:
                                self.fyers_ws.subscribe(symbols=chunk, data_type="SymbolUpdate")
                            time.sleep(0.5)
                        except Exception as e:
                            logger.error(f"Subscription error: {e}")

                try:
                    from fyers_apiv3.FyersWebsocket import data_ws
                    HAS_WS = True
                except ImportError:
                    HAS_WS = False

                if not HAS_WS:
                    self._last_error = "fyers_apiv3 library not installed. WebSocket unavailable."
                    time.sleep(5.0)
                    continue

                self.broker_adapter.auth_service.set_state(ConnectionState.FEED_CONNECTING)
                
                # Fyers WebSocket initialization
                self.fyers_ws = data_ws.FyersDataSocket(
                    access_token=ws_token,
                    save_logs=False,
                    logs_path="",
                    litemode=True,
                    write_to_file=False
                )
                self.fyers_ws.websocket_data = on_message
                self.fyers_ws.on_error = on_error
                self.fyers_ws.on_close = on_close
                self.fyers_ws.on_open = on_open
                
                # This call blocks the thread while the socket is alive
                self.fyers_ws.connect()

                # Reconnect backoff if disconnected
                time.sleep(5.0)

            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Background worker error: {e}")
                time.sleep(5.0)

    def get_all_stocks_dataframe(self) -> pd.DataFrame:
        with self._lock:
            data_rows = [
                {
                    "symbol": s.symbol,
                    "company_name": s.company_name,
                    "exchange": s.exchange,
                    "ltp": s.ltp,
                    "timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                for s in self._snapshots.values() if s.ltp > 0
            ]

        if not data_rows:
            return pd.DataFrame(columns=["symbol", "company_name", "exchange", "ltp", "timestamp"])
        return pd.DataFrame(data_rows)

    def get_filtered_stocks_dataframe(self) -> pd.DataFrame:
        df = self.get_all_stocks_dataframe()
        if df.empty:
            return df

        df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
        df = df.dropna(subset=["ltp"])

        filtered = df[(df["ltp"] >= self.min_price) & (df["ltp"] <= self.max_price)]
        return filtered.sort_values(by="ltp").reset_index(drop=True)

    def get_status_summary(self) -> dict[str, Any]:
        df = self.get_all_stocks_dataframe()
        valid_ltp_count = len(df)
        filtered_df = self.get_filtered_stocks_dataframe()
        
        # Determine specific market closed state if authenticated but zero valid LTPs over time
        auth_status = self.broker_adapter.get_connection_status()

        return {
            "connection_state": auth_status.state.value if auth_status else "UNKNOWN",
            "universe_count": self.universe_count,
            "live_received": len(self.live_symbols_received),
            "valid_ltp": valid_ltp_count,
            "filtered_count": len(filtered_df),
            "min_price": self.min_price,
            "max_price": self.max_price,
            "last_error": self._last_error or auth_status.error_message if auth_status else None,
        }
