"""
Live Market Depth (Level 2) Service for QuantSignal AI.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from quant_signal.brokers.fyers_adapter import FyersBrokerAdapter
from quant_signal.core.types import MarketDepthSnapshot, DepthLevel
from quant_signal.logger import get_logger
from quant_signal.services.fyers_auth_service import ConnectionState

logger = get_logger(__name__)


class MarketDepthService:
    def __init__(self, broker_adapter: FyersBrokerAdapter) -> None:
        self.broker_adapter = broker_adapter
        self._lock = threading.Lock()
        
        self.active_symbol: str | None = None
        self.snapshot: MarketDepthSnapshot | None = None
        
        self.fyers_ws = None
        self._is_running = False

        self._worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._worker_thread.start()

    def subscribe_symbol(self, symbol: str) -> None:
        """Updates the active symbol to monitor for Market Depth."""
        if not symbol or symbol == self.active_symbol:
            return
            
        logger.info(f"MarketDepthService subscribing to {symbol}")
        with self._lock:
            if self.active_symbol and self.fyers_ws:
                try:
                    self.fyers_ws.unsubscribe(symbols=[self.active_symbol], data_type="DepthUpdate")
                except Exception as e:
                    logger.warning(f"Unsubscribe error: {e}")
            
            self.active_symbol = symbol
            self.snapshot = MarketDepthSnapshot(symbol=symbol)
            
            if self.fyers_ws:
                try:
                    self.fyers_ws.subscribe(symbols=[symbol], data_type="DepthUpdate")
                except Exception as e:
                    logger.error(f"Subscribe error: {e}")

    def get_snapshot(self) -> MarketDepthSnapshot | None:
        with self._lock:
            # Return a fast, shallow copy to prevent cross-thread mutation issues in Streamlit
            if not self.snapshot:
                return None
            return MarketDepthSnapshot(
                symbol=self.snapshot.symbol,
                ltp=self.snapshot.ltp,
                bids=list(self.snapshot.bids),
                asks=list(self.snapshot.asks),
                timestamp=self.snapshot.timestamp
            )

    def _background_worker(self) -> None:
        while True:
            try:
                auth_status = self.broker_adapter.get_connection_status()
                if not auth_status.is_authenticated:
                    time.sleep(2.0)
                    continue

                client_id = self.broker_adapter.auth_service.config.client_id
                access_token = self.broker_adapter.auth_service.access_token
                ws_token = f"{client_id}:{access_token}"

                def on_message(message):
                    if not message:
                        return
                    
                    messages = message if isinstance(message, list) else [message]
                    
                    with self._lock:
                        for item in messages:
                            sym = item.get("symbol")
                            if not sym or sym != self.active_symbol or not self.snapshot:
                                continue
                            
                            # Parse LTP if provided
                            if "ltp" in item:
                                self.snapshot.ltp = float(item["ltp"])
                                
                            # Parse Bids
                            if "bids" in item:
                                self.snapshot.bids = []
                                for b in item["bids"]:
                                    self.snapshot.bids.append(DepthLevel(
                                        price=float(b.get("price", 0.0)),
                                        volume=int(b.get("volume", 0)),
                                        orders=int(b.get("ord", 0))
                                    ))
                                # Pad to 5 if less
                                while len(self.snapshot.bids) < 5:
                                    self.snapshot.bids.append(DepthLevel())
                                    
                            # Parse Asks
                            if "asks" in item:
                                self.snapshot.asks = []
                                for a in item["asks"]:
                                    self.snapshot.asks.append(DepthLevel(
                                        price=float(a.get("price", 0.0)),
                                        volume=int(a.get("volume", 0)),
                                        orders=int(a.get("ord", 0))
                                    ))
                                while len(self.snapshot.asks) < 5:
                                    self.snapshot.asks.append(DepthLevel())
                                    
                            self.snapshot.timestamp = datetime.now()

                def on_error(message):
                    logger.error(f"MarketDepth WS Error: {message}")

                def on_close(message):
                    logger.warning("MarketDepth WS Closed.")

                def on_open():
                    logger.info("MarketDepth WS Opened.")
                    if self.active_symbol:
                        try:
                            self.fyers_ws.subscribe(symbols=[self.active_symbol], data_type="DepthUpdate")
                        except Exception as e:
                            logger.error(f"Initial subscribe error: {e}")

                try:
                    from fyers_apiv3.FyersWebsocket import data_ws
                except ImportError:
                    time.sleep(5.0)
                    continue

                self.fyers_ws = data_ws.FyersDataSocket(
                    access_token=ws_token,
                    save_logs=False,
                    logs_path="",
                    litemode=False,
                    write_to_file=False
                )
                self.fyers_ws.websocket_data = on_message
                self.fyers_ws.on_error = on_error
                self.fyers_ws.on_close = on_close
                self.fyers_ws.on_open = on_open
                
                self.fyers_ws.connect()
                
                # If connect() returns or exits, sleep before reconnecting
                time.sleep(5.0)

            except Exception as e:
                logger.error(f"Market Depth worker error: {e}")
                time.sleep(5.0)
