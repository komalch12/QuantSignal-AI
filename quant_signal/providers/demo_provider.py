"""
Demo Market Data Provider for QuantSignal AI.

Provides deterministic, realistic NSE stock data for development and
assignment demonstration WITHOUT requiring a broker account.

IMPORTANT:
- This data is NOT live market data.
- All values are fixed/deterministic (no random generation on each refresh).
- Prices are representative of approximate historical ranges for each stock.
- This provider must NEVER be labelled "LIVE" in any UI component.

Supports all future assignment modules:
- NSE stock scanner
- LTP filter (30 <= LTP <= 500)
- Liquidity filter (Bid/Ask Qty >= 10,00,000)
- Market depth (5-level bid/ask)
- SMMA 20 / SMMA 120 (via get_historical_data)
- Crossover detection
- AI/ML analysis
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from quant_signal.core.types import (
    DepthLevel,
    MarketDepthSnapshot,
    StockSnapshot,
)
from quant_signal.providers.base import MarketDataProvider

# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC NSE DEMO UNIVERSE
# Fields: symbol, company_name, ltp, bid_price, bid_qty, ask_price, ask_qty, volume
# LTP filter: 30 <= LTP <= 500
# Liquidity filter: bid_qty >= 10,00,000 AND ask_qty >= 10,00,000
# ─────────────────────────────────────────────────────────────────────────────
_DEMO_STOCKS: list[dict[str, Any]] = [
    # ── Below ₹30 — filtered OUT by LTP filter ────────────────────────────────
    {"symbol": "YESBANK-EQ",     "company": "Yes Bank Ltd",             "ltp": 21.30, "bid": 21.25, "ask": 21.35, "bid_qty": 5_200_000, "ask_qty": 4_800_000, "vol": 45_000_000},
    {"symbol": "VODAFONEIDEA-EQ","company": "Vodafone Idea Ltd",        "ltp":  8.45, "bid":  8.40, "ask":  8.50, "bid_qty":12_500_000, "ask_qty":11_800_000, "vol":125_000_000},
    {"symbol": "SOUTHBANK-EQ",   "company": "South Indian Bank Ltd",    "ltp": 26.80, "bid": 26.75, "ask": 26.85, "bid_qty": 2_800_000, "ask_qty": 2_500_000, "vol": 18_000_000},
    {"symbol": "UCOBANK-EQ",     "company": "UCO Bank",                 "ltp": 29.40, "bid": 29.35, "ask": 29.45, "bid_qty": 3_200_000, "ask_qty": 2_900_000, "vol": 22_000_000},

    # ── ₹30–₹500 — PASS LTP filter ───────────────────────────────────────────
    {"symbol": "SUZLON-EQ",      "company": "Suzlon Energy Ltd",        "ltp": 58.75, "bid": 58.70, "ask": 58.80, "bid_qty": 3_400_000, "ask_qty": 3_100_000, "vol": 28_000_000},
    {"symbol": "NHPC-EQ",        "company": "NHPC Ltd",                 "ltp": 86.25, "bid": 86.20, "ask": 86.30, "bid_qty": 2_200_000, "ask_qty": 2_050_000, "vol": 15_000_000},
    {"symbol": "IRFC-EQ",        "company": "Indian Railway Finance Corp","ltp":195.40,"bid":195.35,"ask":195.45, "bid_qty": 1_900_000, "ask_qty": 1_750_000, "vol": 12_000_000},
    {"symbol": "RVNL-EQ",        "company": "Rail Vikas Nigam Ltd",     "ltp":342.80, "bid":342.75, "ask":342.85, "bid_qty": 2_500_000, "ask_qty": 2_350_000, "vol": 18_500_000},
    {"symbol": "NTPC-EQ",        "company": "NTPC Ltd",                 "ltp":368.50, "bid":368.45, "ask":368.55, "bid_qty": 3_100_000, "ask_qty": 2_950_000, "vol": 22_000_000},
    {"symbol": "POWERGRID-EQ",   "company": "Power Grid Corp of India", "ltp":318.90, "bid":318.85, "ask":318.95, "bid_qty": 2_800_000, "ask_qty": 2_650_000, "vol": 19_000_000},
    {"symbol": "COALINDIA-EQ",   "company": "Coal India Ltd",           "ltp":445.20, "bid":445.15, "ask":445.25, "bid_qty": 1_600_000, "ask_qty": 1_480_000, "vol":  9_800_000},
    {"symbol": "BANKBARODA-EQ",  "company": "Bank of Baroda",           "ltp":212.35, "bid":212.30, "ask":212.40, "bid_qty": 2_100_000, "ask_qty": 1_950_000, "vol": 14_500_000},
    {"symbol": "CANBK-EQ",       "company": "Canara Bank",              "ltp": 98.45, "bid": 98.40, "ask": 98.50, "bid_qty": 3_800_000, "ask_qty": 3_500_000, "vol": 30_000_000},
    {"symbol": "PNB-EQ",         "company": "Punjab National Bank",     "ltp":105.80, "bid":105.75, "ask":105.85, "bid_qty": 4_200_000, "ask_qty": 3_900_000, "vol": 35_000_000},
    {"symbol": "TATASTEEL-EQ",   "company": "Tata Steel Ltd",           "ltp":148.60, "bid":148.55, "ask":148.65, "bid_qty": 3_600_000, "ask_qty": 3_400_000, "vol": 26_000_000},
    {"symbol": "TATAPOWER-EQ",   "company": "Tata Power Company Ltd",   "ltp":415.30, "bid":415.25, "ask":415.35, "bid_qty": 1_200_000, "ask_qty": 1_100_000, "vol":  8_500_000},
    {"symbol": "ADANIGREEN-EQ",  "company": "Adani Green Energy Ltd",   "ltp":165.40, "bid":165.35, "ask":165.45, "bid_qty":   850_000, "ask_qty":   780_000, "vol":  5_200_000},
    {"symbol": "SAIL-EQ",        "company": "Steel Authority of India", "ltp":128.70, "bid":128.65, "ask":128.75, "bid_qty": 3_200_000, "ask_qty": 2_950_000, "vol": 24_000_000},
    {"symbol": "NMDC-EQ",        "company": "NMDC Ltd",                 "ltp":245.85, "bid":245.80, "ask":245.90, "bid_qty": 1_500_000, "ask_qty": 1_380_000, "vol": 10_200_000},
    {"symbol": "RECLTD-EQ",      "company": "REC Ltd",                  "ltp":487.90, "bid":487.85, "ask":487.95, "bid_qty": 1_100_000, "ask_qty": 1_020_000, "vol":  7_800_000},
    {"symbol": "PFC-EQ",         "company": "Power Finance Corp Ltd",   "ltp":462.50, "bid":462.45, "ask":462.55, "bid_qty": 1_300_000, "ask_qty": 1_180_000, "vol":  9_200_000},
    {"symbol": "UNIONBANK-EQ",   "company": "Union Bank of India",      "ltp":142.65, "bid":142.60, "ask":142.70, "bid_qty": 2_700_000, "ask_qty": 2_550_000, "vol": 19_800_000},
    {"symbol": "IOC-EQ",         "company": "Indian Oil Corporation",   "ltp":152.40, "bid":152.35, "ask":152.45, "bid_qty": 2_900_000, "ask_qty": 2_750_000, "vol": 21_000_000},
    {"symbol": "BPCL-EQ",        "company": "Bharat Petroleum Corp",    "ltp":308.75, "bid":308.70, "ask":308.80, "bid_qty": 1_800_000, "ask_qty": 1_660_000, "vol": 12_500_000},
    {"symbol": "GAIL-EQ",        "company": "GAIL India Ltd",           "ltp":205.60, "bid":205.55, "ask":205.65, "bid_qty": 2_400_000, "ask_qty": 2_260_000, "vol": 17_200_000},
    {"symbol": "IDFCFIRSTB-EQ",  "company": "IDFC First Bank Ltd",      "ltp": 72.35, "bid": 72.30, "ask": 72.40, "bid_qty": 4_500_000, "ask_qty": 4_200_000, "vol": 38_000_000},
    {"symbol": "INDIANB-EQ",     "company": "Indian Bank",              "ltp":498.20, "bid":498.15, "ask":498.25, "bid_qty":   920_000, "ask_qty":   860_000, "vol":  6_400_000},
    {"symbol": "FEDERALBNK-EQ",  "company": "Federal Bank Ltd",         "ltp":185.90, "bid":185.85, "ask":185.95, "bid_qty": 2_050_000, "ask_qty": 1_920_000, "vol": 14_800_000},
    {"symbol": "HFCL-EQ",        "company": "HFCL Ltd",                 "ltp": 83.40, "bid": 83.35, "ask": 83.45, "bid_qty": 1_650_000, "ask_qty": 1_520_000, "vol": 11_500_000},
    {"symbol": "BHEL-EQ",        "company": "Bharat Heavy Electricals", "ltp":238.45, "bid":238.40, "ask":238.50, "bid_qty": 2_350_000, "ask_qty": 2_200_000, "vol": 16_800_000},
    {"symbol": "MAHABANK-EQ",    "company": "Bank of Maharashtra",      "ltp": 53.60, "bid": 53.55, "ask": 53.65, "bid_qty": 4_100_000, "ask_qty": 3_800_000, "vol": 33_500_000},
    {"symbol": "IDEA-EQ",        "company": "Vodafone Idea Ltd",        "ltp": 39.20, "bid": 39.15, "ask": 39.25, "bid_qty": 8_200_000, "ask_qty": 7_900_000, "vol": 72_000_000},

    # ── Above ₹500 — filtered OUT by LTP filter ───────────────────────────────
    {"symbol": "SBIN-EQ",        "company": "State Bank of India",      "ltp":825.60, "bid":825.55, "ask":825.65, "bid_qty": 1_850_000, "ask_qty": 1_700_000, "vol": 13_200_000},
    {"symbol": "RELIANCE-EQ",    "company": "Reliance Industries Ltd",  "ltp":2842.50,"bid":2842.45,"ask":2842.55,"bid_qty":   620_000, "ask_qty":   580_000, "vol":  4_500_000},
    {"symbol": "TCS-EQ",         "company": "Tata Consultancy Services","ltp":3568.80,"bid":3568.75,"ask":3568.85,"bid_qty":   180_000, "ask_qty":   165_000, "vol":  1_200_000},
    {"symbol": "INFY-EQ",        "company": "Infosys Ltd",              "ltp":1582.40,"bid":1582.35,"ask":1582.45,"bid_qty":   420_000, "ask_qty":   390_000, "vol":  2_900_000},
    {"symbol": "HDFCBANK-EQ",    "company": "HDFC Bank Ltd",            "ltp":1685.70,"bid":1685.65,"ask":1685.75,"bid_qty":   380_000, "ask_qty":   352_000, "vol":  2_600_000},
    {"symbol": "ICICIBANK-EQ",   "company": "ICICI Bank Ltd",           "ltp":1248.90,"bid":1248.85,"ask":1248.95,"bid_qty":   560_000, "ask_qty":   520_000, "vol":  3_900_000},
    {"symbol": "ADANIENT-EQ",    "company": "Adani Enterprises Ltd",    "ltp":2245.60,"bid":2245.55,"ask":2245.65,"bid_qty":   210_000, "ask_qty":   195_000, "vol":  1_500_000},
]

# Snapshot timestamp — fixed at a realistic market open time for determinism
_DEMO_TIMESTAMP = datetime(2025, 1, 15, 10, 30, 0)


class DemoMarketDataProvider(MarketDataProvider):
    """
    Deterministic demo data provider for development and assignment demonstration.

    WARNING: This provider returns FIXED SAMPLE DATA.
    It is NOT connected to any live market feed.
    All values are representative approximations only.
    """

    def __init__(self) -> None:
        # Build lookup dict once at startup
        self._data: dict[str, dict[str, Any]] = {
            row["symbol"]: row for row in _DEMO_STOCKS
        }

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "Demo"

    @property
    def is_live(self) -> bool:
        return False  # NEVER returns True

    # ── Symbols ───────────────────────────────────────────────────────────────

    def get_symbols(self) -> list[str]:
        return list(self._data.keys())

    # ── Quotes ────────────────────────────────────────────────────────────────

    def get_quotes(self, symbols: list[str] | None = None) -> list[StockSnapshot]:
        target = symbols or self.get_symbols()
        result = []
        for sym in target:
            if sym not in self._data:
                continue
            row = self._data[sym]
            result.append(StockSnapshot(
                symbol=sym,
                company_name=row["company"],
                exchange="NSE",
                ltp=row["ltp"],
                timestamp=_DEMO_TIMESTAMP,
            ))
        return result

    # ── Market Depth ──────────────────────────────────────────────────────────

    def get_market_depth(self, symbol: str) -> MarketDepthSnapshot:
        """
        Returns a deterministic 5-level order book for the given symbol.
        Bid/ask levels are constructed from the LTP with fixed tick steps.
        """
        row = self._data.get(symbol)
        if not row:
            return MarketDepthSnapshot(symbol=symbol)

        ltp = row["ltp"]
        bid_base = row["bid"]
        ask_base = row["ask"]
        total_bid_qty = row["bid_qty"]
        total_ask_qty = row["ask_qty"]

        # Determine tick step (smaller for low-priced stocks)
        tick = 0.05 if ltp < 100 else (0.10 if ltp < 500 else 0.25)

        # Split total qty across 5 levels with decreasing volumes (deterministic)
        bid_split = [0.35, 0.25, 0.20, 0.12, 0.08]
        ask_split = [0.34, 0.26, 0.19, 0.13, 0.08]

        bids = [
            DepthLevel(
                price=round(bid_base - i * tick, 2),
                volume=int(total_bid_qty * bid_split[i]),
                orders=max(1, 15 - i * 3),
            )
            for i in range(5)
        ]
        asks = [
            DepthLevel(
                price=round(ask_base + i * tick, 2),
                volume=int(total_ask_qty * ask_split[i]),
                orders=max(1, 14 - i * 3),
            )
            for i in range(5)
        ]

        return MarketDepthSnapshot(
            symbol=symbol,
            ltp=ltp,
            bids=bids,
            asks=asks,
            timestamp=_DEMO_TIMESTAMP,
        )

    # ── Historical Data (for SMMA / crossover) ────────────────────────────────

    def get_historical_data(self, symbol: str, timeframe: str, days: int = 180) -> pd.DataFrame:
        """
        Returns a deterministic synthetic OHLCV DataFrame.
        Uses a per-symbol seed so each stock has a unique but repeatable history.
        Suitable for SMMA 20, SMMA 120, and crossover detection development.
        """
        import hashlib
        row = self._data.get(symbol, {})
        base_price = row.get("ltp", 100.0)

        # Deterministic seed per symbol
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % (2**31)

        import random
        rng = random.Random(seed)

        records = []
        price = base_price * 0.85  # Start 15% below current
        base_date = _DEMO_TIMESTAMP - timedelta(days=days)

        for d in range(days):
            date = base_date + timedelta(days=d)
            if date.weekday() >= 5:  # Skip weekends
                continue
            change_pct = (rng.random() - 0.49) * 0.025  # ~±2.5% daily range
            open_ = round(price, 2)
            close = round(price * (1 + change_pct), 2)
            high = round(max(open_, close) * (1 + rng.random() * 0.01), 2)
            low = round(min(open_, close) * (1 - rng.random() * 0.01), 2)
            volume = int(rng.randint(500_000, 5_000_000))
            records.append({
                "timestamp": date,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })
            price = close

        return pd.DataFrame(records)

    # ── Extended Data (for future modules) ───────────────────────────────────

    def get_full_quote_row(self, symbol: str) -> dict[str, Any]:
        """Returns all raw fields for a symbol including bid/ask quantities."""
        return dict(self._data.get(symbol, {}))
