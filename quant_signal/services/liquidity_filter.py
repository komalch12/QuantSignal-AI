"""
Liquidity Filter Service for QuantSignal AI.

Filters NSE stocks based on real-time bid/ask depth volume.
Classifies each stock as High / Medium / Low liquidity.

Filter Rule:
    Bid Quantity > 10,00,000 AND Ask Quantity > 10,00,000

Thresholds:
    High   : min(bid_qty, ask_qty) >= 20,00,000
    Medium : min(bid_qty, ask_qty) >= 10,00,000
    Low    : min(bid_qty, ask_qty) <  10,00,000  (filtered out — not displayed)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from quant_signal.core.types import (
    LiquiditySnapshot,
    LiquidityStatus,
    MarketDepthSnapshot,
)
from quant_signal.logger import get_logger

logger = get_logger(__name__)

# ── Configurable thresholds ───────────────────────────────────────────────────
MIN_FILTER_QTY: int = 1_000_000    # 10,00,000 — minimum to appear in results
HIGH_QTY_THRESHOLD: int = 2_000_000  # 20,00,000 — classified as "High"


class LiquidityFilterService:
    """
    Reusable service to classify and filter NSE stocks by market depth liquidity.

    Usage:
        service = LiquidityFilterService()
        snapshot = service.classify_snapshot(market_depth_snapshot, company_name)
        df = service.filter_dataframe(snapshots_list)
    """

    def __init__(
        self,
        min_filter_qty: int = MIN_FILTER_QTY,
        high_threshold: int = HIGH_QTY_THRESHOLD,
    ) -> None:
        self.min_filter_qty = min_filter_qty
        self.high_threshold = high_threshold

    # ── Core Classification ───────────────────────────────────────────────────

    def classify_status(self, total_bid_qty: int, total_ask_qty: int) -> LiquidityStatus:
        """Classifies liquidity status based on total bid and ask volumes."""
        min_qty = min(total_bid_qty, total_ask_qty)
        if min_qty >= self.high_threshold:
            return LiquidityStatus.HIGH
        elif min_qty >= self.min_filter_qty:
            return LiquidityStatus.MEDIUM
        else:
            return LiquidityStatus.LOW

    def passes_filter(self, total_bid_qty: int, total_ask_qty: int) -> bool:
        """Returns True only if both bid AND ask quantities meet the minimum threshold."""
        return total_bid_qty >= self.min_filter_qty and total_ask_qty >= self.min_filter_qty

    # ── From MarketDepthSnapshot ──────────────────────────────────────────────

    def classify_snapshot(
        self,
        snapshot: MarketDepthSnapshot,
        company_name: str = "",
    ) -> LiquiditySnapshot:
        """Converts a MarketDepthSnapshot into a LiquiditySnapshot with classification."""
        total_bid = sum(b.volume for b in snapshot.bids)
        total_ask = sum(a.volume for a in snapshot.asks)
        status = self.classify_status(total_bid, total_ask)
        return LiquiditySnapshot(
            symbol=snapshot.symbol,
            company_name=company_name,
            ltp=snapshot.ltp,
            total_bid_qty=total_bid,
            total_ask_qty=total_ask,
            status=status,
            timestamp=snapshot.timestamp,
        )

    # ── From List of LiquiditySnapshots ──────────────────────────────────────

    def filter_snapshots(self, snapshots: list[LiquiditySnapshot]) -> list[LiquiditySnapshot]:
        """Filters and sorts snapshots that pass the bid/ask minimum threshold."""
        filtered = [
            s for s in snapshots
            if self.passes_filter(s.total_bid_qty, s.total_ask_qty)
        ]
        return sorted(filtered, key=lambda s: min(s.total_bid_qty, s.total_ask_qty), reverse=True)

    # ── DataFrame Interface ───────────────────────────────────────────────────

    def filter_dataframe(self, snapshots: list[LiquiditySnapshot]) -> pd.DataFrame:
        """
        Converts filtered LiquiditySnapshots into a display-ready Pandas DataFrame.

        Columns: Symbol, Company Name, LTP, Bid Quantity, Ask Quantity,
                 Liquidity Status, Timestamp
        """
        filtered = self.filter_snapshots(snapshots)

        if not filtered:
            return pd.DataFrame(columns=[
                "Symbol", "Company Name", "LTP (Rs)",
                "Bid Quantity", "Ask Quantity", "Liquidity Status", "Timestamp"
            ])

        return pd.DataFrame([
            {
                "Symbol": s.symbol,
                "Company Name": s.company_name,
                "LTP (Rs)": round(s.ltp, 2),
                "Bid Quantity": s.total_bid_qty,
                "Ask Quantity": s.total_ask_qty,
                "Liquidity Status": s.status.value,
                "Timestamp": s.timestamp.strftime("%H:%M:%S"),
            }
            for s in filtered
        ])

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_filter_config(self) -> dict[str, Any]:
        """Returns current filter configuration."""
        return {
            "min_filter_qty": self.min_filter_qty,
            "high_threshold": self.high_threshold,
            "min_filter_qty_display": f"{self.min_filter_qty:,}",
            "high_threshold_display": f"{self.high_threshold:,}",
        }
