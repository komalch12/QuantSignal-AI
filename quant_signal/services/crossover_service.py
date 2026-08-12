"""
Crossover Detection Service for QuantSignal AI.

Phase 6 Implementation: SMMA (20) & SMMA (120) Crossover Detection.
Phase — Step 3A: Historical Crossover Scanner (robust, full-universe, per-bar).

Detects exact bullish (BUY) and bearish (SELL) crossovers across historical data:
  - BUY SIGNAL:
      Previous SMMA20 <= Previous SMMA120 AND Current SMMA20 > Current SMMA120
  - SELL SIGNAL:
      Previous SMMA20 >= Previous SMMA120 AND Current SMMA20 < Current SMMA120

Operates in a broker-independent manner using TechnicalIndicatorService.

Key functions:
  detect_crossovers_in_dataframe()   — core row-by-row BUY/SELL detector.
  scan_historical_crossovers()       — robust wrapper with full edge-case guards.
  CrossoverService                   — service class wrapping the above.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

from quant_signal.logger import get_logger
from quant_signal.services.technical_indicators import TechnicalIndicatorService

logger = get_logger(__name__)


def detect_crossovers_in_dataframe(
    symbol: str,
    company_name: str,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Detects all SMMA 20 / SMMA 120 crossover events in a historical DataFrame.

    Args:
        symbol: Ticker symbol.
        company_name: Human readable company name.
        df: DataFrame containing columns ['timestamp', 'close', 'smma_20', 'smma_120'].

    Returns:
        List of dicts representing detected BUY and SELL crossover events.
    """
    if df.empty or "smma_20" not in df.columns or "smma_120" not in df.columns:
        return []

    # Filter out initial rows where SMMA_20 or SMMA_120 is NaN
    valid_df = df.dropna(subset=["smma_20", "smma_120"]).copy()
    if len(valid_df) < 2:
        return []

    valid_df["prev_smma_20"] = valid_df["smma_20"].shift(1)
    valid_df["prev_smma_120"] = valid_df["smma_120"].shift(1)

    # Drop the first valid row since it has no previous values
    analysis_df = valid_df.dropna(subset=["prev_smma_20", "prev_smma_120"])

    crossovers = []
    for _, row in analysis_df.iterrows():
        curr_20 = float(row["smma_20"])
        curr_120 = float(row["smma_120"])
        prev_20 = float(row["prev_smma_20"])
        prev_120 = float(row["prev_smma_120"])
        close_price = float(row["close"])
        ts = row["timestamp"]

        signal = "NO SIGNAL"

        # BUY: Previous SMMA20 <= Previous SMMA120 AND Current SMMA20 > Current SMMA120
        if prev_20 <= prev_120 and curr_20 > curr_120:
            signal = "BUY"
        # SELL: Previous SMMA20 >= Previous SMMA120 AND Current SMMA20 < Current SMMA120
        elif prev_20 >= prev_120 and curr_20 < curr_120:
            signal = "SELL"

        if signal in ("BUY", "SELL"):
            crossovers.append({
                "symbol": symbol,
                "company_name": company_name,
                "timestamp": ts,
                "signal": signal,
                "ltp": close_price,
                "smma_20": curr_20,
                "smma_120": curr_120,
                "prev_smma_20": prev_20,
                "prev_smma_120": prev_120,
            })

    return crossovers


def scan_historical_crossovers(
    symbol: str,
    company_name: str,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Step 3A — Robust Historical SMMA Crossover Scanner.

    Scans every bar in a historical DataFrame for SMMA 20 / SMMA 120 crossover events.
    Reuses the existing SMMA columns already present in the DataFrame — no second calculation.

    Edge cases explicitly handled:
      - Empty DataFrame or missing SMMA columns → returns [].
      - Fewer than 2 valid (non-NaN SMMA) rows → returns [] with a warning logged.
      - NaN close price → recorded as np.nan, does NOT block the row.
      - Duplicate timestamps → deduplicated (last occurrence kept per timestamp).
      - Both BUY and SELL condition simultaneously → BUY takes precedence (should not
        happen with correct SMMA data but guarded for safety).

    Each crossover event record contains:
      symbol, company_name, timestamp, close_price, crossover_type ('BUY' | 'SELL'),
      signal ('BUY' | 'SELL'),
      prev_smma_20, prev_smma_120, curr_smma_20, curr_smma_120.

    Args:
        symbol:       Ticker symbol string.
        company_name: Human readable company name.
        df:           Historical DataFrame with columns
                      ['timestamp', 'close', 'smma_20', 'smma_120'].

    Returns:
        List of crossover event dicts, sorted by timestamp ascending.
    """
    # ── Guard: empty or missing columns ──────────────────────────────────────
    if df is None or df.empty:
        logger.debug(f"{symbol}: empty DataFrame supplied to scan_historical_crossovers.")
        return []

    required_cols = {"smma_20", "smma_120"}
    if not required_cols.issubset(set(df.columns)):
        logger.warning(
            f"{symbol}: DataFrame missing SMMA columns. "
            f"Found: {list(df.columns)}. Returning no crossovers."
        )
        return []

    # ── Step 1: De-duplicate timestamps (keep last occurrence per timestamp) ──
    work_df = df.copy()
    if "timestamp" in work_df.columns:
        work_df = work_df.drop_duplicates(subset=["timestamp"], keep="last")

    # ── Step 2: Filter rows where both SMMA values are valid (non-NaN) ────────
    valid_df = work_df.dropna(subset=["smma_20", "smma_120"]).copy()

    if len(valid_df) < 2:
        logger.warning(
            f"{symbol}: Insufficient valid SMMA rows ({len(valid_df)}) "
            f"for crossover scan — need at least 2."
        )
        return []

    # ── Step 3: Build shifted previous-bar columns ────────────────────────────
    valid_df = valid_df.reset_index(drop=True)
    valid_df["prev_smma_20"] = valid_df["smma_20"].shift(1)
    valid_df["prev_smma_120"] = valid_df["smma_120"].shift(1)

    # Drop first row — no previous bar available
    analysis_df = valid_df.dropna(subset=["prev_smma_20", "prev_smma_120"]).copy()

    # ── Step 4: Evaluate each bar ─────────────────────────────────────────────
    crossovers: list[dict[str, Any]] = []

    for _, row in analysis_df.iterrows():
        curr_20 = float(row["smma_20"])
        curr_120 = float(row["smma_120"])
        prev_20 = float(row["prev_smma_20"])
        prev_120 = float(row["prev_smma_120"])

        # Resolve close/ltp price — allow NaN (recorded as-is)
        close_price: float | float = (
            float(row["close"]) if "close" in row.index and pd.notna(row["close"]) else np.nan
        )
        ts = row.get("timestamp", None) if "timestamp" in row.index else None

        crossover_type: str | None = None

        # BUY: Previous SMMA20 <= Previous SMMA120 AND Current SMMA20 > Current SMMA120
        if prev_20 <= prev_120 and curr_20 > curr_120:
            crossover_type = "BUY"
        # SELL: Previous SMMA20 >= Previous SMMA120 AND Current SMMA20 < Current SMMA120
        elif prev_20 >= prev_120 and curr_20 < curr_120:
            crossover_type = "SELL"

        if crossover_type is not None:
            crossovers.append({
                "symbol":         symbol,
                "company_name":   company_name,
                "timestamp":      ts,
                "close_price":    close_price,
                "crossover_type": crossover_type,   # 'BUY' or 'SELL'
                "signal":         crossover_type,   # convenience alias
                "prev_smma_20":   prev_20,
                "prev_smma_120":  prev_120,
                "curr_smma_20":   curr_20,
                "curr_smma_120":  curr_120,
            })

    # ── Step 5: Sort by timestamp ascending ──────────────────────────────────
    crossovers.sort(key=lambda x: (x["timestamp"] is None, x["timestamp"]))

    logger.info(
        f"{symbol}: scan_historical_crossovers found {len(crossovers)} event(s) "
        f"across {len(analysis_df)} analysed bars."
    )
    return crossovers


class CrossoverService:
    """
    Broker-independent SMMA Crossover Detection Service.

    Scans single symbols or entire market universe for SMMA 20/120 crossovers.
    """

    def __init__(self, tech_service: TechnicalIndicatorService | None = None) -> None:
        self.tech_service = tech_service or TechnicalIndicatorService()

    def get_symbol_crossovers(
        self,
        symbol: str,
        days: int = 250,
        use_robust_scanner: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Detects all historical crossover events for a single stock symbol.

        Args:
            symbol: Ticker symbol.
            days:   Calendar days of history to load (default 250).
            use_robust_scanner: When True (default), uses the Step 3A
                scan_historical_crossovers() with full edge-case guards.
                When False, falls back to the original detect_crossovers_in_dataframe().
        """
        stock_res = self.tech_service.calculate_stock_indicators(symbol=symbol, days=days)
        history_df = stock_res.get("history_df", pd.DataFrame())

        company_name = symbol
        if hasattr(self.tech_service.provider, "_data"):
            company_name = self.tech_service.provider._data.get(symbol, {}).get("company", symbol)

        if use_robust_scanner:
            return scan_historical_crossovers(
                symbol=symbol,
                company_name=company_name,
                df=history_df,
            )
        return detect_crossovers_in_dataframe(
            symbol=symbol,
            company_name=company_name,
            df=history_df,
        )

    def get_all_crossovers_dataframe(
        self,
        symbols: list[str] | None = None,
        days: int = 250,
    ) -> pd.DataFrame:
        """
        Scans all (or specified) symbols and returns a DataFrame of all historical
        SMMA 20/120 crossover events detected by scan_historical_crossovers().

        Step 3A: uses the robust scanner with full NaN / duplicate / edge-case handling.

        Returned columns (superset of Phase 6 schema for backwards compatibility):
          symbol, company_name, timestamp, signal, crossover_type,
          close_price, ltp (alias of close_price), curr_smma_20, curr_smma_120,
          prev_smma_20, prev_smma_120
        """
        target_symbols = symbols or self.tech_service.provider.get_symbols()
        all_crossovers: list[dict[str, Any]] = []

        company_map: dict[str, str] = {}
        if hasattr(self.tech_service.provider, "_data"):
            company_map = {k: v.get("company", k) for k, v in self.tech_service.provider._data.items()}

        for sym in target_symbols:
            stock_res = self.tech_service.calculate_stock_indicators(symbol=sym, days=days)
            history_df = stock_res.get("history_df", pd.DataFrame())
            comp_name = company_map.get(sym, sym)

            events = scan_historical_crossovers(
                symbol=sym,
                company_name=comp_name,
                df=history_df,
            )
            all_crossovers.extend(events)

        if not all_crossovers:
            return pd.DataFrame(columns=[
                "symbol", "company_name", "timestamp", "signal", "crossover_type",
                "close_price", "ltp", "curr_smma_20", "curr_smma_120",
                "prev_smma_20", "prev_smma_120",
            ])

        df = pd.DataFrame(all_crossovers)
        # Add ltp alias for backwards compatibility with existing UI views
        df["ltp"] = df["close_price"]
        # Rename curr_smma_20/120 to smma_20/120 aliases expected by existing UI
        df["smma_20"] = df["curr_smma_20"]
        df["smma_120"] = df["curr_smma_120"]
        # Sort by timestamp descending so latest crossovers appear first
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    def get_summary_metrics(self, df_crossovers: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Calculates summary metrics across all detected crossovers.
        """
        if df_crossovers is None:
            df_crossovers = self.get_all_crossovers_dataframe()

        if df_crossovers.empty:
            return {
                "total_crossovers": 0,
                "buy_count": 0,
                "sell_count": 0,
                "latest_crossover": None,
            }

        buy_count = len(df_crossovers[df_crossovers["signal"] == "BUY"])
        sell_count = len(df_crossovers[df_crossovers["signal"] == "SELL"])
        latest_row = df_crossovers.iloc[0].to_dict()

        return {
            "total_crossovers": len(df_crossovers),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "latest_crossover": latest_row,
        }

    def map_crossover_to_signal(self, crossover: str) -> str:
        """
        Maps crossover status to signal:
          - 'BUY_CROSSOVER'  -> 'BUY'
          - 'SELL_CROSSOVER' -> 'SELL'
          - 'NONE'           -> 'WATCH'
        """
        if crossover == "BUY_CROSSOVER":
            return "BUY"
        elif crossover == "SELL_CROSSOVER":
            return "SELL"
        return "WATCH"

    def get_crossover_signals_dataframe(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """
        Returns full universe DataFrame containing latest SMMA 20/120 indicators,
        crossover status ('BUY_CROSSOVER', 'SELL_CROSSOVER', 'NONE'), and mapped Signal ('BUY', 'SELL', 'WATCH').

        Sorted with actual crossover signals first (BUY/SELL), followed by NONE.
        """
        target_symbols = symbols or self.tech_service.provider.get_symbols()
        rows = []
        company_map = {}
        if hasattr(self.tech_service.provider, "_data"):
            company_map = {k: v.get("company", k) for k, v in self.tech_service.provider._data.items()}

        for sym in target_symbols:
            stock_res = self.tech_service.calculate_stock_indicators(symbol=sym, days=250)
            comp_name = company_map.get(sym, sym)
            crossover = stock_res.get("crossover", "NONE")
            signal = self.map_crossover_to_signal(crossover)

            history_df = stock_res.get("history_df", pd.DataFrame())
            prev_20 = np.nan
            prev_120 = np.nan

            if not history_df.empty and "smma_20" in history_df.columns:
                valid_rows = history_df.dropna(subset=["smma_20", "smma_120"])
                if len(valid_rows) >= 2:
                    prev_row = valid_rows.iloc[-2]
                    prev_20 = float(prev_row["smma_20"])
                    prev_120 = float(prev_row["smma_120"])

            # Friendly crossover label
            crossover_label = "NO CROSSOVER"
            if crossover == "BUY_CROSSOVER":
                crossover_label = "BUY CROSSOVER"
            elif crossover == "SELL_CROSSOVER":
                crossover_label = "SELL CROSSOVER"

            rows.append({
                "symbol": sym,
                "company_name": comp_name,
                "ltp": stock_res["ltp"],
                "smma_20": stock_res["smma_20"],
                "smma_120": stock_res["smma_120"],
                "prev_smma_20": prev_20,
                "prev_smma_120": prev_120,
                "trend": stock_res["trend"],
                "distance_pct": stock_res["distance_pct"],
                "crossover": crossover_label,
                "crossover_raw": crossover,
                "signal": signal,
                "timestamp": self.tech_service.provider._data.get(sym, {}).get("timestamp", "10:30:00 (Simulated Demo Time)"),
            })

        if not rows:
            return pd.DataFrame(columns=[
                "symbol", "company_name", "ltp", "smma_20", "smma_120", "prev_smma_20", "prev_smma_120",
                "trend", "distance_pct", "crossover", "crossover_raw", "signal", "timestamp"
            ])

        df = pd.DataFrame(rows)
        # Priority sort: BUY_CROSSOVER (1), SELL_CROSSOVER (2), NONE (3), then symbol
        priority_map = {"BUY_CROSSOVER": 1, "SELL_CROSSOVER": 2, "NONE": 3}
        df["sort_priority"] = df["crossover_raw"].map(priority_map).fillna(3)
        df = df.sort_values(by=["sort_priority", "symbol"]).reset_index(drop=True)
        df.drop(columns=["sort_priority"], inplace=True)
        return df

    def get_crossover_summary_cards(self, df: pd.DataFrame | None = None) -> dict[str, int]:
        """
        Returns summary card counts:
          - Total Stocks
          - BUY Crossovers
          - SELL Crossovers
          - No Crossover
        """
        if df is None:
            df = self.get_crossover_signals_dataframe()

        if df.empty:
            return {
                "total_stocks": 0,
                "buy_crossovers": 0,
                "sell_crossovers": 0,
                "no_crossover": 0,
            }

        buy_count = len(df[df["crossover_raw"] == "BUY_CROSSOVER"])
        sell_count = len(df[df["crossover_raw"] == "SELL_CROSSOVER"])
        none_count = len(df[df["crossover_raw"] == "NONE"])

        return {
            "total_stocks": len(df),
            "buy_crossovers": buy_count,
            "sell_crossovers": sell_count,
            "no_crossover": none_count,
        }

