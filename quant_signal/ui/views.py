"""
Dashboard View Layouts for QuantSignal AI.

Phase 4 & Phase 5 Implementation:
- Liquidity Filter and Market Depth using DEVELOPMENT / DEMO DATA MODE.
- Technical Indicators: SMMA 20 and SMMA 120 calculation and interactive crossover chart.

All views are demo-aware and accept either StockScannerService (live) or
DemoScannerService (demo) via duck typing.
"""

from __future__ import annotations

from typing import Union
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from quant_signal.services.stock_scanner import StockScannerService
from quant_signal.services.liquidity_filter import LiquidityFilterService
from quant_signal.services.market_depth_service import MarketDepthService
from quant_signal.services.technical_indicators import TechnicalIndicatorService
from quant_signal.services.crossover_service import CrossoverService, detect_crossovers_in_dataframe, scan_historical_crossovers
from quant_signal.services.ml_signal_service import MLSignalService
from quant_signal.services.crossover_profitability_service import CrossoverProfitabilityService
from quant_signal.ui.components import render_phase4_summary_metrics
from quant_signal.utils.helpers import safe_rerun

AnyScannerService = Union[StockScannerService, "DemoScannerService"]  # type: ignore[name-defined]


def render_stock_scanner_view(scanner_service: AnyScannerService) -> bool:
    """
    Renders the NSE Stock Scanner View with Phase 4 Summary Metrics.

    LTP Filter: ₹30 – ₹500.
    """
    status = scanner_service.get_status_summary()
    is_demo = status.get("connection_state") == "DEMO"

    tab_title = "🔎 NSE Stock Scanner  [DEVELOPMENT / DEMO DATA]" if is_demo else "🔎 Live NSE Stock Scanner"
    st.subheader(tab_title)

    # ── Summary Metrics Cards ──────────────────────────────────────────────────
    if is_demo and hasattr(scanner_service, "get_summary_metrics"):
        metrics = scanner_service.get_summary_metrics()
        render_phase4_summary_metrics(
            total_stocks=metrics["total_stocks"],
            ltp_matching=metrics["ltp_matching"],
            bid_qty_matching=metrics["bid_qty_matching"],
            ask_qty_matching=metrics["ask_qty_matching"],
            liquidity_matching=metrics["liquidity_matching"],
        )

        with st.expander("🛠️ System Diagnostics"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Configuration")
                st.markdown(f"**FYERS:** {status.get('fyers_status', 'Not Configured')}")
                st.markdown(f"**Data Provider:** {status.get('data_provider', 'Demo')}")
                st.markdown(f"**Live Market Data:** {status.get('live_market_data', 'Unavailable')}")
            with c2:
                st.markdown("### Demo Data Universe Info")
                st.markdown(f"**Provider:** DemoMarketDataProvider")
                st.markdown(f"**Data Mode:** DEVELOPMENT / DEMO DATA")
                st.markdown(f"**Total Universe:** {metrics['total_stocks']} NSE Equity Stocks")
                st.markdown(f"**LTP Filter (₹30 - ₹500):** {metrics['ltp_matching']} Stocks Passed")
                st.markdown(f"**Liquidity Filter (>1M Bid & Ask):** {metrics['liquidity_matching']} Stocks Passed")
    else:
        # Live mode header metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        state = status.get("connection_state", "UNKNOWN")
        with m1:
            feed_status = "🟢 Connected" if state in ("FEED_CONNECTED", "AUTHENTICATED") else "🔴 Disconnected"
            st.metric("Fyers Feed", feed_status)
        with m2:
            st.metric("NSE Universe", f"{status['universe_count']:,}")
        with m3:
            st.metric("Live Symbols", f"{status['live_received']:,}")
        with m4:
            st.metric("Valid LTP", f"{status['valid_ltp']:,}")
        with m5:
            st.metric("Matching Stocks", f"{status['filtered_count']:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        auto_refresh = st.checkbox(
            "⚡ Auto-Refresh (1s)",
            value=not is_demo,
            help="Not required in demo mode — data is deterministic and static.",
        )
    with ctrl2:
        if st.button("🔄 Refresh Scanner"):
            safe_rerun()

    st.markdown("---")

    # ── Live Auth Gate ────────────────────────────────────────────────────────
    if not is_demo:
        state = status.get("connection_state", "UNKNOWN")
        if state in ("AUTH_REQUIRED", "AUTH_NOT_CONFIGURED", "AUTH_ERROR", "FEED_DISCONNECTED"):
            st.error("Fyers access token is missing, invalid, or expired. Please authenticate again.")
            return auto_refresh

    # ── Scanner Table ─────────────────────────────────────────────────────────
    df = scanner_service.get_filtered_stocks_dataframe()

    if df.empty:
        st.warning(f"No NSE stocks currently match ₹{status['min_price']:.0f}–₹{status['max_price']:.0f}.")
        return auto_refresh

    ts_col_name = "Demo Timestamp" if is_demo else "Timestamp"
    has_etq = "etq_5m" in df.columns
    has_avg = "avg_ltp_20m" in df.columns

    required_cols = ["symbol", "company_name", "exchange", "ltp"]
    if has_etq:
        required_cols.extend(["etq_5m", "etq_20m", "etq_60m"])
    if has_avg:
        required_cols.extend(["avg_ltp_20m", "avg_ltp_60m"])
    required_cols.append("timestamp")

    display_df = df[[c for c in required_cols if c in df.columns]].copy()
    rename_dict = {
        "symbol":       "Symbol",
        "company_name": "Company Name",
        "exchange":     "Exchange",
        "ltp":          "LTP",
        "timestamp":    ts_col_name,
        "etq_5m":       "ETQ (5m)",
        "etq_20m":      "ETQ (20m)",
        "etq_60m":      "ETQ (60m)",
        "avg_ltp_20m":  "Avg LTP (20m)",
        "avg_ltp_60m":  "Avg LTP (60m)",
    }
    display_df = display_df.rename(columns=rename_dict)
    display_df["LTP"] = display_df["LTP"].apply(lambda x: f"₹{x:,.2f}")

    if "ETQ (5m)" in display_df.columns:
        display_df["ETQ (5m)"] = display_df["ETQ (5m)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
        display_df["ETQ (20m)"] = display_df["ETQ (20m)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
        display_df["ETQ (60m)"] = display_df["ETQ (60m)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")

    if "Avg LTP (20m)" in display_df.columns:
        display_df["Avg LTP (20m)"] = display_df["Avg LTP (20m)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) and x > 0 else "N/A")
        display_df["Avg LTP (60m)"] = display_df["Avg LTP (60m)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) and x > 0 else "N/A")

    column_config_dict = {
        "Symbol":        st.column_config.TextColumn("Symbol",        width="medium"),
        "Company Name":  st.column_config.TextColumn("Company Name",  width="large"),
        "Exchange":      st.column_config.TextColumn("Exchange",      width="small"),
        "LTP":           st.column_config.TextColumn("LTP",           width="medium"),
        "ETQ (5m)":      st.column_config.TextColumn("ETQ (5m)",      width="small"),
        "ETQ (20m)":     st.column_config.TextColumn("ETQ (20m)",     width="small"),
        "ETQ (60m)":     st.column_config.TextColumn("ETQ (60m)",     width="small"),
        "Avg LTP (20m)": st.column_config.TextColumn("Avg LTP (20m)", width="small"),
        "Avg LTP (60m)": st.column_config.TextColumn("Avg LTP (60m)", width="small"),
        ts_col_name:     st.column_config.TextColumn(ts_col_name,     width="large" if is_demo else "medium"),
    }

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config=column_config_dict,
        hide_index=True,
    )

    if is_demo:
        st.caption("ℹ️ Timestamp is simulated because live broker market data is not connected.")

    return auto_refresh




def render_market_depth_view(
    depth_service,
    symbol: str,
    demo_scanner: "DemoScannerService | None" = None,  # type: ignore[name-defined]
) -> bool:
    """
    Renders the Market Depth Tab (Phase 4).

    Displays:
    - Warning: "Market-depth values are deterministic demo values and are NOT live exchange data."
    - Mode Tag: DEVELOPMENT / DEMO DATA
    - Market Depth Table: Symbol, Company Name, LTP, Bid Price, Bid Quantity, Ask Price, Ask Quantity, Demo Timestamp
    - Interactive 5-Level Order Book (DOM) for selected ticker symbol.
    """
    is_demo = demo_scanner is not None

    if is_demo:
        st.subheader("📊 Market Depth Overview  [DEVELOPMENT / DEMO DATA]")
        st.warning("⚠️ Market-depth values are deterministic demo values and are NOT live exchange data.")
    else:
        st.subheader("📊 Market Depth (DOM)")

    auto_refresh = st.checkbox(
        "⚡ Auto-Refresh DOM (1s)",
        value=not is_demo,
        key="dom_refresh",
    )
    st.markdown("---")

    # ── Render Summary Table of All Market Depth Data (Phase 4 Requirement) ──
    if is_demo and demo_scanner is not None:
        st.markdown("### 📋 Market Depth Summary Table")
        depth_df = demo_scanner.get_market_depth_dataframe()

        display_df = depth_df.copy()
        display_df = display_df.rename(columns={
            "symbol":       "Symbol",
            "company_name": "Company Name",
            "ltp":          "LTP",
            "bid_price":    "Bid Price",
            "bid_quantity": "Bid Quantity",
            "ask_price":    "Ask Price",
            "ask_quantity": "Ask Quantity",
            "timestamp":    "Demo Timestamp",
        })

        display_df["LTP"] = display_df["LTP"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Bid Price"] = display_df["Bid Price"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Ask Price"] = display_df["Ask Price"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Bid Quantity"] = display_df["Bid Quantity"].apply(lambda x: f"{x:,}")
        display_df["Ask Quantity"] = display_df["Ask Quantity"].apply(lambda x: f"{x:,}")

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Symbol":         st.column_config.TextColumn("Symbol",         width="medium"),
                "Company Name":   st.column_config.TextColumn("Company Name",   width="large"),
                "LTP":            st.column_config.TextColumn("LTP",            width="small"),
                "Bid Price":      st.column_config.TextColumn("Bid Price",      width="small"),
                "Bid Quantity":   st.column_config.TextColumn("Bid Quantity",   width="medium"),
                "Ask Price":      st.column_config.TextColumn("Ask Price",      width="small"),
                "Ask Quantity":   st.column_config.TextColumn("Ask Quantity",   width="medium"),
                "Demo Timestamp": st.column_config.TextColumn("Demo Timestamp", width="large"),
            },
            hide_index=True,
        )

        st.caption("ℹ️ Timestamp is simulated because live broker market data is not connected.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📖 5-Level Level 2 Order Book (DOM)")

        demo_symbols = demo_scanner.provider.get_symbols()
        clean_symbol = symbol.replace("NSE:", "")
        if clean_symbol not in demo_symbols:
            filtered = demo_scanner.get_filtered_stocks_dataframe()
            clean_symbol = filtered["symbol"].iloc[0] if not filtered.empty else demo_symbols[0]

        selected_symbol = st.selectbox(
            "Select Ticker for Level 2 Order Book",
            options=demo_symbols,
            index=demo_symbols.index(clean_symbol) if clean_symbol in demo_symbols else 0,
        )

        snapshot = demo_scanner.get_market_depth(selected_symbol)

        st.markdown(f"#### Order Book: **{selected_symbol}** | LTP: **₹{snapshot.ltp:,.2f}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color: #a3be8c; text-align: center;'>BID (Buyers)</h4>", unsafe_allow_html=True)
            bids_data = [
                {"Orders": b.orders, "Quantity": f"{b.volume:,}", "Price": f"₹{b.price:,.2f}"}
                for b in snapshot.bids
            ]
            st.dataframe(pd.DataFrame(bids_data), hide_index=True, use_container_width=True)

        with col2:
            st.markdown("<h4 style='color: #bf616a; text-align: center;'>ASK (Sellers)</h4>", unsafe_allow_html=True)
            asks_data = [
                {"Price": f"₹{a.price:,.2f}", "Quantity": f"{a.volume:,}", "Orders": a.orders}
                for a in snapshot.asks
            ]
            st.dataframe(pd.DataFrame(asks_data), hide_index=True, use_container_width=True)

        st.caption("Last Updated: 10:30:00 (Simulated Demo Time) — Timestamp is simulated because live broker market data is not connected.")

    else:
        # Live mode Level 2 DOM
        depth_service.subscribe_symbol(symbol)
        snapshot = depth_service.get_snapshot()

        if not snapshot:
            st.info(f"Waiting for market depth data for {symbol}...")
            return auto_refresh

        st.markdown(f"### LTP: ₹{snapshot.ltp:,.2f}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color: #a3be8c; text-align: center;'>BID (Buyers)</h4>", unsafe_allow_html=True)
            bids_data = [
                {"Orders": b.orders, "Quantity": f"{b.volume:,}", "Price": f"₹{b.price:,.2f}"}
                for b in snapshot.bids
            ]
            st.dataframe(pd.DataFrame(bids_data), hide_index=True, use_container_width=True)

        with col2:
            st.markdown("<h4 style='color: #bf616a; text-align: center;'>ASK (Sellers)</h4>", unsafe_allow_html=True)
            asks_data = [
                {"Price": f"₹{a.price:,.2f}", "Quantity": f"{a.volume:,}", "Orders": a.orders}
                for a in snapshot.asks
            ]
            st.dataframe(pd.DataFrame(asks_data), hide_index=True, use_container_width=True)

    return auto_refresh


def render_liquidity_filter_view(
    liquidity_service: LiquidityFilterService,
    depth_service: MarketDepthService,
    scanner_service: AnyScannerService,
) -> bool:
    """
    Renders the Liquidity Filter Tab (Phase 4).

    Filter Condition:
    Bid Quantity > 1,000,000 AND Ask Quantity > 1,000,000 (and ₹30 <= LTP <= ₹500).
    """
    status = scanner_service.get_status_summary()
    is_demo = status.get("connection_state") == "DEMO"

    if is_demo:
        st.subheader("💧 Liquidity Filter  [DEVELOPMENT / DEMO DATA]")
        st.warning("⚠️ Market-depth values are deterministic demo values and are NOT live exchange data.")
    else:
        st.subheader("💧 Liquidity Filter")

    auto_refresh = st.checkbox(
        "⚡ Auto-Refresh (1s)",
        value=not is_demo,
        key="liq_refresh",
    )
    st.markdown("---")

    st.caption(
        "Filter Rule: **Bid Quantity > 1,000,000 AND Ask Quantity > 1,000,000** (LTP ₹30 - ₹500)"
    )

    if is_demo and hasattr(scanner_service, "get_liquidity_filtered_dataframe"):
        metrics = scanner_service.get_summary_metrics()
        render_phase4_summary_metrics(
            total_stocks=metrics["total_stocks"],
            ltp_matching=metrics["ltp_matching"],
            bid_qty_matching=metrics["bid_qty_matching"],
            ask_qty_matching=metrics["ask_qty_matching"],
            liquidity_matching=metrics["liquidity_matching"],
        )
        st.markdown("<br>", unsafe_allow_html=True)

        liq_df = scanner_service.get_liquidity_filtered_dataframe()

        if liq_df.empty:
            st.warning("No stocks match the Liquidity Filter criteria.")
            return auto_refresh

        display_df = liq_df.copy()
        rename_dict = {
            "symbol":       "Symbol",
            "company_name": "Company Name",
            "ltp":          "LTP",
            "bid_price":    "Bid Price",
            "bid_quantity": "Bid Quantity",
            "ask_price":    "Ask Price",
            "ask_quantity": "Ask Quantity",
            "etq_5m":       "ETQ (5m)",
            "etq_20m":      "ETQ (20m)",
            "etq_60m":      "ETQ (60m)",
            "avg_ltp_20m":  "Avg LTP (20m)",
            "avg_ltp_60m":  "Avg LTP (60m)",
            "timestamp":    "Demo Timestamp",
        }
        display_df = display_df.rename(columns=rename_dict)

        display_df["LTP"] = display_df["LTP"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Bid Price"] = display_df["Bid Price"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Ask Price"] = display_df["Ask Price"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Bid Quantity"] = display_df["Bid Quantity"].apply(lambda x: f"{x:,}")
        display_df["Ask Quantity"] = display_df["Ask Quantity"].apply(lambda x: f"{x:,}")

        if "ETQ (5m)" in display_df.columns:
            display_df["ETQ (5m)"] = display_df["ETQ (5m)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
            display_df["ETQ (20m)"] = display_df["ETQ (20m)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")
            display_df["ETQ (60m)"] = display_df["ETQ (60m)"].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")

        if "Avg LTP (20m)" in display_df.columns:
            display_df["Avg LTP (20m)"] = display_df["Avg LTP (20m)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) and x > 0 else "N/A")
            display_df["Avg LTP (60m)"] = display_df["Avg LTP (60m)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) and x > 0 else "N/A")

        st.markdown(f"**{len(display_df)} Stocks** passed the Liquidity Filter (Bid & Ask Qty > 1,000,000 and ₹30 <= LTP <= ₹500):")

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Symbol":         st.column_config.TextColumn("Symbol",         width="medium"),
                "Company Name":   st.column_config.TextColumn("Company Name",   width="large"),
                "LTP":            st.column_config.TextColumn("LTP",            width="small"),
                "Bid Price":      st.column_config.TextColumn("Bid Price",      width="small"),
                "Bid Quantity":   st.column_config.TextColumn("Bid Quantity",   width="medium"),
                "Ask Price":      st.column_config.TextColumn("Ask Price",      width="small"),
                "Ask Quantity":   st.column_config.TextColumn("Ask Quantity",   width="medium"),
                "ETQ (5m)":       st.column_config.TextColumn("ETQ (5m)",       width="small"),
                "ETQ (20m)":      st.column_config.TextColumn("ETQ (20m)",      width="small"),
                "ETQ (60m)":      st.column_config.TextColumn("ETQ (60m)",      width="small"),
                "Avg LTP (20m)": st.column_config.TextColumn("Avg LTP (20m)", width="small"),
                "Avg LTP (60m)": st.column_config.TextColumn("Avg LTP (60m)", width="small"),
                "Demo Timestamp": st.column_config.TextColumn("Demo Timestamp", width="large"),
            },
            hide_index=True,
        )



        st.caption("ℹ️ Timestamp is simulated because live broker market data is not connected.")

    else:
        # Live mode fallback
        auth_status = scanner_service.broker_adapter.get_connection_status()
        if not auth_status.is_authenticated:
            st.error("Fyers API not connected. Please authenticate first.")
            return auto_refresh

        scanner_df = scanner_service.get_all_stocks_dataframe()
        if scanner_df.empty:
            st.info("Waiting for scanner data...")
            return auto_refresh

        from quant_signal.core.types import LiquiditySnapshot
        snapshots: list[LiquiditySnapshot] = []

        for _, row in scanner_df.iterrows():
            sym = row["symbol"]
            depth_service.subscribe_symbol(sym)
            depth_snap = depth_service.get_snapshot()
            if depth_snap and depth_snap.symbol == sym:
                snapshots.append(liquidity_service.classify_snapshot(depth_snap, row.get("company_name", "")))

        df = liquidity_service.filter_dataframe(snapshots)
        if df.empty:
            st.warning("No stocks currently pass the liquidity filter.")
            return auto_refresh

        st.dataframe(df, use_container_width=True, hide_index=True)

    return auto_refresh


def render_technical_indicators_view(
    tech_service: TechnicalIndicatorService,
    symbol_hint: str = "SUZLON-EQ",
) -> bool:
    """
    Renders the Phase 5 Technical Indicators Dashboard View.

    Displays:
    - Label: "Technical indicators are calculated from DEVELOPMENT / DEMO historical data."
    - Summary Cards (Total Analyzed, Bullish Count, Bearish Count, Avg Distance %)
    - Data Table: Symbol, Company Name, LTP, SMMA 20, SMMA 120, Trend, Distance %, Demo Timestamp
    - Trend Logic: SMMA 20 > SMMA 120 = Bullish 🟢, SMMA 20 < SMMA 120 = Bearish 🔴
    - Interactive Line Chart for selected stock overlaying LTP, SMMA 20, and SMMA 120.
    """
    st.subheader("📈 Technical Indicators  [DEVELOPMENT / DEMO DATA]")
    st.info("ℹ️ Technical indicators are calculated from DEVELOPMENT / DEMO historical data.")

    auto_refresh = st.checkbox("⚡ Auto-Refresh (1s)", value=False, key="tech_refresh")
    st.markdown("---")

    # ── Calculate SMMA Indicators across all demo stocks ─────────────────────
    df_raw = tech_service.get_indicators_dataframe()

    if df_raw.empty:
        st.warning("No technical indicator data available.")
        return auto_refresh

    # Calculate summary counts
    bullish_count = len(df_raw[df_raw["trend"] == "Bullish 🟢"])
    bearish_count = len(df_raw[df_raw["trend"] == "Bearish 🔴"])
    avg_dist = df_raw["distance_pct"].mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Stocks Analyzed", f"{len(df_raw)}")
    with m2:
        st.metric("Bullish Trends (SMMA20 > 120)", f"{bullish_count}")
    with m3:
        st.metric("Bearish Trends (SMMA20 < 120)", f"{bearish_count}")
    with m4:
        st.metric("Avg SMMA Distance %", f"{avg_dist:+.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 SMMA 20 & SMMA 120 Stock Table")

    # Format table for UI display
    display_df = df_raw.copy()
    display_df["Crossover Tag"] = display_df["crossover"].apply(
        lambda c: "🟢 BUY_CROSSOVER" if c == "BUY_CROSSOVER" else ("🔴 SELL_CROSSOVER" if c == "SELL_CROSSOVER" else "NONE")
    )
    display_df = display_df.rename(columns={
        "symbol":        "Symbol",
        "company_name":  "Company Name",
        "ltp":           "LTP",
        "smma_20":       "SMMA 20",
        "smma_120":      "SMMA 120",
        "trend":         "Trend",
        "distance_pct":  "Distance %",
        "Crossover Tag": "Crossover",
        "timestamp":     "Demo Timestamp",
    })

    display_df["LTP"] = display_df["LTP"].apply(lambda x: f"₹{x:,.2f}")
    display_df["SMMA 20"] = display_df["SMMA 20"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
    display_df["SMMA 120"] = display_df["SMMA 120"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
    display_df["Distance %"] = display_df["Distance %"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

    cols_order = [
        "Symbol", "Company Name", "LTP", "SMMA 20", "SMMA 120", "Trend", "Distance %", "Crossover", "Demo Timestamp"
    ]

    st.dataframe(
        display_df[cols_order],
        use_container_width=True,
        column_config={
            "Symbol":         st.column_config.TextColumn("Symbol",         width="medium"),
            "Company Name":   st.column_config.TextColumn("Company Name",   width="large"),
            "LTP":            st.column_config.TextColumn("LTP",            width="small"),
            "SMMA 20":        st.column_config.TextColumn("SMMA 20",        width="small"),
            "SMMA 120":       st.column_config.TextColumn("SMMA 120",       width="small"),
            "Trend":          st.column_config.TextColumn("Trend",          width="small"),
            "Distance %":     st.column_config.TextColumn("Distance %",     width="small"),
            "Crossover":      st.column_config.TextColumn("Crossover",      width="medium"),
            "Demo Timestamp": st.column_config.TextColumn("Demo Timestamp", width="large"),
        },
        hide_index=True,
    )

    st.caption("ℹ️ Timestamp is simulated because live broker market data is not connected.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📉 Interactive SMMA 20 & 120 Price Chart")

    symbols_list = list(df_raw["symbol"])
    clean_hint = symbol_hint.replace("NSE:", "")
    default_idx = symbols_list.index(clean_hint) if clean_hint in symbols_list else 0

    selected_symbol = st.selectbox(
        "Select Ticker Symbol to Plot SMMA Crossover Chart",
        options=symbols_list,
        index=default_idx,
    )

    # Fetch full historical series with SMMA 20 and SMMA 120
    stock_res = tech_service.calculate_stock_indicators(symbol=selected_symbol, days=250)
    history_df = stock_res["history_df"]

    if not history_df.empty and "smma_20" in history_df.columns:
        st.markdown(
            f"#### Price History & SMMA Overlay: **{selected_symbol}** | "
            f"Trend: **{stock_res['trend']}** | Distance: **{stock_res['distance_pct']:+.2f}%**"
        )

        chart_df = history_df[["timestamp", "close", "smma_20", "smma_120"]].copy()
        chart_df = chart_df.rename(columns={
            "close": "LTP / Close Price",
            "smma_20": "SMMA 20",
            "smma_120": "SMMA 120",
        })
        chart_df.set_index("timestamp", inplace=True)

        st.line_chart(
            chart_df,
            use_container_width=True,
        )
    else:
        st.warning(f"Insufficient historical data to plot SMMA chart for {selected_symbol}.")

    return auto_refresh


def render_crossover_signals_view(
    crossover_service: CrossoverService,
    symbol_hint: str = "SUZLON-EQ",
) -> bool:
    """
    Renders the Phase 6 SMMA Crossover Signals Dashboard View (STEP 2).

    Displays:
    - Page Title: Crossover Signals  [DEVELOPMENT / DEMO DATA]
    - DEMO Warning: "Crossover signals are calculated from demo historical data. They are NOT live trading signals."
    - Summary Cards: Total Stocks, BUY Crossovers, SELL Crossovers, No Crossover
    - Search & Filter Controls: All, BUY, SELL, WATCH + Search Symbol/Company
    - Main Signals Table: Symbol, Company Name, LTP, SMMA 20, SMMA 120, Trend, Distance %, Crossover, Signal, Demo Timestamp
    - Crossover Decision Explainability Section (Previous SMMA20, Previous SMMA120, Current SMMA20, Current SMMA120, Crossover Type, Signal, Timestamp)
    - Interactive Price & SMMA Chart with Historical Crossover Points.
    """
    st.subheader("🎯 Crossover Signals  [DEVELOPMENT / DEMO DATA]")
    st.warning(
        "⚠️ **DEVELOPMENT / DEMO DATA**\n\n"
        "Crossover signals are calculated from demo historical data. "
        "They are NOT live trading signals."
    )

    auto_refresh = st.checkbox("⚡ Auto-Refresh (1s)", value=False, key="crossover_refresh")
    st.markdown("---")

    # Fetch signals dataframe across entire demo universe
    df_signals = crossover_service.get_crossover_signals_dataframe()
    cards = crossover_service.get_crossover_summary_cards(df_signals)

    # ── Fetch historical crossovers & win rate metrics ──────────────────────
    prof_service = CrossoverProfitabilityService(crossover_service=crossover_service)
    win_metrics = prof_service.get_win_rate_metrics()


    # ── Summary Cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Stocks Evaluated", f"{cards['total_stocks']}")
    with c2:
        st.metric("BUY Win Rate %", f"{win_metrics['buy_win_rate_pct']:.1f}%", delta=f"{win_metrics['buy_profitable_count']}/{win_metrics['buy_evaluated_trades']} Trades")
    with c3:
        st.metric("SELL Win Rate %", f"{win_metrics['sell_win_rate_pct']:.1f}%", delta=f"{win_metrics['sell_profitable_count']}/{win_metrics['sell_evaluated_trades']} Trades")
    with c4:
        st.metric("Overall Win Rate %", f"{win_metrics['overall_win_rate_pct']:.1f}%", delta=f"{win_metrics['overall_profitable_count']}/{win_metrics['overall_evaluated_trades']} Total")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 SMMA 20/120 Crossover Signals Table")

    # ── 2. Filters & Search ───────────────────────────────────────────────────
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        signal_filter = st.radio(
            "Filter by Signal",
            options=["All", "BUY", "SELL", "WATCH"],
            horizontal=True,
            key="crossover_signal_filter_radio",
        )
    with f_col2:
        search_query = st.text_input(
            "🔍 Search Symbol or Company Name",
            value="",
            placeholder="e.g. SUZLON or Yes Bank",
            key="crossover_search_input",
        )

    # Apply Filters
    filtered_df = df_signals.copy()

    if signal_filter != "All":
        filtered_df = filtered_df[filtered_df["signal"] == signal_filter]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["symbol"].str.lower().str.contains(q) |
            filtered_df["company_name"].str.lower().str.contains(q)
        ]

    if filtered_df.empty:
        st.info("No stocks match the selected signal filter or search query.")
    else:
        # Prepare display DataFrame with badges
        display_df = filtered_df.copy()
        display_df["Crossover Badge"] = display_df["crossover"].apply(
            lambda c: "🟢 BUY CROSSOVER" if "BUY" in c else ("🔴 SELL CROSSOVER" if "SELL" in c else "🟡 NO CROSSOVER")
        )
        display_df["Signal Badge"] = display_df["signal"].apply(
            lambda s: "🟢 BUY" if s == "BUY" else ("🔴 SELL" if s == "SELL" else "🟡 WATCH")
        )

        display_df = display_df.rename(columns={
            "symbol":          "Symbol",
            "company_name":    "Company Name",
            "ltp":             "LTP",
            "smma_20":         "SMMA 20",
            "smma_120":        "SMMA 120",
            "trend":           "Trend",
            "distance_pct":    "Distance %",
            "Crossover Badge": "Crossover",
            "Signal Badge":    "Signal",
            "timestamp":       "Demo Timestamp",
        })

        # Format numeric columns
        display_df["LTP"] = display_df["LTP"].apply(lambda x: f"₹{x:,.2f}")
        display_df["SMMA 20"] = display_df["SMMA 20"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        display_df["SMMA 120"] = display_df["SMMA 120"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        display_df["Distance %"] = display_df["Distance %"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

        cols_order = [
            "Symbol", "Company Name", "LTP", "SMMA 20", "SMMA 120",
            "Trend", "Distance %", "Crossover", "Signal", "Demo Timestamp"
        ]

        st.dataframe(
            display_df[cols_order],
            use_container_width=True,
            column_config={
                "Symbol":         st.column_config.TextColumn("Symbol",         width="medium"),
                "Company Name":   st.column_config.TextColumn("Company Name",   width="large"),
                "LTP":            st.column_config.TextColumn("LTP",            width="small"),
                "SMMA 20":        st.column_config.TextColumn("SMMA 20",        width="small"),
                "SMMA 120":       st.column_config.TextColumn("SMMA 120",       width="small"),
                "Trend":          st.column_config.TextColumn("Trend",          width="small"),
                "Distance %":     st.column_config.TextColumn("Distance %",     width="small"),
                "Crossover":      st.column_config.TextColumn("Crossover",      width="medium"),
                "Signal":         st.column_config.TextColumn("Signal",         width="small"),
                "Demo Timestamp": st.column_config.TextColumn("Demo Timestamp", width="large"),
            },
            hide_index=True,
        )

    st.caption("ℹ️ Timestamp is simulated because live broker market data is not connected.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Crossover Details (Decision Explainability)")

    demo_symbols = crossover_service.tech_service.provider.get_symbols()
    clean_hint = symbol_hint.replace("NSE:", "")
    default_idx = demo_symbols.index(clean_hint) if clean_hint in demo_symbols else 0

    selected_symbol = st.selectbox(
        "Select Ticker Symbol to Inspect Decision Explainability",
        options=demo_symbols,
        index=default_idx,
        key="crossover_explain_symbol",
    )

    # Get indicator details for selected stock
    stock_res = crossover_service.tech_service.calculate_stock_indicators(symbol=selected_symbol, days=250)
    crossover_type = stock_res.get("crossover", "NONE")
    signal_type = crossover_service.map_crossover_to_signal(crossover_type)

    history_df = stock_res.get("history_df", pd.DataFrame())
    prev_20 = np.nan
    prev_120 = np.nan
    curr_20 = stock_res.get("smma_20", np.nan)
    curr_120 = stock_res.get("smma_120", np.nan)

    if not history_df.empty and "smma_20" in history_df.columns:
        valid_rows = history_df.dropna(subset=["smma_20", "smma_120"])
        if len(valid_rows) >= 2:
            prev_row = valid_rows.iloc[-2]
            prev_20 = float(prev_row["smma_20"])
            prev_120 = float(prev_row["smma_120"])

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Previous SMMA20", f"₹{prev_20:,.2f}" if pd.notna(prev_20) else "N/A")
        st.metric("Current SMMA20", f"₹{curr_20:,.2f}" if pd.notna(curr_20) else "N/A")
    with d2:
        st.metric("Previous SMMA120", f"₹{prev_120:,.2f}" if pd.notna(prev_120) else "N/A")
        st.metric("Current SMMA120", f"₹{curr_120:,.2f}" if pd.notna(curr_120) else "N/A")
    with d3:
        crossover_label = "BUY CROSSOVER" if crossover_type == "BUY_CROSSOVER" else ("SELL CROSSOVER" if crossover_type == "SELL_CROSSOVER" else "NO CROSSOVER")
        st.metric("Crossover Type", crossover_label)
        st.metric("Signal", signal_type)
    with d4:
        st.metric("LTP", f"₹{stock_res['ltp']:,.2f}")
        st.metric("Timestamp", "10:30:00 (Simulated)")

    # Decision rationale explanation
    with st.expander("🧠 Decision Logic Explanation", expanded=True):
        if crossover_type == "BUY_CROSSOVER":
            st.success(
                f"🟢 **BUY Signal Triggered**: Previous SMMA20 (₹{prev_20:,.2f}) $\\le$ Previous SMMA120 (₹{prev_120:,.2f}) "
                f"AND Current SMMA20 (₹{curr_20:,.2f}) > Current SMMA120 (₹{curr_120:,.2f})."
            )
        elif crossover_type == "SELL_CROSSOVER":
            st.error(
                f"🔴 **SELL Signal Triggered**: Previous SMMA20 (₹{prev_20:,.2f}) $\\ge$ Previous SMMA120 (₹{prev_120:,.2f}) "
                f"AND Current SMMA20 (₹{curr_20:,.2f}) < Current SMMA120 (₹{curr_120:,.2f})."
            )
        else:
            st.info(
                f"🟡 **Signal = WATCH (No Crossover)**: No fresh crossing event occurred on the latest bar. "
                f"Previous SMMA20 (₹{prev_20:,.2f}), Previous SMMA120 (₹{prev_120:,.2f}) $\\rightarrow$ "
                f"Current SMMA20 (₹{curr_20:,.2f}), Current SMMA120 (₹{curr_120:,.2f}). "
                f"Trend is {stock_res['trend']}, but signal remains WATCH until a new crossover occurs."
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Interactive Price & SMMA Chart with Historical Crossovers")

    if not history_df.empty and "smma_20" in history_df.columns:
        # Step 3C: Historical Crossover Events from Step 3A scanner
        company_name_map = {}
        if hasattr(crossover_service.tech_service.provider, "_data"):
            company_name_map = {
                k: v.get("company", k)
                for k, v in crossover_service.tech_service.provider._data.items()
            }
        comp_name = company_name_map.get(selected_symbol, selected_symbol)

        hist_crossovers = scan_historical_crossovers(
            symbol=selected_symbol,
            company_name=comp_name,
            df=history_df,
        )

        try:
            # Melt history_df for multi-line chart (LTP, SMMA 20, SMMA 120)
            melt_df = history_df[["timestamp", "close", "smma_20", "smma_120"]].copy()
            melt_df = melt_df.rename(columns={
                "close": "LTP / Close",
                "smma_20": "SMMA 20",
                "smma_120": "SMMA 120",
            })
            melted = melt_df.melt(
                id_vars=["timestamp"],
                value_vars=["LTP / Close", "SMMA 20", "SMMA 120"],
                var_name="Metric",
                value_name="Price",
            )

            line_chart = alt.Chart(melted).mark_line().encode(
                x=alt.X("timestamp:T", title="Date / Time"),
                y=alt.Y("Price:Q", title="Price (₹)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(
                        domain=["LTP / Close", "SMMA 20", "SMMA 120"],
                        range=["#1f77b4", "#ff7f0e", "#9467bd"],
                    ),
                    title="Indicator",
                ),
            )

            # Step 3C: BUY and SELL markers from Step 3A historical crossover events
            valid_markers = [
                e for e in hist_crossovers
                if e.get("timestamp") is not None and pd.notna(e.get("close_price"))
            ]

            if valid_markers:
                markers_data = []
                for ev in valid_markers:
                    c_type = ev["crossover_type"]
                    markers_data.append({
                        "timestamp": ev["timestamp"],
                        "price": ev["close_price"],
                        "crossover_type": c_type,
                        "label": "🟢 BUY" if c_type == "BUY" else "🔴 SELL",
                        "smma_20": ev["curr_smma_20"],
                        "smma_120": ev["curr_smma_120"],
                    })
                markers_df = pd.DataFrame(markers_data)

                point_layer = alt.Chart(markers_df).mark_point(filled=True, size=180).encode(
                    x="timestamp:T",
                    y="price:Q",
                    color=alt.Color(
                        "crossover_type:N",
                        scale=alt.Scale(domain=["BUY", "SELL"], range=["#2ca02c", "#d62728"]),
                        title="Crossover Marker",
                    ),
                    shape=alt.Shape(
                        "crossover_type:N",
                        scale=alt.Scale(domain=["BUY", "SELL"], range=["triangle-up", "triangle-down"]),
                        title="Crossover Marker",
                    ),
                    tooltip=["timestamp:T", "crossover_type:N", "price:Q", "smma_20:Q", "smma_120:Q"],
                )

                text_layer = alt.Chart(markers_df).mark_text(align="left", dx=8, dy=-8, fontSize=11, fontWeight="bold").encode(
                    x="timestamp:T",
                    y="price:Q",
                    text="label:N",
                    color=alt.Color(
                        "crossover_type:N",
                        scale=alt.Scale(domain=["BUY", "SELL"], range=["#2ca02c", "#d62728"]),
                    ),
                )

                combined_chart = (line_chart + point_layer + text_layer).properties(height=420)
            else:
                combined_chart = line_chart.properties(height=420)

            st.altair_chart(combined_chart, use_container_width=True)

        except Exception as chart_err:
            logger.warning(f"Altair chart generation fallback for {selected_symbol}: {chart_err}")
            fallback_df = history_df[["timestamp", "close", "smma_20", "smma_120"]].copy()
            fallback_df = fallback_df.rename(columns={
                "close": "LTP / Close Price",
                "smma_20": "SMMA 20",
                "smma_120": "SMMA 120",
            })
            fallback_df.set_index("timestamp", inplace=True)
            st.line_chart(fallback_df, use_container_width=True)

        # Historical Crossovers Table
        if hist_crossovers:
            st.markdown(f"#### 📍 Historical Crossover Events ({len(hist_crossovers)} detected in 250 days)")
            co_df = pd.DataFrame(hist_crossovers)
            co_df["Signal Tag"] = co_df["signal"].apply(lambda s: "🟢 BUY" if s == "BUY" else "🔴 SELL")
            co_df["LTP"] = co_df["close_price"].apply(
                lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A"
            )
            co_df["SMMA 20"] = co_df["curr_smma_20"].apply(lambda x: f"₹{x:,.2f}")
            co_df["SMMA 120"] = co_df["curr_smma_120"].apply(lambda x: f"₹{x:,.2f}")
            co_df["Prev SMMA 20"] = co_df["prev_smma_20"].apply(lambda x: f"₹{x:,.2f}")
            co_df["Prev SMMA 120"] = co_df["prev_smma_120"].apply(lambda x: f"₹{x:,.2f}")
            co_df["Timestamp"] = co_df["timestamp"].apply(
                lambda ts: ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
            )

            st.dataframe(
                co_df[["Signal Tag", "Timestamp", "LTP", "SMMA 20", "SMMA 120", "Prev SMMA 20", "Prev SMMA 120"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(f"No historical crossovers detected for {selected_symbol} in the 250-day window.")
    else:
        st.warning(f"Insufficient historical data to plot chart for {selected_symbol}.")

    # ── Step 3B: Historical Crossover Events Section ───────────────────────
    st.markdown("---")
    st.markdown("## 📅 Historical Crossover Profitability Evaluation  [DEVELOPMENT / DEMO DATA]")
    st.warning(
        "⚠️ **DEVELOPMENT / DEMO DATA** — Trade profitability outcomes are evaluated over a 5-bar holding horizon "
        "using historical candle series. They are NOT guaranteed future trading results."
    )
    st.caption("ℹ️ Timestamp is simulated because live broker market data is not connected.")

    # ── Ticker selection ───────────────────────────────────────────────────────
    hist_col1, hist_col2 = st.columns([1, 2])
    with hist_col1:
        all_demo_symbols = crossover_service.tech_service.provider.get_symbols()
        hist_clean_hint = symbol_hint.replace("NSE:", "")
        hist_default_idx = (
            all_demo_symbols.index(hist_clean_hint)
            if hist_clean_hint in all_demo_symbols
            else 0
        )
        hist_symbol = st.selectbox(
            "📌 Select Ticker Symbol to Inspect Evaluated Trade Outcomes",
            options=all_demo_symbols,
            index=hist_default_idx,
            key="hist_crossover_symbol_select",
        )
    with hist_col2:
        hist_signal_filter = st.radio(
            "Filter by Signal Result",
            options=["All", "PROFITABLE", "UNPROFITABLE", "INSUFFICIENT_DATA"],
            horizontal=True,
            key="hist_crossover_profit_filter",
        )

    # ── Fetch evaluated crossover trade outcomes ─────────────────────────────
    eval_results = prof_service.evaluate_symbol_crossovers(symbol=hist_symbol, days=250, horizon_bars=5)

    # Summary counts for selected symbol
    sym_eval_trades = [r for r in eval_results if r.result in ("PROFITABLE", "UNPROFITABLE")]
    sym_prof_count = sum(1 for r in sym_eval_trades if r.result == "PROFITABLE")
    sym_win_rate = round((sym_prof_count / len(sym_eval_trades)) * 100.0, 1) if sym_eval_trades else 0.0

    hc1, hc2, hc3, hc4, hc5 = st.columns(5)
    with hc1:
        st.metric("Symbol Crossovers", str(len(eval_results)))
    with hc2:
        st.metric("Evaluated Trades", str(len(sym_eval_trades)))
    with hc3:
        st.metric("Profitable Trades", str(sym_prof_count))
    with hc4:
        st.metric("Win Rate %", f"{sym_win_rate:.1f}%")
    with hc5:
        st.metric("Holding Horizon", "5 Bars")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 📋 Evaluated Trade Outcomes Table — {hist_symbol}")

    # Apply result filter
    filtered_results = eval_results
    if hist_signal_filter != "All":
        filtered_results = [r for r in eval_results if r.result == hist_signal_filter]

    if not filtered_results:
        st.info(f"No crossover trades match the selected filter '{hist_signal_filter}' for {hist_symbol}.")
    else:
        prof_rows = []
        for r in filtered_results:
            ts_str = (
                r.crossover_time.strftime("%Y-%m-%d %H:%M")
                if hasattr(r.crossover_time, "strftime")
                else str(r.crossover_time or "N/A")
            )
            out_badge = (
                "🟢 PROFITABLE" if r.result == "PROFITABLE"
                else ("🔴 UNPROFITABLE" if r.result == "UNPROFITABLE" else "🟡 INSUFFICIENT_DATA")
            )
            sig_badge = "🟢 BUY" if r.signal == "BUY" else "🔴 SELL"
            dec_badge = "🟢 ACCEPT" if r.trade_decision == "ACCEPT" else "🔴 AVOID"

            prof_rows.append({
                "Symbol":            r.symbol,
                "Company Name":      r.company_name,
                "Timestamp":         ts_str,
                "Signal":            sig_badge,
                "Trade Decision":    dec_badge,
                "Decision Reason":   r.decision_reason,
                "Entry Price":       f"₹{r.entry_price:,.2f}",
                "Exit Price":        f"₹{r.exit_price:,.2f}" if r.exit_price is not None else "N/A",
                "Horizon":           f"{r.evaluation_horizon} Bars",
                "PnL (₹)":           f"{r.pnl:+,.2f}" if r.pnl is not None else "N/A",
                "Return %":          f"{r.return_pct:+,.2f}%" if r.return_pct is not None else "N/A",
                "Trade Outcome":     out_badge,
                "AI Confidence %":   f"{r.ai_confidence_pct:.1f}%",
                "AI Recommendation": r.ai_recommendation,
                "Avoidance Rationale": r.avoidance_reason,
            })

        prof_df = pd.DataFrame(prof_rows)
        st.dataframe(
            prof_df[[
                "Symbol", "Signal", "Trade Decision", "AI Confidence %", "AI Recommendation",
                "Entry Price", "Exit Price", "PnL (₹)", "Return %", "Trade Outcome", "Decision Reason"
            ]],
            use_container_width=True,
            column_config={
                "Symbol":            st.column_config.TextColumn("Symbol",            width="medium"),
                "Signal":            st.column_config.TextColumn("Signal",            width="small"),
                "Trade Decision":    st.column_config.TextColumn("Trade Decision",    width="medium"),
                "AI Confidence %":   st.column_config.TextColumn("AI Confidence %",   width="small"),
                "AI Recommendation": st.column_config.TextColumn("AI Recommendation", width="medium"),
                "Entry Price":       st.column_config.TextColumn("Entry Price",       width="small"),
                "Exit Price":        st.column_config.TextColumn("Exit Price",        width="small"),
                "PnL (₹)":           st.column_config.TextColumn("PnL (₹)",           width="small"),
                "Return %":          st.column_config.TextColumn("Return %",          width="small"),
                "Trade Outcome":     st.column_config.TextColumn("Trade Outcome",     width="medium"),
                "Decision Reason":   st.column_config.TextColumn("Decision Reason",   width="large"),
            },
            hide_index=True,
        )

        with st.expander("💡 Trade Decision Rationale & Avoidance Explanation", expanded=True):
            for r in filtered_results:
                if r.trade_decision == "AVOID":
                    st.error(f"🔴 **{r.symbol}** | Decision: **AVOID** | Signal: **{r.signal}** @ {r.crossover_time} $\\rightarrow$ {r.decision_reason}")
                else:
                    st.success(f"🟢 **{r.symbol}** | Decision: **ACCEPT** | Signal: **{r.signal}** @ {r.crossover_time} $\\rightarrow$ {r.decision_reason}")


    return auto_refresh



def render_ai_ml_signals_view(
    ml_service: MLSignalService,
    symbol_hint: str = "SUZLON-EQ",
) -> bool:
    """
    Renders the Phase 7 AI / ML Signal Scoring Engine View.

    Displays:
    - Label: "AI/ML Signal Analysis [DEVELOPMENT / DEMO DATA]"
    - Warning: "AI/ML predictions are calculated from demo historical data and Scikit-learn model inference. They are NOT guaranteed financial advice."
    - Summary Cards: Total Evaluated, High-Confidence Signals (>65%), Bullish Recommendations, Avg AI Confidence %
    - Search & Filter Controls: All, STRONG BUY, BUY, HOLD, SELL
    - Main Signals Table: Symbol, Company Name, LTP, SMMA 20, SMMA 120, Trend, Crossover, AI Confidence %, Recommendation, Demo Timestamp
    - Model Explainability & Feature Importance Weights Panel.
    """
    st.subheader("🤖 AI/ML Signal Analysis  [DEVELOPMENT / DEMO DATA]")
    st.warning(
        "⚠️ **DEVELOPMENT / DEMO DATA**\n\n"
        "AI/ML probability scores and recommendations are calculated using Scikit-Learn Random Forest inference "
        "on deterministic demo data. They are NOT live financial advice or guaranteed trade predictions."
    )

    auto_refresh = st.checkbox("⚡ Auto-Refresh (1s)", value=False, key="ml_refresh")
    st.markdown("---")

    # Fetch ML signals DataFrame across entire demo universe
    df_ml = ml_service.get_ml_signals_dataframe()
    cards = ml_service.get_summary_metrics(df_ml)

    # ── Summary Cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Universe Evaluated", f"{cards['total_evaluated']}")
    with c2:
        st.metric("High Confidence (>65%)", f"{cards['high_confidence_count']}")
    with c3:
        st.metric("Bullish AI Signals", f"{cards['bullish_recommendations']}")
    with c4:
        st.metric("Avg AI Confidence %", f"{cards['avg_confidence']:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Scikit-Learn AI Signal Ranking Table")

    # ── Filters & Search ──────────────────────────────────────────────────────
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        rec_filter = st.radio(
            "Filter by AI Recommendation",
            options=["All", "STRONG BUY", "BUY", "HOLD", "SELL"],
            horizontal=True,
            key="ml_rec_filter_radio",
        )
    with f_col2:
        search_query = st.text_input(
            "🔍 Search Symbol or Company Name",
            value="",
            placeholder="e.g. SUZLON or Reliance",
            key="ml_search_input",
        )

    # Apply filters
    filtered_df = df_ml.copy()

    if rec_filter != "All":
        filtered_df = filtered_df[filtered_df["recommendation"].str.contains(rec_filter)]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["symbol"].str.lower().str.contains(q) |
            filtered_df["company_name"].str.lower().str.contains(q)
        ]

    if filtered_df.empty:
        st.info("No stocks match the selected AI recommendation filter or search query.")
    else:
        # Format table for display
        display_df = filtered_df.copy()
        display_df = display_df.rename(columns={
            "symbol":         "Symbol",
            "company_name":   "Company Name",
            "ltp":            "LTP",
            "smma_20":        "SMMA 20",
            "smma_120":       "SMMA 120",
            "trend":          "Trend",
            "crossover":      "Crossover",
            "confidence_pct": "AI Confidence %",
            "recommendation": "AI Recommendation",
            "timestamp":      "Demo Timestamp",
        })

        display_df["LTP"] = display_df["LTP"].apply(lambda x: f"₹{x:,.2f}")
        display_df["SMMA 20"] = display_df["SMMA 20"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        display_df["SMMA 120"] = display_df["SMMA 120"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        display_df["AI Confidence %"] = display_df["AI Confidence %"].apply(lambda x: f"{x:.1f}%")

        cols_order = [
            "Symbol", "Company Name", "LTP", "SMMA 20", "SMMA 120",
            "Trend", "Crossover", "AI Confidence %", "AI Recommendation", "Demo Timestamp"
        ]

        st.dataframe(
            display_df[cols_order],
            use_container_width=True,
            column_config={
                "Symbol":            st.column_config.TextColumn("Symbol",            width="medium"),
                "Company Name":      st.column_config.TextColumn("Company Name",      width="large"),
                "LTP":               st.column_config.TextColumn("LTP",               width="small"),
                "SMMA 20":           st.column_config.TextColumn("SMMA 20",           width="small"),
                "SMMA 120":          st.column_config.TextColumn("SMMA 120",          width="small"),
                "Trend":             st.column_config.TextColumn("Trend",             width="small"),
                "Crossover":         st.column_config.TextColumn("Crossover",         width="medium"),
                "AI Confidence %":   st.column_config.TextColumn("AI Confidence %",   width="medium"),
                "AI Recommendation": st.column_config.TextColumn("AI Recommendation", width="medium"),
                "Demo Timestamp":    st.column_config.TextColumn("Demo Timestamp",    width="large"),
            },
            hide_index=True,
        )

    st.caption("ℹ️ Timestamp is simulated because live broker market data is not connected.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 Model Decision Feature Importance Breakdown")

    importances = ml_service.get_feature_importances()
    imp_df = pd.DataFrame([
        {"Feature": k, "Importance Weight": f"{v * 100:.1f}%"} for k, v in importances.items()
    ])

    ic1, ic2 = st.columns([1, 1])
    with ic1:
        st.markdown("#### Random Forest Feature Weights")
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    with ic2:
        st.markdown("#### Individual Stock AI Inspection & Explainability")
        demo_symbols = ml_service.tech_service.provider.get_symbols()
        clean_hint = symbol_hint.replace("NSE:", "")
        default_idx = demo_symbols.index(clean_hint) if clean_hint in demo_symbols else 0

        selected_symbol = st.selectbox(
            "Select Ticker Symbol to Inspect AI Model Feature Vector",
            options=demo_symbols,
            index=default_idx,
            key="ml_inspect_symbol",
        )

        explain_data = ml_service.get_stock_explainability(selected_symbol)
        st.markdown(f"**Symbol**: **{explain_data['symbol']}** ({explain_data['company_name']})")
        st.markdown(f"**AI Confidence**: **{explain_data['confidence_pct']:.1f}%**")
        st.markdown(f"**AI Recommendation**: **{explain_data['recommendation']}**")

        rec_str = explain_data["recommendation"]
        if "BUY" in rec_str:
            st.success(explain_data["explanation"])
        elif "SELL" in rec_str:
            st.error(explain_data["explanation"])
        else:
            st.info(explain_data["explanation"])

        feat_df = pd.DataFrame(explain_data["features"])
        if not feat_df.empty:
            feat_df["Value"] = feat_df["value"].apply(lambda x: f"{x:+.2f}")
            feat_df["Importance Weight"] = feat_df["importance"].apply(lambda x: f"{x * 100:.1f}%")
            feat_display = feat_df[["feature", "Value", "Importance Weight"]].rename(columns={"feature": "Feature"})
            st.dataframe(feat_display, use_container_width=True, hide_index=True)

    return auto_refresh



