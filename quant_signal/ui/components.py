"""
Reusable Streamlit UI Components for QuantSignal AI.
"""

from __future__ import annotations

from typing import Any
import streamlit as st

from quant_signal.services.fyers_auth_service import FyersAuthService, FyersConnectionStatus
from quant_signal.utils.helpers import format_currency, format_percentage, safe_rerun


def render_header(app_name: str = "QuantSignal AI") -> None:
    """Renders sleek, high-contrast application header."""
    st.markdown(
        f"""
        <div style="
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1.25rem 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        ">
            <h1 style="margin:0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: #ffffff; font-weight: 800; font-size: 2rem; letter-spacing: -0.02em;">
                <span style="color: #f59e0b;">⚡</span> {app_name} <span style="font-size: 0.5em; color: #38bdf8; background: rgba(56, 189, 248, 0.15); padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.3); font-weight: 600; vertical-align: middle;">v1.0</span>
            </h1>
            <p style="margin: 0.4rem 0 0 0; color: #cbd5e1; font-size: 0.95rem; font-family: 'Inter', sans-serif;">
                Production-Ready AI Trading Signal &amp; Quantitative Analysis Dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )




def render_demo_mode_banner() -> None:
    """
    Renders a clearly visible DEVELOPMENT / DEMO DATA MODE banner.

    Must be called when no live broker credentials are configured.
    This banner MUST always be visible during demo mode — never hide or
    suppress it.
    """
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #3d2c00 0%, #5a3e00 100%);
            border: 2px solid #f5a623;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.25rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <p style="
                        margin: 0;
                        font-size: 1.05rem;
                        font-weight: 700;
                        color: #f5a623;
                        font-family: 'Inter', 'Segoe UI', sans-serif;
                        letter-spacing: 0.05em;
                    ">
                        DEVELOPMENT / DEMO DATA MODE
                    </p>
                    <p style="
                        margin: 0.25rem 0 0 0;
                        font-size: 0.88rem;
                        color: #e8c97a;
                        font-family: 'Inter', 'Segoe UI', sans-serif;
                    ">
                        Demo data is being used because no broker account / API credentials are configured.
                        This data is <strong>NOT live market data</strong>. All values are fixed sample data
                        for development and demonstration purposes only.
                    </p>
                </div>
            </div>
            <div style="
                display: flex;
                gap: 2rem;
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid #7a5c00;
            ">
                <span style="font-size: 0.82rem; color: #f0c96b; font-family: monospace;">
                    📊 Data Mode: <strong>DEVELOPMENT / DEMO</strong>
                </span>
                <span style="font-size: 0.82rem; color: #f0c96b; font-family: monospace;">
                    🔌 Live Broker: <strong>NOT CONNECTED</strong>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(
    total_signals: int,
    win_rate: float,
    sharpe_ratio: float,
    current_drawdown: float,
) -> None:
    """Renders top summary metric cards across 4 columns."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Signals", value=str(total_signals))
    with col2:
        st.metric(label="Win Rate", value=format_percentage(win_rate))
    with col3:
        st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")
    with col4:
        st.metric(label="Max Drawdown", value=format_percentage(current_drawdown))


def render_phase4_summary_metrics(
    total_stocks: int,
    ltp_matching: int,
    bid_qty_matching: int,
    ask_qty_matching: int,
    liquidity_matching: int,
) -> None:
    """
    Renders Phase 4 summary metrics across 5 columns:
    - Total Stocks
    - LTP Matching (₹30 - ₹500)
    - Bid Quantity Matching (> 1,000,000)
    - Ask Quantity Matching (> 1,000,000)
    - Liquidity Matching (LTP + Bid & Ask Qty > 1M)
    """
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="Total Stocks", value=f"{total_stocks:,}")
    with m2:
        st.metric(label="LTP Matching (₹30-500)", value=f"{ltp_matching:,}")
    with m3:
        st.metric(label="Bid Qty Matching (>1M)", value=f"{bid_qty_matching:,}")
    with m4:
        st.metric(label="Ask Qty Matching (>1M)", value=f"{ask_qty_matching:,}")
    with m5:
        st.metric(label="Liquidity Matching", value=f"{liquidity_matching:,}")



def render_sidebar_status(
    auth_status: FyersConnectionStatus | None = None,
    is_market_open: bool = True,
    is_demo: bool = False,
) -> dict[str, Any]:
    """Renders sidebar controls and connection/data-mode status.

    In demo mode, shows 'Not Configured' for Fyers and clearly labels
    the data provider as 'Demo'. Never shows 'Fyers Connected' in demo mode.

    Args:
        auth_status: Active FyersConnectionStatus object.
        is_market_open: Whether Indian equity market is currently open.
        is_demo: True when running in DEVELOPMENT / DEMO DATA MODE.

    Returns:
        dict: User selected sidebar parameters.
    """
    st.sidebar.title("🎛️ Trading Controls")

    # ── Authentication Diagnostics ──────────────────────────────────────────
    st.sidebar.markdown("#### 🔌 Connection Status")

    if is_demo:
        # Demo mode — never show "Connected" for Fyers
        st.sidebar.markdown("**FYERS:**  🔴 Not Configured")
        st.sidebar.markdown("**Data Provider:**  🟡 Demo")
        st.sidebar.markdown("**Live Market Data:**  🔴 Unavailable")
    else:
        is_auth = auth_status.is_authenticated if auth_status else False
        status_color = "🟢" if is_auth else "🔴"
        st.sidebar.markdown(
            f"**Fyers API:** {status_color} {'Authenticated' if is_auth else 'Disconnected'}"
        )
        if auth_status and auth_status.is_authenticated:
            st.sidebar.caption(f"👤 User: **{auth_status.user_name}** ({auth_status.fy_id})")
            st.sidebar.caption(f"🔑 Source: `{auth_status.token_source}`")
        elif auth_status and auth_status.error_message:
            st.sidebar.caption(f"⚠️ Error: {auth_status.error_message}")
        st.sidebar.markdown(f"**Data Provider:**  🟢 Fyers Live")
        st.sidebar.markdown(f"**Live Market Data:**  {'🟢 Available' if is_auth else '🔴 Unavailable'}")

    mkt_status = "🟢 Open" if is_market_open else "🟠 Closed"
    st.sidebar.markdown(f"**NSE Market Hours:** {mkt_status}")

    st.sidebar.divider()

    symbol = st.sidebar.text_input("Ticker Symbol", value="NSE:NIFTY50-INDEX")
    timeframe = st.sidebar.selectbox(
        "Candle Timeframe", options=["1m", "3m", "5m", "15m", "1h", "1d"], index=2
    )
    min_confidence = st.sidebar.slider(
        "Min Signal Confidence", min_value=0.5, max_value=0.95, value=0.65, step=0.05
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "min_confidence": min_confidence,
    }


def render_auth_management_widget(auth_service: FyersAuthService) -> None:
    """Renders safe Fyers API Authentication panel."""
    st.subheader("🔑 FYERS API Authentication")

    status = auth_service.get_status()
    config = auth_service.config

    from quant_signal.services.fyers_auth_service import ConnectionState

    has_token = bool(auth_service.access_token)

    # 1. State logic
    if status.state in (ConnectionState.AUTHENTICATED, ConnectionState.FEED_CONNECTED, ConnectionState.MARKET_CLOSED):
        auth_status_str = "Connected"
    elif status.state == ConnectionState.AUTH_NOT_CONFIGURED:
        auth_status_str = "Not Configured"
    else:
        auth_status_str = "Failed"

    # Fyers SDK logic
    try:
        import fyers_apiv3
        sdk_installed = True
    except ImportError:
        sdk_installed = False

    st.markdown("### FYERS API CONFIGURATION")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Client ID**: {'Configured' if config.client_id else 'Missing'}")
        st.markdown(f"**Secret Key**: {'Configured' if config.secret_key else 'Missing'}")
        st.markdown(f"**Redirect URI**: {'Configured' if config.redirect_uri else 'Missing'}")
        st.markdown(f"**Access Token**: {'Present' if has_token else 'Missing'}")
    with c2:
        st.markdown(f"**FYERS SDK**: {'Installed' if sdk_installed else 'Missing'}")
        st.markdown(f"**Authentication**: {auth_status_str}")

    if auth_status_str == "Failed" and status.error_message:
        st.error(f"Error Details: {status.error_message}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Authentication Flow")
        can_generate_url = bool(config.client_id and config.secret_key and config.redirect_uri)
        if st.button("🌐 Generate Login URL", disabled=not can_generate_url):
            try:
                auth_url = auth_service.generate_auth_url()
                st.markdown(f"[👉 Click here to log in to Fyers]({auth_url})")
            except Exception as err:
                st.error(f"Error generating Auth URL: {err}")

        if not can_generate_url:
            st.info("Please configure Client ID, Secret Key, and Redirect URI to generate the URL.")

    with col2:
        st.markdown("#### Exchange Auth Code")
        auth_code_input = st.text_input("Paste Auth Code from Redirect URL", type="password")
        if st.button("⚡ Generate Access Token"):
            if auth_code_input:
                try:
                    res = auth_service.generate_token_from_auth_code(auth_code_input.strip())
                    st.success("Successfully generated & saved access token!")
                    safe_rerun()
                except Exception as err:
                    st.error(f"Failed to generate access token: {err}")
            else:
                st.warning("Please paste authorization code first.")
