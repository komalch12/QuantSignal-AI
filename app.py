"""
QuantSignal AI - Streamlit Web Application Entry Point.

Auto-detects DEVELOPMENT / DEMO DATA MODE when FYERS_CLIENT_ID or
FYERS_SECRET_KEY are absent from the environment.

Phase 5: Technical Indicators (SMMA 20 and SMMA 120) dashboard integration.

In demo mode:
  - DemoScannerService and TechnicalIndicatorService are used
  - FyersBrokerAdapter and FyersAuthService are preserved but NOT invoked
  - UI clearly displays: Data Mode: DEVELOPMENT / DEMO
  - UI clearly displays: Live Broker: NOT CONNECTED
  - "Fyers Connected" is never shown
"""

from __future__ import annotations

import time
import streamlit as st

from config import settings
from quant_signal.brokers.fyers_adapter import FyersBrokerAdapter
from quant_signal.exceptions import QuantSignalException
from quant_signal.logger import get_logger, setup_logger
from quant_signal.services.fyers_auth_service import FyersAuthService
from quant_signal.services.liquidity_filter import LiquidityFilterService
from quant_signal.services.market_depth_service import MarketDepthService
from quant_signal.services.stock_scanner import StockScannerService
from quant_signal.services.demo_scanner_service import DemoScannerService
from quant_signal.services.technical_indicators import TechnicalIndicatorService
from quant_signal.services.crossover_service import CrossoverService
from quant_signal.services.ml_signal_service import MLSignalService
from quant_signal.ui.components import (
    render_auth_management_widget,
    render_demo_mode_banner,
    render_header,
    render_sidebar_status,
)
from quant_signal.ui.views import (
    render_stock_scanner_view,
    render_market_depth_view,
    render_liquidity_filter_view,
    render_technical_indicators_view,
    render_crossover_signals_view,
    render_ai_ml_signals_view,
)
from quant_signal.utils.datetime_utils import is_market_open
from quant_signal.utils.helpers import safe_rerun


# ── 1. Central Logger ─────────────────────────────────────────────────────────
logger = setup_logger(
    name="QuantSignal",
    log_level=settings.app.log_level,
    log_to_file=settings.app.log_to_file,
    log_file_path=settings.app.log_file_path,
)

# ── 2. Streamlit Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title=f"{settings.app.app_name} | Quantitative Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main Streamlit Application Lifecycle — auto-detects demo vs. live mode."""
    logger.info("Initializing QuantSignal AI dashboard interface...")

    # ── Determine data mode ───────────────────────────────────────────────────
    DEMO_MODE: bool = settings.is_demo_mode()

    if DEMO_MODE:
        logger.info(
            "DEVELOPMENT / DEMO DATA MODE activated. "
            "No Fyers credentials configured — using DemoScannerService & TechnicalIndicatorService."
        )

    # ── Render Header ─────────────────────────────────────────────────────────
    render_header(app_name=settings.app.app_name)

    # ── Always show demo banner in demo mode ──────────────────────────────────
    if DEMO_MODE:
        render_demo_mode_banner()

    # ── Always preserve Fyers architecture (even in demo mode) ────────────────
    auth_service = FyersAuthService(config=settings.fyers)
    broker_adapter = FyersBrokerAdapter(config=settings.fyers, auth_service=auth_service)
    liquidity_service = LiquidityFilterService()

    # ── Wire services ─────────────────────────────────────────────────────────
    if DEMO_MODE:
        scanner_service = DemoScannerService(min_price=30.0, max_price=500.0)
        tech_service = TechnicalIndicatorService(provider=scanner_service.provider)
        crossover_service = CrossoverService(tech_service=tech_service)
        ml_service = MLSignalService(tech_service=tech_service, crossover_service=crossover_service)
        depth_service = None  # Scanner handles depth in demo mode
        auth_status = None    # No broker auth in demo mode
    else:
        scanner_service = StockScannerService(
            broker_adapter=broker_adapter, min_price=30.0, max_price=500.0
        )
        depth_service = MarketDepthService(broker_adapter=broker_adapter)
        tech_service = TechnicalIndicatorService()
        crossover_service = CrossoverService(tech_service=tech_service)
        ml_service = MLSignalService(tech_service=tech_service, crossover_service=crossover_service)

        try:
            broker_adapter.authenticate()
        except QuantSignalException as q_err:
            logger.error(f"QuantSignal domain exception during setup: {q_err}")
            st.error(f"Domain Error: {q_err.message}")
        except Exception as err:
            logger.critical(f"Unhandled initialization error: {err}", exc_info=True)
            st.error(f"System Error: Could not initialize engine. {err}")

        auth_status = auth_service.get_status()

    # ── Render Sidebar ────────────────────────────────────────────────────────
    controls = render_sidebar_status(
        auth_status=auth_status,
        is_market_open=is_market_open(),
        is_demo=DEMO_MODE,
    )

    # ── Dashboard Tabs ────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 NSE Scanner" if DEMO_MODE else "🔍 Live NSE Scanner",
        "📊 Market Depth",
        "💧 Liquidity Filter",
        "📈 Technical Indicators",
        "🎯 Crossover Signals",
        "🤖 AI/ML Signals",
        "🔑 Fyers API Auth",
    ])

    should_auto_refresh = False

    with tab1:
        if render_stock_scanner_view(scanner_service=scanner_service):
            should_auto_refresh = True

    with tab2:
        if DEMO_MODE:
            if render_market_depth_view(
                depth_service=None,
                symbol=controls.get("symbol", ""),
                demo_scanner=scanner_service,
            ):
                should_auto_refresh = True
        else:
            if render_market_depth_view(
                depth_service=depth_service,
                symbol=controls.get("symbol", ""),
            ):
                should_auto_refresh = True

    with tab3:
        if DEMO_MODE:
            if render_liquidity_filter_view(
                liquidity_service=liquidity_service,
                depth_service=None,
                scanner_service=scanner_service,
            ):
                should_auto_refresh = True
        else:
            if render_liquidity_filter_view(
                liquidity_service=liquidity_service,
                depth_service=depth_service,
                scanner_service=scanner_service,
            ):
                should_auto_refresh = True

    with tab4:
        if render_technical_indicators_view(
            tech_service=tech_service,
            symbol_hint=controls.get("symbol", "SUZLON-EQ"),
        ):
            should_auto_refresh = True

    with tab5:
        if render_crossover_signals_view(
            crossover_service=crossover_service,
            symbol_hint=controls.get("symbol", "SUZLON-EQ"),
        ):
            should_auto_refresh = True

    with tab6:
        if render_ai_ml_signals_view(
            ml_service=ml_service,
            symbol_hint=controls.get("symbol", "SUZLON-EQ"),
        ):
            should_auto_refresh = True

    with tab7:
        render_auth_management_widget(auth_service=auth_service)

    # ── Auto-refresh (skipped in demo mode by default) ────────────────────────
    if should_auto_refresh and not DEMO_MODE:
        time.sleep(1)
        safe_rerun()


if __name__ == "__main__":
    main()
