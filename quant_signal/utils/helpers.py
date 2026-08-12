"""
General Utility Helper Functions for QuantSignal AI.
"""

from typing import Any


def format_currency(value: float, currency_symbol: str = "₹") -> str:
    """Format numeric value into standard currency string representation.

    Args:
        value: Numeric currency amount.
        currency_symbol: Currency symbol prefix (default Indian Rupee).

    Returns:
        Formatted currency string.
    """
    return f"{currency_symbol}{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal ratio into formatted percentage string.

    Args:
        value: Percentage float (e.g. 0.054 -> '5.40%').
        decimals: Decimal precision points.

    Returns:
        Formatted percentage string.
    """
    return f"{value * 100:.{decimals}f}%"


def safe_float_convert(val: Any, default: float = 0.0) -> float:
    """Safely converts arbitrary input to float with fallback default.

    Args:
        val: Input object.
        default: Fallback return value if conversion fails.
    """
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_rerun() -> None:
    """Reruns the Streamlit script execution safely across Streamlit versions."""
    import streamlit as st

    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

