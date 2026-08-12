"""
Datetime and Indian Market Trading Hours Utilities for QuantSignal AI.
"""

from __future__ import annotations

from datetime import datetime, time
import pytz

# Standard Indian Market Timezone (Python 3.7 compatible via pytz)
IST_ZONE = pytz.timezone("Asia/Kolkata")

# NSE / BSE Normal Trading Hours (09:15 AM - 03:30 PM IST)
MARKET_OPEN_TIME: time = time(9, 15)
MARKET_CLOSE_TIME: time = time(15, 30)


def get_current_ist_time() -> datetime:
    """Returns current timestamp in Indian Standard Time (IST)."""
    return datetime.now(IST_ZONE)


def is_market_open(dt: datetime | None = None) -> bool:
    """Checks whether the given timestamp falls within active Indian equity market hours.

    Args:
        dt: Optional datetime object. Defaults to current IST time.

    Returns:
        bool: True if market is currently open for trading.
    """
    check_dt = dt or get_current_ist_time()

    # Check weekday (Monday = 0, Sunday = 6)
    if check_dt.weekday() >= 5:
        return False

    current_time = check_dt.time()
    return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME


def format_timestamp(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Formats datetime object to standard string representation.

    Args:
        dt: Input datetime.
        fmt: Desired format string.
    """
    return dt.strftime(fmt)
