"""
Quantitative Mathematics, Risk Metrics, and Performance Calculations.
"""

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized Sharpe Ratio for a series of periodic returns.

    Args:
        returns: Array or Series of percentage daily/periodic returns.
        risk_free_rate: Annualized risk-free interest rate benchmark.
        periods_per_year: Trading frequency multiplier (252 for daily equity).

    Returns:
        float: Annualized Sharpe Ratio score.
    """
    if len(returns) == 0:
        return 0.0

    returns_arr = np.asarray(returns)
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1)

    if std_return == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = mean_return - rf_per_period
    return float((excess_returns / std_return) * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    """Calculates Maximum Drawdown (peak-to-trough drop percentage).

    Args:
        equity_curve: Historical cumulative equity or portfolio value array.

    Returns:
        float: Maximum percentage drawdown float (e.g. 0.15 for 15% drawdown).
    """
    if len(equity_curve) == 0:
        return 0.0

    arr = np.asarray(equity_curve)
    cumulative_max = np.maximum.accumulate(arr)
    drawdowns = (cumulative_max - arr) / cumulative_max
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def calculate_win_rate(trades_pnl: list[float] | np.ndarray) -> float:
    """Calculates trade Win Rate ratio.

    Args:
        trades_pnl: List of closed trade PnL figures.

    Returns:
        float: Ratio of winning trades in range [0.0, 1.0].
    """
    if len(trades_pnl) == 0:
        return 0.0

    arr = np.asarray(trades_pnl)
    wins = np.sum(arr > 0)
    return float(wins / len(arr))
