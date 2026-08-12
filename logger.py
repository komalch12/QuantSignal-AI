"""
Root Logger Module Exposer for QuantSignal AI.

Proxies logging setup and helper utilities from `quant_signal.logger`.
"""

from quant_signal.logger import get_logger, log_execution_time, setup_logger

__all__ = ["setup_logger", "get_logger", "log_execution_time"]
