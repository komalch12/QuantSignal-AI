"""
Structured Logging Module for QuantSignal AI.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Default log format string
DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "QuantSignal",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file_path: str = "logs/quantsignal.log",
) -> logging.Logger:
    """Configures and returns a structured Logger instance.

    Args:
        name: Logger module name.
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to sink logs to a file.
        log_file_path: Output filepath for logs.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger: logging.Logger = logging.getLogger(name)
    level: int = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is re-initialized
    if logger.hasHandlers():
        return logger

    formatter: logging.Formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console Handler
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        try:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler: logging.FileHandler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as err:
            logger.warning(f"Could not initialize file logging at {log_file_path}: {err}")

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Utility function to get or create a child logger for a specific module.

    Args:
        module_name: Name of the current module (__name__).

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(f"QuantSignal.{module_name}")


def log_execution_time(logger: logging.Logger | None = None) -> Callable[[F], F]:
    """Decorator to log function entry, exit, execution duration, and exceptions.

    Args:
        logger: Logger instance to output to. Defaults to root module logger.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_logger = logger or get_logger(func.__module__)
            func_name = func.__qualname__
            target_logger.debug(f"Entering '{func_name}'")
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                target_logger.debug(f"Exited '{func_name}' in {elapsed:.4f} seconds")
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                target_logger.error(
                    f"Exception in '{func_name}' after {elapsed:.4f}s: {exc}",
                    exc_info=True,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
