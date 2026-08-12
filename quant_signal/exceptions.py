"""
Custom Exception Hierarchy for QuantSignal AI.
"""

from __future__ import annotations

from typing import Any


class QuantSignalException(Exception):
    """Base exception class for all QuantSignal AI errors."""

    def __init__(self, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.payload: dict[str, Any] = payload or {}

    def __str__(self) -> str:
        if self.payload:
            return f"{self.message} | Context: {self.payload}"
        return self.message


class ConfigurationError(QuantSignalException):
    """Raised when environment variables or config validation fails."""
    pass


class DataValidationError(QuantSignalException):
    """Raised when market data fails validation constraints (e.g. missing columns, invalid OHLC)."""
    pass


class BrokerError(QuantSignalException):
    """Base exception for broker interaction failures."""
    pass


class BrokerAuthenticationError(BrokerError):
    """Raised when broker login or access token validation fails."""
    pass


class BrokerConnectionError(BrokerError):
    """Raised when broker REST API or WebSocket connection drops."""
    pass


class OrderExecutionError(BrokerError):
    """Raised when order placement, modification, or cancellation fails."""
    pass


class IndicatorError(QuantSignalException):
    """Raised when technical indicator calculation fails."""
    pass


class ModelError(QuantSignalException):
    """Base exception for Machine Learning model operations."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when requested model file or path does not exist."""
    pass


class ModelPredictionError(ModelError):
    """Raised when ML model scoring/prediction fails."""
    pass


class StrategyError(QuantSignalException):
    """Raised when trading strategy execution or signal generation fails."""
    pass
