"""
Settings and Configuration Management for QuantSignal AI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from quant_signal.exceptions import ConfigurationError

# Load environment variables from .env file securely from PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv_path = PROJECT_ROOT / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)


@dataclass(frozen=True)
class AppConfig:
    """Application level configurations."""
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "QuantSignal AI"))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() in ("true", "1", "yes"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_to_file: bool = field(default_factory=lambda: os.getenv("LOG_TO_FILE", "true").lower() in ("true", "1", "yes"))
    log_file_path: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "logs/quantsignal.log"))


@dataclass(frozen=True)
class FyersConfig:
    """Fyers API v3 authentication and connection parameters."""
    client_id: str = field(default_factory=lambda: os.getenv("FYERS_CLIENT_ID", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("FYERS_SECRET_KEY", ""))
    redirect_uri: str = field(default_factory=lambda: os.getenv("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/default.ui.html"))
    access_token: str = field(default_factory=lambda: os.getenv("FYERS_ACCESS_TOKEN", ""))
    refresh_token: str = field(default_factory=lambda: os.getenv("FYERS_REFRESH_TOKEN", ""))
    pin: str = field(default_factory=lambda: os.getenv("FYERS_PIN", ""))
    totp_key: str = field(default_factory=lambda: os.getenv("FYERS_TOTP_KEY", ""))
    token_file_path: str = field(default_factory=lambda: os.getenv("FYERS_TOKEN_FILE_PATH", "saved_tokens/fyers_token.json"))

    def validate(self) -> None:
        """Validates critical Fyers credentials."""
        if not self.client_id:
            raise ConfigurationError("FYERS_CLIENT_ID environment variable is missing.")
        if not self.secret_key:
            raise ConfigurationError("FYERS_SECRET_KEY environment variable is missing.")


@dataclass(frozen=True)
class TradingConfig:
    """Trading strategy and risk parameters."""
    default_timeframe: str = field(default_factory=lambda: os.getenv("DEFAULT_TIMEFRAME", "5m"))
    risk_per_trade_percent: float = field(
        default_factory=lambda: float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0"))
    )
    max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    )


@dataclass(frozen=True)
class ModelConfig:
    """Machine Learning model management settings."""
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "saved_models"))
    default_model_name: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL_NAME", "quant_signal_xgb_v1.joblib")
    )


@dataclass(frozen=True)
class Settings:
    """Master Application Settings container."""
    app: AppConfig = field(default_factory=AppConfig)
    fyers: FyersConfig = field(default_factory=FyersConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def is_demo_mode(self) -> bool:
        """
        Returns True when no live broker credentials are configured.

        Demo mode is active when FYERS_CLIENT_ID or FYERS_SECRET_KEY are
        absent from the environment. In demo mode the application uses
        DemoMarketDataProvider and clearly labels data as NOT live.
        """
        return not (self.fyers.client_id and self.fyers.secret_key)


# Global singleton settings instance
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Retrieves or initializes global application settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
