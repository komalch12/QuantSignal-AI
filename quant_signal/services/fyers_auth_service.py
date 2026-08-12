"""
Fyers API v3 Authentication Service for QuantSignal AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from enum import Enum
from dotenv import set_key

from quant_signal.config.settings import FyersConfig
from quant_signal.exceptions import (
    BrokerAuthenticationError,
    BrokerConnectionError,
    ConfigurationError,
)
from quant_signal.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# Try importing fyers_apiv3 library with graceful fallback stub handling
try:
    from fyers_apiv3 import fyersModel
    HAS_FYERS_LIB = True
except ImportError:
    HAS_FYERS_LIB = False
    fyersModel = None

class ConnectionState(str, Enum):
    AUTH_NOT_CONFIGURED = "AUTH_NOT_CONFIGURED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTHENTICATED = "AUTHENTICATED"
    FEED_CONNECTING = "FEED_CONNECTING"
    FEED_CONNECTED = "FEED_CONNECTED"
    FEED_DISCONNECTED = "FEED_DISCONNECTED"
    MARKET_CLOSED = "MARKET_CLOSED"
    AUTH_ERROR = "AUTH_ERROR"


@dataclass
class FyersConnectionStatus:
    """Dataclass representing live Fyers API connection state."""
    state: ConnectionState = ConnectionState.AUTH_NOT_CONFIGURED
    is_authenticated: bool = False
    client_id: str = ""
    user_name: str = "N/A"
    fy_id: str = "N/A"
    email: str = "N/A"
    last_validated: str | None = None
    token_source: str = "None"
    error_message: str | None = None


class FyersAuthService:
    """Production Authentication Service for Fyers API v3."""

    def __init__(self, config: FyersConfig, env_file_path: str = ".env") -> None:
        self.config: FyersConfig = config
        self.env_file_path: Path = Path(env_file_path)
        self.token_file_path: Path = Path(config.token_file_path)
        
        self.access_token: str = config.access_token
        self.refresh_token: str = config.refresh_token
        self._fyers_instance: Any | None = None
        self._status: FyersConnectionStatus = FyersConnectionStatus(client_id=config.client_id)

    def get_status(self) -> FyersConnectionStatus:
        return self._status

    def set_state(self, state: ConnectionState) -> None:
        self._status.state = state

    @log_execution_time()
    def generate_auth_url(self) -> str:
        self.config.validate()
        
        if not HAS_FYERS_LIB:
            return f"https://api-t1.fyers.in/api/v3/generate-authcode?client_id={self.config.client_id}&redirect_uri={self.config.redirect_uri}&response_type=code&state=sample_state"

        try:
            session = fyersModel.SessionModel(
                client_id=self.config.client_id,
                secret_key=self.config.secret_key,
                redirect_uri=self.config.redirect_uri,
                response_type="code",
                grant_type="authorization_code",
            )
            return session.generate_authcode()
        except Exception as err:
            raise BrokerAuthenticationError(f"Auth URL generation failed: {err}") from err

    @log_execution_time()
    def generate_token_from_auth_code(self, auth_code: str) -> dict[str, Any]:
        if not auth_code:
            raise BrokerAuthenticationError("Auth code cannot be empty.")

        self.config.validate()

        if not HAS_FYERS_LIB:
            dummy_access = f"dummy_access_{auth_code[:8]}"
            dummy_refresh = f"dummy_refresh_{auth_code[:8]}"
            self._save_tokens(dummy_access, dummy_refresh, source="auth_code")
            return {"access_token": dummy_access, "refresh_token": dummy_refresh, "s": "ok"}

        try:
            session = fyersModel.SessionModel(
                client_id=self.config.client_id,
                secret_key=self.config.secret_key,
                redirect_uri=self.config.redirect_uri,
                response_type="code",
                grant_type="authorization_code",
            )
            session.set_token(auth_code)
            response = session.generate_token()

            if isinstance(response, dict) and response.get("s") == "ok":
                self._save_tokens(response.get("access_token", ""), response.get("refresh_token", ""), source="auth_code")
                return response
            else:
                raise BrokerAuthenticationError(f"Fyers token exchange failed: {response}")

        except Exception as err:
            raise BrokerAuthenticationError(f"Token generation error: {err}") from err

    @log_execution_time()
    def refresh_access_token(self) -> dict[str, Any]:
        target_refresh = self.refresh_token or self._load_cached_tokens().get("refresh_token", "")
        if not target_refresh:
            raise BrokerAuthenticationError("No refresh token available. Manual login required.")

        self.config.validate()

        if not HAS_FYERS_LIB:
            return {"access_token": "dummy", "s": "ok"}

        try:
            session = fyersModel.SessionModel(
                client_id=self.config.client_id,
                secret_key=self.config.secret_key,
                redirect_uri=self.config.redirect_uri,
                response_type="code",
                grant_type="refresh_token",
            )
            session.set_token(target_refresh)
            response = session.generate_token()

            if isinstance(response, dict) and response.get("s") == "ok":
                self._save_tokens(response.get("access_token", ""), response.get("refresh_token", target_refresh), source="refresh_token")
                return response
            else:
                raise BrokerAuthenticationError(f"Refresh token failed: {response}")

        except Exception as err:
            raise BrokerAuthenticationError(f"Failed to refresh access token: {err}") from err

    @log_execution_time()
    def validate_access_token(self, token_to_test: str | None = None) -> str:
        token = token_to_test or self.access_token
        if not token:
            self._status.state = ConnectionState.AUTH_REQUIRED
            self._status.is_authenticated = False
            self._status.error_message = "Access token is missing."
            return "MISSING"

        if not HAS_FYERS_LIB:
            self._status.state = ConnectionState.AUTHENTICATED
            self._status.is_authenticated = True
            self._status.user_name = "Demo Quant Trader"
            self._status.fy_id = "DEMO123"
            return "VALID"

        try:
            fyers_client = fyersModel.FyersModel(
                client_id=self.config.client_id,
                token=token,
                is_async=False,
                log_path=str(Path("logs").absolute()),
            )
            profile = fyers_client.get_profile()

            if isinstance(profile, dict) and profile.get("s") == "ok":
                data = profile.get("data", {})
                self._fyers_instance = fyers_client
                self.access_token = token
                self._status.state = ConnectionState.AUTHENTICATED
                self._status.is_authenticated = True
                self._status.user_name = data.get("name", "N/A")
                self._status.fy_id = data.get("fy_id", "N/A")
                self._status.email = data.get("email_id", "N/A")
                self._status.last_validated = datetime.now().isoformat()
                self._status.error_message = None
                return "VALID"
            else:
                self._status.state = ConnectionState.AUTH_ERROR
                self._status.is_authenticated = False
                self._status.error_message = profile.get("message", "Validation failed")
                return "INVALID"

        except Exception as err:
            self._status.state = ConnectionState.AUTH_ERROR
            self._status.is_authenticated = False
            self._status.error_message = str(err)
            return "INVALID"

    @log_execution_time()
    def login(self) -> bool:
        if not self.config.client_id or not self.config.secret_key:
            self._status.state = ConnectionState.AUTH_NOT_CONFIGURED
            self._status.error_message = "Client ID or Secret Key missing in configuration."
            return False

        cached_tokens = self._load_cached_tokens()
        candidate_access = self.access_token or cached_tokens.get("access_token", "")
        if candidate_access and self.validate_access_token(candidate_access) == "VALID":
            return True

        candidate_refresh = self.refresh_token or cached_tokens.get("refresh_token", "")
        if candidate_refresh:
            try:
                if self.refresh_access_token().get("access_token") and self.validate_access_token() == "VALID":
                    return True
            except BrokerAuthenticationError:
                pass

        self._status.state = ConnectionState.AUTH_REQUIRED
        self._status.is_authenticated = False
        return False

    def get_fyers_client(self) -> Any | None:
        return self._fyers_instance

    def _save_tokens(self, access_token: str, refresh_token: str, source: str = "unknown") -> None:
        self.access_token = access_token
        if refresh_token:
            self.refresh_token = refresh_token
        self._status.token_source = source
        self._status.state = ConnectionState.AUTHENTICATED
        self._status.is_authenticated = True
        self._status.error_message = None

        try:
            self.token_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "access_token": self.access_token,
                    "refresh_token": self.refresh_token,
                    "client_id": self.config.client_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": source,
                }, f, indent=2)
        except Exception:
            pass

        try:
            if not self.env_file_path.exists():
                self.env_file_path.touch()
            set_key(str(self.env_file_path), "FYERS_ACCESS_TOKEN", self.access_token)
            if self.refresh_token:
                set_key(str(self.env_file_path), "FYERS_REFRESH_TOKEN", self.refresh_token)
        except Exception as e:
            logger.error(f"Failed to update .env: {e}")

    def _load_cached_tokens(self) -> dict[str, str]:
        if not self.token_file_path.exists():
            return {}
        try:
            with open(self.token_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "access_token": data.get("access_token", ""),
                    "refresh_token": data.get("refresh_token", ""),
                }
        except Exception:
            return {}
