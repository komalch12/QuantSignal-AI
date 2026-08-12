"""
Root Config Module Exposer for QuantSignal AI.

Exposes global configuration settings instance and settings loader.
"""

from quant_signal.config.settings import Settings, get_settings

settings: Settings = get_settings()

__all__ = ["settings", "get_settings", "Settings"]
