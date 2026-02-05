"""Common LIVE mode condition checking."""

from __future__ import annotations

import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    from libs.core.key_manager import KeyManager
except Exception:
    KeyManager = None  # type: ignore


def check_live_conditions(exchange: str) -> Tuple[bool, str]:
    """
    Check if LIVE mode conditions are met.

    Returns:
        (can_live, reason)
    """
    live_switch = os.getenv("MASP_DASHBOARD_LIVE", "").strip() == "1"
    if not live_switch:
        return False, "MASP_DASHBOARD_LIVE not set to '1'"

    has_keys = False

    if KeyManager is not None:
        try:
            km = KeyManager()
            raw = km.get_raw_key(exchange)
            has_keys = bool(raw and raw.get("api_key") and raw.get("secret_key"))
        except Exception as exc:
            logger.debug("KeyManager check failed: %s", type(exc).__name__)

    if not has_keys:
        # Check standard naming convention
        api_key = os.getenv(f"{exchange.upper()}_API_KEY")
        secret_key = os.getenv(f"{exchange.upper()}_SECRET_KEY")
        has_keys = bool(api_key and secret_key)

    if not has_keys and exchange.lower() == "upbit":
        # Upbit uses ACCESS_KEY naming convention
        api_key = os.getenv("UPBIT_ACCESS_KEY")
        secret_key = os.getenv("UPBIT_SECRET_KEY")
        has_keys = bool(api_key and secret_key)

    if not has_keys and exchange.lower() == "binance":
        # Binance uses API_SECRET naming convention
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_API_SECRET")
        has_keys = bool(api_key and secret_key)

    if not has_keys:
        return False, f"API keys not configured for {exchange}"

    return True, "All conditions met"
