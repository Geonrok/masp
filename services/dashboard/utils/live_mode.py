"""Common LIVE mode condition checking."""
from __future__ import annotations

import os
from typing import Tuple


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
    try:
        from libs.core.key_manager import KeyManager

        km = KeyManager()
        raw = km.get_raw_key(exchange)
        has_keys = bool(raw and raw.get("api_key") and raw.get("secret_key"))
    except Exception:
        pass

    if not has_keys:
        api_key = os.getenv(f"{exchange.upper()}_API_KEY")
        secret_key = os.getenv(f"{exchange.upper()}_SECRET_KEY")
        has_keys = bool(api_key and secret_key)

    if not has_keys:
        return False, f"API keys not configured for {exchange}"

    return True, "All conditions met"
