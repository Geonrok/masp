"""API Key status panel for MASP Dashboard."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import streamlit as st


def _check_key_status(exchange: str) -> Tuple[bool, bool, str]:
    """
    Check API key status from KeyManager or environment variables.

    Returns:
        (has_api_key, has_secret_key, source)
    """
    try:
        from libs.core.key_manager import KeyManager

        km = KeyManager()
        raw = km.get_raw_key(exchange)
        if raw and raw.get("api_key") and raw.get("secret_key"):
            return True, True, "KeyManager"
    except Exception:
        pass

    env_api = os.getenv(f"{exchange.upper()}_API_KEY")
    env_secret = os.getenv(f"{exchange.upper()}_SECRET_KEY")
    if env_api and env_secret:
        return True, True, "Environment"
    if env_api or env_secret:
        return bool(env_api), bool(env_secret), "Environment"

    return False, False, "Not configured"


def _get_masked_key(exchange: str, key_type: str) -> Optional[str]:
    """Get masked key value (first 4 + ... + last 4) from environment."""
    env_key = f"{exchange.upper()}_{key_type.upper()}"
    value = os.getenv(env_key)
    if value and len(value) > 8:
        return f"{value[:4]}...{value[-4:]}"
    if value:
        return "****"
    return None


def _is_live_mode_enabled() -> bool:
    """Check if LIVE mode is explicitly enabled."""
    return os.getenv("MASP_DASHBOARD_LIVE", "").strip() == "1"


def render_api_key_status_panel() -> None:
    """Render API key status panel."""
    st.subheader("API Key Status")

    live_enabled = _is_live_mode_enabled()
    if live_enabled:
        st.success("LIVE Mode: Enabled (MASP_DASHBOARD_LIVE=1)")
    else:
        st.info("DEMO Mode: Set MASP_DASHBOARD_LIVE=1 to enable LIVE")

    st.divider()

    show_masked = st.checkbox(
        "Show masked key values",
        value=False,
        key="apikey_show_masked",
        help="Show first 4 and last 4 characters of keys",
    )

    exchanges = ["upbit", "bithumb"]

    for exchange in exchanges:
        has_api, has_secret, source = _check_key_status(exchange)

        with st.expander(f"{exchange.upper()}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                if has_api and has_secret:
                    st.success(f"Configured via {source}")
                elif has_api or has_secret:
                    st.warning("Partially configured")
                else:
                    st.error("Not configured")

            with col2:
                can_live = has_api and has_secret and live_enabled
                if can_live:
                    st.success("LIVE Ready")
                else:
                    st.info("DEMO Only")

            if show_masked and source == "Environment":
                api_masked = _get_masked_key(exchange, "API_KEY")
                secret_masked = _get_masked_key(exchange, "SECRET_KEY")
                if api_masked:
                    st.text(f"API Key: {api_masked}")
                if secret_masked:
                    st.text(f"Secret:  {secret_masked}")
            elif show_masked and source == "KeyManager":
                st.text("API Key: ******** (encrypted)")
                st.text("Secret:  ******** (encrypted)")

    with st.expander("Setup Guide"):
        st.markdown(
            """
**Quick Setup (Development):**
```powershell
$env:UPBIT_API_KEY = "your-key"
$env:UPBIT_SECRET_KEY = "your-secret"
$env:MASP_DASHBOARD_LIVE = "1"
```

See `docs/API_KEY_SETUP.md` for production setup.
"""
        )
