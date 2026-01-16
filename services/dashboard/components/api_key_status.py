"""API key status panel (masked, no secrets)."""
from __future__ import annotations

import os

import streamlit as st

from services.dashboard.utils.api_client import ConfigApiClient


def mask_api_key(key: str) -> str:
    """Mask API key for display."""
    if not key or len(key) < 10:
        return "Not configured"
    return f"{key[:6]}...{key[-4:]}"


def render_api_key_status_panel() -> None:
    """API key status display (read-only)."""
    st.subheader("API Key Status")
    st.info(
        "API key management is read-only here. "
        "Use backend Key API or environment variables for updates."
    )

    if not os.getenv("MASP_ADMIN_TOKEN"):
        st.warning("MASP_ADMIN_TOKEN is not set.")
        return

    api = ConfigApiClient()
    keys = api.get_all_keys()

    if not keys:
        st.info("No API keys registered.")
        return

    for exchange, key_info in keys.items():
        with st.expander(exchange.upper()):
            raw_key = key_info.get("api_key", "")
            st.text(f"API Key: {mask_api_key(raw_key)}")
            has_secret = bool(key_info.get("has_secret", False))
            st.text(f"Secret set: {'Yes' if has_secret else 'No'}")
            st.caption("Secret values are not displayed.")
