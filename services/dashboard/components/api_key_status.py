"""API key status panel (masked, no secrets)."""
from __future__ import annotations

import os

import streamlit as st

from services.dashboard.utils.api_client import ConfigApiClient


class ApiKeyStatusPanel:
    def __init__(self, api_client) -> None:
        self.api = api_client

    def render(self) -> None:
        st.subheader("API Key Status")
        keys = self.api.get_all_keys()

        if not keys:
            st.info("No API keys registered.")
            return

        for exchange, key_info in keys.items():
            with st.expander(exchange.upper()):
                st.text(f"API Key: {key_info.get('api_key', 'N/A')}")
                has_secret = bool(key_info.get("has_secret", False))
                st.text(f"Secret set: {'Yes' if has_secret else 'No'}")
                st.caption("Secret values are not displayed.")


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
            st.text(f"API Key: {key_info.get('api_key', 'N/A')}")
            has_secret = bool(key_info.get("has_secret", False))
            st.text(f"Secret set: {'Yes' if has_secret else 'No'}")
            st.caption("Secret values are not displayed.")
