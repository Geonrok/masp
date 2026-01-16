"""API key status panel (masked, no secrets)."""
from __future__ import annotations

import streamlit as st


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
