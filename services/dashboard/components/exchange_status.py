"""Exchange enable/disable status panel."""
from __future__ import annotations

import streamlit as st


class ExchangeStatusPanel:
    EXCHANGES = ["upbit", "bithumb", "binance", "binance_futures"]

    def __init__(self, api_client) -> None:
        self.api = api_client

    def render(self) -> None:
        st.subheader("Exchange Status")

        for exchange in self.EXCHANGES:
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.text(exchange.upper())

            with col2:
                config = self.api.get_exchange_config(exchange)
                if config is None:
                    status = "ERROR"
                    enabled = False
                else:
                    enabled = config.get("enabled", False)
                    status = "ENABLED" if enabled else "DISABLED"
                st.text(status)

            with col3:
                if config is not None:
                    btn_label = "Disable" if enabled else "Enable"
                    if st.button(btn_label, key=f"status_toggle_{exchange}"):
                        success = self.api.toggle_exchange(exchange, not enabled)
                        if success:
                            st.rerun()
                        else:
                            st.error(
                                f"Failed to toggle {exchange}. Check API server."
                            )
