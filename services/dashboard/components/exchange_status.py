"""Exchange enable/disable status panel."""
from __future__ import annotations

import time

import streamlit as st


class ExchangeStatusPanel:
    EXCHANGES = ["upbit", "bithumb", "binance", "binance_futures"]
    _LAST_RERUN_TS_KEY = "masp_last_auto_refresh_rerun_ts"

    def __init__(self, api_client) -> None:
        self.api = api_client

    def render(self) -> None:
        st.subheader("Exchange Status")

        auto_refresh = st.checkbox(
            " Auto Refresh (10s)", value=False, key="auto_refresh_enabled"
        )
        st.session_state["masp_auto_refresh"] = bool(auto_refresh)

        if auto_refresh:
            now = time.time()
            last = float(st.session_state.get(self._LAST_RERUN_TS_KEY, 0.0))

            if (now - last) >= 10.0:
                st.session_state[self._LAST_RERUN_TS_KEY] = now
                st.rerun()

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
