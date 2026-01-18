"""Exchange enable/disable status panel."""
from __future__ import annotations

import math
import os
import time

import streamlit as st


def _get_refresh_interval() -> float:
    """Safely get auto refresh interval from environment."""
    raw = os.getenv("MASP_AUTO_REFRESH_INTERVAL", "10.0")
    try:
        v = float(raw)
        if not math.isfinite(v) or v <= 0:
            return 10.0
        return v
    except ValueError:
        return 10.0


_AUTO_REFRESH_INTERVAL = _get_refresh_interval()


class ExchangeStatusPanel:
    EXCHANGES = ["upbit", "bithumb", "binance", "binance_futures"]
    _LAST_RERUN_TS_KEY = "masp_last_auto_refresh_rerun_ts"

    def __init__(self, api_client) -> None:
        self.api = api_client

    def render(self) -> None:
        st.subheader("Exchange Status")

        auto_refresh = st.checkbox(
            f"Auto Refresh ({_AUTO_REFRESH_INTERVAL:.0f}s)",
            value=False,
            key="auto_refresh_enabled",
        )
        st.session_state["masp_auto_refresh"] = bool(auto_refresh)

        if auto_refresh:
            now = time.time()
            last = float(st.session_state.get(self._LAST_RERUN_TS_KEY, 0.0))

            if (now - last) >= _AUTO_REFRESH_INTERVAL:
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
