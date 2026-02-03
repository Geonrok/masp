"""Strategy configuration panel."""

from __future__ import annotations

import streamlit as st


class StrategyConfigPanel:
    def __init__(self, api_client) -> None:
        self.api = api_client

    def render(self) -> None:
        st.subheader("Strategy Configuration")

        exchanges = ["upbit", "bithumb", "binance", "binance_futures"]
        selected = st.selectbox("Exchange", exchanges)

        config = self.api.get_exchange_config(selected)
        if not config:
            st.error(
                f"Unable to load config for {selected}. Check the API server status."
            )
            return

        with st.form(f"config_form_{selected}"):
            position_size = st.number_input(
                "Position Size (KRW)",
                value=config.get("position_size_krw", 100000),
                min_value=10000,
                step=10000,
            )

            schedule_hour = st.slider(
                "Run Hour",
                min_value=0,
                max_value=23,
                value=config.get("schedule", {}).get("hour", 9),
            )

            schedule_minute = st.slider(
                "Run Minute",
                min_value=0,
                max_value=59,
                value=config.get("schedule", {}).get("minute", 0),
            )

            if st.form_submit_button("Save"):
                updates = {
                    "position_size_krw": position_size,
                    "schedule": {"hour": schedule_hour, "minute": schedule_minute},
                }

                if self.api.update_exchange_config(selected, updates):
                    st.success("Configuration saved.")
                    st.rerun()
                else:
                    st.error("Save failed. Check API logs for details.")
