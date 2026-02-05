"""Strategy indicator display - read only."""

from __future__ import annotations

import os

import streamlit as st

from services.dashboard.constants import SIGNAL_PREVIEW_EXCHANGES
from services.dashboard.utils.api_client import ConfigApiClient


def get_api_client():
    """Return API client if token is configured."""
    if not os.getenv("MASP_ADMIN_TOKEN"):
        return None
    return ConfigApiClient()


def render_strategy_indicators() -> None:
    """Render read-only strategy indicators."""
    api = get_api_client()
    if not api:
        st.warning("MASP_ADMIN_TOKEN is required.")
        return

    exchanges = SIGNAL_PREVIEW_EXCHANGES[:2]  # upbit, bithumb only
    selected = st.selectbox("Select Exchange", exchanges, key="strategy_exchange")

    config = api.get_exchange_config(selected)
    if not config:
        st.error(f"Unable to load config for {selected}.")
        return

    st.subheader(f"{selected.upper()} Strategy Configuration")

    current_strategy = config.get("strategy", "Sept-v3-RSI50-Gate")
    st.info(f"Active Strategy: {current_strategy}")

    st.divider()
    st.subheader("Strategy Parameters (Read Only)")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Position Size", f"{config.get('position_size_krw', 10000):,} KRW")
        st.metric("Enabled", "Yes" if config.get("enabled") else "No")

    with col2:
        schedule = config.get("schedule", {})
        st.metric(
            "Schedule Time",
            f"{schedule.get('hour', 9):02d}:{schedule.get('minute', 0):02d}",
        )
        st.metric("Timezone", schedule.get("timezone", "Asia/Seoul"))

    st.divider()
    st.warning(
        "Parameter editing is disabled. Use Quick Controls for basic settings, "
        "or edit config/schedule_config.json for advanced options."
    )

    with st.expander("Full Configuration"):
        st.json(config)
