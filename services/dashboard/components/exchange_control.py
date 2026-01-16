"""Exchange quick controls (toggle + position size)."""
from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from services.dashboard.utils.api_client import ConfigApiClient


def get_api_client() -> Optional[ConfigApiClient]:
    """Return an authenticated API client using environment settings."""
    token = os.getenv("MASP_ADMIN_TOKEN")
    if not token:
        st.warning("MASP_ADMIN_TOKEN is not set. Controls are disabled.")
        return None
    return ConfigApiClient()


def render_exchange_toggle(exchange_name: str, current_enabled: bool) -> None:
    """Toggle exchange on/off."""
    api = get_api_client()
    if not api:
        return

    new_state = st.toggle(
        f"{exchange_name.upper()} Trading Enabled",
        value=current_enabled,
        key=f"toggle_{exchange_name}",
        help="Enable or disable trading for this exchange.",
    )

    if new_state != current_enabled:
        with st.spinner("Saving..."):
            success = api.toggle_exchange(exchange_name, new_state)

        if success:
            st.success(
                f"{exchange_name.upper()} set to {'ENABLED' if new_state else 'DISABLED'}."
            )
        else:
            st.error("Update failed. Check the API server logs.")
        st.rerun()


def render_position_size_editor(exchange_name: str, current_size: int) -> None:
    """Edit per-trade position size."""
    api = get_api_client()
    if not api:
        return

    with st.form(f"size_form_{exchange_name}"):
        col1, col2 = st.columns([3, 1])

        with col1:
            new_size = st.number_input(
                "Per-trade Size (KRW)",
                min_value=10000,
                max_value=10000000,
                value=current_size,
                step=10000,
                key=f"size_{exchange_name}",
                help="Minimum 10,000 KRW.",
            )

        with col2:
            submitted = st.form_submit_button("Save", use_container_width=True)

    if submitted:
        if new_size == current_size:
            st.info("No changes to save.")
            return

            with st.spinner("Saving..."):
                success = api.update_exchange_config(
                    exchange_name, {"position_size_krw": int(new_size)}
                )

        if success:
            st.success(f"Saved position size: {int(new_size):,} KRW.")
        else:
            st.error("Save failed. Check the API server logs.")
        st.rerun()


def render_exchange_controls(exchanges: list[str]) -> None:
    """Render quick controls for selected exchange."""
    api = get_api_client()
    if not api:
        st.stop()
        return

    selected = st.selectbox("Exchange", exchanges, key="control_exchange")
    config = api.get_exchange_config(selected)

    if not config:
        st.error(f"Unable to load config for {selected}.")
        return

    st.subheader("Enable/Disable")
    render_exchange_toggle(selected, config.get("enabled", False))

    st.divider()

    st.subheader("Position Size")
    render_position_size_editor(selected, config.get("position_size_krw", 10000))

    with st.expander("Current Config"):
        st.json(config)
