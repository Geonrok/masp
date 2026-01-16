"""Exchange control panel - toggle and position size editing."""
from __future__ import annotations

import os

import streamlit as st

from services.dashboard.utils.api_client import ConfigApiClient


def get_api_client():
    """Return API client if token is configured."""
    if not os.getenv("MASP_ADMIN_TOKEN"):
        return None
    return ConfigApiClient()


def render_exchange_toggle(exchange_name: str, current_state: bool) -> None:
    """Render on/off toggle for an exchange."""
    api = get_api_client()
    if not api:
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{exchange_name.upper()}**")
    with col2:
        new_state = st.toggle(
            "Enabled",
            value=current_state,
            key=f"toggle_{exchange_name}",
            label_visibility="collapsed",
        )

    if new_state != current_state:
        with st.spinner("Updating..."):
            success = api.toggle_exchange(exchange_name, new_state)
        if success:
            st.success(f"{exchange_name.upper()} {'enabled' if new_state else 'disabled'}.")
            st.rerun()
        else:
            st.error("Update failed.")


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

    if not submitted:
        return

    new_size_int = int(new_size)
    if new_size_int == int(current_size):
        st.info("No changes to save.")
        return

    with st.spinner("Saving..."):
        success = api.update_exchange_config(
            exchange_name, {"position_size_krw": new_size_int}
        )

    if success:
        st.success(f"Saved position size: {new_size_int:,} KRW.")
    else:
        st.error("Save failed. Check the API server logs.")
    st.rerun()


def render_exchange_controls(exchanges: list[str]) -> None:
    """Render controls for all exchanges."""
    api = get_api_client()
    if not api:
        st.warning("MASP_ADMIN_TOKEN is required for Quick Controls.")
        return

    for exchange in exchanges:
        config = api.get_exchange_config(exchange)
        if not config:
            st.warning(f"Could not load config for {exchange}.")
            continue

        with st.expander(f"{exchange.upper()} Controls", expanded=False):
            render_exchange_toggle(exchange, config.get("enabled", False))
            st.divider()
            render_position_size_editor(
                exchange, config.get("position_size_krw", 10000)
            )
