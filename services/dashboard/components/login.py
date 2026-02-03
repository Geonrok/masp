"""Login component for MASP dashboard."""

from __future__ import annotations

import hmac
import os

import streamlit as st

from services.dashboard.utils import auth


def render_login() -> bool:
    """Render login form and return True on successful login."""
    st.subheader("Admin Login")

    expected = os.getenv("MASP_ADMIN_TOKEN", "")
    if not expected:
        st.error(
            "⚠️ MASP_ADMIN_TOKEN is not configured. Please set it before using the dashboard."
        )
        return False

    with st.form("login_form", clear_on_submit=True):
        token = st.text_input(
            "Admin Token",
            type="password",
            help="Enter MASP admin token",
        )
        submitted = st.form_submit_button("Login", use_container_width=True)

    if not submitted:
        return False

    token = token.strip()
    if not token or not hmac.compare_digest(token, expected):
        st.error("Invalid credentials.")
        return False

    auth.set_token(token)
    return True
