"""Authentication enforcement for Streamlit dashboard."""
from __future__ import annotations

import os

import streamlit as st

from services.dashboard.components.login import render_login
from services.dashboard.utils import auth


def enforce_auth(idle_seconds: int = 1800) -> bool:
    """Render login UI when unauthenticated or expired."""
    if not os.getenv("MASP_ADMIN_TOKEN"):
        st.title("MASP Dashboard")
        st.error("⚠️ MASP_ADMIN_TOKEN is not configured. Set it before using the dashboard.")
        return False

    if auth.require_login(idle_seconds=idle_seconds):
        auth.touch_activity()
        return True

    st.title("MASP Dashboard")
    st.info("Authentication required.")
    if render_login():
        st.success("Login successful.")
        st.rerun()
    return False
