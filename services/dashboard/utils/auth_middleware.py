"""Authentication enforcement for Streamlit dashboard.

Supports Google OAuth authentication via streamlit-google-auth.
"""

from __future__ import annotations

import streamlit as st

from services.dashboard.components.firebase_login import (
    render_firebase_login,
    check_firebase_auth,
    clear_firebase_user,
)


def enforce_auth(idle_seconds: int = 1800) -> bool:
    """Render login UI when unauthenticated.

    Args:
        idle_seconds: Session timeout in seconds (not used with OAuth)

    Returns:
        True if authenticated, False otherwise.
    """
    # Check if user is authenticated
    user = check_firebase_auth()

    if user:
        return True

    # Not authenticated - show login
    st.title("MASP 대시보드")
    st.info("계속하려면 Google로 로그인하세요.")

    result = render_firebase_login()

    if result:
        st.rerun()

    return False


def logout() -> None:
    """Clear authentication and logout."""
    clear_firebase_user()
