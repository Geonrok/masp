"""Google Sign-In component for Streamlit.

Simple OAuth implementation using google-auth-oauthlib.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Path to Google OAuth credentials JSON file
CREDENTIALS_FILE = PROJECT_ROOT / "client_secret_67701556084-ppp3lb0g776m823a8jpml9os5k9vb4t7.apps.googleusercontent.com.json"

# OAuth scopes
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

REDIRECT_URI = "http://localhost:8501"


def _get_auth_url() -> str:
    """Generate Google OAuth authorization URL."""
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_FILE),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = state
    return auth_url


def _exchange_code(code: str) -> dict | None:
    """Exchange authorization code for user info."""
    try:
        flow = Flow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI,
            state=st.session_state.get("oauth_state"),
        )
        flow.fetch_token(code=code)
        credentials = flow.credentials

        # Get user info
        service = build("oauth2", "v2", credentials=credentials)
        user_info = service.userinfo().get().execute()

        return {
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture"),
            "id": user_info.get("id"),
        }
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None


def render_firebase_login() -> dict | None:
    """Render Google Sign-In component.

    Returns:
        dict with user info on successful login, None otherwise.
    """
    # Check if already logged in
    if st.session_state.get("user_info"):
        return st.session_state["user_info"]

    # Check if credentials file exists
    if not CREDENTIALS_FILE.exists():
        st.error(f"OAuth credentials file not found")
        return None

    # Check for OAuth callback (authorization code in URL)
    query_params = st.query_params
    code = query_params.get("code")

    if code:
        # Exchange code for tokens
        user_info = _exchange_code(code)
        if user_info:
            st.session_state["user_info"] = user_info
            st.session_state["connected"] = True
            # Clear URL parameters
            st.query_params.clear()
            st.rerun()
        return None

    # Show login button
    try:
        auth_url = _get_auth_url()

        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 40px 0;">
            <a href="{auth_url}" target="_self" style="
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 24px;
                border: 1px solid #dadce0;
                border-radius: 4px;
                background: white;
                text-decoration: none;
                font-size: 14px;
                font-weight: 500;
                color: #3c4043;
                transition: background 0.2s, box-shadow 0.2s;
            ">
                <svg width="18" height="18" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Sign in with Google
            </a>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

        # Fallback: Development mode
        st.divider()
        st.caption("Development Mode")
        if st.button("Continue as Developer", key="dev_login"):
            st.session_state["user_info"] = {
                "email": "developer@localhost",
                "name": "Developer",
                "picture": "",
            }
            st.session_state["connected"] = True
            st.rerun()

    return None


def check_firebase_auth() -> dict | None:
    """Check if user is authenticated."""
    if st.session_state.get("connected"):
        return st.session_state.get("user_info")
    return None


def clear_firebase_user() -> None:
    """Clear authentication."""
    st.session_state.pop("user_info", None)
    st.session_state.pop("connected", None)
    st.session_state.pop("oauth_state", None)
