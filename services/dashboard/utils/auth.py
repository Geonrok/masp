"""Dashboard auth utilities using session state.

Security features:
- Token fingerprint storage (not raw token)
- Session idle timeout
- Token rotation detection
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Optional

import streamlit as st

_SESSION_KEY = "masp_auth_state"


@dataclass(frozen=True)
class AuthState:
    """Session auth state (stores only token fingerprint)."""

    token_fp: str
    last_activity_ts: float


def _fingerprint(token: str) -> str:
    """Return SHA256 fingerprint of token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _expected_fingerprint() -> str:
    """Get expected token fingerprint from environment."""
    expected = os.getenv("MASP_ADMIN_TOKEN", "")
    return _fingerprint(expected) if expected else ""


def _get_state() -> Optional[AuthState]:
    """Return AuthState from session_state if present."""
    state = st.session_state.get(_SESSION_KEY)
    return state if isinstance(state, AuthState) else None


def set_token(token: str) -> None:
    """Store token fingerprint in session state."""
    st.session_state[_SESSION_KEY] = AuthState(
        token_fp=_fingerprint(token),
        last_activity_ts=time.time(),
    )


def clear_token() -> None:
    """Clear auth state from session state."""
    st.session_state.pop(_SESSION_KEY, None)


def touch_activity() -> None:
    """Update last activity timestamp if logged in."""
    state = _get_state()
    if state is None:
        return
    st.session_state[_SESSION_KEY] = AuthState(
        token_fp=state.token_fp,
        last_activity_ts=time.time(),
    )


def is_session_expired(idle_seconds: int = 1800) -> bool:
    """Return True when session idle time exceeds threshold."""
    state = _get_state()
    if state is None:
        return True
    return (time.time() - state.last_activity_ts) > idle_seconds


def require_login(idle_seconds: int = 1800) -> bool:
    """Return True if valid, non-expired session exists.

    Also invalidates session if MASP_ADMIN_TOKEN was rotated.
    """
    expected_fp = _expected_fingerprint()
    if not expected_fp:
        clear_token()
        return False

    state = _get_state()
    if state is None:
        return False

    if is_session_expired(idle_seconds=idle_seconds):
        clear_token()
        return False

    if not hmac.compare_digest(state.token_fp, expected_fp):
        clear_token()
        return False

    return True
