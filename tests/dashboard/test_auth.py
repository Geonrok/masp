"""Tests for dashboard auth utilities."""

from __future__ import annotations

from unittest.mock import patch

import streamlit as st

from services.dashboard.utils import auth


def setup_function():
    st.session_state.clear()


def test_set_and_require_login(monkeypatch):
    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token")
    auth.set_token("token")
    assert auth.require_login() is True


def test_invalid_token_fails(monkeypatch):
    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token")
    auth.set_token("wrong")
    assert auth.require_login() is False


def test_clear_token(monkeypatch):
    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token")
    auth.set_token("token")
    auth.clear_token()
    assert auth.require_login() is False


def test_session_expired_by_time(monkeypatch):
    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token")
    with patch("services.dashboard.utils.auth.time.time", return_value=1000.0):
        auth.set_token("token")
    with patch("services.dashboard.utils.auth.time.time", return_value=1000.0 + 1801):
        assert auth.is_session_expired() is True
        assert auth.require_login() is False


def test_session_not_expired(monkeypatch):
    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token")
    with patch("services.dashboard.utils.auth.time.time", return_value=1000.0):
        auth.set_token("token")
    with patch("services.dashboard.utils.auth.time.time", return_value=1000.0 + 60):
        assert auth.is_session_expired() is False
        assert auth.require_login() is True


def test_touch_activity_updates_timestamp(monkeypatch):
    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token")
    with patch("services.dashboard.utils.auth.time.time", return_value=1000.0):
        auth.set_token("token")
    with patch("services.dashboard.utils.auth.time.time", return_value=1100.0):
        auth.touch_activity()
    with patch("services.dashboard.utils.auth.time.time", return_value=1100.0 + 1799):
        assert auth.require_login() is True


def test_token_rotation_invalidates_session(monkeypatch):
    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token1")
    auth.set_token("token1")
    assert auth.require_login() is True

    monkeypatch.setenv("MASP_ADMIN_TOKEN", "token2")
    assert auth.require_login() is False


def test_missing_env_var_fails(monkeypatch):
    monkeypatch.delenv("MASP_ADMIN_TOKEN", raising=False)
    auth.set_token("token")
    assert auth.require_login() is False
