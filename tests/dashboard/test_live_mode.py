"""Tests for dashboard LIVE mode utilities."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestCheckLiveConditions:
    """Tests for check_live_conditions function."""

    def test_live_switch_not_set(self):
        """LIVE switch not set -> demo mode."""
        with patch.dict(os.environ, {}, clear=True):
            from services.dashboard.utils.live_mode import check_live_conditions

            can_live, reason = check_live_conditions("upbit")
            assert can_live is False
            assert "MASP_DASHBOARD_LIVE" in reason

    def test_live_switch_set_no_keys(self):
        """LIVE switch set but no keys -> demo mode."""
        with patch.dict(os.environ, {"MASP_DASHBOARD_LIVE": "1"}, clear=True):
            with patch("services.dashboard.utils.live_mode.KeyManager", None):
                from services.dashboard.utils.live_mode import check_live_conditions

                can_live, reason = check_live_conditions("upbit")
                assert can_live is False
                assert "not configured" in reason.lower()

    def test_live_switch_and_env_keys(self):
        """LIVE switch + env keys -> live mode."""
        env = {
            "MASP_DASHBOARD_LIVE": "1",
            "UPBIT_API_KEY": "test-key",
            "UPBIT_SECRET_KEY": "test-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("services.dashboard.utils.live_mode.KeyManager", None):
                from services.dashboard.utils.live_mode import check_live_conditions

                can_live, _ = check_live_conditions("upbit")
                assert can_live is True

    def test_keymanager_priority(self):
        """KeyManager takes priority over env."""
        mock_km_class = MagicMock()
        mock_km_instance = MagicMock()
        mock_km_instance.get_raw_key.return_value = {
            "api_key": "km-key",
            "secret_key": "km-secret",
        }
        mock_km_class.return_value = mock_km_instance

        env = {"MASP_DASHBOARD_LIVE": "1"}
        with patch.dict(os.environ, env, clear=True):
            with patch("services.dashboard.utils.live_mode.KeyManager", mock_km_class):
                from services.dashboard.utils.live_mode import check_live_conditions

                can_live, _ = check_live_conditions("upbit")
                assert can_live is True
                mock_km_instance.get_raw_key.assert_called_once_with("upbit")
