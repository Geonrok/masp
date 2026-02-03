"""
Test suite for MultiExchangeScheduler._get_heartbeat_interval().
Phase 3B: Heartbeat environment variable validation.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from services.multi_exchange_scheduler import MultiExchangeScheduler


@pytest.fixture
def scheduler():
    """Mock scheduler instance for testing."""
    scheduler = MagicMock(spec=MultiExchangeScheduler)
    scheduler._get_heartbeat_interval = (
        MultiExchangeScheduler._get_heartbeat_interval.__get__(
            scheduler, MultiExchangeScheduler
        )
    )
    return scheduler


class TestGetHeartbeatInterval:
    """_get_heartbeat_interval() unit tests (6 cases)."""

    def test_default_when_not_set(self, scheduler):
        """TC-HB-001: env not set -> default 30s."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MASP_HEARTBEAT_SEC", None)
            result = scheduler._get_heartbeat_interval()
            assert result == 30

    def test_valid_value_30(self, scheduler):
        """TC-HB-002: valid value 30 -> 30."""
        with patch.dict(os.environ, {"MASP_HEARTBEAT_SEC": "30"}):
            result = scheduler._get_heartbeat_interval()
            assert result == 30

    def test_valid_value_60(self, scheduler):
        """TC-HB-003: valid value 60 -> 60."""
        with patch.dict(os.environ, {"MASP_HEARTBEAT_SEC": "60"}):
            result = scheduler._get_heartbeat_interval()
            assert result == 60

    def test_below_minimum_clamps_to_5(self, scheduler):
        """TC-HB-004: below min (1) -> clamp to 5."""
        with patch.dict(os.environ, {"MASP_HEARTBEAT_SEC": "1"}):
            result = scheduler._get_heartbeat_interval()
            assert result == 5

    def test_above_maximum_clamps_to_300(self, scheduler):
        """TC-HB-005: above max (600) -> clamp to 300."""
        with patch.dict(os.environ, {"MASP_HEARTBEAT_SEC": "600"}):
            result = scheduler._get_heartbeat_interval()
            assert result == 300

    def test_invalid_string_returns_default(self, scheduler):
        """TC-HB-006: invalid string -> default 30."""
        with patch.dict(os.environ, {"MASP_HEARTBEAT_SEC": "invalid"}):
            result = scheduler._get_heartbeat_interval()
            assert result == 30
