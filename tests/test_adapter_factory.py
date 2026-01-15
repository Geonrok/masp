"""
AdapterFactory tests (Phase 6A Step 2 v3).
"""
import os
import warnings
from unittest.mock import patch

import pytest

from libs.adapters.factory import AdapterFactory


class TestAdapterFactory:
    """AdapterFactory tests."""

    def test_create_upbit_spot_execution_live_mode(self):
        """Live mode returns UpbitSpotExecution."""
        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "1"}):
            from libs.adapters.real_upbit_spot import UpbitSpotExecution

            adapter = AdapterFactory.create_execution(
                "upbit_spot",
                adapter_mode="live",
                access_key="test_key",
                secret_key="test_secret",
            )

            assert isinstance(adapter, UpbitSpotExecution)

    def test_create_upbit_spot_execution_live_guard(self):
        """Live mode enforces live guard."""
        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "0"}):
            with pytest.raises(RuntimeError) as exc_info:
                AdapterFactory.create_execution(
                    "upbit_spot",
                    adapter_mode="live",
                )

            assert "MASP_ENABLE_LIVE_TRADING" in str(exc_info.value)

    def test_deprecated_upbit_alias(self):
        """'upbit' alias emits DeprecationWarning."""
        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "1"}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                adapter = AdapterFactory.create_execution(
                    "upbit",
                    adapter_mode="live",
                    access_key="test_key",
                    secret_key="test_secret",
                )

                assert adapter is not None
                assert len(caught) == 1
                assert issubclass(caught[0].category, DeprecationWarning)
                assert "upbit_spot" in str(caught[0].message)

    def test_create_upbit_spot_execution_paper_mode(self):
        """Paper mode returns PaperExecutionAdapter."""
        adapter = AdapterFactory.create_execution(
            "upbit_spot",
            adapter_mode="paper",
        )

        from libs.adapters.paper_execution import PaperExecutionAdapter
        assert isinstance(adapter, PaperExecutionAdapter)

    def test_unknown_exchange_raises(self):
        """Unknown exchange raises ValueError."""
        with pytest.raises(ValueError):
            AdapterFactory.create_execution("unknown_exchange")

    def test_create_market_data_separate(self):
        """create_market_data and create_execution are separate."""
        market_data = AdapterFactory.create_market_data("upbit_spot")

        from libs.adapters.real_upbit_spot import UpbitSpotMarketData
        assert isinstance(market_data, UpbitSpotMarketData)
