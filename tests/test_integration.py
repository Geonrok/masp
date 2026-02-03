"""
Phase 4C Integration Tests.
Contract-level integration for StrategyEngine + ATLASFuturesStrategy + BinanceFuturesAdapter.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libs.engine.strategy_engine import StrategyEngine
from libs.exchanges.binance_futures import (
    BinanceFuturesAdapter,
    BinanceFuturesConfig,
    OrderResult,
)
from libs.strategies.atlas_futures import (
    ATLASFuturesStrategy,
    Position,
    Signal,
    SignalType,
)
from libs.strategies.loader import get_strategy, load_strategy_class

# ============== Fixtures ==============


def _generate_ohlcv_data(n: int = 300) -> list[dict]:
    """Generate sample OHLCV data with fixed seed."""
    rng = np.random.default_rng(42)
    data = []
    price = 50000.0
    base_ts = 1700000000000

    for i in range(n):
        price += float(rng.normal(0, 100))
        data.append(
            {
                "timestamp": base_ts + i * 14400000,
                "open": price + float(rng.normal(0, 20)),
                "high": price + abs(float(rng.normal(0, 50))),
                "low": price - abs(float(rng.normal(0, 50))),
                "close": price,
                "volume": abs(float(rng.normal(0, 1_000_000))) + 500_000,
            }
        )
    return data


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Mocked Binance adapter for testing."""
    adapter = MagicMock(spec=BinanceFuturesAdapter)
    adapter.config = BinanceFuturesConfig(
        api_key="test",
        secret_key="test",
        testnet=True,
    )
    adapter.initialize = AsyncMock()
    adapter.close = AsyncMock()
    adapter.fetch_ohlcv = AsyncMock(return_value=_generate_ohlcv_data())
    adapter.get_balance = AsyncMock(
        return_value={"total": 10000, "free": 10000, "used": 0}
    )
    adapter.set_leverage = AsyncMock(return_value=True)

    now = datetime.now(timezone.utc)
    adapter.create_market_order = AsyncMock(
        return_value=OrderResult(
            order_id="test123",
            symbol="BTCUSDT",
            side="buy",
            type="market",
            price=50000.0,
            quantity=0.01,
            status="closed",
            timestamp=now,
            raw={"test": True},
        )
    )
    adapter.close_position = AsyncMock(
        return_value=OrderResult(
            order_id="close123",
            symbol="BTCUSDT",
            side="sell",
            type="market",
            price=50000.0,
            quantity=0.01,
            status="closed",
            timestamp=now,
            raw={"test": True},
        )
    )
    return adapter


@pytest.fixture
def strategy() -> ATLASFuturesStrategy:
    """ATLAS strategy instance."""
    return ATLASFuturesStrategy()


@pytest.fixture
def engine(strategy: ATLASFuturesStrategy, mock_adapter: MagicMock) -> StrategyEngine:
    """StrategyEngine with mocked adapter."""
    return StrategyEngine(strategy=strategy, adapter=mock_adapter)


# ============== StrategyEngine Tests ==============


class TestStrategyEngineIntegration:
    """StrategyEngine integration tests."""

    @pytest.mark.asyncio
    async def test_engine_start_stop(
        self, engine: StrategyEngine, mock_adapter: MagicMock
    ):
        """Engine start/stop lifecycle."""
        await engine.start()
        assert engine._running is True
        mock_adapter.initialize.assert_called_once()

        await engine.stop()
        assert engine._running is False
        mock_adapter.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_trading_cycle(
        self, engine: StrategyEngine, mock_adapter: MagicMock
    ):
        """Full trading cycle execution."""
        await engine.start()

        result = await engine.execute_trading_cycle("BTCUSDT")

        assert result is not None
        assert "symbol" in result
        assert "signal" in result
        assert "timestamp" in result
        mock_adapter.fetch_ohlcv.assert_called()

        await engine.stop()

    @pytest.mark.asyncio
    async def test_engine_not_running(self, engine: StrategyEngine):
        """Engine should return None when not running."""
        result = await engine.execute_trading_cycle("BTCUSDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_multi_symbol_execution(
        self, engine: StrategyEngine, mock_adapter: MagicMock
    ):
        """Execute trading cycles for multiple symbols."""
        await engine.start()

        results = []
        for symbol in ["BTCUSDT", "ETHUSDT", "LINKUSDT"]:
            result = await engine.execute_trading_cycle(symbol)
            results.append(result)

        assert len(results) == 3
        assert all(r is not None for r in results)

        await engine.stop()


# ============== Loader Tests ==============


class TestLoaderIntegration:
    """Strategy loader integration tests."""

    def test_load_registered_strategy(self):
        """Load strategy from registry."""
        strategy = get_strategy("mock_strategy")
        assert strategy is not None

    def test_load_strategy_class_dynamic_atlas(self):
        """Dynamic import for ATLAS strategy."""
        strategy_class = load_strategy_class("atlas_futures_p04")
        assert strategy_class is not None
        assert strategy_class.__name__ == "ATLASFuturesStrategy"

        strategy = get_strategy("atlas_futures_p04")
        assert strategy is not None
        assert strategy.STRATEGY_ID == "atlas_futures_p04"

    def test_load_nonexistent_strategy(self):
        """Loading nonexistent strategy returns None."""
        strategy = get_strategy("nonexistent_strategy_xyz")
        assert strategy is None


# ============== Serialization Tests ==============


class TestSerializationIntegration:
    """JSON serialization integration tests."""

    def test_strategy_state_with_position(self, strategy: ATLASFuturesStrategy):
        """Strategy state with positions should serialize."""
        now = datetime.now(timezone.utc)

        pos = Position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            entry_time=now,
            size=0.1,
            leverage=3,
            highest_price=50000.0,
            lowest_price=50000.0,
        )
        strategy.positions["BTCUSDT"] = pos

        state = strategy.get_state()
        json_str = json.dumps(state)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert "BTCUSDT" in parsed["positions"]

    def test_strategy_state_with_signal(self, strategy: ATLASFuturesStrategy):
        """Strategy state with last_signal should serialize."""
        now = datetime.now(timezone.utc)

        strategy.last_signal = Signal(
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            price=50000.0,
            reason="TEST",
            timestamp=now,
        )

        state = strategy.get_state()
        json_str = json.dumps(state)

        parsed = json.loads(json_str)
        assert parsed["last_signal"] is not None
        assert parsed["last_signal"]["signal_type"] == "LONG"

    def test_signal_to_dict(self, strategy: ATLASFuturesStrategy):
        """Signal.to_dict() should produce JSON-safe output."""
        now = datetime.now(timezone.utc)

        signal = Signal(
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            price=50000.0,
            reason="TEST",
            timestamp=now,
        )

        signal_dict = signal.to_dict()
        json_str = json.dumps(signal_dict)

        assert json_str is not None
        assert "T" in signal_dict["timestamp"]


# ============== Sandbox Mode Tests ==============


class TestSandboxModeIntegration:
    """Sandbox mode verification tests."""

    def test_adapter_config_testnet(self, mock_adapter: MagicMock):
        """Adapter should be configured for testnet."""
        assert mock_adapter.config.testnet is True


# ============== Deterministic Tests ==============


class TestDeterministicFlow:
    """Deterministic order flow tests with monkeypatch."""

    @pytest.mark.asyncio
    async def test_forced_long_signal(
        self, engine: StrategyEngine, mock_adapter: MagicMock, monkeypatch
    ):
        """Force LONG signal to test order execution."""
        await engine.start()

        def _force_long(symbol: str, data: pd.DataFrame) -> Signal:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=symbol,
                price=float(data.iloc[-1]["close"]),
                reason="FORCED_LONG",
                timestamp=datetime.now(timezone.utc),
            )

        monkeypatch.setattr(engine.strategy, "generate_signal", _force_long)

        result = await engine.execute_trading_cycle("BTCUSDT")

        assert result is not None
        assert result["signal"] == "LONG"
        mock_adapter.create_market_order.assert_called()

        await engine.stop()


# ============== Error Handling Tests ==============


class TestErrorHandling:
    """Error handling integration tests."""

    @pytest.mark.asyncio
    async def test_engine_handles_adapter_error(
        self, engine: StrategyEngine, mock_adapter: MagicMock
    ):
        """Engine should handle adapter errors gracefully."""
        mock_adapter.fetch_ohlcv = AsyncMock(side_effect=Exception("Network error"))

        await engine.start()
        result = await engine.execute_trading_cycle("BTCUSDT")

        assert result is None

        await engine.stop()

    def test_strategy_handles_empty_positions(self, strategy: ATLASFuturesStrategy):
        """Strategy state with empty positions should serialize."""
        state = strategy.get_state()
        json_str = json.dumps(state)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["positions"] == {}
