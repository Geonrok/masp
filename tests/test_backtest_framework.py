"""
Tests for the enhanced backtesting framework.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from libs.backtest.historical_data import HistoricalDataLoader, OHLCVDataset
from libs.backtest.portfolio_simulator import (
    PortfolioSimulator,
    OrderSide,
    OrderType,
)
from libs.backtest.parameter_optimizer import (
    GridSearchOptimizer,
    OptimizationResult,
    MonteCarloAnalyzer,
)


class TestHistoricalDataLoader:
    """Tests for HistoricalDataLoader."""

    def test_generate_synthetic(self):
        """Test synthetic data generation."""
        loader = HistoricalDataLoader(enable_cache=False)
        dataset = loader.generate_synthetic(
            symbol="TEST",
            days=100,
            initial_price=1000.0,
            volatility=0.02,
        )

        assert dataset.symbol == "TEST"
        assert dataset.length == 100
        assert len(dataset.closes) == 100
        assert len(dataset.opens) == 100
        assert len(dataset.highs) == 100
        assert len(dataset.lows) == 100
        assert len(dataset.volumes) == 100

    def test_dataset_slice(self):
        """Test dataset slicing."""
        loader = HistoricalDataLoader(enable_cache=False)
        dataset = loader.generate_synthetic(days=100)

        sliced = dataset.slice(10, 50)
        assert sliced.length == 40

    def test_ohlcv_consistency(self):
        """Test OHLCV data consistency."""
        loader = HistoricalDataLoader(enable_cache=False)
        dataset = loader.generate_synthetic(days=50)

        # High should be >= open, close, low
        # Low should be <= open, close, high
        for i in range(dataset.length):
            assert dataset.highs[i] >= dataset.opens[i]
            assert dataset.highs[i] >= dataset.closes[i]
            assert dataset.lows[i] <= dataset.opens[i]
            assert dataset.lows[i] <= dataset.closes[i]


class TestPortfolioSimulator:
    """Tests for PortfolioSimulator."""

    def test_initialization(self):
        """Test simulator initialization."""
        sim = PortfolioSimulator(initial_capital=1_000_000)

        assert sim.cash == 1_000_000
        assert len(sim.positions) == 0
        assert len(sim.trades) == 0

    def test_market_order_buy(self):
        """Test market buy order execution."""
        sim = PortfolioSimulator(
            initial_capital=1_000_000,
            commission_rate=0.001,
            slippage_bps=5,
        )

        order = sim.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        assert order.order_id is not None
        assert not order.filled

        # Process bar to fill order
        trades = sim.process_bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open_price=100,
            high=101,
            low=99,
            close=100,
            volume=10000,
        )

        assert len(trades) == 1
        assert trades[0].side == "buy"
        assert trades[0].quantity == 100
        assert sim.cash < 1_000_000  # Cash reduced

    def test_market_order_sell(self):
        """Test market sell order execution."""
        sim = PortfolioSimulator(initial_capital=1_000_000)

        # First buy
        sim.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
        )
        sim.process_bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open_price=100,
            high=101,
            low=99,
            close=100,
            volume=10000,
        )

        # Then sell
        sim.submit_order(
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=100,
        )
        trades = sim.process_bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open_price=100,
            high=101,
            low=99,
            close=100,
            volume=10000,
        )

        assert len(trades) == 1
        assert trades[0].side == "sell"
        assert "TEST" not in sim.positions  # Position closed

    def test_commission_calculation(self):
        """Test commission is correctly calculated."""
        sim = PortfolioSimulator(
            initial_capital=1_000_000,
            commission_rate=0.001,  # 0.1%
            slippage_bps=0,
        )

        sim.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
        )
        trades = sim.process_bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open_price=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
        )

        # Trade value = 100 * 100 = 10000
        # Commission = 10000 * 0.001 = 10
        assert trades[0].commission == pytest.approx(10, rel=0.01)

    def test_equity_calculation(self):
        """Test equity calculation."""
        sim = PortfolioSimulator(initial_capital=100_000)

        sim.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
        )
        sim.process_bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open_price=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
        )

        # Price goes up
        equity = sim.get_equity({"TEST": 110})

        # Position value = 100 * 110 = 11000
        # Started with 100k, bought 10k worth, now have ~90k cash + 11k position
        assert equity > 100_000  # Made profit

    def test_reset(self):
        """Test simulator reset."""
        sim = PortfolioSimulator(initial_capital=1_000_000)

        sim.submit_order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
        )
        sim.process_bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open_price=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
        )

        sim.reset()

        assert sim.cash == 1_000_000
        assert len(sim.positions) == 0
        assert len(sim.trades) == 0


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_comparison(self):
        """Test result comparison by Sharpe ratio."""
        r1 = OptimizationResult(
            params={"a": 1},
            total_return=0.1,
            sharpe_ratio=1.5,
            max_drawdown=10,
            win_rate=60,
            total_trades=100,
        )

        r2 = OptimizationResult(
            params={"a": 2},
            total_return=0.2,
            sharpe_ratio=2.0,
            max_drawdown=15,
            win_rate=55,
            total_trades=80,
        )

        assert r1 < r2  # r1 has lower Sharpe


class TestMonteCarloAnalyzer:
    """Tests for MonteCarloAnalyzer."""

    def test_analyze_trade_sequence(self):
        """Test Monte Carlo analysis on trades."""
        analyzer = MonteCarloAnalyzer(n_simulations=100)

        # Generate sample trades
        trades = [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 75},
            {"pnl": -25},
            {"pnl": 150},
            {"pnl": -100},
            {"pnl": 50},
            {"pnl": -30},
            {"pnl": 80},
            {"pnl": -40},
        ]

        result = analyzer.analyze_trade_sequence(trades)

        assert "original_total_pnl" in result
        assert "mean_simulated_pnl" in result
        assert "confidence_interval_pnl" in result
        assert "probability_profit" in result
        assert result["original_total_pnl"] == sum(t["pnl"] for t in trades)

    def test_insufficient_trades(self):
        """Test error with insufficient trades."""
        analyzer = MonteCarloAnalyzer()
        trades = [{"pnl": 100}]

        result = analyzer.analyze_trade_sequence(trades)
        assert "error" in result


class TestIntegration:
    """Integration tests for the backtest framework."""

    def test_full_backtest_flow(self):
        """Test complete backtest workflow."""
        # Generate data
        loader = HistoricalDataLoader(enable_cache=False)
        dataset = loader.generate_synthetic(
            symbol="TEST",
            days=200,
            initial_price=100,
            volatility=0.02,
        )

        # Create simulator
        sim = PortfolioSimulator(
            initial_capital=100_000,
            commission_rate=0.001,
        )

        # Simple strategy: buy when price crosses above MA
        def simple_strategy(bar_data):
            closes = bar_data.get("closes", [])
            if len(closes) < 20:
                return None

            current_price = closes[-1]
            ma20 = np.mean(closes[-20:])

            if current_price > ma20 * 1.01:  # 1% above MA
                return ("BUY", current_price)
            elif current_price < ma20 * 0.99:  # 1% below MA
                return ("SELL", current_price)
            return None

        # Run backtest
        results = sim.run_backtest(dataset, simple_strategy, position_size=0.1)

        assert len(results) == dataset.length
        assert "equity" in results.columns
        assert "total_pnl" in results.columns

        summary = sim.get_summary()
        assert "total_trades" in summary
        assert "sharpe_ratio" in summary
