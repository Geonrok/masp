"""Tests for portfolio summary component."""
from __future__ import annotations

import math


def test_import_portfolio_summary():
    """Test module imports correctly."""
    from services.dashboard.components import portfolio_summary

    assert hasattr(portfolio_summary, "render_portfolio_summary")
    assert hasattr(portfolio_summary, "_safe_float")
    assert hasattr(portfolio_summary, "_safe_div")
    assert hasattr(portfolio_summary, "_get_demo_portfolio")
    assert hasattr(portfolio_summary, "_aggregate_by_exchange")
    assert hasattr(portfolio_summary, "_aggregate_by_symbol")
    assert hasattr(portfolio_summary, "PortfolioPosition")
    assert hasattr(portfolio_summary, "PortfolioSummary")


def test_safe_float():
    """Test _safe_float handles various inputs."""
    from services.dashboard.components.portfolio_summary import _safe_float

    # Normal values
    assert _safe_float(1.5) == 1.5
    assert _safe_float(0) == 0.0
    assert _safe_float(-10) == -10.0

    # None
    assert _safe_float(None) == 0.0
    assert _safe_float(None, 5.0) == 5.0

    # Non-finite
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("-inf")) == 0.0
    assert _safe_float(float("nan")) == 0.0

    # Invalid types
    assert _safe_float("invalid") == 0.0


def test_safe_div():
    """Test _safe_div handles edge cases."""
    from services.dashboard.components.portfolio_summary import _safe_div

    # Normal division
    assert _safe_div(10, 2) == 5.0
    assert _safe_div(1, 4) == 0.25

    # Zero denominator
    assert _safe_div(10, 0) == 0.0
    assert _safe_div(10, 0, -1.0) == -1.0

    # Inf denominator
    assert _safe_div(10, float("inf")) == 0.0


def test_portfolio_position_properties():
    """Test PortfolioPosition computed properties."""
    from services.dashboard.components.portfolio_summary import PortfolioPosition

    pos = PortfolioPosition(
        symbol="BTC",
        exchange="upbit",
        quantity=0.1,
        avg_price=50_000_000,
        current_price=55_000_000,
    )

    # Cost: 0.1 * 50,000,000 = 5,000,000
    assert pos.cost == 5_000_000

    # Value: 0.1 * 55,000,000 = 5,500,000
    assert pos.value == 5_500_000

    # PnL: 5,500,000 - 5,000,000 = 500,000
    assert pos.pnl == 500_000

    # PnL%: 500,000 / 5,000,000 * 100 = 10%
    assert pos.pnl_percent == 10.0


def test_portfolio_position_loss():
    """Test PortfolioPosition with loss."""
    from services.dashboard.components.portfolio_summary import PortfolioPosition

    pos = PortfolioPosition(
        symbol="ETH",
        exchange="bithumb",
        quantity=1.0,
        avg_price=3_000_000,
        current_price=2_700_000,
    )

    # PnL: 2,700,000 - 3,000,000 = -300,000
    assert pos.pnl == -300_000

    # PnL%: -300,000 / 3,000,000 * 100 = -10%
    assert pos.pnl_percent == -10.0


def test_portfolio_summary_properties():
    """Test PortfolioSummary computed properties."""
    from services.dashboard.components.portfolio_summary import (
        PortfolioPosition,
        PortfolioSummary,
    )

    positions = [
        PortfolioPosition("BTC", "upbit", 0.1, 50_000_000, 55_000_000),
        PortfolioPosition("ETH", "upbit", 1.0, 3_000_000, 3_300_000),
    ]

    summary = PortfolioSummary(
        total_cost=8_000_000,  # 5M + 3M
        total_value=8_800_000,  # 5.5M + 3.3M
        cash_balance=2_000_000,
        positions=positions,
    )

    # Total PnL: 8,800,000 - 8,000,000 = 800,000
    assert summary.total_pnl == 800_000

    # Total PnL%: 800,000 / 8,000,000 * 100 = 10%
    assert summary.total_pnl_percent == 10.0

    # Total assets: 8,800,000 + 2,000,000 = 10,800,000
    assert summary.total_assets == 10_800_000

    # Cash ratio: 2,000,000 / 10,800,000 * 100 ≈ 18.52%
    assert abs(summary.cash_ratio - 18.52) < 0.1

    # Invested ratio: 8,800,000 / 10,800,000 * 100 ≈ 81.48%
    assert abs(summary.invested_ratio - 81.48) < 0.1


def test_demo_portfolio_structure():
    """Test demo portfolio has required structure."""
    from services.dashboard.components.portfolio_summary import _get_demo_portfolio

    portfolio = _get_demo_portfolio()

    # Has positions
    assert len(portfolio.positions) >= 3

    # Has cash
    assert portfolio.cash_balance > 0

    # Has cost and value
    assert portfolio.total_cost > 0
    assert portfolio.total_value > 0

    # Deterministic: calling twice gives same results
    portfolio2 = _get_demo_portfolio()
    assert portfolio.total_cost == portfolio2.total_cost
    assert portfolio.cash_balance == portfolio2.cash_balance


def test_aggregate_by_exchange():
    """Test aggregation by exchange."""
    from services.dashboard.components.portfolio_summary import (
        PortfolioPosition,
        _aggregate_by_exchange,
    )

    positions = [
        PortfolioPosition("BTC", "upbit", 0.1, 50_000_000, 55_000_000),
        PortfolioPosition("ETH", "upbit", 1.0, 3_000_000, 3_300_000),
        PortfolioPosition("XRP", "bithumb", 1000, 800, 750),
    ]

    result = _aggregate_by_exchange(positions)

    # UPBIT: 5,500,000 + 3,300,000 = 8,800,000
    assert result["UPBIT"] == 8_800_000

    # BITHUMB: 1000 * 750 = 750,000
    assert result["BITHUMB"] == 750_000

    # Sorted alphabetically
    assert list(result.keys()) == ["BITHUMB", "UPBIT"]


def test_aggregate_by_symbol():
    """Test aggregation by symbol."""
    from services.dashboard.components.portfolio_summary import (
        PortfolioPosition,
        _aggregate_by_symbol,
    )

    positions = [
        PortfolioPosition("BTC", "upbit", 0.05, 50_000_000, 55_000_000),
        PortfolioPosition("BTC", "bithumb", 0.05, 50_000_000, 55_000_000),
        PortfolioPosition("ETH", "upbit", 1.0, 3_000_000, 3_300_000),
    ]

    result = _aggregate_by_symbol(positions)

    # BTC: 2,750,000 + 2,750,000 = 5,500,000
    assert result["BTC"] == 5_500_000

    # ETH: 3,300,000
    assert result["ETH"] == 3_300_000

    # Sorted by value descending
    assert list(result.keys()) == ["BTC", "ETH"]


def test_aggregate_empty_positions():
    """Test aggregation with empty positions."""
    from services.dashboard.components.portfolio_summary import (
        _aggregate_by_exchange,
        _aggregate_by_symbol,
    )

    assert _aggregate_by_exchange([]) == {}
    assert _aggregate_by_symbol([]) == {}


def test_portfolio_summary_zero_cost():
    """Test PortfolioSummary with zero cost (avoid division by zero)."""
    from services.dashboard.components.portfolio_summary import PortfolioSummary

    summary = PortfolioSummary(
        total_cost=0,
        total_value=0,
        cash_balance=1_000_000,
        positions=[],
    )

    # Should not raise, return 0
    assert summary.total_pnl_percent == 0.0
    assert summary.cash_ratio == 100.0
    assert summary.invested_ratio == 0.0


def test_position_zero_cost():
    """Test PortfolioPosition with zero cost (avoid division by zero)."""
    from services.dashboard.components.portfolio_summary import PortfolioPosition

    pos = PortfolioPosition(
        symbol="FREE",
        exchange="upbit",
        quantity=100,
        avg_price=0,  # Free coins
        current_price=1000,
    )

    # Cost is 0
    assert pos.cost == 0

    # PnL% should be 0 (not inf or nan)
    assert pos.pnl_percent == 0.0
    assert math.isfinite(pos.pnl_percent)
