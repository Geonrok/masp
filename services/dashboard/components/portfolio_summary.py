"""Portfolio summary component for asset allocation and metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Safe Math Helpers
# =============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float with finite check."""
    if value is None:
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safe division with zero and non-finite guards."""
    if den == 0 or not math.isfinite(den):
        return default
    result = num / den
    return result if math.isfinite(result) else default


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PortfolioPosition:
    """Single position in portfolio."""
    symbol: str
    exchange: str
    quantity: float
    avg_price: float
    current_price: float
    
    @property
    def cost(self) -> float:
        """Total cost basis."""
        return _safe_float(self.quantity * self.avg_price)
    
    @property
    def value(self) -> float:
        """Current market value."""
        return _safe_float(self.quantity * self.current_price)
    
    @property
    def pnl(self) -> float:
        """Unrealized PnL."""
        return self.value - self.cost
    
    @property
    def pnl_percent(self) -> float:
        """PnL percentage."""
        return _safe_div(self.pnl, self.cost, 0.0) * 100


@dataclass
class PortfolioSummary:
    """Aggregated portfolio summary."""
    total_cost: float
    total_value: float
    cash_balance: float
    positions: List[PortfolioPosition]
    
    @property
    def total_pnl(self) -> float:
        """Total unrealized PnL."""
        return self.total_value - self.total_cost
    
    @property
    def total_pnl_percent(self) -> float:
        """Total PnL percentage."""
        return _safe_div(self.total_pnl, self.total_cost, 0.0) * 100
    
    @property
    def total_assets(self) -> float:
        """Total assets (positions + cash)."""
        return self.total_value + self.cash_balance
    
    @property
    def cash_ratio(self) -> float:
        """Cash ratio percentage."""
        return _safe_div(self.cash_balance, self.total_assets, 0.0) * 100
    
    @property
    def invested_ratio(self) -> float:
        """Invested ratio percentage."""
        return _safe_div(self.total_value, self.total_assets, 0.0) * 100


# =============================================================================
# Demo Data (Deterministic: fixed values, no randomness)
# =============================================================================


def _get_demo_portfolio() -> PortfolioSummary:
    """Generate deterministic demo portfolio data with fixed values."""
    # Demo positions with realistic crypto data (fixed values for determinism)
    demo_positions = [
        ("BTC", "upbit", 0.15, 55_000_000, 58_000_000),
        ("ETH", "upbit", 2.5, 2_800_000, 3_100_000),
        ("XRP", "bithumb", 5000, 800, 750),
        ("SOL", "upbit", 10, 150_000, 180_000),
        ("ADA", "bithumb", 3000, 600, 580),
        ("AVAX", "upbit", 15, 40_000, 45_000),
    ]
    
    positions = [
        PortfolioPosition(
            symbol=symbol,
            exchange=exchange,
            quantity=qty,
            avg_price=avg_price,
            current_price=current_price,
        )
        for symbol, exchange, qty, avg_price, current_price in demo_positions
    ]
    
    total_cost = sum(p.cost for p in positions)
    total_value = sum(p.value for p in positions)
    cash_balance = 5_000_000  # 5M KRW cash
    
    return PortfolioSummary(
        total_cost=total_cost,
        total_value=total_value,
        cash_balance=cash_balance,
        positions=positions,
    )


# =============================================================================
# Aggregation Functions
# =============================================================================


def _aggregate_by_exchange(positions: List[PortfolioPosition]) -> Dict[str, float]:
    """Aggregate position values by exchange."""
    result: Dict[str, float] = {}
    for pos in positions:
        exchange = pos.exchange.upper()
        result[exchange] = result.get(exchange, 0.0) + pos.value
    return dict(sorted(result.items()))


def _aggregate_by_symbol(positions: List[PortfolioPosition]) -> Dict[str, float]:
    """Aggregate position values by symbol (normalized to uppercase)."""
    result: Dict[str, float] = {}
    for pos in positions:
        # Normalize symbol to uppercase for consistency
        symbol = pos.symbol.upper()
        result[symbol] = result.get(symbol, 0.0) + pos.value
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# =============================================================================
# Visualization
# =============================================================================

# Color palette for charts
CHART_COLORS = [
    "#00C853",  # Green
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#FF5252",  # Red
    "#FFEB3B",  # Yellow
    "#795548",  # Brown
]


def _render_allocation_chart(
    data: Dict[str, float],
    title: str,
    cash_balance: float = 0.0,
    show_cash: bool = True,
) -> None:
    """Render asset allocation pie chart."""
    if not data and cash_balance <= 0:
        st.info("No allocation data to display.")
        return
    
    labels = list(data.keys())
    values = list(data.values())
    
    # Add cash if requested
    if show_cash and cash_balance > 0:
        labels.append("CASH")
        values.append(cash_balance)
    
    # Assign colors with wrapping for large label counts
    colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(labels))]
    
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textposition="inside",
            hole=0.4,  # Donut chart
        )
    )
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_position_table(positions: List[PortfolioPosition]) -> None:
    """Render positions as a table."""
    if not positions:
        st.info("No positions to display.")
        return
    
    # Sort by value descending
    sorted_positions = sorted(positions, key=lambda p: p.value, reverse=True)
    
    data = [
        {
            "Symbol": p.symbol.upper(),
            "Exchange": p.exchange.upper(),
            "Quantity": f"{p.quantity:,.4f}",
            "Avg Price": f"KRW {p.avg_price:,.0f}",
            "Current": f"KRW {p.current_price:,.0f}",
            "Value": f"KRW {p.value:,.0f}",
            "PnL": f"KRW {p.pnl:+,.0f}",
            "PnL%": f"{p.pnl_percent:+.2f}%",
        }
        for p in sorted_positions
    ]
    
    st.dataframe(data, use_container_width=True, hide_index=True)


# =============================================================================
# Main Render Function
# =============================================================================


def render_portfolio_summary(
    portfolio: Optional[PortfolioSummary] = None,
    view_mode: str = "exchange",
) -> None:
    """Render portfolio summary with metrics and allocation chart.
    
    Args:
        portfolio: Portfolio data. If None, uses demo data.
        view_mode: Allocation view mode - "exchange" or "symbol".
    """
    st.subheader("Portfolio Summary")
    
    # Use demo data if not provided
    if portfolio is None:
        st.caption("Demo Data - Connect to live API for real portfolio")
        portfolio = _get_demo_portfolio()
    
    # Handle empty portfolio
    if not portfolio.positions and portfolio.cash_balance <= 0:
        st.info("No portfolio data available.")
        return
    
    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Assets",
            f"KRW {portfolio.total_assets:,.0f}",
        )
    
    with col2:
        st.metric(
            "Invested",
            f"KRW {portfolio.total_value:,.0f}",
            delta=f"{portfolio.invested_ratio:.1f}%",
        )
    
    with col3:
        pnl_color = "normal" if portfolio.total_pnl >= 0 else "inverse"
        st.metric(
            "Total PnL",
            f"KRW {portfolio.total_pnl:+,.0f}",
            delta=f"{portfolio.total_pnl_percent:+.2f}%",
            delta_color=pnl_color,
        )
    
    with col4:
        st.metric(
            "Cash",
            f"KRW {portfolio.cash_balance:,.0f}",
            delta=f"{portfolio.cash_ratio:.1f}%",
        )
    
    # View mode selector with namespaced key
    selected_view = st.radio(
        "Allocation View",
        options=["exchange", "symbol"],
        index=0 if view_mode == "exchange" else 1,
        horizontal=True,
        key="ps_allocation_view_mode",
    )
    
    # Allocation chart
    if selected_view == "exchange":
        allocation_data = _aggregate_by_exchange(portfolio.positions)
        _render_allocation_chart(
            allocation_data,
            title="Allocation by Exchange",
            cash_balance=portfolio.cash_balance,
            show_cash=True,
        )
    else:
        allocation_data = _aggregate_by_symbol(portfolio.positions)
        _render_allocation_chart(
            allocation_data,
            title="Allocation by Symbol",
            cash_balance=portfolio.cash_balance,
            show_cash=True,
        )
    
    # Position details table
    st.subheader("Position Details")
    _render_position_table(portfolio.positions)
