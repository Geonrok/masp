"""Multi-exchange view component for comparing prices across exchanges."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Session state key prefix
_KEY_PREFIX = "multi_exchange."


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def _get_status_color(status: str) -> str:
    """Get color for exchange status."""
    colors = {
        "online": "green",
        "offline": "red",
        "degraded": "orange",
        "maintenance": "yellow",
        "unknown": "gray",
    }
    return colors.get(status.lower(), "gray")


def _get_demo_exchanges() -> List[Dict[str, Any]]:
    """Generate demo exchange data."""
    return [
        {
            "name": "upbit_spot",
            "display_name": "Upbit",
            "exchange_type": "spot",
            "region": "kr",
            "status": "online",
            "latency_ms": 45.2,
            "base_currency": "KRW",
        },
        {
            "name": "bithumb_spot",
            "display_name": "Bithumb",
            "exchange_type": "spot",
            "region": "kr",
            "status": "online",
            "latency_ms": 62.8,
            "base_currency": "KRW",
        },
        {
            "name": "binance_futures",
            "display_name": "Binance Futures",
            "exchange_type": "futures",
            "region": "global",
            "status": "unknown",
            "latency_ms": 0,
            "base_currency": "USDT",
        },
    ]


def _get_demo_price_comparison() -> Dict[str, Any]:
    """Generate demo price comparison."""
    return {
        "symbol": "BTC",
        "exchanges": {
            "upbit_spot": {
                "bid": 54000000,
                "ask": 54050000,
                "mid": 54025000,
                "last": 54030000,
                "spread_pct": 0.09,
            },
            "bithumb_spot": {
                "bid": 53980000,
                "ask": 54020000,
                "mid": 54000000,
                "last": 54010000,
                "spread_pct": 0.07,
            },
        },
        "best_bid": ("upbit_spot", 54000000),
        "best_ask": ("bithumb_spot", 54020000),
        "cross_exchange_spread_pct": -0.04,
        "price_range_pct": 0.05,
    }


def render_exchange_status_card(exchange: Dict[str, Any]) -> None:
    """Render a single exchange status card.

    Args:
        exchange: Exchange info dict
    """
    status = exchange.get("status", "unknown")
    status_color = _get_status_color(status)

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{exchange.get('display_name', exchange.get('name'))}**")
            region = exchange.get("region", "").upper()
            ex_type = exchange.get("exchange_type", "").title()
            st.caption(f"{region} | {ex_type}")

        with col2:
            st.markdown(f":{status_color}[{status.upper()}]")

        with col3:
            latency = exchange.get("latency_ms", 0)
            if latency > 0:
                st.text(f"{latency:.0f}ms")
            else:
                st.text("-")


def render_exchange_list(
    exchanges: Optional[List[Dict[str, Any]]] = None,
    on_health_check: Optional[Callable[[], None]] = None,
) -> None:
    """Render list of exchanges with status.

    Args:
        exchanges: List of exchange info dicts
        on_health_check: Callback for health check button
    """
    st.subheader("Registered Exchanges")

    if exchanges is None:
        st.caption("Demo Data")
        exchanges = _get_demo_exchanges()

    # Summary
    total = len(exchanges)
    online = len([e for e in exchanges if e.get("status") == "online"])
    st.caption(f"{online}/{total} exchanges online")

    # Health check button
    if on_health_check:
        if st.button("Check Health", key=_key("health_check")):
            on_health_check()
            st.rerun()

    # Exchange list
    for exchange in exchanges:
        render_exchange_status_card(exchange)
        st.divider()


def render_price_comparison(
    comparison: Optional[Dict[str, Any]] = None,
    symbol: str = "BTC",
) -> None:
    """Render price comparison across exchanges.

    Args:
        comparison: Price comparison dict
        symbol: Symbol being compared
    """
    st.subheader(f"Price Comparison: {symbol}")

    if comparison is None:
        st.caption("Demo Data")
        comparison = _get_demo_price_comparison()

    exchanges_data = comparison.get("exchanges", {})

    if not exchanges_data:
        st.info("No price data available")
        return

    # Create comparison table
    table_data = []
    for ex_name, prices in exchanges_data.items():
        table_data.append({
            "Exchange": ex_name.replace("_", " ").title(),
            "Bid": f"{prices.get('bid', 0):,.0f}",
            "Ask": f"{prices.get('ask', 0):,.0f}",
            "Last": f"{prices.get('last', 0):,.0f}",
            "Spread": f"{prices.get('spread_pct', 0):.3f}%",
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Best prices
    col1, col2 = st.columns(2)

    with col1:
        best_bid = comparison.get("best_bid")
        if best_bid:
            st.metric(
                "Best Bid (Sell)",
                f"{best_bid[1]:,.0f}",
                delta=best_bid[0].replace("_", " ").title(),
            )

    with col2:
        best_ask = comparison.get("best_ask")
        if best_ask:
            st.metric(
                "Best Ask (Buy)",
                f"{best_ask[1]:,.0f}",
                delta=best_ask[0].replace("_", " ").title(),
            )

    # Cross-exchange spread
    spread = comparison.get("cross_exchange_spread_pct", 0)
    price_range = comparison.get("price_range_pct", 0)

    st.caption(f"Cross-exchange spread: {spread:.3f}% | Price range: {price_range:.3f}%")


def render_arbitrage_opportunities(
    opportunities: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render arbitrage opportunities.

    Args:
        opportunities: List of arbitrage opportunity dicts
    """
    st.subheader("Arbitrage Opportunities")

    if opportunities is None:
        st.caption("Demo Data")
        opportunities = []

    if not opportunities:
        st.info("No arbitrage opportunities detected")
        return

    for opp in opportunities:
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**{opp.get('symbol')}**")
                st.caption(f"Buy: {opp.get('buy_exchange')} @ {opp.get('buy_price'):,.0f}")

            with col2:
                st.caption(f"Sell: {opp.get('sell_exchange')} @ {opp.get('sell_price'):,.0f}")

            with col3:
                profit = opp.get("profit_pct", 0)
                st.metric("Profit", f"{profit:.2f}%")

            st.divider()


def render_multi_exchange_view(
    exchanges: Optional[List[Dict[str, Any]]] = None,
    symbol: str = "BTC",
    comparison_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
    arbitrage_provider: Optional[Callable[[List[str]], List[Dict[str, Any]]]] = None,
    health_check_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Render complete multi-exchange view.

    Args:
        exchanges: List of exchange info dicts
        symbol: Default symbol for comparison
        comparison_provider: Function to get price comparison
        arbitrage_provider: Function to find arbitrage opportunities
        health_check_callback: Callback for health check button
    """
    st.subheader("Multi-Exchange Dashboard")

    # Symbol selector
    symbols = ["BTC", "ETH", "XRP", "SOL", "ADA"]
    selected_symbol = st.selectbox(
        "Select Symbol",
        options=symbols,
        index=symbols.index(symbol) if symbol in symbols else 0,
        key=_key("symbol"),
    )

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Exchange Status", "Price Comparison", "Arbitrage"])

    with tab1:
        render_exchange_list(
            exchanges=exchanges,
            on_health_check=health_check_callback,
        )

    with tab2:
        if comparison_provider:
            comparison = comparison_provider(selected_symbol)
        else:
            comparison = None

        render_price_comparison(
            comparison=comparison,
            symbol=selected_symbol,
        )

    with tab3:
        if arbitrage_provider:
            opportunities = arbitrage_provider(symbols)
        else:
            opportunities = None

        render_arbitrage_opportunities(opportunities=opportunities)
