"""
Multi-Exchange Coordinator - 멀티 거래소 데이터 통합
- 병렬 데이터 수집
- 거래소 간 가격 비교
- Failover 지원
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from libs.adapters.base import MarketDataAdapter, MarketQuote
from libs.adapters.exchange_registry import (
    ExchangeStatus,
    get_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiExchangeQuote:
    """멀티 거래소 시세 통합."""

    symbol: str
    quotes: Dict[str, MarketQuote] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    best_bid: Optional[Tuple[str, float]] = None  # (exchange, price)
    best_ask: Optional[Tuple[str, float]] = None  # (exchange, price)
    spread_pct: float = 0.0

    def add_quote(self, exchange: str, quote: MarketQuote) -> None:
        """Add quote from an exchange."""
        self.quotes[exchange] = quote
        self._update_best()

    def _update_best(self) -> None:
        """Update best bid/ask prices."""
        best_bid_price = 0.0
        best_bid_exchange = None
        best_ask_price = float("inf")
        best_ask_exchange = None

        for exchange, quote in self.quotes.items():
            if quote.bid and quote.bid > best_bid_price:
                best_bid_price = quote.bid
                best_bid_exchange = exchange

            if quote.ask and quote.ask < best_ask_price:
                best_ask_price = quote.ask
                best_ask_exchange = exchange

        if best_bid_exchange:
            self.best_bid = (best_bid_exchange, best_bid_price)

        if best_ask_exchange and best_ask_price < float("inf"):
            self.best_ask = (best_ask_exchange, best_ask_price)

        # Calculate spread
        if self.best_bid and self.best_ask:
            mid_price = (self.best_bid[1] + self.best_ask[1]) / 2
            if mid_price > 0:
                self.spread_pct = (
                    (self.best_ask[1] - self.best_bid[1]) / mid_price * 100
                )

    def get_arbitrage_opportunity(self) -> Optional[Dict[str, Any]]:
        """Check for arbitrage opportunity (buy at one exchange, sell at another).

        Returns:
            Arbitrage info dict or None if no opportunity
        """
        if not self.best_bid or not self.best_ask:
            return None

        # If best bid > best ask on different exchanges, there's arbitrage
        if self.best_bid[0] != self.best_ask[0] and self.best_bid[1] > self.best_ask[1]:
            profit_pct = (self.best_bid[1] - self.best_ask[1]) / self.best_ask[1] * 100
            return {
                "symbol": self.symbol,
                "buy_exchange": self.best_ask[0],
                "buy_price": self.best_ask[1],
                "sell_exchange": self.best_bid[0],
                "sell_price": self.best_bid[1],
                "profit_pct": profit_pct,
                "timestamp": self.timestamp,
            }

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quotes": {
                k: {"bid": v.bid, "ask": v.ask, "last": v.last}
                for k, v in self.quotes.items()
            },
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread_pct": self.spread_pct,
            "timestamp": self.timestamp.isoformat(),
        }


class MultiExchangeCoordinator:
    """멀티 거래소 코디네이터 - 병렬 데이터 수집 및 통합."""

    def __init__(
        self,
        exchanges: Optional[List[str]] = None,
        max_workers: int = 5,
        timeout: float = 10.0,
    ):
        """초기화.

        Args:
            exchanges: 사용할 거래소 이름 리스트 (None=모든 거래소)
            max_workers: 병렬 작업 수
            timeout: 요청 타임아웃(초)
        """
        self.registry = get_registry()
        self.max_workers = max_workers
        self.timeout = timeout

        self._adapters: Dict[str, MarketDataAdapter] = {}
        self._exchanges = exchanges or list(self.registry.get_all().keys())
        self._last_quotes: Dict[str, MultiExchangeQuote] = {}

        self._init_adapters()

    def _init_adapters(self) -> None:
        """Initialize market data adapters for all exchanges."""
        from libs.adapters.factory import AdapterFactory

        for exchange_name in self._exchanges:
            try:
                adapter = AdapterFactory.create_market_data(exchange_name)
                self._adapters[exchange_name] = adapter
                logger.debug("[MultiExchange] Initialized adapter: %s", exchange_name)
            except Exception as e:
                logger.warning(
                    "[MultiExchange] Failed to init adapter %s: %s", exchange_name, e
                )

    def get_quote(
        self, symbol: str, exchange: Optional[str] = None
    ) -> Optional[MarketQuote]:
        """Get quote from a specific exchange or first available.

        Args:
            symbol: Trading symbol
            exchange: Specific exchange (None = first available)

        Returns:
            MarketQuote or None
        """
        if exchange:
            adapter = self._adapters.get(exchange)
            if adapter:
                try:
                    return adapter.get_quote(symbol)
                except Exception as e:
                    logger.warning(
                        "[MultiExchange] Quote failed for %s on %s: %s",
                        symbol,
                        exchange,
                        e,
                    )
                    self.registry.update_status(
                        exchange, ExchangeStatus.DEGRADED, error=True
                    )
            return None

        # Try all exchanges, return first success
        for ex_name, adapter in self._adapters.items():
            try:
                quote = adapter.get_quote(symbol)
                if quote:
                    return quote
            except Exception:
                continue

        return None

    def get_multi_exchange_quote(
        self, symbol: str, exchanges: Optional[List[str]] = None
    ) -> MultiExchangeQuote:
        """Get quotes from multiple exchanges in parallel.

        Args:
            symbol: Trading symbol
            exchanges: List of exchanges to query (None = all)

        Returns:
            MultiExchangeQuote with aggregated data
        """
        target_exchanges = exchanges or list(self._adapters.keys())
        result = MultiExchangeQuote(symbol=symbol)

        def fetch_quote(exchange: str) -> Tuple[str, Optional[MarketQuote], float]:
            adapter = self._adapters.get(exchange)
            if not adapter:
                return exchange, None, 0.0

            start = time.time()
            try:
                quote = adapter.get_quote(symbol)
                latency = (time.time() - start) * 1000
                self.registry.update_status(exchange, ExchangeStatus.ONLINE, latency)
                return exchange, quote, latency
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.warning(
                    "[MultiExchange] Fetch failed for %s on %s: %s", symbol, exchange, e
                )
                self.registry.update_status(
                    exchange, ExchangeStatus.DEGRADED, latency, error=True
                )
                return exchange, None, latency

        # Parallel fetch
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_quote, ex): ex for ex in target_exchanges}

            for future in as_completed(futures, timeout=self.timeout):
                try:
                    exchange, quote, latency = future.result()
                    if quote:
                        result.add_quote(exchange, quote)
                except Exception as e:
                    exchange = futures[future]
                    logger.warning(
                        "[MultiExchange] Future failed for %s: %s", exchange, e
                    )

        self._last_quotes[symbol] = result
        return result

    def get_quotes_batch(
        self, symbols: List[str], exchanges: Optional[List[str]] = None
    ) -> Dict[str, MultiExchangeQuote]:
        """Get quotes for multiple symbols from multiple exchanges.

        Args:
            symbols: List of trading symbols
            exchanges: List of exchanges (None = all)

        Returns:
            Dict of symbol to MultiExchangeQuote
        """
        results = {}

        for symbol in symbols:
            results[symbol] = self.get_multi_exchange_quote(symbol, exchanges)

        return results

    def find_arbitrage_opportunities(
        self, symbols: List[str], min_profit_pct: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities across exchanges.

        Args:
            symbols: List of symbols to check
            min_profit_pct: Minimum profit percentage threshold

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        quotes = self.get_quotes_batch(symbols)

        for symbol, multi_quote in quotes.items():
            arb = multi_quote.get_arbitrage_opportunity()
            if arb and arb["profit_pct"] >= min_profit_pct:
                opportunities.append(arb)

        # Sort by profit descending
        opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)

        return opportunities

    def get_best_exchange_for_buy(self, symbol: str) -> Optional[Tuple[str, float]]:
        """Find the exchange with the best (lowest) ask price.

        Args:
            symbol: Trading symbol

        Returns:
            (exchange_name, price) or None
        """
        multi_quote = self.get_multi_exchange_quote(symbol)
        return multi_quote.best_ask

    def get_best_exchange_for_sell(self, symbol: str) -> Optional[Tuple[str, float]]:
        """Find the exchange with the best (highest) bid price.

        Args:
            symbol: Trading symbol

        Returns:
            (exchange_name, price) or None
        """
        multi_quote = self.get_multi_exchange_quote(symbol)
        return multi_quote.best_bid

    def get_price_comparison(
        self, symbol: str, exchanges: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get price comparison across exchanges.

        Args:
            symbol: Trading symbol
            exchanges: List of exchanges (None = all)

        Returns:
            Comparison dict with prices and spreads
        """
        multi_quote = self.get_multi_exchange_quote(symbol, exchanges)

        prices = {}
        for exchange, quote in multi_quote.quotes.items():
            mid_price = None
            if quote.bid and quote.ask:
                mid_price = (quote.bid + quote.ask) / 2

            prices[exchange] = {
                "bid": quote.bid,
                "ask": quote.ask,
                "mid": mid_price,
                "last": quote.last,
                "spread_pct": (
                    (quote.ask - quote.bid) / quote.bid * 100
                    if quote.bid and quote.ask
                    else 0
                ),
            }

        # Find min/max prices
        all_mids = [p["mid"] for p in prices.values() if p["mid"]]
        price_range_pct = 0.0
        if all_mids and len(all_mids) > 1:
            min_mid = min(all_mids)
            max_mid = max(all_mids)
            price_range_pct = (max_mid - min_mid) / min_mid * 100 if min_mid > 0 else 0

        return {
            "symbol": symbol,
            "exchanges": prices,
            "best_bid": multi_quote.best_bid,
            "best_ask": multi_quote.best_ask,
            "cross_exchange_spread_pct": multi_quote.spread_pct,
            "price_range_pct": price_range_pct,
            "timestamp": multi_quote.timestamp.isoformat(),
        }

    def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all exchanges.

        Returns:
            Dict of exchange name to status info
        """
        result = {}

        for exchange_name in self._exchanges:
            info = self.registry.get(exchange_name)
            if info:
                result[exchange_name] = {
                    "display_name": info.display_name,
                    "status": info.status.value,
                    "last_check": (
                        info.last_check.isoformat() if info.last_check else None
                    ),
                    "latency_ms": info.latency_ms,
                    "error_count": info.error_count,
                    "adapter_available": exchange_name in self._adapters,
                }

        return result

    def health_check(self) -> Dict[str, ExchangeStatus]:
        """Perform health check on all exchanges.

        Returns:
            Dict of exchange name to status
        """
        results = {}

        for exchange_name in self._exchanges:
            status = self.registry.check_health(exchange_name)
            results[exchange_name] = status

        return results


# =============================================================================
# Factory function
# =============================================================================


def create_multi_exchange_coordinator(
    exchanges: Optional[List[str]] = None,
) -> MultiExchangeCoordinator:
    """Create a MultiExchangeCoordinator instance.

    Args:
        exchanges: List of exchange names (None = all available)

    Returns:
        MultiExchangeCoordinator instance
    """
    return MultiExchangeCoordinator(exchanges=exchanges)
