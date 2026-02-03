"""
Mock adapters for Phase 0 testing.
No real API calls - generates deterministic mock data.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from libs.adapters.base import (
    ExecutionAdapter,
    MarketDataAdapter,
    MarketQuote,
    OrderResult,
)


class MockMarketDataAdapter(MarketDataAdapter):
    """
    Mock market data adapter.
    Generates deterministic prices based on symbol hash.
    """

    # Base prices for different asset types
    BASE_PRICES = {
        # Crypto
        "BTC": 45000.0,
        "ETH": 2500.0,
        "SOL": 100.0,
        "XRP": 0.50,
        "ADA": 0.40,
        # KR Stocks (in KRW)
        "005930": 72000.0,  # Samsung
        "000660": 130000.0,  # SK Hynix
        "035420": 180000.0,  # NAVER
        "051910": 450000.0,  # LG Chem
        "006400": 400000.0,  # Samsung SDI
        # Futures
        "101S6000": 350.0,
        "101S3000": 280.0,
        "101S9000": 320.0,
    }

    def __init__(self):
        self._market_open = True

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for a symbol."""
        # Check direct match
        for key, price in self.BASE_PRICES.items():
            if key in symbol:
                return price

        # Default based on symbol hash
        hash_bytes = hashlib.sha256(symbol.encode()).digest()
        return 100.0 + (hash_bytes[0] / 255) * 900  # 100-1000 range

    def _add_spread(self, base_price: float, symbol: str) -> tuple[float, float, float]:
        """Add bid/ask spread to base price."""
        # Use symbol hash for deterministic spread
        hash_bytes = hashlib.sha256(symbol.encode()).digest()
        spread_pct = 0.001 + (hash_bytes[1] / 255) * 0.004  # 0.1%-0.5% spread

        spread = base_price * spread_pct
        bid = base_price - spread / 2
        ask = base_price + spread / 2
        last = base_price + (hash_bytes[2] - 128) / 128 * spread  # Last within spread

        return round(bid, 4), round(ask, 4), round(last, 4)

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Get mock quote for a symbol."""
        base_price = self._get_base_price(symbol)
        bid, ask, last = self._add_spread(base_price, symbol)

        # Mock volume based on symbol
        hash_bytes = hashlib.sha256(symbol.encode()).digest()
        volume = 1000000 + hash_bytes[3] * 100000  # 1M-26M

        return MarketQuote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            volume_24h=volume,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """Get mock quotes for multiple symbols."""
        return {symbol: self.get_quote(symbol) for symbol in symbols}

    def is_market_open(self) -> bool:
        """Mock always returns market open."""
        return self._market_open

    def set_market_open(self, is_open: bool) -> None:
        """Set mock market state (for testing)."""
        self._market_open = is_open


class MockExecutionAdapter(ExecutionAdapter):
    """
    Mock execution adapter.
    Simulates order execution without real API calls.
    """

    def __init__(self):
        self._orders: dict[str, dict] = {}
        self._balances: dict[str, float] = {
            "USDT": 100000.0,
            "KRW": 100000000.0,
            "BTC": 1.0,
            "ETH": 10.0,
        }

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """Place a mock order."""
        order_id = f"MOCK-{uuid.uuid4().hex[:8].upper()}"

        # Get mock fill price
        market_adapter = MockMarketDataAdapter()
        quote = market_adapter.get_quote(symbol)

        if quote:
            fill_price = quote.ask if side == "BUY" else quote.bid
        else:
            fill_price = price or 100.0

        # Store order
        self._orders[order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "fill_price": fill_price,
            "status": "FILLED",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            status="FILLED",
            message="Mock order filled successfully",
            mock=True,
        )

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get mock order status."""
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a mock order."""
        if order_id in self._orders:
            self._orders[order_id]["status"] = "CANCELLED"
            return True
        return False

    def get_balance(self, asset: str) -> Optional[float]:
        """Get mock balance."""
        return self._balances.get(asset)

    def set_balance(self, asset: str, amount: float) -> None:
        """Set mock balance (for testing)."""
        self._balances[asset] = amount
