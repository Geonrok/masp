"""
Base adapter interfaces for market data and execution.
These define the contract for real adapters in future phases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MarketQuote:
    """Market quote data."""
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume_24h: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class OrderResult:
    """Result of an order attempt."""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: Optional[float] = None
    status: str = "unknown"
    message: Optional[str] = None
    mock: bool = True


class MarketDataAdapter(ABC):
    """
    Abstract base class for market data adapters.
    
    Implementations should fetch real market data from exchanges/brokers.
    Phase 0 uses MockMarketDataAdapter.
    """
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            MarketQuote or None if unavailable
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """
        Get quotes for multiple symbols.
        
        Args:
            symbols: List of trading symbols
        
        Returns:
            Dict mapping symbol to MarketQuote
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        raise NotImplementedError


class ExecutionAdapter(ABC):
    """
    Abstract base class for execution adapters.
    
    Implementations should execute real orders via exchange/broker APIs.
    Phase 0 uses MockExecutionAdapter.
    """
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,  # BUY/SELL
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: MARKET, LIMIT, etc.
            price: Limit price (for LIMIT orders)
        
        Returns:
            OrderResult with execution details
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get status of an existing order.
        
        Args:
            order_id: Order identifier
        
        Returns:
            Order status dict or None if not found
        """
        raise NotImplementedError
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order identifier
        
        Returns:
            True if cancelled successfully
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_balance(self, asset: str) -> Optional[float]:
        """
        Get balance for an asset.
        
        Args:
            asset: Asset symbol (e.g., "USDT", "BTC")
        
        Returns:
            Available balance or None
        """
        raise NotImplementedError


