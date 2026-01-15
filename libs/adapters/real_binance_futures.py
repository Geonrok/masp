"""
Binance Futures Market Data Adapter (Phase 2A - Read Only)

Provides real-time market data from Binance Futures exchange.
Phase 2A: Read-only, no order execution.
Phase 2C: Full trading implementation.
"""

import logging
import requests
from typing import List, Optional, Dict, Any
from libs.adapters.base import MarketDataAdapter, ExecutionAdapter, MarketQuote, OrderResult

logger = logging.getLogger(__name__)


class BinanceFuturesMarketData(MarketDataAdapter):
    """
    Binance Futures Market Data Adapter
    
    Phase 2A: Real market data (read-only, Public API)
    Supports: get_quote, get_quotes
    
    API Documentation: https://binance-docs.github.io/apidocs/futures/en/
    """
    
    BASE_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        """
        Initialize Binance adapter.
        
        Args:
            api_key: Binance API key (optional for public API)
            api_secret: Binance API secret (optional for public API)
            testnet: Use testnet instead of live (default: False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Multi-Asset-Strategy-Platform/1.0"
        })
        logger.info(f"[Binance] MarketData adapter initialized ({'testnet' if testnet else 'live'}, read-only mode)")
    
    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Symbol in "BTC/USDT:PERP" format
        
        Returns:
            MarketQuote object or None if unavailable
        """
        try:
            binance_symbol = self._convert_symbol(symbol)
            url = f"{self.base_url}/fapi/v1/ticker/bookTicker"
            params = {"symbol": binance_symbol}
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            return MarketQuote(
                symbol=symbol,
                bid=float(data.get("bidPrice", 0)),
                ask=float(data.get("askPrice", 0)),
                last=(float(data.get("bidPrice", 0)) + float(data.get("askPrice", 0))) / 2,  # Mid price
                volume_24h=0.0,  # Not available in bookTicker, would need separate call
                timestamp=""
            )
        
        except requests.exceptions.RequestException as e:
            logger.error(f"[Binance] Failed to get quote for {symbol}: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"[Binance] Failed to parse quote data for {symbol}: {e}")
            return None
    
    def get_quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """
        Get quotes for multiple symbols.
        
        Args:
            symbols: List of symbols in "BTC/USDT:PERP" format
        
        Returns:
            Dict mapping symbol to MarketQuote
        """
        result = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                result[symbol] = quote
        return result
    
    def is_market_open(self) -> bool:
        """
        Check if Binance market is open.
        
        Returns:
            True (Binance Futures operates 24/7)
        """
        return True
    
    def _convert_symbol(self, symbol: str) -> str:
        """
        Convert symbol format: 'BTC/USDT:PERP' → 'BTCUSDT'
        
        Args:
            symbol: Symbol in "BTC/USDT:PERP" format
        
        Returns:
            Binance symbol "BTCUSDT"
        """
        # Remove :PERP suffix if present
        if ":PERP" in symbol:
            symbol = symbol.replace(":PERP", "")
        
        # Remove slash
        return symbol.replace("/", "")


class BinanceFuturesExecution(ExecutionAdapter):
    """
    Binance Futures Execution Adapter
    
    Phase 2A: Disabled (RuntimeError)
    Phase 2C: Real order execution
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        logger.warning(
            "[Binance] Execution adapter initialized. "
            "Phase 2A: Order execution DISABLED. "
            "Use adapter_mode='paper' for simulation."
        )
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """Phase 2A: Order execution disabled."""
        raise RuntimeError(
            "[Binance] place_order() is disabled in Phase 2A (read-only mode). "
            "For order simulation, use adapter_mode='paper'. "
            "For real trading, wait for Phase 2C implementation."
        )
    
    def get_order_status(self, order_id: str) -> Optional[dict]:
        """Phase 2A: Disabled"""
        raise RuntimeError("[Binance] get_order_status() disabled in Phase 2A")
    
    def cancel_order(self, order_id: str) -> bool:
        """Phase 2A: Disabled"""
        raise RuntimeError("[Binance] cancel_order() disabled in Phase 2A")
    
    def get_balance(self, asset: str) -> Optional[float]:
        """Phase 2A: Disabled"""
        raise RuntimeError("[Binance] get_balance() disabled in Phase 2A")
