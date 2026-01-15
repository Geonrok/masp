"""
Binance Futures Adapter (ccxt-based).
Supports testnet and production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)


@dataclass
class BinanceFuturesConfig:
    """Binance Futures adapter config."""
    api_key: str
    secret_key: str
    testnet: bool = True
    default_leverage: int = 3
    recv_window: int = 5000


@dataclass
class OrderResult:
    """Order result."""
    order_id: str
    symbol: str
    side: str
    type: str
    price: float
    quantity: float
    status: str
    timestamp: datetime
    raw: Dict[str, Any]


class BinanceFuturesAdapter:
    """
    Binance Futures adapter.

    Features:
        - Testnet/production support
        - Market/limit orders
        - Position queries
        - Leverage configuration
        - Balance queries
    """

    def __init__(self, config: BinanceFuturesConfig):
        self.config = config
        self._exchange: Optional[ccxt.binance] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize exchange connection."""
        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            "recvWindow": self.config.recv_window,
        }

        if self.config.testnet:
            options["urls"] = {
                "api": {
                    "public": "https://testnet.binancefuture.com",
                    "private": "https://testnet.binancefuture.com",
                }
            }

        self._exchange = ccxt.binance(
            {
                "apiKey": self.config.api_key,
                "secret": self.config.secret_key,
                "enableRateLimit": True,
                "options": options,
            }
        )

        if self.config.testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("[BINANCE] Sandbox mode enabled (TESTNET)")
        else:
            logger.info("[BINANCE] Connecting to PRODUCTION")

        await self._exchange.load_markets()
        self._initialized = True

        if self.config.testnet:
            try:
                balance = await self._exchange.fetch_balance()
                usdt = balance.get("USDT", {})
                logger.info(
                    "[BINANCE] Testnet connection verified, USDT: %.2f",
                    usdt.get("total", 0),
                )
            except Exception as exc:
                logger.error("[BINANCE] Testnet verification failed: %s", exc)
                raise RuntimeError("Testnet connection failed") from exc
        else:
            logger.info("[BINANCE] Initialized PRODUCTION")

    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._initialized = False
            logger.info("[BINANCE] Connection closed")

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")

    async def get_balance(self) -> Dict[str, float]:
        """Get USDT balance."""
        self._check_initialized()
        balance = await self._exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "total": float(usdt.get("total", 0)),
            "free": float(usdt.get("free", 0)),
            "used": float(usdt.get("used", 0)),
        }

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage."""
        self._check_initialized()
        try:
            await self._exchange.set_leverage(leverage, symbol)
            logger.info("[BINANCE] Set leverage %s: %sx", symbol, leverage)
            return True
        except Exception as exc:
            logger.error("[BINANCE] Failed to set leverage: %s", exc)
            return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all positions."""
        self._check_initialized()
        positions = await self._exchange.fetch_positions()
        return [
            {
                "symbol": pos["symbol"],
                "side": "LONG" if pos["side"] == "long" else "SHORT",
                "size": abs(float(pos["contracts"])),
                "entry_price": float(pos["entryPrice"]) if pos["entryPrice"] else 0,
                "unrealized_pnl": float(pos["unrealizedPnl"]) if pos["unrealizedPnl"] else 0,
                "leverage": int(pos["leverage"]) if pos["leverage"] else 1,
                "liquidation_price": float(pos["liquidationPrice"]) if pos["liquidationPrice"] else 0,
            }
            for pos in positions
            if abs(float(pos.get("contracts", 0))) > 0
        ]

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch a position by symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                return pos
        return None

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Create market order."""
        self._check_initialized()

        params = {"reduceOnly": reduce_only}
        order = await self._exchange.create_market_order(
            symbol=symbol,
            side=side.lower(),
            amount=amount,
            params=params,
        )

        result = OrderResult(
            order_id=order["id"],
            symbol=order["symbol"],
            side=order["side"].upper(),
            type=order["type"].upper(),
            price=float(order["average"]) if order["average"] else 0,
            quantity=float(order["filled"]),
            status=order["status"],
            timestamp=datetime.now(),
            raw=order,
        )

        logger.info("[BINANCE] Market order: %s %s %s @ %s", side.upper(), amount, symbol, result.price)
        return result

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Create limit order."""
        self._check_initialized()

        params = {"reduceOnly": reduce_only}
        order = await self._exchange.create_limit_order(
            symbol=symbol,
            side=side.lower(),
            amount=amount,
            price=price,
            params=params,
        )

        result = OrderResult(
            order_id=order["id"],
            symbol=order["symbol"],
            side=order["side"].upper(),
            type=order["type"].upper(),
            price=float(order["price"]),
            quantity=float(order["amount"]),
            status=order["status"],
            timestamp=datetime.now(),
            raw=order,
        )

        logger.info("[BINANCE] Limit order: %s %s %s @ %s", side.upper(), amount, symbol, price)
        return result

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        self._check_initialized()
        try:
            await self._exchange.cancel_order(order_id, symbol)
            logger.info("[BINANCE] Cancelled order %s", order_id)
            return True
        except Exception as exc:
            logger.error("[BINANCE] Failed to cancel order: %s", exc)
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol."""
        self._check_initialized()
        orders = await self._exchange.fetch_open_orders(symbol)
        cancelled = 0
        for order in orders:
            if await self.cancel_order(symbol, order["id"]):
                cancelled += 1
        return cancelled

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch OHLCV data."""
        self._check_initialized()
        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return [
            {
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
            for candle in ohlcv
        ]

    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get current ticker."""
        self._check_initialized()
        ticker = await self._exchange.fetch_ticker(symbol)
        return {
            "last": float(ticker["last"]),
            "bid": float(ticker["bid"]) if ticker["bid"] else 0,
            "ask": float(ticker["ask"]) if ticker["ask"] else 0,
            "volume": float(ticker["quoteVolume"]) if ticker["quoteVolume"] else 0,
        }

    async def close_position(self, symbol: str) -> Optional[OrderResult]:
        """Close a position fully."""
        position = await self.get_position(symbol)
        if not position or position["size"] == 0:
            return None

        side = "sell" if position["side"] == "LONG" else "buy"
        return await self.create_market_order(
            symbol=symbol,
            side=side,
            amount=position["size"],
            reduce_only=True,
        )
