"""
Binance Futures Market Data & Execution Adapter

Phase 9: Full USDT-M Futures trading support.

Features:
- Real-time market data (quotes, OHLCV, orderbook)
- Order execution (market, limit, stop)
- Position management
- Leverage and margin type control
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from libs.adapters.base import (
    OHLCV,
    ExecutionAdapter,
    MarketDataAdapter,
    MarketQuote,
    OrderResult,
)
from libs.adapters.binance_api import BinanceAPI, BinanceMarket

logger = logging.getLogger(__name__)


class BinanceFuturesMarketData(MarketDataAdapter):
    """
    Binance USDT-M Futures Market Data Adapter.

    Symbol format: "BTC/USDT:PERP" or "BTC/USDT" -> "BTCUSDT"
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        self.api = BinanceAPI(
            api_key=api_key,
            api_secret=api_secret,
            market=BinanceMarket.FUTURES_USDT,
            testnet=testnet,
        )
        self.testnet = testnet
        logger.info(
            "[BinanceFutures] MarketData initialized (%s)",
            "testnet" if testnet else "live",
        )

    def _convert_symbol(self, symbol: str) -> str:
        """Convert 'BTC/USDT:PERP' or 'BTC/USDT' -> 'BTCUSDT'."""
        # Remove :PERP suffix
        if ":PERP" in symbol:
            symbol = symbol.replace(":PERP", "")
        return symbol.replace("/", "")

    def _to_standard_symbol(self, binance_symbol: str) -> str:
        """Convert 'BTCUSDT' -> 'BTC/USDT:PERP'."""
        for quote in ["USDT", "BUSD"]:
            if binance_symbol.endswith(quote):
                base = binance_symbol[: -len(quote)]
                return f"{base}/{quote}:PERP"
        return binance_symbol

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Get current quote for a symbol."""
        try:
            binance_symbol = self._convert_symbol(symbol)
            data = self.api.get_ticker(binance_symbol)

            bid = float(data.get("bidPrice", 0))
            ask = float(data.get("askPrice", 0))
            last = (bid + ask) / 2 if bid and ask else bid or ask

            return MarketQuote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume_24h=0.0,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error("[BinanceFutures] Failed to get quote for %s: %s", symbol, e)
            return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """Get quotes for multiple symbols."""
        result = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                result[symbol] = quote
        return result

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 100,
    ) -> List[OHLCV]:
        """Get OHLCV data."""
        try:
            binance_symbol = self._convert_symbol(symbol)
            klines = self.api.get_klines(binance_symbol, interval=interval, limit=limit)

            result = []
            for k in klines:
                result.append(
                    OHLCV(
                        timestamp=datetime.fromtimestamp(k[0] / 1000).isoformat(),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                    )
                )
            return result
        except Exception as e:
            logger.error("[BinanceFutures] Failed to get OHLCV for %s: %s", symbol, e)
            return []

    def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict]:
        """Get order book."""
        try:
            binance_symbol = self._convert_symbol(symbol)
            data = self.api.get_orderbook(binance_symbol, limit=depth)

            return {
                "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
            }
        except Exception as e:
            logger.error(
                "[BinanceFutures] Failed to get orderbook for %s: %s", symbol, e
            )
            return None

    def is_market_open(self) -> bool:
        """Binance Futures operates 24/7."""
        return True

    def get_all_symbols(self) -> List[str]:
        """Get all available perpetual pairs."""
        try:
            info = self.api.get_exchange_info()
            symbols = []
            for s in info.get("symbols", []):
                if (
                    s.get("status") == "TRADING"
                    and s.get("contractType") == "PERPETUAL"
                ):
                    symbols.append(f"{s['baseAsset']}/{s['quoteAsset']}:PERP")
            return symbols
        except Exception as e:
            logger.error("[BinanceFutures] Failed to get symbols: %s", e)
            return []


@dataclass
class BinanceFuturesOrderResult(OrderResult):
    """Extended order result for Binance Futures."""

    client_order_id: Optional[str] = None
    executed_qty: float = 0.0
    avg_price: float = 0.0
    reduce_only: bool = False
    position_side: str = "BOTH"
    commission: float = 0.0


class BinanceFuturesExecution(ExecutionAdapter):
    """
    Binance USDT-M Futures Execution Adapter.

    Supports:
    - Market/Limit/Stop orders
    - Long/Short positions
    - Leverage control (1x-125x)
    - Cross/Isolated margin
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        default_leverage: int = 10,
    ):
        self.api = BinanceAPI(
            api_key=api_key,
            api_secret=api_secret,
            market=BinanceMarket.FUTURES_USDT,
            testnet=testnet,
        )
        self.testnet = testnet
        self.default_leverage = default_leverage
        self._leverage_cache: Dict[str, int] = {}

        # Verify credentials
        if not self.api.api_key or not self.api.api_secret:
            logger.warning(
                "[BinanceFutures] API credentials not provided. "
                "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
            )
        else:
            # Sync time
            self.api.sync_time()

        logger.info(
            "[BinanceFutures] Execution initialized (%s, leverage=%dx)",
            "testnet" if testnet else "live",
            default_leverage,
        )

    def _convert_symbol(self, symbol: str) -> str:
        """Convert 'BTC/USDT:PERP' -> 'BTCUSDT'."""
        if ":PERP" in symbol:
            symbol = symbol.replace(":PERP", "")
        return symbol.replace("/", "")

    def _ensure_leverage(self, symbol: str, leverage: Optional[int] = None) -> int:
        """Ensure leverage is set for a symbol."""
        binance_symbol = self._convert_symbol(symbol)
        target_leverage = leverage or self.default_leverage

        if self._leverage_cache.get(binance_symbol) == target_leverage:
            return target_leverage

        try:
            self.api.set_leverage(binance_symbol, target_leverage)
            self._leverage_cache[binance_symbol] = target_leverage
            logger.info(
                "[BinanceFutures] Leverage set: %s -> %dx", symbol, target_leverage
            )
            return target_leverage
        except Exception as e:
            logger.warning(
                "[BinanceFutures] Failed to set leverage for %s: %s", symbol, e
            )
            return self.default_leverage

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,  # USDT amount
        price: Optional[float] = None,
        leverage: Optional[int] = None,
        reduce_only: bool = False,
        **kwargs,
    ) -> BinanceFuturesOrderResult:
        """
        Place a futures order.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT:PERP")
            side: "BUY" (long) or "SELL" (short)
            order_type: "MARKET", "LIMIT", "STOP_MARKET"
            units: Quantity in base asset
            amount_krw: Amount in USDT (for calculating quantity)
            price: Limit/Stop price
            leverage: Position leverage (default: 10x)
            reduce_only: Close position only

        Returns:
            BinanceFuturesOrderResult
        """
        if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
            return BinanceFuturesOrderResult(
                symbol=symbol,
                order_id="",
                status="REJECTED",
                message="Live trading disabled. Set MASP_ENABLE_LIVE_TRADING=1",
            )

        try:
            binance_symbol = self._convert_symbol(symbol)

            # Set leverage
            actual_leverage = self._ensure_leverage(symbol, leverage)

            # Calculate quantity from USDT amount if needed
            quantity = units
            if amount_krw and not units:
                # Get current price for calculation
                ticker = self.api.get_ticker(binance_symbol)
                mark_price = float(ticker.get("bidPrice", 0))
                if mark_price > 0:
                    quantity = amount_krw / mark_price
                    logger.info(
                        "[BinanceFutures] Calculated qty: %.6f from %.2f USDT @ %.2f",
                        quantity,
                        amount_krw,
                        mark_price,
                    )

            if not quantity:
                return BinanceFuturesOrderResult(
                    symbol=symbol,
                    order_id="",
                    status="REJECTED",
                    message="Quantity or amount required",
                )

            result = self.api.place_order(
                symbol=binance_symbol,
                side=side.upper(),
                order_type=order_type.upper(),
                quantity=quantity,
                price=price,
                reduce_only=reduce_only,
            )

            # Parse response
            status_map = {
                "NEW": "OPEN",
                "PARTIALLY_FILLED": "PARTIAL",
                "FILLED": "FILLED",
                "CANCELED": "CANCELED",
                "REJECTED": "REJECTED",
                "EXPIRED": "EXPIRED",
            }

            return BinanceFuturesOrderResult(
                symbol=symbol,
                order_id=str(result.get("orderId", "")),
                client_order_id=result.get("clientOrderId"),
                status=status_map.get(result.get("status", ""), "UNKNOWN"),
                message=f"Order {result.get('status')}",
                executed_qty=float(result.get("executedQty", 0)),
                avg_price=(
                    float(result.get("avgPrice", 0)) if result.get("avgPrice") else 0
                ),
                reduce_only=reduce_only,
            )

        except Exception as e:
            logger.error("[BinanceFutures] Order failed: %s", e)
            return BinanceFuturesOrderResult(
                symbol=symbol,
                order_id="",
                status="REJECTED",
                message=str(e),
            )

    def close_position(self, symbol: str) -> BinanceFuturesOrderResult:
        """
        Close an existing position.

        Args:
            symbol: Trading pair

        Returns:
            BinanceFuturesOrderResult
        """
        position = self.get_position(symbol)
        if not position or position.get("position_amt", 0) == 0:
            return BinanceFuturesOrderResult(
                symbol=symbol,
                order_id="",
                status="REJECTED",
                message="No position to close",
            )

        amt = position["position_amt"]
        side = "SELL" if amt > 0 else "BUY"
        quantity = abs(amt)

        return self.place_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            units=quantity,
            reduce_only=True,
        )

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a symbol."""
        try:
            binance_symbol = self._convert_symbol(symbol)
            return self.api.get_position(binance_symbol)
        except Exception as e:
            logger.error("[BinanceFutures] Get position failed: %s", e)
            return None

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            positions = self.api.get_positions()
            return [
                {
                    "symbol": self._to_standard_symbol(p["symbol"]),
                    "position_amt": float(p["positionAmt"]),
                    "entry_price": float(p["entryPrice"]),
                    "mark_price": float(p["markPrice"]),
                    "unrealized_pnl": float(p["unRealizedProfit"]),
                    "leverage": int(p.get("leverage", 1)),
                }
                for p in positions
                if float(p["positionAmt"]) != 0
            ]
        except Exception as e:
            logger.error("[BinanceFutures] Get all positions failed: %s", e)
            return []

    def _to_standard_symbol(self, binance_symbol: str) -> str:
        """Convert 'BTCUSDT' -> 'BTC/USDT:PERP'."""
        for quote in ["USDT", "BUSD"]:
            if binance_symbol.endswith(quote):
                base = binance_symbol[: -len(quote)]
                return f"{base}/{quote}:PERP"
        return binance_symbol

    def get_order_status(self, order_id: str, symbol: str = "") -> Optional[Dict]:
        """Get order status."""
        try:
            if not symbol:
                logger.error("[BinanceFutures] Symbol required for get_order_status")
                return None

            binance_symbol = self._convert_symbol(symbol)
            result = self.api.get_order(binance_symbol, order_id=int(order_id))

            return {
                "order_id": str(result.get("orderId")),
                "symbol": symbol,
                "status": result.get("status"),
                "side": result.get("side"),
                "type": result.get("type"),
                "quantity": float(result.get("origQty", 0)),
                "executed_qty": float(result.get("executedQty", 0)),
                "price": float(result.get("price", 0)),
                "avg_price": (
                    float(result.get("avgPrice", 0)) if result.get("avgPrice") else 0
                ),
            }
        except Exception as e:
            logger.error("[BinanceFutures] Get order status failed: %s", e)
            return None

    def cancel_order(self, order_id: str, symbol: str = "") -> bool:
        """Cancel an order."""
        try:
            if not symbol:
                logger.error("[BinanceFutures] Symbol required for cancel_order")
                return False

            binance_symbol = self._convert_symbol(symbol)
            self.api.cancel_order(binance_symbol, order_id=int(order_id))
            return True
        except Exception as e:
            logger.error("[BinanceFutures] Cancel order failed: %s", e)
            return False

    def get_balance(self, asset: str = "USDT") -> Optional[float]:
        """Get balance for an asset."""
        try:
            balance = self.api.get_balance(asset.upper())
            if balance:
                return balance.get("available", 0.0)
            return 0.0
        except Exception as e:
            logger.error("[BinanceFutures] Get balance failed: %s", e)
            return None

    def get_all_balances(self) -> Dict[str, float]:
        """Get all non-zero balances."""
        try:
            balances = self.api.get_balances()
            return {asset: info["available"] for asset, info in balances.items()}
        except Exception as e:
            logger.error("[BinanceFutures] Get all balances failed: %s", e)
            return {}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        try:
            binance_symbol = self._convert_symbol(symbol) if symbol else None
            orders = self.api.get_open_orders(binance_symbol)
            return [
                {
                    "order_id": str(o.get("orderId")),
                    "symbol": self._to_standard_symbol(o.get("symbol", "")),
                    "side": o.get("side"),
                    "type": o.get("type"),
                    "quantity": float(o.get("origQty", 0)),
                    "price": float(o.get("price", 0)),
                    "status": o.get("status"),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error("[BinanceFutures] Get open orders failed: %s", e)
            return []

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            binance_symbol = self._convert_symbol(symbol)
            self.api.set_leverage(binance_symbol, leverage)
            self._leverage_cache[binance_symbol] = leverage
            return True
        except Exception as e:
            logger.error("[BinanceFutures] Set leverage failed: %s", e)
            return False

    def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Set margin type.

        Args:
            symbol: Trading pair
            margin_type: "ISOLATED" or "CROSSED"
        """
        try:
            binance_symbol = self._convert_symbol(symbol)
            self.api.set_margin_type(binance_symbol, margin_type)
            return True
        except Exception as e:
            # Already set to the same margin type is not an error
            if "No need to change margin type" in str(e):
                return True
            logger.error("[BinanceFutures] Set margin type failed: %s", e)
            return False
