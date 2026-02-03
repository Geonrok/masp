"""
Binance Spot Market Data & Execution Adapter

Phase 9: Full spot trading support.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from libs.adapters.base import (
    ExecutionAdapter,
    MarketDataAdapter,
    MarketQuote,
    OHLCV,
    OrderResult,
)
from libs.adapters.binance_api import BinanceAPI, BinanceMarket

logger = logging.getLogger(__name__)


class BinanceSpotMarketData(MarketDataAdapter):
    """
    Binance Spot Market Data Adapter.

    Provides real-time market data from Binance Spot exchange.

    Symbol format: "BTC/USDT" -> "BTCUSDT"
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
            market=BinanceMarket.SPOT,
            testnet=testnet,
        )
        self.testnet = testnet
        logger.info(
            "[BinanceSpot] MarketData initialized (%s)",
            "testnet" if testnet else "live",
        )

    def _convert_symbol(self, symbol: str) -> str:
        """Convert 'BTC/USDT' -> 'BTCUSDT'."""
        return symbol.replace("/", "")

    def _reverse_symbol(self, binance_symbol: str) -> str:
        """Convert 'BTCUSDT' -> 'BTC/USDT' (best effort)."""
        # Common quote assets
        for quote in ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"]:
            if binance_symbol.endswith(quote):
                base = binance_symbol[: -len(quote)]
                return f"{base}/{quote}"
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
            logger.error("[BinanceSpot] Failed to get quote for %s: %s", symbol, e)
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
            logger.error("[BinanceSpot] Failed to get OHLCV for %s: %s", symbol, e)
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
            logger.error("[BinanceSpot] Failed to get orderbook for %s: %s", symbol, e)
            return None

    def is_market_open(self) -> bool:
        """Binance Spot operates 24/7."""
        return True

    def get_all_symbols(self) -> List[str]:
        """Get all available USDT pairs."""
        try:
            info = self.api.get_exchange_info()
            symbols = []
            for s in info.get("symbols", []):
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                    symbols.append(f"{s['baseAsset']}/{s['quoteAsset']}")
            return symbols
        except Exception as e:
            logger.error("[BinanceSpot] Failed to get symbols: %s", e)
            return []


@dataclass
class BinanceOrderResult(OrderResult):
    """Extended order result for Binance."""

    client_order_id: Optional[str] = None
    executed_qty: float = 0.0
    cummulative_quote_qty: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""


class BinanceSpotExecution(ExecutionAdapter):
    """
    Binance Spot Execution Adapter.

    Supports real order execution on Binance Spot.
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
            market=BinanceMarket.SPOT,
            testnet=testnet,
        )
        self.testnet = testnet
        self._positions: Dict[str, Dict] = {}

        # Verify credentials
        if not self.api.api_key or not self.api.api_secret:
            logger.warning(
                "[BinanceSpot] API credentials not provided. "
                "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
            )

        # Sync time
        self.api.sync_time()

        logger.info(
            "[BinanceSpot] Execution initialized (%s)", "testnet" if testnet else "live"
        )

    def _convert_symbol(self, symbol: str) -> str:
        """Convert 'BTC/USDT' -> 'BTCUSDT'."""
        return symbol.replace("/", "")

    def _base_asset(self, symbol: str) -> str:
        """Extract base asset: 'BTC/USDT' -> 'BTC'."""
        if "/" in symbol:
            return symbol.split("/")[0]
        return symbol

    def _quote_asset(self, symbol: str) -> str:
        """Extract quote asset: 'BTC/USDT' -> 'USDT'."""
        if "/" in symbol:
            return symbol.split("/")[1]
        return "USDT"

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,  # For USDT amount
        price: Optional[float] = None,
        **kwargs,
    ) -> BinanceOrderResult:
        """
        Place an order.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            units: Quantity in base asset
            amount_krw: Amount in quote asset (USDT) - for market buys
            price: Limit price

        Returns:
            BinanceOrderResult
        """
        if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
            return BinanceOrderResult(
                symbol=symbol,
                order_id="",
                status="REJECTED",
                message="Live trading disabled. Set MASP_ENABLE_LIVE_TRADING=1",
            )

        try:
            binance_symbol = self._convert_symbol(symbol)

            result = self.api.place_order(
                symbol=binance_symbol,
                side=side.upper(),
                order_type=order_type.upper(),
                quantity=units,
                quote_quantity=amount_krw,  # USDT amount for spot
                price=price,
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

            fills = result.get("fills", [])
            total_commission = sum(float(f.get("commission", 0)) for f in fills)
            commission_asset = fills[0].get("commissionAsset", "") if fills else ""

            return BinanceOrderResult(
                symbol=symbol,
                order_id=str(result.get("orderId", "")),
                client_order_id=result.get("clientOrderId"),
                status=status_map.get(result.get("status", ""), "UNKNOWN"),
                message=f"Order {result.get('status')}",
                executed_qty=float(result.get("executedQty", 0)),
                cummulative_quote_qty=float(result.get("cummulativeQuoteQty", 0)),
                avg_price=(
                    float(result.get("avgPrice", 0)) if result.get("avgPrice") else 0
                ),
                commission=total_commission,
                commission_asset=commission_asset,
            )

        except Exception as e:
            logger.error("[BinanceSpot] Order failed: %s", e)
            return BinanceOrderResult(
                symbol=symbol,
                order_id="",
                status="REJECTED",
                message=str(e),
            )

    def get_order_status(self, order_id: str, symbol: str = "") -> Optional[Dict]:
        """Get order status."""
        try:
            if not symbol:
                logger.error("[BinanceSpot] Symbol required for get_order_status")
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
            logger.error("[BinanceSpot] Get order status failed: %s", e)
            return None

    def cancel_order(self, order_id: str, symbol: str = "") -> bool:
        """Cancel an order."""
        try:
            if not symbol:
                logger.error("[BinanceSpot] Symbol required for cancel_order")
                return False

            binance_symbol = self._convert_symbol(symbol)
            self.api.cancel_order(binance_symbol, order_id=int(order_id))
            return True
        except Exception as e:
            logger.error("[BinanceSpot] Cancel order failed: %s", e)
            return False

    def get_balance(self, asset: str) -> Optional[float]:
        """Get balance for an asset."""
        try:
            balance = self.api.get_balance(asset.upper())
            if balance:
                return balance.get("free", 0.0)
            return 0.0
        except Exception as e:
            logger.error("[BinanceSpot] Get balance failed: %s", e)
            return None

    def get_all_balances(self) -> Dict[str, float]:
        """Get all non-zero balances."""
        try:
            balances = self.api.get_balances()
            return {asset: info["free"] for asset, info in balances.items()}
        except Exception as e:
            logger.error("[BinanceSpot] Get all balances failed: %s", e)
            return {}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        try:
            binance_symbol = self._convert_symbol(symbol) if symbol else None
            orders = self.api.get_open_orders(binance_symbol)
            return [
                {
                    "order_id": str(o.get("orderId")),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "type": o.get("type"),
                    "quantity": float(o.get("origQty", 0)),
                    "price": float(o.get("price", 0)),
                    "status": o.get("status"),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error("[BinanceSpot] Get open orders failed: %s", e)
            return []
