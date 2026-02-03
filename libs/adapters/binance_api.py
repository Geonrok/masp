"""
Binance API Client - Spot & Futures Unified API

Supports:
- Binance Spot (api.binance.com)
- Binance USDT-M Futures (fapi.binance.com)
- Binance COIN-M Futures (dapi.binance.com)

API Documentation:
- Spot: https://binance-docs.github.io/apidocs/spot/en/
- Futures: https://binance-docs.github.io/apidocs/futures/en/
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class BinanceMarket(Enum):
    """Binance market types."""

    SPOT = "spot"
    FUTURES_USDT = "futures_usdt"  # USDT-M Futures
    FUTURES_COIN = "futures_coin"  # COIN-M Futures


@dataclass
class BinanceEndpoints:
    """API endpoints for different markets."""

    base_url: str
    testnet_url: str
    ws_url: str

    # Common endpoints
    ticker: str
    orderbook: str
    klines: str
    exchange_info: str

    # Account endpoints
    account: str
    order: str
    open_orders: str
    all_orders: str

    # Position (futures only)
    position: Optional[str] = None


SPOT_ENDPOINTS = BinanceEndpoints(
    base_url="https://api.binance.com",
    testnet_url="https://testnet.binance.vision",
    ws_url="wss://stream.binance.com:9443",
    ticker="/api/v3/ticker/bookTicker",
    orderbook="/api/v3/depth",
    klines="/api/v3/klines",
    exchange_info="/api/v3/exchangeInfo",
    account="/api/v3/account",
    order="/api/v3/order",
    open_orders="/api/v3/openOrders",
    all_orders="/api/v3/allOrders",
)

FUTURES_USDT_ENDPOINTS = BinanceEndpoints(
    base_url="https://fapi.binance.com",
    testnet_url="https://testnet.binancefuture.com",
    ws_url="wss://fstream.binance.com",
    ticker="/fapi/v1/ticker/bookTicker",
    orderbook="/fapi/v1/depth",
    klines="/fapi/v1/klines",
    exchange_info="/fapi/v1/exchangeInfo",
    account="/fapi/v2/account",
    order="/fapi/v1/order",
    open_orders="/fapi/v1/openOrders",
    all_orders="/fapi/v1/allOrders",
    position="/fapi/v2/positionRisk",
)

FUTURES_COIN_ENDPOINTS = BinanceEndpoints(
    base_url="https://dapi.binance.com",
    testnet_url="https://testnet.binancefuture.com",
    ws_url="wss://dstream.binance.com",
    ticker="/dapi/v1/ticker/bookTicker",
    orderbook="/dapi/v1/depth",
    klines="/dapi/v1/klines",
    exchange_info="/dapi/v1/exchangeInfo",
    account="/dapi/v1/account",
    order="/dapi/v1/order",
    open_orders="/dapi/v1/openOrders",
    all_orders="/dapi/v1/allOrders",
    position="/dapi/v1/positionRisk",
)


def get_endpoints(market: BinanceMarket) -> BinanceEndpoints:
    """Get endpoints for a specific market."""
    if market == BinanceMarket.SPOT:
        return SPOT_ENDPOINTS
    elif market == BinanceMarket.FUTURES_USDT:
        return FUTURES_USDT_ENDPOINTS
    elif market == BinanceMarket.FUTURES_COIN:
        return FUTURES_COIN_ENDPOINTS
    else:
        raise ValueError(f"Unknown market: {market}")


class BinanceAPI:
    """
    Unified Binance API client for Spot and Futures.

    Features:
    - HMAC-SHA256 request signing
    - Automatic timestamp synchronization
    - Rate limit handling
    - Testnet support

    Usage:
        # Spot
        api = BinanceAPI(api_key, api_secret, market=BinanceMarket.SPOT)

        # USDT-M Futures
        api = BinanceAPI(api_key, api_secret, market=BinanceMarket.FUTURES_USDT)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        market: BinanceMarket = BinanceMarket.FUTURES_USDT,
        testnet: bool = False,
        timeout: int = 10,
    ):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.market = market
        self.testnet = testnet
        self.timeout = timeout

        self.endpoints = get_endpoints(market)
        self.base_url = (
            self.endpoints.testnet_url if testnet else self.endpoints.base_url
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "MASP/1.0",
            }
        )
        if self.api_key:
            self.session.headers["X-MBX-APIKEY"] = self.api_key

        # Server time offset for timestamp sync
        self._time_offset = 0

        logger.info(
            "[BinanceAPI] Initialized: market=%s, testnet=%s, base_url=%s",
            market.value,
            testnet,
            self.base_url,
        )

    def _get_timestamp(self) -> int:
        """Get current timestamp with server offset."""
        return int(time.time() * 1000) + self._time_offset

    def _sign(self, params: Dict[str, Any]) -> str:
        """Create HMAC-SHA256 signature."""
        if not self.api_secret:
            raise ValueError("API secret required for signed requests")

        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> Any:
        """Make API request."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            if not self.api_key:
                raise ValueError("API key required for signed requests")
            params["timestamp"] = self._get_timestamp()
            params["signature"] = self._sign(params)

        try:
            if method == "GET":
                resp = self.session.get(url, params=params, timeout=self.timeout)
            elif method == "POST":
                resp = self.session.post(url, params=params, timeout=self.timeout)
            elif method == "DELETE":
                resp = self.session.delete(url, params=params, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Handle rate limits
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning(
                    "[BinanceAPI] Rate limited, retry after %ds", retry_after
                )
                raise Exception(f"Rate limited, retry after {retry_after}s")

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            error_msg = self._extract_error(e.response)
            logger.error(
                "[BinanceAPI] HTTP Error: %s - %s", e.response.status_code, error_msg
            )
            raise Exception(f"Binance API Error: {error_msg}") from e
        except requests.exceptions.RequestException as e:
            logger.error("[BinanceAPI] Request failed: %s", e)
            raise ConnectionError(f"Network error: {e}") from e

    @staticmethod
    def _extract_error(response) -> str:
        """Extract error message from response."""
        try:
            data = response.json()
            code = data.get("code", "")
            msg = data.get("msg", "")
            return f"[{code}] {msg}" if code else str(data)
        except Exception:
            return response.text[:200] if response.text else "Unknown error"

    def sync_time(self) -> int:
        """
        Synchronize with server time.

        Returns:
            Time offset in milliseconds
        """
        try:
            if self.market == BinanceMarket.SPOT:
                endpoint = "/api/v3/time"
            else:
                endpoint = "/fapi/v1/time"

            local_time = int(time.time() * 1000)
            data = self._request("GET", endpoint)
            server_time = data["serverTime"]
            self._time_offset = server_time - local_time

            logger.info("[BinanceAPI] Time synced, offset: %dms", self._time_offset)
            return self._time_offset
        except Exception as e:
            logger.warning("[BinanceAPI] Time sync failed: %s", e)
            return 0

    # =========================================================================
    # Public API
    # =========================================================================

    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker (best bid/ask)."""
        params = {"symbol": symbol}
        return self._request("GET", self.endpoints.ticker, params)

    def get_all_tickers(self) -> List[Dict]:
        """Get all tickers."""
        return self._request("GET", self.endpoints.ticker)

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book."""
        params = {"symbol": symbol, "limit": limit}
        return self._request("GET", self.endpoints.orderbook, params)

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        """
        Get klines/candlesticks.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
            limit: Number of klines (max 1500)
            start_time: Start time in ms
            end_time: End time in ms

        Returns:
            List of klines: [open_time, open, high, low, close, volume, ...]
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._request("GET", self.endpoints.klines, params)

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Get exchange information."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", self.endpoints.exchange_info, params)

    # =========================================================================
    # Account API (Signed)
    # =========================================================================

    def get_account(self) -> Dict:
        """Get account information."""
        return self._request("GET", self.endpoints.account, signed=True)

    def get_balances(self) -> Dict[str, Dict]:
        """
        Get all balances.

        Returns:
            Dict of asset to balance info
        """
        account = self.get_account()

        if self.market == BinanceMarket.SPOT:
            balances = account.get("balances", [])
            return {
                b["asset"]: {
                    "free": float(b["free"]),
                    "locked": float(b["locked"]),
                    "total": float(b["free"]) + float(b["locked"]),
                }
                for b in balances
                if float(b["free"]) > 0 or float(b["locked"]) > 0
            }
        else:
            # Futures
            assets = account.get("assets", [])
            return {
                a["asset"]: {
                    "wallet_balance": float(a["walletBalance"]),
                    "unrealized_pnl": float(a["unrealizedProfit"]),
                    "margin_balance": float(a["marginBalance"]),
                    "available": float(a["availableBalance"]),
                }
                for a in assets
                if float(a["walletBalance"]) > 0
            }

    def get_balance(self, asset: str) -> Optional[Dict]:
        """Get balance for a specific asset."""
        balances = self.get_balances()
        return balances.get(asset)

    # =========================================================================
    # Order API (Signed)
    # =========================================================================

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Optional[float] = None,
        quote_quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        position_side: str = "BOTH",
    ) -> Dict:
        """
        Place an order.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "BUY" or "SELL"
            order_type: "MARKET", "LIMIT", "STOP_MARKET", etc.
            quantity: Order quantity
            quote_quantity: Quote asset quantity (for MARKET orders)
            price: Limit price
            time_in_force: "GTC", "IOC", "FOK"
            reduce_only: Reduce position only (futures)
            position_side: "BOTH", "LONG", "SHORT" (futures hedge mode)

        Returns:
            Order response
        """
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
        }

        if quantity:
            params["quantity"] = f"{quantity:.8f}".rstrip("0").rstrip(".")

        if quote_quantity and self.market == BinanceMarket.SPOT:
            params["quoteOrderQty"] = f"{quote_quantity:.8f}".rstrip("0").rstrip(".")

        if price and order_type != "MARKET":
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = time_in_force

        # Futures-specific
        if self.market != BinanceMarket.SPOT:
            if reduce_only:
                params["reduceOnly"] = "true"
            params["positionSide"] = position_side

        return self._request("POST", self.endpoints.order, params, signed=True)

    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """Get order status."""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._request("GET", self.endpoints.order, params, signed=True)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """Cancel an order."""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._request("DELETE", self.endpoints.order, params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", self.endpoints.open_orders, params, signed=True)

    def get_all_orders(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get all orders (open and closed)."""
        params = {"symbol": symbol, "limit": limit}
        return self._request("GET", self.endpoints.all_orders, params, signed=True)

    # =========================================================================
    # Futures-specific API
    # =========================================================================

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get position information (futures only)."""
        if self.market == BinanceMarket.SPOT:
            raise ValueError("Positions not available for spot market")

        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", self.endpoints.position, params, signed=True)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol."""
        positions = self.get_positions(symbol)
        for pos in positions:
            if pos["symbol"] == symbol:
                return {
                    "symbol": pos["symbol"],
                    "position_amt": float(pos["positionAmt"]),
                    "entry_price": float(pos["entryPrice"]),
                    "mark_price": float(pos["markPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "liquidation_price": float(pos.get("liquidationPrice", 0)),
                    "leverage": int(pos.get("leverage", 1)),
                    "margin_type": pos.get("marginType", "cross"),
                }
        return None

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol (futures only)."""
        if self.market == BinanceMarket.SPOT:
            raise ValueError("Leverage not available for spot market")

        params = {"symbol": symbol, "leverage": leverage}
        endpoint = (
            "/fapi/v1/leverage"
            if self.market == BinanceMarket.FUTURES_USDT
            else "/dapi/v1/leverage"
        )
        return self._request("POST", endpoint, params, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        Set margin type (ISOLATED or CROSSED).

        Args:
            symbol: Trading pair
            margin_type: "ISOLATED" or "CROSSED"
        """
        if self.market == BinanceMarket.SPOT:
            raise ValueError("Margin type not available for spot market")

        params = {"symbol": symbol, "marginType": margin_type.upper()}
        endpoint = (
            "/fapi/v1/marginType"
            if self.market == BinanceMarket.FUTURES_USDT
            else "/dapi/v1/marginType"
        )
        return self._request("POST", endpoint, params, signed=True)
