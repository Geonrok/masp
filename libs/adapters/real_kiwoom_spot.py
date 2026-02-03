"""
Kiwoom REST API Spot Market Adapter

Provides market data and execution for KOSPI/KOSDAQ stocks via Kiwoom REST API.
Supports fractional share trading for small capital portfolios.

API Documentation: https://apiportal.koreainvestment.com (Kiwoom uses similar OpenAPI structure)

Environment Variables:
    KIWOOM_APP_KEY: API App Key
    KIWOOM_APP_SECRET: API App Secret
    KIWOOM_ACCOUNT_NO: Trading account number
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

from libs.adapters.base import (
    ExecutionAdapter,
    MarketDataAdapter,
    MarketQuote,
    OrderResult,
)

logger = logging.getLogger(__name__)


# Kiwoom fractional trading available stocks (as of 2025)
# Source: Kiwoom Securities fractional trading service
FRACTIONAL_TRADING_SYMBOLS = {
    # Large-cap KOSPI stocks
    "005930",
    "000660",
    "005380",
    "005490",
    "035420",
    "000270",
    "051910",
    "006400",
    "035720",
    "028260",
    "068270",
    "105560",
    "055550",
    "034730",
    "003550",
    "015760",
    "017670",
    "096770",
    "066570",
    "032830",
    "012330",
    "034220",
    "036570",
    "003490",
    "018260",
    "086790",
    "009150",
    "010950",
    "003670",
    "042700",
    # Additional popular stocks
    "033780",
    "011170",
    "024110",
    "004020",
    "009540",
    "018880",
    "030200",
    "021240",
    "000810",
    "004990",
}


@dataclass
class OHLCVCandle:
    """OHLCV candlestick data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class KiwoomSpotMarketData(MarketDataAdapter):
    """
    Kiwoom REST API Market Data Adapter

    Provides:
        - get_quote(symbol): Current price query
        - get_quotes(symbols): Multiple symbols query
        - get_ohlcv(symbol, interval, limit): Historical OHLCV data
        - is_market_open(): Market hours check

    Authentication:
        Uses KIWOOM_APP_KEY and KIWOOM_APP_SECRET environment variables.
        OAuth2 client_credentials flow.
    """

    # Korean stock market hours (KST)
    MARKET_OPEN = dt_time(9, 0)
    MARKET_CLOSE = dt_time(15, 30)

    # API endpoints
    BASE_URL = "https://openapi.kiwoom.com:8080"

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
    ):
        """
        Initialize Kiwoom market data adapter.

        Args:
            app_key: Kiwoom API app key (or KIWOOM_APP_KEY env)
            app_secret: Kiwoom API app secret (or KIWOOM_APP_SECRET env)
        """
        self._app_key = app_key or os.getenv("KIWOOM_APP_KEY", "")
        self._app_secret = app_secret or os.getenv("KIWOOM_APP_SECRET", "")
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._session: Optional[Any] = None

        if not self._app_key or not self._app_secret:
            logger.warning(
                "[Kiwoom] API credentials not provided. "
                "Set KIWOOM_APP_KEY and KIWOOM_APP_SECRET environment variables."
            )

        logger.info("[Kiwoom] MarketData adapter initialized")

    def _run_async(self, coro):
        """Run async coroutine synchronously.

        Handles the case where we may or may not already be in an async context.
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                self._session = None
                self._access_token = None
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            # No running loop - safe to use asyncio.run directly
            return asyncio.run(coro)

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    async def _ensure_token(self) -> bool:
        """Ensure we have a valid access token."""
        # Check if token is still valid
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires:
                return True

        if not self._app_key or not self._app_secret:
            logger.error("[Kiwoom] Cannot authenticate: missing credentials")
            return False

        try:
            session = await self._get_session()

            # OAuth2 token request
            url = f"{self.BASE_URL}/oauth2/token"
            data = {
                "grant_type": "client_credentials",
                "appkey": self._app_key,
                "secretkey": self._app_secret,
            }

            async with session.post(url, json=data) as resp:
                if resp.status != 200:
                    logger.error(f"[Kiwoom] Token request failed: {resp.status}")
                    return False

                result = await resp.json()
                self._access_token = result.get("access_token")
                expires_in = result.get("expires_in", 86400)  # Default 24h

                if self._access_token:
                    from datetime import timedelta

                    self._token_expires = datetime.now() + timedelta(
                        seconds=expires_in - 60
                    )
                    logger.info("[Kiwoom] Token acquired successfully")
                    return True
                else:
                    logger.error("[Kiwoom] No access token in response")
                    return False

        except Exception as e:
            logger.error(f"[Kiwoom] Token acquisition error: {e}")
            return False

    def _get_headers(self) -> dict:
        """Get API request headers."""
        return {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
        }

    async def _get_quote_async(self, symbol: str) -> Optional[MarketQuote]:
        """Get quote for a symbol (async)."""
        if not await self._ensure_token():
            return None

        try:
            session = await self._get_session()

            # Stock current price API (similar to ka10001)
            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
            headers = self._get_headers()
            headers["tr_id"] = "FHKST01010100"  # Current price TR

            params = {
                "FID_COND_MRKT_DIV_CODE": "J",  # KOSPI/KOSDAQ
                "FID_INPUT_ISCD": symbol,
            }

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"[Kiwoom] Quote request failed for {symbol}: {resp.status}"
                    )
                    return None

                result = await resp.json()
                output = result.get("output", {})

                if not output:
                    logger.warning(f"[Kiwoom] No data for {symbol}")
                    return None

                price = float(output.get("stck_prpr", 0))  # Current price
                bid = float(output.get("stck_hgpr", price))  # High (approx bid)
                ask = float(output.get("stck_lwpr", price))  # Low (approx ask)
                volume = float(output.get("acml_vol", 0))  # Accumulated volume

                return MarketQuote(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    last=price,
                    volume_24h=volume,
                    timestamp=datetime.now().isoformat(),
                )

        except Exception as e:
            logger.error(f"[Kiwoom] Error getting quote for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock code (6-digit, e.g., "005930")

        Returns:
            MarketQuote or None if unavailable
        """
        return self._run_async(self._get_quote_async(symbol))

    def get_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of stock codes

        Returns:
            Dict mapping symbol to MarketQuote
        """
        result = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                result[symbol] = quote
        return result

    async def _get_ohlcv_async(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 100,
    ) -> List[OHLCVCandle]:
        """Get OHLCV data (async)."""
        if not await self._ensure_token():
            return []

        try:
            session = await self._get_session()

            # Daily price API
            url = (
                f"{self.BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
            )
            headers = self._get_headers()
            headers["tr_id"] = "FHKST01010400"  # Daily price TR

            # Period code mapping
            period_map = {"1d": "D", "1w": "W", "1M": "M"}
            if interval not in period_map:
                raise ValueError(
                    f"[Kiwoom] Unsupported interval '{interval}'. Supported: {list(period_map.keys())}"
                )

            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_PERIOD_DIV_CODE": period_map[interval],
                "FID_ORG_ADJ_PRC": "0",  # Adjusted price
            }

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"[Kiwoom] OHLCV request failed for {symbol}: {resp.status}"
                    )
                    return []

                result = await resp.json()
                output_list = result.get("output", [])

                if not output_list:
                    logger.warning(f"[Kiwoom] No OHLCV data for {symbol}")
                    return []

                candles = []
                for item in output_list[:limit]:
                    try:
                        date_str = item.get("stck_bsop_date", "")
                        if date_str:
                            ts = datetime.strptime(date_str, "%Y%m%d")
                        else:
                            continue

                        candles.append(
                            OHLCVCandle(
                                timestamp=ts,
                                open=float(item.get("stck_oprc", 0)),
                                high=float(item.get("stck_hgpr", 0)),
                                low=float(item.get("stck_lwpr", 0)),
                                close=float(item.get("stck_clpr", 0)),
                                volume=float(item.get("acml_vol", 0)),
                            )
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning(f"[Kiwoom] Failed to parse candle: {e}")
                        continue

                # Return oldest first
                candles.reverse()
                logger.debug(f"[Kiwoom] get_ohlcv: {symbol} - {len(candles)} candles")
                return candles

        except Exception as e:
            logger.error(f"[Kiwoom] Error getting OHLCV for {symbol}: {e}")
            return []

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 100,
    ) -> List[OHLCVCandle]:
        """
        Get OHLCV candle data.

        Args:
            symbol: Stock code (6-digit)
            interval: "1d" (daily), "1w" (weekly), "1M" (monthly)
            limit: Number of candles

        Returns:
            List of OHLCVCandle (oldest first)
        """
        return self._run_async(self._get_ohlcv_async(symbol, interval, limit))

    def is_market_open(self) -> bool:
        """
        Check if Korean stock market is open.

        Market hours: 09:00 - 15:30 KST, weekdays only.

        Returns:
            True if market is open
        """
        now = datetime.now(ZoneInfo("Asia/Seoul"))

        # Check weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        current_time = now.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def is_fractional_available(self, symbol: str) -> bool:
        """Check if fractional trading is available for symbol."""
        return symbol in FRACTIONAL_TRADING_SYMBOLS

    async def _close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None


class KiwoomSpotExecution(ExecutionAdapter):
    """
    Kiwoom REST API Execution Adapter

    Provides:
        - place_order(): Place buy/sell orders (supports fractional)
        - get_order_status(): Check order status
        - cancel_order(): Cancel pending orders
        - get_balance(): Get account balance

    Fractional Trading:
        - Minimum order amount: 1,000 KRW
        - Supports fractional shares for ~370 stocks
        - Orders are executed at market close single price auction
    """

    BASE_URL = "https://openapi.kiwoom.com:8080"

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        account_no: Optional[str] = None,
    ):
        """
        Initialize Kiwoom execution adapter.

        Args:
            app_key: Kiwoom API app key (or KIWOOM_APP_KEY env)
            app_secret: Kiwoom API app secret (or KIWOOM_APP_SECRET env)
            account_no: Trading account number (or KIWOOM_ACCOUNT_NO env)
        """
        self._app_key = app_key or os.getenv("KIWOOM_APP_KEY", "")
        self._app_secret = app_secret or os.getenv("KIWOOM_APP_SECRET", "")
        self._account_no = account_no or os.getenv("KIWOOM_ACCOUNT_NO", "")
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._session: Optional[Any] = None
        self._trade_logger = None

        if not self._app_key or not self._app_secret:
            logger.warning(
                "[Kiwoom] API credentials not provided. "
                "Set KIWOOM_APP_KEY and KIWOOM_APP_SECRET environment variables."
            )

        if not self._account_no:
            logger.warning(
                "[Kiwoom] Account number not provided. "
                "Set KIWOOM_ACCOUNT_NO environment variable for trading."
            )

        logger.info("[Kiwoom] Execution adapter initialized")

    def set_trade_logger(self, trade_logger):
        """Set trade logger for recording trades."""
        self._trade_logger = trade_logger

    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                self._session = None
                self._access_token = None
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(coro)

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    async def _ensure_token(self) -> bool:
        """Ensure we have a valid access token."""
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires:
                return True

        if not self._app_key or not self._app_secret:
            logger.error("[Kiwoom] Cannot authenticate: missing credentials")
            return False

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/oauth2/token"
            data = {
                "grant_type": "client_credentials",
                "appkey": self._app_key,
                "secretkey": self._app_secret,
            }

            async with session.post(url, json=data) as resp:
                if resp.status != 200:
                    logger.error(f"[Kiwoom] Token request failed: {resp.status}")
                    return False

                result = await resp.json()
                self._access_token = result.get("access_token")
                expires_in = result.get("expires_in", 86400)

                if self._access_token:
                    from datetime import timedelta

                    self._token_expires = datetime.now() + timedelta(
                        seconds=expires_in - 60
                    )
                    logger.info("[Kiwoom] Execution token acquired")
                    return True
                return False

        except Exception as e:
            logger.error(f"[Kiwoom] Token acquisition error: {e}")
            return False

    def _get_headers(self, tr_id: str) -> dict:
        """Get API request headers."""
        return {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
            "tr_id": tr_id,
        }

    async def _place_order_async(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """Place an order (async)."""
        if not await self._ensure_token():
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                quantity=quantity,
                message="Authentication failed",
                mock=False,
            )

        if not self._account_no:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                quantity=quantity,
                message="Account number not configured",
                mock=False,
            )

        try:
            session = await self._get_session()

            # Determine if fractional order
            is_fractional = quantity != int(quantity) or (
                quantity < 1 and symbol in FRACTIONAL_TRADING_SYMBOLS
            )

            if is_fractional and symbol not in FRACTIONAL_TRADING_SYMBOLS:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    message=f"Fractional trading not available for {symbol}",
                    mock=False,
                )

            # Order TR ID
            if side.upper() == "BUY":
                tr_id = (
                    "TTTC0802U" if not is_fractional else "TTTC0852U"
                )  # Fractional buy
            else:
                tr_id = (
                    "TTTC0801U" if not is_fractional else "TTTC0851U"
                )  # Fractional sell

            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
            headers = self._get_headers(tr_id)

            # Order type code
            ord_dvsn = "01" if order_type == "MARKET" else "00"  # 00=limit, 01=market

            body = {
                "CANO": self._account_no[:8],  # Account number (first 8 digits)
                "ACNT_PRDT_CD": (
                    self._account_no[8:10] if len(self._account_no) > 8 else "01"
                ),
                "PDNO": symbol,
                "ORD_DVSN": ord_dvsn,
                "ORD_QTY": str(int(quantity)) if not is_fractional else "0",
                "ORD_UNPR": str(int(price)) if price else "0",
            }

            # For fractional orders, use amount-based ordering
            if is_fractional:
                body["ORD_QTY"] = "0"
                # Calculate order amount based on current price
                if price:
                    body["ORD_AMT"] = str(int(quantity * price))

            async with session.post(url, headers=headers, json=body) as resp:
                result = await resp.json()

                if resp.status == 200 and result.get("rt_cd") == "0":
                    order_id = result.get("output", {}).get("ODNO", "")
                    logger.info(
                        f"[Kiwoom] Order placed: {side} {quantity} {symbol} -> {order_id}"
                    )

                    order_result = OrderResult(
                        success=True,
                        order_id=order_id,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        status="submitted",
                        message="Order submitted successfully",
                        mock=False,
                    )

                    if self._trade_logger:
                        self._trade_logger.log_order(order_result)

                    return order_result
                else:
                    error_msg = result.get("msg1", "Unknown error")
                    logger.error(f"[Kiwoom] Order failed: {error_msg}")
                    return OrderResult(
                        success=False,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        status="rejected",
                        message=error_msg,
                        mock=False,
                    )

        except Exception as e:
            logger.error(f"[Kiwoom] Order error: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                quantity=quantity,
                message=str(e),
                mock=False,
            )

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place an order.

        Args:
            symbol: Stock code (6-digit)
            side: "BUY" or "SELL"
            quantity: Order quantity (supports fractional for eligible stocks)
            order_type: "MARKET" or "LIMIT"
            price: Limit price (required for LIMIT orders)

        Returns:
            OrderResult with execution details
        """
        return self._run_async(
            self._place_order_async(symbol, side, quantity, order_type, price)
        )

    async def _get_order_status_async(self, order_id: str) -> Optional[dict]:
        """Get order status (async)."""
        if not await self._ensure_token():
            return None

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
            headers = self._get_headers("TTTC8001R")

            params = {
                "CANO": self._account_no[:8],
                "ACNT_PRDT_CD": (
                    self._account_no[8:10] if len(self._account_no) > 8 else "01"
                ),
                "INQR_STRT_DT": datetime.now().strftime("%Y%m%d"),
                "INQR_END_DT": datetime.now().strftime("%Y%m%d"),
                "SLL_BUY_DVSN_CD": "00",  # All
                "INQR_DVSN": "00",
                "PDNO": "",
                "CCLD_DVSN": "00",
                "ORD_GNO_BRNO": "",
                "ODNO": order_id,
                "INQR_DVSN_3": "00",
                "INQR_DVSN_1": "",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            }

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    return None

                result = await resp.json()
                output_list = result.get("output1", [])

                for order in output_list:
                    if order.get("odno") == order_id:
                        return {
                            "order_id": order_id,
                            "symbol": order.get("pdno"),
                            "side": (
                                "BUY"
                                if order.get("sll_buy_dvsn_cd") == "02"
                                else "SELL"
                            ),
                            "quantity": float(order.get("ord_qty", 0)),
                            "filled_quantity": float(order.get("tot_ccld_qty", 0)),
                            "price": float(order.get("ord_unpr", 0)),
                            "status": self._parse_order_status(
                                order.get("ord_dvsn_cd")
                            ),
                        }
                return None

        except Exception as e:
            logger.error(f"[Kiwoom] Get order status error: {e}")
            return None

    def _parse_order_status(self, status_code: str) -> str:
        """Parse order status code to string."""
        status_map = {
            "00": "submitted",
            "01": "filled",
            "02": "partially_filled",
            "03": "cancelled",
        }
        return status_map.get(status_code, "unknown")

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get status of an existing order.

        Args:
            order_id: Order identifier

        Returns:
            Order status dict or None if not found
        """
        return self._run_async(self._get_order_status_async(order_id))

    async def _cancel_order_async(self, order_id: str) -> bool:
        """Cancel an order (async)."""
        if not await self._ensure_token():
            return False

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/order-rvsecncl"
            headers = self._get_headers("TTTC0803U")

            body = {
                "CANO": self._account_no[:8],
                "ACNT_PRDT_CD": (
                    self._account_no[8:10] if len(self._account_no) > 8 else "01"
                ),
                "KRX_FWDG_ORD_ORGNO": "",
                "ORGN_ODNO": order_id,
                "ORD_DVSN": "00",
                "RVSE_CNCL_DVSN_CD": "02",  # 02 = cancel
                "ORD_QTY": "0",  # All remaining
                "ORD_UNPR": "0",
                "QTY_ALL_ORD_YN": "Y",
            }

            async with session.post(url, headers=headers, json=body) as resp:
                result = await resp.json()

                if resp.status == 200 and result.get("rt_cd") == "0":
                    logger.info(f"[Kiwoom] Order cancelled: {order_id}")
                    return True
                else:
                    error_msg = result.get("msg1", "Cancel failed")
                    logger.error(f"[Kiwoom] Cancel failed: {error_msg}")
                    return False

        except Exception as e:
            logger.error(f"[Kiwoom] Cancel order error: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order identifier

        Returns:
            True if cancelled successfully
        """
        return self._run_async(self._cancel_order_async(order_id))

    async def _get_balance_async(self, asset: str) -> Optional[float]:
        """Get balance (async)."""
        if not await self._ensure_token():
            return None

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
            headers = self._get_headers("TTTC8434R")

            params = {
                "CANO": self._account_no[:8],
                "ACNT_PRDT_CD": (
                    self._account_no[8:10] if len(self._account_no) > 8 else "01"
                ),
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            }

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    return None

                result = await resp.json()

                if asset.upper() == "KRW":
                    # Get available cash
                    output2 = result.get("output2", [{}])[0]
                    return float(output2.get("dnca_tot_amt", 0))
                else:
                    # Get stock holding
                    output1 = result.get("output1", [])
                    for holding in output1:
                        if holding.get("pdno") == asset:
                            return float(holding.get("hldg_qty", 0))
                    return 0.0

        except Exception as e:
            logger.error(f"[Kiwoom] Get balance error: {e}")
            return None

    def get_balance(self, asset: str) -> Optional[float]:
        """
        Get balance for an asset.

        Args:
            asset: "KRW" for cash, or stock code for holdings

        Returns:
            Available balance or None
        """
        return self._run_async(self._get_balance_async(asset))

    def get_all_balances(self) -> dict:
        """Get all account balances including cash and holdings."""
        return self._run_async(self._get_all_balances_async())

    async def _get_all_balances_async(self) -> dict:
        """Get all balances (async)."""
        if not await self._ensure_token():
            return {"KRW": 0.0, "holdings": {}}

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
            headers = self._get_headers("TTTC8434R")

            params = {
                "CANO": self._account_no[:8],
                "ACNT_PRDT_CD": (
                    self._account_no[8:10] if len(self._account_no) > 8 else "01"
                ),
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            }

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    return {"KRW": 0.0, "holdings": {}}

                result = await resp.json()

                # Cash balance
                output2 = result.get("output2", [{}])[0]
                cash = float(output2.get("dnca_tot_amt", 0))

                # Stock holdings
                holdings = {}
                for item in result.get("output1", []):
                    symbol = item.get("pdno")
                    if symbol:
                        holdings[symbol] = {
                            "quantity": float(item.get("hldg_qty", 0)),
                            "avg_price": float(item.get("pchs_avg_pric", 0)),
                            "current_price": float(item.get("prpr", 0)),
                            "profit_loss": float(item.get("evlu_pfls_amt", 0)),
                            "profit_loss_rate": float(item.get("evlu_pfls_rt", 0)),
                        }

                return {"KRW": cash, "holdings": holdings}

        except Exception as e:
            logger.error(f"[Kiwoom] Get all balances error: {e}")
            return {"KRW": 0.0, "holdings": {}}

    @staticmethod
    def get_fractional_symbols() -> set:
        """Get set of symbols available for fractional trading."""
        return FRACTIONAL_TRADING_SYMBOLS.copy()

    async def _close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
