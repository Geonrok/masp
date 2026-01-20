"""
eBest (LS증권) Spot Market Data Adapter

Provides real-time market data from LS증권 Open API for KOSPI/KOSDAQ stocks.
Uses the `ebest` Python package (v1.0.2+) for REST API access.

API Documentation: https://openapi.ls-sec.co.kr
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import List, Optional, Dict, Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

from libs.adapters.base import MarketDataAdapter, MarketQuote

logger = logging.getLogger(__name__)


@dataclass
class OHLCVCandle:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class EbestSpotMarketData(MarketDataAdapter):
    """
    eBest (LS증권) Spot Market Data Adapter

    Provides:
        - get_quote(symbol): 현재가 조회 (t1102)
        - get_quotes(symbols): 복수 종목 현재가
        - get_ohlcv(symbol, interval, limit): 기간별 주가 (t1305)
        - get_all_symbols(): 전체 종목 코드 (t8430)
        - is_market_open(): 장 운영 시간 체크

    Authentication:
        Uses EBEST_APP_KEY and EBEST_APP_SECRET environment variables.
    """

    # Korean stock market hours (KST)
    MARKET_OPEN = dt_time(9, 0)
    MARKET_CLOSE = dt_time(15, 30)

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
    ):
        """
        Initialize eBest market data adapter.

        Args:
            app_key: LS증권 Open API app key (or EBEST_APP_KEY env)
            app_secret: LS증권 Open API app secret (or EBEST_APP_SECRET env)
        """
        self._app_key = app_key or os.getenv("EBEST_APP_KEY", "")
        self._app_secret = app_secret or os.getenv("EBEST_APP_SECRET", "")
        self._api: Optional[Any] = None
        self._logged_in = False

        if not self._app_key or not self._app_secret:
            logger.warning(
                "[eBest] API credentials not provided. "
                "Set EBEST_APP_KEY and EBEST_APP_SECRET environment variables."
            )

        logger.info("[eBest] MarketData adapter initialized")

    def _run_async(self, coro):
        """Run async coroutine synchronously.

        Handles the case where we may or may not already be in an async context.
        Reuses API connection when possible for efficiency.
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use nest_asyncio or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Reset API state for new thread/loop
                self._api = None
                self._logged_in = False
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            # No running loop - safe to use asyncio.run directly
            # Keep existing API state for connection reuse
            return asyncio.run(coro)

    async def _ensure_login(self) -> bool:
        """Ensure API is logged in."""
        if self._logged_in and self._api:
            return True

        if not self._app_key or not self._app_secret:
            logger.error("[eBest] Cannot login: missing credentials")
            return False

        try:
            from ebest import OpenApi
            self._api = OpenApi()
            success = await self._api.login(self._app_key, self._app_secret)
            if success:
                self._logged_in = True
                logger.info("[eBest] Login successful")
                return True
            else:
                logger.error("[eBest] Login failed")
                return False
        except ImportError:
            logger.error("[eBest] 'ebest' package not installed. Run: pip install ebest")
            return False
        except Exception as e:
            logger.error(f"[eBest] Login error: {e}")
            return False

    async def _get_quote_async(self, symbol: str) -> Optional[MarketQuote]:
        """Get quote for a symbol (async)."""
        if not await self._ensure_login():
            return None

        try:
            # t1102: 주식 현재가(시세) 조회
            # symbol format: 6-digit code (e.g., "005930" for Samsung)
            data = {
                "t1102InBlock": {
                    "shcode": symbol,
                }
            }

            result = await self._api.request("t1102", data)
            if not result:
                logger.warning(f"[eBest] No data for {symbol}")
                return None

            # ResponseValue has .body attribute containing the parsed JSON
            body = result.body if hasattr(result, 'body') else result
            out_block = body.get("t1102OutBlock", {}) if isinstance(body, dict) else {}
            if not out_block:
                logger.warning(f"[eBest] Empty response for {symbol}")
                return None

            # Parse response
            price = float(out_block.get("price", 0))
            offer = float(out_block.get("offerho", price))  # 매도호가
            bid = float(out_block.get("bidho", price))      # 매수호가
            volume = float(out_block.get("volume", 0))      # 거래량

            return MarketQuote(
                symbol=symbol,
                bid=bid,
                ask=offer,
                last=price,
                volume_24h=volume,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"[eBest] Error getting quote for {symbol}: {e}")
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
        if not await self._ensure_login():
            return []

        try:
            # t1305: Period-based stock price query
            # dwmcode: 1=daily, 2=weekly, 3=monthly
            # Note: eBest API only supports daily/weekly/monthly, not intraday
            dwm_map = {
                "1d": "1",
                "1w": "2",
                "1M": "3",
            }
            if interval not in dwm_map:
                raise ValueError(
                    f"[eBest] Unsupported interval '{interval}'. "
                    f"Supported: {list(dwm_map.keys())}. "
                    "Note: Intraday data (1m, 5m, etc.) is not available via t1305."
                )
            dwmcode = dwm_map[interval]

            # API limit is 500 candles per request
            actual_limit = min(limit, 500)
            if limit > 500:
                logger.warning(
                    "[eBest] Requested %d candles but API limit is 500. Returning %d.",
                    limit, actual_limit
                )

            data = {
                "t1305InBlock": {
                    "shcode": symbol,
                    "dwmcode": dwmcode,
                    "cnt": actual_limit,
                    "date": "",  # Empty for latest
                    "idx": 0,
                }
            }

            result = await self._api.request("t1305", data)
            if not result:
                logger.warning(f"[eBest] No OHLCV data for {symbol}")
                return []

            # ResponseValue has .body attribute containing the parsed JSON
            body = result.body if hasattr(result, 'body') else result
            out_block = body.get("t1305OutBlock1", []) if isinstance(body, dict) else []
            if not out_block:
                logger.warning(f"[eBest] Empty OHLCV response for {symbol}")
                return []

            candles = []
            for item in reversed(out_block):  # Oldest first
                try:
                    date_str = item.get("date", "")
                    if date_str:
                        ts = datetime.strptime(date_str, "%Y%m%d")
                    else:
                        continue

                    candles.append(OHLCVCandle(
                        timestamp=ts,
                        open=float(item.get("open", 0)),
                        high=float(item.get("high", 0)),
                        low=float(item.get("low", 0)),
                        close=float(item.get("close", 0)),
                        volume=float(item.get("volume", 0)),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning(f"[eBest] Failed to parse candle: {e}")
                    continue

            logger.debug(f"[eBest] get_ohlcv: {symbol} - {len(candles)} candles")
            return candles

        except Exception as e:
            logger.error(f"[eBest] Error getting OHLCV for {symbol}: {e}")
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
            limit: Number of candles (max 500)

        Returns:
            List of OHLCVCandle (oldest first)
        """
        return self._run_async(self._get_ohlcv_async(symbol, interval, limit))

    async def _get_all_symbols_async(self, market: str = "ALL") -> List[str]:
        """Get all stock symbols (async)."""
        if not await self._ensure_login():
            return []

        try:
            # t8430: 주식 종목 조회
            # gubun: 0=전체, 1=코스피, 2=코스닥
            gubun_map = {
                "ALL": "0",
                "KOSPI": "1",
                "KOSDAQ": "2",
            }
            gubun = gubun_map.get(market.upper(), "0")

            data = {
                "t8430InBlock": {
                    "gubun": gubun,
                }
            }

            result = await self._api.request("t8430", data)
            if not result:
                logger.warning(f"[eBest] No symbol data for market={market}")
                return []

            # ResponseValue has .body attribute containing the parsed JSON
            body = result.body if hasattr(result, 'body') else result
            out_block = body.get("t8430OutBlock", []) if isinstance(body, dict) else []
            if not out_block:
                logger.warning(f"[eBest] Empty symbol response for market={market}")
                return []

            symbols = []
            for item in out_block:
                code = item.get("shcode", "")
                if code:
                    symbols.append(code)

            logger.info(f"[eBest] Loaded {len(symbols)} symbols for {market}")
            return symbols

        except Exception as e:
            logger.error(f"[eBest] Error getting symbols: {e}")
            return []

    def get_all_symbols(self, market: str = "ALL") -> List[str]:
        """
        Get all stock symbols.

        Args:
            market: "ALL", "KOSPI", or "KOSDAQ"

        Returns:
            List of stock codes
        """
        return self._run_async(self._get_all_symbols_async(market))

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
