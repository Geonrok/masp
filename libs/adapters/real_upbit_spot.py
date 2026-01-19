"""
Upbit Spot Market Data Adapter (Phase 2A - Read Only)

Provides real-time market data from Upbit Korean cryptocurrency exchange.
Phase 2A: Read-only, no order execution.
Phase 2C: Full trading implementation with API authentication.
"""

import hashlib
import logging
import os
import random
import re
import ccxt
import threading
import time
import uuid
from typing import List, Optional, Dict, Any, Mapping
from urllib.parse import urlencode, unquote

import requests
import jwt
from libs.adapters.base import MarketDataAdapter, ExecutionAdapter, MarketQuote, OrderResult
from libs.adapters.rate_limit import TokenBucket
from libs.core.market_cache import MarketCache

logger = logging.getLogger(__name__)

RE_REMAINING_REQ = re.compile(
    r"group=(?P<group>[^;]+)(?:;\s*min=(?P<min>\d+))?;\s*sec=(?P<sec>\d+)"
)

# Shared Circuit Breaker State (모듈 ?�벨)
_UPBIT_CIRCUIT_OPEN_UNTIL: float = 0.0
_UPBIT_CIRCUIT_LOCK = threading.Lock()


def _circuit_is_open() -> bool:
    """Check if circuit breaker is open."""
    with _UPBIT_CIRCUIT_LOCK:
        return time.time() < _UPBIT_CIRCUIT_OPEN_UNTIL


def _open_circuit(cooldown: float) -> None:
    """Open circuit breaker for specified duration."""
    global _UPBIT_CIRCUIT_OPEN_UNTIL
    with _UPBIT_CIRCUIT_LOCK:
        _UPBIT_CIRCUIT_OPEN_UNTIL = time.time() + cooldown


class UpbitSpotMarketData(MarketDataAdapter):
    """
    Upbit Spot Market Data Adapter
    
    Phase 2A: Real market data (read-only, Public API)
    Supports: get_quote, get_quotes, get_orderbook
    
    API Documentation: https://docs.upbit.com/reference
    """
    
    BASE_URL = "https://api.upbit.com/v1"
    MAX_RETRIES = 3
    MAX_BACKOFF_SECONDS = 8
    BASE_BACKOFF_SECONDS = 1.0
    CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60

    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None,
                 cache: Optional[MarketCache] = None):
        """
        Initialize Upbit adapter.
        
        Args:
            access_key: Upbit API access key (optional for public API)
            secret_key: Upbit API secret key (optional for public API)
            cache: MarketCache instance (creates default if None)
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Multi-Asset-Strategy-Platform/1.0"
        })
        
        # [CRITICAL PATCH #2] Cache integration for rate limit protection
        self._cache = cache if cache else MarketCache(default_ttl=5.0)
        logger.info(
            "[Upbit] MarketData adapter initialized (read-only mode) with cache (TTL: 5s)"
        )

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Get current quote for a symbol (with cache, Decorrelated Jitter, and 418 protection).
        """
        cached = self._cache.get(symbol)
        if cached:
            logger.debug(f"[Upbit] Cache HIT: {symbol}")
            return cached

        if _circuit_is_open():
            logger.warning(f"[Upbit] Circuit breaker open. Skipping request for {symbol}")
            return None

        retry_count = 0
        previous_sleep = self.BASE_BACKOFF_SECONDS

        while True:
            cached = self._cache.get(symbol)
            if cached:
                logger.debug(f"[Upbit] Cache HIT: {symbol}")
                return cached

            try:
                market = self._convert_symbol(symbol)
                url = f"{self.BASE_URL}/ticker"
                params = {"markets": market}

                response = self.session.get(url, params=params, timeout=5)

                if response.status_code == 418:
                    _open_circuit(self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    logger.critical(
                        "[Upbit] CRITICAL: IP Ban detected (418) for %s. Circuit breaker opened for %ds",
                        symbol,
                        self.CIRCUIT_BREAKER_COOLDOWN_SECONDS,
                    )
                    raise RuntimeError(
                        f"[Upbit] IP Ban (418) detected for {symbol}. "
                        f"Circuit breaker opened for {self.CIRCUIT_BREAKER_COOLDOWN_SECONDS}s"
                    )

                if response.status_code == 429:
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f"[Upbit] Rate limit exceeded max retries for {symbol}")
                        raise RuntimeError(
                            f"429 Too Many Requests after {self.MAX_RETRIES} retries"
                        )

                    upper = max(self.BASE_BACKOFF_SECONDS, previous_sleep * 3)
                    wait_time = min(
                        self.MAX_BACKOFF_SECONDS,
                        random.uniform(self.BASE_BACKOFF_SECONDS, upper),
                    )
                    logger.warning(
                        "[Upbit] Rate limit (429). Retry %d/%d after %.2fs for %s (source=decorrelated)",
                        retry_count + 1,
                        self.MAX_RETRIES,
                        wait_time,
                        symbol,
                    )
                    time.sleep(wait_time)
                    previous_sleep = wait_time
                    retry_count += 1
                    continue

                response.raise_for_status()

                data = response.json()[0]

                quote = MarketQuote(
                    symbol=symbol,
                    bid=float(data.get("trade_price", 0)),
                    ask=float(data.get("trade_price", 0)),
                    last=float(data.get("trade_price", 0)),
                    volume_24h=float(data.get("acc_trade_volume_24h", 0)),
                    timestamp=str(data.get("timestamp", ""))
                )

                self._cache.set(symbol, quote)
                logger.debug(f"[Upbit] Cache SET: {symbol}")

                return quote

            except requests.exceptions.HTTPError as e:
                logger.error(f"[Upbit] HTTP error for {symbol}: {e}")
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"[Upbit] Request failed for {symbol}: {e}")
                return None
            except (KeyError, IndexError, ValueError) as e:
                logger.error(f"[Upbit] Failed to parse quote data for {symbol}: {e}")
                return None
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """
        Get quotes for multiple symbols (with Circuit Breaker, 418/429 protection).
        """
        if _circuit_is_open():
            logger.warning("[Upbit] Circuit breaker open. Skipping request (get_quotes).")
            return {}

        retry_count = 0
        previous_sleep = self.BASE_BACKOFF_SECONDS

        while True:
            try:
                markets = ",".join([self._convert_symbol(s) for s in symbols])
                url = f"{self.BASE_URL}/ticker"
                params = {"markets": markets}

                response = self.session.get(url, params=params, timeout=10)

                if response.status_code == 418:
                    _open_circuit(self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    logger.critical(
                        "[Upbit] CRITICAL: IP Ban detected (418). Circuit breaker opened for %ds (get_quotes)",
                        self.CIRCUIT_BREAKER_COOLDOWN_SECONDS,
                    )
                    raise RuntimeError(
                        f"[Upbit] IP Ban (418) detected. Circuit breaker opened for "
                        f"{self.CIRCUIT_BREAKER_COOLDOWN_SECONDS}s"
                    )

                if response.status_code == 429:
                    if retry_count >= self.MAX_RETRIES:
                        logger.error("[Upbit] Rate limit exceeded max retries (get_quotes)")
                        raise RuntimeError(
                            f"429 Too Many Requests after {self.MAX_RETRIES} retries"
                        )

                    upper = max(self.BASE_BACKOFF_SECONDS, previous_sleep * 3)
                    wait_time = min(
                        self.MAX_BACKOFF_SECONDS,
                        random.uniform(self.BASE_BACKOFF_SECONDS, upper),
                    )
                    logger.warning(
                        "[Upbit] Rate limit (429). Retry %d/%d after %.2fs (get_quotes, source=decorrelated)",
                        retry_count + 1,
                        self.MAX_RETRIES,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    previous_sleep = wait_time
                    retry_count += 1
                    continue

                response.raise_for_status()

                data_list = response.json()

                result = {}
                for data in data_list:
                    symbol = self._revert_symbol(data["market"])
                    result[symbol] = MarketQuote(
                        symbol=symbol,
                        bid=float(data.get("trade_price", 0)),
                        ask=float(data.get("trade_price", 0)),
                        last=float(data.get("trade_price", 0)),
                        volume_24h=float(data.get("acc_trade_volume_24h", 0)),
                        timestamp=str(data.get("timestamp", ""))
                    )

                return result

            except requests.exceptions.RequestException as e:
                logger.error(f"[Upbit] Failed to get quotes: {e}")
                return {}
            except (KeyError, ValueError) as e:
                logger.error(f"[Upbit] Failed to parse quotes data: {e}")
                return {}
    
    def is_market_open(self) -> bool:
        """
        Check if Upbit market is open.
        
        Returns:
            True (Upbit operates 24/7)
        """
        return True
    
    def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get orderbook for a symbol (with Circuit Breaker, 418/429 protection).
        """
        if _circuit_is_open():
            logger.warning(f"[Upbit] Circuit breaker open. Skipping request for {symbol} (get_orderbook).")
            return None

        retry_count = 0
        previous_sleep = self.BASE_BACKOFF_SECONDS

        while True:
            try:
                market = self._convert_symbol(symbol)
                url = f"{self.BASE_URL}/orderbook"
                params = {"markets": market}

                response = self.session.get(url, params=params, timeout=5)

                if response.status_code == 418:
                    _open_circuit(self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    logger.critical(
                        "[Upbit] CRITICAL: IP Ban detected (418) for %s. Circuit breaker opened for %ds (get_orderbook)",
                        symbol,
                        self.CIRCUIT_BREAKER_COOLDOWN_SECONDS,
                    )
                    raise RuntimeError(
                        f"[Upbit] IP Ban (418) detected for {symbol}. "
                        f"Circuit breaker opened for {self.CIRCUIT_BREAKER_COOLDOWN_SECONDS}s"
                    )

                if response.status_code == 429:
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f"[Upbit] Rate limit exceeded max retries for {symbol} (get_orderbook)")
                        raise RuntimeError(
                            f"429 Too Many Requests after {self.MAX_RETRIES} retries"
                        )

                    upper = max(self.BASE_BACKOFF_SECONDS, previous_sleep * 3)
                    wait_time = min(
                        self.MAX_BACKOFF_SECONDS,
                        random.uniform(self.BASE_BACKOFF_SECONDS, upper),
                    )
                    logger.warning(
                        "[Upbit] Rate limit (429). Retry %d/%d after %.2fs for %s (get_orderbook, source=decorrelated)",
                        retry_count + 1,
                        self.MAX_RETRIES,
                        wait_time,
                        symbol,
                    )
                    time.sleep(wait_time)
                    previous_sleep = wait_time
                    retry_count += 1
                    continue

                response.raise_for_status()

                data = response.json()[0]

                return {
                    "symbol": symbol,
                    "timestamp": data.get("timestamp"),
                    "bids": [
                        {"price": float(unit["bid_price"]), "size": float(unit["bid_size"])}
                        for unit in data.get("orderbook_units", [])
                    ],
                    "asks": [
                        {"price": float(unit["ask_price"]), "size": float(unit["ask_size"])}
                        for unit in data.get("orderbook_units", [])
                    ]
                }

            except requests.exceptions.RequestException as e:
                logger.error(f"[Upbit] Failed to get orderbook for {symbol}: {e}")
                return None
            except (KeyError, IndexError, ValueError) as e:
                logger.error(f"[Upbit] Failed to parse orderbook data for {symbol}: {e}")
                return None

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 200,
        to: str = None,
    ) -> List:
        """
        Get OHLCV candle data.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/KRW")
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
            limit: Number of candles (max 200)
            to: End datetime (ISO format, optional)
        
        Returns:
            List of OHLCV objects (oldest first)
        """
        from dataclasses import dataclass
        from datetime import datetime as dt
        
        @dataclass
        class OHLCVCandle:
            timestamp: dt
            open: float
            high: float
            low: float
            close: float
            volume: float
        
        if _circuit_is_open():
            logger.warning("[Upbit] Circuit breaker open, returning empty for get_ohlcv")
            return []
        
        market = self._convert_symbol(symbol)
        
        # Interval mapping (Upbit API format)
        interval_map = {
            "1m": ("minutes", 1),
            "3m": ("minutes", 3),
            "5m": ("minutes", 5),
            "15m": ("minutes", 15),
            "30m": ("minutes", 30),
            "1h": ("minutes", 60),
            "4h": ("minutes", 240),
            "1d": ("days", None),
            "1w": ("weeks", None),
            "1M": ("months", None),
        }
        
        if interval not in interval_map:
            logger.warning(f"[Upbit] Unknown interval {interval}, using 1d")
            interval = "1d"
        
        candle_type, unit = interval_map[interval]
        
        if candle_type == "minutes":
            url = f"{self.BASE_URL}/candles/{candle_type}/{unit}"
        else:
            url = f"{self.BASE_URL}/candles/{candle_type}"
        
        params = {
            "market": market,
            "count": min(limit, 200),
        }
        if to:
            params["to"] = to
        
        try:
            time.sleep(random.uniform(0.05, 0.15))  # Decorrelated jitter
            
            resp = self.session.get(url, params=params, timeout=5)
            
            if resp.status_code == 418:
                _open_circuit(self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                logger.error("[Upbit] 418 received in get_ohlcv, circuit opened")
                return []
            
            if resp.status_code == 429:
                logger.warning("[Upbit] 429 rate limit in get_ohlcv, backing off")
                time.sleep(1.0)
                return []
            
            resp.raise_for_status()
            data = resp.json()
            
            candles = []
            for item in reversed(data):  # Oldest first
                try:
                    ts_str = item.get("candle_date_time_kst", item.get("candle_date_time_utc", ""))
                    if ts_str:
                        ts = dt.fromisoformat(ts_str.replace("T", " ").split("+")[0])
                    else:
                        ts = dt.now()
                    
                    candles.append(OHLCVCandle(
                        timestamp=ts,
                        open=float(item["opening_price"]),
                        high=float(item["high_price"]),
                        low=float(item["low_price"]),
                        close=float(item["trade_price"]),
                        volume=float(item["candle_acc_trade_volume"]),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning(f"[Upbit] Failed to parse candle: {e}")
                    continue
            
            logger.debug(f"[Upbit] get_ohlcv: {symbol} - {len(candles)} candles")
            return candles
            
        except requests.RequestException as exc:
            logger.error(f"[Upbit] OHLCV request failed for {symbol}: {exc}")
            return []

    def _convert_symbol(self, symbol: str) -> str:
        """
        Convert symbol format: 'BTC/KRW' ??'KRW-BTC'
        
        Args:
            symbol: Symbol in "BTC/KRW" format
        
        Returns:
            Upbit market code "KRW-BTC"
        """
        base, quote = symbol.split("/")
        return f"{quote}-{base}"
    
    def _revert_symbol(self, market: str) -> str:
        """
        Revert symbol format: 'KRW-BTC' ??'BTC/KRW'
        
        Args:
            market: Upbit market code "KRW-BTC"
        
        Returns:
            Symbol in "BTC/KRW" format
        """
        quote, base = market.split("-")
        return f"{base}/{quote}"


class UpbitSpotExecution(ExecutionAdapter):
    """
    Upbit Spot Execution Adapter

    Phase 6A: Real order execution with API authentication
    """

    BASE_URL = "https://api.upbit.com/v1"
    VALID_SIDES = {"BUY", "SELL"}
    VALID_ORDER_TYPES = {"MARKET", "LIMIT"}
    MIN_ORDER_KRW = 5000
    VALID_JWT_ALGORITHMS = {"HS512", "HS256"}
    MAX_RECOVERY_ATTEMPTS = 3
    RECOVERY_TIMEOUT = 30
    MAX_429_RETRIES = 3
    MAX_BACKOFF_SECONDS = 8
    BASE_BACKOFF_SECONDS = 1.0
    CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60

    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.access_key = access_key or os.getenv("UPBIT_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("UPBIT_SECRET_KEY")
        jwt_alg_env = os.getenv("UPBIT_JWT_ALG", "HS512").upper()
        if jwt_alg_env not in self.VALID_JWT_ALGORITHMS:
            raise ValueError(f"Invalid JWT algorithm: {jwt_alg_env}")
        self.jwt_alg = jwt_alg_env
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Multi-Asset-Strategy-Platform/1.0",
        })
        self.client = ccxt.upbit({
            "apiKey": self.access_key,
            "secret": self.secret_key,
            "enableRateLimit": True,
            "options": {
                "createMarketBuyOrderRequiresPrice": False,
            },
        })
        self._order_bucket = TokenBucket(rate_per_sec=8, capacity=8)
        self._default_bucket = TokenBucket(rate_per_sec=30, capacity=30)
        self._rate_limit_info: Dict[str, Any] = {
            "standard": None,
            "remaining_req": None,
            "updated_at": None,
        }
        self._last_standard_remaining: Optional[int] = None
        self._last_identifier: Optional[str] = None
        self._last_jwt_payload: Optional[Dict[str, str]] = None
        self._last_response_headers: Dict[str, str] = {}

        logger.info("[Upbit] Execution adapter initialized")

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float = None,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        *,
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,
    ) -> OrderResult:
        """
        Place an order on Upbit.

        Args:
            symbol: Trading pair (e.g., "BTC/KRW")
            side: "BUY" or "SELL"
            quantity: [DEPRECATED] Use units or amount_krw instead
            order_type: "MARKET" or "LIMIT"
            price: Limit price (for LIMIT orders)
            units: Coin quantity (for SELL or LIMIT BUY)
            amount_krw: KRW amount (for MARKET BUY)
        """
        self._ensure_live_trading()
        self._ensure_credentials()

        side_upper = side.upper()
        if side_upper not in self.VALID_SIDES:
            raise ValueError(f"Invalid side: {side_upper}")

        order_type_upper = order_type.upper()
        if order_type_upper not in self.VALID_ORDER_TYPES:
            raise ValueError(f"Invalid order type: {order_type_upper}")

        # Handle parameter compatibility
        if quantity is not None and units is None and amount_krw is None:
            # Legacy: quantity used as units for SELL/LIMIT, amount_krw for MARKET BUY
            if side_upper == "BUY" and order_type_upper == "MARKET":
                amount_krw = quantity
            else:
                units = quantity

        # Validate parameters
        if side_upper == "BUY" and order_type_upper == "MARKET":
            if amount_krw is None and units is None:
                raise ValueError("[Upbit] MARKET BUY requires amount_krw")
            if amount_krw is not None and amount_krw < self.MIN_ORDER_KRW:
                raise ValueError(f"[Upbit] MARKET BUY requires at least {self.MIN_ORDER_KRW} KRW")
        elif side_upper == "SELL":
            if units is None:
                raise ValueError("[Upbit] SELL requires units")
            if units <= 0:
                raise ValueError("Units must be positive")

        market = self._convert_symbol(symbol)
        identifier = str(uuid.uuid4())
        self._last_identifier = identifier

        params: Dict[str, Any] = {
            "market": market,
            "side": "bid" if side_upper == "BUY" else "ask",
            "identifier": identifier,
        }

        if order_type_upper == "MARKET":
            if side_upper == "BUY":
                if price is not None:
                    raise ValueError(
                        "[Upbit] MARKET BUY does not accept 'price' parameter. "
                        f"Use 'amount_krw' for KRW amount. Got price={price}"
                    )
                params["ord_type"] = "price"
                params["price"] = str(int(amount_krw))
            else:
                params["ord_type"] = "market"
                params["volume"] = str(units)
        else:
            if price is None or price <= 0:
                raise ValueError("Limit price must be positive")
            params["ord_type"] = "limit"
            params["price"] = str(price)
            params["volume"] = str(units)

        # Determine quantity for result
        result_quantity = units if units is not None else amount_krw

        try:
            data = self._request("POST", "/orders", params=params, is_order=True)
            return self._to_order_result(data, symbol, side_upper, result_quantity, price)
        except requests.exceptions.Timeout:
            start_time = time.monotonic()
            attempts = 0
            while attempts < self.MAX_RECOVERY_ATTEMPTS and (time.monotonic() - start_time) < self.RECOVERY_TIMEOUT:
                attempts += 1
                recovered = self._get_order_by_identifier(identifier)
                if recovered:
                    return self._to_order_result(recovered, symbol, side_upper, result_quantity, price)
                if attempts < self.MAX_RECOVERY_ATTEMPTS:
                    time.sleep(2 ** attempts)
            return OrderResult(success=False, message="Timeout and order not found", mock=False)
        except Exception as exc:
            return OrderResult(success=False, message=str(exc), mock=False)

    def get_order_status(self, order_id: str) -> Optional[dict]:
        self._ensure_live_trading()
        self._ensure_credentials()
        return self._request("GET", "/order", params={"uuid": order_id}, is_order=True)

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_live_trading()
        self._ensure_credentials()
        data = self._request("DELETE", "/order", params={"uuid": order_id}, is_order=True)
        return bool(data and data.get("uuid"))

    def get_balance(self, asset: str) -> Optional[float]:
        self._ensure_live_trading()
        self._ensure_credentials()
        balances = self.get_all_balances()
        for entry in balances:
            if entry.get("currency") == asset:
                return float(entry.get("balance", 0))
        return 0.0

    def get_all_balances(self) -> List[Dict[str, Any]]:
        self._ensure_live_trading()
        self._ensure_credentials()
        data = self._request("GET", "/accounts", params={}, is_order=False)
        return data if isinstance(data, list) else []

    def _ensure_live_trading(self) -> None:
        if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
            raise RuntimeError(
                "[Upbit] Live trading disabled. Set MASP_ENABLE_LIVE_TRADING=1"
            )

    def _ensure_credentials(self) -> None:
        if not self.access_key or not self.secret_key:
            raise ValueError("UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY are required")


    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 200,
        to: str = None,
    ) -> List:
        """
        Get OHLCV candle data.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/KRW")
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
            limit: Number of candles (max 200)
            to: End datetime (ISO format, optional)
        
        Returns:
            List of OHLCV objects (oldest first)
        """
        from dataclasses import dataclass
        from datetime import datetime as dt
        
        @dataclass
        class OHLCVCandle:
            timestamp: dt
            open: float
            high: float
            low: float
            close: float
            volume: float
        
        if _circuit_is_open():
            logger.warning("[Upbit] Circuit breaker open, returning empty for get_ohlcv")
            return []
        
        market = self._convert_symbol(symbol)
        
        # Interval mapping (Upbit API format)
        interval_map = {
            "1m": ("minutes", 1),
            "3m": ("minutes", 3),
            "5m": ("minutes", 5),
            "15m": ("minutes", 15),
            "30m": ("minutes", 30),
            "1h": ("minutes", 60),
            "4h": ("minutes", 240),
            "1d": ("days", None),
            "1w": ("weeks", None),
            "1M": ("months", None),
        }
        
        if interval not in interval_map:
            logger.warning(f"[Upbit] Unknown interval {interval}, using 1d")
            interval = "1d"
        
        candle_type, unit = interval_map[interval]
        
        if candle_type == "minutes":
            url = f"{self.BASE_URL}/candles/{candle_type}/{unit}"
        else:
            url = f"{self.BASE_URL}/candles/{candle_type}"
        
        params = {
            "market": market,
            "count": min(limit, 200),
        }
        if to:
            params["to"] = to
        
        try:
            time.sleep(random.uniform(0.05, 0.15))  # Decorrelated jitter
            
            resp = self.session.get(url, params=params, timeout=5)
            
            if resp.status_code == 418:
                _open_circuit(self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                logger.error("[Upbit] 418 received in get_ohlcv, circuit opened")
                return []
            
            if resp.status_code == 429:
                logger.warning("[Upbit] 429 rate limit in get_ohlcv, backing off")
                time.sleep(1.0)
                return []
            
            resp.raise_for_status()
            data = resp.json()
            
            candles = []
            for item in reversed(data):  # Oldest first
                try:
                    ts_str = item.get("candle_date_time_kst", item.get("candle_date_time_utc", ""))
                    if ts_str:
                        ts = dt.fromisoformat(ts_str.replace("T", " ").split("+")[0])
                    else:
                        ts = dt.now()
                    
                    candles.append(OHLCVCandle(
                        timestamp=ts,
                        open=float(item["opening_price"]),
                        high=float(item["high_price"]),
                        low=float(item["low_price"]),
                        close=float(item["trade_price"]),
                        volume=float(item["candle_acc_trade_volume"]),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning(f"[Upbit] Failed to parse candle: {e}")
                    continue
            
            logger.debug(f"[Upbit] get_ohlcv: {symbol} - {len(candles)} candles")
            return candles
            
        except requests.RequestException as exc:
            logger.error(f"[Upbit] OHLCV request failed for {symbol}: {exc}")
            return []

    def _convert_symbol(self, symbol: str) -> str:
        base, quote = symbol.split("/")
        return f"{quote}-{base}"

    def _build_query_string(self, params: Dict[str, Any]) -> str:
        return unquote(urlencode(params, doseq=True))

    def _build_auth_header(self, params: Dict[str, Any]) -> str:
        query = self._build_query_string(params)
        query_hash = hashlib.sha512(query.encode()).hexdigest()

        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
            "query_hash": query_hash,
            "query_hash_alg": "SHA512",
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.jwt_alg)
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        self._last_jwt_payload = payload
        return f"Bearer {token}"

    def _rate_limit(self, is_order: bool) -> None:
        bucket = self._order_bucket if is_order else self._default_bucket
        bucket.consume()

    def _update_rate_limit_headers(self, headers: Mapping[str, str], *, is_order: bool) -> None:
        now = int(time.time())

        standard_info = None
        try:
            limit = headers.get("X-RateLimit-Limit")
            remaining = headers.get("X-RateLimit-Remaining")
            reset_ts = headers.get("X-RateLimit-Reset")
            retry_after = headers.get("Retry-After")

            if limit and remaining:
                standard_info = {
                    "limit": int(limit),
                    "remaining": int(remaining),
                    "reset_at": int(reset_ts) if reset_ts else None,
                    "retry_after": int(retry_after) if retry_after else None,
                }
                if self._last_standard_remaining != standard_info["remaining"]:
                    logger.debug(
                        "[Upbit] Standard Rate Limit: %d/%d remaining",
                        standard_info["remaining"],
                        standard_info["limit"],
                    )
                    self._last_standard_remaining = standard_info["remaining"]
        except (ValueError, TypeError) as exc:
            logger.debug("[Upbit] Standard header parse failed: %s", exc)

        remaining_req_info = None
        remaining_req = headers.get("Remaining-Req")
        if remaining_req:
            match = RE_REMAINING_REQ.search(remaining_req)
            if match:
                try:
                    remaining_req_info = {
                        "group": match.group("group"),
                        "min": int(match.group("min")) if match.group("min") else None,
                        "sec": int(match.group("sec")),
                    }
                except (ValueError, TypeError) as exc:
                    logger.debug("[Upbit] Remaining-Req parse failed: %s", exc)

        self._rate_limit_info = {
            "standard": standard_info,
            "remaining_req": remaining_req_info,
            "updated_at": now,
        }

        if os.getenv("MASP_SYNC_TOKEN_BUCKET") == "1":
            self._sync_token_bucket(is_order=is_order)

    def _sync_token_bucket(self, is_order: bool) -> None:
        remaining_req = self._rate_limit_info.get("remaining_req")
        if remaining_req and remaining_req.get("sec") is not None:
            try:
                group = remaining_req.get("group", "")
                sec_remaining = remaining_req["sec"]
                if is_order and group not in {"order", "orders"}:
                    logger.debug(
                        "[Upbit] Skipping sync: is_order=True but group=%s",
                        group,
                    )
                    return
                if not is_order and group in {"order", "orders"}:
                    logger.debug(
                        "[Upbit] Skipping sync: is_order=False but group=%s",
                        group,
                    )
                    return
                bucket = self._order_bucket if is_order else self._default_bucket
                bucket.set_tokens(sec_remaining)
                logger.debug(
                    "[Upbit] TokenBucket synced: %d tokens (is_order=%s, group=%s)",
                    sec_remaining,
                    is_order,
                    group,
                )
            except (ValueError, TypeError, AttributeError) as exc:
                logger.debug("[Upbit] TokenBucket sync failed: %s", exc)

    def _request(self, method: str, path: str, params: Dict[str, Any], is_order: bool) -> Any:
        """Execute HTTP request with rate limiting, Decorrelated Jitter backoff, and 418 Circuit Breaker."""
        if _circuit_is_open():
            raise RuntimeError("[Upbit] Circuit breaker is open. Request blocked.")

        retry_count = 0
        previous_sleep = self.BASE_BACKOFF_SECONDS

        while True:
            self._rate_limit(is_order)
            headers = {
                "Authorization": self._build_auth_header(params),
            }
            url = f"{self.BASE_URL}{path}"
            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                params=params if method in {"GET", "DELETE"} else None,
                json=params if method == "POST" else None,
                timeout=5,
            )
            self._last_response_headers = dict(response.headers)
            self._update_rate_limit_headers(response.headers, is_order=is_order)

            if response.status_code == 418:
                _open_circuit(self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                logger.critical(
                    "[Upbit] CRITICAL: IP Ban detected (418). Circuit breaker opened for %ds (method=%s, path=%s, is_order=%s)",
                    self.CIRCUIT_BREAKER_COOLDOWN_SECONDS,
                    method,
                    path,
                    is_order,
                )
                raise RuntimeError(
                    f"[Upbit] IP Ban (418) detected. Circuit breaker opened for "
                    f"{self.CIRCUIT_BREAKER_COOLDOWN_SECONDS}s"
                )

            if response.status_code == 429:
                if retry_count >= self.MAX_429_RETRIES:
                    raise RuntimeError(
                        f"[Upbit] Rate limited after {self.MAX_429_RETRIES} retries"
                    )

                retry_after = response.headers.get("Retry-After")
                retry_after_used = False
                wait_time = self.MAX_BACKOFF_SECONDS
                if retry_after:
                    try:
                        parsed = int(retry_after)
                        wait_time = max(0, min(parsed, self.MAX_BACKOFF_SECONDS))
                        retry_after_used = True
                    except (ValueError, TypeError):
                        upper = max(self.BASE_BACKOFF_SECONDS, previous_sleep * 3)
                        wait_time = min(
                            self.MAX_BACKOFF_SECONDS,
                            random.uniform(self.BASE_BACKOFF_SECONDS, upper),
                        )
                        previous_sleep = wait_time
                else:
                    upper = max(self.BASE_BACKOFF_SECONDS, previous_sleep * 3)
                    wait_time = min(
                        self.MAX_BACKOFF_SECONDS,
                        random.uniform(self.BASE_BACKOFF_SECONDS, upper),
                    )
                    previous_sleep = wait_time

                backoff_source = "Retry-After" if retry_after_used else "decorrelated"
                logger.warning(
                    "[Upbit] Rate limited (429). Retry %d/%d after %.2fs (method=%s, path=%s, is_order=%s, source=%s)",
                    retry_count + 1,
                    self.MAX_429_RETRIES,
                    wait_time,
                    method,
                    path,
                    is_order,
                    backoff_source,
                )
                time.sleep(wait_time)
                retry_count += 1
                continue

            if response.status_code >= 400:
                message = response.text
                try:
                    payload = response.json()
                    if isinstance(payload, dict):
                        message = payload.get("error", {}).get("message", message)
                except ValueError:
                    pass
                raise RuntimeError(f"[Upbit] API error {response.status_code}: {message}")

            return response.json()

    def _get_order_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            return self._request("GET", "/order", params={"identifier": identifier}, is_order=True)
        except Exception:
            return None

    def _to_order_result(
        self,
        data: Dict[str, Any],
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
    ) -> OrderResult:
        if not isinstance(data, dict):
            return OrderResult(success=False, message="Invalid response", mock=False)

        state = data.get("state", "")
        if state == "cancel":
            executed_vol = float(data.get("executed_volume", 0))
            status = "FILLED" if executed_vol > 0 else "CANCELLED"
        else:
            status_map = {
                "wait": "PENDING",
                "done": "FILLED",
            }
            status = status_map.get(state, state or "unknown")

        executed_vol = float(data.get("executed_volume", 0))
        success = state in {"wait", "done"} or (state == "cancel" and executed_vol > 0)
        return OrderResult(
            success=success,
            order_id=data.get("uuid"),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=float(data.get("price") or price or 0),
            status=status,
            message=data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else None,
            mock=False,
        )
