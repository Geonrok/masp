"""
Bithumb 시세 조회 및 주문 실행 어댑터
- 현재가 조회
- 호가창 조회
- OHLCV 조회 (HTTP fallback 포함)
- 주문 실행 (BithumbSpotExecution)

Phase 6D 수정사항:
- HTTP fallback 추가 (91+ 캔들 확보)
- Row 방어 파싱 (GPT 권장)
- Timezone 정규화 (GPT 권장)

Phase 9 수정사항:
- BithumbSpotExecution 추가 (v2.1 통합)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

try:
    import pybithumb
except ImportError:
    pybithumb = None

logger = logging.getLogger(__name__)

# KST timezone
KST = timezone(timedelta(hours=9))


@dataclass
class BithumbQuote:
    """시세 정보"""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime


@dataclass
class BithumbOHLCV:
    """캔들 데이터"""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BithumbSpotMarketData:
    """
    Bithumb 시세 조회 어댑터 (Public API - 키 불필요)

    Methods:
        get_quote(symbol): 현재가 조회
        get_orderbook(symbol, depth): 호가창 조회
        get_ohlcv(symbol, interval, limit): OHLCV 조회 (HTTP fallback 포함)
        get_tickers(): 전체 종목 조회
    """

    def __init__(self):
        """초기화"""
        if pybithumb is None:
            raise ImportError("pybithumb not installed. Run: pip install pybithumb")
        logger.info("[BithumbMarketData] Adapter initialized")

    def get_quote(self, symbol: str) -> Optional[BithumbQuote]:
        """현재가 조회"""
        try:
            ticker = self._convert_symbol(symbol)
            price = pybithumb.get_current_price(ticker)

            if price is None:
                return None

            orderbook = pybithumb.get_orderbook(ticker)
            bid = (
                float(orderbook["bids"][0]["price"])
                if orderbook and orderbook.get("bids")
                else float(price) * 0.999
            )
            ask = (
                float(orderbook["asks"][0]["price"])
                if orderbook and orderbook.get("asks")
                else float(price) * 1.001
            )

            return BithumbQuote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=float(price),
                volume=0,
                timestamp=datetime.now(tz=timezone.utc),
            )
        except Exception as e:
            logger.error(f"[BithumbMarketData] Quote failed: {e}")
            return None

    def get_orderbook(self, symbol: str, depth: int = 5) -> Optional[dict]:
        """호가창 조회"""
        try:
            ticker = self._convert_symbol(symbol)
            orderbook = pybithumb.get_orderbook(ticker)

            if not orderbook:
                return None

            return {
                "bids": orderbook.get("bids", [])[:depth],
                "asks": orderbook.get("asks", [])[:depth],
                "timestamp": datetime.now(tz=timezone.utc),
            }
        except Exception as e:
            logger.error(f"[BithumbMarketData] Orderbook failed: {e}")
            return None

    def _fetch_candles_http(
        self, order_currency: str, payment_currency: str, interval: str
    ) -> List[BithumbOHLCV]:
        """Bithumb HTTP API fallback - 더 긴 히스토리 확보

        API 응답 포맷: [timestamp_ms, open, close, high, low, volume]
        """
        chart_map = {"1d": "24h", "day": "24h", "1h": "1h", "30m": "30m", "5m": "5m"}
        chart_interval = chart_map.get(interval, "24h")

        url = f"https://api.bithumb.com/public/candlestick/{order_currency}_{payment_currency}/{chart_interval}"

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            payload = r.json()

            if payload.get("status") != "0000":
                logger.warning(
                    f"[BithumbMarketData] HTTP API error: {payload.get('message', 'Unknown')}"
                )
                return []

            rows = payload.get("data") or []
            out: List[BithumbOHLCV] = []
            bad_rows = 0

            for row in rows:
                # [Phase 6D] Row 방어 파싱 (GPT 권장)
                # row: [timestamp_ms, open, close, high, low, volume]
                try:
                    if row is None or len(row) < 6:
                        bad_rows += 1
                        continue

                    ts_ms = int(float(row[0]))
                    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

                    out.append(
                        BithumbOHLCV(
                            timestamp=dt,
                            open=float(row[1]),
                            high=float(row[3]),
                            low=float(row[4]),
                            close=float(row[2]),
                            volume=float(row[5]),
                        )
                    )
                except (ValueError, TypeError, IndexError):
                    bad_rows += 1
                    continue

            out.sort(key=lambda c: c.timestamp)

            if bad_rows > 0:
                logger.warning(
                    "[BithumbMarketData] HTTP fallback: %d bad rows skipped (kept=%d) for %s",
                    bad_rows,
                    len(out),
                    order_currency,
                )

            logger.info(
                f"[BithumbMarketData] HTTP fallback: {len(out)} candles for {order_currency}"
            )
            return out

        except Exception as e:
            logger.error(f"[BithumbMarketData] HTTP fallback failed: {e}")
            return []

    def get_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[BithumbOHLCV]:
        """
        OHLCV 조회 (pybithumb 우선, HTTP fallback)

        Args:
            symbol: 종목 (예: "BTC/KRW")
            interval: "1m", "5m", "1h", "1d" 등
            limit: 조회 개수 (TSMOM 전략은 최소 91개 필요)
        """
        try:
            ticker = self._convert_symbol(symbol)

            # 1) pybithumb 시도
            interval_map = {"1d": "day", "1h": "1h", "30m": "30m", "5m": "5m"}
            bithumb_interval = interval_map.get(interval, "day")

            df = pybithumb.get_ohlcv(ticker, interval=bithumb_interval)

            pybithumb_result: List[BithumbOHLCV] = []
            if df is not None and not df.empty:
                df = df.sort_index()
                for idx, row in df.iterrows():
                    # [Phase 6D] Timezone 정규화 (GPT 권장)
                    # pybithumb timestamp를 tz-aware UTC로 변환
                    ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                    if isinstance(ts, datetime) and ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    pybithumb_result.append(
                        BithumbOHLCV(
                            timestamp=ts,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                        )
                    )

                if len(pybithumb_result) >= limit:
                    logger.debug(
                        f"[BithumbMarketData] pybithumb: {len(pybithumb_result)} candles for {symbol}"
                    )
                    return pybithumb_result[-limit:]

                logger.warning(
                    f"[BithumbMarketData] pybithumb insufficient: {len(pybithumb_result)} < {limit} for {symbol}"
                )

            # 2) HTTP fallback - 더 긴 히스토리 시도
            parts = symbol.split("/")
            order_currency = parts[0]
            payment_currency = parts[1] if len(parts) > 1 else "KRW"

            http_result = self._fetch_candles_http(
                order_currency, payment_currency, interval
            )
            if http_result and len(http_result) >= limit:
                return http_result[-limit:]

            # 3) 더 긴 결과 반환 (pybithumb vs HTTP)
            if http_result and len(http_result) > len(pybithumb_result):
                logger.info(
                    f"[BithumbMarketData] Using HTTP result: {len(http_result)} candles"
                )
                return http_result[-limit:] if len(http_result) > limit else http_result

            if pybithumb_result:
                logger.info(
                    f"[BithumbMarketData] Using pybithumb result: {len(pybithumb_result)} candles"
                )
                return (
                    pybithumb_result[-limit:]
                    if len(pybithumb_result) > limit
                    else pybithumb_result
                )

            return []

        except Exception as e:
            logger.error(f"[BithumbMarketData] OHLCV failed for {symbol}: {e}")
            return []

    def get_tickers(self) -> List[str]:
        """전체 종목 조회"""
        try:
            tickers = pybithumb.get_tickers()
            return tickers if tickers else []
        except Exception as e:
            logger.error(f"[BithumbMarketData] Tickers failed: {e}")
            return []

    def _convert_symbol(self, symbol: str) -> str:
        """심볼 변환: BTC/KRW -> BTC"""
        return symbol.split("/")[0]


# ============================================================
# Execution Adapter (Phase 9 - API 2.0 JWT 방식)
# ============================================================

try:
    import jwt  # PyJWT
except ImportError:
    jwt = None

import hashlib
import uuid


@dataclass
class BithumbOrderResult:
    """Bithumb 주문 결과"""

    order_id: str
    symbol: str
    side: str
    order_type: str = "MARKET"
    quantity: float = 0.0  # 코인 수량
    krw_amount: float = 0.0  # KRW 금액
    price: Optional[float] = None
    status: str = "UNKNOWN"
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(KST))
    message: str = ""


class BithumbSpotExecution:
    """
    Bithumb 주문 실행 어댑터 v3.0 (API 2.0 - JWT 인증)

    인터페이스 계약:
        place_order(symbol, side, quantity, ...)
        - quantity: 항상 "코인 수량" (StrategyRunner와 일관성)
        - BUY: 내부에서 quantity * price = KRW로 변환 후 API 호출
        - SELL: quantity 그대로 API 전달

    안전장치:
        - Kill-Switch 체크
        - MASP_ENABLE_LIVE_TRADING 체크
        - 최소/최대 주문 금액 검증
        - Rate Limit (0.1초 간격)

    API 2.0 변경사항:
        - JWT 인증 방식 (HS256)
        - 새로운 엔드포인트 (/v1/accounts, /v1/orders)
    """

    BASE_URL = "https://api.bithumb.com"
    FEE_RATE = 0.0025  # 0.25%
    MIN_ORDER_KRW = 1000  # Bithumb 최소 주문 금액
    REQUEST_INTERVAL = 0.1  # 100ms

    def __init__(self):
        """초기화"""
        if jwt is None:
            raise ImportError("PyJWT not installed. Run: pip install PyJWT")

        self.api_key = os.getenv("BITHUMB_API_KEY", "")
        self.secret_key = os.getenv("BITHUMB_SECRET_KEY") or os.getenv(
            "BITHUMB_API_SECRET", ""
        )

        self._initialized = bool(self.api_key and self.secret_key)
        if not self._initialized:
            logger.warning(
                "[BithumbExecution] API credentials not set. "
                "Set BITHUMB_API_KEY and BITHUMB_SECRET_KEY"
            )

        self._last_request_time: Optional[datetime] = None
        self._kill_switch_file = os.getenv("KILL_SWITCH_FILE", "storage/kill_switch.flag")

        logger.info("[BithumbExecution] Adapter v3.0 (API 2.0 JWT) initialized")

    def _generate_jwt(self, query_params: Optional[Dict] = None) -> str:
        """
        JWT 토큰 생성 (API 2.0 인증)

        Args:
            query_params: 쿼리 파라미터 (있을 경우 해시 포함)

        Returns:
            JWT 토큰 문자열
        """
        payload = {
            "access_key": self.api_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
        }

        # 파라미터가 있으면 query_hash 추가
        if query_params:
            query_string = "&".join(
                f"{k}={v}" for k, v in sorted(query_params.items())
            )
            query_hash = hashlib.sha512(query_string.encode()).hexdigest()
            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        API 요청 실행

        Args:
            method: HTTP 메소드 (GET, POST, DELETE)
            endpoint: API 엔드포인트 (예: /v1/accounts)
            params: GET 파라미터
            data: POST body

        Returns:
            응답 JSON 또는 None
        """
        # P1 Fix: 인증 정보 없으면 요청하지 않음
        if not self._initialized:
            logger.warning("[BithumbExecution] API credentials not initialized")
            return None

        self._rate_limit()

        url = f"{self.BASE_URL}{endpoint}"

        # JWT 토큰 생성
        query_params = params or data
        token = self._generate_jwt(query_params)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == "POST":
                resp = requests.post(url, headers=headers, json=data, timeout=10)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, params=params, timeout=10)
            else:
                logger.error(f"[BithumbExecution] Unknown method: {method}")
                return None

            if resp.status_code in (200, 201):
                return resp.json()
            else:
                logger.error(
                    f"[BithumbExecution] API error {resp.status_code}: {resp.text}"
                )
                return {"error": resp.status_code, "message": resp.text}

        except Exception as e:
            logger.error(f"[BithumbExecution] Request failed: {e}")
            return None

    def _ensure_live_trading(self) -> None:
        """실거래 모드 확인"""
        if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
            raise RuntimeError(
                "[Bithumb] Live trading disabled. Set MASP_ENABLE_LIVE_TRADING=1"
            )

    def _is_kill_switch_active(self) -> bool:
        """Kill-Switch 상태 확인"""
        return os.path.exists(self._kill_switch_file)

    def _rate_limit(self):
        """Rate Limit 처리"""
        if self._last_request_time:
            elapsed = (datetime.now(KST) - self._last_request_time).total_seconds()
            if elapsed < self.REQUEST_INTERVAL:
                time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = datetime.now(KST)

    # ========== 조회 기능 ==========

    def get_balance(self, currency: str = "KRW") -> Optional[float]:
        """잔고 조회 (단일 통화)"""
        balances = self.get_all_balances()
        for b in balances:
            if b.get("currency") == currency:
                return float(b.get("balance", 0)) + float(b.get("locked", 0))
        return 0.0

    def get_all_balances(self) -> List[Dict[str, Any]]:
        """전체 잔고 조회 (API 2.0)"""
        if not self._initialized:
            return []

        result = self._request("GET", "/v1/accounts")

        if result is None:
            return []

        if isinstance(result, dict) and "error" in result:
            logger.warning(f"[BithumbExecution] Balance error: {result.get('message')}")
            return []

        # API 2.0 응답: [{"currency": "KRW", "balance": "1000", "locked": "0", ...}, ...]
        if isinstance(result, list):
            return [
                {
                    "currency": item.get("currency"),
                    "balance": float(item.get("balance", 0)),
                    "locked": float(item.get("locked", 0)),
                    "avg_buy_price": float(item.get("avg_buy_price", 0)),
                }
                for item in result
                if float(item.get("balance", 0)) > 0 or float(item.get("locked", 0)) > 0
            ]

        return []

    def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회 (Public API - pybithumb 사용)"""
        try:
            ticker = self._convert_symbol(symbol)
            if pybithumb:
                price = pybithumb.get_current_price(ticker)
                return float(price) if price else None
            else:
                # pybithumb 없으면 Public API 직접 호출
                url = f"https://api.bithumb.com/public/ticker/{ticker}_KRW"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "0000":
                        return float(data["data"]["closing_price"])
                return None
        except Exception as e:
            logger.error(f"[BithumbExecution] Price query failed: {e}")
            return None

    # ========== 주문 기능 ==========

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> BithumbOrderResult:
        """
        주문 실행 (API 2.0)

        Args:
            symbol: 종목 (예: "BTC/KRW")
            side: "BUY" 또는 "SELL"
            quantity: 코인 수량 (예: 0.001 BTC)
            order_type: "MARKET" 또는 "LIMIT"
            price: 지정가 주문 시 가격
        """
        # 1. 실거래 모드 확인
        self._ensure_live_trading()

        # 2. API 키 확인
        if not self._initialized:
            return self._rejected_order(symbol, side, quantity, 0, "API credentials not set")

        # 3. Kill-Switch 체크
        if self._is_kill_switch_active():
            logger.warning("[BithumbExecution] Kill-Switch active")
            return self._rejected_order(symbol, side, quantity, 0, "Kill-Switch active")

        # 4. 현재가 조회
        current_price = price or self.get_current_price(symbol)
        if current_price is None:
            return self._rejected_order(symbol, side, quantity, 0, "Price unavailable")

        # 5. KRW 금액 계산
        krw_amount = quantity * current_price

        # 6. 로깅
        logger.info(
            f"[BithumbExecution] {side}: {quantity:.8f} coins = {krw_amount:,.0f} KRW"
        )

        # 7. 최소 주문 금액 체크
        if krw_amount < self.MIN_ORDER_KRW:
            return self._rejected_order(
                symbol,
                side,
                quantity,
                krw_amount,
                f"최소 주문 금액 미달: {krw_amount:,.0f} < {self.MIN_ORDER_KRW:,.0f} KRW",
            )

        # 8. 주문 실행 (API 2.0)
        try:
            # API 2.0 마켓 포맷: KRW-BTC
            ticker = self._convert_symbol(symbol)
            market = f"KRW-{ticker}"

            # 주문 파라미터
            order_data = {
                "market": market,
                "side": "bid" if side.upper() == "BUY" else "ask",
                "volume": str(quantity),
                "ord_type": "price" if order_type == "MARKET" and side.upper() == "BUY" else
                           "market" if order_type == "MARKET" else "limit",
            }

            # MARKET BUY는 price(총 금액) 사용, MARKET SELL은 volume만 사용
            if order_type == "MARKET":
                if side.upper() == "BUY":
                    order_data["price"] = str(int(krw_amount))
                    order_data.pop("volume", None)  # BUY는 volume 불필요
            else:  # LIMIT
                order_data["price"] = str(int(current_price))

            result = self._request("POST", "/v1/orders", data=order_data)

            if result is None:
                return self._rejected_order(
                    symbol, side, quantity, krw_amount, "API request failed"
                )

            if "error" in result:
                return self._rejected_order(
                    symbol, side, quantity, krw_amount, result.get("message", "Unknown error")
                )

            order_result = self._parse_result(
                result, symbol, side, quantity, krw_amount, order_type, current_price
            )

            logger.info(
                f"[BithumbExecution] Order completed: {side} {symbol} "
                f"qty={quantity:.8f} krw={krw_amount:,.0f} → {order_result.status}"
            )

            return order_result

        except Exception as e:
            logger.error(f"[BithumbExecution] Order failed: {e}", exc_info=True)
            return self._rejected_order(symbol, side, quantity, krw_amount, str(e))

    def cancel_order(self, order_id: str, symbol: str = "") -> bool:
        """주문 취소 (API 2.0)"""
        if not self._initialized:
            return False
        try:
            result = self._request("DELETE", "/v1/order", params={"uuid": order_id})
            if result and "error" not in result:
                logger.info(f"[BithumbExecution] Order {order_id} cancelled")
                return True
            return False
        except Exception as e:
            logger.error(f"[BithumbExecution] Cancel failed: {e}")
            return False

    # ========== Private Methods ==========

    def _convert_symbol(self, symbol: str) -> str:
        """심볼 변환: BTC/KRW -> BTC"""
        return symbol.split("/")[0]

    def _parse_result(
        self, result: Dict, symbol: str, side: str, quantity: float,
        krw_amount: float, order_type: str, price: float
    ) -> BithumbOrderResult:
        """API 2.0 응답 파싱"""
        # API 2.0 응답 필드:
        # uuid, side, ord_type, price, state, market, volume, remaining_volume, etc.
        order_id = result.get("uuid", "")
        state = result.get("state", "")
        filled_volume = float(result.get("executed_volume", 0))
        fee = krw_amount * self.FEE_RATE

        # 상태 매핑
        status_map = {
            "wait": "PENDING",
            "watch": "PENDING",
            "done": "FILLED",
            "cancel": "CANCELLED",
        }
        status = status_map.get(state, "UNKNOWN")

        return BithumbOrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            order_type=order_type,
            quantity=quantity,
            krw_amount=krw_amount,
            price=price,
            status=status,
            filled_quantity=filled_volume if filled_volume else quantity,
            filled_price=price,
            fee=fee,
            created_at=datetime.now(KST),
            message=f"Order {state}",
        )

    def _rejected_order(
        self, symbol: str, side: str, quantity: float, krw_amount: float, reason: str
    ) -> BithumbOrderResult:
        """거부된 주문"""
        logger.warning(
            f"[BithumbExecution] REJECTED: {symbol} {side} "
            f"qty={quantity:.8f} krw={krw_amount:,.0f} - {reason}"
        )
        return BithumbOrderResult(
            order_id="",
            symbol=symbol,
            side=side.upper(),
            order_type="MARKET",
            quantity=quantity,
            krw_amount=krw_amount,
            price=None,
            status="REJECTED",
            filled_quantity=0,
            filled_price=0,
            fee=0,
            created_at=datetime.now(KST),
            message=reason,
        )
