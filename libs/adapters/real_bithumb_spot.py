"""
Bithumb 시세 조회 어댑터
- 현재가 조회
- 호가창 조회
- OHLCV 조회 (HTTP fallback 포함)

Phase 6D 수정사항:
- HTTP fallback 추가 (91+ 캔들 확보)
- Row 방어 파싱 (GPT 권장)
- Timezone 정규화 (GPT 권장)
"""

import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

try:
    import pybithumb
except ImportError:
    pybithumb = None

logger = logging.getLogger(__name__)


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
