"""
Bithumb 시세 조회 어댑터
- 현재가 조회
- 호가창 조회
- OHLCV 조회
"""

import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

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
        get_ohlcv(symbol, interval, limit): OHLCV 조회
        get_tickers(): 전체 종목 조회
    """
    
    def __init__(self):
        """초기화"""
        if pybithumb is None:
            raise ImportError("pybithumb not installed. Run: pip install pybithumb")
        logger.info("[BithumbMarketData] Adapter initialized")
    
    def get_quote(self, symbol: str) -> Optional[BithumbQuote]:
        """
        현재가 조회
        
        Args:
            symbol: 종목 (예: "BTC/KRW")
            
        Returns:
            BithumbQuote: 시세 정보
        """
        try:
            ticker = self._convert_symbol(symbol)
            price = pybithumb.get_current_price(ticker)
            
            if price is None:
                return None
            
            # 호가 조회
            orderbook = pybithumb.get_orderbook(ticker)
            bid = float(orderbook['bids'][0]['price']) if orderbook and orderbook.get('bids') else price * 0.999
            ask = float(orderbook['asks'][0]['price']) if orderbook and orderbook.get('asks') else price * 1.001
            
            return BithumbQuote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=float(price),
                volume=0,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"[BithumbMarketData] Quote failed: {e}")
            return None
    
    def get_orderbook(self, symbol: str, depth: int = 5) -> Optional[dict]:
        """
        호가창 조회
        
        Args:
            symbol: 종목
            depth: 호가 깊이
        """
        try:
            ticker = self._convert_symbol(symbol)
            orderbook = pybithumb.get_orderbook(ticker)
            
            if not orderbook:
                return None
            
            return {
                "bids": orderbook.get('bids', [])[:depth],
                "asks": orderbook.get('asks', [])[:depth],
                "timestamp": datetime.now()
            }
        except Exception as e:
            logger.error(f"[BithumbMarketData] Orderbook failed: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[BithumbOHLCV]:
        """
        OHLCV 조회
        
        Args:
            symbol: 종목
            interval: "1m", "5m", "1h", "1d" 등
            limit: 조회 개수
        """
        try:
            ticker = self._convert_symbol(symbol)
            
            # interval 변환
            interval_map = {
                "1m": "1m", "5m": "5m", "10m": "10m", "30m": "30m",
                "1h": "1h", "6h": "6h", "12h": "12h", "1d": "24h"
            }
            bithumb_interval = interval_map.get(interval, "24h")
            
            df = pybithumb.get_ohlcv(ticker, interval=bithumb_interval)
            
            if df is None or df.empty:
                return []
            
            # ✅ CRITICAL: 시간순 정렬 (Gemini 권장)
            # pybithumb은 시간순 정렬을 보장하지 않음
            df = df.sort_index()
            
            result = []
            for idx, row in df.tail(limit).iterrows():
                result.append(BithumbOHLCV(
                    timestamp=idx,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                ))
            
            return result
        except Exception as e:
            logger.error(f"[BithumbMarketData] OHLCV failed: {e}")
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
