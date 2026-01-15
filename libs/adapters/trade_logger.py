"""
거래 로거 - 모든 거래 기록 CSV 저장
- 일별 파일 분리
- 월별 디렉토리 구조
- 요약 통계 조회
- Thread Safe 쓰기 지원
- Formula Injection 방어
"""

import csv
import logging
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    거래 로거
    
    저장 경로: logs/trades/YYYY-MM/trades_YYYY-MM-DD.csv
    
    Methods:
        log_trade(trade): 거래 기록 (Thread Safe)
        get_trades(date): 특정 날짜 조회
        get_daily_summary(date): 일일 요약
    """
    
    DEFAULT_LOG_DIR = "logs/trades"
    
    CSV_HEADERS = [
        "timestamp", "exchange", "order_id", "symbol", "side",
        "quantity", "price", "fee", "pnl", "status", "message"
    ]
    
    def __init__(self, log_dir: Optional[str] = None):
        """초기화"""
        self.log_dir = Path(log_dir or self.DEFAULT_LOG_DIR)
        self._lock = threading.Lock()
        self._ensure_directory()
        logger.info(f"[TradeLogger] Initialized: {self.log_dir}")
    
    def _ensure_directory(self) -> None:
        """디렉토리 생성"""
        today = date.today()
        month_dir = self.log_dir / today.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, trade_date: Optional[date] = None) -> Path:
        """파일 경로 반환"""
        d = trade_date or date.today()
        month_dir = self.log_dir / d.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)
        return month_dir / f"trades_{d.strftime('%Y-%m-%d')}.csv"
    
    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        """안전한 float 변환"""
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return default
        try:
            return float(s)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _norm_side(v: Any) -> str:
        """BUY/SELL 정규화"""
        s = ("" if v is None else str(v)).strip().upper()
        if s in ("B", "BUY", "LONG"):
            return "BUY"
        if s in ("S", "SELL", "SHORT"):
            return "SELL"
        return s
    
    @staticmethod
    def _sanitize_cell(v: Any) -> str:
        """
        CSV Formula Injection 방어
        Excel이 수식으로 해석하는 접두(= + - @)는 '로 이스케이프
        """
        s = "" if v is None else str(v)
        if s.startswith(("=", "+", "-", "@")):
            return "'" + s
        return s
    
    def log_trade(self, trade: Dict) -> bool:
        """
        거래 기록 (Thread Safe)
        
        Args:
            trade: dict with keys:
                - order_id, exchange, symbol, side
                - quantity, price, fee, pnl
                - status, message, timestamp (optional)
        
        Returns:
            bool: 성공 여부
        """
        file_path = self._get_file_path()
        file_exists = file_path.exists()
        
        row = {
            "timestamp": trade.get("timestamp", datetime.now().isoformat()),
            "exchange": self._sanitize_cell(trade.get("exchange", "unknown")),
            "order_id": self._sanitize_cell(trade.get("order_id", "")),
            "symbol": self._sanitize_cell(trade.get("symbol", "")),
            "side": self._norm_side(trade.get("side", "")),
            "quantity": self._to_float(trade.get("quantity", 0)),
            "price": self._to_float(trade.get("price", 0)),
            "fee": self._to_float(trade.get("fee", 0)),
            "pnl": self._to_float(trade.get("pnl", 0)),
            "status": self._sanitize_cell(trade.get("status", "")),
            "message": self._sanitize_cell(trade.get("message", "")),
        }
        
        try:
            with self._lock:
                with open(file_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()
            
            logger.debug(f"[TradeLogger] Logged: {row['symbol']} {row['side']}")
            return True
            
        except Exception as e:
            logger.error(f"[TradeLogger] Failed: {e}")
            return False
    
    def get_trades(self, trade_date: Optional[date] = None) -> List[Dict]:
        """특정 날짜 거래 조회"""
        file_path = self._get_file_path(trade_date)
        
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"[TradeLogger] Read failed: {e}")
            return []
    
    def get_daily_summary(self, trade_date: Optional[date] = None) -> Dict:
        """일일 거래 요약"""
        trades = self.get_trades(trade_date)
        d = trade_date or date.today()
        
        if not trades:
            return {
                "date": d.isoformat(),
                "total_trades": 0,
                "buy_count": 0,
                "sell_count": 0,
                "total_volume": 0.0,
                "total_fee": 0.0,
                "total_pnl": 0.0,
            }
        
        buy = sell = 0
        total_volume = total_fee = total_pnl = 0.0
        
        for t in trades:
            side = self._norm_side(t.get("side"))
            if side == "BUY":
                buy += 1
            elif side == "SELL":
                sell += 1
            
            qty = self._to_float(t.get("quantity", 0))
            price = self._to_float(t.get("price", 0))
            total_volume += qty * price
            total_fee += self._to_float(t.get("fee", 0))
            total_pnl += self._to_float(t.get("pnl", 0))
        
        return {
            "date": d.isoformat(),
            "total_trades": len(trades),
            "buy_count": buy,
            "sell_count": sell,
            "total_volume": total_volume,
            "total_fee": total_fee,
            "total_pnl": total_pnl,
        }
    
    def get_trade_count(self, trade_date: Optional[date] = None) -> int:
        """거래 횟수 반환"""
        return len(self.get_trades(trade_date))
