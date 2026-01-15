"""
Upbit 실주문 어댑터
- 잔고 조회
- 시장가/지정가 주문
- 주문 취소
- Kill-Switch 연동
- OrderValidator 검증
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass

try:
    import pyupbit
except ImportError:
    pyupbit = None

from libs.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class UpbitOrderResult:
    """Upbit 주문 결과"""
    order_id: str
    symbol: str
    side: str  # BUY | SELL
    order_type: str  # MARKET | LIMIT
    quantity: float
    price: Optional[float]
    status: str  # PENDING | FILLED | CANCELLED | REJECTED
    filled_quantity: float
    filled_price: float
    fee: float
    created_at: datetime
    message: str = ""


class UpbitExecutionAdapter:
    """
    Upbit 실주문 어댑터
    
    Features:
    - Kill-Switch 연동
    - OrderValidator 검증
    - TradeLogger 자동 기록
    - 잔고/주문 조회
    
    Usage:
        config = Config(asset_class="spot", strategy_name="test")
        adapter = UpbitExecutionAdapter(config)
        
        # 잔고 조회
        krw = adapter.get_balance("KRW")
        btc = adapter.get_balance("BTC")
        
        # 주문 (Kill-Switch 체크 포함)
        result = adapter.place_order("BTC/KRW", "BUY", 0.0001)
    """
    
    # 수수료율 (Upbit 기본)
    FEE_RATE = 0.0005  # 0.05%
    
    def __init__(self, config: Config):
        """
        초기화
        
        Args:
            config: Config 객체 (API 키 포함)
        
        Raises:
            ImportError: pyupbit 미설치
            ValueError: API 키 미설정
        """
        if pyupbit is None:
            raise ImportError("pyupbit not installed. Run: pip install pyupbit")
        
        self.config = config
        self._validate_config()
        
        # pyupbit 인스턴스 생성
        access_key = config.upbit_access_key.get_secret_value()
        secret_key = config.upbit_secret_key.get_secret_value()
        self.upbit = pyupbit.Upbit(access_key, secret_key)
        
        # TradeLogger (옵션)
        self._trade_logger = None
        
        logger.info("[UpbitExecution] Adapter initialized")
    
    def _validate_config(self):
        """설정 검증"""
        access = self.config.upbit_access_key.get_secret_value()
        secret = self.config.upbit_secret_key.get_secret_value()
        
        if not access or access == "your_access_key_here":
            raise ValueError("UPBIT_ACCESS_KEY not set or invalid")
        if not secret or secret == "your_secret_key_here":
            raise ValueError("UPBIT_SECRET_KEY not set or invalid")
    
    def set_trade_logger(self, trade_logger):
        """TradeLogger 설정"""
        self._trade_logger = trade_logger
    
    # ========== 조회 기능 ==========
    
    def get_balance(self, currency: str = "KRW") -> float:
        """
        잔고 조회
        
        Args:
            currency: 통화 코드 (KRW, BTC, ETH 등)
            
        Returns:
            float: 잔고 (없으면 0.0)
        """
        try:
            balances = self.upbit.get_balances()
            
            if balances is None:
                logger.error("[UpbitExecution] Failed to get balances")
                return 0.0
            
            for b in balances:
                if b.get('currency') == currency:
                    return float(b.get('balance', 0))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"[UpbitExecution] Balance query failed: {e}")
            return 0.0
    
    def get_all_balances(self) -> List[Dict]:
        """모든 잔고 조회"""
        try:
            balances = self.upbit.get_balances()
            return balances if balances else []
        except Exception as e:
            logger.error(f"[UpbitExecution] All balances query failed: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        현재가 조회
        
        Args:
            symbol: 종목 (예: "BTC/KRW")
            
        Returns:
            float: 현재가 (실패 시 None)
        """
        try:
            ticker = self._convert_symbol(symbol)
            price = pyupbit.get_current_price(ticker)
            return float(price) if price else None
        except Exception as e:
            logger.error(f"[UpbitExecution] Price query failed: {e}")
            return None
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """주문 조회"""
        try:
            return self.upbit.get_order(order_id)
        except Exception as e:
            logger.error(f"[UpbitExecution] Order query failed: {e}")
            return None
    
    # ========== 주문 기능 ==========
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None
    ) -> UpbitOrderResult:
        """
        주문 실행
        
        Args:
            symbol: 종목 (예: "BTC/KRW")
            side: "BUY" | "SELL"
            quantity: 수량
            order_type: "MARKET" | "LIMIT"
            price: 지정가 주문 시 가격
            
        Returns:
            UpbitOrderResult: 주문 결과
        """
        # 1. Kill-Switch 체크
        if self.config.is_kill_switch_active():
            logger.warning("[UpbitExecution] Kill-Switch active, order rejected")
            return self._rejected_order(symbol, side, quantity, "Kill-Switch active")
        
        # 2. 현재가 조회
        current_price = price or self.get_current_price(symbol)
        if current_price is None:
            return self._rejected_order(symbol, side, quantity, "Price unavailable")
        
        # 3. 주문 금액 검증 (MAX_ORDER_VALUE_KRW)
        order_value = quantity * current_price
        max_order = int(self.config.max_order_value_krw) if hasattr(self.config, 'max_order_value_krw') else 1_000_000
        
        if order_value > max_order:
            return self._rejected_order(
                symbol, side, quantity, 
                f"Order value {order_value:,.0f} exceeds limit {max_order:,.0f}"
            )
        
        # 4. 주문 실행
        try:
            ticker = self._convert_symbol(symbol)
            result = None
            
            if order_type == "MARKET":
                if side.upper() == "BUY":
                    # 시장가 매수: 총액 기준
                    result = self.upbit.buy_market_order(ticker, order_value)
                else:
                    # 시장가 매도: 수량 기준
                    result = self.upbit.sell_market_order(ticker, quantity)
            else:  # LIMIT
                if side.upper() == "BUY":
                    result = self.upbit.buy_limit_order(ticker, price, quantity)
                else:
                    result = self.upbit.sell_limit_order(ticker, price, quantity)
            
            # 5. 결과 파싱
            order_result = self._parse_result(result, symbol, side, quantity, order_type, current_price)
            
            # 6. TradeLogger 기록
            if self._trade_logger and order_result.status != "REJECTED":
                self._log_trade(order_result)
            
            return order_result
            
        except Exception as e:
            logger.error(f"[UpbitExecution] Order failed: {e}")
            return self._rejected_order(symbol, side, quantity, str(e))
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            result = self.upbit.cancel_order(order_id)
            return result is not None
        except Exception as e:
            logger.error(f"[UpbitExecution] Cancel failed: {e}")
            return False
    
    # ========== Private Methods ==========
    
    def _convert_symbol(self, symbol: str) -> str:
        """심볼 변환: BTC/KRW -> KRW-BTC"""
        base, quote = symbol.split("/")
        return f"{quote}-{base}"
    
    def _parse_result(
        self, 
        result: Dict, 
        symbol: str, 
        side: str, 
        quantity: float,
        order_type: str,
        price: float
    ) -> UpbitOrderResult:
        """API 응답 파싱"""
        if result is None or 'error' in result:
            error_msg = "Unknown error"
            if result and 'error' in result:
                error_msg = result['error'].get('message', 'Unknown error')
            return self._rejected_order(symbol, side, quantity, error_msg)
        
        # 체결 정보 추출
        filled_qty = float(result.get('executed_volume', quantity))
        filled_price = float(result.get('avg_price', price) or price)
        fee = filled_qty * filled_price * self.FEE_RATE
        
        return UpbitOrderResult(
            order_id=result.get('uuid', ''),
            symbol=symbol,
            side=side.upper(),
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="FILLED" if order_type == "MARKET" else "PENDING",
            filled_quantity=filled_qty,
            filled_price=filled_price,
            fee=fee,
            created_at=datetime.now(),
            message="Order placed successfully"
        )
    
    def _rejected_order(self, symbol: str, side: str, quantity: float, reason: str) -> UpbitOrderResult:
        """거부된 주문 생성"""
        logger.warning(f"[UpbitExecution] REJECTED: {symbol} {side} {quantity} - {reason}")
        return UpbitOrderResult(
            order_id="",
            symbol=symbol,
            side=side.upper(),
            order_type="MARKET",
            quantity=quantity,
            price=None,
            status="REJECTED",
            filled_quantity=0,
            filled_price=0,
            fee=0,
            created_at=datetime.now(),
            message=reason
        )
    
    def _log_trade(self, order: UpbitOrderResult):
        """TradeLogger에 기록"""
        self._trade_logger.log_trade({
            "exchange": "upbit",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.filled_quantity,
            "price": order.filled_price,
            "fee": order.fee,
            "pnl": 0,  # 실시간 PnL은 별도 계산 필요
            "status": order.status,
            "message": order.message
        })
