"""
Bithumb 실주문 어댑터
- 잔고 조회
- 시장가/지정가 주문
- Kill-Switch 연동
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass

from libs.adapters.bithumb_api_v2 import BithumbAPIV2
from libs.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class BithumbOrderResult:
    """Bithumb 주문 결과"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    filled_price: float
    fee: float
    created_at: datetime
    message: str = ""


class BithumbExecutionAdapter:
    """
    Bithumb 실주문 어댑터
    
    Features:
    - Kill-Switch 연동
    - TradeLogger 자동 기록
    - 잔고/주문 조회
    """
    
    FEE_RATE = 0.0025  # 0.25% (Bithumb 기본)
    
    def __init__(self, config: Config):
        """초기화"""
        self.config = config
        self._validate_config()
        
        # Bithumb API v2 인스턴스 생성
        api_key = config.bithumb_api_key.get_secret_value()
        secret_key = config.bithumb_secret_key.get_secret_value()
        self.bithumb = BithumbAPIV2(api_key, secret_key)
        
        self._trade_logger = None
        logger.info("[BithumbExecution] Adapter initialized")
    
    def _validate_config(self):
        """설정 검증"""
        api_key = self.config.bithumb_api_key.get_secret_value()
        secret_key = self.config.bithumb_secret_key.get_secret_value()
        
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("BITHUMB_API_KEY not set or invalid")
        if not secret_key or secret_key == "your_secret_key_here":
            raise ValueError("BITHUMB_SECRET_KEY not set or invalid")
    
    def set_trade_logger(self, trade_logger):
        """TradeLogger 설정"""
        self._trade_logger = trade_logger
    
    # ========== 조회 기능 ==========
    
    def get_balance(self, currency: str = "KRW") -> float:
        """잔고 조회"""
        try:
            accounts = self.bithumb.get_accounts()
            for account in accounts:
                if account.get("currency") == currency:
                    return float(account.get("balance", 0))
            return 0.0
        except Exception as e:
            logger.error(f"[BithumbExecution] Balance query failed: {e}")
            return 0.0
    
    def get_all_balances(self) -> Dict[str, float]:
        """모든 잔고 조회"""
        try:
            accounts = self.bithumb.get_accounts()
            balances: Dict[str, float] = {}
            for account in accounts:
                currency = account.get("currency")
                if not currency:
                    continue
                balance = float(account.get("balance", 0))
                if balance > 0:
                    balances[currency] = balance
            return balances
        except Exception as e:
            logger.error(f"[BithumbExecution] All balances failed: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        try:
            market = self._convert_symbol(symbol)
            result = self.bithumb.get_ticker([market])
            if isinstance(result, list) and result:
                price = result[0].get("trade_price")
            elif isinstance(result, dict):
                price = result.get("trade_price")
            else:
                price = None
            return float(price) if price else None
        except Exception as e:
            logger.error(f"[BithumbExecution] Price query failed: {e}")
            return None
    
    # ========== 주문 기능 ==========
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float = None,  # 기존 호환성 (deprecated)
        order_type: str = "MARKET",
        price: Optional[float] = None,
        *,  # keyword-only 강제 (ChatGPT 권장)
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,
    ) -> BithumbOrderResult:
        """
        주문 실행
        
        Args:
            symbol: 종목 (예: "BTC/KRW")
            side: "BUY" 또는 "SELL"
            quantity: [DEPRECATED] 코인 수량 - units 사용 권장
            units: 코인 수량 (BUY/SELL 모두 사용 가능)
            amount_krw: KRW 금액 (BUY 전용, 내부에서 units로 변환)
            order_type: "MARKET" 또는 "LIMIT"
            price: 지정가 주문 시 가격
        
        Contract (ChatGPT 게이트 요구사항):
            - BUY: (units XOR amount_krw) 중 하나만 허용
            - SELL: units만 허용 (amount_krw 금지)
            - quantity는 units로 매핑 (하위 호환)
        """
        # 0. [ChatGPT 필수 수정 #1] Live 모드에서 deprecated quantity 거부
        # quantity(positional)는 "KRW를 unit으로 오인"하는 사고 백도어
        if quantity is not None:
            raise ValueError(
                "DEPRECATED: 'quantity' parameter is not allowed in Live mode. "
                "Use 'units=' for coin quantity or 'amount_krw=' for KRW amount."
            )
        
        # 하위 호환은 Paper 모드에서만 허용 (이 어댑터는 Live 전용)
        
        # 1. 계약 강제 (ChatGPT 권장: 상호배타 체크)
        side_u = side.upper()
        if side_u == "BUY":
            if units is None and amount_krw is None:
                raise ValueError("BUY requires exactly one of: units or amount_krw")
            if units is not None and amount_krw is not None:
                raise ValueError("BUY requires exactly one of: units or amount_krw (not both)")
        else:  # SELL
            if units is None:
                raise ValueError("SELL requires units")
            if amount_krw is not None:
                raise ValueError("SELL does not accept amount_krw (use units only)")
        
        # 3. Kill-Switch 체크
        if self.config.is_kill_switch_active():
            logger.warning("[BithumbExecution] Kill-Switch active")
            return self._rejected_order(symbol, side_u, units or 0, "Kill-Switch active")
        
        # 4. 현재가 조회
        current_price = price or self.get_current_price(symbol)
        if current_price is None:
            return self._rejected_order(symbol, side_u, units or 0, "Price unavailable")
        
        # 5. amount_krw → units 변환 (ChatGPT 권장)
        if amount_krw is not None and units is None:
            # BUY에서 amount_krw 사용 시 units로 변환
            fee_buffer = 0.003  # 수수료/슬리피지 버퍼
            units = (float(amount_krw) * (1 - fee_buffer)) / float(current_price)
            logger.info(f"[BithumbExecution] BUY: {amount_krw:,.0f} KRW → {units:.8f} units (fee buffer: {fee_buffer})")
        
        # 6. KRW 금액 계산
        krw_amount = float(amount_krw) if amount_krw is not None else units * current_price
        
        # 7. side별 로깅 (디버깅용)
        if side_u == "BUY":
            logger.info(f"[BithumbExecution] BUY: {units:.8f} coins = {krw_amount:,.0f} KRW")
        else:
            logger.info(f"[BithumbExecution] SELL: {units:.8f} coins")
        
        # 5. 최소 주문 금액 체크 (Bithumb: 5,000 KRW - Gemini 권장)
        MIN_ORDER_KRW = 5000
        if krw_amount < MIN_ORDER_KRW:
            return self._rejected_order(
                symbol, side_u, units,
                f"최소 주문 금액 미달: {krw_amount:,.0f} < {MIN_ORDER_KRW:,} KRW"
            )
        
        # 9. 최대 주문 금액 체크
        max_order = int(getattr(self.config, 'max_order_value_krw', 1_000_000))
        if krw_amount > max_order:
            return self._rejected_order(
                symbol, side_u, units,
                f"Order value {krw_amount:,.0f} exceeds limit {max_order:,.0f}"
            )
        
        # 7. 주문 실행
        try:
            market = self._convert_symbol(symbol)
            result = None

            order_type_u = order_type.upper()
            if order_type_u == "MARKET":
                if side_u == "BUY":
                    amount_value = float(amount_krw) if amount_krw is not None else float(krw_amount)
                    result = self.bithumb.post_order(
                        market=market,
                        side="bid",
                        ord_type="price",
                        price=str(int(amount_value)),
                    )
                else:
                    result = self.bithumb.post_order(
                        market=market,
                        side="ask",
                        ord_type="market",
                        volume=f"{units:.8f}",
                    )
            else:  # LIMIT
                result = self.bithumb.post_order(
                    market=market,
                    side="bid" if side_u == "BUY" else "ask",
                    ord_type="limit",
                    volume=f"{units:.8f}",
                    price=str(price),
                )
            # [ChatGPT ?? #1 ??] quantity ===> units ??
            order_result = self._parse_result(result, symbol, side, units, order_type_u, current_price)
            
            # [ChatGPT 잔여 이슈 #2] UNKNOWN 주문도 TradeLogger 기록 방지
            # status가 FILLED 또는 PENDING인 경우만 기록 (REJECTED, UNKNOWN 제외)
            if self._trade_logger and order_result.status in ("FILLED", "PENDING"):
                self._log_trade(order_result)
            
            return order_result
            
        except Exception as e:
            logger.error(f"[BithumbExecution] Order failed: {e}")
            return self._rejected_order(symbol, side, units or 0, str(e))
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """주문 취소"""
        try:
            result = self.bithumb.cancel_order(order_id)
            return result is not None
        except Exception as e:
            logger.error(f"[BithumbExecution] Cancel failed: {e}")
            return False
    
    # ========== Private Methods ==========
    
    def _convert_symbol(self, symbol: str) -> str:
        """심볼 변환: BTC/KRW -> KRW-BTC"""
        base, quote = symbol.split("/")
        return f"{quote}-{base}"
    
    def _parse_result(self, result, symbol, side, quantity, order_type, price) -> BithumbOrderResult:
        """
        API 응답 파싱
        
        [ChatGPT 권장 보강 A] 실제 order_id 추출
        
        Bithumb API v2 반환값 형식:
        - 성공: {"uuid": "...", ...}
        - 실패: None 또는 에러 dict
        """
        if result is None:
            return self._rejected_order(symbol, side, quantity, "Order failed: None response")
        
        order_id = None
        raw_response = str(result)  # ?? ?? ???

        if isinstance(result, dict):
            order_id = result.get("uuid") or result.get("order_id") or result.get("orderId")
            if not order_id and "error" in result:
                error = result.get("error")
                if isinstance(error, dict):
                    error_msg = error.get("message") or error.get("msg") or str(error)
                else:
                    error_msg = str(error)
                return self._rejected_order(symbol, side, quantity, f"API error: {error_msg}")
        elif isinstance(result, str):
            order_id = result
        else:
            order_id = str(result)
            logger.warning(f"[BithumbExecution] Unknown result format: {type(result)}")
        
        # order_id 유효성 검사 (ChatGPT 권고: 심볼로 fallback 방지)
        if not order_id or order_id == symbol or order_id == "None":
            logger.warning(f"[BithumbExecution] Invalid order_id: {order_id}, raw: {raw_response}")
            # [ChatGPT 추가 권고] UNKNOWN 대신 status=UNKNOWN 처리
            # 취소/추적 불가능한 주문을 "성공"으로 간주하지 않음
            return BithumbOrderResult(
                order_id=f"UNKNOWN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                symbol=symbol,
                side=side.upper(),
                order_type=order_type,
                quantity=quantity,
                price=price,
                status="UNKNOWN",  # ✅ FILLED가 아닌 UNKNOWN
                filled_quantity=quantity,
                filled_price=price,
                fee=(quantity or 0) * (price or 0) * self.FEE_RATE,
                created_at=datetime.now(),
                message=f"Order may have succeeded but order_id invalid. Raw: {raw_response[:100]}"
            )
        
        fee = (quantity or 0) * (price or 0) * self.FEE_RATE
        
        return BithumbOrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="FILLED" if order_type == "MARKET" else "PENDING",
            filled_quantity=quantity,
            filled_price=price,
            fee=fee,
            created_at=datetime.now(),
            message=f"Order placed successfully. Raw: {raw_response[:100]}"
        )
    
    def _rejected_order(self, symbol, side, quantity, reason) -> BithumbOrderResult:
        """거부된 주문"""
        logger.warning(f"[BithumbExecution] REJECTED: {symbol} {side} {quantity} - {reason}")
        return BithumbOrderResult(
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
    
    def _log_trade(self, order: BithumbOrderResult):
        """TradeLogger에 기록"""
        self._trade_logger.log_trade({
            "exchange": "bithumb",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.filled_quantity,
            "price": order.filled_price,
            "fee": order.fee,
            "pnl": 0,
            "status": order.status,
            "message": order.message
        })
