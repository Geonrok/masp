"""
BithumbExecutionAdapter v2.1 - 최종 확정 버전

AI 검수 합의:
- ChatGPT: 계약 강제, 게이트 테스트 권장
- Gemini: AWS ap-northeast-2, REST 안정성
- DeepSeek: StrategyRunner는 quantity(코인 수량) 전달, Adapter 내부 변환
- Perplexity: 배포 승인

핵심 수정:
- place_order()는 quantity(코인 수량)를 받음 (StrategyRunner와 일관성)
- BUY시 Adapter 내부에서 quantity → krw_amount 변환
- SELL시 quantity 그대로 전달
"""

import logging
import os
import time
from typing import Optional, Dict
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

try:
    import pybithumb
except ImportError:
    pybithumb = None

from libs.core.config import Config

logger = logging.getLogger(__name__)

# KST timezone
KST = timezone(timedelta(hours=9))


@dataclass
class BithumbOrderResult:
    """Bithumb 주문 결과"""

    order_id: str
    symbol: str
    side: str
    order_type: str = "MARKET"
    quantity: float = 0.0  # 코인 수량
    krw_amount: float = 0.0  # KRW 금액 (로깅/리포트용)
    price: Optional[float] = None
    status: str = "UNKNOWN"
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(KST))
    message: str = ""


class BithumbExecutionAdapter:
    """
    Bithumb 실주문 어댑터 v2.1

    인터페이스 계약:
        place_order(symbol, side, quantity, ...)
        - quantity: 항상 "코인 수량" (StrategyRunner와 일관성)
        - BUY: 내부에서 quantity * price = KRW로 변환 후 API 호출
        - SELL: quantity 그대로 API 전달

    안전장치:
        - Kill-Switch 체크
        - MASP_ENABLE_LIVE_TRADING 체크 (Factory에서)
        - 최소/최대 주문 금액 검증
        - Rate Limit (0.1초 간격)
    """

    FEE_RATE = 0.0025  # 0.25%
    MIN_ORDER_KRW = 1000  # Bithumb 최소 주문 금액
    REQUEST_INTERVAL = 0.1  # 100ms

    def __init__(self, config: Config, **kwargs):
        """
        초기화

        Note: **kwargs는 무시됨 (TypeError 방지, ChatGPT 피드백)
        """
        if pybithumb is None:
            raise ImportError("pybithumb not installed. Run: pip install pybithumb")

        self.config = config
        self._validate_config()

        # pybithumb 인스턴스
        api_key = config.bithumb_api_key.get_secret_value()
        secret_key = config.bithumb_secret_key.get_secret_value()
        self.bithumb = pybithumb.Bithumb(api_key, secret_key)

        self._trade_logger = None
        self._last_request_time: Optional[datetime] = None

        logger.info("[BithumbExecution] Adapter v2.1 initialized")

    def _validate_config(self):
        """설정 검증 (DeepSeek 피드백 반영)"""
        try:
            api_key = self.config.bithumb_api_key.get_secret_value()
            secret_key = self.config.bithumb_secret_key.get_secret_value()
        except AttributeError as e:
            raise ValueError(f"Config에 bithumb_api_key/secret_key 없음: {e}")

        if not api_key or api_key == "your_api_key_here":
            raise ValueError("BITHUMB_API_KEY not set or invalid")
        if not secret_key or secret_key == "your_secret_key_here":
            raise ValueError("BITHUMB_SECRET_KEY not set or invalid")

        # 테스트 키 차단
        test_patterns = ["test", "demo", "example", "placeholder"]
        if any(pattern in api_key.lower() for pattern in test_patterns):
            raise ValueError("테스트 API 키는 사용할 수 없습니다")

    def set_trade_logger(self, trade_logger):
        """TradeLogger 설정"""
        self._trade_logger = trade_logger

    def _rate_limit(self):
        """Rate Limit 처리 (Gemini 피드백)"""
        if self._last_request_time:
            elapsed = (datetime.now(KST) - self._last_request_time).total_seconds()
            if elapsed < self.REQUEST_INTERVAL:
                time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = datetime.now(KST)

    # ========== 조회 기능 ==========

    def get_balance(self, currency: str = "KRW") -> float:
        """잔고 조회"""
        try:
            self._rate_limit()
            balance = self.bithumb.get_balance(currency)
            if balance is None:
                return 0.0
            return float(balance[0]) if isinstance(balance, tuple) else float(balance)
        except Exception as e:
            logger.error(f"[BithumbExecution] Balance query failed: {e}")
            return 0.0

    def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        try:
            self._rate_limit()
            ticker = self._convert_symbol(symbol)
            price = pybithumb.get_current_price(ticker)
            return float(price) if price else None
        except Exception as e:
            logger.error(f"[BithumbExecution] Price query failed: {e}")
            return None

    # ========== 주문 기능 (핵심) ==========

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,  # ✅ 코인 수량 (StrategyRunner와 일관성)
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> BithumbOrderResult:
        """
        주문 실행

        Args:
            symbol: 종목 (예: "BTC/KRW")
            side: "BUY" 또는 "SELL"
            quantity: 코인 수량 (예: 0.001 BTC)
                      - StrategyRunner가 position_size_krw / price로 계산하여 전달
            order_type: "MARKET" 또는 "LIMIT"
            price: 지정가 주문 시 가격

        Returns:
            BithumbOrderResult

        Internal Logic:
            - BUY: quantity * current_price = KRW 금액 → pybithumb.buy_market_order(ticker, krw_amount)
            - SELL: quantity 그대로 → pybithumb.sell_market_order(ticker, quantity)
        """
        # 1. Kill-Switch 체크
        if self.config.is_kill_switch_active():
            logger.warning("[BithumbExecution] Kill-Switch active")
            return self._rejected_order(symbol, side, quantity, 0, "Kill-Switch active")

        # 2. 현재가 조회
        current_price = price or self.get_current_price(symbol)
        if current_price is None:
            return self._rejected_order(symbol, side, quantity, 0, "Price unavailable")

        # 3. KRW 금액 계산 (DeepSeek 핵심 피드백)
        krw_amount = quantity * current_price

        # 4. side별 로깅 (디버깅용)
        if side.upper() == "BUY":
            logger.info(
                f"[BithumbExecution] BUY: {quantity:.8f} coins = {krw_amount:,.0f} KRW"
            )
        else:
            logger.info(
                f"[BithumbExecution] SELL: {quantity:.8f} coins (≈ {krw_amount:,.0f} KRW)"
            )

        # 5. 최소 주문 금액 체크 (Gemini 피드백: UnderMinTotalBid 방지)
        if krw_amount < self.MIN_ORDER_KRW:
            return self._rejected_order(
                symbol,
                side,
                quantity,
                krw_amount,
                f"최소 주문 금액 미달: {krw_amount:,.0f} < {self.MIN_ORDER_KRW:,.0f} KRW",
            )

        # 6. 최대 주문 금액 체크
        max_order = int(getattr(self.config, "max_order_value_krw", 1_000_000))
        if krw_amount > max_order:
            return self._rejected_order(
                symbol,
                side,
                quantity,
                krw_amount,
                f"최대 주문 금액 초과: {krw_amount:,.0f} > {max_order:,.0f} KRW",
            )

        # 7. 주문 실행
        try:
            self._rate_limit()
            ticker = self._convert_symbol(symbol)
            result = None

            if order_type == "MARKET":
                if side.upper() == "BUY":
                    # ✅ 핵심: BUY는 KRW 금액 전달 (pybithumb API 요구사항)
                    result = self.bithumb.buy_market_order(ticker, krw_amount)
                else:
                    # SELL은 코인 수량 전달
                    result = self.bithumb.sell_market_order(ticker, quantity)
            else:  # LIMIT
                if side.upper() == "BUY":
                    result = self.bithumb.buy_limit_order(ticker, price, krw_amount)
                else:
                    result = self.bithumb.sell_limit_order(ticker, price, quantity)

            order_result = self._parse_result(
                result, symbol, side, quantity, krw_amount, order_type, current_price
            )

            if self._trade_logger and order_result.status != "REJECTED":
                self._log_trade(order_result)

            logger.info(
                f"[BithumbExecution] Order completed: {side} {symbol} "
                f"qty={quantity:.8f} krw={krw_amount:,.0f} → {order_result.status}"
            )

            return order_result

        except Exception as e:
            logger.error(f"[BithumbExecution] Order failed: {e}", exc_info=True)
            return self._rejected_order(symbol, side, quantity, krw_amount, str(e))

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """주문 취소"""
        try:
            self._rate_limit()
            ticker = self._convert_symbol(symbol)
            result = self.bithumb.cancel_order(order_id, ticker)
            return result is not None
        except Exception as e:
            logger.error(f"[BithumbExecution] Cancel failed: {e}")
            return False

    # ========== Private Methods ==========

    def _convert_symbol(self, symbol: str) -> str:
        """심볼 변환: BTC/KRW -> BTC"""
        return symbol.split("/")[0]

    def _parse_result(
        self, result, symbol, side, quantity, krw_amount, order_type, price
    ) -> BithumbOrderResult:
        """API 응답 파싱"""
        if result is None:
            return self._rejected_order(
                symbol, side, quantity, krw_amount, "Order failed - null response"
            )

        order_id = result if isinstance(result, str) else str(result)
        fee = krw_amount * self.FEE_RATE

        return BithumbOrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            order_type=order_type,
            quantity=quantity,
            krw_amount=krw_amount,
            price=price,
            status="FILLED" if order_type == "MARKET" else "PENDING",
            filled_quantity=quantity,
            filled_price=price,
            fee=fee,
            created_at=datetime.now(KST),
            message="Order placed successfully",
        )

    def _rejected_order(
        self, symbol, side, quantity, krw_amount, reason
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

    def _log_trade(self, order: BithumbOrderResult):
        """TradeLogger에 기록"""
        self._trade_logger.log_trade(
            {
                "exchange": "bithumb",
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.filled_quantity,
                "krw_amount": order.krw_amount,
                "price": order.filled_price,
                "fee": order.fee,
                "pnl": 0,
                "status": order.status,
                "message": order.message,
                "timestamp": order.created_at.isoformat(),
            }
        )
