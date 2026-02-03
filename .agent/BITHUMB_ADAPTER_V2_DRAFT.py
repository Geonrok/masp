"""
BithumbExecutionAdapter v2.0 - AI 검수 피드백 반영

주요 수정사항:
1. quantity → amount_krw 기반 통일 (DeepSeek 피드백)
2. UnifiedOrderResult 베이스 클래스 (ChatGPT, DeepSeek 피드백)
3. 예외 처리 세분화 (모든 AI 피드백)
4. Rate Limit 추가 (Gemini 피드백)
5. datetime aware 통일 (ChatGPT 피드백)
6. Live ACK 추가 (ChatGPT 피드백)
"""

import logging
import os
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

try:
    import pybithumb
except ImportError:
    pybithumb = None

from libs.core.config import Config

logger = logging.getLogger(__name__)

# KST timezone
KST = timezone(timedelta(hours=9))


# ============================================================
# 공통 반환 타입 인터페이스 (모든 거래소 공통)
# ============================================================
@dataclass
class UnifiedOrderResult(ABC):
    """모든 거래소 공통 주문 결과 인터페이스"""

    exchange: str
    order_id: str
    symbol: str
    side: str
    status: str  # FILLED, PENDING, REJECTED
    message: str = ""

    # 확장 필드 (거래소별로 다를 수 있음)
    quantity: float = 0.0  # 코인 수량
    krw_amount: float = 0.0  # KRW 금액
    filled_price: float = 0.0
    fee: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(KST))


@dataclass
class BithumbOrderResult(UnifiedOrderResult):
    """Bithumb 주문 결과"""

    exchange: str = "bithumb"
    order_type: str = "MARKET"
    filled_quantity: float = 0.0


# ============================================================
# BithumbExecutionAdapter v2.0
# ============================================================
class BithumbExecutionAdapter:
    """
    Bithumb 실주문 어댑터 v2.0

    주요 변경사항:
    - place_order(amount_krw)로 통일 (BUY/SELL 모두 KRW 금액 기준)
    - 내부에서 코인 수량 자동 계산
    - Rate Limit 처리 추가
    - 예외 처리 세분화
    """

    FEE_RATE = 0.0025  # 0.25%
    MIN_ORDER_KRW = 1000  # Bithumb 최소 주문 금액
    REQUEST_INTERVAL = 0.1  # 100ms

    def __init__(self, config: Config, **kwargs):
        """초기화 - kwargs 무시하여 TypeError 방지"""
        if pybithumb is None:
            raise ImportError("pybithumb not installed. Run: pip install pybithumb")

        # Live ACK 체크 (ChatGPT 피드백)
        if os.getenv("MASP_ENABLE_LIVE_TRADING") == "1":
            if os.getenv("MASP_ACK_BITHUMB_LIVE") != "1":
                logger.warning(
                    "[BithumbExecution] MASP_ACK_BITHUMB_LIVE=1 권장. "
                    "Live 거래 ACK 없이 진행합니다."
                )

        self.config = config
        self._validate_config()

        # pybithumb 인스턴스
        api_key = config.bithumb_api_key.get_secret_value()
        secret_key = config.bithumb_secret_key.get_secret_value()
        self.bithumb = pybithumb.Bithumb(api_key, secret_key)

        self._trade_logger = None
        self._last_request_time: Optional[datetime] = None

        logger.info("[BithumbExecution] Adapter v2.0 initialized")

    def _validate_config(self):
        """설정 검증 - 강화된 버전 (DeepSeek 피드백)"""
        try:
            api_key = self.config.bithumb_api_key.get_secret_value()
            secret_key = self.config.bithumb_secret_key.get_secret_value()
        except AttributeError as e:
            raise ValueError(f"Config에 bithumb_api_key/secret_key 없음: {e}")

        # 기본 검증
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("BITHUMB_API_KEY not set or invalid")
        if not secret_key or secret_key == "your_secret_key_here":
            raise ValueError("BITHUMB_SECRET_KEY not set or invalid")

        # 테스트 키 차단 (DeepSeek 피드백)
        test_patterns = ["test", "demo", "example", "placeholder"]
        if any(pattern in api_key.lower() for pattern in test_patterns):
            raise ValueError("테스트 API 키는 사용할 수 없습니다")

        # 최소 길이 검증
        if len(api_key) < 20 or len(secret_key) < 20:
            raise ValueError("API 키 형식이 올바르지 않습니다")

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
        except ConnectionError as e:
            logger.error(f"[BithumbExecution] Network error: {e}")
            return 0.0
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
        except ConnectionError as e:
            logger.error(f"[BithumbExecution] Network error: {e}")
            return None
        except Exception as e:
            logger.error(f"[BithumbExecution] Price query failed: {e}")
            return None

    # ========== 주문 기능 (핵심 수정) ==========

    def place_order(
        self,
        symbol: str,
        side: str,
        amount_krw: float,  # ✅ KRW 금액 기준으로 통일
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> BithumbOrderResult:
        """
        주문 실행

        Args:
            symbol: 종목 (예: "BTC/KRW")
            side: "BUY" 또는 "SELL"
            amount_krw: 주문 금액 (KRW)
            order_type: "MARKET" 또는 "LIMIT"
            price: 지정가 주문 시 가격

        Returns:
            BithumbOrderResult

        Note:
            - BUY: amount_krw만큼 매수
            - SELL: amount_krw에 해당하는 코인 수량 매도
        """
        # 1. Kill-Switch 체크
        if self.config.is_kill_switch_active():
            logger.warning("[BithumbExecution] Kill-Switch active")
            return self._rejected_order(symbol, side, amount_krw, "Kill-Switch active")

        # 2. 최소 주문 금액 체크 (Gemini 피드백: UnderMinTotalBid 방지)
        if amount_krw < self.MIN_ORDER_KRW:
            return self._rejected_order(
                symbol,
                side,
                amount_krw,
                f"최소 주문 금액 미달: {amount_krw:,.0f} < {self.MIN_ORDER_KRW:,.0f} KRW",
            )

        # 3. 현재가 조회
        current_price = price or self.get_current_price(symbol)
        if current_price is None:
            return self._rejected_order(symbol, side, amount_krw, "Price unavailable")

        # 4. 코인 수량 계산
        coin_quantity = amount_krw / current_price

        # 5. 최대 주문 금액 체크
        max_order = int(getattr(self.config, "max_order_value_krw", 1_000_000))
        if amount_krw > max_order:
            return self._rejected_order(
                symbol,
                side,
                amount_krw,
                f"최대 주문 금액 초과: {amount_krw:,.0f} > {max_order:,.0f} KRW",
            )

        # 6. 주문 실행
        try:
            self._rate_limit()
            ticker = self._convert_symbol(symbol)
            result = None

            if order_type == "MARKET":
                if side.upper() == "BUY":
                    # ✅ 핵심 수정: BUY는 KRW 금액 전달 (DeepSeek 피드백)
                    result = self.bithumb.buy_market_order(ticker, amount_krw)
                else:
                    # SELL은 코인 수량 전달
                    result = self.bithumb.sell_market_order(ticker, coin_quantity)
            else:  # LIMIT
                if side.upper() == "BUY":
                    result = self.bithumb.buy_limit_order(ticker, price, coin_quantity)
                else:
                    result = self.bithumb.sell_limit_order(ticker, price, coin_quantity)

            order_result = self._parse_result(
                result,
                symbol,
                side,
                coin_quantity,
                amount_krw,
                order_type,
                current_price,
            )

            if self._trade_logger and order_result.status != "REJECTED":
                self._log_trade(order_result)

            logger.info(
                f"[BithumbExecution] {side} {symbol}: "
                f"{amount_krw:,.0f} KRW ({coin_quantity:.8f} coins) -> {order_result.status}"
            )

            return order_result

        except pybithumb.BithumbError as e:
            logger.error(f"[BithumbExecution] Bithumb API error: {e}")
            return self._rejected_order(symbol, side, amount_krw, f"API Error: {e}")
        except ConnectionError as e:
            logger.error(f"[BithumbExecution] Network error: {e}")
            return self._rejected_order(symbol, side, amount_krw, f"Network Error: {e}")
        except ValueError as e:
            logger.error(f"[BithumbExecution] Invalid parameter: {e}")
            return self._rejected_order(
                symbol, side, amount_krw, f"Invalid Parameter: {e}"
            )
        except Exception as e:
            logger.error(f"[BithumbExecution] Unexpected error: {e}", exc_info=True)
            return self._rejected_order(
                symbol, side, amount_krw, f"Unexpected Error: {e}"
            )

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
                symbol, side, krw_amount, "Order failed - null response"
            )

        # pybithumb은 order_id만 반환하는 경우가 많음
        order_id = result if isinstance(result, str) else str(result)
        fee = krw_amount * self.FEE_RATE

        return BithumbOrderResult(
            exchange="bithumb",
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            order_type=order_type,
            quantity=quantity,
            krw_amount=krw_amount,
            filled_quantity=quantity,
            filled_price=price,
            fee=fee,
            status="FILLED" if order_type == "MARKET" else "PENDING",
            created_at=datetime.now(KST),  # ✅ timezone-aware (ChatGPT 피드백)
            message="Order placed successfully",
        )

    def _rejected_order(self, symbol, side, krw_amount, reason) -> BithumbOrderResult:
        """거부된 주문"""
        logger.warning(
            f"[BithumbExecution] REJECTED: {symbol} {side} {krw_amount:,.0f} KRW - {reason}"
        )
        return BithumbOrderResult(
            exchange="bithumb",
            order_id="",
            symbol=symbol,
            side=side.upper(),
            order_type="MARKET",
            quantity=0,
            krw_amount=krw_amount,
            filled_quantity=0,
            filled_price=0,
            fee=0,
            status="REJECTED",
            created_at=datetime.now(KST),
            message=reason,
        )

    def _log_trade(self, order: BithumbOrderResult):
        """TradeLogger에 기록"""
        self._trade_logger.log_trade(
            {
                "exchange": order.exchange,
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
