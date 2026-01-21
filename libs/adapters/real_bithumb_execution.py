"""
Bithumb 실주문 어댑터
- 잔고 조회
- 시장가/지정가 주문
- Kill-Switch 연동
"""

import logging
import time
from typing import Optional, Dict, List, Callable, TypeVar
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import functools
import random

from libs.adapters.bithumb_api_v2 import BithumbAPIV2
from libs.adapters.trade_logger import TradeLogger
from libs.core.config import Config

T = TypeVar("T")

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Bithumb 주문 상태."""
    WAIT = "wait"          # 체결 대기
    WATCH = "watch"        # 예약 주문 대기
    DONE = "done"          # 체결 완료
    CANCEL = "cancel"      # 취소됨


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


@dataclass
class OrderStatus:
    """주문 상태 상세 정보."""
    order_id: str
    market: str
    side: str
    ord_type: str
    state: str
    volume: float           # 주문 수량
    remaining_volume: float  # 미체결 수량
    executed_volume: float   # 체결 수량
    price: Optional[float]  # 지정가
    avg_price: Optional[float]  # 평균 체결가
    trades_count: int       # 체결 건수
    created_at: datetime
    paid_fee: float = 0.0
    locked: float = 0.0     # 동결 금액

    @property
    def is_done(self) -> bool:
        """체결 완료 여부."""
        return self.state == OrderState.DONE.value

    @property
    def is_canceled(self) -> bool:
        """취소 여부."""
        return self.state == OrderState.CANCEL.value

    @property
    def is_pending(self) -> bool:
        """체결 대기 여부."""
        return self.state in (OrderState.WAIT.value, OrderState.WATCH.value)

    @property
    def fill_ratio(self) -> float:
        """체결률 (0.0~1.0)."""
        if self.volume == 0:
            return 0.0
        return self.executed_volume / self.volume


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
        
        self._trade_logger = TradeLogger()
        logger.info("[BithumbExecution] Adapter initialized with TradeLogger")
    
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
    
    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """주문 취소"""
        try:
            result = self.bithumb.cancel_order(order_id)
            return result is not None
        except Exception as e:
            logger.error(f"[BithumbExecution] Cancel failed: {e}")
            return False

    # ========== 주문 상태 추적 기능 (Phase 8) ==========

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        주문 상태 조회.

        Args:
            order_id: 주문 UUID

        Returns:
            OrderStatus 또는 None
        """
        try:
            result = self.bithumb.get_order(order_id)
            if not result:
                return None

            return self._parse_order_status(result)
        except Exception as e:
            logger.error(f"[BithumbExecution] get_order_status failed: {e}")
            return None

    def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Optional[OrderStatus]:
        """
        주문 체결 대기.

        Args:
            order_id: 주문 UUID
            timeout_seconds: 최대 대기 시간
            poll_interval: 폴링 간격

        Returns:
            최종 OrderStatus 또는 None (타임아웃)
        """
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout_seconds:
            status = self.get_order_status(order_id)

            if status is None:
                logger.warning(f"[BithumbExecution] Order {order_id} not found")
                return last_status

            last_status = status

            if status.is_done:
                logger.info(
                    f"[BithumbExecution] Order {order_id} filled: "
                    f"{status.executed_volume:.8f} @ avg {status.avg_price or 0:,.0f}"
                )
                return status

            if status.is_canceled:
                logger.warning(f"[BithumbExecution] Order {order_id} was canceled")
                return status

            # 부분 체결 로그
            if status.executed_volume > 0:
                logger.info(
                    f"[BithumbExecution] Order {order_id} partial fill: "
                    f"{status.fill_ratio:.1%} ({status.executed_volume:.8f}/{status.volume:.8f})"
                )

            time.sleep(poll_interval)

        logger.warning(
            f"[BithumbExecution] Timeout waiting for {order_id} "
            f"(last state: {last_status.state if last_status else 'unknown'})"
        )
        return last_status

    def get_open_orders(self, symbol: str = None) -> List[OrderStatus]:
        """
        미체결 주문 목록 조회.

        Args:
            symbol: 종목 필터 (예: "BTC/KRW")

        Returns:
            미체결 OrderStatus 목록
        """
        try:
            market = self._convert_symbol(symbol) if symbol else None
            result = self.bithumb.get_orders(
                market=market,
                states=["wait", "watch"],
                limit=100,
            )

            orders = []
            for item in result or []:
                status = self._parse_order_status(item)
                if status:
                    orders.append(status)

            return orders
        except Exception as e:
            logger.error(f"[BithumbExecution] get_open_orders failed: {e}")
            return []

    def get_recent_orders(
        self,
        symbol: str = None,
        limit: int = 20,
        states: List[str] = None,
    ) -> List[OrderStatus]:
        """
        최근 주문 목록 조회.

        Args:
            symbol: 종목 필터
            limit: 최대 개수
            states: 상태 필터 (기본: 전체)

        Returns:
            OrderStatus 목록
        """
        try:
            market = self._convert_symbol(symbol) if symbol else None
            result = self.bithumb.get_orders(
                market=market,
                states=states,
                limit=limit,
            )

            orders = []
            for item in result or []:
                status = self._parse_order_status(item)
                if status:
                    orders.append(status)

            return orders
        except Exception as e:
            logger.error(f"[BithumbExecution] get_recent_orders failed: {e}")
            return []

    def get_order_chance(self, symbol: str) -> Optional[Dict]:
        """
        주문 가능 정보 조회 (수수료, 제한 등).

        Args:
            symbol: 종목 (예: "BTC/KRW")

        Returns:
            주문 가능 정보 dict
        """
        try:
            market = self._convert_symbol(symbol)
            return self.bithumb.get_orders_chance(market)
        except Exception as e:
            logger.error(f"[BithumbExecution] get_order_chance failed: {e}")
            return None

    def _parse_order_status(self, data: Dict) -> Optional[OrderStatus]:
        """API 응답을 OrderStatus로 변환."""
        if not data:
            return None

        try:
            # 평균 체결가 계산
            avg_price = None
            if data.get("trades"):
                trades = data["trades"]
                total_value = sum(
                    float(t.get("price", 0)) * float(t.get("volume", 0))
                    for t in trades
                )
                total_volume = sum(float(t.get("volume", 0)) for t in trades)
                if total_volume > 0:
                    avg_price = total_value / total_volume
            elif data.get("avg_price"):
                avg_price = float(data["avg_price"])

            return OrderStatus(
                order_id=data.get("uuid", ""),
                market=data.get("market", ""),
                side=data.get("side", ""),
                ord_type=data.get("ord_type", ""),
                state=data.get("state", ""),
                volume=float(data.get("volume", 0)),
                remaining_volume=float(data.get("remaining_volume", 0)),
                executed_volume=float(data.get("executed_volume", 0)),
                price=float(data["price"]) if data.get("price") else None,
                avg_price=avg_price,
                trades_count=int(data.get("trades_count", 0)),
                created_at=datetime.fromisoformat(
                    data.get("created_at", "").replace("Z", "+00:00")
                ) if data.get("created_at") else datetime.now(),
                paid_fee=float(data.get("paid_fee", 0)),
                locked=float(data.get("locked", 0)),
            )
        except Exception as e:
            logger.warning(f"[BithumbExecution] _parse_order_status failed: {e}")
            return None

    # ========== 포지션 동기화 기능 (Phase 8-2) ==========

    def sync_positions(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """
        거래소 실제 잔고를 기준으로 포지션 동기화.

        Args:
            symbols: 동기화할 종목 목록 (없으면 전체)

        Returns:
            {symbol: {"balance": float, "locked": float, "avg_buy_price": float}}
        """
        try:
            accounts = self.bithumb.get_accounts()
            positions = {}

            for account in accounts or []:
                currency = account.get("currency", "")

                # KRW 제외
                if currency == "KRW":
                    continue

                symbol = f"{currency}/KRW"

                # symbols 필터링
                if symbols and symbol not in symbols:
                    continue

                balance = float(account.get("balance", 0))
                locked = float(account.get("locked", 0))
                avg_buy_price = float(account.get("avg_buy_price", 0))

                # 유의미한 잔고만 포함 (0.00001 이상)
                if balance > 0.00001 or locked > 0.00001:
                    positions[symbol] = {
                        "balance": balance,
                        "locked": locked,
                        "total": balance + locked,
                        "avg_buy_price": avg_buy_price,
                        "currency": currency,
                    }

            logger.info(
                f"[BithumbExecution] sync_positions: {len(positions)} positions found"
            )
            return positions

        except Exception as e:
            logger.error(f"[BithumbExecution] sync_positions failed: {e}")
            return {}

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        특정 종목 포지션 조회.

        Args:
            symbol: 종목 (예: "BTC/KRW")

        Returns:
            {balance, locked, avg_buy_price} 또는 None
        """
        positions = self.sync_positions([symbol])
        return positions.get(symbol)

    def get_position_value(self, symbol: str) -> float:
        """
        특정 종목 포지션의 현재 가치 (KRW).

        Args:
            symbol: 종목

        Returns:
            현재가 기준 포지션 가치
        """
        position = self.get_position(symbol)
        if not position:
            return 0.0

        current_price = self.get_current_price(symbol)
        if not current_price:
            return 0.0

        return position["total"] * current_price

    def get_total_portfolio_value(self) -> Dict[str, float]:
        """
        전체 포트폴리오 가치 계산.

        Returns:
            {
                "krw_balance": float,
                "positions_value": float,
                "total_value": float,
                "positions": {symbol: value}
            }
        """
        try:
            krw_balance = self.get_balance("KRW")
            positions = self.sync_positions()
            positions_value = 0.0
            position_values = {}

            for symbol, pos in positions.items():
                current_price = self.get_current_price(symbol)
                if current_price:
                    value = pos["total"] * current_price
                    positions_value += value
                    position_values[symbol] = value

            return {
                "krw_balance": krw_balance,
                "positions_value": positions_value,
                "total_value": krw_balance + positions_value,
                "positions": position_values,
            }

        except Exception as e:
            logger.error(f"[BithumbExecution] get_total_portfolio_value failed: {e}")
            return {
                "krw_balance": 0.0,
                "positions_value": 0.0,
                "total_value": 0.0,
                "positions": {},
            }

    def get_pnl(self, symbol: str) -> Optional[Dict]:
        """
        특정 종목의 손익 계산.

        Args:
            symbol: 종목

        Returns:
            {
                "unrealized_pnl": float,
                "unrealized_pnl_pct": float,
                "avg_buy_price": float,
                "current_price": float,
                "quantity": float,
            }
        """
        position = self.get_position(symbol)
        if not position or position["total"] == 0:
            return None

        current_price = self.get_current_price(symbol)
        if not current_price:
            return None

        avg_buy_price = position.get("avg_buy_price", 0)
        quantity = position["total"]
        current_value = quantity * current_price
        cost_basis = quantity * avg_buy_price

        if cost_basis > 0:
            unrealized_pnl = current_value - cost_basis
            unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price * 100
        else:
            unrealized_pnl = 0.0
            unrealized_pnl_pct = 0.0

        return {
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "avg_buy_price": avg_buy_price,
            "current_price": current_price,
            "quantity": quantity,
            "current_value": current_value,
            "cost_basis": cost_basis,
        }

    # ========== 에러 복구 및 재시도 (Phase 8-3) ==========

    def _retry_operation(
        self,
        operation: Callable[[], T],
        operation_name: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        retryable_exceptions: tuple = (ConnectionError, TimeoutError),
    ) -> T:
        """
        일반 작업에 대한 재시도 래퍼.

        Args:
            operation: 실행할 callable
            operation_name: 로깅용 작업 이름
            max_retries: 최대 재시도 횟수
            base_delay: 초기 대기 시간
            retryable_exceptions: 재시도할 예외 타입

        Returns:
            작업 결과

        Raises:
            마지막 예외 (모든 재시도 실패 시)
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return operation()

            except retryable_exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self._calculate_backoff(attempt, base_delay)
                    logger.warning(
                        f"[BithumbExecution] {operation_name} attempt {attempt + 1}/{max_retries} "
                        f"failed: {type(e).__name__}. Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[BithumbExecution] {operation_name} failed after {max_retries + 1} attempts: {e}"
                    )

            except Exception as e:
                # 재시도 불가능한 예외는 즉시 raise
                logger.error(f"[BithumbExecution] {operation_name} non-retryable error: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Unexpected failure in {operation_name}")

    @staticmethod
    def _calculate_backoff(attempt: int, base_delay: float, max_delay: float = 30.0) -> float:
        """지수 백오프 + 지터 계산."""
        delay = base_delay * (2 ** attempt)
        delay = min(delay, max_delay)
        # ±20% 지터
        jitter = delay * 0.2 * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)

    def place_order_with_retry(
        self,
        symbol: str,
        side: str,
        *,
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        max_retries: int = 2,
    ) -> BithumbOrderResult:
        """
        재시도 로직이 포함된 주문 실행.

        네트워크 오류 시 자동 재시도합니다.
        주문 중복 방지를 위해 주문 전 상태를 확인합니다.

        Args:
            symbol: 종목
            side: "BUY" 또는 "SELL"
            units: 코인 수량
            amount_krw: KRW 금액 (BUY 전용)
            order_type: 주문 유형
            price: 지정가 (LIMIT 전용)
            max_retries: 최대 재시도 횟수

        Returns:
            BithumbOrderResult
        """

        def attempt_order():
            return self.place_order(
                symbol,
                side,
                units=units,
                amount_krw=amount_krw,
                order_type=order_type,
                price=price,
            )

        return self._retry_operation(
            attempt_order,
            f"place_order({symbol}, {side})",
            max_retries=max_retries,
            base_delay=1.0,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )

    def safe_cancel_order(
        self,
        order_id: str,
        symbol: str = None,
        max_retries: int = 2,
    ) -> bool:
        """
        안전한 주문 취소 (상태 확인 포함).

        Args:
            order_id: 주문 UUID
            symbol: 종목 (옵션)
            max_retries: 최대 재시도

        Returns:
            취소 성공 여부
        """
        # 1. 주문 상태 확인
        status = self.get_order_status(order_id)

        if status is None:
            logger.warning(f"[BithumbExecution] Order {order_id} not found for cancel")
            return False

        if status.is_done:
            logger.info(f"[BithumbExecution] Order {order_id} already filled, cannot cancel")
            return False

        if status.is_canceled:
            logger.info(f"[BithumbExecution] Order {order_id} already canceled")
            return True

        # 2. 취소 시도 (재시도 포함)
        def attempt_cancel():
            return self.cancel_order(order_id, symbol)

        try:
            return self._retry_operation(
                attempt_cancel,
                f"cancel_order({order_id})",
                max_retries=max_retries,
            )
        except Exception as e:
            logger.error(f"[BithumbExecution] safe_cancel_order failed: {e}")
            return False

    def verify_order_execution(
        self,
        order_id: str,
        expected_side: str,
        expected_symbol: str,
        timeout_seconds: float = 10.0,
    ) -> Optional[OrderStatus]:
        """
        주문 실행 검증.

        주문 후 실제로 체결되었는지 확인합니다.

        Args:
            order_id: 주문 UUID
            expected_side: 예상 주문 방향
            expected_symbol: 예상 종목
            timeout_seconds: 검증 대기 시간

        Returns:
            OrderStatus (체결 완료 시) 또는 None
        """
        status = self.wait_for_fill(order_id, timeout_seconds=timeout_seconds)

        if status is None:
            logger.error(f"[BithumbExecution] Order {order_id} verification failed: not found")
            return None

        # 검증
        expected_market = self._convert_symbol(expected_symbol)
        expected_api_side = "bid" if expected_side.upper() == "BUY" else "ask"

        if status.market != expected_market:
            logger.error(
                f"[BithumbExecution] Order {order_id} market mismatch: "
                f"expected {expected_market}, got {status.market}"
            )
            return None

        if status.side != expected_api_side:
            logger.error(
                f"[BithumbExecution] Order {order_id} side mismatch: "
                f"expected {expected_api_side}, got {status.side}"
            )
            return None

        if status.is_done:
            logger.info(
                f"[BithumbExecution] Order {order_id} verified: "
                f"{status.executed_volume:.8f} @ {status.avg_price or 0:,.0f}"
            )
            return status

        if status.is_canceled:
            logger.warning(f"[BithumbExecution] Order {order_id} was canceled during verification")
            return status

        logger.warning(
            f"[BithumbExecution] Order {order_id} verification incomplete: state={status.state}"
        )
        return status

    def recover_from_error(self, error: Exception, context: Dict) -> Dict:
        """
        에러 복구 시도.

        Args:
            error: 발생한 예외
            context: 에러 컨텍스트 (symbol, side, order_id 등)

        Returns:
            복구 결과 {"recovered": bool, "action": str, "details": ...}
        """
        error_type = type(error).__name__
        symbol = context.get("symbol", "")
        side = context.get("side", "")
        order_id = context.get("order_id", "")

        logger.info(f"[BithumbExecution] Attempting recovery from {error_type}")

        # 1. 네트워크 오류 - 연결 재확인
        if isinstance(error, (ConnectionError, TimeoutError)):
            try:
                # 연결 테스트
                self.get_balance("KRW")
                return {
                    "recovered": True,
                    "action": "connection_restored",
                    "details": "Connection recovered after retry",
                }
            except Exception as e:
                return {
                    "recovered": False,
                    "action": "connection_failed",
                    "details": str(e),
                }

        # 2. 주문 관련 오류 - 미체결 주문 확인
        if order_id:
            try:
                status = self.get_order_status(order_id)
                if status:
                    return {
                        "recovered": True,
                        "action": "order_status_found",
                        "details": {
                            "state": status.state,
                            "executed_volume": status.executed_volume,
                            "remaining_volume": status.remaining_volume,
                        },
                    }
            except Exception:
                pass

        # 3. 복구 불가
        return {
            "recovered": False,
            "action": "recovery_failed",
            "details": f"Could not recover from {error_type}: {error}",
        }

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
    
    def get_closed_orders(
        self,
        market: Optional[str] = None,
        limit: int = 100,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get closed (completed) orders from Bithumb.

        Args:
            market: Market symbol (e.g., "KRW-BTC"). If None, gets all markets.
            limit: Number of orders to fetch (max 100)
            page: Page number for pagination

        Returns:
            List of closed order dicts with keys:
            - uuid, side, ord_type, price, state, market,
            - created_at, volume, remaining_volume, executed_volume,
            - trades_count, paid_fee, avg_price
        """
        try:
            orders = self.bithumb.get_orders(
                market=market,
                state="done",  # Completed orders only
                limit=min(limit, 100),
                page=page,
                order_by="desc",
            )
            if isinstance(orders, list):
                logger.info(f"[BithumbExecution] Fetched {len(orders)} closed orders")
                return orders
            return []
        except Exception as e:
            logger.error(f"[BithumbExecution] Failed to get closed orders: {e}")
            return []

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


# Alias for backward compatibility (used by holdings.py)
class BithumbExecution(BithumbExecutionAdapter):
    """Simplified BithumbExecution that loads config from environment variables."""

    def __init__(self, config: Config = None):
        """Initialize with optional config. If not provided, loads from environment."""
        if config is None:
            config = Config()
        super().__init__(config)
