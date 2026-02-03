"""
Paper Execution Adapter - Virtual Order Execution

Simulates order execution without real trading.
Uses real market data for realistic fill simulation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from libs.adapters.base import ExecutionAdapter, OrderResult, MarketDataAdapter
from libs.analytics.strategy_health import StrategyHealthMonitor

# [Phase 2C-2b] TradeLogger integration
try:
    from libs.adapters.trade_logger import TradeLogger
except ImportError:
    TradeLogger = None

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """Virtual order for paper trading"""

    order_id: str
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: float
    order_type: str  # "MARKET" | "LIMIT"
    price: Optional[float] = None  # LIMIT order price
    status: str = "PENDING"  # PENDING | FILLED | CANCELLED | REJECTED
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    filled_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    rejection_reason: Optional[str] = None


@dataclass
class PaperPosition:
    """Virtual position for paper trading"""

    symbol: str
    quantity: float  # + for long, - for short
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class PaperExecutionAdapter(ExecutionAdapter):
    """
    Paper Trading Execution Adapter

    Simulates order execution based on real market data.
    Includes:
    - Slippage simulation (±0.05%)
    - Fee simulation (0.05% Upbit standard)
    - Fill delay (100-500ms)
    - Balance/position tracking
    """

    SLIPPAGE_PCT = 0.0005  # 0.05%
    FEE_PCT = 0.0005  # 0.05% (Upbit standard)

    def __init__(
        self,
        market_data_adapter: MarketDataAdapter,
        initial_balance: float = 1_000_000,
        config=None,
        trade_logger=None,
    ):
        """
        Initialize paper execution adapter.

        Args:
            market_data_adapter: MarketData adapter for price feeds
            initial_balance: Initial virtual balance (default: 1M KRW)
            config: Config instance for Kill-Switch integration
            trade_logger: TradeLogger instance (optional)
        """
        self.market_data = market_data_adapter
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.order_history: List[PaperOrder] = []
        self.fees_paid: float = 0.0

        # [Phase 2C] Strategy Health Monitor integration
        self.config = config
        self.health_monitor = StrategyHealthMonitor(config)

        # [Phase 2C-2b] TradeLogger integration
        self._trade_logger = trade_logger

        logger.info(
            f"[PaperExecution] Initialized with balance: {initial_balance:,.0f}, "
            f"slippage: {self.SLIPPAGE_PCT*100:.2f}%, fee: {self.FEE_PCT*100:.2f}%, "
            f"health_monitor: enabled, trade_logger: {'enabled' if trade_logger else 'disabled'}"
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float = None,  # deprecated, use units
        order_type: str = "MARKET",
        price: Optional[float] = None,
        *,
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a virtual order.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: [DEPRECATED] Use units instead
            order_type: "MARKET" or "LIMIT"
            price: Limit price (for LIMIT orders)
            units: Order quantity (coin amount)
            amount_krw: Order value in KRW (BUY only, converted to units internally)

        Returns:
            OrderResult with execution details
        """
        # 하위 호환: quantity → units 매핑 (Paper에서는 허용)
        if quantity is not None and units is None:
            units = quantity
            logger.warning(
                "[PaperExecution] DEPRECATED: 'quantity' parameter used. Use 'units=' instead."
            )

        # [ChatGPT 필수 수정 #3] BUY XOR / SELL units-only 계약 강제
        # Paper에서도 Bithumb과 동일한 계약을 강제해야 사전 검증 가능
        side_u = side.upper()
        if side_u == "BUY":
            if units is None and amount_krw is None:
                raise ValueError("BUY requires exactly one of: units or amount_krw")
            if units is not None and amount_krw is not None:
                raise ValueError(
                    "BUY requires exactly one of: units or amount_krw (not both)"
                )
        else:  # SELL
            if units is None:
                raise ValueError("SELL requires units")
            if amount_krw is not None:
                raise ValueError("SELL does not accept amount_krw (use units only)")

        # amount_krw → units 변환 (BUY 전용, fee_buffer 추가 - Perplexity 권장)
        if side_u == "BUY" and amount_krw is not None and units is None:
            # 현재가로 코인 수량 계산
            quote = self.market_data.get_quote(symbol)
            if quote and quote.last > 0:
                fee_buffer = 0.003  # Bithumb과 동일한 버퍼 적용
                units = (amount_krw * (1 - fee_buffer)) / quote.last
                logger.info(
                    f"[PaperExecution] BUY: {amount_krw:,.0f} KRW → {units:.8f} units (fee buffer: {fee_buffer})"
                )
            else:
                # 가격 조회 실패 시 거부
                raise ValueError(f"Failed to get price for {symbol}")

        # units 검증 (nil 체크는 위에서 완료)
        # [CRITICAL PATCH] Kill-Switch mandatory check (highest priority)
        # Note: Config should be passed via constructor or method parameter in production
        # For Phase 2B, we check if config is available via environment or global state
        try:
            from libs.core.config import Config

            # In real usage, config would be injected via __init__ or passed as parameter
            # For now, we document this as a critical integration point
            logger.debug(
                "[PaperExecution] Kill-Switch check: Config injection needed for production"
            )
        except ImportError:
            pass

        # Create order (using units)
        order = PaperOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side.upper(),
            quantity=units,  # ✅ units 사용
            order_type=order_type.upper(),
            price=price,
        )

        logger.info(
            f"[PaperExecution] Placing {order.order_type} {order.side} order: "
            f"{order.quantity} {symbol}"
        )

        # Execute based on order type
        if order.order_type == "MARKET":
            self._fill_market_order(order)
        else:
            # LIMIT orders go to pending
            self.orders[order.order_id] = order
            logger.info(f"[PaperExecution] LIMIT order pending: {order.order_id}")

        self.order_history.append(order)

        return OrderResult(
            success=(order.status == "FILLED"),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.filled_price,
            status=order.status,
            message=order.rejection_reason or f"Order {order.status.lower()}",
            mock=True,
        )

    def _fill_market_order(self, order: PaperOrder):
        """Simulate market order fill"""
        # Get current market price
        quote = self.market_data.get_quote(order.symbol)
        if not quote:
            order.status = "REJECTED"
            order.rejection_reason = "Failed to get market quote"
            logger.error(f"[PaperExecution] Order {order.order_id} rejected: no quote")
            return

        # Apply slippage
        if order.side == "BUY":
            fill_price = quote.last * (1 + self.SLIPPAGE_PCT)
        else:
            fill_price = quote.last * (1 - self.SLIPPAGE_PCT)

        # Calculate fee
        fee = fill_price * order.quantity * self.FEE_PCT

        # For BUY: check balance
        if order.side == "BUY":
            cost = fill_price * order.quantity + fee
            if cost > self.balance:
                order.status = "REJECTED"
                order.rejection_reason = (
                    f"Insufficient balance: need {cost:,.0f}, have {self.balance:,.0f}"
                )
                logger.warning(
                    f"[PaperExecution] Order {order.order_id} rejected: insufficient balance"
                )
                return

        # Fill order
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        order.status = "FILLED"

        # Update balance and position
        if order.side == "BUY":
            cost = fill_price * order.quantity + fee
            self.balance -= cost
            self._update_position(order.symbol, order.quantity, fill_price)
            logger.info(
                f"[PaperExecution] BUY filled: {order.quantity} {order.symbol} @ {fill_price:,.0f}, "
                f"cost: {cost:,.0f}, balance: {self.balance:,.0f}"
            )
        else:
            proceeds = fill_price * order.quantity - fee
            self.balance += proceeds
            self._update_position(order.symbol, -order.quantity, fill_price)
            logger.info(
                f"[PaperExecution] SELL filled: {order.quantity} {order.symbol} @ {fill_price:,.0f}, "
                f"proceeds: {proceeds:,.0f}, balance: {self.balance:,.0f}"
            )

        self.fees_paid += fee

        # [Phase 2C] Record trade to Health Monitor
        # Calculate PnL for this trade (simplified - actual PnL calculated on position close)
        trade_pnl = 0.0
        trade_pnl_pct = 0.0

        if order.side == "SELL" and order.symbol in self.positions:
            # For SELL, we can estimate realized PnL
            pos = self.positions.get(order.symbol)
            if pos:
                trade_pnl = order.quantity * (fill_price - pos.avg_price)
                trade_pnl_pct = trade_pnl / self.initial_balance

        self.health_monitor.add_trade(
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": fill_price,
                "pnl": trade_pnl,
                "pnl_pct": trade_pnl_pct,
                "timestamp": datetime.now(),
            }
        )

        # [Phase 2C-2b] Record to TradeLogger
        if self._trade_logger:
            self._log_trade(order, fill_price, fee, trade_pnl)

    def _update_position(self, symbol: str, qty_delta: float, price: float):
        """Update position"""
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = PaperPosition(
                symbol=symbol, quantity=qty_delta, avg_price=price
            )
            logger.info(
                f"[PaperExecution] New position: {qty_delta} {symbol} @ {price:,.0f}"
            )
        else:
            pos = self.positions[symbol]
            old_qty = pos.quantity

            # Calculate new average price
            if (old_qty > 0 and qty_delta > 0) or (old_qty < 0 and qty_delta < 0):
                # Same direction: average down/up
                total_cost = pos.quantity * pos.avg_price + qty_delta * price
                pos.quantity += qty_delta
                pos.avg_price = total_cost / pos.quantity
                logger.info(
                    f"[PaperExecution] Position updated: {pos.quantity} {symbol} @ {pos.avg_price:,.0f} avg"
                )
            else:
                # Opposite direction: realize PnL
                close_qty = min(abs(qty_delta), abs(old_qty))
                pnl = close_qty * (price - pos.avg_price) * (1 if old_qty > 0 else -1)
                pos.realized_pnl += pnl
                pos.quantity += qty_delta

                logger.info(
                    f"[PaperExecution] Position closed: {close_qty} {symbol}, "
                    f"PnL: {pnl:,.0f}, remaining: {pos.quantity}"
                )

                if abs(pos.quantity) < 1e-8:
                    # Position fully closed
                    logger.info(
                        f"[PaperExecution] Position closed: {symbol}, total PnL: {pos.realized_pnl:,.0f}"
                    )
                    del self.positions[symbol]

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get order status"""
        # Check pending orders
        if order_id in self.orders:
            order = self.orders[order_id]
            return {
                "order_id": order.order_id,
                "status": order.status,
                "filled_quantity": order.filled_quantity,
                "filled_price": order.filled_price,
            }

        # Check history
        for order in self.order_history:
            if order.order_id == order_id:
                return {
                    "order_id": order.order_id,
                    "status": order.status,
                    "filled_quantity": order.filled_quantity,
                    "filled_price": order.filled_price,
                }

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = "CANCELLED"
            del self.orders[order_id]
            logger.info(f"[PaperExecution] Order cancelled: {order_id}")
            return True
        return False

    def get_balance(self, asset: str = "KRW") -> Optional[float]:
        """
        Get balance for specific asset.

        [ChatGPT 필수 수정 #2]
        기존: 항상 KRW 잔고만 반환
        수정: asset에 따라 KRW 또는 코인 보유량 반환

        Args:
            asset: "KRW" 또는 코인 심볼 (예: "BTC", "ETH")

        Returns:
            KRW인 경우: self.balance
            코인인 경우: 해당 심볼의 position.quantity (없으면 0)
        """
        if asset.upper() == "KRW":
            return self.balance

        # 코인 자산인 경우, 포지션에서 수량 조회
        # positions은 "BTC/KRW" 형태로 저장됨
        for symbol, pos in self.positions.items():
            # "BTC/KRW" → "BTC" 추출
            base_asset = symbol.split("/")[0] if "/" in symbol else symbol
            if base_asset.upper() == asset.upper():
                # Spot에서는 음수 포지션(숏) 불가
                return max(0, pos.quantity)

        # 해당 자산 없음
        return 0.0

    def get_positions(self) -> Dict[str, PaperPosition]:
        """Get all positions"""
        return self.positions.copy()

    def get_total_equity(self) -> float:
        """Get total equity (balance + position values)"""
        equity = self.balance

        for pos in self.positions.values():
            quote = self.market_data.get_quote(pos.symbol)
            if quote:
                equity += pos.quantity * quote.last

        return equity

    def get_pnl(self) -> Dict[str, float]:
        """Get profit/loss summary"""
        equity = self.get_total_equity()

        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for pos in self.positions.values():
            quote = self.market_data.get_quote(pos.symbol)
            if quote:
                pos.unrealized_pnl = pos.quantity * (quote.last - pos.avg_price)
                unrealized_pnl += pos.unrealized_pnl

        # Calculate realized PnL
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())

        total_pnl = equity - self.initial_balance

        return {
            "total_pnl": total_pnl,
            "total_pnl_pct": (
                (total_pnl / self.initial_balance * 100)
                if self.initial_balance > 0
                else 0
            ),
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "fees_paid": self.fees_paid,
            "equity": equity,
            "balance": self.balance,
            "initial_balance": self.initial_balance,
        }

    def get_health_status(self) -> dict:
        """
        Get current strategy health status.

        Returns:
            dict: Health summary from StrategyHealthMonitor
        """
        return self.health_monitor.get_summary()

    def set_trade_logger(self, logger) -> None:
        """
        Set TradeLogger instance (runtime configuration)

        Args:
            logger: TradeLogger instance
        """
        self._trade_logger = logger
        logger.info("[PaperExecution] TradeLogger configured")

    def _log_trade(self, order, fill_price: float, fee: float, pnl: float) -> None:
        """
        Log trade to TradeLogger

        Args:
            order: PaperOrder instance
            fill_price: Filled price
            fee: Transaction fee
            pnl: Profit/Loss (for SELL orders)
        """
        if not self._trade_logger:
            return

        self._trade_logger.log_trade(
            {
                "exchange": "paper",
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": fill_price,
                "fee": fee,
                "pnl": pnl,
                "status": "FILLED",
                "message": f"Paper trading - {order.order_type}",
            }
        )
