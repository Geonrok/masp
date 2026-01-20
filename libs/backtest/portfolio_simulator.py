"""
Portfolio Simulator

Full portfolio simulation with realistic transaction costs,
slippage modeling, and position tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Callable

import numpy as np
import pandas as pd

from libs.backtest.historical_data import OHLCVDataset

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by simulator."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class SimulatedOrder:
    """Simulated order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Current position in a symbol."""

    symbol: str
    quantity: float
    avg_cost: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Market value based on avg_cost (update with current price)."""
        return self.quantity * self.avg_cost

    def update_unrealized(self, current_price: float) -> None:
        """Update unrealized PnL based on current price."""
        if self.quantity > 0:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity
        elif self.quantity < 0:
            self.unrealized_pnl = (self.avg_cost - current_price) * abs(self.quantity)


@dataclass
class Trade:
    """Executed trade record."""

    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""

    timestamp: datetime
    cash: float
    positions: Dict[str, Position]
    equity: float
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0


class PortfolioSimulator:
    """
    Full portfolio simulator with realistic execution modeling.

    Features:
    - Multi-asset position tracking
    - Transaction costs (commission + slippage)
    - Partial fills simulation
    - Margin/leverage support (optional)
    - Equity curve tracking
    - Drawdown calculation
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        commission_rate: float = 0.0015,  # 0.15% per trade
        slippage_model: str = "fixed",  # "fixed", "proportional", "volatility"
        slippage_bps: float = 5.0,  # 5 basis points
        enable_shorting: bool = False,
        max_leverage: float = 1.0,
        margin_requirement: float = 1.0,
    ):
        """
        Initialize portfolio simulator.

        Args:
            initial_capital: Starting capital
            commission_rate: Commission as fraction of trade value
            slippage_model: How to model slippage
            slippage_bps: Slippage in basis points
            enable_shorting: Allow short positions
            max_leverage: Maximum leverage allowed
            margin_requirement: Margin requirement for positions
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.enable_shorting = enable_shorting
        self.max_leverage = max_leverage
        self.margin_requirement = margin_requirement

        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.orders: List[SimulatedOrder] = []
        self.equity_curve: List[PortfolioState] = []

        # Tracking
        self.peak_equity = initial_capital
        self._order_counter = 0
        self._trade_counter = 0

        logger.info(
            f"[PortfolioSimulator] Initialized: capital={initial_capital:,.0f}, "
            f"commission={commission_rate*100:.2f}%, slippage={slippage_bps}bps"
        )

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.orders.clear()
        self.equity_curve.clear()
        self.peak_equity = self.initial_capital
        self._order_counter = 0
        self._trade_counter = 0

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"TRD-{self._trade_counter:06d}"

    def _calculate_slippage(
        self,
        price: float,
        side: OrderSide,
        quantity: float,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate slippage based on model.

        Args:
            price: Base price
            side: Order side
            quantity: Order quantity
            volatility: Current volatility (for volatility model)

        Returns:
            Slippage amount (always positive, added to buys, subtracted from sells)
        """
        if self.slippage_model == "fixed":
            # Fixed basis points
            slippage = price * (self.slippage_bps / 10000)

        elif self.slippage_model == "proportional":
            # Proportional to order size
            base_slippage = price * (self.slippage_bps / 10000)
            size_factor = 1 + (quantity / 10000)  # Larger orders = more slippage
            slippage = base_slippage * size_factor

        elif self.slippage_model == "volatility":
            # Based on volatility
            vol = volatility or 0.02
            slippage = price * vol * (self.slippage_bps / 100)

        else:
            slippage = 0.0

        return abs(slippage)

    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for trade."""
        return abs(trade_value) * self.commission_rate

    def _get_fill_price(
        self,
        order: SimulatedOrder,
        current_price: float,
        high: float,
        low: float,
    ) -> Optional[float]:
        """
        Determine fill price for order.

        Args:
            order: Order to fill
            current_price: Current market price
            high: Period high
            low: Period low

        Returns:
            Fill price or None if not filled
        """
        if order.order_type == OrderType.MARKET:
            # Fill at current price with slippage
            slippage = self._calculate_slippage(
                current_price, order.side, order.quantity
            )
            if order.side == OrderSide.BUY:
                return current_price + slippage
            else:
                return current_price - slippage

        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return None

            if order.side == OrderSide.BUY:
                # Buy limit fills if low <= limit price
                if low <= order.limit_price:
                    return min(order.limit_price, current_price)
            else:
                # Sell limit fills if high >= limit price
                if high >= order.limit_price:
                    return max(order.limit_price, current_price)

        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                return None

            if order.side == OrderSide.BUY:
                # Buy stop triggers if high >= stop price
                if high >= order.stop_price:
                    slippage = self._calculate_slippage(
                        order.stop_price, order.side, order.quantity
                    )
                    return order.stop_price + slippage
            else:
                # Sell stop triggers if low <= stop price
                if low <= order.stop_price:
                    slippage = self._calculate_slippage(
                        order.stop_price, order.side, order.quantity
                    )
                    return order.stop_price - slippage

        return None

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> SimulatedOrder:
        """
        Submit a new order.

        Args:
            symbol: Symbol to trade
            side: Buy or sell
            quantity: Number of units
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            timestamp: Order timestamp

        Returns:
            SimulatedOrder
        """
        order = SimulatedOrder(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            timestamp=timestamp,
        )

        self.orders.append(order)
        logger.debug(
            f"[PortfolioSimulator] Order submitted: {order.order_id} "
            f"{side.value} {quantity} {symbol}"
        )

        return order

    def process_bar(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> List[Trade]:
        """
        Process a bar/candle for order execution and position updates.

        Args:
            symbol: Symbol
            timestamp: Bar timestamp
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume

        Returns:
            List of executed trades
        """
        executed_trades = []

        # Process pending orders for this symbol
        pending_orders = [
            o for o in self.orders
            if o.symbol == symbol and not o.filled
        ]

        for order in pending_orders:
            fill_price = self._get_fill_price(order, close, high, low)

            if fill_price is not None:
                # Execute the order
                trade = self._execute_order(order, fill_price, timestamp)
                if trade:
                    executed_trades.append(trade)

        # Update position unrealized PnL
        if symbol in self.positions:
            self.positions[symbol].update_unrealized(close)

        return executed_trades

    def _execute_order(
        self,
        order: SimulatedOrder,
        fill_price: float,
        timestamp: datetime,
    ) -> Optional[Trade]:
        """
        Execute an order and update positions.

        Args:
            order: Order to execute
            fill_price: Execution price
            timestamp: Execution timestamp

        Returns:
            Trade record or None
        """
        trade_value = order.quantity * fill_price
        commission = self._calculate_commission(trade_value)
        slippage = abs(fill_price - (order.limit_price or fill_price))

        # Check if we have enough cash for buy
        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission
            if total_cost > self.cash:
                logger.warning(
                    f"[PortfolioSimulator] Insufficient cash for {order.order_id}: "
                    f"need {total_cost:,.0f}, have {self.cash:,.0f}"
                )
                return None

            # Deduct cash
            self.cash -= total_cost

            # Update position
            if order.symbol not in self.positions:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_cost=fill_price,
                )
            else:
                pos = self.positions[order.symbol]
                # Calculate new average cost
                total_qty = pos.quantity + order.quantity
                if total_qty > 0:
                    pos.avg_cost = (
                        (pos.quantity * pos.avg_cost) + (order.quantity * fill_price)
                    ) / total_qty
                pos.quantity = total_qty

        else:  # SELL
            if order.symbol not in self.positions:
                if not self.enable_shorting:
                    logger.warning(
                        f"[PortfolioSimulator] Cannot sell {order.symbol}: no position"
                    )
                    return None
                # Create short position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=-order.quantity,
                    avg_cost=fill_price,
                )
            else:
                pos = self.positions[order.symbol]

                if pos.quantity < order.quantity and not self.enable_shorting:
                    logger.warning(
                        f"[PortfolioSimulator] Cannot sell {order.quantity} {order.symbol}: "
                        f"only have {pos.quantity}"
                    )
                    return None

                # Calculate realized PnL
                sell_qty = min(order.quantity, pos.quantity) if pos.quantity > 0 else order.quantity
                if pos.quantity > 0:
                    realized_pnl = (fill_price - pos.avg_cost) * sell_qty
                    pos.realized_pnl += realized_pnl
                else:
                    realized_pnl = 0.0

                pos.quantity -= order.quantity

                # Remove position if zero
                if abs(pos.quantity) < 1e-9:
                    del self.positions[order.symbol]

            # Add cash from sale
            self.cash += trade_value - commission

        # Mark order as filled
        order.filled = True
        order.fill_price = fill_price
        order.fill_quantity = order.quantity
        order.commission = commission
        order.slippage = slippage

        # Create trade record
        trade = Trade(
            trade_id=self._generate_trade_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            timestamp=timestamp,
            pnl=0.0,  # Updated on close
        )

        self.trades.append(trade)

        logger.info(
            f"[PortfolioSimulator] Trade executed: {trade.trade_id} "
            f"{trade.side} {trade.quantity} {trade.symbol} @ {trade.price:,.2f}"
        )

        return trade

    def get_equity(self, prices: Dict[str, float]) -> float:
        """
        Calculate current equity.

        Args:
            prices: Current prices for each symbol

        Returns:
            Total equity (cash + positions value)
        """
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                position_value += pos.quantity * prices[symbol]
            else:
                position_value += pos.quantity * pos.avg_cost

        return self.cash + position_value

    def get_portfolio_state(
        self,
        timestamp: datetime,
        prices: Dict[str, float],
    ) -> PortfolioState:
        """
        Get current portfolio state snapshot.

        Args:
            timestamp: Current timestamp
            prices: Current prices

        Returns:
            PortfolioState
        """
        equity = self.get_equity(prices)
        total_pnl = equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = self.peak_equity - equity
        drawdown_pct = (drawdown / self.peak_equity) * 100 if self.peak_equity > 0 else 0

        # Calculate daily PnL
        if self.equity_curve:
            daily_pnl = equity - self.equity_curve[-1].equity
        else:
            daily_pnl = 0.0

        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            equity=equity,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
        )

        self.equity_curve.append(state)
        return state

    def run_backtest(
        self,
        dataset: OHLCVDataset,
        strategy_fn: Callable[[Dict[str, Any]], Optional[tuple]],
        position_size: float = 0.1,
    ) -> pd.DataFrame:
        """
        Run backtest on historical data with a strategy function.

        Args:
            dataset: Historical OHLCV data
            strategy_fn: Function that takes bar data and returns (side, price) or None
            position_size: Fraction of equity to risk per trade

        Returns:
            DataFrame with backtest results
        """
        self.reset()
        results = []

        for i, row in dataset.data.iterrows():
            timestamp = pd.to_datetime(row["timestamp"])
            symbol = dataset.symbol

            bar_data = {
                "index": i,
                "timestamp": timestamp,
                "symbol": symbol,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "closes": dataset.closes[:i+1] if i > 0 else [row["close"]],
            }

            # Get strategy signal
            signal = strategy_fn(bar_data)

            if signal is not None:
                side, price = signal

                # Calculate position size
                equity = self.get_equity({symbol: row["close"]})
                trade_value = equity * position_size
                quantity = trade_value / price

                if side == "BUY":
                    self.submit_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        timestamp=timestamp,
                    )
                elif side == "SELL":
                    self.submit_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=quantity,
                        timestamp=timestamp,
                    )

            # Process bar
            self.process_bar(
                symbol=symbol,
                timestamp=timestamp,
                open_price=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Record state
            state = self.get_portfolio_state(timestamp, {symbol: row["close"]})
            results.append({
                "timestamp": timestamp,
                "equity": state.equity,
                "cash": state.cash,
                "total_pnl": state.total_pnl,
                "total_pnl_pct": state.total_pnl_pct,
                "drawdown_pct": state.drawdown_pct,
            })

        return pd.DataFrame(results)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get backtest summary statistics.

        Returns:
            Dictionary with summary metrics
        """
        if not self.equity_curve:
            return {"error": "No data"}

        equities = [s.equity for s in self.equity_curve]
        final_equity = equities[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Calculate returns
        returns = np.diff(equities) / equities[:-1] if len(equities) > 1 else []

        # Max drawdown
        max_dd = max(s.drawdown_pct for s in self.equity_curve)

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            "total_commission": sum(t.commission for t in self.trades),
        }
