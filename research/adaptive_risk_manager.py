"""
Adaptive Risk Management System
Strategy Quality: B+ → A+ Upgrade

Features:
1. Dynamic Position Sizing (Kelly Criterion + Risk Parity)
2. Regime-Based Risk Adjustment
3. Drawdown Control with Recovery Mode
4. Correlation-Based Portfolio Risk Limit
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""

    BULL_LOW_VOL = "bull_low_vol"  # Best for momentum
    BULL_HIGH_VOL = "bull_high_vol"  # Good but reduce size
    BEAR_LOW_VOL = "bear_low_vol"  # Cautious
    BEAR_HIGH_VOL = "bear_high_vol"  # Maximum caution / cash
    SIDEWAYS = "sideways"  # Reduce trading


@dataclass
class RiskParameters:
    """Risk parameters for position sizing."""

    max_portfolio_risk: float = 0.02  # Max 2% daily portfolio VaR
    max_position_size: float = 0.10  # Max 10% per position
    min_position_size: float = 0.01  # Min 1% per position
    max_correlation: float = 0.7  # Max avg correlation
    max_drawdown_limit: float = 0.20  # -20% triggers recovery mode
    kelly_fraction: float = 0.25  # Use 1/4 Kelly for safety
    volatility_target: float = 0.15  # 15% annual vol target


class AdaptiveRiskManager:
    """
    Dynamic risk management that adapts to market conditions.

    Key improvements over static risk:
    1. Reduces exposure in high-volatility regimes
    2. Implements drawdown control
    3. Uses correlation to limit concentration risk
    4. Applies fractional Kelly for position sizing
    """

    def __init__(self, params: RiskParameters = None):
        self.params = params or RiskParameters()
        self.current_drawdown = 0.0
        self.peak_equity = 1.0
        self.recovery_mode = False

    def detect_regime(self, btc_returns: pd.Series, lookback: int = 30) -> MarketRegime:
        """
        Detect current market regime based on BTC.

        Args:
            btc_returns: BTC daily returns
            lookback: Days to look back

        Returns:
            Current market regime
        """
        recent = btc_returns.tail(lookback)

        # Trend: positive cumulative return = bull
        cumulative_return = (1 + recent).prod() - 1
        is_bull = cumulative_return > 0

        # Volatility: compare to historical
        current_vol = recent.std() * np.sqrt(252)
        historical_vol = btc_returns.std() * np.sqrt(252)
        is_high_vol = current_vol > historical_vol * 1.2

        # Trend strength
        trend_strength = abs(cumulative_return) / (recent.std() * np.sqrt(lookback))

        if trend_strength < 0.5:
            return MarketRegime.SIDEWAYS

        if is_bull:
            return (
                MarketRegime.BULL_HIGH_VOL if is_high_vol else MarketRegime.BULL_LOW_VOL
            )
        else:
            return (
                MarketRegime.BEAR_HIGH_VOL if is_high_vol else MarketRegime.BEAR_LOW_VOL
            )

    def get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier based on regime."""
        multipliers = {
            MarketRegime.BULL_LOW_VOL: 1.0,  # Full size
            MarketRegime.BULL_HIGH_VOL: 0.7,  # Reduce 30%
            MarketRegime.SIDEWAYS: 0.5,  # Half size
            MarketRegime.BEAR_LOW_VOL: 0.3,  # Cautious
            MarketRegime.BEAR_HIGH_VOL: 0.1,  # Minimal
        }
        return multipliers.get(regime, 0.5)

    def calculate_kelly_size(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion position size.

        Kelly = W - (1-W)/R
        where W = win rate, R = win/loss ratio

        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)

        Returns:
            Optimal position size (0-1)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Apply fractional Kelly for safety
        kelly = max(0, kelly * self.params.kelly_fraction)

        return min(kelly, self.params.max_position_size)

    def calculate_volatility_adjusted_size(
        self, symbol_volatility: float, base_size: float
    ) -> float:
        """
        Adjust position size to target constant volatility.

        Args:
            symbol_volatility: Annualized volatility of the symbol
            base_size: Base position size

        Returns:
            Volatility-adjusted position size
        """
        if symbol_volatility <= 0:
            return 0.0

        vol_adjustment = self.params.volatility_target / symbol_volatility
        adjusted_size = base_size * vol_adjustment

        return np.clip(
            adjusted_size, self.params.min_position_size, self.params.max_position_size
        )

    def check_correlation_limit(
        self,
        current_positions: pd.DataFrame,
        new_symbol: str,
        returns_data: pd.DataFrame,
    ) -> bool:
        """
        Check if adding new position exceeds correlation limit.

        Args:
            current_positions: Current portfolio positions
            new_symbol: Symbol to potentially add
            returns_data: Historical returns for correlation calculation

        Returns:
            True if can add position, False if too correlated
        """
        if current_positions.empty:
            return True

        # Calculate correlations
        correlations = []
        for symbol in current_positions.index:
            if symbol in returns_data.columns and new_symbol in returns_data.columns:
                corr = returns_data[symbol].corr(returns_data[new_symbol])
                correlations.append(abs(corr))

        if not correlations:
            return True

        avg_correlation = np.mean(correlations)
        return avg_correlation < self.params.max_correlation

    def update_drawdown(self, current_equity: float) -> None:
        """Update drawdown tracking."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity

        # Enter/exit recovery mode
        if self.current_drawdown >= self.params.max_drawdown_limit:
            self.recovery_mode = True
        elif self.current_drawdown < self.params.max_drawdown_limit * 0.5:
            self.recovery_mode = False

    def get_recovery_multiplier(self) -> float:
        """Get position multiplier during recovery mode."""
        if not self.recovery_mode:
            return 1.0

        # Gradually reduce size as drawdown increases
        # At max_drawdown_limit: 50% size
        # At 2x max_drawdown_limit: 10% size
        severity = self.current_drawdown / self.params.max_drawdown_limit
        return max(0.1, 1.0 - 0.5 * severity)

    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],
        returns_data: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate portfolio Value at Risk.

        Args:
            positions: Dict of symbol -> weight
            returns_data: Historical returns
            confidence: VaR confidence level

        Returns:
            Daily VaR as positive number
        """
        # Get returns for positions
        position_symbols = [s for s in positions.keys() if s in returns_data.columns]
        if not position_symbols:
            return 0.0

        weights = np.array([positions[s] for s in position_symbols])
        weights = weights / weights.sum()  # Normalize

        returns_matrix = returns_data[position_symbols].dropna()
        if returns_matrix.empty:
            return 0.0

        portfolio_returns = returns_matrix @ weights

        # Historical VaR
        var = -np.percentile(portfolio_returns, (1 - confidence) * 100)

        return var

    def get_optimal_position_size(
        self,
        symbol: str,
        btc_returns: pd.Series,
        symbol_returns: pd.Series,
        current_positions: Dict[str, float],
        returns_data: pd.DataFrame,
        trade_stats: Optional[Dict] = None,
    ) -> float:
        """
        Calculate optimal position size considering all factors.

        Args:
            symbol: Symbol to size
            btc_returns: BTC returns for regime detection
            symbol_returns: Symbol returns for volatility
            current_positions: Current portfolio
            returns_data: Full returns data for correlation
            trade_stats: Optional trade statistics for Kelly

        Returns:
            Optimal position size (0 to max_position_size)
        """
        # 1. Base size (equal weight among max positions)
        max_positions = 20
        base_size = 1.0 / max_positions

        # 2. Apply Kelly if trade stats available
        if trade_stats:
            kelly_size = self.calculate_kelly_size(
                trade_stats.get("win_rate", 0.5),
                trade_stats.get("avg_win", 0.02),
                trade_stats.get("avg_loss", 0.02),
            )
            base_size = min(base_size, kelly_size) if kelly_size > 0 else base_size

        # 3. Volatility adjustment
        symbol_vol = symbol_returns.std() * np.sqrt(252)
        vol_adjusted = self.calculate_volatility_adjusted_size(symbol_vol, base_size)

        # 4. Regime adjustment
        regime = self.detect_regime(btc_returns)
        regime_multiplier = self.get_regime_multiplier(regime)

        # 5. Recovery mode adjustment
        recovery_multiplier = self.get_recovery_multiplier()

        # 6. Correlation check (binary)
        positions_df = pd.DataFrame({"weight": current_positions})
        can_add = self.check_correlation_limit(positions_df, symbol, returns_data)
        correlation_multiplier = 1.0 if can_add else 0.0

        # Final size
        final_size = (
            vol_adjusted
            * regime_multiplier
            * recovery_multiplier
            * correlation_multiplier
        )

        return np.clip(final_size, 0, self.params.max_position_size)

    def get_risk_report(self) -> Dict:
        """Generate current risk status report."""
        return {
            "current_drawdown": f"{self.current_drawdown:.2%}",
            "peak_equity": self.peak_equity,
            "recovery_mode": self.recovery_mode,
            "recovery_multiplier": self.get_recovery_multiplier(),
            "max_drawdown_limit": f"{self.params.max_drawdown_limit:.2%}",
        }


class DynamicStopLoss:
    """
    Adaptive stop-loss that adjusts to volatility.

    Features:
    - ATR-based trailing stops
    - Time-decay stops
    - Profit-taking levels
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        profit_target_atr: float = 3.0,
        max_holding_days: int = 30,
    ):
        self.atr_multiplier = atr_multiplier
        self.profit_target_atr = profit_target_atr
        self.max_holding_days = max_holding_days

    def calculate_stop_price(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        position_side: str,  # 'long' or 'short'
        days_held: int,
    ) -> Tuple[float, str]:
        """
        Calculate adaptive stop price.

        Returns:
            (stop_price, reason)
        """
        # Base stop: entry - 2*ATR for long
        if position_side == "long":
            base_stop = entry_price - self.atr_multiplier * atr

            # Trail stop up if in profit
            if current_price > entry_price:
                profit_atr = (current_price - entry_price) / atr
                if profit_atr > 1:
                    # Trail at 1 ATR below current
                    trailing_stop = current_price - atr
                    base_stop = max(base_stop, trailing_stop)

            # Time-decay: tighten stop as time passes
            time_decay = min(days_held / self.max_holding_days, 0.5)
            tightened_stop = entry_price - (1 - time_decay) * self.atr_multiplier * atr
            base_stop = max(base_stop, tightened_stop)

            return base_stop, (
                "trailing"
                if base_stop > entry_price - self.atr_multiplier * atr
                else "initial"
            )

        else:  # short
            base_stop = entry_price + self.atr_multiplier * atr

            if current_price < entry_price:
                profit_atr = (entry_price - current_price) / atr
                if profit_atr > 1:
                    trailing_stop = current_price + atr
                    base_stop = min(base_stop, trailing_stop)

            time_decay = min(days_held / self.max_holding_days, 0.5)
            tightened_stop = entry_price + (1 - time_decay) * self.atr_multiplier * atr
            base_stop = min(base_stop, tightened_stop)

            return base_stop, (
                "trailing"
                if base_stop < entry_price + self.atr_multiplier * atr
                else "initial"
            )

    def should_take_profit(
        self, entry_price: float, current_price: float, atr: float, position_side: str
    ) -> bool:
        """Check if profit target reached."""
        if position_side == "long":
            profit_atr = (current_price - entry_price) / atr
        else:
            profit_atr = (entry_price - current_price) / atr

        return profit_atr >= self.profit_target_atr


def example_usage():
    """Example usage of adaptive risk manager."""
    import numpy as np
    import pandas as pd

    # Generate sample data
    np.random.seed(42)
    n_days = 252

    # BTC returns (for regime detection)
    btc_returns = pd.Series(
        np.random.normal(0.001, 0.03, n_days),
        index=pd.date_range("2024-01-01", periods=n_days, freq="D"),
    )

    # Symbol returns
    symbol_returns = pd.Series(
        np.random.normal(0.002, 0.05, n_days),
        index=pd.date_range("2024-01-01", periods=n_days, freq="D"),
    )

    # Initialize manager
    params = RiskParameters(
        max_portfolio_risk=0.02, max_position_size=0.10, volatility_target=0.15
    )
    manager = AdaptiveRiskManager(params)

    # Detect regime
    regime = manager.detect_regime(btc_returns)
    print(f"Current Regime: {regime.value}")
    print(f"Regime Multiplier: {manager.get_regime_multiplier(regime)}")

    # Calculate position size
    trade_stats = {"win_rate": 0.55, "avg_win": 0.03, "avg_loss": 0.02}

    returns_data = pd.DataFrame({"BTC": btc_returns, "ETH": symbol_returns})

    position_size = manager.get_optimal_position_size(
        symbol="ETH",
        btc_returns=btc_returns,
        symbol_returns=symbol_returns,
        current_positions={},
        returns_data=returns_data,
        trade_stats=trade_stats,
    )

    print(f"\nOptimal Position Size: {position_size:.2%}")

    # Simulate drawdown
    manager.update_drawdown(0.85)  # 15% drawdown
    print(f"\nRisk Report: {manager.get_risk_report()}")

    return manager


if __name__ == "__main__":
    example_usage()
