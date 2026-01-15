"""
Mock Strategy - Deterministic strategy for Phase 0 testing.
Generates reproducible BUY/SELL/HOLD/SKIP decisions based on symbol and date.
"""

import hashlib
from datetime import datetime

import pytz

from libs.strategies.base import Action, BaseStrategy, Decision, StrategyContext


class MockStrategy(BaseStrategy):
    """
    Mock strategy that generates deterministic decisions.
    
    The decision for each symbol is determined by:
    - Hash of (symbol + current_date)
    - This ensures reproducibility: same symbol on same day = same decision
    - Different symbols get different decisions for variety
    """
    
    strategy_id = "mock_strategy"
    name = "Mock Strategy"
    version = "1.0.0"
    description = "Deterministic mock strategy for testing"
    
    # Action mapping based on hash modulo
    ACTION_MAP = {
        0: Action.BUY,
        1: Action.SELL,
        2: Action.HOLD,
        3: Action.SKIP,
    }
    
    # Notes explaining each action
    NOTES_MAP = {
        Action.BUY: "Mock signal: Simulated bullish indicator crossover",
        Action.SELL: "Mock signal: Simulated bearish divergence detected",
        Action.HOLD: "Mock signal: No clear trend, maintaining current position",
        Action.SKIP: "Mock signal: Insufficient data or market conditions unfavorable",
    }
    
    def __init__(self):
        super().__init__()
        self._kst = pytz.timezone("Asia/Seoul")
    
    def _get_deterministic_action(self, symbol: str, date_str: str) -> Action:
        """
        Get deterministic action based on symbol and date.
        
        Args:
            symbol: Trading symbol
            date_str: Date string (YYYY-MM-DD)
        
        Returns:
            Action enum value
        """
        # Create hash input
        hash_input = f"{symbol}:{date_str}"
        
        # Generate hash
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
        
        # Map to action
        action_idx = hash_int % 4
        return self.ACTION_MAP[action_idx]
    
    def _generate_mock_metrics(self, symbol: str, action: Action) -> dict:
        """Generate mock metrics for the decision."""
        # Use symbol hash for reproducible metrics
        hash_bytes = hashlib.sha256(symbol.encode()).digest()
        
        # Generate mock values
        rsi = 30 + (hash_bytes[0] % 40)  # RSI between 30-70
        macd = (hash_bytes[1] - 128) / 100  # MACD between -1.28 and 1.27
        volume_ratio = 0.5 + (hash_bytes[2] / 255)  # Volume ratio 0.5-1.5
        
        return {
            "mock_rsi": round(rsi, 2),
            "mock_macd": round(macd, 4),
            "mock_volume_ratio": round(volume_ratio, 2),
            "confidence": round(0.5 + (hash_bytes[3] / 510), 2),  # 0.5-1.0
        }
    
    def execute(self, ctx: StrategyContext) -> list[Decision]:
        """
        Execute mock strategy.
        
        Args:
            ctx: Strategy context
        
        Returns:
            List of deterministic decisions
        """
        decisions = []
        
        # Get current date in KST for deterministic hash
        now_kst = datetime.now(self._kst)
        date_str = now_kst.strftime("%Y-%m-%d")
        
        for symbol in ctx.symbols:
            # Get deterministic action
            action = self._get_deterministic_action(symbol, date_str)
            
            # Generate notes
            notes = self.NOTES_MAP[action]
            
            # Generate mock metrics
            metrics = self._generate_mock_metrics(symbol, action)
            
            decision = Decision(
                symbol=symbol,
                action=action,
                notes=notes,
                metrics=metrics,
            )
            decisions.append(decision)
        
        return decisions


# Alternative mock strategy for variety
class TrendFollowingMockStrategy(BaseStrategy):
    """
    Another mock strategy with different behavior.
    Simulates a trend-following approach.
    """
    
    strategy_id = "trend_following_mock"
    name = "Trend Following Mock"
    version = "1.0.0"
    description = "Mock trend-following strategy"
    
    def __init__(self):
        super().__init__()
        self._kst = pytz.timezone("Asia/Seoul")
    
    def execute(self, ctx: StrategyContext) -> list[Decision]:
        """Execute trend following mock strategy."""
        decisions = []
        
        now_kst = datetime.now(self._kst)
        hour = now_kst.hour
        
        for symbol in ctx.symbols:
            # Different logic: based on hour and symbol
            hash_input = f"{symbol}:{hour}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()
            score = hash_bytes[0] / 255  # 0-1 score
            
            if score > 0.7:
                action = Action.BUY
                notes = "Trend mock: Strong upward momentum detected"
            elif score < 0.3:
                action = Action.SELL
                notes = "Trend mock: Strong downward momentum detected"
            elif score > 0.5:
                action = Action.HOLD
                notes = "Trend mock: Weak upward trend, holding"
            else:
                action = Action.SKIP
                notes = "Trend mock: No clear trend direction"
            
            decisions.append(Decision(
                symbol=symbol,
                action=action,
                notes=notes,
                metrics={"trend_score": round(score, 3)},
            ))
        
        return decisions


