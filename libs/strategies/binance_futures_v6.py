"""
Binance Futures Strategy v6.0 - AI Consensus Edition.

This strategy was designed through 6 rounds of AI collaboration:
- ChatGPT, Gemini, DeepSeek, Perplexity, Claude, Grok, Clova

Key Features:
1. Market Regime Detection (Bull/Neutral/Bear)
2. Multi-Timeframe Analysis (1D -> 4H)
3. Signal Generation (Supertrend + KAMA + TSMOM_VW)
4. Quality Filters (CHOP, ADX, Volume, OBV)
5. BTC Gate (-5% threshold)
6. Regime-Adaptive Position Sizing

Performance Targets (Clova Backtest Validated):
- Win Rate: 48-52% (OOS)
- Annual Return: 25-45%
- Max MDD: 25%
- Risk per Trade: 1%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from libs.strategies.indicators import (
    ADX,
    ATR,
    Choppiness,
    EMA_series,
    KAMA_series,
    MACD_series,
    OBV_signal,
    Supertrend,
    TSMOM_volume_weighted,
)

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime states."""

    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"


class SignalType(str, Enum):
    """Signal types."""

    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"


@dataclass
class BinanceFuturesV6Config:
    """
    Binance Futures v6 Strategy Configuration.

    All parameters are based on AI consensus from 6 rounds of collaboration.
    """

    # Strategy metadata
    strategy_id: str = "binance_futures_v6"
    name: str = "Binance Futures v6 - AI Consensus"
    version: str = "6.0.0"

    # Target symbols (default top USDT-M pairs)
    symbols: List[str] = field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "LINKUSDT",
            "DOTUSDT",
        ]
    )

    # Timeframes
    higher_timeframe: str = "1d"  # For trend bias
    lower_timeframe: str = "4h"  # For entry signals

    # === Market Regime Detection (Clova Data) ===
    btc_ema_period: int = 200
    btc_52w_bars: int = 365  # Approximate 52 weeks in daily bars
    regime_bear_threshold: float = -30.0  # 52w high drawdown for BEAR
    regime_neutral_threshold: float = -15.0  # 52w high drawdown for NEUTRAL

    # === BTC Gate (Grok Confirmed: -5%) ===
    btc_gate_threshold: float = -5.0  # 24h BTC change threshold

    # === Multi-Timeframe (Grok Implementation) ===
    mtf_ema_fast: int = 50
    mtf_ema_slow: int = 200
    mtf_macd_fast: int = 12
    mtf_macd_slow: int = 26
    mtf_macd_signal: int = 9

    # === Supertrend (Primary Signal) ===
    supertrend_atr_period: int = 10
    supertrend_factor: float = 3.0

    # === KAMA (Confirmation) ===
    kama_period: int = 5
    kama_fast_sc: int = 2
    kama_slow_sc: int = 30
    kama_slope_period: int = 3

    # === TSMOM Volume-Weighted (Clova) ===
    tsmom_period: int = 20

    # === Choppiness Index (Gemini WFA: 55) ===
    chop_period: int = 14
    chop_threshold: float = 55.0  # < 55 = trending

    # === ADX Asymmetric (All AI Consensus) ===
    adx_period: int = 14
    adx_long_threshold: float = 25.0
    adx_short_threshold: float = 30.0

    # === Volume Filter (Grok) ===
    volume_sma_period: int = 20
    volume_multiplier: float = 1.0  # Current > SMA * multiplier

    # === OBV Filter (Grok: +5-10% win rate) ===
    obv_ema_period: int = 20

    # === Risk Management ===
    leverage_default: int = 5
    leverage_max: int = 10
    risk_per_trade: float = 0.01  # 1% per trade
    atr_stop_multiplier: float = 1.5

    max_positions: int = 5
    position_size_pct: float = 5.0  # % of account per position

    # === Drawdown Limits ===
    daily_loss_limit: float = -3.0  # %
    weekly_loss_limit: float = -7.0  # %
    max_drawdown: float = -15.0  # %
    max_mdd_stop: float = -25.0  # %

    # === Position Sizing by Regime ===
    position_mult_bull: float = 1.0
    position_mult_neutral: float = 0.5
    position_mult_bear_short: float = 0.5
    position_mult_bear_long: float = 0.0  # No longs in bear market

    # === Performance Targets (Clova Validated) ===
    expected_win_rate_bull: float = 0.55
    expected_win_rate_neutral: float = 0.50
    expected_win_rate_bear: float = 0.38
    target_win_rate: float = 0.52
    target_annual_return: float = 0.45


@dataclass
class Position:
    """Position tracking."""

    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    size: float
    leverage: int
    stop_loss: float
    regime_at_entry: MarketRegime
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    bars_held: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == float("inf"):
            self.lowest_price = self.entry_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "size": self.size,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "regime_at_entry": self.regime_at_entry.value,
            "highest_price": self.highest_price,
            "lowest_price": (
                self.lowest_price if self.lowest_price != float("inf") else None
            ),
            "bars_held": self.bars_held,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class Signal:
    """Signal output."""

    signal_type: SignalType
    symbol: str
    price: float
    reason: str
    timestamp: datetime
    regime: MarketRegime
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    filters_passed: Optional[Dict[str, bool]] = None
    indicators: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "price": self.price,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "regime": self.regime.value,
            "stop_loss": self.stop_loss,
            "position_size": self.position_size,
            "filters_passed": self.filters_passed,
            "indicators": self.indicators,
        }


class MarketRegimeDetector:
    """
    Market Regime Detection based on Clova backtest data.

    Bear market (2022): 38% win rate, 42% MDD
    Bull market (2023-2025): 60% win rate, 18% MDD
    """

    def __init__(self, config: BinanceFuturesV6Config):
        self.cfg = config

    def detect(
        self, btc_price: float, btc_ema_200: float, btc_52w_high: float
    ) -> MarketRegime:
        """
        Detect market regime.

        Args:
            btc_price: Current BTC price
            btc_ema_200: BTC 200-period EMA
            btc_52w_high: BTC 52-week high

        Returns:
            MarketRegime: BULL, NEUTRAL, or BEAR
        """
        if btc_52w_high <= 0:
            return MarketRegime.NEUTRAL

        drawdown_from_high = ((btc_price - btc_52w_high) / btc_52w_high) * 100
        below_ema = btc_price < btc_ema_200

        # Bear: Below EMA AND -30% from high
        if below_ema and drawdown_from_high <= self.cfg.regime_bear_threshold:
            return MarketRegime.BEAR

        # Neutral: Below EMA OR -15% from high
        if below_ema or drawdown_from_high <= self.cfg.regime_neutral_threshold:
            return MarketRegime.NEUTRAL

        return MarketRegime.BULL

    def get_position_multiplier(self, regime: MarketRegime, signal: str) -> float:
        """Get position size multiplier based on regime."""
        if regime == MarketRegime.BEAR:
            if signal == "LONG":
                return self.cfg.position_mult_bear_long  # 0 = no longs
            return self.cfg.position_mult_bear_short  # 0.5
        elif regime == MarketRegime.NEUTRAL:
            # Allow both directions in neutral with reduced size
            return self.cfg.position_mult_neutral  # 0.5
        return self.cfg.position_mult_bull  # 1.0

    def get_expected_win_rate(self, regime: MarketRegime) -> float:
        """Get expected win rate for regime (Clova data)."""
        if regime == MarketRegime.BEAR:
            return self.cfg.expected_win_rate_bear
        elif regime == MarketRegime.NEUTRAL:
            return self.cfg.expected_win_rate_neutral
        return self.cfg.expected_win_rate_bull


class MTFAnalyzer:
    """
    Multi-Timeframe Analysis (Grok Implementation).

    Uses 1D for trend bias, 4H for entries.
    Effect: Sharpe 0.33 -> 0.80
    """

    def __init__(self, config: BinanceFuturesV6Config):
        self.cfg = config

    def calculate_trend_bias(
        self, df_1d: pd.DataFrame
    ) -> Literal["LONG", "SHORT", "NEUTRAL"]:
        """
        Calculate 1D trend bias using EMA and MACD.

        Conditions:
        - LONG: EMA50 > EMA200 AND MACD > Signal
        - SHORT: EMA50 < EMA200 AND MACD < Signal
        - NEUTRAL: Conflicting signals
        """
        if len(df_1d) < self.cfg.mtf_ema_slow:
            return "NEUTRAL"

        close = df_1d["close"].values

        # EMA crossover
        ema_fast = EMA_series(close, self.cfg.mtf_ema_fast)
        ema_slow = EMA_series(close, self.cfg.mtf_ema_slow)
        ema_bullish = ema_fast[-1] > ema_slow[-1]

        # MACD
        macd_line, signal_line, _ = MACD_series(
            close,
            self.cfg.mtf_macd_fast,
            self.cfg.mtf_macd_slow,
            self.cfg.mtf_macd_signal,
        )
        macd_bullish = macd_line[-1] > signal_line[-1]

        if ema_bullish and macd_bullish:
            return "LONG"
        elif not ema_bullish and not macd_bullish:
            return "SHORT"
        return "NEUTRAL"

    def check_alignment(
        self,
        higher_tf_bias: Literal["LONG", "SHORT", "NEUTRAL"],
        lower_tf_signal: Literal["LONG", "SHORT", "NEUTRAL"],
    ) -> bool:
        """Check if higher and lower timeframes are aligned."""
        if higher_tf_bias == "NEUTRAL":
            return False
        return higher_tf_bias == lower_tf_signal


class SignalGenerator:
    """
    Signal generation using Supertrend + KAMA + TSMOM_VW.

    Conflict Resolution (Gemini Protocol):
    - Supertrend LONG + KAMA slope < 0 -> NEUTRAL
    - All indicators must align for signal
    """

    def __init__(self, config: BinanceFuturesV6Config):
        self.cfg = config

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all required indicators."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values

        # Supertrend
        st_value, st_direction = Supertrend(
            high, low, close, self.cfg.supertrend_atr_period, self.cfg.supertrend_factor
        )

        # KAMA
        kama = KAMA_series(
            close, self.cfg.kama_period, self.cfg.kama_fast_sc, self.cfg.kama_slow_sc
        )
        kama_slope = (
            kama[-1] - kama[-self.cfg.kama_slope_period]
            if len(kama) > self.cfg.kama_slope_period
            else 0
        )

        # TSMOM Volume-Weighted
        tsmom_vw = TSMOM_volume_weighted(close, volume, self.cfg.tsmom_period)

        # Choppiness Index
        chop = Choppiness(high, low, close, self.cfg.chop_period)

        # ADX
        adx = ADX(high, low, close, self.cfg.adx_period)

        # ATR (use supertrend ATR period)
        atr = ATR(high, low, close, self.cfg.supertrend_atr_period)

        # OBV Signal
        obv_bullish = OBV_signal(close, volume, self.cfg.obv_ema_period)

        # Volume Filter
        vol_sma = (
            np.mean(volume[-self.cfg.volume_sma_period :])
            if len(volume) >= self.cfg.volume_sma_period
            else np.mean(volume)
        )
        volume_pass = volume[-1] > vol_sma * self.cfg.volume_multiplier

        return {
            "supertrend_value": st_value,
            "supertrend_direction": st_direction,  # 1 = up/long, -1 = down/short
            "kama": kama[-1] if len(kama) > 0 else close[-1],
            "kama_slope": kama_slope,
            "tsmom_vw": tsmom_vw,
            "chop": chop,
            "adx": adx,
            "atr": atr,
            "obv_bullish": obv_bullish,
            "volume_pass": volume_pass,
            "current_price": close[-1],
        }

    def resolve_signal_conflict(
        self,
        supertrend_signal: Literal["LONG", "SHORT", "NEUTRAL"],
        kama_slope: float,
        tsmom_vw: float,
    ) -> Literal["LONG", "SHORT", "NEUTRAL"]:
        """
        Resolve signal conflicts (Gemini Protocol).

        If indicators conflict, return NEUTRAL (no entry).
        """
        if supertrend_signal == "LONG":
            if kama_slope < 0:  # Conflict: ST up but KAMA down
                return "NEUTRAL"
            if tsmom_vw < 0:  # Conflict: ST up but momentum down
                return "NEUTRAL"
            return "LONG"

        elif supertrend_signal == "SHORT":
            if kama_slope > 0:  # Conflict: ST down but KAMA up
                return "NEUTRAL"
            if tsmom_vw > 0:  # Conflict: ST down but momentum up
                return "NEUTRAL"
            return "SHORT"

        return "NEUTRAL"

    def get_supertrend_signal(
        self, direction: int
    ) -> Literal["LONG", "SHORT", "NEUTRAL"]:
        """Convert Supertrend direction to signal."""
        if direction == 1:
            return "LONG"
        elif direction == -1:
            return "SHORT"
        return "NEUTRAL"


class QualityFilter:
    """
    Quality filters for signal validation.

    All must pass for entry:
    1. CHOP < 55 (trending market)
    2. ADX > 25/30 (trend strength)
    3. Volume > 20 SMA (liquidity)
    4. OBV confirmation (momentum)
    """

    def __init__(self, config: BinanceFuturesV6Config):
        self.cfg = config

    def check_all(
        self, indicators: Dict[str, Any], signal: Literal["LONG", "SHORT"]
    ) -> Dict[str, bool]:
        """Check all quality filters."""
        results = {}

        # 1. Choppiness (< threshold = trending)
        results["chop"] = indicators["chop"] < self.cfg.chop_threshold

        # 2. ADX (asymmetric thresholds)
        if signal == "LONG":
            results["adx"] = indicators["adx"] > self.cfg.adx_long_threshold
        else:
            results["adx"] = indicators["adx"] > self.cfg.adx_short_threshold

        # 3. Volume
        results["volume"] = indicators["volume_pass"]

        # 4. OBV
        if signal == "LONG":
            results["obv"] = indicators["obv_bullish"]
        else:
            results["obv"] = not indicators["obv_bullish"]

        return results


class BinanceFuturesV6Strategy:
    """
    Binance Futures Strategy v6.0 - AI Consensus Edition.

    Designed through 6 rounds of multi-AI collaboration.

    Core Components:
    1. MarketRegimeDetector - Bull/Neutral/Bear classification
    2. MTFAnalyzer - 1D->4H timeframe alignment
    3. SignalGenerator - Supertrend+KAMA+TSMOM signals
    4. QualityFilter - CHOP/ADX/Volume/OBV gates
    5. RiskManager - Regime-adaptive position sizing
    """

    STRATEGY_ID = "binance_futures_v6"
    NAME = "Binance Futures v6 - AI Consensus"
    VERSION = "6.0.0"
    DESCRIPTION = "Multi-AI consensus strategy for Binance USDT-M Futures"

    def __init__(self, config: Optional[BinanceFuturesV6Config] = None):
        self.config = config or BinanceFuturesV6Config()

        # Components
        self.regime_detector = MarketRegimeDetector(self.config)
        self.mtf_analyzer = MTFAnalyzer(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.quality_filter = QualityFilter(self.config)

        # State
        self.positions: Dict[str, Position] = {}
        self.current_regime: MarketRegime = MarketRegime.NEUTRAL
        self.last_signal: Optional[Signal] = None

        # Risk tracking
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_drawdown: float = 0.0
        self.consecutive_losses: int = 0

        # BTC data cache
        self._btc_price: float = 0.0
        self._btc_ema_200: float = 0.0
        self._btc_52w_high: float = 0.0
        self._btc_change_24h: float = 0.0

        logger.info("[BFv6] Initialized %s", self.VERSION)

    def update_btc_data(
        self, price: float, ema_200: float, high_52w: float, change_24h: float
    ) -> None:
        """Update BTC market data for regime detection and BTC gate."""
        self._btc_price = price
        self._btc_ema_200 = ema_200
        self._btc_52w_high = high_52w
        self._btc_change_24h = change_24h

        # Update regime
        self.current_regime = self.regime_detector.detect(price, ema_200, high_52w)
        logger.debug(
            "[BFv6] Regime: %s, BTC: %.2f, 24h: %.2f%%",
            self.current_regime.value,
            price,
            change_24h,
        )

    def check_btc_gate(self, signal: str) -> bool:
        """
        BTC Gate Filter (Grok: -5% threshold).

        If BTC drops > 5% in 24h, block long entries.
        """
        if signal == "LONG" and self._btc_change_24h <= self.config.btc_gate_threshold:
            return False
        return True

    def check_risk_limits(self) -> tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        if self.daily_pnl <= self.config.daily_loss_limit:
            return (
                False,
                f"Daily loss limit: {self.daily_pnl:.2f}% <= {self.config.daily_loss_limit}%",
            )

        if self.weekly_pnl <= self.config.weekly_loss_limit:
            return (
                False,
                f"Weekly loss limit: {self.weekly_pnl:.2f}% <= {self.config.weekly_loss_limit}%",
            )

        if self.current_drawdown <= self.config.max_mdd_stop:
            return (
                False,
                f"Max drawdown: {self.current_drawdown:.2f}% <= {self.config.max_mdd_stop}%",
            )

        return True, "OK"

    def calculate_position_size(
        self, account_value: float, entry_price: float, atr: float, signal: str
    ) -> Dict[str, float]:
        """
        Calculate position size with regime adjustment.

        Formula (Gemini): Position = TargetRisk / StopDistance
        """
        # Base risk
        target_risk = account_value * self.config.risk_per_trade

        # Stop distance
        stop_distance = self.config.atr_stop_multiplier * atr
        stop_distance_pct = stop_distance / entry_price

        # Base position
        base_position = target_risk / stop_distance_pct if stop_distance_pct > 0 else 0

        # Leverage
        leverage = base_position / account_value if account_value > 0 else 0
        leverage = min(leverage, self.config.leverage_max)

        # Regime multiplier
        regime_mult = self.regime_detector.get_position_multiplier(
            self.current_regime, signal
        )

        # Drawdown adjustment
        dd_mult = 1.0
        if self.current_drawdown < -7:
            dd_mult = 0.25
        elif self.current_drawdown < -5:
            dd_mult = 0.5
        elif self.current_drawdown < -3:
            dd_mult = 0.75

        final_leverage = leverage * regime_mult * dd_mult
        final_position = account_value * final_leverage

        return {
            "position_notional": round(final_position, 2),
            "leverage": round(final_leverage, 2),
            "regime_multiplier": regime_mult,
            "dd_multiplier": dd_mult,
            "stop_loss_long": entry_price - stop_distance,
            "stop_loss_short": entry_price + stop_distance,
            "risk_amount": target_risk * regime_mult * dd_mult,
        }

    def generate_signal(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        account_value: float = 10000.0,
    ) -> Signal:
        """
        Generate trading signal for a symbol.

        Filter Chain:
        1. Risk limits check
        2. Market regime check
        3. MTF alignment check
        4. Signal generation (Supertrend+KAMA+TSMOM)
        5. BTC Gate
        6. Quality filters (CHOP, ADX, Volume, OBV)
        7. Position sizing
        """
        now = datetime.now()
        filters_passed = {}

        # === 1. Check Existing Position (BEFORE risk limits - always allow exits) ===
        if symbol in self.positions:
            exit_signal = self._check_exit_conditions(symbol, df_4h)
            if exit_signal:
                return exit_signal
            self._update_position(symbol, df_4h)
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=df_4h["close"].iloc[-1],
                reason="Position held",
                timestamp=now,
                regime=self.current_regime,
            )

        # === 2. Risk Limits (only for new entries) ===
        risk_ok, risk_reason = self.check_risk_limits()
        if not risk_ok:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=df_4h["close"].iloc[-1] if len(df_4h) > 0 else 0,
                reason=f"Risk limit: {risk_reason}",
                timestamp=now,
                regime=self.current_regime,
                filters_passed={"risk": False},
            )
        filters_passed["risk"] = True

        # === 3. Max Positions Check ===
        if len(self.positions) >= self.config.max_positions:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=df_4h["close"].iloc[-1],
                reason=f"Max positions ({self.config.max_positions}) reached",
                timestamp=now,
                regime=self.current_regime,
                filters_passed={"max_positions": False},
            )
        filters_passed["max_positions"] = True

        # === 4. MTF Analysis ===
        mtf_bias = self.mtf_analyzer.calculate_trend_bias(df_1d)
        filters_passed["mtf_bias"] = mtf_bias

        if mtf_bias == "NEUTRAL":
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=df_4h["close"].iloc[-1],
                reason="MTF bias neutral",
                timestamp=now,
                regime=self.current_regime,
                filters_passed=filters_passed,
            )

        # === 5. Calculate Indicators ===
        indicators = self.signal_generator.calculate_indicators(df_4h)

        # === 6. Supertrend Signal ===
        st_signal = self.signal_generator.get_supertrend_signal(
            indicators["supertrend_direction"]
        )

        # === 7. MTF Alignment ===
        if not self.mtf_analyzer.check_alignment(mtf_bias, st_signal):
            filters_passed["mtf_alignment"] = False
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=indicators["current_price"],
                reason=f"MTF misalignment: {mtf_bias} vs {st_signal}",
                timestamp=now,
                regime=self.current_regime,
                filters_passed=filters_passed,
                indicators=indicators,
            )
        filters_passed["mtf_alignment"] = True

        # === 8. Signal Conflict Resolution ===
        resolved_signal = self.signal_generator.resolve_signal_conflict(
            st_signal, indicators["kama_slope"], indicators["tsmom_vw"]
        )

        if resolved_signal == "NEUTRAL":
            filters_passed["signal_conflict"] = False
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=indicators["current_price"],
                reason="Signal conflict (KAMA/TSMOM divergence)",
                timestamp=now,
                regime=self.current_regime,
                filters_passed=filters_passed,
                indicators=indicators,
            )
        filters_passed["signal_conflict"] = True

        # === 9. BTC Gate ===
        btc_gate_pass = self.check_btc_gate(resolved_signal)
        filters_passed["btc_gate"] = btc_gate_pass

        if not btc_gate_pass:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=indicators["current_price"],
                reason=f"BTC Gate blocked: 24h change {self._btc_change_24h:.2f}%",
                timestamp=now,
                regime=self.current_regime,
                filters_passed=filters_passed,
                indicators=indicators,
            )

        # === 10. Quality Filters ===
        quality_results = self.quality_filter.check_all(indicators, resolved_signal)
        filters_passed.update(quality_results)

        if not all(quality_results.values()):
            failed = [k for k, v in quality_results.items() if not v]
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=indicators["current_price"],
                reason=f"Quality filter failed: {', '.join(failed)}",
                timestamp=now,
                regime=self.current_regime,
                filters_passed=filters_passed,
                indicators=indicators,
            )

        # === 11. Regime Position Check ===
        position_mult = self.regime_detector.get_position_multiplier(
            self.current_regime, resolved_signal
        )

        if position_mult == 0:
            filters_passed["regime_position"] = False
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=indicators["current_price"],
                reason=f"Regime {self.current_regime.value} blocks {resolved_signal}",
                timestamp=now,
                regime=self.current_regime,
                filters_passed=filters_passed,
                indicators=indicators,
            )
        filters_passed["regime_position"] = True

        # === 12. Calculate Position Size ===
        position_info = self.calculate_position_size(
            account_value,
            indicators["current_price"],
            indicators["atr"],
            resolved_signal,
        )

        # === 13. Generate Entry Signal ===
        signal_type = SignalType.LONG if resolved_signal == "LONG" else SignalType.SHORT
        stop_loss = (
            position_info["stop_loss_long"]
            if resolved_signal == "LONG"
            else position_info["stop_loss_short"]
        )

        reason = (
            f"{resolved_signal}: ST={indicators['supertrend_direction']}, "
            f"KAMA_slope={indicators['kama_slope']:.4f}, "
            f"TSMOM={indicators['tsmom_vw']:.4f}, "
            f"CHOP={indicators['chop']:.1f}, ADX={indicators['adx']:.1f}"
        )

        signal = Signal(
            signal_type=signal_type,
            symbol=symbol,
            price=indicators["current_price"],
            reason=reason,
            timestamp=now,
            regime=self.current_regime,
            stop_loss=stop_loss,
            position_size=position_info["position_notional"],
            filters_passed=filters_passed,
            indicators=indicators,
        )

        self.last_signal = signal
        logger.info(
            "[BFv6] Signal: %s %s @ %.2f (Regime: %s)",
            signal_type.value,
            symbol,
            indicators["current_price"],
            self.current_regime.value,
        )

        return signal

    def _check_exit_conditions(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Check exit conditions for existing position."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        now = datetime.now()
        current_price = df["close"].iloc[-1]

        # Calculate ATR for trailing stop
        atr = ATR(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            self.config.supertrend_atr_period,
        )

        if position.side == "LONG":
            # Trailing stop for long
            trailing_stop = position.highest_price - (
                atr * self.config.atr_stop_multiplier
            )

            if current_price < trailing_stop or current_price < position.stop_loss:
                return Signal(
                    signal_type=SignalType.EXIT_LONG,
                    symbol=symbol,
                    price=current_price,
                    reason=f"Stop hit: price {current_price:.2f} < stop {min(trailing_stop, position.stop_loss):.2f}",
                    timestamp=now,
                    regime=self.current_regime,
                )
        else:  # SHORT
            # Trailing stop for short
            trailing_stop = position.lowest_price + (
                atr * self.config.atr_stop_multiplier
            )

            if current_price > trailing_stop or current_price > position.stop_loss:
                return Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    symbol=symbol,
                    price=current_price,
                    reason=f"Stop hit: price {current_price:.2f} > stop {max(trailing_stop, position.stop_loss):.2f}",
                    timestamp=now,
                    regime=self.current_regime,
                )

        return None

    def _update_position(self, symbol: str, df: pd.DataFrame) -> None:
        """Update position state."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        current_price = df["close"].iloc[-1]
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]

        pos.bars_held += 1

        if pos.side == "LONG":
            pos.highest_price = max(pos.highest_price, current_high)
            pos.pnl_pct = (
                ((current_price - pos.entry_price) / pos.entry_price)
                * 100
                * pos.leverage
            )
        else:
            pos.lowest_price = min(pos.lowest_price, current_low)
            pos.pnl_pct = (
                ((pos.entry_price - current_price) / pos.entry_price)
                * 100
                * pos.leverage
            )

    def open_position(self, symbol: str, signal: Signal) -> None:
        """Open a new position."""
        side = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"

        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            size=signal.position_size or (self.config.position_size_pct / 100.0),
            leverage=self.config.leverage_default,
            stop_loss=signal.stop_loss or signal.price,
            regime_at_entry=self.current_regime,
            highest_price=signal.price,
            lowest_price=signal.price,
        )

        logger.info(
            "[BFv6] Opened %s %s @ %.2f (Regime: %s)",
            side,
            symbol,
            signal.price,
            self.current_regime.value,
        )

    def close_position(self, symbol: str, signal: Signal) -> float:
        """Close position and return PnL percent."""
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]

        if pos.side == "LONG":
            pnl_pct = (
                ((signal.price - pos.entry_price) / pos.entry_price)
                * 100
                * pos.leverage
            )
        else:
            pnl_pct = (
                ((pos.entry_price - signal.price) / pos.entry_price)
                * 100
                * pos.leverage
            )

        # Update risk tracking
        self.daily_pnl += pnl_pct
        self.weekly_pnl += pnl_pct

        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        del self.positions[symbol]

        logger.info(
            "[BFv6] Closed %s %s @ %.2f, PnL=%.2f%%",
            pos.side,
            symbol,
            signal.price,
            pnl_pct,
        )

        return pnl_pct

    def get_state(self) -> Dict[str, Any]:
        """Return complete strategy state."""
        return {
            "strategy_id": self.STRATEGY_ID,
            "version": self.VERSION,
            "regime": self.current_regime.value,
            "expected_win_rate": self.regime_detector.get_expected_win_rate(
                self.current_regime
            ),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "last_signal": self.last_signal.to_dict() if self.last_signal else None,
            "btc_data": {
                "price": self._btc_price,
                "ema_200": self._btc_ema_200,
                "52w_high": self._btc_52w_high,
                "change_24h": self._btc_change_24h,
            },
            "risk": {
                "daily_pnl": self.daily_pnl,
                "weekly_pnl": self.weekly_pnl,
                "current_drawdown": self.current_drawdown,
                "consecutive_losses": self.consecutive_losses,
            },
            "config": {
                "symbols": self.config.symbols,
                "leverage": self.config.leverage_default,
                "max_positions": self.config.max_positions,
                "btc_gate_threshold": self.config.btc_gate_threshold,
                "chop_threshold": self.config.chop_threshold,
                "adx_long_threshold": self.config.adx_long_threshold,
                "adx_short_threshold": self.config.adx_short_threshold,
            },
        }

    def reset(self) -> None:
        """Reset strategy state."""
        self.positions.clear()
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.consecutive_losses = 0
        self.last_signal = None
        logger.info("[BFv6] Strategy reset")

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0
        logger.debug("[BFv6] Daily counters reset")

    def reset_weekly(self) -> None:
        """Reset weekly counters."""
        self.weekly_pnl = 0.0
        logger.debug("[BFv6] Weekly counters reset")
