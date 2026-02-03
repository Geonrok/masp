"""
ATLAS-Futures P0-4 Squeeze-Surge Strategy.
Protocol: v2.6.2-r1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

try:
    import ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"


@dataclass
class ATLASFuturesConfig:
    """ATLAS-Futures v2.6.2-r1 config."""

    strategy_id: str = "atlas_futures_p04"
    name: str = "P0-4 Squeeze-Surge"
    version: str = "v2.6.2-r1"

    timeframe: str = "4h"
    symbols: List[str] = field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    )
    leverage: int = 3
    max_positions: int = 3
    position_size_pct: float = 1.6

    bbwp_lookback: int = 252
    bbwp_threshold: float = 10.0
    squeeze_min_bars: int = 6
    adx_period: int = 14
    adx_threshold: float = 12.0
    ema_trend_period: int = 200
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    chandelier_multiplier: float = 3.0

    rvol_lookback: int = 20
    rvol_major: float = 1.5
    rvol_alt: float = 3.0
    major_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])

    failure_bars_major: int = 4
    failure_bars_alt: int = 5
    failure_threshold_pct: float = -0.16

    daily_loss_limit: float = -5.0
    weekly_loss_limit: float = -10.0
    monthly_loss_limit: float = -15.0
    max_drawdown: float = -10.0
    max_drawdown_stop: float = -20.0
    consecutive_loss_pause: int = 10

    link_concentration_threshold: float = 90.0


@dataclass
class Position:
    """Position state."""

    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    size: float
    leverage: int
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
        """Return JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "size": self.size,
            "leverage": self.leverage,
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
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict."""
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "price": self.price,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


class ATLASFuturesStrategy:
    """
    ATLAS-Futures P0-4 Squeeze-Surge strategy.

    Core idea: detect volatility squeeze and breakout surge.
    """

    STRATEGY_ID = "atlas_futures_p04"
    NAME = "P0-4 Squeeze-Surge"
    VERSION = "v2.6.2-r1"
    DESCRIPTION = "ATLAS-Futures volatility squeeze + surge strategy"

    def __init__(self, config: Optional[ATLASFuturesConfig] = None):
        self.config = config or ATLASFuturesConfig()
        self.positions: Dict[str, Position] = {}
        self.last_signal: Optional[Signal] = None

        self.consecutive_losses: int = 0
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_drawdown: float = 0.0

        self.track_b_pnl: float = 0.0
        self.track_c_pnl: float = 0.0
        self.link_blocked: bool = False

        self._squeeze_counters: Dict[str, int] = {}

        logger.info("[ATLAS] Initialized %s", self.VERSION)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        if ta is None:
            raise ImportError("ta library required: pip install ta")

        df = df.copy()

        bb = ta.volatility.BollingerBands(
            df["close"],
            window=self.config.bb_period,
            window_dev=self.config.bb_std,
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()

        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        df["bbwp"] = (
            df["bb_width"]
            .rolling(self.config.bbwp_lookback)
            .apply(
                lambda x: percentileofscore(x, x.iloc[-1]) if len(x) >= 10 else 50,
                raw=False,
            )
        )

        df["adx"] = ta.trend.adx(
            df["high"],
            df["low"],
            df["close"],
            window=self.config.adx_period,
        )

        df["ema_200"] = ta.trend.ema_indicator(
            df["close"],
            window=self.config.ema_trend_period,
        )

        df["atr"] = ta.volatility.average_true_range(
            df["high"],
            df["low"],
            df["close"],
            window=self.config.atr_period,
        )

        avg_volume = df["volume"].rolling(self.config.rvol_lookback).mean()
        df["rvol"] = df["volume"] / avg_volume

        return df

    def _count_squeeze_bars(self, symbol: str, bbwp: float) -> int:
        """Count consecutive squeeze bars."""
        if bbwp < self.config.bbwp_threshold:
            self._squeeze_counters[symbol] = self._squeeze_counters.get(symbol, 0) + 1
        else:
            self._squeeze_counters[symbol] = 0
        return self._squeeze_counters.get(symbol, 0)

    def _get_rvol_threshold(self, symbol: str) -> float:
        """RVOL threshold by symbol."""
        if symbol in self.config.major_symbols:
            return self.config.rvol_major
        return self.config.rvol_alt

    def _get_failure_bars(self, symbol: str) -> int:
        """Failure exit bars by symbol."""
        if symbol in self.config.major_symbols:
            return self.config.failure_bars_major
        return self.config.failure_bars_alt

    def check_risk_limits(self) -> tuple[bool, str]:
        """Risk limit checks."""
        if self.consecutive_losses >= self.config.consecutive_loss_pause:
            return False, (
                f"Consecutive losses: {self.consecutive_losses} >= "
                f"{self.config.consecutive_loss_pause}"
            )

        if self.daily_pnl <= self.config.daily_loss_limit:
            return False, (
                f"Daily loss limit: {self.daily_pnl:.2f}% <= {self.config.daily_loss_limit}%"
            )

        if self.weekly_pnl <= self.config.weekly_loss_limit:
            return False, (
                f"Weekly loss limit: {self.weekly_pnl:.2f}% <= {self.config.weekly_loss_limit}%"
            )

        if self.monthly_pnl <= self.config.monthly_loss_limit:
            return False, (
                f"Monthly loss limit: {self.monthly_pnl:.2f}% <= {self.config.monthly_loss_limit}%"
            )

        if self.current_drawdown <= self.config.max_drawdown_stop:
            return False, (
                f"Max drawdown stop: {self.current_drawdown:.2f}% <= {self.config.max_drawdown_stop}%"
            )

        return True, "OK"

    def check_track_switch(self) -> Optional[str]:
        """SSOT track switch check."""
        if self.track_b_pnl < 0 and self.track_c_pnl > 0:
            self.link_blocked = True
            logger.warning("[ATLAS] Track switch: B -> C (LINK blocked)")
            return "SWITCH_TO_C"

        if self.track_b_pnl > self.track_c_pnl + 5:
            self.link_blocked = False
            logger.info("[ATLAS] Track switch: C -> B (LINK unblocked)")
            return "REVERT_TO_B"

        return None

    def check_link_concentration(self) -> None:
        """
        Check LINK concentration (block if > threshold).
        """
        if not self.positions:
            return

        total_value = sum(
            abs(pos.size * pos.entry_price) for pos in self.positions.values()
        )

        if total_value == 0:
            return

        link_value = sum(
            abs(pos.size * pos.entry_price)
            for symbol, pos in self.positions.items()
            if "LINK" in symbol
        )

        concentration = (link_value / total_value) * 100

        if concentration > self.config.link_concentration_threshold:
            self.link_blocked = True
            logger.warning(
                "[ATLAS] LINK concentration %.1f%% > %.1f%% threshold - blocking",
                concentration,
                self.config.link_concentration_threshold,
            )

    def check_entry_conditions(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[Signal]:
        """Entry conditions."""
        if len(data) < self.config.ema_trend_period:
            return None

        latest = data.iloc[-1]
        now = datetime.now()

        squeeze_bars = self._count_squeeze_bars(symbol, latest["bbwp"])

        squeeze_ok = (
            latest["bbwp"] < self.config.bbwp_threshold
            and squeeze_bars >= self.config.squeeze_min_bars
        )
        adx_ok = latest["adx"] >= self.config.adx_threshold
        rvol_ok = latest["rvol"] > self._get_rvol_threshold(symbol)

        if not (squeeze_ok and adx_ok and rvol_ok):
            return None

        if latest["close"] > latest["ema_200"] and latest["close"] > latest["bb_upper"]:
            stop_loss = latest["close"] - (
                latest["atr"] * self.config.chandelier_multiplier
            )
            return Signal(
                signal_type=SignalType.LONG,
                symbol=symbol,
                price=latest["close"],
                stop_loss=stop_loss,
                position_size=self.config.position_size_pct / 100.0,
                reason=(
                    "LONG: Squeeze breakout "
                    f"(BBWP={latest['bbwp']:.1f}%, ADX={latest['adx']:.1f}, RVOL={latest['rvol']:.2f})"
                ),
                timestamp=now,
                metadata={
                    "bbwp": latest["bbwp"],
                    "adx": latest["adx"],
                    "rvol": latest["rvol"],
                    "squeeze_bars": squeeze_bars,
                },
            )

        if latest["close"] < latest["ema_200"] and latest["close"] < latest["bb_lower"]:
            stop_loss = latest["close"] + (
                latest["atr"] * self.config.chandelier_multiplier
            )
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=symbol,
                price=latest["close"],
                stop_loss=stop_loss,
                position_size=self.config.position_size_pct / 100.0,
                reason=(
                    "SHORT: Squeeze breakdown "
                    f"(BBWP={latest['bbwp']:.1f}%, ADX={latest['adx']:.1f}, RVOL={latest['rvol']:.2f})"
                ),
                timestamp=now,
                metadata={
                    "bbwp": latest["bbwp"],
                    "adx": latest["adx"],
                    "rvol": latest["rvol"],
                    "squeeze_bars": squeeze_bars,
                },
            )

        return None

    def check_exit_conditions(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[Signal]:
        """Exit conditions."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        latest = data.iloc[-1]
        now = datetime.now()

        if position.side == "LONG":
            chandelier_stop = position.highest_price - (
                latest["atr"] * self.config.chandelier_multiplier
            )
            if latest["close"] < chandelier_stop:
                return Signal(
                    signal_type=SignalType.EXIT_LONG,
                    symbol=symbol,
                    price=latest["close"],
                    reason=f"Chandelier Exit: {latest['close']:.2f} < {chandelier_stop:.2f}",
                    timestamp=now,
                )
        else:
            chandelier_stop = position.lowest_price + (
                latest["atr"] * self.config.chandelier_multiplier
            )
            if latest["close"] > chandelier_stop:
                return Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    symbol=symbol,
                    price=latest["close"],
                    reason=f"Chandelier Exit: {latest['close']:.2f} > {chandelier_stop:.2f}",
                    timestamp=now,
                )

        failure_bars = self._get_failure_bars(symbol)
        if position.bars_held >= failure_bars:
            pnl_pct = self._calculate_pnl_pct(position, latest["close"])
            if pnl_pct < self.config.failure_threshold_pct:
                exit_type = (
                    SignalType.EXIT_LONG
                    if position.side == "LONG"
                    else SignalType.EXIT_SHORT
                )
                return Signal(
                    signal_type=exit_type,
                    symbol=symbol,
                    price=latest["close"],
                    reason=f"Failure Exit: {position.bars_held} bars, PnL={pnl_pct:.2f}%",
                    timestamp=now,
                )

        return None

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate signal for a symbol."""
        now = datetime.now()

        data = self.calculate_indicators(data)

        risk_ok, risk_reason = self.check_risk_limits()
        if not risk_ok:
            signal = Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=data.iloc[-1]["close"],
                reason=f"Risk limit: {risk_reason}",
                timestamp=now,
            )
            self.last_signal = signal
            return signal

        if symbol == "LINKUSDT" and self.link_blocked:
            signal = Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=data.iloc[-1]["close"],
                reason="LINK blocked due to concentration",
                timestamp=now,
            )
            self.last_signal = signal
            return signal

        if symbol in self.positions:
            exit_signal = self.check_exit_conditions(symbol, data)
            if exit_signal:
                self.last_signal = exit_signal
                return exit_signal
            self._update_position(symbol, data.iloc[-1])

        if (
            len(self.positions) < self.config.max_positions
            and symbol not in self.positions
        ):
            entry_signal = self.check_entry_conditions(symbol, data)
            if entry_signal:
                self.last_signal = entry_signal
                return entry_signal

        signal = Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            price=data.iloc[-1]["close"],
            reason="No signal",
            timestamp=now,
        )
        self.last_signal = signal
        return signal

    def _update_position(self, symbol: str, latest: pd.Series) -> None:
        """Update position state."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        pos.bars_held += 1

        if pos.side == "LONG":
            pos.highest_price = max(pos.highest_price, latest["high"])
        else:
            pos.lowest_price = min(pos.lowest_price, latest["low"])

        pos.pnl_pct = self._calculate_pnl_pct(pos, latest["close"])

    def _calculate_pnl_pct(self, position: Position, current_price: float) -> float:
        """Calculate PnL percent."""
        if position.side == "LONG":
            return (
                ((current_price - position.entry_price) / position.entry_price)
                * 100
                * position.leverage
            )
        return (
            ((position.entry_price - current_price) / position.entry_price)
            * 100
            * position.leverage
        )

    def open_position(self, symbol: str, signal: Signal) -> None:
        """Open position."""
        side = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            size=signal.position_size or self.config.position_size_pct / 100.0,
            leverage=self.config.leverage,
            highest_price=signal.price,
            lowest_price=signal.price,
        )
        logger.info("[ATLAS] Opened %s %s @ %.2f", side, symbol, signal.price)

    def close_position(self, symbol: str, signal: Signal) -> float:
        """Close position and return PnL percent."""
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        pnl_pct = self._calculate_pnl_pct(pos, signal.price)

        self.daily_pnl += pnl_pct
        self.weekly_pnl += pnl_pct
        self.monthly_pnl += pnl_pct

        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        del self.positions[symbol]
        logger.info(
            "[ATLAS] Closed %s %s @ %.2f, PnL=%.2f%%",
            pos.side,
            symbol,
            signal.price,
            pnl_pct,
        )

        return pnl_pct

    def get_state(self) -> Dict[str, Any]:
        """Return strategy state (JSON-safe)."""
        return {
            "strategy_id": self.STRATEGY_ID,
            "version": self.VERSION,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "last_signal": self.last_signal.to_dict() if self.last_signal else None,
            "risk": {
                "consecutive_losses": self.consecutive_losses,
                "daily_pnl": self.daily_pnl,
                "weekly_pnl": self.weekly_pnl,
                "monthly_pnl": self.monthly_pnl,
                "current_drawdown": self.current_drawdown,
            },
            "track": {
                "b_pnl": self.track_b_pnl,
                "c_pnl": self.track_c_pnl,
                "link_blocked": self.link_blocked,
            },
            "config": {
                key: value
                for key, value in self.config.__dict__.items()
                if not callable(value)
            },
        }

    def reset(self) -> None:
        """Reset strategy state."""
        self.positions.clear()
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self._squeeze_counters.clear()
        logger.info("[ATLAS] Strategy reset")
