"""
ATLAS-Futures P0-4 Squeeze-Surge Strategy.
Protocol: v2.6.2-r1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from scipy.stats import percentileofscore

from libs.strategies.base import BaseStrategy
from libs.strategies.base import Signal as BaseSignal
from libs.strategies.base import TradeSignal

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
        default_factory=lambda: ["ALL_USDT_PERP"]
    )
    leverage: int = 3
    max_positions: int = 3
    position_size_pct: float = 20.0

    # BTC Gate
    btc_gate_enabled: bool = True
    btc_gate_ma_period: int = 50
    btc_gate_symbol: str = "BTCUSDT"

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


class ATLASFuturesStrategy(BaseStrategy):
    """
    ATLAS-Futures P0-4 Squeeze-Surge strategy.

    Core idea: detect volatility squeeze and breakout surge.
    """

    strategy_id: str = "atlas_futures_p04"
    name: str = "P0-4 Squeeze-Surge"
    version: str = "v2.6.2-r1"
    description: str = "ATLAS-Futures volatility squeeze + surge strategy"

    # Backward compat aliases
    STRATEGY_ID = "atlas_futures_p04"
    NAME = "P0-4 Squeeze-Surge"
    VERSION = "v2.6.2-r1"
    DESCRIPTION = "ATLAS-Futures volatility squeeze + surge strategy"

    def __init__(self, config: Optional[ATLASFuturesConfig] = None):
        super().__init__(name="ATLAS-Futures")
        self.config = config or ATLASFuturesConfig()
        self.positions: Dict[str, Position] = {}
        self.last_signal: Optional[Signal] = None

        self.consecutive_losses: int = 0
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.peak_equity: float = 100.0
        self.current_drawdown: float = 0.0

        self.track_b_pnl: float = 0.0
        self.track_c_pnl: float = 0.0
        self.link_blocked: bool = False

        self._squeeze_counters: Dict[str, int] = {}
        self._cumulative_pnl: float = 0.0  # never resets daily, for drawdown
        self._generation: int = 0  # increments per generate_signals() cycle
        self._last_gen: Dict[str, int] = {}  # per-symbol last processed generation

        # BTC Gate state: updated via update_btc_gate() before signal generation
        self._btc_gate_long: bool = False  # True = BTC > SMA(50) → longs allowed
        self._btc_gate_short: bool = False  # True = BTC < SMA(50) → shorts allowed

        # Date tracking for auto-reset
        self._last_reset_date = datetime.now().date()
        self._last_reset_week = datetime.now().isocalendar()[1]
        self._last_reset_month = datetime.now().month

        # Market data adapter
        self._market_data = None

        logger.info("[ATLAS] Initialized %s", self.VERSION)

    def set_market_data(self, adapter) -> None:
        """Set market data adapter for BaseStrategy generate_signals."""
        self._market_data = adapter

    def update_btc_gate(self, btc_daily_df: Optional[pd.DataFrame] = None) -> None:
        """Update BTC gate using daily close vs SMA(50).

        Uses the PREVIOUS day's close (shift-1) to avoid look-ahead bias.
        Call this once per 4h bar cycle, before generate_signal()/generate_signals().

        Args:
            btc_daily_df: BTC daily OHLCV DataFrame with 'close' column.
                          If None, attempts to fetch via market_data adapter.
        """
        if not self.config.btc_gate_enabled:
            self._btc_gate_long = True
            self._btc_gate_short = True
            return

        df = btc_daily_df
        if df is None and self._market_data:
            try:
                ohlcv_list = self._market_data.get_ohlcv(
                    self.config.btc_gate_symbol, interval="1d", limit=100
                )
                if ohlcv_list and len(ohlcv_list) > 0:
                    df = pd.DataFrame(
                        [{"close": c.close} for c in ohlcv_list]
                    )
            except Exception as exc:
                logger.warning("[ATLAS] BTC gate data fetch failed: %s", exc)

        if df is None or len(df) < self.config.btc_gate_ma_period + 1:
            logger.warning("[ATLAS] BTC gate: insufficient data, blocking both directions")
            self._btc_gate_long = False
            self._btc_gate_short = False
            return

        close = df["close"].astype(float)
        sma = close.rolling(self.config.btc_gate_ma_period).mean()

        # Use previous day's values (shift-1 equivalent: second-to-last row)
        prev_close = float(close.iloc[-2])
        prev_sma = float(sma.iloc[-2])

        self._btc_gate_long = prev_close > prev_sma
        self._btc_gate_short = prev_close < prev_sma

        logger.debug(
            "[ATLAS] BTC Gate: close=%.2f sma50=%.2f → long=%s short=%s",
            prev_close, prev_sma, self._btc_gate_long, self._btc_gate_short,
        )

    def _auto_reset_risk_counters(self) -> None:
        """Auto-reset daily/weekly/monthly PnL at date boundaries."""
        now = datetime.now()
        today = now.date()
        if today != self._last_reset_date:
            self.daily_pnl = 0.0
            self._last_reset_date = today
            logger.debug("[ATLAS] Daily PnL reset")
        current_week = now.isocalendar()[1]
        if current_week != self._last_reset_week:
            self.weekly_pnl = 0.0
            self._last_reset_week = current_week
            logger.debug("[ATLAS] Weekly PnL reset")
        if now.month != self._last_reset_month:
            self.monthly_pnl = 0.0
            self._last_reset_month = now.month
            logger.debug("[ATLAS] Monthly PnL reset")

    def check_gate(self) -> bool:
        """BaseStrategy interface: gate check (risk limits only).

        Note: BTC directional gate is checked per-signal in check_entry_conditions(),
        not here, because this gate applies to all symbols uniformly while BTC gate
        is direction-specific (long vs short).
        """
        risk_ok, _ = self.check_risk_limits()
        return risk_ok

    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """BaseStrategy interface: generate signals for multiple symbols."""
        self._generation += 1  # new bar cycle
        # Update BTC gate once per cycle (fetches BTC daily data)
        self.update_btc_gate()
        self._last_gen["_btc_gate_gen"] = self._generation  # mark as done for this gen
        results: List[TradeSignal] = []
        for symbol in symbols:
            try:
                # ATLAS requires DataFrame - if no market data, return HOLD
                if self._market_data:
                    ohlcv_list = self._market_data.get_ohlcv(
                        symbol, interval="4h", limit=300
                    )
                    if ohlcv_list and len(ohlcv_list) > 0:
                        df = pd.DataFrame(
                            [
                                {
                                    "open": c.open,
                                    "high": c.high,
                                    "low": c.low,
                                    "close": c.close,
                                    "volume": c.volume,
                                }
                                for c in ohlcv_list
                            ]
                        )
                        sig = self.generate_signal(symbol, df)
                        results.append(self._convert_signal(sig))
                        continue
                results.append(
                    TradeSignal(
                        symbol=symbol,
                        signal=BaseSignal.HOLD,
                        price=0,
                        timestamp=datetime.now(),
                        reason="No data",
                    )
                )
            except Exception as exc:
                logger.error("[ATLAS] Error for %s: %s", symbol, exc)
                results.append(
                    TradeSignal(
                        symbol=symbol,
                        signal=BaseSignal.HOLD,
                        price=0,
                        timestamp=datetime.now(),
                        reason=f"Error: {exc}",
                    )
                )
        return results

    def _convert_signal(self, sig: Signal) -> TradeSignal:
        """Convert internal Signal to BaseStrategy TradeSignal."""
        if sig.signal_type == SignalType.LONG:
            base_signal = BaseSignal.BUY
        elif sig.signal_type in (
            SignalType.SHORT,
            SignalType.EXIT_LONG,
            SignalType.EXIT_SHORT,
        ):
            base_signal = BaseSignal.SELL
        else:
            base_signal = BaseSignal.HOLD
        return TradeSignal(
            symbol=sig.symbol,
            signal=base_signal,
            price=sig.price,
            timestamp=sig.timestamp,
            reason=sig.reason,
        )

    def update_position(self, symbol: str, quantity: float) -> None:
        """Sync internal positions when changed externally."""
        super().update_position(symbol, quantity)
        if quantity <= 0 and symbol in self.positions:
            del self.positions[symbol]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        if ta is None:
            raise ImportError("ta library required for ATLAS-Futures: pip install ta")

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
        """Count consecutive squeeze bars (idempotent per generation)."""
        gen_key = f"sq_{symbol}"
        if self._last_gen.get(gen_key) == self._generation:
            return self._squeeze_counters.get(symbol, 0)
        self._last_gen[gen_key] = self._generation

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
        """Entry conditions (includes BTC directional gate)."""
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

        # LONG entry: BTC gate must allow longs (BTC > SMA50)
        if (
            self._btc_gate_long
            and latest["close"] > latest["ema_200"]
            and latest["close"] > latest["bb_upper"]
        ):
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
                    f"(BBWP={latest['bbwp']:.1f}%, ADX={latest['adx']:.1f}, "
                    f"RVOL={latest['rvol']:.2f}, BTC_gate=LONG)"
                ),
                timestamp=now,
                metadata={
                    "bbwp": latest["bbwp"],
                    "adx": latest["adx"],
                    "rvol": latest["rvol"],
                    "squeeze_bars": squeeze_bars,
                    "btc_gate": "LONG",
                },
            )

        # SHORT entry: BTC gate must allow shorts (BTC < SMA50)
        if (
            self._btc_gate_short
            and latest["close"] < latest["ema_200"]
            and latest["close"] < latest["bb_lower"]
        ):
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
                    f"(BBWP={latest['bbwp']:.1f}%, ADX={latest['adx']:.1f}, "
                    f"RVOL={latest['rvol']:.2f}, BTC_gate=SHORT)"
                ),
                timestamp=now,
                metadata={
                    "bbwp": latest["bbwp"],
                    "adx": latest["adx"],
                    "rvol": latest["rvol"],
                    "squeeze_bars": squeeze_bars,
                    "btc_gate": "SHORT",
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
            if latest["low"] <= chandelier_stop:
                # Use stop price if gap-down didn't breach open, else use open
                exit_price = max(chandelier_stop, latest["open"])
                if latest["open"] <= chandelier_stop:
                    exit_price = latest["open"]
                return Signal(
                    signal_type=SignalType.EXIT_LONG,
                    symbol=symbol,
                    price=exit_price,
                    reason=f"Chandelier Exit: low={latest['low']:.2f} <= stop={chandelier_stop:.2f}",
                    timestamp=now,
                )
        else:
            chandelier_stop = position.lowest_price + (
                latest["atr"] * self.config.chandelier_multiplier
            )
            if latest["high"] >= chandelier_stop:
                exit_price = min(chandelier_stop, latest["open"])
                if latest["open"] >= chandelier_stop:
                    exit_price = latest["open"]
                return Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    symbol=symbol,
                    price=exit_price,
                    reason=f"Chandelier Exit: high={latest['high']:.2f} >= stop={chandelier_stop:.2f}",
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

    def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Signal:
        """Generate signal for a symbol."""
        now = datetime.now()

        # Increment generation if called directly (not via generate_signals)
        gen_key = f"direct_{symbol}"
        if self._last_gen.get(gen_key) != self._generation:
            if self._generation == 0:
                self._generation = 1
            self._last_gen[gen_key] = self._generation

        self._auto_reset_risk_counters()
        # Update BTC gate once per generation (generate_signals already calls this)
        btc_gate_key = "_btc_gate_gen"
        if self._last_gen.get(btc_gate_key) != self._generation:
            self.update_btc_gate()
            self._last_gen[btc_gate_key] = self._generation
        self.check_link_concentration()
        self.check_track_switch()

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
            self._update_position(symbol, data.iloc[-1])

            # BTC Gate flip exit: close position if gate no longer allows direction
            pos = self.positions[symbol]
            gate_flipped = (
                (pos.side == "LONG" and not self._btc_gate_long)
                or (pos.side == "SHORT" and not self._btc_gate_short)
            )
            if gate_flipped:
                exit_type = (
                    SignalType.EXIT_LONG
                    if pos.side == "LONG"
                    else SignalType.EXIT_SHORT
                )
                gate_signal = Signal(
                    signal_type=exit_type,
                    symbol=symbol,
                    price=data.iloc[-1]["close"],
                    reason=f"BTC Gate flip: {pos.side} gate closed",
                    timestamp=now,
                )
                self.last_signal = gate_signal
                return gate_signal

            exit_signal = self.check_exit_conditions(symbol, data)
            if exit_signal:
                self.last_signal = exit_signal
                return exit_signal

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
        """Update position state (idempotent per generation cycle)."""
        if symbol not in self.positions:
            return

        # Per-generation dedup: only increment bars_held once per cycle
        gen_key = f"pos_{symbol}"
        if self._last_gen.get(gen_key) == self._generation:
            # Already processed this generation — only update price tracking
            pos = self.positions[symbol]
            if pos.side == "LONG":
                pos.highest_price = max(pos.highest_price, latest["high"])
            else:
                pos.lowest_price = min(pos.lowest_price, latest["low"])
            pos.pnl_pct = self._calculate_pnl_pct(pos, latest["close"])
            return
        self._last_gen[gen_key] = self._generation

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
        self._cumulative_pnl += pnl_pct

        # Update track PnL for SSOT switch
        if symbol in self.config.major_symbols:
            self.track_b_pnl += pnl_pct
        else:
            self.track_c_pnl += pnl_pct

        # Update drawdown guard (uses cumulative, not daily)
        equity = 100.0 + self._cumulative_pnl
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            self.current_drawdown = (
                (equity - self.peak_equity) / self.peak_equity
            ) * 100

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
            "btc_gate": {
                "long_allowed": self._btc_gate_long,
                "short_allowed": self._btc_gate_short,
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
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self._cumulative_pnl = 0.0
        self.peak_equity = 100.0
        self.current_drawdown = 0.0
        self.track_b_pnl = 0.0
        self.track_c_pnl = 0.0
        self.link_blocked = False
        self.last_signal = None
        self._squeeze_counters.clear()
        logger.info("[ATLAS] Strategy reset")
