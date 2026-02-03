"""
Sept_v3_RSI50_Gate Strategy v3.0

7-Signal OR + 거래량 상위 30% 필터

Entry:
    Gate pass (BTC > MA30) AND (7중 OR 신호) AND (거래량 상위 30%)

Exit:
    1. Signal Exit: 7중 OR 신호 불충족
    2. BTC Gate Exit: BTC < MA30 -> Close all positions
    3. Portfolio Stop: -15% from peak -> Close all + 3 day cooldown

Signals (7개 중 1개 이상 = 7중 OR):
    1. KAMA(5): price > KAMA
    2. TSMOM(90): price > price[90]
    3. EMA Cross: EMA12 > EMA26
    4. Momentum(20): price > price[20]
    5. SMA Cross: SMA20 > SMA50
    6. RSI(14) > 50
    7. Higher Low: price > min(price[1:20])

v3 Changes from v2:
    - Signal: 7중 OR 유지 (min_signals=1)
    - Portfolio Stop: -20% → -15%
    - Max Positions: 10개 → 무제한
    - Volume Filter: 상위 30% (핵심 변경)

OOS Performance (v3):
    - Sharpe: 2.27
    - MDD: -37.0%
    - Return: 11,763%
    - Avg Positions: 4.1
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from libs.adapters.factory import AdapterFactory
from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import (
    EMA,
    KAMA,
    MA,
    RSI,
    TSMOM_signal,
)

logger = logging.getLogger(__name__)


class SeptV3Rsi50GateStrategy(BaseStrategy):
    """
    Sept_v3_RSI50_Gate Strategy v3.

    7중 OR 신호 + 거래량 상위 30% 필터 + BTC Gate Exit + Portfolio Stop.

    Parameters:
        gate_ma_period: BTC Gate MA period (default 30)
        kama_period: KAMA period (default 5)
        tsmom_lookback: TSMOM lookback (default 90)
        ema_fast: EMA fast period (default 12)
        ema_slow: EMA slow period (default 26)
        sma_fast: SMA fast period (default 20)
        sma_slow: SMA slow period (default 50)
        momentum_period: Momentum lookback (default 20)
        rsi_period: RSI period (default 14)
        rsi_threshold: RSI threshold (default 50)
        higher_low_period: Higher low lookback (default 20)
        min_signals: Minimum signals required (default 1, 7중 OR)
        volume_filter_pct: Volume filter percentage (default 0.30)
        portfolio_stop: Portfolio stop loss (default -0.15)
        cooldown_days: Days to wait after stop (default 3)
        max_positions: Max positions, 0=unlimited (default 0)
        position_size_krw: Position size per symbol (default 10,000)
    """

    strategy_id = "sept_v3_rsi50_gate"
    name = "Sept-v3-RSI50-Gate"
    version = "3.0.0"
    description = "7중 OR 신호 + 거래량 상위 30% 필터 + BTC Gate Exit + Portfolio Stop"

    # Default parameters
    DEFAULT_GATE_MA_PERIOD = 30
    DEFAULT_KAMA_PERIOD = 5
    DEFAULT_TSMOM_LOOKBACK = 90
    DEFAULT_EMA_FAST = 12
    DEFAULT_EMA_SLOW = 26
    DEFAULT_SMA_FAST = 20
    DEFAULT_SMA_SLOW = 50
    DEFAULT_MOMENTUM_PERIOD = 20
    DEFAULT_RSI_PERIOD = 14
    DEFAULT_RSI_THRESHOLD = 50
    DEFAULT_HIGHER_LOW_PERIOD = 20
    DEFAULT_MIN_SIGNALS = 1  # v3: 7중 OR (1개 이상)
    DEFAULT_VOLUME_FILTER_PCT = 0.30  # v3: 상위 30%
    DEFAULT_PORTFOLIO_STOP = -0.15  # v3: -15%
    DEFAULT_COOLDOWN_DAYS = 3
    DEFAULT_MAX_POSITIONS = 0  # v3: 무제한
    DEFAULT_POSITION_SIZE = 10000

    def __init__(
        self,
        gate_ma_period: int = None,
        kama_period: int = None,
        tsmom_lookback: int = None,
        ema_fast: int = None,
        ema_slow: int = None,
        sma_fast: int = None,
        sma_slow: int = None,
        momentum_period: int = None,
        rsi_period: int = None,
        rsi_threshold: int = None,
        higher_low_period: int = None,
        min_signals: int = None,
        volume_filter_pct: float = None,
        portfolio_stop: float = None,
        cooldown_days: int = None,
        max_positions: int = None,
        position_size_krw: float = None,
        market_data_adapter=None,
    ):
        super().__init__(name=self.name)

        # Signal parameters
        self.gate_ma_period = gate_ma_period or self.DEFAULT_GATE_MA_PERIOD
        self.kama_period = kama_period or self.DEFAULT_KAMA_PERIOD
        self.tsmom_lookback = tsmom_lookback or self.DEFAULT_TSMOM_LOOKBACK
        self.ema_fast = ema_fast or self.DEFAULT_EMA_FAST
        self.ema_slow = ema_slow or self.DEFAULT_EMA_SLOW
        self.sma_fast = sma_fast or self.DEFAULT_SMA_FAST
        self.sma_slow = sma_slow or self.DEFAULT_SMA_SLOW
        self.momentum_period = momentum_period or self.DEFAULT_MOMENTUM_PERIOD
        self.rsi_period = rsi_period or self.DEFAULT_RSI_PERIOD
        self.rsi_threshold = rsi_threshold or self.DEFAULT_RSI_THRESHOLD
        self.higher_low_period = higher_low_period or self.DEFAULT_HIGHER_LOW_PERIOD

        # v3 parameters
        self.min_signals = (
            min_signals if min_signals is not None else self.DEFAULT_MIN_SIGNALS
        )
        self.volume_filter_pct = (
            volume_filter_pct
            if volume_filter_pct is not None
            else self.DEFAULT_VOLUME_FILTER_PCT
        )

        # Risk management parameters
        self.portfolio_stop = (
            portfolio_stop
            if portfolio_stop is not None
            else self.DEFAULT_PORTFOLIO_STOP
        )
        self.cooldown_days = (
            cooldown_days if cooldown_days is not None else self.DEFAULT_COOLDOWN_DAYS
        )
        self.max_positions = (
            max_positions if max_positions is not None else self.DEFAULT_MAX_POSITIONS
        )
        self.position_size_krw = position_size_krw or self.DEFAULT_POSITION_SIZE

        # Market data adapter
        self._market_data = market_data_adapter
        if self._market_data is None:
            try:
                self._market_data = AdapterFactory.create_market_data("upbit_spot")
            except Exception as exc:
                logger.warning(f"[Strategy] Market data adapter not available: {exc}")

        # Internal state
        self._price_cache: Dict[str, List[float]] = {}
        self._volume_cache: Dict[str, float] = {}
        self._btc_prices: List[float] = []
        self._gate_status: bool = False

        # Risk management state
        self._peak_portfolio_value: float = 0.0
        self._cooldown_remaining: int = 0
        self._last_portfolio_value: float = 0.0
        self._in_portfolio_stop: bool = False

        logger.info(f"[Strategy] {self.name} v{self.version} initialized")
        logger.info(
            f"  v3 Changes: min_signals={self.min_signals}, "
            f"volume_filter={self.volume_filter_pct*100:.0f}%, "
            f"portfolio_stop={self.portfolio_stop*100:.0f}%"
        )

    def set_market_data(self, adapter) -> None:
        """Set market data adapter."""
        self._market_data = adapter

    def update_prices(
        self, symbol: str, prices: List[float], volume: float = None
    ) -> None:
        """Update cached price and volume data."""
        self._price_cache[symbol] = prices
        if volume is not None:
            self._volume_cache[symbol] = volume
        if symbol == "BTC/KRW":
            self._btc_prices = prices

    def update_portfolio_value(self, value: float) -> bool:
        """
        Update portfolio value for risk management.
        Returns True if portfolio stop was triggered.
        """
        self._last_portfolio_value = value

        if value > self._peak_portfolio_value:
            self._peak_portfolio_value = value

        if self._peak_portfolio_value > 0:
            drawdown = (value - self._peak_portfolio_value) / self._peak_portfolio_value

            if drawdown <= self.portfolio_stop and not self._in_portfolio_stop:
                logger.warning(
                    f"[Strategy] PORTFOLIO STOP triggered: "
                    f"Drawdown {drawdown*100:.1f}% <= {self.portfolio_stop*100:.0f}%"
                )
                self._in_portfolio_stop = True
                self._cooldown_remaining = self.cooldown_days
                return True

        return False

    def process_daily_update(self) -> None:
        """Process daily update for cooldown tracking."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            logger.info(
                f"[Strategy] Cooldown: {self._cooldown_remaining} days remaining"
            )

            if self._cooldown_remaining == 0:
                logger.info("[Strategy] Cooldown ended, trading resumed")
                self._in_portfolio_stop = False
                self._peak_portfolio_value = self._last_portfolio_value

    def is_in_cooldown(self) -> bool:
        """Check if strategy is in cooldown period."""
        return self._cooldown_remaining > 0

    def reset_risk_state(self, portfolio_value: float) -> None:
        """Reset risk management state."""
        self._peak_portfolio_value = portfolio_value
        self._last_portfolio_value = portfolio_value
        self._cooldown_remaining = 0
        self._in_portfolio_stop = False
        logger.info(f"[Strategy] Risk state reset, peak = {portfolio_value:,.0f}")

    def check_gate(self) -> bool:
        """Gate check: BTC > MA(30)."""
        if len(self._btc_prices) < self.gate_ma_period:
            if self._market_data:
                try:
                    ohlcv = self._market_data.get_ohlcv(
                        "BTC/KRW",
                        interval="1d",
                        limit=self.gate_ma_period + 10,
                    )
                    if ohlcv:
                        self._btc_prices = [candle.close for candle in ohlcv]
                except Exception as exc:
                    logger.error(f"[Strategy] Failed to get BTC prices: {exc}")

        if len(self._btc_prices) < self.gate_ma_period:
            self._gate_status = False
            return False

        btc_close = self._btc_prices[-1]
        btc_ma = MA(self._btc_prices, self.gate_ma_period)
        self._gate_status = btc_close > btc_ma

        return self._gate_status

    def _get_prices(self, symbol: str, min_length: int) -> List[float]:
        """Get price data from cache or adapter."""
        if symbol in self._price_cache and len(self._price_cache[symbol]) >= min_length:
            return self._price_cache[symbol]

        if self._market_data:
            try:
                ohlcv = self._market_data.get_ohlcv(
                    symbol, interval="1d", limit=min_length + 10
                )
                if ohlcv:
                    prices = [candle.close for candle in ohlcv]
                    self._price_cache[symbol] = prices
                    # Update volume
                    if ohlcv:
                        vol = sum(c.close * c.volume for c in ohlcv[-20:]) / min(
                            20, len(ohlcv)
                        )
                        self._volume_cache[symbol] = vol
                    return prices
            except Exception as exc:
                logger.error(f"[Strategy] Failed to get {symbol} prices: {exc}")

        return self._price_cache.get(symbol, [])

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        prices = self._price_cache.get(symbol, [])
        if prices:
            return prices[-1]

        if self._market_data:
            try:
                quote = self._market_data.get_quote(symbol)
                if quote:
                    return quote.last
            except Exception as exc:
                logger.error(f"[Strategy] Failed to get {symbol} quote: {exc}")

        return None

    def _get_volume(self, symbol: str) -> float:
        """Get average volume for symbol."""
        return self._volume_cache.get(symbol, 0.0)

    # ===== Individual Signal Checks =====

    def _check_kama_signal(self, symbol: str) -> bool:
        prices = self._get_prices(symbol, self.kama_period + 10)
        if len(prices) < self.kama_period + 1:
            return False
        return prices[-1] > KAMA(prices, self.kama_period)

    def _check_tsmom_signal(self, symbol: str) -> bool:
        prices = self._get_prices(symbol, self.tsmom_lookback + 10)
        if len(prices) <= self.tsmom_lookback:
            return False
        return TSMOM_signal(prices, self.tsmom_lookback)

    def _check_ema_cross_signal(self, symbol: str) -> bool:
        prices = self._get_prices(symbol, self.ema_slow + 10)
        if len(prices) < self.ema_slow:
            return False
        return EMA(prices, self.ema_fast) > EMA(prices, self.ema_slow)

    def _check_momentum_signal(self, symbol: str) -> bool:
        prices = self._get_prices(symbol, self.momentum_period + 10)
        if len(prices) <= self.momentum_period:
            return False
        return prices[-1] > prices[-(self.momentum_period + 1)]

    def _check_sma_cross_signal(self, symbol: str) -> bool:
        prices = self._get_prices(symbol, self.sma_slow + 10)
        if len(prices) < self.sma_slow:
            return False
        return MA(prices, self.sma_fast) > MA(prices, self.sma_slow)

    def _check_rsi_signal(self, symbol: str) -> bool:
        prices = self._get_prices(symbol, self.rsi_period + 10)
        if len(prices) < self.rsi_period + 1:
            return False
        return RSI(prices, self.rsi_period) > self.rsi_threshold

    def _check_higher_low_signal(self, symbol: str) -> bool:
        prices = self._get_prices(symbol, self.higher_low_period + 10)
        if len(prices) <= self.higher_low_period:
            return False
        return prices[-1] > min(prices[-(self.higher_low_period + 1) : -1])

    def _count_active_signals(self, symbol: str) -> int:
        """Count how many of 7 signals are True."""
        count = 0
        if self._check_kama_signal(symbol):
            count += 1
        if self._check_tsmom_signal(symbol):
            count += 1
        if self._check_ema_cross_signal(symbol):
            count += 1
        if self._check_momentum_signal(symbol):
            count += 1
        if self._check_sma_cross_signal(symbol):
            count += 1
        if self._check_rsi_signal(symbol):
            count += 1
        if self._check_higher_low_signal(symbol):
            count += 1
        return count

    def _check_signal(self, symbol: str) -> bool:
        """v3: Check if minimum signals are met."""
        return self._count_active_signals(symbol) >= self.min_signals

    def generate_signal(
        self,
        symbol: str,
        gate_pass: Optional[bool] = None,
    ) -> TradeSignal:
        """Generate signal for a single symbol."""
        price = self._get_current_price(symbol)
        if price is None:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=0,
                timestamp=datetime.now(),
                reason="Price unavailable",
            )

        # Cooldown check
        if self.is_in_cooldown():
            if self.has_position(symbol):
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=price,
                    timestamp=datetime.now(),
                    reason=f"Cooldown active ({self._cooldown_remaining}d remaining)",
                )
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=price,
                timestamp=datetime.now(),
                reason=f"Cooldown ({self._cooldown_remaining}d remaining)",
            )

        if gate_pass is None:
            gate_pass = self.check_gate()

        # BTC Gate Exit
        if not gate_pass:
            if self.has_position(symbol):
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=price,
                    timestamp=datetime.now(),
                    reason="BTC Gate FAIL - Close all",
                )
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=price,
                timestamp=datetime.now(),
                reason="BTC Gate FAIL - No entry",
            )

        # v3: Check minimum signals
        signal_count = self._count_active_signals(symbol)
        has_signal = signal_count >= self.min_signals
        has_pos = self.has_position(symbol)

        if gate_pass and has_signal:
            if not has_pos:
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.BUY,
                    price=price,
                    timestamp=datetime.now(),
                    reason=f"Gate PASS + {signal_count}/7 signals",
                )

        if not has_signal:
            if has_pos:
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=price,
                    timestamp=datetime.now(),
                    reason=f"Only {signal_count}/7 signals (min {self.min_signals})",
                )

        return TradeSignal(
            symbol=symbol,
            signal=Signal.HOLD,
            price=price,
            timestamp=datetime.now(),
            reason="Holding position" if has_pos else "No action",
        )

    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """Generate signals for multiple symbols with volume filter."""
        signals: List[TradeSignal] = []

        # Cooldown check
        if self.is_in_cooldown():
            logger.info(
                f"[Strategy] In cooldown: {self._cooldown_remaining} days remaining"
            )
            for symbol in symbols:
                if self.has_position(symbol):
                    price = self._get_current_price(symbol)
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.SELL,
                            price=price or 0,
                            timestamp=datetime.now(),
                            reason=f"Portfolio Stop Cooldown ({self._cooldown_remaining}d)",
                        )
                    )
                else:
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.HOLD,
                            price=0,
                            timestamp=datetime.now(),
                            reason=f"Cooldown ({self._cooldown_remaining}d)",
                        )
                    )
            return signals

        # Gate check
        gate_pass = self.check_gate()
        logger.info(f"[Strategy] Gate Status: {'PASS' if gate_pass else 'FAIL'}")

        # BTC Gate Exit
        if not gate_pass:
            logger.warning("[Strategy] BTC Gate FAIL - Closing all positions")
            for symbol in symbols:
                price = self._get_current_price(symbol)
                if self.has_position(symbol):
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.SELL,
                            price=price or 0,
                            timestamp=datetime.now(),
                            reason="BTC Gate FAIL - Close all",
                        )
                    )
                else:
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.HOLD,
                            price=price or 0,
                            timestamp=datetime.now(),
                            reason="BTC Gate FAIL - No entry",
                        )
                    )
            return signals

        # Collect symbols that pass signal condition
        candidates = []
        for symbol in symbols:
            signal_count = self._count_active_signals(symbol)
            if signal_count >= self.min_signals:
                volume = self._get_volume(symbol)
                candidates.append((symbol, signal_count, volume))

        # v3: Apply volume filter - top N%
        if candidates and self.volume_filter_pct < 1.0:
            candidates.sort(key=lambda x: x[2], reverse=True)
            cutoff = max(1, int(len(candidates) * self.volume_filter_pct))
            candidates = candidates[:cutoff]
            logger.info(
                f"[Strategy] Volume filter: {len(candidates)} symbols (top {self.volume_filter_pct*100:.0f}%)"
            )

        target_symbols = set(s for s, _, _ in candidates)

        # Generate signals
        for symbol in symbols:
            price = self._get_current_price(symbol)
            has_pos = self.has_position(symbol)

            if symbol in target_symbols:
                if not has_pos:
                    signal_count = next((c for s, c, _ in candidates if s == symbol), 0)
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.BUY,
                            price=price or 0,
                            timestamp=datetime.now(),
                            reason=f"Gate PASS + {signal_count}/7 + Vol Top",
                        )
                    )
                else:
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.HOLD,
                            price=price or 0,
                            timestamp=datetime.now(),
                            reason="Holding position",
                        )
                    )
            else:
                if has_pos:
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.SELL,
                            price=price or 0,
                            timestamp=datetime.now(),
                            reason="Signal or volume filter failed",
                        )
                    )
                else:
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            signal=Signal.HOLD,
                            price=price or 0,
                            timestamp=datetime.now(),
                            reason="No action",
                        )
                    )

        # v3: Max positions = 0 means unlimited
        if self.max_positions > 0:
            buy_signals = [s for s in signals if s.signal == Signal.BUY]
            current_positions = sum(1 for s in symbols if self.has_position(s))
            available_slots = self.max_positions - current_positions

            if len(buy_signals) > available_slots:
                logger.info(
                    f"[Strategy] Limiting BUY signals: {len(buy_signals)} -> {available_slots}"
                )
                for i, signal in enumerate(signals):
                    if signal.signal == Signal.BUY:
                        if available_slots > 0:
                            available_slots -= 1
                        else:
                            signals[i] = TradeSignal(
                                symbol=signal.symbol,
                                signal=Signal.HOLD,
                                price=signal.price,
                                timestamp=signal.timestamp,
                                reason="Max positions reached",
                            )

        return signals

    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "gate_ma_period": self.gate_ma_period,
            "kama_period": self.kama_period,
            "tsmom_lookback": self.tsmom_lookback,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "momentum_period": self.momentum_period,
            "rsi_period": self.rsi_period,
            "rsi_threshold": self.rsi_threshold,
            "higher_low_period": self.higher_low_period,
            "min_signals": self.min_signals,
            "volume_filter_pct": self.volume_filter_pct,
            "portfolio_stop": self.portfolio_stop,
            "cooldown_days": self.cooldown_days,
            "max_positions": self.max_positions,
            "position_size_krw": self.position_size_krw,
        }

    def get_risk_state(self) -> dict:
        """Return current risk management state."""
        return {
            "peak_portfolio_value": self._peak_portfolio_value,
            "last_portfolio_value": self._last_portfolio_value,
            "cooldown_remaining": self._cooldown_remaining,
            "in_portfolio_stop": self._in_portfolio_stop,
            "current_drawdown": (
                (self._last_portfolio_value - self._peak_portfolio_value)
                / self._peak_portfolio_value
                if self._peak_portfolio_value > 0
                else 0.0
            ),
        }
