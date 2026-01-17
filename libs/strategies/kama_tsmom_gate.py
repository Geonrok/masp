"""
KAMA-TSMOM-Gate Strategy v1.0.

Entry:
    Gate pass (BTC > MA30) and (KAMA(5) or TSMOM(90)).
Exit:
    Gate fail or (KAMA fail and TSMOM fail).
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional

from libs.adapters.factory import AdapterFactory
from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import MA, KAMA, TSMOM_signal

logger = logging.getLogger(__name__)


class KamaTsmomGateStrategy(BaseStrategy):
    """
    KAMA-TSMOM-Gate strategy.

    Parameters:
        gate_ma_period: Gate MA period (default 30).
        kama_period: KAMA period (default 5).
        tsmom_lookback: TSMOM lookback (default 90).
        position_size_krw: Position size per symbol (default 10,000).
    """

    DEFAULT_GATE_MA_PERIOD = 30
    DEFAULT_KAMA_PERIOD = 5
    DEFAULT_TSMOM_LOOKBACK = 90
    DEFAULT_POSITION_SIZE = 10000

    def __init__(
        self,
        gate_ma_period: int = None,
        kama_period: int = None,
        tsmom_lookback: int = None,
        position_size_krw: float = None,
        market_data_adapter=None,
    ):
        super().__init__(name="KAMA-TSMOM-Gate")

        self.gate_ma_period = gate_ma_period or self.DEFAULT_GATE_MA_PERIOD
        self.kama_period = kama_period or self.DEFAULT_KAMA_PERIOD
        self.tsmom_lookback = tsmom_lookback or self.DEFAULT_TSMOM_LOOKBACK
        self.position_size_krw = position_size_krw or self.DEFAULT_POSITION_SIZE

        self._market_data = market_data_adapter
        if self._market_data is None:
            try:
                self._market_data = AdapterFactory.create_market_data("upbit_spot")
            except Exception as exc:
                logger.warning(f"[Strategy] Market data adapter not available: {exc}")

        self._price_cache: Dict[str, List[float]] = {}
        self._btc_prices: List[float] = []
        self._gate_status: bool = False

        logger.info(f"[Strategy] {self.name} initialized")
        logger.info(
            "  Gate MA: %s, KAMA: %s, TSMOM: %s",
            self.gate_ma_period,
            self.kama_period,
            self.tsmom_lookback,
        )

    def set_market_data(self, adapter) -> None:
        """Set market data adapter."""
        self._market_data = adapter

    def update_prices(self, symbol: str, prices: List[float]) -> None:
        """Update cached price data."""
        self._price_cache[symbol] = prices
        if symbol == "BTC/KRW":
            self._btc_prices = prices

    def check_gate(self) -> bool:
        """
        Gate check: BTC > MA(30).
        """
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
            logger.warning(
                "[Strategy] Insufficient BTC data: %s < %s",
                len(self._btc_prices),
                self.gate_ma_period,
            )
            self._gate_status = False
            return False

        btc_close = self._btc_prices[-1]
        btc_ma = MA(self._btc_prices, self.gate_ma_period)

        self._gate_status = btc_close > btc_ma

        logger.debug(
            "[Strategy] Gate: BTC %s %s MA(%s) %s",
            f"{btc_close:,.0f}",
            ">" if self._gate_status else "<=",
            self.gate_ma_period,
            f"{btc_ma:,.0f}",
        )

        return self._gate_status

    def _check_kama_signal(self, symbol: str) -> bool:
        """KAMA(5) signal: Close > KAMA(5)."""
        prices = self._get_prices(symbol, self.kama_period + 10)
        if len(prices) < self.kama_period + 1:
            return False
        close = prices[-1]
        kama = KAMA(prices, self.kama_period)
        return close > kama

    def _check_tsmom_signal(self, symbol: str) -> bool:
        """TSMOM(90) signal: Close > Close[-90]."""
        prices = self._get_prices(symbol, self.tsmom_lookback + 10)
        if len(prices) <= self.tsmom_lookback:
            logger.debug(
                "[Strategy] TSMOM insufficient data for %s: %d <= %d",
                symbol,
                len(prices),
                self.tsmom_lookback,
            )
            return False
        return TSMOM_signal(prices, self.tsmom_lookback)

    def _get_prices(self, symbol: str, min_length: int) -> List[float]:
        """Get price data from cache or adapter."""
        if symbol in self._price_cache and len(self._price_cache[symbol]) >= min_length:
            return self._price_cache[symbol]

        if self._market_data:
            try:
                ohlcv = self._market_data.get_ohlcv(
                    symbol,
                    interval="1d",
                    limit=min_length + 10,
                )
                if ohlcv:
                    prices = [candle.close for candle in ohlcv]
                    self._price_cache[symbol] = prices
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

    def generate_signal(
        self,
        symbol: str,
        gate_pass: Optional[bool] = None,
    ) -> TradeSignal:
        """
        Generate signal for a single symbol.

        Args:
            symbol: Trading symbol.
            gate_pass: Pre-computed gate status (None = compute internally).
        """
        price = self._get_current_price(symbol)
        if price is None:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=0,
                timestamp=datetime.now(),
                reason="Price unavailable",
            )

        if gate_pass is None:
            gate_pass = self.check_gate()
        kama_signal = self._check_kama_signal(symbol)
        tsmom_signal = self._check_tsmom_signal(symbol)

        has_pos = self.has_position(symbol)

        if gate_pass and (kama_signal or tsmom_signal):
            if not has_pos:
                reasons = []
                if kama_signal:
                    reasons.append("KAMA")
                if tsmom_signal:
                    reasons.append("TSMOM")

                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.BUY,
                    price=price,
                    timestamp=datetime.now(),
                    reason=f"Gate PASS + {' + '.join(reasons)}",
                )

        if not gate_pass or (not kama_signal and not tsmom_signal):
            if has_pos:
                if not gate_pass:
                    reason = "Gate FAIL"
                else:
                    reason = "KAMA FAIL + TSMOM FAIL"

                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=price,
                    timestamp=datetime.now(),
                    reason=reason,
                )

        return TradeSignal(
            symbol=symbol,
            signal=Signal.HOLD,
            price=price,
            timestamp=datetime.now(),
            reason="No action",
        )

    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """Generate signals for multiple symbols."""
        signals: List[TradeSignal] = []

        gate_pass = self.check_gate()
        logger.info(f"[Strategy] Gate Status: {'PASS' if gate_pass else 'FAIL'}")

        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol, gate_pass=gate_pass)
                signals.append(signal)

                if signal.signal != Signal.HOLD:
                    logger.info(
                        "[Strategy] %s: %s - %s",
                        symbol,
                        signal.signal.value,
                        signal.reason,
                    )

            except Exception as exc:
                logger.error(f"[Strategy] Error for {symbol}: {exc}")

        return signals

    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "gate_ma_period": self.gate_ma_period,
            "kama_period": self.kama_period,
            "tsmom_lookback": self.tsmom_lookback,
            "position_size_krw": self.position_size_krw,
        }
