"""
Trend-Momentum-Gate (TMG) Strategy

Cross-sectional momentum with BTC trend filter for crypto spot trading.

Core logic:
    1. BTC Gate: Only invest when BTC > SMA(50) (using t-1 data)
    2. Universe: Top 50% by lagged 30-day dollar volume (no look-ahead)
    3. Selection: Top N coins by 30-day momentum (cross-sectional ranking)
    4. Sizing: Inverse-volatility weighted, targeting 20% annual portfolio vol
    5. Rebalance: Weekly

Parameters (5, all with economic rationale):
    - btc_ma_period: BTC trend filter period (50)
    - mom_lookback: Momentum ranking lookback (30 days)
    - top_n: Number of coins to hold (10)
    - vol_target: Annual portfolio volatility target (0.20)
    - rebal_days: Rebalance interval in days (7)

Design principles:
    - Single codebase for backtest AND production (no code divergence)
    - ALL signals use t-1 data (no look-ahead)
    - Exchange-aware BTC gate (BTC/KRW vs BTC/USDT)
    - Cross-sectional ranking ensures <50% signal activation
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import MA

logger = logging.getLogger(__name__)

# Stablecoin and leveraged token patterns
_STABLECOIN_PATTERN = re.compile(
    r"^(USDT|USDC|BUSD|DAI|TUSD|USDP|FDUSD|USDD|PYUSD|UST)/",
    re.IGNORECASE,
)
_LEVERAGED_PATTERN = re.compile(
    r"(UP|DOWN|BULL|BEAR|3L|3S|2L|2S)/",
    re.IGNORECASE,
)


class TrendMomentumGateStrategy(BaseStrategy):
    """
    Cross-sectional momentum strategy with BTC trend gate.

    Uses generate_signal() for StrategyRunner compatibility.
    On first call per cycle, computes rankings for ALL symbols (cross-sectional),
    then returns cached results for subsequent calls.
    """

    strategy_id: str = "trend_momentum_gate"
    name: str = "Trend-Momentum-Gate"
    version: str = "1.0.0"
    description: str = (
        "Cross-sectional momentum with BTC trend filter. "
        "Buys top N coins by 30-day momentum when BTC > SMA(50). "
        "Inverse-volatility weighted, weekly rebalance."
    )

    def __init__(
        self,
        btc_ma_period: int = 50,
        mom_lookback: int = 30,
        top_n: int = 10,
        vol_target: float = 0.20,
        rebal_days: int = 7,
        exchange: Optional[str] = None,
        market_data_adapter=None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(name=name, config=config)

        self.btc_ma_period = btc_ma_period
        self.mom_lookback = mom_lookback
        self.top_n = top_n
        self.vol_target = vol_target
        self.rebal_days = rebal_days

        self._exchange = exchange or ""
        self._market_data = market_data_adapter

        # Minimum bars needed for any calculation
        self._min_bars = max(self.btc_ma_period, self.mom_lookback) + 10

        # Cross-sectional signal cache
        self._signals_cache: Dict[str, TradeSignal] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)

        # All symbols known to the strategy (set by StrategyRunner or backtest)
        self._all_symbols: List[str] = []

        # Drawdown stop state
        self._peak_value: float = 0.0
        self._in_cooldown: bool = False
        self._cooldown_until: Optional[datetime] = None
        self._drawdown_limit: float = -0.25  # -25%
        self._cooldown_days: int = 7

        logger.info("[TMG] Initialized v%s", self.version)
        logger.info(
            "  BTC_MA=%d, MOM=%d, TOP_N=%d, VOL=%.0f%%, REBAL=%dd",
            self.btc_ma_period,
            self.mom_lookback,
            self.top_n,
            self.vol_target * 100,
            self.rebal_days,
        )

    def set_market_data(self, adapter) -> None:
        """Set market data adapter for live trading."""
        self._market_data = adapter

    def set_exchange(self, exchange: str) -> None:
        """Set exchange name for BTC symbol resolution."""
        self._exchange = exchange

    def set_all_symbols(self, symbols: List[str]) -> None:
        """Set the full universe of symbols for cross-sectional ranking."""
        self._all_symbols = list(symbols)

    # ----------------------------------------------------------------
    # BTC Gate (exchange-aware, t-1 data)
    # ----------------------------------------------------------------

    def _get_btc_symbol(self) -> str:
        """Return the correct BTC symbol for the current exchange."""
        if self._exchange in ("upbit", "upbit_spot", "bithumb", "bithumb_spot"):
            return "BTC/KRW"
        return "BTC/USDT"

    def check_gate(self) -> bool:
        """
        BTC trend gate: True if BTC[t-1] > SMA(BTC[t-1], btc_ma_period).

        Uses t-1 close to avoid look-ahead.
        """
        if not self._market_data:
            return True  # fail-open if no data

        btc_symbol = self._get_btc_symbol()
        try:
            limit = self.btc_ma_period + 10
            ohlcv = self._market_data.get_ohlcv(btc_symbol, interval="1d", limit=limit)
            if not ohlcv or len(ohlcv) < self.btc_ma_period + 1:
                logger.warning("[TMG] Insufficient BTC data for gate")
                return True

            closes = [c.close for c in ohlcv]
            # t-1: use all data EXCEPT the last bar (today's incomplete candle)
            btc_prev_close = closes[-2]
            btc_ma = MA(closes[:-1], period=self.btc_ma_period)
            gate = btc_prev_close > btc_ma

            logger.debug(
                "[TMG] Gate: BTC=%.2f, MA(%d)=%.2f, gate=%s",
                btc_prev_close,
                self.btc_ma_period,
                btc_ma,
                gate,
            )
            return gate

        except Exception as exc:
            logger.error("[TMG] Gate check failed: %s", exc)
            return True  # fail-open

    # ----------------------------------------------------------------
    # Universe filtering (no look-ahead)
    # ----------------------------------------------------------------

    @staticmethod
    def is_excluded(symbol: str) -> bool:
        """Check if symbol is a stablecoin or leveraged token."""
        return bool(
            _STABLECOIN_PATTERN.search(symbol) or _LEVERAGED_PATTERN.search(symbol)
        )

    def _filter_universe(self, symbols: List[str]) -> List[str]:
        """
        Filter universe by lagged dollar volume (top 50%).

        Uses t-1 data for volume ranking to avoid look-ahead.
        """
        if not self._market_data:
            return [s for s in symbols if not self.is_excluded(s)]

        vol_scores: Dict[str, float] = {}
        for symbol in symbols:
            if self.is_excluded(symbol):
                continue
            try:
                ohlcv = self._market_data.get_ohlcv(symbol, interval="1d", limit=35)
                if not ohlcv or len(ohlcv) < 32:
                    continue
                # Use t-1 data: last 30 bars EXCLUDING current bar
                bars = ohlcv[:-1][-30:]
                avg_dvol = np.mean([b.close * b.volume for b in bars])
                vol_scores[symbol] = avg_dvol
            except Exception:
                continue

        if not vol_scores:
            return [s for s in symbols if not self.is_excluded(s)]

        # Top 50% by dollar volume
        threshold = np.median(list(vol_scores.values()))
        eligible = [s for s, v in vol_scores.items() if v >= threshold]

        logger.debug(
            "[TMG] Universe: %d/%d eligible (threshold=%.0f)",
            len(eligible),
            len(symbols),
            threshold,
        )
        return eligible

    # ----------------------------------------------------------------
    # Core: Cross-sectional momentum ranking
    # ----------------------------------------------------------------

    def _compute_momentum(self, symbol: str) -> Optional[float]:
        """Compute 30-day momentum for a symbol using t-1 data."""
        if not self._market_data:
            return None
        try:
            limit = self.mom_lookback + 5
            ohlcv = self._market_data.get_ohlcv(symbol, interval="1d", limit=limit)
            if not ohlcv or len(ohlcv) < self.mom_lookback + 2:
                return None

            closes = [c.close for c in ohlcv]
            # t-1 momentum: close[t-1] / close[t-1-lookback] - 1
            curr = closes[-2]
            prev = closes[-2 - self.mom_lookback]
            if prev <= 0:
                return None
            return (curr / prev) - 1.0

        except Exception:
            return None

    def _compute_vol_weight(self, symbol: str) -> float:
        """Compute inverse-volatility weight for position sizing."""
        if not self._market_data:
            return 1.0 / self.top_n

        try:
            ohlcv = self._market_data.get_ohlcv(symbol, interval="1d", limit=35)
            if not ohlcv or len(ohlcv) < 32:
                return 1.0 / self.top_n

            closes = np.array([c.close for c in ohlcv[:-1][-30:]], dtype=float)
            if len(closes) < 2:
                return 1.0 / self.top_n

            daily_returns = np.diff(closes) / closes[:-1]
            ann_vol = float(np.std(daily_returns) * np.sqrt(365))
            if ann_vol < 0.01:
                # Near-zero vol → default equal weight, capped
                return min(1.0 / self.top_n, 0.25)

            # Target vol per position = portfolio vol target / N
            target_per_pos = self.vol_target / self.top_n
            raw_weight = target_per_pos / ann_vol
            # Cap at 25% per coin, floor at 1%
            return max(0.01, min(raw_weight, 0.25))

        except Exception:
            return 1.0 / self.top_n

    def _compute_cross_sectional_signals(
        self, symbols: List[str], gate_pass: bool
    ) -> Dict[str, TradeSignal]:
        """
        Core algorithm: cross-sectional momentum ranking.

        Returns dict of symbol -> TradeSignal for all symbols.
        """
        now = datetime.now()
        result: Dict[str, TradeSignal] = {}

        # Cooldown check
        if self._in_cooldown:
            if self._cooldown_until and now < self._cooldown_until:
                for symbol in symbols:
                    result[symbol] = TradeSignal(
                        symbol=symbol,
                        signal=Signal.SELL if self.has_position(symbol) else Signal.HOLD,
                        price=0,
                        timestamp=now,
                        reason="Drawdown cooldown",
                    )
                return result
            else:
                self._in_cooldown = False
                self._cooldown_until = None
                self._peak_value = 0.0  # Reset peak so drawdown doesn't retrigger
                logger.info("[TMG] Cooldown ended, resuming trading")

        # Gate OFF → sell everything
        if not gate_pass:
            for symbol in symbols:
                result[symbol] = TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL if self.has_position(symbol) else Signal.HOLD,
                    price=0,
                    timestamp=now,
                    reason="BTC gate OFF",
                )
            return result

        # 1. Filter universe (lagged volume, no stablecoins/leveraged)
        eligible = self._filter_universe(symbols)

        # 2. Compute momentum for eligible coins (t-1 data)
        mom_scores: Dict[str, float] = {}
        for symbol in eligible:
            mom = self._compute_momentum(symbol)
            if mom is not None:
                mom_scores[symbol] = mom

        # 3. Rank and select top N
        ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
        top_n_symbols = set(s for s, _ in ranked[: self.top_n])

        logger.info(
            "[TMG] Selected %d/%d coins (top momentum from %d eligible)",
            len(top_n_symbols),
            len(symbols),
            len(mom_scores),
        )
        if ranked[:3]:
            top3 = ", ".join(f"{s}({m:+.1%})" for s, m in ranked[:3])
            logger.info("[TMG] Top 3: %s", top3)

        # 4. Generate signals with vol-weighted sizing
        for symbol in symbols:
            if symbol in top_n_symbols:
                strength = self._compute_vol_weight(symbol)
                mom_val = mom_scores.get(symbol, 0)
                result[symbol] = TradeSignal(
                    symbol=symbol,
                    signal=Signal.BUY,
                    price=0,
                    timestamp=now,
                    reason=f"Momentum rank top-{self.top_n} ({mom_val:+.1%})",
                    strength=strength,
                )
            elif self.has_position(symbol):
                result[symbol] = TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=0,
                    timestamp=now,
                    reason="Dropped from top-N",
                )
            else:
                result[symbol] = TradeSignal(
                    symbol=symbol,
                    signal=Signal.HOLD,
                    price=0,
                    timestamp=now,
                    reason="Not selected",
                )

        return result

    # ----------------------------------------------------------------
    # Drawdown stop
    # ----------------------------------------------------------------

    def update_portfolio_value(self, value: float) -> None:
        """
        Track portfolio value for drawdown stop.

        Must be called by the execution layer after each rebalance.
        """
        if value > self._peak_value:
            self._peak_value = value

        if self._peak_value > 0:
            drawdown = (value - self._peak_value) / self._peak_value
            if drawdown <= self._drawdown_limit and not self._in_cooldown:
                self._in_cooldown = True
                self._cooldown_until = datetime.now() + timedelta(
                    days=self._cooldown_days
                )
                logger.warning(
                    "[TMG] DRAWDOWN STOP: %.1f%% (limit %.1f%%). Cooldown until %s",
                    drawdown * 100,
                    self._drawdown_limit * 100,
                    self._cooldown_until.strftime("%Y-%m-%d"),
                )

    # ----------------------------------------------------------------
    # Public interface (StrategyRunner compatible)
    # ----------------------------------------------------------------

    def generate_signal(
        self, symbol: str, gate_pass: bool = True
    ) -> TradeSignal:
        """
        Generate signal for a single symbol (StrategyRunner interface).

        On first call per cycle, computes cross-sectional rankings for ALL
        known symbols, then returns from cache for subsequent calls.
        """
        now = datetime.now()

        # Check if cache is fresh
        if self._cache_time and (now - self._cache_time) < self._cache_ttl:
            cached = self._signals_cache.get(symbol)
            if cached is not None:
                return cached

        # Cache miss or expired → recompute all signals
        symbols = self._all_symbols if self._all_symbols else [symbol]
        self._signals_cache = self._compute_cross_sectional_signals(
            symbols, gate_pass
        )
        self._cache_time = now

        return self._signals_cache.get(
            symbol,
            TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=0,
                timestamp=now,
                reason="No data",
            ),
        )

    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """
        Generate signals for multiple symbols (backtest interface).

        Directly computes cross-sectional rankings without caching.
        """
        gate = self.check_gate()
        signals_dict = self._compute_cross_sectional_signals(symbols, gate)
        return [
            signals_dict.get(
                s,
                TradeSignal(
                    symbol=s,
                    signal=Signal.HOLD,
                    price=0,
                    timestamp=datetime.now(),
                    reason="No data",
                ),
            )
            for s in symbols
        ]

    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "btc_ma_period": self.btc_ma_period,
            "mom_lookback": self.mom_lookback,
            "top_n": self.top_n,
            "vol_target": self.vol_target,
            "rebal_days": self.rebal_days,
            "exchange": self._exchange,
        }
