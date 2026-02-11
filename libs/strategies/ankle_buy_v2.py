"""
Ankle Buy v2.0 Strategy - Crypto Spot

SMA breakout strategy with BTC gate and hourly TP/stop management.

Entry (checked once daily, executed next bar):
    - Yesterday's candle is green (close > open)
    - Yesterday's open < SMA(25 or 33, on open prices) < yesterday's close
    - BTC Gate ON (BTC close[t-1] > BTC SMA50[t-1])

Exit (checked hourly, priority order):
    1. BTC Gate OFF → full liquidation
    2. Stop loss: price <= entry - ATR(14) * 2
    3. TP: +10% increments, sell 10% of original qty per level (multi-level)
    4. SMA exit: price < today's open AND price <= upper SMA

Performance (OOS validated, Codex PASS 2026-02-09):
    - Binance: Sharpe 0.98, CAGR 16.7%, MDD -21.2%
    - Upbit: Sharpe 1.45, CAGR 19.9%, MDD -10.2%
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import ATR, MA

logger = logging.getLogger(__name__)

STATE_DIR = Path("state")
TP_INCREMENT = 0.10  # +10% per level
TP_SELL_FRACTION = 0.10  # sell 10% of original qty per level
SMA_SHORT = 25
SMA_LONG = 33
BTC_GATE_SMA = 50
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0
MIN_BARS = 60  # minimum daily bars needed


class AnkleBuyV2Strategy(BaseStrategy):
    """Ankle Buy v2.0 - SMA breakout with BTC gate for crypto spot."""

    strategy_id: str = "ankle_buy_v2"
    name: str = "Ankle Buy v2.0"
    version: str = "2.0.0"
    description: str = (
        "SMA(25/33) breakout + green candle + BTC SMA(50) gate. "
        "ATR*2 stop, +10% multi-level TP, SMA exit. "
        "OOS validated: Binance Sharpe 0.98, Upbit 1.45."
    )

    def __init__(
        self,
        market_data_adapter=None,
        btc_symbol: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(name=name, config=config)
        self._market_data = market_data_adapter
        self._btc_symbol = btc_symbol  # Auto-detected if None
        self._ohlcv_cache: Dict[str, dict] = {}
        self._btc_cache: Optional[dict] = None
        self._gate_status: Optional[bool] = None

        # Position info: {symbol: {entry_price, original_qty, stop_loss, tp_sold: set(), upper_sma}}
        self._pos_info: Dict[str, dict] = {}

        # State path: per-exchange isolation (set by set_exchange_name, default to "default")
        self._exchange_name = "default"
        self._state_path = STATE_DIR / "ankle_buy_v2_default.json"
        self._load_state()

        logger.info(
            "[AnkleBuyV2] Initialized v%s | SMA(%d,%d) BTC Gate SMA(%d)",
            self.version,
            SMA_SHORT,
            SMA_LONG,
            BTC_GATE_SMA,
        )

    def set_market_data(self, adapter) -> None:
        """Set market data adapter for OHLCV fetching."""
        self._market_data = adapter

    def set_exchange_name(self, exchange_name: str) -> None:
        """Set exchange name for state file isolation. Called by StrategyRunner."""
        self._exchange_name = exchange_name
        suffix = "" if self.config.get("btc_gate_enabled", True) else "_nogate"
        new_path = STATE_DIR / f"ankle_buy_v2_{exchange_name}{suffix}.json"
        if new_path != self._state_path:
            self._state_path = new_path
            self._pos_info.clear()
            self._load_state()
            logger.info("[AnkleBuyV2] State path set: %s", self._state_path)

    # ------------------------------------------------------------------
    # State persistence (Docker restart resilience)
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load position info from disk."""
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                for sym, info in data.items():
                    info["tp_sold"] = set(info.get("tp_sold", []))
                    self._pos_info[sym] = info
                logger.info(
                    "[AnkleBuyV2] Loaded state: %d positions", len(self._pos_info)
                )
            except Exception as exc:
                logger.warning("[AnkleBuyV2] State load failed: %s", exc)

    def _save_state(self) -> None:
        """Persist position info to disk."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for sym, info in self._pos_info.items():
                data[sym] = {**info, "tp_sold": sorted(info["tp_sold"])}
            self._state_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("[AnkleBuyV2] State save failed: %s", exc)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _detect_btc_symbol(self, sample_symbol: str) -> str:
        """Detect BTC symbol from the format of other symbols."""
        if self._btc_symbol:
            return self._btc_symbol
        if "/KRW" in sample_symbol:
            self._btc_symbol = "BTC/KRW"
        else:
            self._btc_symbol = "BTC/USDT"
        return self._btc_symbol

    def _fetch_ohlcv(self, symbol: str) -> Optional[dict]:
        """Fetch daily OHLCV via market data adapter."""
        if not self._market_data:
            return None
        try:
            ohlcv_list = self._market_data.get_ohlcv(
                symbol, interval="1d", limit=MIN_BARS + 20
            )
            if not ohlcv_list or len(ohlcv_list) < MIN_BARS:
                return None
            return {
                "open": np.array([c.open for c in ohlcv_list], dtype=float),
                "high": np.array([c.high for c in ohlcv_list], dtype=float),
                "low": np.array([c.low for c in ohlcv_list], dtype=float),
                "close": np.array([c.close for c in ohlcv_list], dtype=float),
                "volume": np.array([c.volume for c in ohlcv_list], dtype=float),
            }
        except Exception as exc:
            logger.debug("[AnkleBuyV2] OHLCV fetch failed for %s: %s", symbol, exc)
            return None

    def _get_ohlcv(self, symbol: str) -> Optional[dict]:
        """Get OHLCV with caching (cache valid within one run_once cycle)."""
        if symbol not in self._ohlcv_cache:
            data = self._fetch_ohlcv(symbol)
            if data is not None:
                self._ohlcv_cache[symbol] = data
        return self._ohlcv_cache.get(symbol)

    def _get_btc_data(self) -> Optional[dict]:
        """Get BTC OHLCV (cached)."""
        if self._btc_cache is None:
            btc_sym = self._btc_symbol or "BTC/USDT"
            self._btc_cache = self._fetch_ohlcv(btc_sym)
        return self._btc_cache

    # ------------------------------------------------------------------
    # BTC Gate
    # ------------------------------------------------------------------

    def check_gate(self) -> bool:
        """
        BTC Gate: BTC close[t-1] > BTC SMA(50)[t-1].

        Called once per run_once cycle by the runner.
        Can be disabled via config: {"btc_gate_enabled": false}
        """
        if not self.config.get("btc_gate_enabled", True):
            logger.info("[AnkleBuyV2] BTC Gate disabled by config")
            self._gate_status = True
            return True

        btc = self._get_btc_data()
        if btc is None or len(btc["close"]) < BTC_GATE_SMA + 2:
            logger.warning("[AnkleBuyV2] BTC data insufficient, gate defaults to True")
            self._gate_status = True
            return True

        close = btc["close"]
        # t-1 values (yesterday)
        prev_close = float(close[-2])
        prev_sma50 = float(np.mean(close[-(BTC_GATE_SMA + 1) : -1]))
        self._gate_status = prev_close > prev_sma50

        logger.info(
            "[AnkleBuyV2] BTC Gate: close[t-1]=%.2f, SMA50[t-1]=%.2f → %s",
            prev_close,
            prev_sma50,
            "ON" if self._gate_status else "OFF",
        )
        return self._gate_status

    def check_gate_realtime(self, btc_price: Optional[float] = None) -> bool:
        """
        Real-time BTC Gate: current BTC price > SMA50(today).

        Used by WebSocket monitor for immediate exit decisions.
        Falls back to check_gate() if btc_price is not provided.
        """
        if not self.config.get("btc_gate_enabled", True):
            return True

        if btc_price is None:
            return self.check_gate()

        btc = self._get_btc_data()
        if btc is None or len(btc["close"]) < BTC_GATE_SMA + 1:
            return True

        close = btc["close"]
        sma50_today = float(np.mean(close[-BTC_GATE_SMA:]))
        result = btc_price > sma50_today

        logger.debug(
            "[AnkleBuyV2] BTC Gate(RT): price=%.2f, SMA50=%.2f → %s",
            btc_price,
            sma50_today,
            "ON" if result else "OFF",
        )
        return result

    def get_btc_sma50_today(self) -> Optional[float]:
        """Return today's BTC SMA50 value for WS monitor hysteresis."""
        btc = self._get_btc_data()
        if btc is None or len(btc["close"]) < BTC_GATE_SMA + 1:
            return None
        return float(np.mean(btc["close"][-BTC_GATE_SMA:]))

    def get_position_info(self, symbol: str) -> Optional[dict]:
        """Expose position info for WS monitor (stop/TP/SMA checks)."""
        return self._pos_info.get(symbol)

    # ------------------------------------------------------------------
    # Entry / Exit logic
    # ------------------------------------------------------------------

    def _check_entry_signal(self, data: dict) -> bool:
        """
        Check if yesterday's candle triggered an entry signal.

        Entry conditions (all on t-1 bar):
            1. Green candle: close > open
            2. SMA breakout: open < SMA(25 or 33, on open prices) < close
        """
        opens = data["open"]
        closes = data["close"]
        n = len(opens)
        if n < max(SMA_SHORT, SMA_LONG) + 2:
            return False

        prev_open = float(opens[-2])
        prev_close = float(closes[-2])

        # Green candle check
        if prev_close <= prev_open:
            return False

        # SMA on open prices (t-1)
        sma_short = float(np.mean(opens[-(SMA_SHORT + 1) : -1]))
        sma_long = float(np.mean(opens[-(SMA_LONG + 1) : -1]))

        # Breakout: open < SMA < close
        short_break = prev_open < sma_short < prev_close
        long_break = prev_open < sma_long < prev_close

        return short_break or long_break

    def _compute_stop_loss(self, data: dict, entry_price: float) -> float:
        """Compute stop loss: entry - ATR(14) * 2."""
        atr = ATR(data["high"], data["low"], data["close"], period=ATR_PERIOD)
        if atr > 0:
            return entry_price - ATR_STOP_MULT * atr
        return entry_price * 0.90  # Fallback 10% stop

    def _compute_upper_sma(self, data: dict) -> float:
        """Compute the upper (higher) of SMA(25) and SMA(33) on open prices."""
        opens = data["open"]
        sma_short = float(np.mean(opens[-SMA_SHORT:]))
        sma_long = float(np.mean(opens[-SMA_LONG:]))
        return max(sma_short, sma_long)

    def _check_tp_levels(self, symbol: str, current_price: float) -> tuple[float, str]:
        """
        Check take-profit levels (multi-level).

        Returns (sell_fraction, reason) where sell_fraction is the fraction
        of CURRENT balance to sell. 0.0 means no TP.
        """
        info = self._pos_info.get(symbol)
        if not info:
            return 0.0, ""

        entry = info["entry_price"]
        original_qty = info["original_qty"]
        tp_sold = info["tp_sold"]

        # Find all TP levels that should have been hit
        new_levels = []
        level = 1
        while True:
            tp_price = entry * (1.0 + TP_INCREMENT * level)
            if current_price >= tp_price and level not in tp_sold:
                new_levels.append(level)
            if tp_price > current_price:
                break
            level += 1

        if not new_levels:
            return 0.0, ""

        # Total qty to sell = number of new levels * fraction of original
        total_sell_qty = len(new_levels) * TP_SELL_FRACTION * original_qty

        # Remaining qty from _positions
        remaining = self._positions.get(symbol, 0)
        if remaining <= 0:
            return 0.0, ""

        # Cap sell qty at remaining
        total_sell_qty = min(total_sell_qty, remaining)
        sell_fraction = total_sell_qty / remaining

        # Mark levels as sold
        for lv in new_levels:
            tp_sold.add(lv)
        self._save_state()

        levels_str = ",".join(f"+{lv*10}%" for lv in new_levels)
        reason = f"TP {levels_str} (sell {sell_fraction:.1%} of balance)"
        return min(sell_fraction, 1.0), reason

    def _check_sma_exit(self, data: dict) -> bool:
        """
        SMA exit: current price < today's open AND price <= upper SMA.

        The "upper SMA" is the higher of SMA(25) and SMA(33) on open prices.
        """
        current_close = float(data["close"][-1])
        today_open = float(data["open"][-1])
        upper_sma = self._compute_upper_sma(data)

        return current_close < today_open and current_close <= upper_sma

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        gate_pass: Optional[bool] = None,
        gate_pass_realtime: Optional[bool] = None,
    ) -> TradeSignal:
        """
        Generate signal for a single symbol.

        Called per symbol by StrategyRunner.run_once().
        gate_pass is computed from check_gate() by the runner (t-1, for entry).
        gate_pass_realtime is from WS monitor (current price, for exit).
        """
        now = datetime.now()

        # Auto-detect BTC symbol
        self._detect_btc_symbol(symbol)

        # Skip BTC itself (we don't trade the gate asset)
        if symbol == self._btc_symbol:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=0,
                timestamp=now,
                reason="Gate asset (BTC)",
            )

        data = self._get_ohlcv(symbol)
        if data is None:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=0,
                timestamp=now,
                reason="Data unavailable",
            )

        current_price = float(data["close"][-1])
        has_pos = self.has_position(symbol)

        # --- EXIT LOGIC (if holding position) ---
        if has_pos:
            info = self._pos_info.get(symbol, {})

            # Priority 1: BTC Gate OFF → full liquidation
            # Use realtime gate for exit (more responsive), fallback to t-1 gate
            effective_gate = (
                gate_pass_realtime if gate_pass_realtime is not None else gate_pass
            )
            if effective_gate is False:
                self._cleanup_position(symbol)
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=current_price,
                    timestamp=now,
                    reason="BTC Gate OFF → liquidate",
                    strength=1.0,
                )

            # Priority 2: Stop loss
            stop = info.get("stop_loss", 0)
            if stop > 0 and current_price <= stop:
                self._cleanup_position(symbol)
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=current_price,
                    timestamp=now,
                    reason=f"Stop loss hit: {current_price:.4f} <= {stop:.4f}",
                    strength=1.0,
                )

            # Priority 3: Take profit (partial, multi-level)
            tp_fraction, tp_reason = self._check_tp_levels(symbol, current_price)
            if tp_fraction > 0:
                # Check if TP sold everything
                remaining_after = self._positions.get(symbol, 0) * (1 - tp_fraction)
                if remaining_after < 1e-10:
                    self._cleanup_position(symbol)
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=current_price,
                    timestamp=now,
                    reason=tp_reason,
                    strength=tp_fraction,
                )

            # Priority 4: SMA exit
            if self._check_sma_exit(data):
                self._cleanup_position(symbol)
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=current_price,
                    timestamp=now,
                    reason="SMA exit: price < open AND <= upper SMA",
                    strength=1.0,
                )

            # Hold position
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=current_price,
                timestamp=now,
                reason="Position held",
            )

        # --- ENTRY LOGIC (no position) ---
        # Gate OFF → don't signal BUY (runner also blocks, but be explicit)
        if gate_pass is False:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=current_price,
                timestamp=now,
                reason="BTC Gate OFF",
            )

        # Check entry signal
        if self._check_entry_signal(data):
            # Compute stop loss and upper SMA for position tracking
            stop = self._compute_stop_loss(data, current_price)
            upper_sma = self._compute_upper_sma(data)

            # Get current position qty (will be updated after execution by runner sync)
            # For now, record with placeholder qty; update on next sync
            self._pos_info[symbol] = {
                "entry_price": current_price,
                "original_qty": 0,  # Updated after execution
                "stop_loss": stop,
                "tp_sold": set(),
                "upper_sma": upper_sma,
                "entry_date": now.isoformat(),
            }
            self._save_state()

            return TradeSignal(
                symbol=symbol,
                signal=Signal.BUY,
                price=current_price,
                timestamp=now,
                reason=f"SMA breakout: stop={stop:.4f}, upperSMA={upper_sma:.4f}",
                strength=1.0,
            )

        return TradeSignal(
            symbol=symbol,
            signal=Signal.HOLD,
            price=current_price,
            timestamp=now,
            reason="No signal",
        )

    def _cleanup_position(self, symbol: str) -> None:
        """Remove position tracking info."""
        self._pos_info.pop(symbol, None)
        self._save_state()

    # ------------------------------------------------------------------
    # Position sync hook
    # ------------------------------------------------------------------

    def update_position(self, symbol: str, quantity: float) -> None:
        """
        Called by runner after exchange balance sync.

        Updates internal position tracking and records original_qty
        for new positions (needed for TP calculation).
        """
        super().update_position(symbol, quantity)

        if quantity > 0 and symbol in self._pos_info:
            info = self._pos_info[symbol]
            # Record original_qty on first sync after entry
            if info.get("original_qty", 0) == 0:
                info["original_qty"] = quantity
                self._save_state()
                logger.info(
                    "[AnkleBuyV2] %s position synced: qty=%.6f, entry=%.4f, stop=%.4f",
                    symbol,
                    quantity,
                    info["entry_price"],
                    info["stop_loss"],
                )
        elif quantity <= 0 and symbol in self._pos_info:
            # Position closed externally
            self._cleanup_position(symbol)

    # ------------------------------------------------------------------
    # Cache reset (called at start of each run cycle)
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear OHLCV cache. Call at start of each run cycle."""
        self._ohlcv_cache.clear()
        self._btc_cache = None
        self._gate_status = None

    def generate_signals(
        self,
        symbols: list[str],
        gate_pass_realtime: Optional[bool] = None,
    ) -> list[TradeSignal]:
        """Generate signals for multiple symbols (batch mode)."""
        self.clear_cache()
        gate = self.check_gate()
        return [
            self.generate_signal(
                sym,
                gate_pass=gate,
                gate_pass_realtime=gate_pass_realtime,
            )
            for sym in symbols
        ]
