"""
Ankle Buy v2.0 WebSocket Real-Time Monitor

Responsibilities:
1. Daily candle close -> immediate entry signal processing
2. BTC Gate real-time -> instant full liquidation when Gate OFF
3. Position exit monitoring -> stop loss, TP-A, SMA exit

Architecture:
- Uses strategy._pos_info as SSOT for position state
- Runs blocking exchange operations in thread executor
- Hourly cron provides backup coverage if WS disconnects
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Dict, Optional, Set

if TYPE_CHECKING:
    from libs.adapters.trade_logger import TradeLogger
    from libs.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

GATE_HYSTERESIS_PCT = 0.5  # SMA50 + 0.5% to restore Gate ON


class AnkleBuyWSMonitor:
    """
    WebSocket-based real-time monitor for Ankle Buy v2.0 strategy.

    Entry: Daily candle close -> BUY signals (liquidity-sorted, balance-limited)
    Exit: Real-time WebSocket ticks -> stop loss / TP-A / SMA exit / Gate OFF
    """

    def __init__(
        self,
        exchange_name: str,
        strategy,
        execution,
        market_data,
        symbols: list,
        trade_logger: Optional["TradeLogger"] = None,
        notifier: Optional["TelegramNotifier"] = None,
        min_position: float = 50000,
        quote_currency: str = "KRW",
    ):
        self.exchange_name = exchange_name
        self.strategy = strategy
        self.execution = execution
        self.market_data = market_data
        self.symbols = symbols
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.min_position = min_position
        self.quote_currency = quote_currency

        # BTC Gate state
        self.btc_sma50: Optional[float] = None
        self.gate_status: bool = True

        # SMA exit cache (refreshed at daily close)
        self._sma_cache: Dict[str, dict] = {}

        # Double execution prevention
        self._executing: Set[str] = set()
        self._exec_lock = threading.Lock()

        # WebSocket
        self.ws = None

        # BTC symbol
        if quote_currency == "USDT":
            self._btc_strategy = "BTC/USDT"
        else:
            self._btc_strategy = "BTC/KRW"

    # ------------------------------------------------------------------
    # Symbol conversion
    # ------------------------------------------------------------------

    def _strategy_to_ws(self, symbol: str) -> str:
        """Convert strategy symbol (BTC/KRW) to WS format."""
        base, quote = symbol.split("/")
        if self.exchange_name in ("upbit", "upbit_spot"):
            return f"{quote}-{base}"
        elif self.exchange_name == "binance_spot":
            return f"{base}{quote}".lower()
        elif self.exchange_name in ("bithumb", "bithumb_spot"):
            return f"{base}_{quote}"
        return symbol

    def _ws_to_strategy(self, ws_symbol: str) -> Optional[str]:
        """Convert WS format to strategy symbol (BTC/KRW)."""
        try:
            if self.exchange_name in ("upbit", "upbit_spot"):
                parts = ws_symbol.split("-")
                if len(parts) == 2:
                    return f"{parts[1]}/{parts[0]}"
            elif self.exchange_name == "binance_spot":
                upper = ws_symbol.upper()
                if upper.endswith("USDT"):
                    return f"{upper[:-4]}/USDT"
            elif self.exchange_name in ("bithumb", "bithumb_spot"):
                parts = ws_symbol.split("_")
                if len(parts) == 2:
                    return f"{parts[0]}/{parts[1]}"
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Tick parsing
    # ------------------------------------------------------------------

    def _parse_tick(self, message: dict) -> tuple:
        """Parse exchange-specific tick -> (ws_symbol, price). (None, 0) on failure."""
        try:
            if self.exchange_name in ("upbit", "upbit_spot"):
                code = message.get("code")
                price = message.get("trade_price")
                if code and price:
                    return str(code), float(price)

            elif self.exchange_name == "binance_spot":
                evt = message.get("e")
                if evt == "24hrTicker":
                    return message.get("s", "").lower(), float(message.get("c", 0))
                if evt == "kline":
                    k = message.get("k", {})
                    return k.get("s", "").lower(), float(k.get("c", 0))

            elif self.exchange_name in ("bithumb", "bithumb_spot"):
                content = message.get("content", {})
                sym = content.get("symbol")
                price = content.get("closePrice")
                if sym and price:
                    return str(sym), float(price)
        except (ValueError, TypeError):
            pass
        return None, 0

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start WebSocket connection and subscriptions."""
        if self.exchange_name in ("upbit", "upbit_spot"):
            from libs.realtime.websocket_client import UpbitWebSocket

            self.ws = UpbitWebSocket(on_message=self._on_tick)
        elif self.exchange_name == "binance_spot":
            from libs.realtime.websocket_client import BinanceWebSocket

            self.ws = BinanceWebSocket(on_message=self._on_tick)
        elif self.exchange_name in ("bithumb", "bithumb_spot"):
            from libs.realtime.bithumb_websocket import BithumbWebSocket

            self.ws = BithumbWebSocket(on_message=self._on_tick)
        else:
            logger.error("[WS-Monitor] Unsupported exchange: %s", self.exchange_name)
            return

        connected = await self.ws.connect()
        if not connected:
            logger.error("[WS-Monitor] Failed to connect to %s", self.exchange_name)
            return

        logger.info("[WS-Monitor] Connected to %s", self.exchange_name)

        # Initialize BTC SMA50
        self._refresh_btc_sma50()

        # Subscribe to BTC
        await self._subscribe_btc()

        # Subscribe to held positions
        await self._subscribe_positions()

        # Binance: also subscribe to kline_1d for daily close detection
        if self.exchange_name == "binance_spot":
            btc_ws = self._strategy_to_ws(self._btc_strategy)
            await self.ws.subscribe_kline(btc_ws, "1d")

        # Start receive loop
        asyncio.create_task(self.ws.run())
        logger.info("[WS-Monitor] Receive loop started for %s", self.exchange_name)

    async def stop(self):
        """Stop WebSocket monitor."""
        if self.ws:
            await self.ws.disconnect()
        logger.info("[WS-Monitor] Stopped for %s", self.exchange_name)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def _subscribe_btc(self):
        btc_ws = self._strategy_to_ws(self._btc_strategy)
        if self.exchange_name in ("upbit", "upbit_spot"):
            await self.ws.subscribe_ticker([btc_ws])
        elif self.exchange_name == "binance_spot":
            await self.ws.subscribe_ticker(btc_ws)
        elif self.exchange_name in ("bithumb", "bithumb_spot"):
            await self.ws.subscribe_ticker([btc_ws])

    async def _subscribe_positions(self):
        held = [s for s in self.strategy._pos_info.keys() if s != self._btc_strategy]
        if not held:
            return

        ws_symbols = [self._strategy_to_ws(s) for s in held]
        if self.exchange_name in ("upbit", "upbit_spot"):
            await self.ws.subscribe_ticker(ws_symbols)
        elif self.exchange_name == "binance_spot":
            for s in ws_symbols:
                await self.ws.subscribe_ticker(s)
        elif self.exchange_name in ("bithumb", "bithumb_spot"):
            await self.ws.subscribe_ticker(ws_symbols)

        logger.info("[WS-Monitor] Subscribed to %d positions", len(held))

    async def update_subscriptions(self):
        """Update subscriptions after position changes."""
        await self._subscribe_positions()

    # ------------------------------------------------------------------
    # Tick handling
    # ------------------------------------------------------------------

    def _on_tick(self, message: dict):
        """WebSocket tick callback - called per tick from receive loop."""
        ws_symbol, price = self._parse_tick(message)
        if ws_symbol is None or price <= 0:
            return

        strategy_symbol = self._ws_to_strategy(ws_symbol)
        if strategy_symbol is None:
            return

        # Binance kline daily close detection
        if (
            self.exchange_name == "binance_spot"
            and message.get("e") == "kline"
            and message.get("k", {}).get("x") is True
        ):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_daily_close())
            except RuntimeError:
                pass

        # BTC tick -> gate check
        if strategy_symbol == self._btc_strategy:
            self._check_gate_realtime(price)
            return

        # Position tick -> exit check
        info = self.strategy.get_position_info(strategy_symbol)
        if info:
            self._check_exit_realtime(strategy_symbol, price, info)

    # ------------------------------------------------------------------
    # BTC Gate real-time (with hysteresis)
    # ------------------------------------------------------------------

    def _check_gate_realtime(self, btc_price: float):
        if self.btc_sma50 is None:
            return

        if self.gate_status:
            # ON -> OFF: price < SMA50
            if btc_price < self.btc_sma50:
                self.gate_status = False
                logger.warning(
                    "[WS-Monitor] BTC Gate OFF! %.2f < SMA50 %.2f",
                    btc_price,
                    self.btc_sma50,
                )
                self._schedule_liquidate_all("BTC Gate OFF (realtime)")
        else:
            # OFF -> ON: price > SMA50 * (1 + hysteresis)
            threshold = self.btc_sma50 * (1 + GATE_HYSTERESIS_PCT / 100)
            if btc_price > threshold:
                self.gate_status = True
                logger.info(
                    "[WS-Monitor] BTC Gate ON recovered: %.2f > %.2f",
                    btc_price,
                    threshold,
                )

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _check_exit_realtime(self, symbol: str, price: float, info: dict):
        """Check exit conditions for held position."""
        # 1. Stop loss
        stop = info.get("stop_loss", 0)
        if stop > 0 and price <= stop:
            self._schedule_sell(symbol, 1.0, f"Stop loss: {price:.4f} <= {stop:.4f}")
            return

        # 2. TP-A (calls strategy method - mutates tp_sold state)
        tp_fraction, tp_reason = self.strategy._check_tp_levels(symbol, price)
        if tp_fraction > 0:
            self._schedule_sell(symbol, tp_fraction, tp_reason)
            return

        # 3. SMA exit (cached values from daily close)
        cached = self._sma_cache.get(symbol)
        if cached:
            upper_sma = cached["upper_sma"]
            today_open = cached["today_open"]
            if price < today_open and price <= upper_sma:
                self._schedule_sell(symbol, 1.0, "SMA exit (realtime)")

    # ------------------------------------------------------------------
    # Order execution (via thread executor)
    # ------------------------------------------------------------------

    def _schedule_sell(self, symbol: str, fraction: float, reason: str):
        """Schedule sell in thread executor (non-blocking)."""
        with self._exec_lock:
            if symbol in self._executing:
                return
            self._executing.add(symbol)

        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(
                None, self._execute_sell_sync, symbol, fraction, reason
            )
            fut.add_done_callback(lambda f: self._on_exec_done(symbol, f))
        except RuntimeError:
            with self._exec_lock:
                self._executing.discard(symbol)

    def _execute_sell_sync(self, symbol: str, fraction: float, reason: str):
        """Execute sell order (blocking, runs in executor thread)."""
        try:
            base_asset = symbol.split("/")[0]
            balance = float(self.execution.get_balance(base_asset) or 0)
            if balance <= 0:
                if fraction >= 0.999:
                    self.strategy._cleanup_position(symbol)
                return

            sell_units = balance * fraction
            min_order = 5 if self.quote_currency == "USDT" else 5000

            # Check minimum sell value
            quote = self.market_data.get_quote(symbol)
            price = 0.0
            if quote:
                price = float(getattr(quote, "last", 0) or 0)
                sell_value = sell_units * price
                if sell_value < min_order:
                    logger.info(
                        "[WS-Monitor] Sell too small: %s %.2f < %.2f",
                        symbol,
                        sell_value,
                        min_order,
                    )
                    if fraction >= 0.999:
                        self.strategy._cleanup_position(symbol)
                    return

            order = self.execution.place_order(
                symbol, "SELL", order_type="MARKET", units=sell_units
            )

            if fraction >= 0.999:
                self.strategy._cleanup_position(symbol)

            logger.info(
                "[WS-Monitor] SELL %s %.0f%%: %s (order=%s)",
                symbol,
                fraction * 100,
                reason,
                getattr(order, "order_id", "N/A"),
            )
            self._send_notification(symbol, f"SELL ({fraction:.0%})", reason)

        except Exception as e:
            logger.error("[WS-Monitor] Sell failed for %s: %s", symbol, e)

    def _on_exec_done(self, symbol: str, future):
        """Callback when executor sell completes."""
        with self._exec_lock:
            self._executing.discard(symbol)
        exc = future.exception()
        if exc:
            logger.error("[WS-Monitor] Execution error for %s: %s", symbol, exc)

    def _schedule_liquidate_all(self, reason: str):
        """Close all positions immediately."""
        held = list(self.strategy._pos_info.keys())
        for symbol in held:
            self._schedule_sell(symbol, 1.0, reason)

    def is_executing(self, symbol: str) -> bool:
        """Check if a symbol is currently being executed (for cron dedup)."""
        with self._exec_lock:
            return symbol in self._executing

    # ------------------------------------------------------------------
    # Daily close entry
    # ------------------------------------------------------------------

    async def _on_daily_close(self):
        """Handle daily candle close - entry logic."""
        logger.info("[WS-Monitor] Daily close triggered for %s", self.exchange_name)

        # Run all blocking I/O (OHLCV fetch, signal gen, orders) in executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._daily_close_sync)

        # Update subscriptions for new positions (async WS call)
        await self.update_subscriptions()

    def _daily_close_sync(self):
        """Daily close logic (blocking, runs in executor)."""
        # 1. Clear caches and refresh
        self.strategy.clear_cache()
        self._refresh_btc_sma50()
        self._refresh_sma_cache()

        # 2. Check gate (t-1, for entry)
        gate = self.strategy.check_gate()
        if not gate:
            logger.info("[WS-Monitor] BTC Gate OFF (t-1), no entries")
            return

        # 3. Collect BUY signals
        from libs.strategies.base import Signal

        buy_signals = []
        for symbol in self.symbols:
            if symbol == self._btc_strategy:
                continue
            try:
                signal = self.strategy.generate_signal(
                    symbol,
                    gate_pass=gate,
                    gate_pass_realtime=self.gate_status,
                )
                if signal and signal.signal == Signal.BUY:
                    volume = 0
                    try:
                        quote = self.market_data.get_quote(symbol)
                        if quote:
                            volume = getattr(quote, "volume_24h", 0) or 0
                    except Exception:
                        pass
                    buy_signals.append((symbol, signal, volume))
            except Exception as e:
                logger.debug("[WS-Monitor] Signal error for %s: %s", symbol, e)

        if not buy_signals:
            logger.info("[WS-Monitor] No BUY signals")
            return

        # 4. Sort by volume descending (liquidity priority)
        buy_signals.sort(key=lambda x: x[2], reverse=True)

        # 5. Execute buys
        self._execute_buys_sync(buy_signals)

    def _execute_buys_sync(self, buy_signals: list):
        """Execute buy orders (blocking, runs in executor)."""
        try:
            raw_balance = self.execution.get_balance(self.quote_currency)
            balance = float(raw_balance) if raw_balance else 0.0
        except Exception as e:
            logger.error("[WS-Monitor] Balance check failed: %s", e)
            return

        per_symbol = self.min_position
        if self.symbols:
            per_symbol = max(balance / len(self.symbols), self.min_position)

        bought = 0
        for symbol, signal, volume in buy_signals:
            if balance < self.min_position:
                logger.info(
                    "[WS-Monitor] Balance exhausted: %.2f < %.2f",
                    balance,
                    self.min_position,
                )
                break

            amount = min(per_symbol, balance)
            if amount < self.min_position:
                break

            try:
                order = self.execution.place_order(
                    symbol, "BUY", order_type="MARKET", amount_krw=amount
                )
                balance -= amount
                bought += 1

                # Sync position so WS monitor can track it immediately
                # (generate_signal already set _pos_info, but original_qty=0)
                self._sync_position_after_buy(symbol)

                logger.info(
                    "[WS-Monitor] BUY %s: %.2f %s (order=%s)",
                    symbol,
                    amount,
                    self.quote_currency,
                    getattr(order, "order_id", "N/A"),
                )
                self._send_notification(
                    symbol, "BUY", f"Daily entry: {amount:.0f} {self.quote_currency}"
                )
            except Exception as e:
                logger.error("[WS-Monitor] BUY failed for %s: %s", symbol, e)

        logger.info(
            "[WS-Monitor] Daily entry: %d buys, remaining %.2f %s",
            bought,
            balance,
            self.quote_currency,
        )

    def _sync_position_after_buy(self, symbol: str):
        """Sync position from exchange after BUY so WS monitor can track it."""
        try:
            base_asset = symbol.split("/")[0]
            qty = float(self.execution.get_balance(base_asset) or 0)
            if qty > 0 and hasattr(self.strategy, "update_position"):
                self.strategy.update_position(symbol, qty)
                logger.info(
                    "[WS-Monitor] Position synced after BUY: %s qty=%.6f", symbol, qty
                )
        except Exception as e:
            logger.warning("[WS-Monitor] Position sync failed for %s: %s", symbol, e)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _refresh_btc_sma50(self):
        val = self.strategy.get_btc_sma50_today()
        if val:
            self.btc_sma50 = val
            logger.info("[WS-Monitor] BTC SMA50 refreshed: %.2f", val)

    def _refresh_sma_cache(self):
        """Refresh SMA exit values for all held positions."""
        for symbol in list(self.strategy._pos_info.keys()):
            try:
                data = self.strategy._get_ohlcv(symbol)
                if data is None:
                    continue
                upper_sma = self.strategy._compute_upper_sma(data)
                today_open = float(data["open"][-1])
                self._sma_cache[symbol] = {
                    "upper_sma": upper_sma,
                    "today_open": today_open,
                }
            except Exception as e:
                logger.debug("[WS-Monitor] SMA cache failed for %s: %s", symbol, e)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _send_notification(self, symbol: str, action: str, reason: str):
        if self.notifier and getattr(self.notifier, "enabled", False):
            try:
                from libs.notifications.telegram import format_trade_message

                msg = format_trade_message(
                    self.exchange_name, symbol, action, 0, 0, reason
                )
                self.notifier.send_message_sync(msg)
            except Exception:
                pass
