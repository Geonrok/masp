"""
Strategy Runner - 전략 신호를 실거래 주문으로 변환
- 전략 신호 수신
- Kill-Switch 체크
- Health Monitor 검증
- 주문 실행
- TradeLogger 기록
"""

import logging
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load .env without overriding system environment variables.
load_dotenv(override=False)

from libs.adapters.factory import AdapterFactory
from libs.adapters.trade_logger import TradeLogger
from libs.analytics.daily_report import DailyReportGenerator
from libs.analytics.strategy_health import StrategyHealthMonitor
from libs.core.config import Config
from libs.notifications.telegram import (
    TelegramNotifier,
    format_daily_summary,
    format_trade_message,
)
from libs.risk.stop_loss_manager import (
    CompositeStopManager,
    FixedPercentageStop,
    TimeBasedStop,
    TrailingStop,
)
from libs.strategies.base import Signal as StrategySignal
from libs.strategies.loader import get_strategy

logger = logging.getLogger(__name__)
MIN_ORDER_KRW = 5000


class StrategyRunner:
    """
    전략 실행기

    Usage:
        runner = StrategyRunner(
            strategy_name="ma_crossover_v1",
            exchange="paper",  # "paper" | "upbit" | "bithumb"
            symbols=["BTC/KRW"],
            position_size_krw=10000
        )
        runner.run_once()  # 1회 실행
        runner.run_loop(interval_seconds=60)  # 반복 실행
    """

    def __init__(
        self,
        strategy_name: str,
        exchange: str = "paper",
        symbols: List[str] = None,
        position_size_krw: float = 10000,
        position_size_usdt: float = 0,
        leverage: int = 1,
        config: Config = None,
        strategy=None,
        market_data=None,
        execution=None,
        enable_stop_loss: bool = False,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        trailing_stop_pct: float = 0.03,
        max_holding_hours: float = 120.0,
        strategy_config: Optional[dict] = None,
        force_paper: bool = False,
        exchange_label: Optional[str] = None,
    ):
        self.strategy_name = strategy_name
        self.exchange = exchange
        self._exchange_label = exchange_label or exchange
        self._force_paper = force_paper
        self.symbols = symbols or ["BTC/KRW"]
        self.position_size_krw = position_size_krw
        self.position_size_usdt = position_size_usdt
        self.leverage = leverage

        # Determine quote currency based on exchange
        self._is_usdt_based = exchange in {"binance_spot", "binance_futures"}
        self._is_kr_stock = exchange in {
            "ebest",
            "ebest_spot",
            "ebest_kospi",
            "ebest_kosdaq",
        }
        self._quote_currency = "USDT" if self._is_usdt_based else "KRW"
        self._position_size = (
            position_size_usdt if self._is_usdt_based else position_size_krw
        )
        self._effective_size = self._position_size

        # Dynamic sizing mode (set externally by scheduler)
        self._position_size_mode = "fixed"  # "fixed" | "equal_weight"
        self._min_position_krw = 50000
        self._min_position_usdt = 50

        # Config 로드
        self.config = config or Config(
            asset_class="crypto_spot", strategy_name=strategy_name
        )

        # --- Phase 6D-Fix: 전략 로딩 canonicalization + fail-fast ---
        self.strategy, loaded_key = self._load_strategy_with_fallback(
            strategy, strategy_name
        )
        logger.info(
            "[StrategyRunner] Strategy loaded: requested=%s loaded=%s type=%s",
            strategy_name,
            loaded_key,
            type(self.strategy).__name__,
        )

        # Inject strategy-specific config (before set_exchange_name, state path may depend on it)
        if strategy_config and hasattr(self.strategy, "config"):
            self.strategy.config.update(strategy_config)

        # 로그 디렉토리 (exchange_label for isolation, e.g. "upbit_ankle_nogate")
        log_label = exchange_label or exchange
        log_base = Path(f"logs/{log_label}_trades")

        # 컴포넌트 초기화
        self.trade_logger = TradeLogger(log_dir=str(log_base / "trades"))
        self.health_monitor = StrategyHealthMonitor(self.config)
        self.daily_reporter = DailyReportGenerator(
            self.trade_logger, self.health_monitor, report_dir=str(log_base / "reports")
        )

        # 실행 어댑터
        execution_exchange = exchange
        adapter_mode = "paper"
        live_trading_enabled = os.getenv("MASP_ENABLE_LIVE_TRADING") == "1"

        if force_paper:
            adapter_mode = "paper"
        elif exchange in {"upbit", "upbit_spot"}:
            execution_exchange = "upbit_spot"
            adapter_mode = "live" if live_trading_enabled else "paper"
        elif exchange in {"bithumb", "bithumb_spot"}:
            execution_exchange = "bithumb"
            adapter_mode = "live" if live_trading_enabled else "paper"
        elif exchange == "binance_spot":
            execution_exchange = "binance_spot"
            adapter_mode = "live" if live_trading_enabled else "paper"
        elif exchange == "binance_futures":
            execution_exchange = "binance_futures"
            adapter_mode = "live" if live_trading_enabled else "paper"
        elif exchange in {"ebest", "ebest_spot", "ebest_kospi", "ebest_kosdaq"}:
            execution_exchange = exchange
            adapter_mode = "live" if live_trading_enabled else "paper"

        self.execution = execution or AdapterFactory.create_execution(
            execution_exchange,
            adapter_mode=adapter_mode,
            config=self.config if exchange != "paper" else None,
            trade_logger=self.trade_logger,
        )
        self._execution_adapter = self.execution

        # 시세 어댑터
        if exchange in ["paper", "upbit", "upbit_spot"]:
            md_exchange = "upbit_spot"
        elif exchange in ["bithumb", "bithumb_spot"]:
            md_exchange = "bithumb_spot"
        elif exchange == "binance_spot":
            md_exchange = "binance_spot"
        elif exchange == "binance_futures":
            md_exchange = "binance_futures"
        elif exchange in {"ebest", "ebest_spot", "ebest_kospi", "ebest_kosdaq"}:
            md_exchange = "ebest_spot"
        else:
            md_exchange = "upbit_spot"

        self.market_data = market_data or AdapterFactory.create_market_data(md_exchange)

        # MarketData 주입 (duck-typing)
        if hasattr(self.strategy, "set_market_data"):
            self.strategy.set_market_data(self.market_data)
            logger.info(
                "[StrategyRunner] Injected %s market data into strategy", md_exchange
            )
        # Exchange name 주입 (state isolation)
        if hasattr(self.strategy, "set_exchange_name"):
            self.strategy.set_exchange_name(exchange)
        else:
            logger.warning(
                "[StrategyRunner] Strategy has no set_market_data(): %s",
                type(self.strategy).__name__,
            )

        # 포지션 상태
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._last_signals: Dict[str, str] = {}  # symbol -> signal

        # Stop Loss Manager (선택적 활성화)
        self._enable_stop_loss = enable_stop_loss
        self._stop_loss_manager: Optional[CompositeStopManager] = None
        if enable_stop_loss:
            self._stop_loss_manager = CompositeStopManager()
            # Trailing stop with activation threshold
            self._stop_loss_manager.add_strategy(
                TrailingStop(
                    trail_pct=trailing_stop_pct,
                    activation_pct=trailing_stop_pct * 0.7,  # Activate at 70% of trail
                    initial_stop_pct=stop_loss_pct,
                )
            )
            # Fixed take profit
            self._stop_loss_manager.add_strategy(
                FixedPercentageStop(
                    stop_loss_pct=stop_loss_pct * 2,  # Wider emergency stop
                    take_profit_pct=take_profit_pct,
                )
            )
            # Time-based exit
            self._stop_loss_manager.add_strategy(
                TimeBasedStop(
                    max_holding_hours=max_holding_hours,
                    fallback_stop_pct=stop_loss_pct,
                )
            )
            logger.info(
                "[StrategyRunner] StopLossManager enabled: SL=%.1f%%, TP=%.1f%%, Trail=%.1f%%",
                stop_loss_pct * 100,
                take_profit_pct * 100,
                trailing_stop_pct * 100,
            )

        # Telegram 알림 (best-effort, 설정 없으면 비활성화)
        self._notifier = TelegramNotifier()
        if self._notifier.enabled:
            logger.info("[StrategyRunner] Telegram notifications enabled")

        logger.info("[StrategyRunner] Initialized: %s on %s", strategy_name, exchange)

    def _sync_positions_from_exchange(self) -> None:
        """
        거래소의 실제 보유량을 전략의 _positions에 동기화.

        BTC Gate 실패 시 SELL 시그널이 제대로 생성되려면
        전략이 실제 보유 포지션을 알아야 합니다.

        Supports two formats:
        - Upbit: List[Dict] with {'currency': 'BTC', 'balance': '0.001', 'locked': '0'}
        - Bithumb: Dict[str, float] with {'BTC': 0.001, 'KRW': 1000}
        """
        if not hasattr(self.execution, "get_all_balances"):
            logger.debug(
                "[StrategyRunner] Execution adapter has no get_all_balances method"
            )
            return

        try:
            balances = self.execution.get_all_balances()
            if not balances:
                return

            synced = 0

            # Handle Dict[str, float] format (Bithumb)
            if isinstance(balances, dict):
                for currency, total in balances.items():
                    if currency in ("KRW", "USDT", "P"):
                        continue  # Skip quote currencies and special tokens

                    try:
                        total = float(total)
                    except (ValueError, TypeError):
                        continue

                    if total <= 0:
                        continue

                    if self._is_usdt_based:
                        symbol = f"{currency}/USDT"
                    else:
                        symbol = f"{currency}/KRW"

                    if self.strategy and hasattr(self.strategy, "update_position"):
                        self.strategy.update_position(symbol, total)
                        synced += 1

                    self._positions[symbol] = total

            # Handle List[Dict] format (Upbit)
            else:
                for item in balances:
                    currency = item.get("currency", "")
                    if currency in ("KRW", "USDT"):
                        continue

                    try:
                        balance = float(item.get("balance", 0))
                        locked = float(item.get("locked", 0))
                        total = balance + locked
                    except (ValueError, TypeError):
                        continue

                    if total <= 0:
                        continue

                    if self._is_usdt_based:
                        symbol = f"{currency}/USDT"
                    else:
                        symbol = f"{currency}/KRW"

                    if self.strategy and hasattr(self.strategy, "update_position"):
                        self.strategy.update_position(symbol, total)
                        synced += 1

                    self._positions[symbol] = total

            if synced > 0:
                logger.info(
                    "[StrategyRunner] Synced %d positions from exchange", synced
                )

        except Exception as exc:
            logger.warning("[StrategyRunner] Position sync failed: %s", exc)

    @staticmethod
    def _canonicalize_strategy_name(name: str) -> List[str]:
        """
        전략 이름 정규화 - 다양한 형식 시도
        예: "KAMA-TSMOM-Gate" -> "kama_tsmom_gate"
        """
        s = (name or "").strip()
        if not s:
            return []

        lowered = s.lower()

        # separators -> underscore
        snake = re.sub(r"[\s\-\/]+", "_", lowered)
        # collapse multiple underscores
        compact = re.sub(r"_+", "_", snake).strip("_")

        candidates: List[str] = []
        for x in [s, lowered, snake, compact]:
            if x and x not in candidates:
                candidates.append(x)
        return candidates

    @classmethod
    def _load_strategy_with_fallback(cls, injected_strategy, strategy_name: str):
        """
        전략 로딩 + fail-fast
        - injected_strategy가 있으면 그대로 사용
        - 없으면 canonicalization 후보를 순회하며 get_strategy() 시도
        - 모두 실패 시 ValueError
        """
        if injected_strategy is not None:
            return injected_strategy, "<injected>"

        candidates = cls._canonicalize_strategy_name(strategy_name)
        for key in candidates:
            try:
                st = get_strategy(key)
                if st is not None:
                    return st, key
            except Exception as exc:
                logger.warning("[StrategyRunner] get_strategy(%s) raised: %s", key, exc)

        raise ValueError(f"Unknown strategy: {strategy_name} (tried: {candidates})")

    def run_once(self) -> Dict:
        """
        Run one cycle for all symbols with dynamic limits.

        Dynamic limits:
            1. Balance-based: skip BUY if available_krw < position_size_krw
            2. Time-based: stop after MASP_MAX_EXECUTION_TIME (default 1800s)
            3. Rate limit: 0.1s between symbols
        """
        results = {}
        total = len(self.symbols)
        start_time = time.time()
        max_execution_time = int(os.getenv("MASP_MAX_EXECUTION_TIME", "1800"))

        # 거래소 보유량을 전략에 동기화 (BTC Gate 실패 시 SELL 시그널 생성용)
        self._sync_positions_from_exchange()

        if os.getenv("STOP_TRADING") == "1" or self.config.is_kill_switch_active():
            logger.critical("Kill-Switch Activated! Trading Halted.")
            raise RuntimeError("Kill-Switch Enforced")

        health = self.health_monitor.check_health()
        if health.status.value in ["CRITICAL", "HALTED"]:
            logger.warning("[StrategyRunner] Health %s - skipping", health.status.value)
            raise RuntimeError(f"Health {health.status.value}")

        gate_pass = self._compute_gate_pass()

        try:
            raw_balance = self.execution.get_balance(self._quote_currency)
            available_balance = float(raw_balance) if raw_balance is not None else 0.0
        except Exception as exc:
            logger.warning(
                "[StrategyRunner] Balance check failed: %s, proceeding without limit",
                exc,
            )
            available_balance = float("inf")

        # Dynamic position sizing
        if self._position_size_mode == "equal_weight" and self.symbols:
            min_pos = (
                self._min_position_usdt
                if self._is_usdt_based
                else self._min_position_krw
            )
            if available_balance != float("inf") and len(self.symbols) > 0:
                dynamic = available_balance / len(self.symbols)
                self._effective_size = max(dynamic, min_pos)
            else:
                self._effective_size = min_pos
        else:
            self._effective_size = self._position_size

        balance_label = (
            "unlimited"
            if available_balance == float("inf")
            else f"{available_balance:,.2f} {self._quote_currency}"
        )
        max_buy = (
            "unlimited"
            if available_balance == float("inf")
            else (
                int(available_balance // self._effective_size)
                if self._effective_size > 0
                else 0
            )
        )

        logger.info(
            "[StrategyRunner] Processing %d symbols, Balance: %s, Max BUY: %s",
            total,
            balance_label,
            max_buy,
        )

        # Check stop loss conditions for existing positions (before processing new signals)
        if self._enable_stop_loss and self._stop_loss_manager:
            stop_loss_results = self._check_stop_loss_conditions()
            for sl_symbol, sl_result in stop_loss_results.items():
                if sl_result.get("action") == "SELL":
                    results[sl_symbol] = sl_result
                    logger.info(
                        "[StopLoss] %s triggered for %s: %s",
                        sl_result.get("reason", "unknown"),
                        sl_symbol,
                        sl_result,
                    )

        for i, symbol in enumerate(self.symbols):
            # Skip symbols already handled by stop loss
            if symbol in results:
                continue

            elapsed = time.time() - start_time
            if elapsed > max_execution_time:
                logger.warning(
                    "[StrategyRunner] Max execution time reached (%ss), processed %d/%d symbols",
                    max_execution_time,
                    i,
                    total,
                )
                break

            if i % 10 == 0 or i == total - 1:
                progress = (i + 1) / total * 100 if total else 100
                logger.info(
                    "[Progress] %d/%d (%.0f%%) - %s", i + 1, total, progress, symbol
                )

            try:
                if i > 0:
                    time.sleep(0.1)

                signal = self._generate_trade_signal(symbol, gate_pass)

                # --- Phase 6D-Fix: signal=None은 HOLD로 숨기지 않는다 ---
                if signal is None:
                    logger.error("[%s] Strategy returned None signal", symbol)
                    results[symbol] = {
                        "action": "ERROR",
                        "reason": "Strategy returned None",
                    }
                    continue

                action, effective_gate = self._parse_signal(signal, gate_pass)

                if action == "BUY" and not effective_gate:
                    logger.warning("[%s] Gate CLOSED. BUY blocked.", symbol)
                    results[symbol] = {"action": "BLOCKED", "reason": "Gate Veto"}
                    continue

                if (
                    action == "BUY"
                    and available_balance != float("inf")
                    and available_balance < self._effective_size
                ):
                    logger.warning("[%s] BUY skipped: insufficient balance", symbol)
                    results[symbol] = {
                        "action": "SKIP",
                        "reason": "insufficient_balance",
                    }
                    continue

                quote = self.market_data.get_quote(symbol)
                if quote is None:
                    results[symbol] = {"action": "ERROR", "reason": "Price unavailable"}
                    continue

                result = self._execute_trade_signal(symbol, signal, quote)
                results[symbol] = result

                if (
                    action == "BUY"
                    and result.get("action") == "BUY"
                    and available_balance != float("inf")
                ):
                    available_balance -= self._effective_size
                    logger.info(
                        "[StrategyRunner] %s BUY executed, remaining: %.2f %s",
                        symbol,
                        available_balance,
                        self._quote_currency,
                    )

            except Exception as exc:
                logger.error("[%s] Error: %s", symbol, exc, exc_info=True)
                results[symbol] = {"action": "ERROR", "reason": str(exc)}

        elapsed = time.time() - start_time
        actions: Dict[str, int] = {}
        for result in results.values():
            action = result.get("action", "UNKNOWN")
            actions[action] = actions.get(action, 0) + 1

        logger.info(
            "[Summary] Processed %d/%d symbols in %.1fs | Actions: %s",
            len(results),
            total,
            elapsed,
            actions,
        )

        return results

    def _parse_signal(self, signal, default_gate_pass: bool) -> Tuple[str, bool]:
        """
        Gate 상태 추출 (방어적 접근).
        NOTE: signal=None은 상위에서 ERROR 처리하므로 여기서는 보조 방어만 유지.
        """
        if signal is None:
            return "ERROR", default_gate_pass

        action = getattr(signal, "action", None)
        if action is None:
            raw_signal = getattr(signal, "signal", None)
            if isinstance(raw_signal, StrategySignal):
                action = raw_signal.value
            elif raw_signal is not None:
                action = str(raw_signal)

        gate_pass = getattr(signal, "gate_pass", default_gate_pass)
        if gate_pass is None:
            gate_pass = default_gate_pass

        # per-symbol 로그 폭발 방지: debug 권장
        logger.debug("Signal: %s, Gate: %s", action, "OPEN" if gate_pass else "CLOSED")
        return action, gate_pass

    def _compute_gate_pass(self) -> bool:
        if self.strategy and hasattr(self.strategy, "check_gate"):
            try:
                return bool(self.strategy.check_gate())
            except Exception as exc:
                logger.warning("[StrategyRunner] Gate check failed: %s", exc)
        return True

    def _generate_trade_signal(self, symbol: str, gate_pass: bool):
        if self.strategy and hasattr(self.strategy, "generate_signal"):
            try:
                return self.strategy.generate_signal(symbol, gate_pass=gate_pass)
            except TypeError:
                return self.strategy.generate_signal(symbol)
        if self.strategy and hasattr(self.strategy, "generate_signals"):
            signals = self.strategy.generate_signals([symbol])
            return signals[0] if signals else None
        return None

    def _execute_trade_signal(self, symbol: str, signal, quote) -> Dict:
        """주문 실행 로직."""
        action, _gate_pass = self._parse_signal(signal, True)
        current_price = self._extract_price(quote)

        if action == "ERROR":
            return {"action": "ERROR", "reason": "signal=None"}

        if action == "HOLD":
            return {"action": "HOLD", "reason": getattr(signal, "reason", "N/A")}

        # Minimum order size based on quote currency
        min_order = 5 if self._is_usdt_based else MIN_ORDER_KRW

        if action == "BUY":
            # For USDT-based exchanges, use amount_krw parameter (adapter will interpret as USDT)
            order = self.execution.place_order(
                symbol,
                "BUY",
                order_type="MARKET",
                amount_krw=self._effective_size,
            )
            # Register position for stop loss tracking
            if self._enable_stop_loss and current_price > 0:
                quantity = self._effective_size / current_price
                self._register_position_for_stop_loss(
                    symbol, "long", current_price, quantity
                )

            # Telegram 알림 (best-effort)
            self._send_trade_notification(
                symbol, "BUY", self._effective_size, current_price, "FILLED"
            )
            return {"action": "BUY", "order_id": order.order_id or order.symbol}

        if action == "SELL":
            base_asset = self._base_asset(symbol)
            balance = self.execution.get_balance(base_asset) or 0
            estimated_value = balance * current_price

            if estimated_value < min_order:
                logger.info(
                    "[%s] Dust Skip: %.2f %s < %.2f",
                    symbol,
                    estimated_value,
                    self._quote_currency,
                    min_order,
                )
                # Remove from stop loss tracker if exists
                if self._stop_loss_manager:
                    self._stop_loss_manager.close_position(symbol)
                return {
                    "action": "SKIP",
                    "reason": f"Dust ({estimated_value:.2f} {self._quote_currency})",
                }

            # Partial sell support: use signal.strength (0..1) to determine sell qty
            sell_strength = getattr(signal, "strength", 1.0)
            if sell_strength is None:
                sell_strength = 1.0
            sell_strength = max(0.0, min(1.0, float(sell_strength)))
            sell_units = balance * sell_strength
            sell_value = sell_units * current_price

            if sell_value < min_order:
                logger.info(
                    "[%s] Partial sell too small: %.2f %s < %.2f",
                    symbol,
                    sell_value,
                    self._quote_currency,
                    min_order,
                )
                return {
                    "action": "SKIP",
                    "reason": f"Partial sell too small ({sell_value:.2f} {self._quote_currency})",
                }

            order = self.execution.place_order(
                symbol,
                "SELL",
                order_type="MARKET",
                units=sell_units,
            )
            # Remove from stop loss tracker only on full sell
            if sell_strength >= 0.999 and self._stop_loss_manager:
                self._stop_loss_manager.close_position(symbol)

            # Telegram 알림 (best-effort)
            sell_pct = f" ({sell_strength:.0%})" if sell_strength < 0.999 else ""
            self._send_trade_notification(
                symbol, f"SELL{sell_pct}", sell_units, current_price, "FILLED"
            )
            return {"action": "SELL", "order_id": order.order_id or order.symbol}

        return {"action": "UNKNOWN", "reason": str(action)}

    def _extract_price(self, quote) -> float:
        if quote is None:
            return 0.0
        if hasattr(quote, "last") and quote.last is not None:
            return float(quote.last)
        if isinstance(quote, dict):
            return float(quote.get("trade_price") or quote.get("last") or 0)
        return 0.0

    def _base_asset(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol.split("/")[0]
        if "-" in symbol:
            return symbol.split("-")[1]
        return symbol

    def _check_stop_loss_conditions(self) -> Dict[str, Dict]:
        """
        Check stop loss conditions for all tracked positions.

        Returns:
            Dict of symbol -> result with executed SELL orders
        """
        results = {}
        if not self._stop_loss_manager:
            return results

        # Get current prices for all positions
        prices = {}
        for symbol in self._stop_loss_manager.positions:
            try:
                quote = self.market_data.get_quote(symbol)
                if quote:
                    prices[symbol] = self._extract_price(quote)
            except Exception as exc:
                logger.warning("[StopLoss] Failed to get price for %s: %s", symbol, exc)

        # Check all positions
        exit_signals = self._stop_loss_manager.check_all_positions(prices)

        for symbol, signal in exit_signals.items():
            if not signal.should_exit:
                continue

            # Execute SELL order
            try:
                base_asset = self._base_asset(symbol)
                balance = self.execution.get_balance(base_asset) or 0
                current_price = prices.get(symbol, 0)

                if balance <= 0:
                    logger.warning(
                        "[StopLoss] No balance for %s, removing from tracker", symbol
                    )
                    self._stop_loss_manager.close_position(symbol)
                    continue

                min_order = 5 if self._is_usdt_based else MIN_ORDER_KRW
                estimated_value = balance * current_price

                if estimated_value < min_order:
                    logger.info(
                        "[StopLoss] Dust skip for %s: %.2f", symbol, estimated_value
                    )
                    self._stop_loss_manager.close_position(symbol)
                    continue

                order = self.execution.place_order(
                    symbol,
                    "SELL",
                    order_type="MARKET",
                    units=balance,
                )

                # Remove from stop loss tracker
                self._stop_loss_manager.close_position(symbol)

                # Telegram notification
                reason_str = signal.reason.value if signal.reason else "stop_loss"
                self._send_trade_notification(
                    symbol, "SELL", balance, current_price, f"STOP_LOSS ({reason_str})"
                )

                results[symbol] = {
                    "action": "SELL",
                    "reason": f"StopLoss: {reason_str}",
                    "pnl_percent": signal.pnl_percent,
                    "order_id": order.order_id or order.symbol,
                }

                logger.info(
                    "[StopLoss] Executed SELL for %s: reason=%s, pnl=%.2f%%",
                    symbol,
                    reason_str,
                    (signal.pnl_percent or 0) * 100,
                )

            except Exception as exc:
                logger.error(
                    "[StopLoss] Failed to execute SELL for %s: %s", symbol, exc
                )
                results[symbol] = {"action": "ERROR", "reason": str(exc)}

        return results

    def _register_position_for_stop_loss(
        self, symbol: str, side: str, entry_price: float, quantity: float
    ) -> None:
        """Register a new position with the stop loss manager."""
        if not self._stop_loss_manager:
            return

        self._stop_loss_manager.open_position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
        )
        logger.info(
            "[StopLoss] Registered position: %s %s @ %.2f x %.6f",
            side,
            symbol,
            entry_price,
            quantity,
        )

    def run_loop(self, interval_seconds: int = 60, max_iterations: int = None):
        iteration = 0
        logger.info("[StrategyRunner] Starting loop (interval=%ss)", interval_seconds)

        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                logger.info("[StrategyRunner] Iteration %d", iteration)

                results = self.run_once()
                logger.info("[StrategyRunner] Results: %s", results)

                if results.get("status") == "HALTED":
                    logger.warning("[StrategyRunner] Halted - stopping loop")
                    break

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("[StrategyRunner] Interrupted by user")

        self.generate_daily_report()

    def _send_trade_notification(
        self, symbol: str, side: str, quantity: float, price: float, status: str
    ) -> None:
        """Send trade notification via Telegram (best-effort, swallow errors)."""
        if not self._notifier.enabled:
            return
        try:
            mode = "PAPER" if self._force_paper else ""
            msg = format_trade_message(
                self._exchange_label, symbol, side, quantity, price, status, mode=mode
            )
            self._notifier.send_message_sync(msg)
        except Exception as exc:
            logger.debug("[Telegram] Notification failed (swallowed): %s", exc)

    def generate_daily_report(self) -> str:
        report = self.daily_reporter.generate()
        # Send daily summary via Telegram (best-effort)
        if self._notifier.enabled:
            try:
                trades_today = self.trade_logger.get_trade_count(date.today())
                pnl = (
                    self.trade_logger.get_daily_pnl(date.today())
                    if hasattr(self.trade_logger, "get_daily_pnl")
                    else 0
                )
                msg = format_daily_summary(
                    self._exchange_label, trades_today, pnl, self._quote_currency
                )
                self._notifier.send_message_sync(msg)
            except Exception as exc:
                logger.debug("[Telegram] Daily summary failed (swallowed): %s", exc)
        return report

    def get_status(self) -> Dict:
        return {
            "strategy": self.strategy_name,
            "exchange": self.exchange,
            "symbols": self.symbols,
            "positions": self._positions.copy(),
            "health": self.health_monitor.get_summary(),
            "trades_today": self.trade_logger.get_trade_count(date.today()),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Runner")
    parser.add_argument("--strategy", default="ma_crossover_v1", help="Strategy name")
    parser.add_argument(
        "--exchange",
        default="paper",
        choices=[
            "paper",
            "upbit",
            "bithumb",
            "binance_spot",
            "binance_futures",
            "ebest",
            "ebest_spot",
            "ebest_kospi",
            "ebest_kosdaq",
        ],
    )
    parser.add_argument("--symbol", default="BTC/KRW", help="Trading symbol")
    parser.add_argument(
        "--size", type=float, default=10000, help="Position size (KRW/USDT)"
    )
    parser.add_argument(
        "--leverage", type=int, default=1, help="Leverage (for futures)"
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Loop interval (seconds)"
    )
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations")
    parser.add_argument("--once", action="store_true", help="Run once only")

    args = parser.parse_args()

    # Determine position size based on exchange
    is_usdt = args.exchange in {"binance_spot", "binance_futures"}

    runner = StrategyRunner(
        strategy_name=args.strategy,
        exchange=args.exchange,
        symbols=[args.symbol],
        position_size_krw=0 if is_usdt else args.size,
        position_size_usdt=args.size if is_usdt else 0,
        leverage=args.leverage,
    )

    if args.once:
        result = runner.run_once()
        print(f"Result: {result}")
    else:
        runner.run_loop(interval_seconds=args.interval, max_iterations=args.iterations)
