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
import time
from datetime import datetime, date
from typing import Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv

# Load .env without overriding system environment variables.
load_dotenv(override=False)

from libs.core.config import Config
from libs.adapters.factory import AdapterFactory
from libs.adapters.trade_logger import TradeLogger
from libs.analytics.strategy_health import StrategyHealthMonitor
from libs.analytics.daily_report import DailyReportGenerator
from libs.strategies.base import Signal as StrategySignal
from libs.strategies.loader import get_strategy

logger = logging.getLogger(__name__)
MIN_ORDER_KRW = 5000


class StrategyRunner:
    """
    전략 실행기
    
    Usage:
        runner = StrategyRunner(
            strategy_name="ma_crossover",
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
        config: Config = None,
        strategy=None,
        market_data=None,
        execution=None
    ):
        """
        초기화
        
        Args:
            strategy_name: 전략 이름
            exchange: "paper" | "upbit" | "bithumb"
            symbols: 거래 종목 리스트
            position_size_krw: 1회 주문 금액 (KRW)
            config: Config 객체
        """
        self.strategy_name = strategy_name
        self.exchange = exchange
        self.symbols = symbols or ["BTC/KRW"]
        self.position_size_krw = position_size_krw
        
        # Config 로드
        self.config = config or Config(
            asset_class="crypto_spot",
            strategy_name=strategy_name
        )

        self.strategy = strategy or get_strategy(strategy_name)
        
        # 로그 디렉토리
        log_base = Path(f"logs/{exchange}_trades")
        
        # 컴포넌트 초기화
        self.trade_logger = TradeLogger(log_dir=str(log_base / "trades"))
        self.health_monitor = StrategyHealthMonitor(self.config)
        self.daily_reporter = DailyReportGenerator(
            self.trade_logger,
            self.health_monitor,
            report_dir=str(log_base / "reports")
        )
        
        # 실행 어댑터
        execution_exchange = exchange
        adapter_mode = "paper"
        live_trading_enabled = os.getenv("MASP_ENABLE_LIVE_TRADING") == "1"

        if exchange in {"upbit", "upbit_spot"}:
            execution_exchange = "upbit_spot"
            adapter_mode = "live" if live_trading_enabled else "paper"
        elif exchange in {"bithumb", "bithumb_spot"}:
            execution_exchange = "bithumb"
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
        else:
            md_exchange = "upbit_spot"
        self.market_data = market_data or AdapterFactory.create_market_data(md_exchange)

        if self.strategy and hasattr(self.strategy, "set_market_data"):
            self.strategy.set_market_data(self.market_data)
            logger.info(
                "[StrategyRunner] Injected %s market data into strategy",
                md_exchange,
            )
        
        # 포지션 상태
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._last_signals: Dict[str, str] = {}  # symbol -> signal
        
        logger.info(f"[StrategyRunner] Initialized: {strategy_name} on {exchange}")
    
    def run_once(self) -> Dict:
        """
        Run one cycle for all symbols with dynamic limits.

        Dynamic limits:
            1. Balance-based: skip BUY if available_krw < position_size_krw
            2. Time-based: stop after max_execution_time (5 minutes)
            3. Rate limit: 0.1s between symbols
        """
        results = {}
        total = len(self.symbols)
        start_time = time.time()
        max_execution_time = 300

        if os.getenv("STOP_TRADING") == "1" or self.config.is_kill_switch_active():
            logger.critical("Kill-Switch Activated! Trading Halted.")
            raise RuntimeError("Kill-Switch Enforced")

        health = self.health_monitor.check_health()
        if health.status.value in ["CRITICAL", "HALTED"]:
            logger.warning(f"[StrategyRunner] Health {health.status.value} - skipping")
            raise RuntimeError(f"Health {health.status.value}")

        gate_pass = self._compute_gate_pass()

        try:
            raw_balance = self.execution.get_balance("KRW")
            available_krw = float(raw_balance) if raw_balance is not None else 0.0
        except Exception as exc:
            logger.warning(
                "[StrategyRunner] Balance check failed: %s, proceeding without limit",
                exc,
            )
            available_krw = float("inf")

        balance_label = (
            "unlimited" if available_krw == float("inf") else f"{available_krw:,.0f} KRW"
        )
        max_buy = (
            "unlimited"
            if available_krw == float("inf")
            else int(available_krw // self.position_size_krw)
        )
        logger.info(
            "[StrategyRunner] Processing %d symbols, Balance: %s, Max BUY: %s",
            total,
            balance_label,
            max_buy,
        )

        for i, symbol in enumerate(self.symbols):
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
                logger.info("[Progress] %d/%d (%.0f%%) - %s", i + 1, total, progress, symbol)

            try:
                if i > 0:
                    time.sleep(0.1)

                signal = self._generate_trade_signal(symbol, gate_pass)
                action, effective_gate = self._parse_signal(signal, gate_pass)

                if action == "BUY" and not effective_gate:
                    logger.warning(f"[{symbol}] Gate CLOSED. BUY blocked.")
                    results[symbol] = {"action": "BLOCKED", "reason": "Gate Veto"}
                    continue

                if (
                    action == "BUY"
                    and available_krw != float("inf")
                    and available_krw < self.position_size_krw
                ):
                    logger.warning("[%s] BUY skipped: insufficient balance", symbol)
                    results[symbol] = {"action": "SKIP", "reason": "insufficient_balance"}
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
                    and available_krw != float("inf")
                ):
                    available_krw -= self.position_size_krw
                    remaining_label = f"{available_krw:,.0f} KRW"
                    logger.info(
                        "[StrategyRunner] %s BUY executed, remaining: %s",
                        symbol,
                        remaining_label,
                    )
            except Exception as exc:
                logger.error(f"[{symbol}] Error: {exc}", exc_info=True)
                results[symbol] = {"action": "ERROR", "reason": str(exc)}

        elapsed = time.time() - start_time
        actions = {}
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

    # Step 0 confirmed: TradeSignal.signal (Signal enum), no TradeSignal.gate_pass/action.
    # generate_signal signature (KAMA-TSMOM-Gate): (symbol, gate_pass: Optional[bool] = None).
    def _parse_signal(self, signal, default_gate_pass: bool) -> tuple[str, bool]:
        """Gate 상태 추출 (방어적 접근)."""
        if signal is None:
            return "HOLD", default_gate_pass
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
        logger.info("Signal: %s, Gate: %s", action, "OPEN" if gate_pass else "CLOSED")
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
        """Upbit 특화 주문 로직."""
        action, _gate_pass = self._parse_signal(signal, True)
        current_price = self._extract_price(quote)

        if action == "HOLD":
            return {"action": "HOLD", "reason": getattr(signal, "reason", "N/A")}

        if action == "BUY":
            # BithumbExecutionAdapter는 amount_krw= 파라미터를 지원
            # position_size_krw를 amount_krw로 전달하면 내부에서 코인 수량으로 변환
            order = self.execution.place_order(
                symbol,
                "BUY",
                order_type="MARKET",
                amount_krw=self.position_size_krw,  # ✅ amount_krw 사용
            )
            return {"action": "BUY", "order_id": order.order_id or order.symbol}

        if action == "SELL":
            base_asset = self._base_asset(symbol)
            balance = self.execution.get_balance(base_asset) or 0
            estimated_value = balance * current_price

            if estimated_value < MIN_ORDER_KRW:
                logger.info(
                    "[%s] Dust Skip: %d KRW < %d",
                    symbol,
                    int(estimated_value),
                    MIN_ORDER_KRW,
                )
                return {"action": "SKIP", "reason": f"Dust ({estimated_value:.0f} KRW)"}

            # SELL은 코인 수량(balance)을 units= 파라미터로 전달
            order = self.execution.place_order(
                symbol,
                "SELL",
                order_type="MARKET",
                units=balance,  # ✅ units 명시
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

    def run_loop(self, interval_seconds: int = 60, max_iterations: int = None):
        """
        반복 실행
        
        Args:
            interval_seconds: 실행 간격 (초)
            max_iterations: 최대 반복 횟수 (None=무한)
        """
        iteration = 0
        
        logger.info(f"[StrategyRunner] Starting loop (interval={interval_seconds}s)")
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                logger.info(f"[StrategyRunner] Iteration {iteration}")
                
                results = self.run_once()
                logger.info(f"[StrategyRunner] Results: {results}")
                
                if results.get("status") == "HALTED":
                    logger.warning("[StrategyRunner] Halted - stopping loop")
                    break
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("[StrategyRunner] Interrupted by user")
        
        # 종료 시 Daily Report 생성
        self.generate_daily_report()
    
    def generate_daily_report(self) -> str:
        """Daily Report 생성"""
        return self.daily_reporter.generate()
    
    def get_status(self) -> Dict:
        """현재 상태 조회"""
        return {
            "strategy": self.strategy_name,
            "exchange": self.exchange,
            "symbols": self.symbols,
            "positions": self._positions.copy(),
            "health": self.health_monitor.get_summary(),
            "trades_today": self.trade_logger.get_trade_count(date.today())
        }


# CLI 실행
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Runner")
    parser.add_argument("--strategy", default="ma_crossover", help="Strategy name")
    parser.add_argument("--exchange", default="paper", choices=["paper", "upbit", "bithumb"])
    parser.add_argument("--symbol", default="BTC/KRW", help="Trading symbol")
    parser.add_argument("--size", type=float, default=10000, help="Position size (KRW)")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval (seconds)")
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations")
    parser.add_argument("--once", action="store_true", help="Run once only")
    
    args = parser.parse_args()
    
    runner = StrategyRunner(
        strategy_name=args.strategy,
        exchange=args.exchange,
        symbols=[args.symbol],
        position_size_krw=args.size
    )
    
    if args.once:
        result = runner.run_once()
        print(f"Result: {result}")
    else:
        runner.run_loop(
            interval_seconds=args.interval,
            max_iterations=args.iterations
        )
