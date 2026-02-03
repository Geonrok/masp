"""
Paper Trading Verification Test
Phase 7E-2: Multi-Exchange Paper Trading 검증

실행 방법:
    python tools/paper_trading_test.py --exchange=upbit
    python tools/paper_trading_test.py --exchange=bithumb
    python tools/paper_trading_test.py --exchange=all

환경 설정:
    - MASP_ENABLE_LIVE_TRADING=0 (또는 미설정) - Paper 모드
    - MASP_ENABLE_LIVE_TRADING=1 - Live 모드 (주의!)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)


class PaperTradingTest:
    """Paper trading verification test suite."""

    def __init__(self, exchanges: List[str]):
        self.exchanges = exchanges
        self.results: Dict[str, Any] = {}
        self.live_mode = os.getenv("MASP_ENABLE_LIVE_TRADING") == "1"

    def run(self) -> bool:
        """Run all paper trading tests."""
        print("=" * 60)
        print("Paper Trading Verification Test")
        print(f"Mode: {'LIVE (주의!)' if self.live_mode else 'PAPER'}")
        print(f"Exchanges: {', '.join(self.exchanges)}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        if self.live_mode:
            print("\n⚠️  WARNING: LIVE TRADING MODE IS ENABLED!")
            print("    This will execute REAL orders with REAL money!")
            response = input("    Type 'YES' to continue: ")
            if response != "YES":
                print("    Aborted.")
                return False

        all_passed = True

        for exchange in self.exchanges:
            print(f"\n{'='*60}")
            print(f"Testing {exchange.upper()}")
            print("=" * 60)

            result = self._test_exchange(exchange)
            self.results[exchange] = result

            if not result.get("passed", False):
                all_passed = False

        self._print_summary()
        self._save_results()

        return all_passed

    def _test_exchange(self, exchange: str) -> Dict[str, Any]:
        """Test a single exchange."""
        result = {
            "exchange": exchange,
            "passed": False,
            "tests": {},
        }

        # Test 1: Strategy Runner initialization
        print("\n[1] Testing StrategyRunner initialization...")
        try:
            from services.strategy_runner import StrategyRunner

            runner = StrategyRunner(
                strategy_name="KAMA-TSMOM-Gate",
                exchange=exchange,
                symbols=["BTC/KRW"],
                position_size_krw=10000,
            )

            result["tests"]["init"] = {
                "passed": True,
                "message": f"Initialized with {type(runner.execution).__name__}",
            }
            print(f"    PASS: {result['tests']['init']['message']}")
        except Exception as e:
            result["tests"]["init"] = {"passed": False, "message": str(e)}
            print(f"    FAIL: {e}")
            return result

        # Test 2: Market Data
        print("\n[2] Testing MarketData adapter...")
        try:
            quote = runner.market_data.get_quote("BTC/KRW")
            if quote and quote.last > 0:
                result["tests"]["market_data"] = {
                    "passed": True,
                    "message": f"BTC/KRW: {quote.last:,.0f} KRW",
                }
                print(f"    PASS: {result['tests']['market_data']['message']}")
            else:
                result["tests"]["market_data"] = {
                    "passed": False,
                    "message": "No quote data",
                }
                print(f"    FAIL: {result['tests']['market_data']['message']}")
        except Exception as e:
            result["tests"]["market_data"] = {"passed": False, "message": str(e)}
            print(f"    FAIL: {e}")

        # Test 3: Balance Check
        print("\n[3] Testing Balance check...")
        try:
            balance = runner.execution.get_balance("KRW")
            result["tests"]["balance"] = {
                "passed": True,
                "message": f"KRW Balance: {balance:,.0f}" if balance else "N/A (Paper)",
            }
            print(f"    PASS: {result['tests']['balance']['message']}")
        except Exception as e:
            result["tests"]["balance"] = {"passed": False, "message": str(e)}
            print(f"    FAIL: {e}")

        # Test 4: Strategy Signal Generation
        print("\n[4] Testing Strategy signal generation...")
        try:
            gate_pass = runner._compute_gate_pass()
            signal = runner._generate_trade_signal("BTC/KRW", gate_pass)

            if signal is not None:
                action = getattr(signal, "action", None) or getattr(
                    signal, "signal", "UNKNOWN"
                )
                result["tests"]["signal"] = {
                    "passed": True,
                    "message": f"Signal: {action}, Gate: {'OPEN' if gate_pass else 'CLOSED'}",
                }
                print(f"    PASS: {result['tests']['signal']['message']}")
            else:
                result["tests"]["signal"] = {
                    "passed": False,
                    "message": "Strategy returned None",
                }
                print(f"    FAIL: {result['tests']['signal']['message']}")
        except Exception as e:
            result["tests"]["signal"] = {"passed": False, "message": str(e)}
            print(f"    FAIL: {e}")

        # Test 5: Run Once (Paper execution)
        print("\n[5] Testing run_once (Paper mode execution)...")
        try:
            run_result = runner.run_once()

            actions = {}
            for sym_result in run_result.values():
                action = sym_result.get("action", "UNKNOWN")
                actions[action] = actions.get(action, 0) + 1

            result["tests"]["run_once"] = {
                "passed": True,
                "message": f"Actions: {actions}",
                "details": run_result,
            }
            print(f"    PASS: {result['tests']['run_once']['message']}")
        except RuntimeError as e:
            if "Kill-Switch" in str(e) or "Health" in str(e):
                result["tests"]["run_once"] = {
                    "passed": True,
                    "message": f"Safety stop: {e}",
                }
                print(f"    PASS (Safety): {result['tests']['run_once']['message']}")
            else:
                result["tests"]["run_once"] = {"passed": False, "message": str(e)}
                print(f"    FAIL: {e}")
        except Exception as e:
            result["tests"]["run_once"] = {"passed": False, "message": str(e)}
            print(f"    FAIL: {e}")

        # Calculate overall result
        passed_count = sum(
            1 for t in result["tests"].values() if t.get("passed", False)
        )
        total_count = len(result["tests"])
        result["passed"] = passed_count >= total_count - 1  # Allow 1 failure

        return result

    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        for exchange, result in self.results.items():
            status = "PASS" if result.get("passed", False) else "FAIL"
            print(f"\n{exchange.upper()}: {status}")

            for test_name, test_result in result.get("tests", {}).items():
                test_status = "PASS" if test_result.get("passed", False) else "FAIL"
                print(
                    f"  [{test_status:4}] {test_name}: {test_result.get('message', '')}"
                )

        total_passed = sum(1 for r in self.results.values() if r.get("passed", False))
        total = len(self.results)
        print("\n" + "-" * 60)
        print(f"Overall: {total_passed}/{total} exchanges passed")
        print("=" * 60)

    def _save_results(self):
        """Save test results to file."""
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        results_file = (
            log_dir
            / f"paper_trading_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "mode": "LIVE" if self.live_mode else "PAPER",
                    "exchanges": self.exchanges,
                    "results": self.results,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(f"\nResults saved: {results_file}")


def test_scheduler_status():
    """Test MultiExchangeScheduler status."""
    print("\n" + "=" * 60)
    print("MultiExchangeScheduler Status")
    print("=" * 60)

    try:
        from services.multi_exchange_scheduler import MultiExchangeScheduler

        scheduler = MultiExchangeScheduler()
        status = scheduler.get_status()

        print(f"\nRunning: {status.get('running', False)}")
        print(f"Config: {status.get('config_path', 'N/A')}")

        for name, info in status.get("exchanges", {}).items():
            print(f"\n{name.upper()}:")
            print(f"  Strategy: {info.get('strategy', 'N/A')}")
            print(f"  Symbols: {info.get('symbols', 'N/A')}")
            print(f"  Schedule: {info.get('schedule', 'N/A')}")
            print(f"  Next Run: {info.get('next_run', 'N/A')}")

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence strategy/adapter logs during test
    logging.getLogger("libs").setLevel(logging.ERROR)
    logging.getLogger("services").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Paper Trading Verification Test")
    parser.add_argument(
        "--exchange",
        choices=["upbit", "bithumb", "all"],
        default="all",
        help="Exchange to test",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show scheduler status only",
    )
    args = parser.parse_args()

    if args.status:
        test_scheduler_status()
        return

    if args.exchange == "all":
        exchanges = ["upbit", "bithumb"]
    else:
        exchanges = [args.exchange]

    tester = PaperTradingTest(exchanges=exchanges)
    success = tester.run()

    # Also show scheduler status
    test_scheduler_status()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
