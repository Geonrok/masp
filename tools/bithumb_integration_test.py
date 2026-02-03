"""
Bithumb Integration Test Suite
Phase 8: 빗썸 실거래 통합 검증

실행 전 확인:
1. .env에 BITHUMB_API_KEY, BITHUMB_SECRET_KEY 설정
2. MASP_ENABLE_LIVE_TRADING=1 (실거래 테스트 시)
3. 테스트 계정에 충분한 KRW 잔고 (10,000원 이상)

사용법:
    python tools/bithumb_integration_test.py --mode=readonly  # 조회만
    python tools/bithumb_integration_test.py --mode=dryrun   # 소액 거래 포함
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)


class TestResult:
    """Test result container."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details: Dict = {}
        self.duration_ms = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


class BithumbIntegrationTest:
    """Bithumb integration test suite."""

    def __init__(self, mode: str = "readonly"):
        self.mode = mode
        self.results: List[TestResult] = []
        self.adapter = None
        self.market_data = None

    def run(self) -> bool:
        """Run all tests."""
        print("=" * 60)
        print("Bithumb Integration Test Suite")
        print(f"Mode: {self.mode}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Initialize
        if not self._init_adapters():
            return False

        # Run tests
        self._test_api_connectivity()
        self._test_balance_query()
        self._test_market_data()
        self._test_order_status_tracking()
        self._test_position_sync()

        if self.mode == "dryrun":
            self._test_small_order()

        # Summary
        return self._print_summary()

    def _init_adapters(self) -> bool:
        """Initialize adapters."""
        result = TestResult("Adapter Initialization")
        import time

        start = time.time()

        try:
            from libs.adapters.factory import AdapterFactory
            from libs.core.config import Config

            # Check environment
            api_key = os.getenv("BITHUMB_API_KEY")
            if not api_key or api_key == "your_api_key_here":
                result.message = "BITHUMB_API_KEY not configured"
                result.details = {"env_check": "failed"}
                self.results.append(result)
                return False

            # Create adapters
            self.market_data = AdapterFactory.create_market_data("bithumb_spot")

            if self.mode == "dryrun":
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    result.message = (
                        "MASP_ENABLE_LIVE_TRADING must be 1 for dryrun mode"
                    )
                    self.results.append(result)
                    return False

                config = Config(
                    asset_class="crypto_spot", strategy_name="integration_test"
                )
                from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

                self.adapter = BithumbExecutionAdapter(config)
            else:
                # Readonly mode - still need adapter for balance checks
                try:
                    config = Config(
                        asset_class="crypto_spot", strategy_name="integration_test"
                    )
                    from libs.adapters.real_bithumb_execution import (
                        BithumbExecutionAdapter,
                    )

                    self.adapter = BithumbExecutionAdapter(config)
                except Exception as e:
                    print(f"  Note: Execution adapter not available: {e}")
                    self.adapter = None

            result.passed = True
            result.message = "Adapters initialized"
            result.details = {
                "market_data": type(self.market_data).__name__,
                "execution": type(self.adapter).__name__ if self.adapter else "N/A",
            }

        except Exception as e:
            result.message = f"Failed: {e}"
            result.details = {"error": str(e)}

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)
        print(f"\n[1] {result.name}: {'PASS' if result.passed else 'FAIL'}")
        print(f"    {result.message}")
        return result.passed

    def _test_api_connectivity(self):
        """Test API connectivity."""
        result = TestResult("API Connectivity")
        import time

        start = time.time()

        try:
            # Test public API
            quote = self.market_data.get_quote("BTC/KRW")
            if quote and quote.last > 0:
                result.passed = True
                result.message = f"BTC/KRW price: {quote.last:,.0f} KRW"
                result.details = {
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "last": quote.last,
                }
            else:
                result.message = "Failed to get BTC quote"

        except Exception as e:
            result.message = f"Error: {e}"

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)
        print(f"\n[2] {result.name}: {'PASS' if result.passed else 'FAIL'}")
        print(f"    {result.message}")

    def _test_balance_query(self):
        """Test balance query."""
        result = TestResult("Balance Query")
        import time

        start = time.time()

        if not self.adapter:
            result.message = "Skipped (no execution adapter)"
            self.results.append(result)
            print(f"\n[3] {result.name}: SKIP")
            return

        try:
            krw_balance = self.adapter.get_balance("KRW")
            all_balances = self.adapter.get_all_balances()

            result.passed = True
            result.message = f"KRW: {krw_balance:,.0f}"
            result.details = {
                "krw_balance": krw_balance,
                "all_balances": all_balances,
            }

        except Exception as e:
            result.message = f"Error: {e}"

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)
        print(f"\n[3] {result.name}: {'PASS' if result.passed else 'FAIL'}")
        print(f"    {result.message}")

    def _test_market_data(self):
        """Test market data functions."""
        result = TestResult("Market Data")
        import time

        start = time.time()

        try:
            # Test OHLCV
            ohlcv = self.market_data.get_ohlcv("BTC/KRW", interval="1d", limit=5)

            # Test orderbook
            orderbook = self.market_data.get_orderbook("BTC/KRW", depth=5)

            if ohlcv and len(ohlcv) > 0 and orderbook:
                result.passed = True
                result.message = f"OHLCV: {len(ohlcv)} candles, Orderbook: OK"
                result.details = {
                    "ohlcv_count": len(ohlcv),
                    "latest_close": ohlcv[-1].close if ohlcv else None,
                    "orderbook_bids": len(orderbook.get("bids", [])),
                    "orderbook_asks": len(orderbook.get("asks", [])),
                }
            else:
                result.message = "Incomplete data"

        except Exception as e:
            result.message = f"Error: {e}"

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)
        print(f"\n[4] {result.name}: {'PASS' if result.passed else 'FAIL'}")
        print(f"    {result.message}")

    def _test_order_status_tracking(self):
        """Test order status tracking."""
        result = TestResult("Order Status Tracking")
        import time

        start = time.time()

        if not self.adapter:
            result.message = "Skipped (no execution adapter)"
            self.results.append(result)
            print(f"\n[5] {result.name}: SKIP")
            return

        try:
            # Get recent orders
            recent_orders = self.adapter.get_recent_orders(limit=5)
            open_orders = self.adapter.get_open_orders()

            result.passed = True
            result.message = f"Recent: {len(recent_orders)}, Open: {len(open_orders)}"
            result.details = {
                "recent_orders_count": len(recent_orders),
                "open_orders_count": len(open_orders),
                "recent_order_states": (
                    [o.state for o in recent_orders[:3]] if recent_orders else []
                ),
            }

        except Exception as e:
            result.message = f"Error: {e}"

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)
        print(f"\n[5] {result.name}: {'PASS' if result.passed else 'FAIL'}")
        print(f"    {result.message}")

    def _test_position_sync(self):
        """Test position synchronization."""
        result = TestResult("Position Sync")
        import time

        start = time.time()

        if not self.adapter:
            result.message = "Skipped (no execution adapter)"
            self.results.append(result)
            print(f"\n[6] {result.name}: SKIP")
            return

        try:
            positions = self.adapter.sync_positions()
            portfolio = self.adapter.get_total_portfolio_value()

            result.passed = True
            result.message = f"Positions: {len(positions)}, Total: {portfolio['total_value']:,.0f} KRW"
            result.details = {
                "position_count": len(positions),
                "position_symbols": list(positions.keys()),
                "total_value": portfolio["total_value"],
                "krw_balance": portfolio["krw_balance"],
                "positions_value": portfolio["positions_value"],
            }

        except Exception as e:
            result.message = f"Error: {e}"

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)
        print(f"\n[6] {result.name}: {'PASS' if result.passed else 'FAIL'}")
        print(f"    {result.message}")

    def _test_small_order(self):
        """Test small order execution (dryrun mode only)."""
        result = TestResult("Small Order Test")
        import time

        start = time.time()

        if self.mode != "dryrun":
            result.message = "Skipped (readonly mode)"
            self.results.append(result)
            print(f"\n[7] {result.name}: SKIP")
            return

        if not self.adapter:
            result.message = "Skipped (no execution adapter)"
            self.results.append(result)
            print(f"\n[7] {result.name}: SKIP")
            return

        try:
            # Check KRW balance
            krw_balance = self.adapter.get_balance("KRW")
            test_amount = 6000  # Minimum + buffer

            if krw_balance < test_amount:
                result.message = (
                    f"Insufficient balance: {krw_balance:,.0f} < {test_amount:,} KRW"
                )
                self.results.append(result)
                print(f"\n[7] {result.name}: SKIP")
                return

            print(f"\n[7] {result.name}: EXECUTING...")
            print(f"    Test amount: {test_amount:,} KRW")

            # Place buy order
            buy_result = self.adapter.place_order(
                "BTC/KRW",
                "BUY",
                amount_krw=test_amount,
            )

            if buy_result.status == "REJECTED":
                result.message = f"Buy rejected: {buy_result.message}"
                self.results.append(result)
                print(f"    {result.message}")
                return

            print(f"    Buy order: {buy_result.order_id}")

            # Wait for fill
            buy_status = self.adapter.wait_for_fill(
                buy_result.order_id,
                timeout_seconds=10.0,
            )

            if not buy_status or not buy_status.is_done:
                result.message = "Buy order not filled"
                result.details = {
                    "buy_order_id": buy_result.order_id,
                    "buy_status": buy_status.state if buy_status else "unknown",
                }
                self.results.append(result)
                print(f"    {result.message}")
                return

            print(f"    Buy filled: {buy_status.executed_volume:.8f} BTC")

            # Check BTC position
            btc_position = self.adapter.get_position("BTC/KRW")
            btc_balance = btc_position["balance"] if btc_position else 0

            if btc_balance < 0.00001:
                result.message = "BTC balance too small for sell"
                self.results.append(result)
                print(f"    {result.message}")
                return

            # Sell back
            import time as t

            t.sleep(1)  # Brief pause

            sell_result = self.adapter.place_order(
                "BTC/KRW",
                "SELL",
                units=btc_balance,
            )

            if sell_result.status == "REJECTED":
                result.message = f"Sell rejected: {sell_result.message}"
                self.results.append(result)
                print(f"    {result.message}")
                return

            print(f"    Sell order: {sell_result.order_id}")

            # Wait for sell fill
            sell_status = self.adapter.wait_for_fill(
                sell_result.order_id,
                timeout_seconds=10.0,
            )

            if sell_status and sell_status.is_done:
                result.passed = True
                result.message = "Round-trip complete"
                result.details = {
                    "buy_order_id": buy_result.order_id,
                    "buy_volume": buy_status.executed_volume,
                    "sell_order_id": sell_result.order_id,
                    "sell_volume": sell_status.executed_volume,
                }
                print(f"    Sell filled: {sell_status.executed_volume:.8f} BTC")
            else:
                result.message = "Sell order not filled"
                result.details = {
                    "sell_order_id": sell_result.order_id,
                    "sell_status": sell_status.state if sell_status else "unknown",
                }

        except Exception as e:
            result.message = f"Error: {e}"
            import traceback

            traceback.print_exc()

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)
        print(f"    Result: {'PASS' if result.passed else 'FAIL'} - {result.message}")

    def _print_summary(self) -> bool:
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        skipped = sum(1 for r in self.results if "skip" in r.message.lower())

        for r in self.results:
            status = (
                "PASS"
                if r.passed
                else ("SKIP" if "skip" in r.message.lower() else "FAIL")
            )
            print(f"  [{status:4}] {r.name}: {r.message}")

        print("-" * 60)
        print(
            f"  Total: {total}, Passed: {passed}, Skipped: {skipped}, Failed: {total - passed - skipped}"
        )

        # Save results
        results_file = (
            PROJECT_ROOT
            / "logs"
            / f"bithumb_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "mode": self.mode,
                    "results": [r.to_dict() for r in self.results],
                    "summary": {
                        "total": total,
                        "passed": passed,
                        "skipped": skipped,
                        "failed": total - passed - skipped,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n  Results saved: {results_file}")
        print("=" * 60)

        return (total - skipped) > 0 and passed >= (total - skipped)


def main():
    parser = argparse.ArgumentParser(description="Bithumb Integration Test")
    parser.add_argument(
        "--mode",
        choices=["readonly", "dryrun"],
        default="readonly",
        help="Test mode: readonly (no trades) or dryrun (small trades)",
    )
    args = parser.parse_args()

    tester = BithumbIntegrationTest(mode=args.mode)
    success = tester.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
