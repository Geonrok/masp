"""
Live trading test runner (v2.2 Final).
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ROOT_DIR)

from libs.adapters.factory import AdapterFactory
from libs.core.config import Config
from _helpers import resolve_market, get_fee_rates
from _teardown import cancel_all_open_orders, verify_no_open_orders
from _live_test_utils import require_live_guard, log_event, get_loss_cap_krw


TIMEOUT_SECONDS = 300


class LiveTestRunner:
    def __init__(self) -> None:
        self.test_start_time = datetime.now(timezone.utc)
        self.market = resolve_market("BTC/KRW")
        self.results: dict[str, str] = {}
        self.execution = None
        self.fee_rates = None
        self.timeout_event = threading.Event()

    def run(self) -> bool:
        timer = threading.Timer(TIMEOUT_SECONDS, self._timeout_handler)
        timer.start()
        try:
            print("=" * 60)
            print("MASP Live Trading Tests (v2.2 Final)")
            print(f"Start: {self.test_start_time}")
            print(f"Timeout: {TIMEOUT_SECONDS}s")
            print(f"Loss cap: {get_loss_cap_krw():,} KRW")
            print("=" * 60)

            require_live_guard()

            config = Config(asset_class="crypto_spot", strategy_name="live_test")
            self.execution = AdapterFactory.create_execution(
                "upbit_spot",
                adapter_mode="live",
                config=config,
            )

            print("\nFee lookup...")
            self.fee_rates = get_fee_rates(self.market, self.execution)
            print(f"✅ bid_fee: {self.fee_rates['bid_fee']*100:.4f}%")
            print(f"✅ ask_fee: {self.fee_rates['ask_fee']*100:.4f}%")

            log_event(
                {
                    "type": "test_start",
                    "start_time": self.test_start_time.isoformat(),
                    "market": self.market,
                    "fee_rates": {
                        "bid": float(self.fee_rates["bid_fee"]),
                        "ask": float(self.fee_rates["ask_fee"]),
                    },
                }
            )

            tests = [
                ("Pre-Check", self._test_pre_check),
                ("Kill-Switch", self._test_kill_switch),
                ("Balance", self._test_balance),
                ("Buy", self._test_buy),
                ("Sell", self._test_sell),
            ]

            buy_passed = None
            for name, test_func in tests:
                if self.timeout_event.is_set():
                    print("\nTimeout reached - stopping")
                    self.results[name] = "⏰ TIMEOUT"
                    break
                if name == "Sell" and buy_passed is False:
                    self.results[name] = "⏭️ SKIP (Buy failed)"
                    continue
                print(f"\n{'=' * 60}")
                print(f"Running: {name}")
                print("=" * 60)
                try:
                    success = test_func()
                    self.results[name] = "✅ PASS" if success else "❌ FAIL"
                    if name == "Buy":
                        buy_passed = bool(success)
                    if not success and name in ["Pre-Check", "Kill-Switch"]:
                        print(f"\n❌ {name} failed - stopping")
                        break
                except Exception as exc:
                    self.results[name] = f"❌ ERROR: {exc}"
                    log_event({"type": "test_error", "test": name, "error": str(exc)})

            return all("PASS" in str(r) for r in self.results.values())
        finally:
            timer.cancel()
            self._teardown()
            self._print_summary()

    def _timeout_handler(self) -> None:
        print("\nTimeout triggered")
        self.timeout_event.set()

    def _test_pre_check(self) -> bool:
        from test_pre_check import test_pre_check
        return test_pre_check()

    def _test_kill_switch(self) -> bool:
        from test_kill_switch import test_kill_switch
        return test_kill_switch()

    def _test_balance(self) -> bool:
        from test_live_balance import test_live_balance
        return test_live_balance(self.execution)

    def _test_buy(self) -> bool:
        from test_live_buy import test_live_buy
        result = test_live_buy(self.execution, self.fee_rates)
        return result.get("status") == "passed"

    def _test_sell(self) -> bool:
        from test_live_sell import test_live_sell
        result = test_live_sell(self.execution, self.fee_rates)
        return result.get("status") in ["passed", "skipped"]

    def _teardown(self) -> None:
        print("\n" + "=" * 60)
        print("TEARDOWN: cancel open orders")
        print("=" * 60)
        if self.execution is None:
            print("⚠️ No execution instance - skip")
            return
        result = cancel_all_open_orders(
            self.execution,
            market=self.market,
            since_ts=self.test_start_time,
        )
        verified = verify_no_open_orders(
            self.execution,
            market=self.market,
            since_ts=self.test_start_time,
        )
        self.results["Teardown"] = "✅ PASS" if verified else "⚠️ WARN"
        log_event(
            {
                "type": "teardown",
                "cancelled": result["cancelled"],
                "failed": result["failed"],
                "verified": verified,
            }
        )

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, result in self.results.items():
            print(f"  {name}: {result}")
        passed = sum(1 for r in self.results.values() if "PASS" in str(r))
        total = len(self.results)
        print(f"\n  Result: {passed}/{total} PASS")
        print(f"  Duration: {(datetime.now(timezone.utc) - self.test_start_time).total_seconds():.1f}s")
        log_event(
            {
                "type": "test_complete",
                "results": {k: str(v) for k, v in self.results.items()},
                "passed": passed,
                "total": total,
            }
        )


if __name__ == "__main__":
    runner = LiveTestRunner()
    success = runner.run()
    sys.exit(0 if success else 1)
