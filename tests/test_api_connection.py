"""
API connection test script (extended).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    print("=" * 60)
    print("API Connection Test (Extended)")
    print("=" * 60)

    results = []

    # 1. .env file check
    print("\n[1] .env file check")
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        print(f"  OK: {env_path}")
        results.append(True)
    else:
        print("  NOT FOUND")
        results.append(False)

    # 2. Upbit connection test
    print("\n[2] Upbit connection test")
    try:
        from libs.adapters.real_upbit_spot import UpbitSpotMarketData

        adapter = UpbitSpotMarketData()
        quote = adapter.get_quote("BTC/KRW")
        if quote:
            print(f"  OK - BTC/KRW: {quote.last:,.0f} KRW")
            results.append(True)
        else:
            print("  FAIL")
            results.append(False)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results.append(False)

    # 3. Kill-Switch
    print("\n[3] Kill-Switch status")
    try:
        from pathlib import Path

        ks_file = "kill_switch.txt"
        if Path(ks_file).exists():
            print("  ACTIVE - trading blocked")
            results.append(False)
        else:
            print("  INACTIVE (normal)")
            results.append(True)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results.append(False)

    # 4. Trading Limits (dotenv)
    print("\n[4] Trading Limits check")
    try:
        from dotenv import load_dotenv

        load_dotenv()

        max_order = os.getenv("MAX_ORDER_VALUE_KRW", "1000000")
        max_position = os.getenv("MAX_POSITION_PCT", "0.10")
        max_daily_loss = os.getenv("MAX_DAILY_LOSS_KRW", "100000")

        print(f"  Max Order: {int(max_order):,} KRW")
        print(f"  Max Position: {float(max_position) * 100:.0f}%")
        print(f"  Max Daily Loss: {int(max_daily_loss):,} KRW")
        print("  PASS")
        results.append(True)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results.append(False)

    # 5. Adapter Mode
    print("\n[5] Adapter Mode check")
    try:
        from dotenv import load_dotenv

        load_dotenv()

        adapter_mode = os.getenv("ADAPTER_MODE", "paper")

        if adapter_mode.lower() == "live":
            print("  LIVE MODE - trading enabled")
            print("  CAUTION: real funds will be used")
        else:
            print("  PAPER MODE - simulated trading")

        print(f"  Mode: {adapter_mode}")
        results.append(True)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    test_names = [
        ".env file",
        "Upbit connection",
        "Kill-Switch",
        "Trading Limits",
        "Adapter Mode",
    ]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"  {i+1}. {name}: {status}")

    print(f"\nTotal: {passed}/{total} PASS")

    if passed == total:
        print("\nAll Tests PASSED")
        print("Ready for trading")
    else:
        print(f"\nWARNING: {total - passed} test(s) failed")

    print("\n" + "=" * 60)
    print("Next actions:")
    print("  1. Fill .env with real API keys")
    print("  2. Check ADAPTER_MODE (paper recommended)")
    print("  3. Verify trading limits")
    print("=" * 60)

    return 0 if passed >= 4 else 1  # PASS if 4/5 or more


if __name__ == "__main__":
    sys.exit(main())
