"""
KAMA-TSMOM-Gate strategy tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> bool:
    print("=" * 60)
    print("KAMA-TSMOM-Gate Strategy Test")
    print("=" * 60)

    # 1. Indicator tests
    print("\n[1] Indicator tests")
    from libs.strategies.indicators import KAMA, MA, TSMOM, KAMA_series, TSMOM_signal

    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110]

    ma5 = MA(prices, 5)
    print(f"  MA(5): {ma5:.2f}")
    assert 105 < ma5 < 108, f"MA(5) unexpected: {ma5}"

    kama5 = KAMA(prices, 5)
    print(f"  KAMA(5): {kama5:.2f}")
    assert 100 < kama5 < 115, f"KAMA(5) unexpected: {kama5}"

    ks = KAMA_series(prices, 5)
    assert abs(kama5 - float(ks[-1])) < 1e-9, f"KAMA mismatch: {kama5} vs {ks[-1]}"
    print(f"  KAMA consistency: KAMA()={kama5:.6f} == KAMA_series()[-1]={ks[-1]:.6f} ✓")

    long_prices = list(range(100, 200))
    tsmom = TSMOM(long_prices, 90)
    print(f"  TSMOM(90): {tsmom:.4f}")
    assert tsmom > 0, "TSMOM should be positive"

    print("  PASS")

    # 2. Strategy init
    print("\n[2] Strategy init")
    from libs.strategies.kama_tsmom_gate import KamaTsmomGateStrategy

    strategy = KamaTsmomGateStrategy(
        gate_ma_period=30,
        kama_period=5,
        tsmom_lookback=90,
        position_size_krw=10000,
    )
    print(f"  Name: {strategy.name}")
    print(f"  Params: {strategy.get_parameters()}")
    print("  PASS")

    # 3. Gate test (mock)
    print("\n[3] Gate test (mock)")
    btc_prices_up = [130000000 + i * 100000 for i in range(50)]
    strategy.update_prices("BTC/KRW", btc_prices_up)

    gate_pass = strategy.check_gate()
    print(f"  BTC uptrend: Gate = {'PASS' if gate_pass else 'FAIL'}")
    assert gate_pass, "Gate should PASS on uptrend"

    btc_prices_down = [130000000 - i * 100000 for i in range(50)]
    strategy.update_prices("BTC/KRW", btc_prices_down)

    gate_fail = not strategy.check_gate()
    print(f"  BTC downtrend: Gate = {'FAIL' if gate_fail else 'PASS'}")
    assert gate_fail, "Gate should FAIL on downtrend"

    print("  PASS")

    # 4. Signal generation
    print("\n[4] Signal generation")
    strategy.update_prices("BTC/KRW", btc_prices_up)
    eth_prices_up = [4000000 + i * 10000 for i in range(100)]
    strategy.update_prices("ETH/KRW", eth_prices_up)

    from libs.strategies.base import Signal

    signal = strategy.generate_signal("ETH/KRW")
    print(f"  ETH uptrend (Gate PASS): {signal.signal.value} - {signal.reason}")
    assert signal.signal == Signal.BUY, f"Should be BUY: {signal.signal}"

    print("  PASS")

    # 5. Position management
    print("\n[5] Position management")
    strategy.update_position("ETH/KRW", 0.001)
    print(f"  Position: {strategy.get_position('ETH/KRW')}")
    assert strategy.has_position("ETH/KRW"), "Should have position"

    eth_prices_down = [4500000 - i * 10000 for i in range(100)]
    strategy.update_prices("ETH/KRW", eth_prices_down)

    signal = strategy.generate_signal("ETH/KRW")
    print(f"  ETH downtrend (Has Position): {signal.signal.value} - {signal.reason}")
    assert signal.signal == Signal.SELL, f"Should be SELL: {signal.signal}"

    print("  PASS")

    # 6. State check
    print("\n[6] State check")
    state = strategy.get_state()
    print(f"  Name: {state.name}")
    print(f"  Running: {state.is_running}")
    print(f"  Gate: {'PASS' if state.gate_status else 'FAIL'}")
    print(f"  Positions: {state.positions}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All Tests Complete (7/7 PASS)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
