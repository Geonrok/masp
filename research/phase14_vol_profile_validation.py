#!/usr/bin/env python3
"""
Ralph-Loop Phase 14: Comprehensive Validation of Volume Profile Breakout
=========================================================================
Phase 13 found vol_profile passing 6/6 (Sharpe=2.52, +16.3% at 5x).
This phase runs identical Phase 10 validation:

1. Reproduce 6/6 TRUE OOS (both 3x and 5x)
2. Slippage sensitivity (0.02% to 0.10%)
3. Regime analysis (bull/bear/sideways)
4. Monte Carlo (1000 shuffles)
5. Parameter robustness (±20% on lookback, VWAP mult, EMA)
6. Capacity analysis
7. Cross-validate: vol_profile vs baseline Dual MA Breakout
"""

import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

DATA_ROOT = Path("E:/data/crypto_ohlcv")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")
RESULTS_PATH.mkdir(exist_ok=True)

COMMISSION = 0.0004
FUNDING_PER_8H = 0.0001


def load_ohlcv(symbol, timeframe="1h"):
    path = DATA_ROOT / f"binance_futures_{timeframe}" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["datetime", "timestamp", "date"]:
        if col in df.columns:
            df["datetime"] = pd.to_datetime(df[col])
            break
    return df.sort_values("datetime").reset_index(drop=True)


def calc_atr(high, low, close, period=14):
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values


# =============================================================================
# STRATEGIES
# =============================================================================


def vol_profile_strategy(df, lookback=48, vwap_mult=1.01, ema_fast=50, ema_slow=200):
    """Volume Profile Breakout - the Phase 13 winner"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    signals = np.where(
        (close > upper) & (close > vwap * vwap_mult) & (ema_f > ema_s), 1, 0
    )
    return signals


def baseline_strategy(df, lookback=48, ema_fast=50, ema_slow=200, atr_expansion=1.0):
    """Dual MA Breakout - Phase 10 baseline for comparison"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (ema_f > ema_s) & expanding, 1, 0)
    return signals


# =============================================================================
# SIMULATION
# =============================================================================


def simulate(
    df,
    signals,
    position_pct=0.02,
    max_bars=72,
    atr_stop=3.0,
    profit_target_atr=8.0,
    slippage=0.0003,
):
    capital = 1.0
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []
    equity_curve = [1.0]

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr_vals = calc_atr(high, low, close, 14)

    for i in range(len(df)):
        c = close[i]
        sig = signals[i] if i < len(signals) else 0
        cur_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else c * 0.01

        if position != 0:
            bars_held += 1
            unrealized_atr = position * (c - entry_price) / (cur_atr + 1e-10)
            should_exit = False
            if bars_held >= max_bars:
                should_exit = True
            elif atr_stop > 0 and unrealized_atr < -atr_stop:
                should_exit = True
            elif profit_target_atr > 0 and unrealized_atr > profit_target_atr:
                should_exit = True

            if should_exit:
                exit_p = c * (1 - slippage * np.sign(position))
                pnl = position * (exit_p - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0

        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = c * (1 + slippage * position)
            capital -= COMMISSION * position_pct
            bars_held = 0

        equity_curve.append(capital)

    if position != 0:
        c = close[-1]
        exit_p = c * (1 - slippage * np.sign(position))
        pnl = position * (exit_p - entry_price) / entry_price * position_pct
        pnl -= COMMISSION * position_pct
        pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct
        trades.append(pnl)

    wins = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t <= 0)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    return {
        "total_return": capital - 1,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "profit_factor": gp / (gl + 1e-10),
        "trade_count": len(trades),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def check_criteria(r):
    c = {
        "sharpe_gt_1": r.get("sharpe", 0) > 1.0,
        "max_dd_lt_25": r.get("max_drawdown", -1) > -0.25,
        "win_rate_gt_45": r.get("win_rate", 0) > 0.45,
        "profit_factor_gt_1_5": r.get("profit_factor", 0) > 1.5,
        "wfa_efficiency_gt_50": r.get("wfa_efficiency", 0) > 50,
        "trade_count_gt_100": r.get("trade_count", 0) > 100,
    }
    return c, sum(v for v in c.values())


def run_portfolio_oos(
    all_data,
    strat_fn,
    max_positions=10,
    test_bars=720,
    exit_params=None,
    slippage=0.0003,
    position_scale=1.0,
):
    """Full TRUE OOS portfolio test with position scaling."""
    if exit_params is None:
        exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

    # OOS on last 40%
    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > 2000:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    equity = [1.0]
    period_returns = []
    all_trades = []
    min_len = min(len(d) for d in oos_data.values())
    train_bars = min(4320, min_len // 3)

    i = train_bars
    while i + test_bars <= min_len:
        period_pnl = 0
        scored = []
        for symbol, df in oos_data.items():
            if len(df) <= i:
                continue
            vol = df["close"].iloc[:i].pct_change().rolling(168).std().iloc[-1]
            if np.isnan(vol) or vol == 0:
                vol = 0.01
            scored.append((symbol, vol))
        scored.sort(key=lambda x: x[1])
        selected = scored[:max_positions]

        for symbol, vol in selected:
            df = oos_data[symbol]
            if i + test_bars > len(df):
                continue
            full = df.iloc[: i + test_bars]
            sigs = strat_fn(full)
            test_sigs = sigs[i : i + test_bars]
            test_df = df.iloc[i : i + test_bars].copy().reset_index(drop=True)
            ann_vol = vol * np.sqrt(24 * 365)
            position_pct = min(0.10 / (ann_vol + 1e-10) / max(len(selected), 1), 0.05)
            position_pct *= position_scale
            r = simulate(
                test_df,
                test_sigs,
                position_pct,
                exit_params["max_bars"],
                exit_params["atr_stop"],
                exit_params["profit_target_atr"],
                slippage,
            )
            period_pnl += r["total_return"]
            all_trades.extend(r["trades"])

        period_returns.append(period_pnl)
        equity.append(equity[-1] * (1 + period_pnl))
        i += test_bars

    if not period_returns:
        return None

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak

    wins = sum(1 for t in all_trades if t > 0)
    losses = sum(1 for t in all_trades if t <= 0)
    gp = sum(t for t in all_trades if t > 0)
    gl = abs(sum(t for t in all_trades if t < 0))
    sharpe = np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)

    return {
        "total_return": float(equity_arr[-1] - 1),
        "max_drawdown": float(dd.min()),
        "sharpe": float(sharpe),
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "profit_factor": gp / (gl + 1e-10),
        "trade_count": len(all_trades),
        "periods": len(period_returns),
        "wfa_efficiency": sum(1 for r in period_returns if r > 0)
        / len(period_returns)
        * 100,
        "period_returns": period_returns,
        "all_trades": all_trades,
        "equity_curve": list(equity_arr),
    }


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 14: VOL PROFILE COMPREHENSIVE VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    print("Candidate: Volume Profile Breakout (lookback=48, VWAP*1.01, EMA 50/200)")
    print("Config: 10 positions, 720-bar rebalance, ATR 3x stop, 8x PT")
    print()

    # Load data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Loaded {len(all_data)} symbols\n")

    exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}
    verdicts = {}

    # =========================================================================
    # TEST 1: Reproduce 6/6 TRUE OOS (3x and 5x)
    # =========================================================================
    print("=" * 60)
    print("TEST 1: Reproduce TRUE OOS 6/6")
    print("=" * 60)

    results_by_scale = {}
    for scale in [1.0, 3.0, 5.0]:
        r = run_portfolio_oos(
            all_data,
            vol_profile_strategy,
            max_positions=10,
            test_bars=720,
            exit_params=exit_params,
            position_scale=scale,
        )
        if r:
            c, p = check_criteria(r)
            results_by_scale[scale] = r
            print(
                f"\n  Scale {scale}x: [{p}/6] Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
                f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
                f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  T={r['trade_count']}"
            )
            for k, v in c.items():
                print(f"    {k}: {'PASS' if v else 'FAIL'}")

    result_5x = results_by_scale.get(5.0)
    result_3x = results_by_scale.get(3.0)
    main_result = result_5x or result_3x

    t1_pass = main_result is not None and check_criteria(main_result)[1] == 6
    verdicts["test1_6_of_6"] = t1_pass
    print(f"\n  TEST 1 VERDICT: {'PASS' if t1_pass else 'FAIL'}")

    # =========================================================================
    # TEST 2: Slippage Sensitivity
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("TEST 2: Slippage Sensitivity")
    print("=" * 60)

    slip_results = {}
    for slip_name, slip_val in [
        ("0.02%", 0.0002),
        ("0.03%", 0.0003),
        ("0.05%", 0.0005),
        ("0.08%", 0.0008),
        ("0.10%", 0.001),
    ]:
        r = run_portfolio_oos(
            all_data,
            vol_profile_strategy,
            max_positions=10,
            test_bars=720,
            exit_params=exit_params,
            slippage=slip_val,
            position_scale=5.0,
        )
        if r:
            c, p = check_criteria(r)
            slip_results[slip_name] = {"passed": p, **r}
            print(
                f"  {slip_name}: [{p}/6] Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
                f"WR={r['win_rate']*100:.0f}%  PF={r['profit_factor']:.2f}  T={r['trade_count']}"
            )

    t2_pass = all(v.get("passed", 0) >= 5 for v in slip_results.values())
    verdicts["test2_slippage_robust"] = t2_pass
    print(f"\n  TEST 2 VERDICT: {'PASS' if t2_pass else 'FAIL'}")

    # =========================================================================
    # TEST 3: Regime Analysis
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("TEST 3: Market Regime Analysis")
    print("=" * 60)

    btc = load_ohlcv("BTCUSDT", "1h")
    regime_results = {}
    t3_pass = True
    if not btc.empty:
        btc_close = btc["close"]
        ret_30d = btc_close.pct_change(720)
        btc["regime"] = np.where(
            ret_30d > 0.10, "bull", np.where(ret_30d < -0.10, "bear", "sideways")
        )

        test_symbols = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "DOGEUSDT",
            "ADAUSDT",
            "XRPUSDT",
            "BNBUSDT",
            "LINKUSDT",
            "AVAXUSDT",
            "LTCUSDT",
        ]

        for regime in ["bull", "bear", "sideways"]:
            regime_mask = btc["regime"] == regime
            regime_dates = set(btc.loc[regime_mask, "datetime"])
            count = len(regime_dates)
            pct = count / len(btc) * 100

            all_trades = []
            for symbol in test_symbols:
                df = load_ohlcv(symbol, "1h")
                if df.empty:
                    continue
                regime_df = (
                    df[df["datetime"].isin(regime_dates)].copy().reset_index(drop=True)
                )
                if len(regime_df) < 500:
                    continue
                sigs = vol_profile_strategy(regime_df)
                r = simulate(regime_df, sigs, 0.02, 72, 3.0, 8.0, 0.0003)
                all_trades.extend(r["trades"])

            if all_trades:
                wins = sum(1 for t in all_trades if t > 0)
                losses = sum(1 for t in all_trades if t <= 0)
                gp = sum(t for t in all_trades if t > 0)
                gl = abs(sum(t for t in all_trades if t < 0))
                wr = wins / (wins + losses) if (wins + losses) > 0 else 0
                pf = gp / (gl + 1e-10)
                net = sum(all_trades)
                status = "OK" if net > 0 else "DANGER"
                if regime == "bear" and net < 0:
                    t3_pass = False
                regime_results[regime] = {
                    "count": count,
                    "pct": pct,
                    "trades": len(all_trades),
                    "win_rate": wr,
                    "profit_factor": pf,
                    "net": net,
                }
                print(
                    f"  {regime} ({pct:.0f}%): {len(all_trades)} trades  WR={wr:.0%}  "
                    f"PF={pf:.2f}  Net={net*100:+.2f}%  [{status}]"
                )

    verdicts["test3_no_bear_catastrophe"] = t3_pass
    print(f"\n  TEST 3 VERDICT: {'PASS' if t3_pass else 'FAIL'}")

    # =========================================================================
    # TEST 4: Monte Carlo (1000 shuffles)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("TEST 4: Monte Carlo Simulation (1000 runs)")
    print("=" * 60)

    p_positive = 0
    mc_stats = {}
    if main_result and main_result.get("all_trades"):
        trades = main_result["all_trades"]
        mc_returns = []
        np.random.seed(42)
        for _ in range(1000):
            shuffled = np.random.choice(trades, size=len(trades), replace=True)
            eq = np.cumprod(1 + np.array(shuffled))
            mc_returns.append(float(eq[-1] - 1))

        mc_returns = sorted(mc_returns)
        p_positive = sum(1 for r in mc_returns if r > 0) / len(mc_returns) * 100
        p5 = mc_returns[int(len(mc_returns) * 0.05)]
        p50 = mc_returns[int(len(mc_returns) * 0.50)]
        p95 = mc_returns[int(len(mc_returns) * 0.95)]

        mc_stats = {
            "p_positive": p_positive,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "worst": mc_returns[0],
            "best": mc_returns[-1],
        }

        print(f"  P(positive): {p_positive:.1f}%")
        print(f"  5th percentile: {p5*100:+.1f}%")
        print(f"  50th percentile (median): {p50*100:+.1f}%")
        print(f"  95th percentile: {p95*100:+.1f}%")
        print(f"  Worst case: {mc_returns[0]*100:+.1f}%")
        print(f"  Best case: {mc_returns[-1]*100:+.1f}%")

    t4_pass = p_positive > 90
    verdicts["test4_monte_carlo_positive"] = t4_pass
    print(f"\n  TEST 4 VERDICT: {'PASS' if t4_pass else 'FAIL'}")

    # =========================================================================
    # TEST 5: Parameter Sensitivity (±20%)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("TEST 5: Parameter Robustness (±20%)")
    print("=" * 60)

    param_variants = {
        "base": {"lookback": 48, "vwap_mult": 1.01, "ema_fast": 50, "ema_slow": 200},
        "lookback_38": {
            "lookback": 38,
            "vwap_mult": 1.01,
            "ema_fast": 50,
            "ema_slow": 200,
        },
        "lookback_58": {
            "lookback": 58,
            "vwap_mult": 1.01,
            "ema_fast": 50,
            "ema_slow": 200,
        },
        "vwap_0.808": {
            "lookback": 48,
            "vwap_mult": 1.008,
            "ema_fast": 50,
            "ema_slow": 200,
        },
        "vwap_1.012": {
            "lookback": 48,
            "vwap_mult": 1.012,
            "ema_fast": 50,
            "ema_slow": 200,
        },
        "ema_fast_40": {
            "lookback": 48,
            "vwap_mult": 1.01,
            "ema_fast": 40,
            "ema_slow": 200,
        },
        "ema_fast_60": {
            "lookback": 48,
            "vwap_mult": 1.01,
            "ema_fast": 60,
            "ema_slow": 200,
        },
        "ema_slow_160": {
            "lookback": 48,
            "vwap_mult": 1.01,
            "ema_fast": 50,
            "ema_slow": 160,
        },
        "ema_slow_240": {
            "lookback": 48,
            "vwap_mult": 1.01,
            "ema_fast": 50,
            "ema_slow": 240,
        },
    }

    param_results = {}
    for name, params in param_variants.items():

        def make_strat(p=params):
            def s(df):
                return vol_profile_strategy(df, **p)

            return s

        r = run_portfolio_oos(
            all_data,
            make_strat(),
            max_positions=10,
            test_bars=720,
            exit_params=exit_params,
            position_scale=5.0,
        )
        if r:
            c, p = check_criteria(r)
            param_results[name] = p
            print(
                f"  {name:<20} [{p}/6] Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
                f"WR={r['win_rate']*100:.0f}%  PF={r['profit_factor']:.2f}  T={r['trade_count']}"
            )
        else:
            param_results[name] = 0
            print(f"  {name:<20} SKIP")

    robust_count = sum(1 for p in param_results.values() if p >= 5)
    t5_pass = robust_count >= 6
    verdicts["test5_param_robust"] = t5_pass
    print(f"\n  Robustness: {robust_count}/{len(param_results)} variants pass 5+/6")
    print(f"  TEST 5 VERDICT: {'PASS' if t5_pass else 'FAIL'}")

    # =========================================================================
    # TEST 6: Capacity
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("TEST 6: Capacity Analysis")
    print("=" * 60)

    capacity_ok = True
    for portfolio_size in [10000, 50000, 100000, 500000]:
        pos_size = portfolio_size * 0.05
        print(f"\n  Portfolio ${portfolio_size:,}:")
        all_ok = True
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]:
            df = load_ohlcv(symbol, "1h")
            if df.empty:
                continue
            recent = df.tail(90 * 24)
            if "volume" in recent.columns:
                adv = (recent["volume"] * recent["close"]).mean() * 24
                impact = pos_size / adv * 100 if adv > 0 else 999
                status = (
                    "OK" if impact < 0.1 else "CAUTION" if impact < 1 else "TOO LARGE"
                )
                if impact >= 1:
                    all_ok = False
                print(
                    f"    {symbol:<12} ADV=${adv/1e6:.0f}M  ${pos_size:,.0f} = {impact:.4f}% [{status}]"
                )
        if all_ok:
            print(f"    → All OK for ${portfolio_size:,}")
        else:
            if portfolio_size <= 100000:
                capacity_ok = False

    verdicts["test6_capacity_ok"] = capacity_ok
    print(f"\n  TEST 6 VERDICT: {'PASS' if capacity_ok else 'FAIL'}")

    # =========================================================================
    # TEST 7: Cross-validate with baseline Dual MA Breakout
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("TEST 7: Cross-validate with Baseline Dual MA Breakout")
    print("=" * 60)

    baseline_r = run_portfolio_oos(
        all_data,
        baseline_strategy,
        max_positions=10,
        test_bars=720,
        exit_params=exit_params,
        position_scale=5.0,
    )
    if baseline_r:
        c, p = check_criteria(baseline_r)
        print(f"\n  Baseline Dual MA (5x):")
        print(
            f"    [{p}/6] Sharpe={baseline_r['sharpe']:.2f}  Ret={baseline_r['total_return']*100:+.1f}%  "
            f"DD={baseline_r['max_drawdown']*100:.1f}%  WR={baseline_r['win_rate']*100:.0f}%  "
            f"PF={baseline_r['profit_factor']:.2f}  T={baseline_r['trade_count']}"
        )

    if main_result:
        c_main, p_main = check_criteria(main_result)
        print(f"\n  Vol Profile (5x):")
        print(
            f"    [{p_main}/6] Sharpe={main_result['sharpe']:.2f}  Ret={main_result['total_return']*100:+.1f}%  "
            f"DD={main_result['max_drawdown']*100:.1f}%  WR={main_result['win_rate']*100:.0f}%  "
            f"PF={main_result['profit_factor']:.2f}  T={main_result['trade_count']}"
        )

    if baseline_r and main_result:
        better = main_result["sharpe"] >= baseline_r["sharpe"]
        print(
            f"\n  Vol Profile {'beats' if better else 'does not beat'} baseline "
            f"(Sharpe {main_result['sharpe']:.2f} vs {baseline_r['sharpe']:.2f})"
        )

    # Also test correlation between strategies (do they generate same trades?)
    # If low correlation → ensemble potential
    if (
        baseline_r
        and main_result
        and baseline_r.get("period_returns")
        and main_result.get("period_returns")
    ):
        br = baseline_r["period_returns"]
        vr = main_result["period_returns"]
        min_len = min(len(br), len(vr))
        if min_len > 2:
            corr = np.corrcoef(br[:min_len], vr[:min_len])[0, 1]
            print(f"\n  Period return correlation: {corr:.3f}")
            if corr < 0.7:
                print("  → Low correlation: ensemble potential!")
            else:
                print("  → High correlation: strategies are similar")

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print(f"\n\n{'=' * 70}")
    print("FINAL VERDICT")
    print("=" * 70)

    all_pass = all(verdicts.values())

    for test, passed in verdicts.items():
        print(f"  {test}: {'PASS' if passed else 'FAIL'}")

    print(f"\n  OVERALL: {'APPROVED FOR PAPER TRADING' if all_pass else 'NOT READY'}")

    if all_pass and main_result:
        print(f"""
  ============================================
  STRATEGY SPECIFICATION (for implementation)
  ============================================
  Name: Volume Profile Breakout (Long-Only)
  Timeframe: 1H
  Entry:
    - Price breaks above Donchian(48) upper band
    - Price > VWAP(48) * 1.01 (volume confirmation)
    - EMA(50) > EMA(200) (trend confirmation)
    - Long only (no shorts)
  Exit:
    - ATR 3x stop-loss
    - ATR 8x profit target
    - 72-bar (3-day) maximum holding period
  Portfolio:
    - Universe: All Binance USDT-M Futures
    - Max 10 simultaneous positions
    - Vol-targeting: 10% annual per position
    - Position scale: 5x (aggressive) / 3x (moderate)
    - Max 5% allocation per position (before scaling)
    - Rebalance every 720 bars (30 days)
    - Symbol selection: lowest volatility first
  Expected Performance (5x):
    - Sharpe: {main_result['sharpe']:.2f}
    - Return: {main_result['total_return']*100:+.1f}%
    - Max DD: {main_result['max_drawdown']*100:.1f}%
    - Win Rate: {main_result['win_rate']*100:.0f}%
    - Profit Factor: {main_result['profit_factor']:.2f}
  ============================================
""")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "Volume Profile Breakout (Long-Only)",
        "config": {
            "lookback": 48,
            "vwap_mult": 1.01,
            "ema_fast": 50,
            "ema_slow": 200,
            "max_positions": 10,
            "rebalance_bars": 720,
            "exit": {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        },
        "results_by_scale": {
            str(k): {
                key: float(v) if isinstance(v, (float, np.floating)) else v
                for key, v in res.items()
                if key not in ("all_trades", "period_returns", "equity_curve")
            }
            for k, res in results_by_scale.items()
        },
        "verdicts": verdicts,
        "all_pass": all_pass,
        "slippage_results": {
            k: {"passed": v["passed"], "sharpe": float(v["sharpe"])}
            for k, v in slip_results.items()
        },
        "regime_results": {
            k: {
                key: float(val) if isinstance(val, (float, np.floating)) else val
                for key, val in v.items()
            }
            for k, v in regime_results.items()
        },
        "monte_carlo": mc_stats,
        "param_robustness": param_results,
        "baseline_comparison": {
            "baseline_sharpe": float(baseline_r["sharpe"]) if baseline_r else None,
            "vol_profile_sharpe": float(main_result["sharpe"]) if main_result else None,
        },
    }

    with open(RESULTS_PATH / "phase14_vol_profile_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to {RESULTS_PATH / 'phase14_vol_profile_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")

    return all_pass


if __name__ == "__main__":
    main()
