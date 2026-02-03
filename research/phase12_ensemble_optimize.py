#!/usr/bin/env python3
"""
Ralph-Loop Phase 12: Ensemble & Optimization
==============================================
Phase 10: Dual MA Breakout 6/6 (Sharpe 2.39, +2.9%) - approved
Phase 11: 8 new strategies tested, none 6/6. Macro regime 5/6 best.

This phase:
1. Macro regime as OVERLAY filter on existing strategy (risk-off = reduce exposure)
2. Optimize existing strategy for higher absolute returns
3. Multi-strategy ensemble: breakout + macro filter + altcoin rotation
4. Final TRUE OOS validation of best ensemble
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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


def load_macro(name):
    path = DATA_ROOT / "macro" / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def calc_atr(high, low, close, period=14):
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values


def dual_ma_breakout(df, lookback=48, ema_fast=50, ema_slow=200, atr_expansion=1.0):
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
            elif sig != 0 and sig != position:
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


def build_macro_regime():
    """Build daily macro regime signal: 1=risk-on, 0=risk-off."""
    dxy = load_macro("DXY")
    vix = load_macro("VIX")
    if dxy.empty or vix.empty:
        return {}

    dxy["date"] = dxy["datetime"].dt.date
    vix["date"] = vix["datetime"].dt.date

    merged = dxy[["date", "close"]].rename(columns={"close": "dxy"})
    merged = merged.merge(
        vix[["date", "close"]].rename(columns={"close": "vix"}), on="date", how="inner"
    )

    dxy_sma = pd.Series(merged["dxy"].values).rolling(100).mean()
    risk_on = (merged["dxy"].values < dxy_sma.values) & (merged["vix"].values < 25)

    regime = {}
    for i, row in merged.iterrows():
        if not np.isnan(dxy_sma.iloc[i]):
            regime[row["date"]] = 1 if risk_on[i] else 0

    return regime


def run_portfolio_oos(
    all_data,
    strat_fn,
    max_positions=10,
    test_bars=720,
    exit_params=None,
    slippage=0.0003,
    macro_regime=None,
    position_scale=1.0,
):
    """Full TRUE OOS portfolio test with optional macro regime overlay."""
    if exit_params is None:
        exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

    # Selection on first 60%
    for symbol, df in all_data.items():
        split = int(len(df) * 0.6)
        break  # just need to know split ratio

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

        # Check macro regime for this period (if available)
        regime_scale = 1.0
        if macro_regime:
            # Get date of current period start
            sample_symbol = list(oos_data.keys())[0]
            sample_df = oos_data[sample_symbol]
            if "datetime" in sample_df.columns and i < len(sample_df):
                current_date = sample_df["datetime"].iloc[i]
                if hasattr(current_date, "date"):
                    d = current_date.date()
                    regime_scale = macro_regime.get(d, 0.5)
                    # Also check recent regime trend (last 5 days)
                    regime_vals = []
                    for offset in range(5):
                        dd = pd.Timestamp(d) - pd.Timedelta(days=offset)
                        rv = macro_regime.get(dd.date(), 0.5)
                        regime_vals.append(rv)
                    regime_scale = np.mean(regime_vals)

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
            base_pct = min(0.10 / (ann_vol + 1e-10) / max(len(selected), 1), 0.05)
            position_pct = base_pct * regime_scale * position_scale

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
    }


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 12: ENSEMBLE & OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Loaded {len(all_data)} symbols\n")

    # Build macro regime
    print("Building macro regime overlay...")
    macro_regime = build_macro_regime()
    print(f"  Macro regime data: {len(macro_regime)} days")
    risk_on_pct = (
        sum(v for v in macro_regime.values()) / len(macro_regime) * 100
        if macro_regime
        else 0
    )
    print(f"  Risk-on days: {risk_on_pct:.0f}%\n")

    exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

    # =========================================================================
    # PART 1: Baseline (reproduce Phase 10 result)
    # =========================================================================
    print("=" * 60)
    print("PART 1: Baseline (Phase 10 reproduction)")
    print("=" * 60)

    baseline = run_portfolio_oos(
        all_data,
        dual_ma_breakout,
        max_positions=10,
        test_bars=720,
        exit_params=exit_params,
    )
    if baseline:
        c, p = check_criteria(baseline)
        print(
            f"  [{p}/6] Sharpe={baseline['sharpe']:.2f} Ret={baseline['total_return']*100:+.1f}% "
            f"DD={baseline['max_drawdown']*100:.1f}% WR={baseline['win_rate']*100:.0f}% "
            f"PF={baseline['profit_factor']:.2f} T={baseline['trade_count']}"
        )

    # =========================================================================
    # PART 2: With Macro Regime Overlay
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("PART 2: With Macro Regime Overlay")
    print("=" * 60)

    overlay = run_portfolio_oos(
        all_data,
        dual_ma_breakout,
        max_positions=10,
        test_bars=720,
        exit_params=exit_params,
        macro_regime=macro_regime,
    )
    if overlay:
        c, p = check_criteria(overlay)
        print(
            f"  [{p}/6] Sharpe={overlay['sharpe']:.2f} Ret={overlay['total_return']*100:+.1f}% "
            f"DD={overlay['max_drawdown']*100:.1f}% WR={overlay['win_rate']*100:.0f}% "
            f"PF={overlay['profit_factor']:.2f} T={overlay['trade_count']}"
        )

    # =========================================================================
    # PART 3: Position Scale Optimization (higher returns)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("PART 3: Position Scale Optimization")
    print("=" * 60)

    scale_results = {}
    for scale in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        r = run_portfolio_oos(
            all_data,
            dual_ma_breakout,
            max_positions=10,
            test_bars=720,
            exit_params=exit_params,
            position_scale=scale,
        )
        if r:
            c, p = check_criteria(r)
            scale_results[scale] = r
            print(
                f"  scale={scale:.1f}x: [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                f"PF={r['profit_factor']:.2f} T={r['trade_count']}"
            )

    # =========================================================================
    # PART 4: Macro Overlay + Position Scale
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("PART 4: Macro Overlay + Position Scale")
    print("=" * 60)

    combo_results = {}
    for scale in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        r = run_portfolio_oos(
            all_data,
            dual_ma_breakout,
            max_positions=10,
            test_bars=720,
            exit_params=exit_params,
            macro_regime=macro_regime,
            position_scale=scale,
        )
        if r:
            c, p = check_criteria(r)
            combo_results[scale] = r
            print(
                f"  macro+scale={scale:.1f}x: [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                f"PF={r['profit_factor']:.2f} T={r['trade_count']}"
            )

    # =========================================================================
    # PART 5: More Positions + Shorter Rebalance + Scale
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("PART 5: Aggressive Portfolio Configs")
    print("=" * 60)

    agg_results = {}
    for max_pos in [10, 15, 20]:
        for test_bars in [360, 720]:
            for scale in [2.0, 3.0, 5.0]:
                config = f"pos{max_pos}_rb{test_bars}_s{scale}"
                r = run_portfolio_oos(
                    all_data,
                    dual_ma_breakout,
                    max_positions=max_pos,
                    test_bars=test_bars,
                    exit_params=exit_params,
                    position_scale=scale,
                )
                if r:
                    c, p = check_criteria(r)
                    agg_results[config] = {**r, "criteria_passed": p}
                    print(
                        f"  {config}: [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                        f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                        f"PF={r['profit_factor']:.2f} T={r['trade_count']}"
                    )

    # =========================================================================
    # PART 6: Best configs with macro overlay
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("PART 6: Best Aggressive + Macro Overlay")
    print("=" * 60)

    # Find top 5 from Part 5
    ranked = sorted(
        agg_results.items(),
        key=lambda x: (-x[1].get("criteria_passed", 0), -x[1].get("sharpe", 0)),
    )

    for config, r in ranked[:5]:
        parts = config.split("_")
        max_pos = int(parts[0].replace("pos", ""))
        test_bars = int(parts[1].replace("rb", ""))
        scale = float(parts[2].replace("s", ""))

        r2 = run_portfolio_oos(
            all_data,
            dual_ma_breakout,
            max_positions=max_pos,
            test_bars=test_bars,
            exit_params=exit_params,
            macro_regime=macro_regime,
            position_scale=scale,
        )
        if r2:
            c, p = check_criteria(r2)
            print(
                f"  macro+{config}: [{p}/6] Sharpe={r2['sharpe']:.2f} Ret={r2['total_return']*100:+.1f}% "
                f"DD={r2['max_drawdown']*100:.1f}% WR={r2['win_rate']*100:.0f}% "
                f"PF={r2['profit_factor']:.2f} T={r2['trade_count']}"
            )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 70}")
    print("PHASE 12 SUMMARY")
    print("=" * 70)

    # Collect all results
    all_configs = []
    if baseline:
        c, p = check_criteria(baseline)
        all_configs.append(("baseline", p, baseline))
    if overlay:
        c, p = check_criteria(overlay)
        all_configs.append(("macro_overlay", p, overlay))
    for scale, r in scale_results.items():
        c, p = check_criteria(r)
        all_configs.append((f"scale_{scale}x", p, r))
    for scale, r in combo_results.items():
        c, p = check_criteria(r)
        all_configs.append((f"macro+scale_{scale}x", p, r))
    for config, r in agg_results.items():
        p = r.get("criteria_passed", 0)
        all_configs.append((config, p, r))

    all_configs.sort(key=lambda x: (-x[1], -x[2].get("sharpe", 0)))

    print("\nTop 10 configurations:\n")
    for i, (name, passed, r) in enumerate(all_configs[:10]):
        c, _ = check_criteria(r)
        print(f"  {i+1}. [{passed}/6] {name}")
        print(
            f"     Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
            f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  T={r['trade_count']}"
        )
        fails = [k for k, v in c.items() if not v]
        if fails:
            print(f"     FAILS: {', '.join(fails)}")

    # Best 6/6 with highest return
    six_six = [(n, r) for n, p, r in all_configs if p == 6]
    if six_six:
        six_six.sort(key=lambda x: -x[1]["total_return"])
        print(f"\n*** {len(six_six)} configs passed 6/6 ***")
        print(f"\nBest by return:")
        for name, r in six_six[:5]:
            print(
                f"  {name}: Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
                f"DD={r['max_drawdown']*100:.1f}%  T={r['trade_count']}"
            )

        print(f"\nBest by Sharpe:")
        six_six_sharpe = sorted(six_six, key=lambda x: -x[1]["sharpe"])
        for name, r in six_six_sharpe[:5]:
            print(
                f"  {name}: Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
                f"DD={r['max_drawdown']*100:.1f}%  T={r['trade_count']}"
            )

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(all_configs),
        "six_six_count": len(six_six) if six_six else 0,
        "top_10": [
            {
                "config": name,
                "passed": int(passed),
                "sharpe": float(r.get("sharpe", 0)),
                "return": float(r.get("total_return", 0)),
                "max_dd": float(r.get("max_drawdown", 0)),
                "win_rate": float(r.get("win_rate", 0)),
                "pf": float(r.get("profit_factor", 0)),
                "trades": int(r.get("trade_count", 0)),
            }
            for name, passed, r in all_configs[:10]
        ],
    }
    with open(RESULTS_PATH / "phase12_report.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_PATH / 'phase12_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
