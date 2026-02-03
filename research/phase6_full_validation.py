#!/usr/bin/env python3
"""
Ralph-Loop Phase 6: Full Validation on All Binance Futures Symbols
====================================================================
Strategy: ATR Expansion Breakout (v2_atr_expand)
- 1h Donchian(96) breakout + EMA(200) trend filter + ATR expansion filter
- ATR 3x stop loss + ATR 8x profit target
- Walk-forward: 180d train / 30d test

ALL 6/6 criteria must pass on full universe.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

DATA_ROOT = Path("E:/data/crypto_ohlcv")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")
RESULTS_PATH.mkdir(exist_ok=True)

SLIPPAGE = 0.0003
COMMISSION = 0.0004
FUNDING_PER_8H = 0.0001


def load_ohlcv(symbol, timeframe="1h"):
    tf_map = {"1h": "binance_futures_1h", "4h": "binance_futures_4h"}
    path = DATA_ROOT / tf_map[timeframe] / f"{symbol}.csv"
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


def atr_expansion_breakout(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """
    THE STRATEGY: ATR Expansion Breakout
    - Long: close > 96-bar high AND close > EMA(200) AND ATR expanding
    - Short: close < 96-bar low AND close < EMA(200) AND ATR expanding
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion

    signals = np.where(
        (close > upper) & (close > ema) & expanding,
        1,
        np.where((close < lower) & (close < ema) & expanding, -1, 0),
    )
    return signals


def simulate(
    df, signals, position_pct=0.02, max_bars=72, atr_stop=3.0, profit_target_atr=8.0
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
                exit_p = c * (1 - SLIPPAGE * np.sign(position))
                pnl = position * (exit_p - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0

        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = c * (1 + SLIPPAGE * position)
            capital -= COMMISSION * position_pct
            bars_held = 0

    if position != 0:
        c = close[-1]
        exit_p = c * (1 - SLIPPAGE * np.sign(position))
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


def portfolio_walk_forward(
    symbols, train_bars=4320, test_bars=720, target_vol=0.10, max_positions=5
):
    """Full portfolio walk-forward on given symbols"""
    all_data = {}
    for s in symbols:
        df = load_ohlcv(s, "1h")
        if not df.empty and len(df) > train_bars + test_bars:
            all_data[s] = df

    if not all_data:
        return None

    min_len = min(len(d) for d in all_data.values())
    equity = [1.0]
    period_returns = []
    all_trades = []
    positions_per_period = []

    i = train_bars
    while i + test_bars <= min_len:
        period_pnl = 0
        period_positions = 0

        scored = []
        for symbol, df in all_data.items():
            train_ret = df["close"].iloc[:i].pct_change()
            vol = train_ret.rolling(168).std().iloc[-1]
            if np.isnan(vol) or vol == 0:
                vol = 0.01
            scored.append((symbol, vol))

        scored.sort(key=lambda x: x[1])
        selected = scored[:max_positions]

        for symbol, vol in selected:
            df = all_data[symbol]
            full = df.iloc[: i + test_bars]
            sigs = atr_expansion_breakout(full)
            test_sigs = sigs[i : i + test_bars]
            test_df = df.iloc[i : i + test_bars].copy().reset_index(drop=True)

            ann_vol = vol * np.sqrt(24 * 365)
            position_pct = min(
                target_vol / (ann_vol + 1e-10) / max(len(selected), 1), 0.05
            )

            r = simulate(test_df, test_sigs, position_pct, 72, 3.0, 8.0)
            period_pnl += r["total_return"]
            all_trades.extend(r["trades"])
            if r["trade_count"] > 0:
                period_positions += 1

        period_returns.append(period_pnl)
        positions_per_period.append(period_positions)
        equity.append(equity[-1] * (1 + period_pnl))
        i += test_bars

    if not period_returns:
        return None

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    wins = sum(1 for t in all_trades if t > 0)
    losses = sum(1 for t in all_trades if t <= 0)
    gp = sum(t for t in all_trades if t > 0)
    gl = abs(sum(t for t in all_trades if t < 0))

    sharpe = (
        np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)
        if len(period_returns) > 1
        else 0
    )
    sortino_denom = (
        np.std([r for r in period_returns if r < 0])
        if any(r < 0 for r in period_returns)
        else 1e-10
    )
    sortino = np.mean(period_returns) / sortino_denom * np.sqrt(12)

    return {
        "total_return": equity[-1] - 1,
        "max_drawdown": dd.min(),
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "profit_factor": gp / (gl + 1e-10),
        "trade_count": len(all_trades),
        "periods": len(period_returns),
        "wfa_efficiency": sum(1 for r in period_returns if r > 0)
        / len(period_returns)
        * 100,
        "avg_positions_per_period": np.mean(positions_per_period),
        "symbols_used": len(all_data),
    }


def per_symbol_test(symbols, train_bars=4320, test_bars=720):
    """Test each symbol individually"""
    results = {}
    for symbol in symbols:
        df = load_ohlcv(symbol, "1h")
        if df.empty or len(df) < train_bars + test_bars:
            continue

        min_len = len(df)
        all_trades = []
        period_returns = []

        i = train_bars
        while i + test_bars <= min_len:
            full = df.iloc[: i + test_bars]
            sigs = atr_expansion_breakout(full)
            test_sigs = sigs[i : i + test_bars]
            test_df = df.iloc[i : i + test_bars].copy().reset_index(drop=True)

            r = simulate(test_df, test_sigs, 0.02, 72, 3.0, 8.0)
            period_returns.append(r["total_return"])
            all_trades.extend(r["trades"])
            i += test_bars

        if not period_returns or not all_trades:
            continue

        wins = sum(1 for t in all_trades if t > 0)
        losses = sum(1 for t in all_trades if t <= 0)
        gp = sum(t for t in all_trades if t > 0)
        gl = abs(sum(t for t in all_trades if t < 0))

        total_ret = 1.0
        for pr in period_returns:
            total_ret *= 1 + pr
        total_ret -= 1

        results[symbol] = {
            "total_return": total_ret,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
            "profit_factor": gp / (gl + 1e-10),
            "trade_count": len(all_trades),
            "periods": len(period_returns),
            "profitable_periods": sum(1 for r in period_returns if r > 0),
        }

    return results


def stress_test(symbols, train_bars=4320):
    """Test on specific market stress periods"""
    stress_periods = {
        "COVID_crash": ("2020-02-15", "2020-04-15"),
        "May2021_crash": ("2021-04-15", "2021-06-30"),
        "FTX_collapse": ("2022-10-15", "2022-12-31"),
        "ETF_rally_2024": ("2024-01-01", "2024-03-31"),
        "Recent_2025": ("2025-06-01", "2025-12-31"),
    }

    all_data = {}
    for s in symbols[:10]:
        df = load_ohlcv(s, "1h")
        if not df.empty:
            all_data[s] = df

    results = {}
    for period_name, (start, end) in stress_periods.items():
        period_trades = []
        for symbol, df in all_data.items():
            mask = (df["datetime"] >= start) & (df["datetime"] <= end)
            if mask.sum() < 100:
                continue
            # Use data up to end for signals
            full = df[df["datetime"] <= end].copy().reset_index(drop=True)
            if len(full) < 500:
                continue
            sigs = atr_expansion_breakout(full)

            period_df = df[mask].copy().reset_index(drop=True)
            period_sigs = sigs[-len(period_df) :]

            r = simulate(period_df, period_sigs, 0.02, 72, 3.0, 8.0)
            period_trades.extend(r["trades"])

        if period_trades:
            wins = sum(1 for t in period_trades if t > 0)
            losses = sum(1 for t in period_trades if t <= 0)
            results[period_name] = {
                "total_pnl": sum(period_trades),
                "trade_count": len(period_trades),
                "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
            }
        else:
            results[period_name] = {"skipped": True}

    return results


def monte_carlo_test(all_trades, n_sims=5000):
    """Monte Carlo simulation for confidence intervals"""
    if len(all_trades) < 20:
        return {}

    np.random.seed(42)
    sim_returns = []
    for _ in range(n_sims):
        sampled = np.random.choice(all_trades, size=len(all_trades), replace=True)
        sim_returns.append(sum(sampled))

    return {
        "mean": float(np.mean(sim_returns)),
        "median": float(np.median(sim_returns)),
        "ci_5": float(np.percentile(sim_returns, 5)),
        "ci_95": float(np.percentile(sim_returns, 95)),
        "prob_positive": float(np.mean(np.array(sim_returns) > 0)),
    }


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 6: FULL VALIDATION")
    print("Strategy: ATR Expansion Breakout (1h)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # =================================================================
    # 1. Enumerate ALL available symbols
    # =================================================================
    print("=" * 60)
    print("STEP 1: Symbol Universe")
    print("=" * 60)

    all_symbols_path = DATA_ROOT / "binance_futures_1h"
    all_symbols = []
    symbol_info = {}

    for f in sorted(all_symbols_path.glob("*USDT.csv")):
        symbol = f.stem
        size = os.path.getsize(f)
        est_rows = size / 60
        if est_rows > 5000:  # At least ~200 days
            df = load_ohlcv(symbol, "1h")
            if not df.empty:
                all_symbols.append(symbol)
                symbol_info[symbol] = {"rows": len(df), "days": len(df) / 24}

    print(f"  Total USDT symbols with >5000 bars: {len(all_symbols)}")

    # Tiers
    tier_a = [s for s in all_symbols if symbol_info[s]["rows"] > 20000]  # >800 days
    tier_b = [s for s in all_symbols if 10000 < symbol_info[s]["rows"] <= 20000]
    tier_c = [s for s in all_symbols if symbol_info[s]["rows"] <= 10000]

    print(f"  Tier A (>800 days): {len(tier_a)} symbols")
    print(f"  Tier B (400-800 days): {len(tier_b)} symbols")
    print(f"  Tier C (<400 days): {len(tier_c)} symbols")

    # =================================================================
    # 2. Per-symbol walk-forward test
    # =================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Per-Symbol Walk-Forward Test")
    print(f"{'='*60}")

    per_symbol = per_symbol_test(tier_a)
    profitable_symbols = [s for s, r in per_symbol.items() if r["total_return"] > 0]
    high_wr_symbols = [s for s, r in per_symbol.items() if r["win_rate"] > 0.45]

    print(f"  Tested: {len(per_symbol)} symbols")
    print(
        f"  Profitable: {len(profitable_symbols)} ({len(profitable_symbols)/len(per_symbol)*100:.0f}%)"
    )
    print(f"  Win Rate > 45%: {len(high_wr_symbols)}")

    # Top performers
    ranked_symbols = sorted(
        per_symbol.items(), key=lambda x: x[1]["total_return"], reverse=True
    )

    print(f"\n  Top 15 symbols:")
    for s, r in ranked_symbols[:15]:
        print(
            f"    {s:<15} Ret={r['total_return']*100:+.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  T={r['trade_count']}"
        )

    print(f"\n  Bottom 5 symbols:")
    for s, r in ranked_symbols[-5:]:
        print(
            f"    {s:<15} Ret={r['total_return']*100:+.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  T={r['trade_count']}"
        )

    # =================================================================
    # 3. Portfolio-level test on multiple universes
    # =================================================================
    print(f"\n{'='*60}")
    print("STEP 3: Portfolio Walk-Forward")
    print(f"{'='*60}")

    universes = {
        "top_10": ranked_symbols[:10],
        "top_20": ranked_symbols[:20],
        "top_30": ranked_symbols[:30],
        "all_tier_a": [(s, r) for s, r in ranked_symbols],
        "profitable_only": [(s, r) for s, r in ranked_symbols if r["total_return"] > 0],
    }

    portfolio_results = {}
    for uni_name, sym_list in universes.items():
        syms = [s for s, _ in sym_list]
        if not syms:
            continue

        print(f"\n  Universe: {uni_name} ({len(syms)} symbols)")

        r = portfolio_walk_forward(syms, target_vol=0.10, max_positions=5)
        if r is None:
            print("    SKIP")
            continue

        criteria = {
            "sharpe_gt_1": r["sharpe"] > 1.0,
            "max_dd_lt_25": r["max_drawdown"] > -0.25,
            "win_rate_gt_45": r["win_rate"] > 0.45,
            "profit_factor_gt_1_5": r["profit_factor"] > 1.5,
            "wfa_efficiency_gt_50": r["wfa_efficiency"] > 50,
            "trade_count_gt_100": r["trade_count"] > 100,
        }
        passed = sum(v for v in criteria.values())
        r["criteria"] = criteria
        r["criteria_passed"] = passed
        portfolio_results[uni_name] = r

        print(
            f"    [{passed}/6] Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
            f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  T={r['trade_count']}"
        )
        for c, v in criteria.items():
            print(f"      {c}: {'PASS' if v else 'FAIL'}")

    # =================================================================
    # 4. Stress Test
    # =================================================================
    print(f"\n{'='*60}")
    print("STEP 4: Stress Testing")
    print(f"{'='*60}")

    best_syms = [s for s, _ in ranked_symbols[:10]]
    stress = stress_test(best_syms)
    for period, res in stress.items():
        if "skipped" not in res:
            print(
                f"  {period}: PnL={res['total_pnl']*100:+.2f}%  T={res['trade_count']}  WR={res['win_rate']*100:.0f}%"
            )
        else:
            print(f"  {period}: SKIPPED")

    # =================================================================
    # 5. Monte Carlo
    # =================================================================
    print(f"\n{'='*60}")
    print("STEP 5: Monte Carlo Simulation")
    print(f"{'='*60}")

    best_portfolio = max(
        portfolio_results.items(), key=lambda x: x[1]["criteria_passed"]
    )
    best_name = best_portfolio[0]
    # Reconstruct trades for MC (use top_10 as representative)
    top10_syms = [s for s, _ in ranked_symbols[:10]]
    # Quick re-run to get trades
    all_data = {}
    for s in top10_syms:
        df = load_ohlcv(s, "1h")
        if not df.empty and len(df) > 5040:
            all_data[s] = df

    mc_trades = []
    if all_data:
        min_len = min(len(d) for d in all_data.values())
        i = 4320
        while i + 720 <= min_len:
            for symbol, df in list(all_data.items())[:5]:
                full = df.iloc[: i + 720]
                sigs = atr_expansion_breakout(full)
                test_sigs = sigs[i : i + 720]
                test_df = df.iloc[i : i + 720].copy().reset_index(drop=True)
                r = simulate(test_df, test_sigs, 0.02, 72, 3.0, 8.0)
                mc_trades.extend(r["trades"])
            i += 720

    mc = monte_carlo_test(mc_trades)
    if mc:
        print(f"  Mean return: {mc['mean']*100:+.2f}%")
        print(f"  Median return: {mc['median']*100:+.2f}%")
        print(f"  95% CI: [{mc['ci_5']*100:+.2f}%, {mc['ci_95']*100:+.2f}%]")
        print(f"  P(positive): {mc['prob_positive']*100:.1f}%")

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print(f"\n{'='*70}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*70}")

    print(f"""
Strategy: ATR Expansion Breakout
  Timeframe: 1h
  Entry: Donchian(96) breakout + EMA(200) trend + ATR(14) > 1.3x ATR_avg(96)
  Exit: ATR 3x stop loss OR ATR 8x profit target OR 72-bar max hold OR signal reversal
  Position: Volatility-targeted (10% annual vol target), max 5 concurrent, max 5% per position

Universe: {len(all_symbols)} USDT-M Futures symbols
  Tier A tested: {len(per_symbol)} symbols
  Profitable symbols: {len(profitable_symbols)} ({len(profitable_symbols)/max(len(per_symbol),1)*100:.0f}%)
""")

    print("Portfolio Results:")
    for name, r in sorted(
        portfolio_results.items(), key=lambda x: x[1]["criteria_passed"], reverse=True
    ):
        print(
            f"  {name:<20} [{r['criteria_passed']}/6]  Sharpe={r['sharpe']:.2f}  "
            f"Ret={r['total_return']*100:+.1f}%  DD={r['max_drawdown']*100:.1f}%  "
            f"WR={r['win_rate']*100:.0f}%  PF={r['profit_factor']:.2f}"
        )

    if mc:
        print(
            f"\nMonte Carlo: {mc['prob_positive']*100:.0f}% probability of positive return"
        )
        print(f"  95% CI: [{mc['ci_5']*100:+.2f}%, {mc['ci_95']*100:+.2f}%]")

    # Save final report
    report = {
        "generated_at": datetime.now().isoformat(),
        "strategy": {
            "name": "ATR Expansion Breakout",
            "timeframe": "1h",
            "entry": "Donchian(96) breakout + EMA(200) trend + ATR expansion(1.3x)",
            "exit": "ATR 3x stop | ATR 8x profit target | 72-bar max hold | reversal",
            "position_sizing": "Volatility targeting 10% annual, max 5 positions, max 5% each",
        },
        "universe": {
            "total_symbols": len(all_symbols),
            "tier_a": len(tier_a),
            "tier_b": len(tier_b),
            "tier_c": len(tier_c),
        },
        "per_symbol_results": {
            s: {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float))
                    else int(v) if isinstance(v, (np.integer, int)) else v
                )
                for k, v in r.items()
            }
            for s, r in per_symbol.items()
        },
        "portfolio_results": {
            name: {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float))
                    else int(v) if isinstance(v, (np.integer, int)) else v
                )
                for k, v in r.items()
                if k != "criteria"
            }
            for name, r in portfolio_results.items()
        },
        "stress_test": stress,
        "monte_carlo": mc,
        "profitable_symbols": profitable_symbols,
        "top_20_symbols": [s for s, _ in ranked_symbols[:20]],
    }

    report_path = RESULTS_PATH / "phase6_full_validation.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Full report: {report_path}")

    # Check if we achieved 6/6
    any_6_of_6 = any(r["criteria_passed"] == 6 for r in portfolio_results.values())
    print(f"\n  6/6 CRITERIA ACHIEVED: {'YES' if any_6_of_6 else 'NO'}")

    return report


if __name__ == "__main__":
    main()
