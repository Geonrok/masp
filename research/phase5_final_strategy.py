#!/usr/bin/env python3
"""
Ralph-Loop Final: All-Criteria Strategy
=========================================
Goal: Pass ALL 6/6 criteria on walk-forward validation
Base: Breakout_96 on 1h (Sharpe 1.49, 4/6 - failed WR & PF)

Improvements to try:
1. Wider ATR filter to skip choppy markets → higher WR
2. Volume confirmation for entries → fewer false breakouts
3. Partial profit taking → higher PF
4. Asymmetric hold: let winners run, cut losers fast
5. Cross-sectional selection: only trade strongest breakouts
6. Multiple lookback periods for robustness
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


# =============================================================================
# Strategy Variants
# =============================================================================


def strat_v1_vol_confirmed_breakout(df, lookback=96, ema_period=200, vol_mult=1.5):
    """Breakout with volume confirmation: only enter when volume > vol_mult * avg"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    vol_avg = volume.rolling(lookback).mean()
    vol_ok = volume > vol_avg * vol_mult

    signals = np.where(
        (close > upper) & (close > ema) & vol_ok,
        1,
        np.where((close < lower) & (close < ema) & vol_ok, -1, 0),
    )
    return signals


def strat_v2_atr_filter_breakout(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """Breakout with ATR expansion filter: only trade when vol is expanding"""
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


def strat_v3_multi_lookback(df, ema_period=200):
    """Multi-lookback breakout: require breakout on 2 of 3 timeframes"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    ema = close.ewm(span=ema_period, adjust=False).mean()

    scores = np.zeros(len(df))
    for lb in [48, 96, 144]:
        upper = high.rolling(lb).max().shift(1)
        lower = low.rolling(lb).min().shift(1)
        scores += np.where(close > upper, 1, np.where(close < lower, -1, 0))

    # Require 2 of 3 agree + EMA filter
    signals = np.where(
        (scores >= 2) & (close > ema),
        1,
        np.where((scores <= -2) & (close < ema), -1, 0),
    )
    return signals


def strat_v4_momentum_breakout(df, lookback=96, ema_period=200, mom_period=48):
    """Breakout + momentum confirmation"""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    # Momentum: require positive ROC
    roc = close.pct_change(mom_period)

    signals = np.where(
        (close > upper) & (close > ema) & (roc > 0),
        1,
        np.where((close < lower) & (close < ema) & (roc < 0), -1, 0),
    )
    return signals


def strat_v5_breakout_rsi_filter(
    df, lookback=96, ema_period=200, rsi_period=14, rsi_long_min=50, rsi_short_max=50
):
    """Breakout + RSI direction filter: RSI must confirm trend direction"""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    signals = np.where(
        (close > upper) & (close > ema) & (rsi > rsi_long_min),
        1,
        np.where((close < lower) & (close < ema) & (rsi < rsi_short_max), -1, 0),
    )
    return signals


def strat_v6_breakout_vol_atr(
    df, lookback=96, ema_period=200, vol_mult=1.2, atr_expansion=1.2
):
    """Combined: volume + ATR expansion + breakout"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    vol_avg = volume.rolling(lookback).mean()
    vol_ok = volume > vol_avg * vol_mult

    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion

    signals = np.where(
        (close > upper) & (close > ema) & vol_ok & expanding,
        1,
        np.where((close < lower) & (close < ema) & vol_ok & expanding, -1, 0),
    )
    return signals


def strat_v7_breakout_trend_strength(
    df, lookback=96, ema_period=200, adx_period=14, adx_min=25
):
    """Breakout + ADX trend strength filter"""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    # Simplified ADX: use directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr_vals = calc_atr(high.values, low.values, close.values, adx_period)
    atr = pd.Series(atr_vals, index=df.index)

    plus_di = (
        pd.Series(plus_dm, index=df.index).rolling(adx_period).mean()
        / (atr + 1e-10)
        * 100
    )
    minus_di = (
        pd.Series(minus_dm, index=df.index).rolling(adx_period).mean()
        / (atr + 1e-10)
        * 100
    )

    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = dx.rolling(adx_period).mean()

    signals = np.where(
        (close > upper) & (close > ema) & (adx > adx_min),
        1,
        np.where((close < lower) & (close < ema) & (adx > adx_min), -1, 0),
    )
    return signals


# =============================================================================
# Simulation with asymmetric exit
# =============================================================================
def simulate_asymmetric(
    df,
    signals,
    position_pct=0.02,
    max_bars=72,
    atr_stop=3.0,
    profit_target_atr=0.0,
    trailing_atr=0.0,
    breakeven_atr=0.0,
):
    """
    Advanced simulation:
    - ATR stop loss
    - Optional profit target (ATR multiple)
    - Optional trailing stop (kick in after breakeven_atr profit)
    """
    capital = 1.0
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []
    best_price = 0

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
            if position == 1:
                best_price = max(best_price, c)
            else:
                best_price = min(best_price, c)

            should_exit = False
            unrealized_atr = position * (c - entry_price) / (cur_atr + 1e-10)

            # Max hold
            if bars_held >= max_bars:
                should_exit = True

            # Stop loss
            elif atr_stop > 0 and unrealized_atr < -atr_stop:
                should_exit = True

            # Profit target
            elif profit_target_atr > 0 and unrealized_atr > profit_target_atr:
                should_exit = True

            # Trailing stop (only after breakeven)
            elif trailing_atr > 0 and breakeven_atr > 0:
                if unrealized_atr > breakeven_atr:
                    # Activate trailing
                    trail_dist = trailing_atr * cur_atr
                    if position == 1 and c < best_price - trail_dist:
                        should_exit = True
                    elif position == -1 and c > best_price + trail_dist:
                        should_exit = True

            # Signal reversal
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
            best_price = c

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


# =============================================================================
# Portfolio Walk-Forward
# =============================================================================
def portfolio_wf(
    symbols,
    strat_func,
    label,
    train_bars=4320,
    test_bars=720,
    target_vol=0.10,
    max_positions=5,
    atr_stop=3.0,
    profit_target_atr=0.0,
    trailing_atr=0.0,
    breakeven_atr=0.0,
    max_hold=72,
):
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

    i = train_bars
    while i + test_bars <= min_len:
        period_pnl = 0

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
            sigs = strat_func(full)
            test_sigs = sigs[i : i + test_bars]
            test_df = df.iloc[i : i + test_bars].copy().reset_index(drop=True)

            ann_vol = vol * np.sqrt(24 * 365)
            position_pct = min(
                target_vol / (ann_vol + 1e-10) / max(len(selected), 1), 0.05
            )

            r = simulate_asymmetric(
                test_df,
                test_sigs,
                position_pct,
                max_hold,
                atr_stop,
                profit_target_atr,
                trailing_atr,
                breakeven_atr,
            )
            period_pnl += r["total_return"]
            all_trades.extend(r["trades"])

        period_returns.append(period_pnl)
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

    return {
        "name": label,
        "total_return": equity[-1] - 1,
        "max_drawdown": dd.min(),
        "sharpe": sharpe,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "profit_factor": gp / (gl + 1e-10),
        "trade_count": len(all_trades),
        "periods": len(period_returns),
        "wfa_efficiency": sum(1 for r in period_returns if r > 0)
        / len(period_returns)
        * 100,
    }


def check_criteria(r):
    c = {
        "sharpe_gt_1": r["sharpe"] > 1.0,
        "max_dd_lt_25": r["max_drawdown"] > -0.25,
        "win_rate_gt_45": r["win_rate"] > 0.45,
        "profit_factor_gt_1_5": r["profit_factor"] > 1.5,
        "wfa_efficiency_gt_50": r["wfa_efficiency"] > 50,
        "trade_count_gt_100": r["trade_count"] > 100,
    }
    return c, sum(v for v in c.values())


def main():
    print("=" * 70)
    print("RALPH-LOOP FINAL: ALL-CRITERIA STRATEGY SEARCH")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Use all symbols with sufficient data
    all_symbols_path = DATA_ROOT / "binance_futures_1h"
    available = []
    for f in sorted(all_symbols_path.glob("*USDT.csv")):
        pd.read_csv(f, nrows=5)
        # Check row count from file size (rough estimate)
        import os

        size = os.path.getsize(f)
        est_rows = size / 60  # ~60 bytes per row
        if est_rows > 10000:  # > ~400 days
            available.append(f.stem)

    print(f"Symbols with >10000 bars: {len(available)}")

    # Use top liquidity symbols
    symbols_10 = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "LINKUSDT",
        "AVAXUSDT",
        "LTCUSDT",
    ]

    # Extended universe
    symbols_20 = symbols_10 + [
        "MATICUSDT",
        "DOTUSDT",
        "UNIUSDT",
        "AAVEUSDT",
        "FILUSDT",
        "ATOMUSDT",
        "NEARUSDT",
        "APTUSDT",
        "ARBUSDT",
        "OPUSDT",
    ]

    # =================================================================
    # Test all strategy + exit combinations
    # =================================================================
    strategies = {
        "v1_vol_confirm": lambda df: strat_v1_vol_confirmed_breakout(df, 96, 200, 1.5),
        "v1_vol_1.2": lambda df: strat_v1_vol_confirmed_breakout(df, 96, 200, 1.2),
        "v2_atr_expand": lambda df: strat_v2_atr_filter_breakout(df, 96, 200, 1.3),
        "v2_atr_expand_1.2": lambda df: strat_v2_atr_filter_breakout(df, 96, 200, 1.2),
        "v3_multi_lb": lambda df: strat_v3_multi_lookback(df, 200),
        "v4_mom_confirm": lambda df: strat_v4_momentum_breakout(df, 96, 200, 48),
        "v5_rsi_filter": lambda df: strat_v5_breakout_rsi_filter(
            df, 96, 200, 14, 50, 50
        ),
        "v5_rsi_55_45": lambda df: strat_v5_breakout_rsi_filter(
            df, 96, 200, 14, 55, 45
        ),
        "v6_vol_atr": lambda df: strat_v6_breakout_vol_atr(df, 96, 200, 1.2, 1.2),
        "v7_adx_25": lambda df: strat_v7_breakout_trend_strength(df, 96, 200, 14, 25),
        "v7_adx_20": lambda df: strat_v7_breakout_trend_strength(df, 96, 200, 14, 20),
    }

    exit_configs = [
        ("atr3", 3.0, 0, 0, 0, 72),
        ("atr3_pt6", 3.0, 6.0, 0, 0, 72),
        ("atr3_pt8", 3.0, 8.0, 0, 0, 72),
        ("atr2_pt5", 2.0, 5.0, 0, 0, 72),
        ("atr3_trail2_be1", 3.0, 0, 2.0, 1.0, 96),
        ("atr3_trail3_be2", 3.0, 0, 3.0, 2.0, 96),
        ("atr3_trail2_be2", 3.0, 0, 2.0, 2.0, 96),
    ]

    all_results = []

    for strat_name, strat_func in strategies.items():
        for exit_name, atr_s, pt, trail, be, mh in exit_configs:
            label = f"{strat_name}_{exit_name}"
            print(f"  {label}...", end=" ", flush=True)

            r = portfolio_wf(
                symbols_10,
                strat_func,
                label,
                target_vol=0.10,
                max_positions=5,
                atr_stop=atr_s,
                profit_target_atr=pt,
                trailing_atr=trail,
                breakeven_atr=be,
                max_hold=mh,
            )

            if r is None:
                print("SKIP")
                continue

            criteria, passed = check_criteria(r)
            r["criteria"] = criteria
            r["criteria_passed"] = passed
            all_results.append(r)

            marker = " ***" if passed >= 5 else (" **" if passed >= 4 else "")
            print(
                f"[{passed}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                f"PF={r['profit_factor']:.2f} T={r['trade_count']}{marker}"
            )

    # =================================================================
    # Top candidates: test on wider universe
    # =================================================================
    top_candidates = sorted(
        all_results, key=lambda x: (x["criteria_passed"], x["sharpe"]), reverse=True
    )[:5]

    print(f"\n{'='*70}")
    print("TOP 5 → Testing on 20-symbol universe")
    print(f"{'='*70}")

    final_results = []
    for r in top_candidates:
        label = r["name"]
        # Parse config from name
        parts = label.split("_")
        "_".join(parts[:-1])
        parts[-1] if len(parts) > 1 else "atr3"

        # Find the matching strategy and exit
        for sn, sf in strategies.items():
            if label.startswith(sn):
                strat_func = sf
                exit_label = label[len(sn) + 1 :]
                break

        for en, atr_s, pt, trail, be, mh in exit_configs:
            if en == exit_label:
                print(f"\n  {label} on 20 symbols...", end=" ", flush=True)

                r20 = portfolio_wf(
                    symbols_20,
                    strat_func,
                    f"{label}_20sym",
                    target_vol=0.10,
                    max_positions=5,
                    atr_stop=atr_s,
                    profit_target_atr=pt,
                    trailing_atr=trail,
                    breakeven_atr=be,
                    max_hold=mh,
                )

                if r20:
                    criteria, passed = check_criteria(r20)
                    r20["criteria"] = criteria
                    r20["criteria_passed"] = passed
                    final_results.append(r20)

                    print(
                        f"[{passed}/6] Sharpe={r20['sharpe']:.2f} Ret={r20['total_return']*100:+.1f}% "
                        f"DD={r20['max_drawdown']*100:.1f}% WR={r20['win_rate']*100:.0f}% "
                        f"PF={r20['profit_factor']:.2f} T={r20['trade_count']}"
                    )
                break

    # =================================================================
    # Final ranking
    # =================================================================
    print(f"\n{'='*70}")
    print("FINAL RANKING")
    print(f"{'='*70}")

    all_combined = all_results + final_results
    ranked = sorted(
        all_combined, key=lambda x: (x["criteria_passed"], x["sharpe"]), reverse=True
    )

    for rank, r in enumerate(ranked[:20], 1):
        marker = (
            "*** 6/6 ***"
            if r["criteria_passed"] == 6
            else ("** 5/6 **" if r["criteria_passed"] == 5 else "")
        )
        print(
            f"  {rank:2d}. [{r['criteria_passed']}/6] {r['name']:<45} "
            f"Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
            f"PF={r['profit_factor']:.2f} T={r['trade_count']} {marker}"
        )

    # Detail on best
    if ranked:
        best = ranked[0]
        print(f"\n{'='*70}")
        print(f"BEST: {best['name']}")
        print(f"{'='*70}")
        for c, v in best["criteria"].items():
            print(f"  {c}: {'PASS' if v else 'FAIL'}")

    # Save
    report = {
        "generated_at": datetime.now().isoformat(),
        "configs_tested": len(all_combined),
        "best": ranked[0]["name"] if ranked else None,
        "best_criteria_passed": int(ranked[0]["criteria_passed"]) if ranked else 0,
        "ranking": [
            {
                "rank": i + 1,
                "name": r["name"],
                "criteria_passed": int(r["criteria_passed"]),
                "sharpe": float(r["sharpe"]),
                "total_return": float(r["total_return"]),
                "max_drawdown": float(r["max_drawdown"]),
                "win_rate": float(r["win_rate"]),
                "profit_factor": float(r["profit_factor"]),
                "wfa_efficiency": float(r["wfa_efficiency"]),
                "trade_count": int(r["trade_count"]),
            }
            for i, r in enumerate(ranked)
        ],
    }

    report_path = RESULTS_PATH / "phase5_final_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report: {report_path}")

    return report


if __name__ == "__main__":
    main()
