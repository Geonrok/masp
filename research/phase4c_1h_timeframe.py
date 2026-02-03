#!/usr/bin/env python3
"""
Ralph-Loop Phase 4C: 1H Timeframe Strategy Test
=================================================
Test TSMOM and improved strategies on 1-hour data
Hypothesis: shorter timeframe → more trades → better Sharpe
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

# Costs adjusted for 1h
SLIPPAGE = 0.0003  # tighter for 1h (more liquid times)
COMMISSION = 0.0004  # same taker fee
FUNDING_PER_8H = 0.0001


def load_ohlcv(symbol, timeframe="1h"):
    tf_map = {
        "1h": "binance_futures_1h",
        "4h": "binance_futures_4h",
        "1d": "binance_futures_1d",
    }
    path = DATA_ROOT / tf_map[timeframe] / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["datetime", "timestamp", "date"]:
        if col in df.columns:
            df["datetime"] = pd.to_datetime(df[col])
            break
    return df.sort_values("datetime").reset_index(drop=True)


def simulate(
    df, signals, position_pct=0.02, max_bars=72, atr_stop=0.0, trailing_atr=0.0
):
    """Simulate with optional ATR stop / trailing stop"""
    capital = 1.0
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []
    best_price = 0

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # ATR
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14).mean().values

    for i in range(len(df)):
        c = close[i]
        sig = signals[i] if i < len(signals) else 0
        cur_atr = atr[i] if not np.isnan(atr[i]) else c * 0.01

        if position != 0:
            bars_held += 1
            if position == 1:
                best_price = max(best_price, c)
            else:
                best_price = min(best_price, c)

            should_exit = False

            if bars_held >= max_bars:
                should_exit = True
            elif atr_stop > 0:
                if position == 1 and c < entry_price - atr_stop * cur_atr:
                    should_exit = True
                elif position == -1 and c > entry_price + atr_stop * cur_atr:
                    should_exit = True
            elif trailing_atr > 0:
                if position == 1 and c < best_price - trailing_atr * cur_atr:
                    should_exit = True
                elif position == -1 and c > best_price + trailing_atr * cur_atr:
                    should_exit = True
            elif sig != 0 and sig != position:
                should_exit = True
            elif sig == 0 and position != 0:
                should_exit = True

            if should_exit:
                exit_p = c * (1 - SLIPPAGE * np.sign(position))
                pnl = position * (exit_p - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct  # 1h = 1/8 of 8h
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
# Strategies (periods in 1h bars)
# =============================================================================


def strat_tsmom(df, lookback=168):
    """TSMOM on 1h: lookback=168 = 7 days"""
    ret = df["close"].pct_change(lookback)
    return np.sign(ret).fillna(0).astype(int).values


def strat_tsmom_ema_filter(df, lookback=168, ema_period=200):
    """TSMOM + EMA trend filter: only trade with the longer trend"""
    close = df["close"]
    ret = close.pct_change(lookback)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    sig = np.sign(ret)
    # Only long above EMA, only short below
    filtered = np.where(
        (sig > 0) & (close > ema), 1, np.where((sig < 0) & (close < ema), -1, 0)
    )
    return filtered


def strat_breakout_1h(df, lookback=48, ema_period=200):
    """Donchian breakout on 1h with EMA filter"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    signals = np.where(
        (close > upper) & (close > ema),
        1,
        np.where((close < lower) & (close < ema), -1, 0),
    )
    return signals


def strat_mean_reversion_1h(df, bb_window=48, bb_std=2.5, rsi_window=14):
    """Short-term mean reversion: BB + RSI oversold/overbought"""
    close = df["close"]

    # Bollinger
    sma = close.rolling(bb_window).mean()
    std = close.rolling(bb_window).std()
    upper = sma + bb_std * std
    lower = sma - bb_std * std

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Long: below lower BB AND RSI < 25
    # Short: above upper BB AND RSI > 75
    signals = np.where(
        (close < lower) & (rsi < 25), 1, np.where((close > upper) & (rsi > 75), -1, 0)
    )
    return signals


def strat_ema_crossover(df, fast=12, slow=48):
    """EMA crossover"""
    close = df["close"]
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return np.where(ema_f > ema_s, 1, -1)


def strat_multi_signal_1h(df):
    """
    Multi-signal: combine 3 independent signals on 1h.
    Score >= 2 → long, <= -2 → short
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    scores = np.zeros(len(df))

    # Signal 1: TSMOM 168 (7-day momentum)
    ret_168 = close.pct_change(168)
    scores += np.sign(ret_168).fillna(0).values

    # Signal 2: EMA trend (24/96)
    ema24 = close.ewm(span=24, adjust=False).mean()
    ema96 = close.ewm(span=96, adjust=False).mean()
    scores += np.where(ema24 > ema96, 1, -1)

    # Signal 3: Donchian breakout (48)
    upper = high.rolling(48).max().shift(1)
    lower = low.rolling(48).min().shift(1)
    scores += np.where(close > upper, 1, np.where(close < lower, -1, 0))

    return np.where(scores >= 2, 1, np.where(scores <= -2, -1, 0))


def strat_vol_breakout_1h(df, lookback=24, atr_mult=1.0):
    """Volatility breakout: price moves > ATR multiple from open"""
    close = df["close"]
    open_p = df["open"]
    high = df["high"]
    low = df["low"]

    tr = pd.DataFrame(
        {
            "hl": high - low,
            "hc": abs(high - close.shift(1)),
            "lc": abs(low - close.shift(1)),
        }
    ).max(axis=1)
    atr = tr.rolling(lookback).mean()

    # EMA filter
    ema200 = close.ewm(span=200, adjust=False).mean()

    signals = np.where(
        (close > open_p + atr * atr_mult) & (close > ema200),
        1,
        np.where((close < open_p - atr * atr_mult) & (close < ema200), -1, 0),
    )
    return signals


def strat_keltner_breakout(df, ema_period=48, atr_period=14, atr_mult=2.0):
    """Keltner Channel breakout"""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema = close.ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame(
        {
            "hl": high - low,
            "hc": abs(high - close.shift(1)),
            "lc": abs(low - close.shift(1)),
        }
    ).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr

    signals = np.where(close > upper, 1, np.where(close < lower, -1, 0))
    return signals


# =============================================================================
# Walk-Forward Portfolio Engine
# =============================================================================
def portfolio_wf(
    symbols,
    strat_func,
    strat_name,
    train_bars=4320,
    test_bars=720,  # 180d train, 30d test (1h)
    target_vol=0.10,
    max_positions=5,
    atr_stop=0.0,
    trailing_atr=0.0,
    max_hold=72,
):
    """Walk-forward portfolio backtest on 1h data"""
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

        # Score symbols by inverse volatility
        scored = []
        for symbol, df in all_data.items():
            train_ret = df["close"].iloc[:i].pct_change()
            vol = train_ret.rolling(168).std().iloc[-1]
            if np.isnan(vol) or vol == 0:
                vol = 0.01
            scored.append((symbol, vol))

        scored.sort(key=lambda x: x[1])  # low vol first
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

            r = simulate(
                test_df, test_sigs, position_pct, max_hold, atr_stop, trailing_atr
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
        "name": strat_name,
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
    print("RALPH-LOOP PHASE 4C: 1H TIMEFRAME TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Check data availability
    test_path = DATA_ROOT / "binance_futures_1h"
    if test_path.exists():
        files = list(test_path.glob("*USDT.csv"))
        print(f"1h data available: {len(files)} symbols")
    else:
        print("ERROR: 1h data not found!")
        return

    symbols = [
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

    # Verify data length
    for s in symbols:
        df = load_ohlcv(s, "1h")
        if df.empty:
            print(f"  {s}: NO DATA")
        else:
            print(f"  {s}: {len(df)} bars ({len(df)/24:.0f} days)")

    # Strategy matrix
    strategies = [
        ("TSMOM_168", lambda df: strat_tsmom(df, 168), {}),
        ("TSMOM_336", lambda df: strat_tsmom(df, 336), {}),
        ("TSMOM_EMA_168", lambda df: strat_tsmom_ema_filter(df, 168, 200), {}),
        ("TSMOM_EMA_336", lambda df: strat_tsmom_ema_filter(df, 336, 400), {}),
        ("Breakout_48", lambda df: strat_breakout_1h(df, 48, 200), {}),
        ("Breakout_96", lambda df: strat_breakout_1h(df, 96, 400), {}),
        ("MeanRev_BB_RSI", lambda df: strat_mean_reversion_1h(df), {}),
        ("EMA_12_48", lambda df: strat_ema_crossover(df, 12, 48), {}),
        ("EMA_24_96", lambda df: strat_ema_crossover(df, 24, 96), {}),
        ("Multi_Signal", lambda df: strat_multi_signal_1h(df), {}),
        ("Vol_Breakout", lambda df: strat_vol_breakout_1h(df), {}),
        ("Keltner_Break", lambda df: strat_keltner_breakout(df), {}),
    ]

    # Stop configs
    stop_configs = [
        ("no_stop", 0, 0, 72),
        ("trail_3atr", 0, 3.0, 72),
        ("trail_4atr", 0, 4.0, 96),
        ("atr_3x", 3.0, 0, 72),
        ("short_hold", 0, 0, 24),  # max 24 bars = 1 day
        ("medium_hold", 0, 0, 48),  # max 48 bars = 2 days
    ]

    all_results = []

    for strat_name, strat_func, _ in strategies:
        for stop_name, atr_s, trail_s, max_h in stop_configs:
            key = f"{strat_name}_{stop_name}"
            print(f"\n  Testing {key}...", end=" ", flush=True)

            r = portfolio_wf(
                symbols,
                strat_func,
                key,
                target_vol=0.10,
                max_positions=5,
                atr_stop=atr_s,
                trailing_atr=trail_s,
                max_hold=max_h,
            )

            if r is None:
                print("SKIP")
                continue

            criteria, passed = check_criteria(r)
            r["criteria"] = criteria
            r["criteria_passed"] = passed
            all_results.append(r)

            print(
                f"Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
                f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
                f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  "
                f"T={r['trade_count']}  [{passed}/6]"
            )

    # Also test with 15% vol target on best candidates
    print(f"\n{'='*70}")
    print("AGGRESSIVE VOL TARGET (15%) on promising strategies")
    print(f"{'='*70}")

    promising_strats = [
        s
        for s in strategies
        if any(
            r["criteria_passed"] >= 3 for r in all_results if r["name"].startswith(s[0])
        )
    ]
    if not promising_strats:
        # Fall back to top strategies
        promising_strats = strategies[:6]

    for strat_name, strat_func, _ in promising_strats:
        for stop_name, atr_s, trail_s, max_h in [
            ("no_stop", 0, 0, 72),
            ("trail_3atr", 0, 3.0, 72),
        ]:
            key = f"{strat_name}_{stop_name}_15pct"
            print(f"\n  Testing {key}...", end=" ", flush=True)

            r = portfolio_wf(
                symbols,
                strat_func,
                key,
                target_vol=0.15,
                max_positions=5,
                atr_stop=atr_s,
                trailing_atr=trail_s,
                max_hold=max_h,
            )

            if r is None:
                print("SKIP")
                continue

            criteria, passed = check_criteria(r)
            r["criteria"] = criteria
            r["criteria_passed"] = passed
            all_results.append(r)

            print(
                f"Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
                f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
                f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  "
                f"T={r['trade_count']}  [{passed}/6]"
            )

    # Final ranking
    print(f"\n{'='*70}")
    print("1H STRATEGY RANKING")
    print(f"{'='*70}")

    ranked = sorted(
        all_results, key=lambda x: (x["criteria_passed"], x["sharpe"]), reverse=True
    )

    for rank, r in enumerate(ranked[:20], 1):
        marker = (
            "***"
            if r["criteria_passed"] >= 5
            else ("**" if r["criteria_passed"] >= 4 else "")
        )
        print(
            f"  {rank:2d}. {r['name']:<40} [{r['criteria_passed']}/6]  "
            f"Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
            f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  T={r['trade_count']}  {marker}"
        )

    # Detail on best
    if ranked:
        best = ranked[0]
        print(f"\n{'='*70}")
        print(f"BEST: {best['name']}")
        print(f"{'='*70}")
        for c, v in best["criteria"].items():
            print(f"  {c}: {'PASS' if v else 'FAIL'}")

        print("\n  vs 4h baseline (Sharpe=0.64, 4/6):")
        imp = (
            "IMPROVED"
            if best["criteria_passed"] > 4 or best["sharpe"] > 0.64
            else "NOT IMPROVED"
        )
        print(f"  Result: {imp}")
        print(f"  Sharpe: 0.64 → {best['sharpe']:.2f}")
        print(f"  Criteria: 4/6 → {best['criteria_passed']}/6")

    # Save
    report = {
        "generated_at": datetime.now().isoformat(),
        "timeframe": "1h",
        "configs_tested": len(all_results),
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

    report_path = RESULTS_PATH / "phase4c_1h_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report: {report_path}")

    return report


if __name__ == "__main__":
    main()
