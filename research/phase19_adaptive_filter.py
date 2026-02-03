#!/usr/bin/env python3
"""
Phase 19: 적응형 추세 필터 테스트

기존: EMA(50) > EMA(200) → 고정 파라미터
테스트:
A. KAMA(10,2,30) > KAMA(50,2,30) — Kaufman Adaptive MA
B. KAMA(10) > KAMA(50) — 단순 KAMA 크로스
C. KAMA 단독 방향 (KAMA 기울기 > 0)
D. DEMA (Double EMA) — 더 빠른 반응
E. TEMA (Triple EMA) — 더더 빠른 반응
F. Ehlers Filter (Super Smoother)
G. KAMA + VWAP 복합

수정 기준: Sharpe>1, DD<25%, WR>30%, PF>1.3, WFA>50%, T>100
OOS: min_oos_bars=16000 (19종목, 18윈도우, ~2년)
"""

import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

DATA_ROOT = Path("E:/data/crypto_ohlcv")
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


# ============================================================
# ADAPTIVE MOVING AVERAGES
# ============================================================


def kama(series, period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average
    - 추세장: EMA(fast_sc)처럼 빠르게 반응
    - 횡보장: EMA(slow_sc)처럼 느리게 반응
    """
    close = series.values.astype(float)
    n = len(close)
    result = np.full(n, np.nan)

    if n < period + 1:
        return pd.Series(result, index=series.index)

    # Efficiency Ratio
    fast_alpha = 2.0 / (fast_sc + 1)
    slow_alpha = 2.0 / (slow_sc + 1)

    result[period] = close[period]

    for i in range(period + 1, n):
        direction = abs(close[i] - close[i - period])
        volatility = sum(
            abs(close[j] - close[j - 1]) for j in range(i - period + 1, i + 1)
        )
        if volatility == 0:
            er = 0
        else:
            er = direction / volatility

        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        result[i] = result[i - 1] + sc * (close[i] - result[i - 1])

    return pd.Series(result, index=series.index)


def dema(series, period):
    """Double Exponential Moving Average — lag 줄임"""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2


def tema(series, period):
    """Triple Exponential Moving Average — lag 더 줄임"""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3


def super_smoother(series, period=10):
    """Ehlers Super Smoother — 노이즈 제거에 최적화"""
    close = series.values.astype(float)
    n = len(close)
    result = np.full(n, np.nan)

    a = np.exp(-np.sqrt(2) * np.pi / period)
    b = 2 * a * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3

    if n < 3:
        return pd.Series(result, index=series.index)

    result[0] = close[0]
    result[1] = close[1]
    for i in range(2, n):
        result[i] = (
            c1 * (close[i] + close[i - 1]) / 2 + c2 * result[i - 1] + c3 * result[i - 2]
        )

    return pd.Series(result, index=series.index)


# ============================================================
# STRATEGIES
# ============================================================


def strat_baseline_ema(df, lookback=48):
    """기존 VWAP Breakout (EMA 50/200)"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (close > vwap * 1.02) & (ema_f > ema_s), 1, 0)


def strat_kama_cross(df, lookback=48, fast_p=10, slow_p=50):
    """KAMA(fast) > KAMA(slow) 크로스"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    kama_f = kama(close, period=fast_p, fast_sc=2, slow_sc=30)
    kama_s = kama(close, period=slow_p, fast_sc=2, slow_sc=30)
    return np.where((close > upper) & (close > vwap * 1.02) & (kama_f > kama_s), 1, 0)


def strat_kama_slope(df, lookback=48, kama_period=20):
    """KAMA 기울기 > 0 (상승 중일 때만)"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    k = kama(close, period=kama_period, fast_sc=2, slow_sc=30)
    k_slope = k - k.shift(10)  # 10봉 전 대비 방향
    return np.where((close > upper) & (close > vwap * 1.02) & (k_slope > 0), 1, 0)


def strat_kama_dual(df, lookback=48):
    """KAMA(10,2,30) > KAMA(50,2,30) — 원래 EMA 50/200 대체"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    kf = kama(close, period=10, fast_sc=2, slow_sc=30)
    ks = kama(close, period=50, fast_sc=5, slow_sc=50)
    return np.where((close > upper) & (close > vwap * 1.02) & (kf > ks), 1, 0)


def strat_dema_cross(df, lookback=48):
    """DEMA(50) > DEMA(200)"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    df_ = dema(close, 50)
    ds = dema(close, 200)
    return np.where((close > upper) & (close > vwap * 1.02) & (df_ > ds), 1, 0)


def strat_tema_cross(df, lookback=48):
    """TEMA(50) > TEMA(200)"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    tf = tema(close, 50)
    ts = tema(close, 200)
    return np.where((close > upper) & (close > vwap * 1.02) & (tf > ts), 1, 0)


def strat_super_smoother_cross(df, lookback=48):
    """Super Smoother(50) > Super Smoother(200)"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    sf = super_smoother(close, 50)
    ss = super_smoother(close, 200)
    return np.where((close > upper) & (close > vwap * 1.02) & (sf > ss), 1, 0)


def strat_kama_no_vwap(df, lookback=48):
    """KAMA 크로스만 (VWAP 없이) — KAMA가 단독으로 충분한지"""
    close = df["close"]
    high = df["high"]
    upper = high.rolling(lookback).max().shift(1)
    kf = kama(close, period=10, fast_sc=2, slow_sc=30)
    ks = kama(close, period=50, fast_sc=2, slow_sc=30)
    return np.where((close > upper) & (kf > ks), 1, 0)


def strat_kama_ema_hybrid(df, lookback=48):
    """KAMA(20) slope > 0 AND EMA(50) > EMA(200) — 둘 다 확인"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    k = kama(close, period=20, fast_sc=2, slow_sc=30)
    k_slope = k - k.shift(10)
    return np.where(
        (close > upper) & (close > vwap * 1.02) & (ema_f > ema_s) & (k_slope > 0), 1, 0
    )


# ============================================================
# SIMULATION & PORTFOLIO
# ============================================================


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
            if should_exit:
                exit_p = c * (1 - slippage * np.sign(position))
                pnl = position * (exit_p - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0
        if position == 0 and sig != 0:
            position = 1
            entry_price = c * (1 + slippage)
            capital -= COMMISSION * position_pct
            bars_held = 0

    if position != 0:
        c = close[-1]
        exit_p = c * (1 - slippage)
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


def check_criteria_revised(r):
    c = {
        "sharpe_gt_1": r.get("sharpe", 0) > 1.0,
        "max_dd_lt_25": r.get("max_drawdown", -1) > -0.25,
        "win_rate_gt_30": r.get("win_rate", 0) > 0.30,
        "profit_factor_gt_1_3": r.get("profit_factor", 0) > 1.3,
        "wfa_efficiency_gt_50": r.get("wfa_efficiency", 0) > 50,
        "trade_count_gt_100": r.get("trade_count", 0) > 100,
    }
    return c, sum(v for v in c.values())


def run_portfolio_oos(
    all_data,
    strat_fn,
    btc_df=None,
    min_oos_bars=16000,
    max_positions=10,
    test_bars=720,
    position_scale=5.0,
    train_bars=4320,
):
    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > min_oos_bars:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    # BTC OOS for regime
    btc_oos = None
    if btc_df is not None:
        btc_split = int(len(btc_df) * 0.6)
        btc_oos = btc_df.iloc[btc_split:].copy().reset_index(drop=True)

    min_len = min(len(d) for d in oos_data.values())

    equity = [1.0]
    period_returns = []
    all_trades = []
    regime_pnls = {"BULL": [], "BEAR": [], "SIDEWAYS": []}

    i = train_bars
    while i + test_bars <= min_len:
        # Regime tag
        regime = "UNKNOWN"
        if btc_oos is not None and i + test_bars <= len(btc_oos):
            btc_s = btc_oos["close"].iloc[i]
            btc_e = btc_oos["close"].iloc[min(i + test_bars - 1, len(btc_oos) - 1)]
            btc_ret = (btc_e - btc_s) / btc_s
            if btc_ret > 0.10:
                regime = "BULL"
            elif btc_ret < -0.10:
                regime = "BEAR"
            else:
                regime = "SIDEWAYS"

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
            r = simulate(test_df, test_sigs, position_pct, 72, 3.0, 8.0, 0.0003)
            period_pnl += r["total_return"]
            all_trades.extend(r["trades"])

        period_returns.append(period_pnl)
        equity.append(equity[-1] * (1 + period_pnl))
        if regime in regime_pnls:
            regime_pnls[regime].append(period_pnl)
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

    # Regime returns
    regime_ret = {}
    for reg, pnls in regime_pnls.items():
        if pnls:
            total = 1.0
            for p in pnls:
                total *= 1 + p
            regime_ret[reg] = total - 1
        else:
            regime_ret[reg] = None

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
        "regime_returns": regime_ret,
    }


def main():
    print("=" * 90)
    print("PHASE 19: ADAPTIVE TREND FILTER TEST (KAMA, DEMA, TEMA, Super Smoother)")
    print("=" * 90)
    print(f"Started: {datetime.now().isoformat()}\n")

    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Loaded {len(all_data)} symbols")

    btc_df = all_data.get("BTCUSDT")
    qualifying = sum(1 for df in all_data.values() if int(len(df) * 0.4) > 16000)
    print(f"Qualifying (OOS>16000): {qualifying}\n")

    strategies = {
        # Baseline
        "A_baseline_EMA": strat_baseline_ema,
        # KAMA variants
        "B_kama_10_50": lambda df: strat_kama_cross(df, fast_p=10, slow_p=50),
        "C_kama_20_100": lambda df: strat_kama_cross(df, fast_p=20, slow_p=100),
        "D_kama_dual": strat_kama_dual,
        "E_kama_slope_20": lambda df: strat_kama_slope(df, kama_period=20),
        "F_kama_slope_50": lambda df: strat_kama_slope(df, kama_period=50),
        # DEMA / TEMA
        "G_dema_50_200": strat_dema_cross,
        "H_tema_50_200": strat_tema_cross,
        # Ehlers
        "I_super_smoother": strat_super_smoother_cross,
        # No VWAP (KAMA standalone)
        "J_kama_no_vwap": strat_kama_no_vwap,
        # Hybrid
        "K_kama_ema_hybrid": strat_kama_ema_hybrid,
    }

    header = (
        f"{'Strategy':<22} {'Rev':>4} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} "
        f"{'WR':>5} {'PF':>6} {'WFA':>5} {'T':>5} "
        f"{'Bull':>7} {'Bear':>7} {'Side':>7}"
    )
    print(header)
    print("-" * 105)

    results = []
    for name, fn in strategies.items():
        r = run_portfolio_oos(
            all_data, fn, btc_df=btc_df, min_oos_bars=16000, position_scale=5.0
        )
        if r:
            c, p = check_criteria_revised(r)
            rr = r["regime_returns"]
            bull = (
                f"{rr.get('BULL', 0)*100:+.1f}%"
                if rr.get("BULL") is not None
                else "N/A"
            )
            bear = (
                f"{rr.get('BEAR', 0)*100:+.1f}%"
                if rr.get("BEAR") is not None
                else "N/A"
            )
            side = (
                f"{rr.get('SIDEWAYS', 0)*100:+.1f}%"
                if rr.get("SIDEWAYS") is not None
                else "N/A"
            )

            fails = [k for k, v in c.items() if not v]
            results.append((name, p, r))

            print(
                f"  {name:<20} [{p}/6] {r['sharpe']:>6.2f} {r['total_return']*100:>+7.1f}% "
                f"{r['max_drawdown']*100:>6.1f}% {r['win_rate']*100:>4.0f}% "
                f"{r['profit_factor']:>5.2f} {r['wfa_efficiency']:>4.0f}% {r['trade_count']:>4} "
                f"{bull:>7} {bear:>7} {side:>7}"
            )
            if fails:
                print(f"    FAILS: {', '.join(fails)}")
        else:
            print(f"  {name:<20} NO DATA")

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY: REVISED 6/6 PASS + REGIME COMPARISON")
    print("=" * 90)

    passed = [(n, p, r) for n, p, r in results if p == 6]
    if passed:
        passed.sort(key=lambda x: -x[2]["sharpe"])
        print("\n6/6 PASSED (sorted by Sharpe):")
        for name, p, r in passed:
            rr = r["regime_returns"]
            bear_ret = rr.get("BEAR", 0)
            bear_str = f"{bear_ret*100:+.1f}%" if bear_ret is not None else "N/A"
            bear_ok = (
                " [BEAR OK]"
                if bear_ret is not None and bear_ret > 0
                else " [BEAR LOSS]" if bear_ret is not None and bear_ret < 0 else ""
            )
            print(
                f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                f"Bear={bear_str}{bear_ok}"
            )

        # Best overall considering all regimes
        print("\nBest bear-market survivor among 6/6:")
        bear_ranked = sorted(
            passed, key=lambda x: -(x[2]["regime_returns"].get("BEAR", -999) or -999)
        )
        for name, p, r in bear_ranked[:3]:
            rr = r["regime_returns"]
            print(
                f"  {name}: Bear={rr.get('BEAR',0)*100:+.1f}% "
                f"Side={rr.get('SIDEWAYS',0)*100:+.1f}% "
                f"Bull={rr.get('BULL',0)*100:+.1f}%"
            )
    else:
        print("\nNo strategy passed 6/6 revised criteria")
        best = sorted(results, key=lambda x: -x[1])[:5]
        for name, p, r in best:
            print(f"  [{p}/6] {name}: Sharpe={r['sharpe']:.2f}")

    # vs baseline comparison
    baseline = next((r for n, p, r in results if n == "A_baseline_EMA"), None)
    if baseline:
        print(
            f"\nBaseline (EMA 50/200): Sharpe={baseline['sharpe']:.2f} "
            f"Ret={baseline['total_return']*100:+.1f}%"
        )
        better = [
            (n, r)
            for n, p, r in results
            if r["sharpe"] > baseline["sharpe"] and n != "A_baseline_EMA"
        ]
        if better:
            print("Strategies that BEAT baseline:")
            for name, r in sorted(better, key=lambda x: -x[1]["sharpe"]):
                diff = r["sharpe"] - baseline["sharpe"]
                print(f"  {name}: Sharpe={r['sharpe']:.2f} (+{diff:.2f})")
        else:
            print("No adaptive filter beats EMA baseline")

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
