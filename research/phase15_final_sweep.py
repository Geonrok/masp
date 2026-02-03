#!/usr/bin/env python3
"""
Phase 15: Final Strategy Sweep - 아직 테스트하지 않은 모든 변형
================================================================
1. 타임프레임 변경 (4H)
2. 종목 선택 방식 변경 (모멘텀 기반 vs 저변동성)
3. 트레일링 스탑
4. 숏 포지션 (하락장 전용)
5. 평균회귀 전략 (1H 포트폴리오 프레임워크)
6. 매크로 오버레이 + Vol Profile
7. 다중 타임프레임 필터 (1D 추세 + 1H 진입)
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


def calc_atr(high, low, close, period=14):
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values


# =============================================================================
# STRATEGY VARIANTS
# =============================================================================


def strat_vol_profile(df, lookback=48):
    """Current best: Vol Profile Breakout (baseline for comparison)"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s), 1, 0)
    return signals


def strat_mean_reversion(df, lookback=20, zscore_entry=-2.0):
    """Mean reversion: buy when z-score < -2, sell when z-score > 0"""
    close = df["close"]
    ma = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    zscore = (close - ma) / (std + 1e-10)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    # Only buy dips in uptrend
    signals = np.where((zscore < zscore_entry) & (ema_f > ema_s), 1, 0)
    return signals


def strat_mean_reversion_rsi(df, rsi_period=14, rsi_entry=30):
    """RSI mean reversion: buy RSI < 30 in uptrend"""
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((rsi < rsi_entry) & (ema_f > ema_s), 1, 0)
    return signals


def strat_mean_reversion_bb(df, period=20, std_mult=2.0):
    """Bollinger Band mean reversion: buy at lower band in uptrend"""
    close = df["close"]
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    lower = ma - std_mult * std
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close < lower) & (ema_f > ema_s), 1, 0)
    return signals


def strat_vol_profile_short(df, lookback=48):
    """Vol Profile but SHORT only: sell breakdowns in downtrend"""
    close = df["close"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    lower = low.rolling(lookback).min().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close < lower) & (close < vwap * 0.99) & (ema_f < ema_s), -1, 0)
    return signals


def strat_vol_profile_longshort(df, lookback=48):
    """Vol Profile long AND short"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    long_sig = (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s)
    short_sig = (close < lower) & (close < vwap * 0.99) & (ema_f < ema_s)
    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals


def strat_momentum_rotation(df, lookback=48):
    """Momentum-based: buy if return over lookback > 0 and breakout"""
    close = df["close"]
    high = df["high"]
    ret = close.pct_change(lookback)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close > upper) & (ret > 0.05) & (ema_f > ema_s), 1, 0)
    return signals


def strat_vol_profile_tight_exit(df, lookback=48):
    """Vol Profile with tighter exits (for trailing stop simulation)"""
    # Same entry, exits handled in simulate with different params
    return strat_vol_profile(df, lookback)


def strat_multi_tf(df, lookback=48):
    """Multi-timeframe: 1D trend (via 24-bar proxy) + 1H breakout"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)

    # Daily trend proxy: 24h EMA
    ema_daily = close.ewm(span=24 * 20, adjust=False).mean()  # ~20-day EMA
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # Daily must be trending up too
    daily_trend = close > ema_daily
    signals = np.where(
        (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s) & daily_trend, 1, 0
    )
    return signals


def strat_vol_profile_adx(df, lookback=48, adx_period=14, adx_thresh=25):
    """Vol Profile + ADX filter: only trade in strong trends"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # ADX calculation
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr = pd.Series(
        calc_atr(high.values, low.values, close.values, adx_period), index=df.index
    )
    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(adx_period).mean()

    signals = np.where(
        (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s) & (adx > adx_thresh),
        1,
        0,
    )
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
    trailing_stop_atr=0,
):
    capital = 1.0
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []
    max_favorable = 0  # for trailing stop

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
            if unrealized_atr > max_favorable:
                max_favorable = unrealized_atr

            should_exit = False
            if bars_held >= max_bars:
                should_exit = True
            elif atr_stop > 0 and unrealized_atr < -atr_stop:
                should_exit = True
            elif profit_target_atr > 0 and unrealized_atr > profit_target_atr:
                should_exit = True
            elif trailing_stop_atr > 0 and max_favorable > trailing_stop_atr:
                # Trailing: once we hit trailing_stop_atr profit, trail at 50% of max
                if unrealized_atr < max_favorable * 0.5:
                    should_exit = True

            if should_exit:
                exit_p = c * (1 - slippage * np.sign(position))
                pnl = position * (exit_p - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0
                max_favorable = 0

        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = c * (1 + slippage * position)
            capital -= COMMISSION * position_pct
            bars_held = 0
            max_favorable = 0

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


def run_portfolio_oos(
    all_data,
    strat_fn,
    max_positions=10,
    test_bars=720,
    exit_params=None,
    position_scale=5.0,
    select_by="low_vol",
    trailing_stop_atr=0,
):
    if exit_params is None:
        exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

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

            if select_by == "momentum":
                # Select by recent momentum (highest return)
                ret = df["close"].iloc[i - 720 : i].pct_change().sum() if i > 720 else 0
                scored.append((symbol, vol, ret))
            else:
                scored.append((symbol, vol, 0))

        if select_by == "momentum":
            scored.sort(key=lambda x: -x[2])  # highest momentum first
        else:
            scored.sort(key=lambda x: x[1])  # lowest vol first

        selected = scored[:max_positions]

        for symbol, vol, _ in selected:
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
                0.0003,
                trailing_stop_atr=trailing_stop_atr,
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
    print("PHASE 15: FINAL STRATEGY SWEEP - ALL REMAINING VARIATIONS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load 1H data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data_1h = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data_1h[symbol] = df
    print(f"Loaded {len(all_data_1h)} symbols (1H)\n")

    # Load 4H data
    all_path_4h = DATA_ROOT / "binance_futures_4h"
    all_data_4h = {}
    if all_path_4h.exists():
        for f in sorted(all_path_4h.glob("*USDT.csv")):
            symbol = f.stem
            df = load_ohlcv(symbol, "4h")
            if not df.empty and len(df) > 2500:
                all_data_4h[symbol] = df
    print(f"Loaded {len(all_data_4h)} symbols (4H)\n")

    exit_default = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}
    exit_tight = {"max_bars": 48, "atr_stop": 2.0, "profit_target_atr": 6.0}
    exit_wide = {"max_bars": 96, "atr_stop": 4.0, "profit_target_atr": 10.0}
    # For 4H: scale bars proportionally (72 bars 1H = 18 bars 4H)
    exit_4h = {"max_bars": 18, "atr_stop": 3.0, "profit_target_atr": 8.0}

    all_results = []

    # -----------------------------------------------------------------
    # A. BASELINE (vol_profile, 1H, low_vol selection, default exit)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("A. BASELINE: Vol Profile 1H (for comparison)")
    print("=" * 60)
    r = run_portfolio_oos(all_data_1h, strat_vol_profile, position_scale=5.0)
    if r:
        c, p = check_criteria(r)
        all_results.append(("A_baseline_vol_profile", p, r))
        print(
            f"  [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    # -----------------------------------------------------------------
    # B. TIMEFRAME: 4H
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("B. Vol Profile on 4H timeframe")
    print("=" * 60)
    if all_data_4h:
        # 4H: lookback=12 (48h / 4h), test_bars=180 (720/4)
        def strat_vp_4h(df):
            return strat_vol_profile(df, lookback=12)

        r = run_portfolio_oos(
            all_data_4h,
            strat_vp_4h,
            test_bars=180,
            exit_params=exit_4h,
            position_scale=5.0,
        )
        if r:
            c, p = check_criteria(r)
            all_results.append(("B_vol_profile_4H", p, r))
            print(
                f"  [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
            )
        else:
            print("  SKIP (insufficient data)")
    else:
        print("  SKIP (no 4H data)")

    # -----------------------------------------------------------------
    # C. SYMBOL SELECTION: Momentum-based
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("C. Vol Profile with Momentum-based symbol selection")
    print("=" * 60)
    r = run_portfolio_oos(
        all_data_1h, strat_vol_profile, position_scale=5.0, select_by="momentum"
    )
    if r:
        c, p = check_criteria(r)
        all_results.append(("C_momentum_selection", p, r))
        print(
            f"  [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    # -----------------------------------------------------------------
    # D. EXIT VARIANTS
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("D. Exit variants")
    print("=" * 60)

    # Tight exit
    r = run_portfolio_oos(
        all_data_1h, strat_vol_profile, exit_params=exit_tight, position_scale=5.0
    )
    if r:
        c, p = check_criteria(r)
        all_results.append(("D1_tight_exit", p, r))
        print(
            f"  Tight [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    # Wide exit
    r = run_portfolio_oos(
        all_data_1h, strat_vol_profile, exit_params=exit_wide, position_scale=5.0
    )
    if r:
        c, p = check_criteria(r)
        all_results.append(("D2_wide_exit", p, r))
        print(
            f"  Wide  [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    # Trailing stop
    r = run_portfolio_oos(
        all_data_1h, strat_vol_profile, position_scale=5.0, trailing_stop_atr=4.0
    )
    if r:
        c, p = check_criteria(r)
        all_results.append(("D3_trailing_stop", p, r))
        print(
            f"  Trail [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    # -----------------------------------------------------------------
    # E. MEAN REVERSION STRATEGIES
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("E. Mean Reversion strategies (1H portfolio)")
    print("=" * 60)

    for name, fn in [
        ("E1_zscore_mr", strat_mean_reversion),
        ("E2_rsi_mr", strat_mean_reversion_rsi),
        ("E3_bollinger_mr", strat_mean_reversion_bb),
    ]:
        r = run_portfolio_oos(all_data_1h, fn, position_scale=5.0)
        if r:
            c, p = check_criteria(r)
            all_results.append((name, p, r))
            print(
                f"  {name} [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
            )

    # -----------------------------------------------------------------
    # F. SHORT / LONG-SHORT
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("F. Short and Long-Short variants")
    print("=" * 60)

    r = run_portfolio_oos(all_data_1h, strat_vol_profile_short, position_scale=5.0)
    if r:
        c, p = check_criteria(r)
        all_results.append(("F1_short_only", p, r))
        print(
            f"  Short    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    r = run_portfolio_oos(all_data_1h, strat_vol_profile_longshort, position_scale=5.0)
    if r:
        c, p = check_criteria(r)
        all_results.append(("F2_long_short", p, r))
        print(
            f"  LongShrt [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    # -----------------------------------------------------------------
    # G. ADDITIONAL FILTERS
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("G. Additional filter variants")
    print("=" * 60)

    r = run_portfolio_oos(all_data_1h, strat_momentum_rotation, position_scale=5.0)
    if r:
        c, p = check_criteria(r)
        all_results.append(("G1_momentum_filter", p, r))
        print(
            f"  Momentum [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    r = run_portfolio_oos(all_data_1h, strat_multi_tf, position_scale=5.0)
    if r:
        c, p = check_criteria(r)
        all_results.append(("G2_multi_timeframe", p, r))
        print(
            f"  MultiTF  [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    r = run_portfolio_oos(all_data_1h, strat_vol_profile_adx, position_scale=5.0)
    if r:
        c, p = check_criteria(r)
        all_results.append(("G3_adx_filter", p, r))
        print(
            f"  ADX      [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}"
        )

    # -----------------------------------------------------------------
    # H. ENSEMBLE: vol_profile + mean_reversion (if MR passes)
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("H. Ensemble check (correlation between strategies)")
    print("=" * 60)

    baseline_pr = None
    for name, p, r in all_results:
        if name == "A_baseline_vol_profile":
            baseline_pr = r.get("period_returns", [])

    if baseline_pr:
        for name, p, r in all_results:
            if name.startswith("A_"):
                continue
            pr = r.get("period_returns", [])
            if pr and baseline_pr:
                ml = min(len(baseline_pr), len(pr))
                if ml > 2:
                    corr = np.corrcoef(baseline_pr[:ml], pr[:ml])[0, 1]
                    print(
                        f"  {name} vs baseline: corr={corr:.3f} {'(LOW - ensemble!)' if corr < 0.5 else '(high)'}"
                    )

    # -----------------------------------------------------------------
    # FINAL RANKING
    # -----------------------------------------------------------------
    all_results.sort(key=lambda x: (-x[1], -x[2].get("sharpe", 0)))

    print(f"\n\n{'=' * 70}")
    print("FINAL RANKING - ALL VARIATIONS")
    print("=" * 70)

    for i, (name, passed, r) in enumerate(all_results):
        c, _ = check_criteria(r)
        fails = [k for k, v in c.items() if not v]
        fail_str = f"  FAILS: {', '.join(fails)}" if fails else ""
        print(f"  {i+1}. [{passed}/6] {name}")
        print(
            f"     Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
            f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  T={r['trade_count']}{fail_str}"
        )

    six_six = [(n, r) for n, p, r in all_results if p == 6]
    print(f"\n*** {len(six_six)} configs passed 6/6 ***")
    for name, r in six_six:
        print(f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%")

    # Any new 6/6 that beats baseline?
    baseline_sharpe = 2.52
    better = [
        (n, r)
        for n, r in six_six
        if r["sharpe"] > baseline_sharpe and n != "A_baseline_vol_profile"
    ]
    if better:
        print(f"\n*** NEW STRATEGIES BEAT BASELINE (Sharpe > {baseline_sharpe}) ***")
        for name, r in better:
            print(
                f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%"
            )
    else:
        print(
            f"\nNo new strategy beats Vol Profile baseline (Sharpe={baseline_sharpe})"
        )
        print("→ Vol Profile Breakout CONFIRMED as final strategy")

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tested": len(all_results),
        "six_six_count": len(six_six),
        "results": [
            {
                "name": n,
                "passed": int(p),
                "sharpe": float(r.get("sharpe", 0)),
                "return": float(r.get("total_return", 0)),
                "max_dd": float(r.get("max_drawdown", 0)),
                "win_rate": float(r.get("win_rate", 0)),
                "pf": float(r.get("profit_factor", 0)),
                "trades": int(r.get("trade_count", 0)),
            }
            for n, p, r in all_results
        ],
    }
    with open(RESULTS_PATH / "phase15_final_sweep.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_PATH / 'phase15_final_sweep.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
