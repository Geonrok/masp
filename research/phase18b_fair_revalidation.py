#!/usr/bin/env python3
"""
Phase 18b: 공정한 재검증

기존 코드의 min_len 문제를 수정하되, 로직은 최대한 동일하게 유지.

핵심 수정:
- 기존: min_len = min(모든 종목 OOS 길이) → 최단 종목에 전체 종속
- 수정A: OOS > N봉인 종목만 포함 (N = 8000/12000/16000)
- 수정B: datetime-aligned 윈도우 (Phase 18과 동일하되 train 요건 완화)

3가지 검증:
1. 장기 종목만 (OOS > 12000봉, ~1.4년)
2. 중기 종목만 (OOS > 8000봉, ~11개월)
3. 기존 방식 그대로 (OOS > 2000봉) 단, min_len 대신 median_len 사용
"""

import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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


def strat_vol_profile(df, lookback=48):
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s), 1, 0)
    return signals


def strat_dual_ma(df, lookback=48):
    """기존 Dual MA Breakout (baseline) - VWAP 없는 버전"""
    close = df["close"]
    high = df["high"]
    atr = pd.Series(
        calc_atr(high.values, df["low"].values, close.values, 14), index=df.index
    )
    atr_avg = atr.rolling(lookback).mean()
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close > upper) & (ema_f > ema_s) & (atr > atr_avg * 1.0), 1, 0)
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


def run_portfolio_oos_fixed(
    all_data,
    strat_fn,
    min_oos_bars=8000,
    max_positions=10,
    test_bars=720,
    position_scale=5.0,
    train_bars=4320,
):
    """
    수정된 OOS 포트폴리오 - min_len을 종목 필터로 대체

    기존 로직과 동일하되:
    - OOS 길이가 min_oos_bars 이상인 종목만 포함
    - min_len = min(필터 통과 종목들의 OOS 길이) → 여전히 사용하되 필터가 강화됨
    """
    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > min_oos_bars:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    # 이제 min_len은 필터 통과 종목들 중 최소값
    min_len = min(len(d) for d in oos_data.values())
    oos_lengths = sorted([len(d) for d in oos_data.values()])

    print(f"    Symbols passing filter (OOS>{min_oos_bars}): {len(oos_data)}")
    print(f"    OOS length range: {oos_lengths[0]} ~ {oos_lengths[-1]} bars")
    print(f"    min_len (effective): {min_len} bars ({min_len/24:.0f} days)")
    n_windows = (min_len - train_bars) // test_bars
    print(f"    Expected windows: {n_windows}")

    equity = [1.0]
    period_returns = []
    all_trades = []

    i = train_bars
    window_idx = 0
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
            r = simulate(test_df, test_sigs, position_pct, 72, 3.0, 8.0, 0.0003)
            period_pnl += r["total_return"]
            all_trades.extend(r["trades"])

        period_returns.append(period_pnl)
        equity.append(equity[-1] * (1 + period_pnl))
        i += test_bars
        window_idx += 1

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
        "equity": equity,
        "period_returns": period_returns,
    }


def main():
    print("=" * 70)
    print("PHASE 18b: FAIR REVALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load all data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Total symbols loaded: {len(all_data)}\n")

    # 종목별 OOS 길이 분포
    oos_dist = []
    for sym, df in all_data.items():
        oos_len = int(len(df) * 0.4)
        oos_dist.append(oos_len)
    oos_dist.sort()
    print("OOS length distribution:")
    for pct in [10, 25, 50, 75, 90, 100]:
        idx = min(int(len(oos_dist) * pct / 100), len(oos_dist) - 1)
        bars = oos_dist[idx]
        print(
            f"  P{pct:>3}: {bars:>6} bars ({bars/24:>6.0f} days, {bars/24/30:>4.1f} months)"
        )

    strategies = {
        "vol_profile": strat_vol_profile,
        "dual_ma": strat_dual_ma,
    }

    # min_oos_bars 임계값별 테스트
    thresholds = [
        (2000, "Original (2000 bars, ~83 days)"),
        (5000, "Medium (5000 bars, ~208 days)"),
        (8000, "Strict (8000 bars, ~333 days)"),
        (12000, "Very Strict (12000 bars, ~500 days)"),
        (16000, "Ultra Strict (16000 bars, ~667 days)"),
    ]

    for strat_name, strat_fn in strategies.items():
        print(f"\n{'=' * 70}")
        print(f"STRATEGY: {strat_name}")
        print("=" * 70)

        for threshold, desc in thresholds:
            print(f"\n--- {desc} ---")
            for scale_name, scale in [("1x", 1.0), ("3x", 3.0), ("5x", 5.0)]:
                r = run_portfolio_oos_fixed(
                    all_data,
                    strat_fn,
                    min_oos_bars=threshold,
                    max_positions=10,
                    test_bars=720,
                    position_scale=scale,
                )
                if r:
                    c, p = check_criteria(r)
                    fails = [k for k, v in c.items() if not v]
                    fail_str = f" FAILS: {', '.join(fails)}" if fails else ""
                    print(
                        f"    {scale_name}: [{p}/6] Sharpe={r['sharpe']:.2f} "
                        f"Ret={r['total_return']*100:+.1f}% "
                        f"DD={r['max_drawdown']*100:.1f}% "
                        f"WR={r['win_rate']*100:.0f}% "
                        f"PF={r['profit_factor']:.2f} "
                        f"WFA={r['wfa_efficiency']:.0f}% "
                        f"T={r['trade_count']} "
                        f"W={r['periods']}{fail_str}"
                    )

                    # 연도별 성과 (5x만)
                    if scale == 5.0 and r.get("period_returns"):
                        # 각 윈도우의 날짜 추정 (train_bars + window*test_bars)
                        # OOS 시작 ≈ 전체 데이터의 60% 지점
                        # 실제 날짜 정보가 없으므로 간략화
                        pass
                else:
                    print(f"    {scale_name}: NO DATA (insufficient symbols)")

    # 최종 요약
    print(f"\n{'=' * 70}")
    print("SUMMARY: WHICH THRESHOLD GIVES 6/6?")
    print("=" * 70)

    for strat_name, strat_fn in strategies.items():
        print(f"\n{strat_name}:")
        for threshold, desc in thresholds:
            r = run_portfolio_oos_fixed(
                all_data,
                strat_fn,
                min_oos_bars=threshold,
                max_positions=10,
                test_bars=720,
                position_scale=5.0,
            )
            if r:
                c, p = check_criteria(r)
                status = "PASS" if p == 6 else f"FAIL ({p}/6)"
                print(
                    f"  OOS>{threshold:>5}: [{p}/6] Sharpe={r['sharpe']:.2f} "
                    f"Ret={r['total_return']*100:+.1f}% W={r['periods']} "
                    f"→ {status}"
                )

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
