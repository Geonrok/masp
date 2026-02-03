#!/usr/bin/env python3
"""
Phase 18d: 수정 기준 6/6 통과 7개 전략의 시장 레짐별 성과 분석

- BTC 30일 수익률 기준 레짐 분류: Bull(>10%), Bear(<-10%), Sideways
- 각 윈도우를 레짐별로 분류 후 전략별 성과 비교
- 횡보/하락장 생존 능력이 채택 기준
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


# ============================================================
# STRATEGIES (6/6 revised criteria)
# ============================================================


def strat_vol_profile(df, lookback=48):
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s), 1, 0)


def strat_dual_ma(df, lookback=48):
    close = df["close"]
    high = df["high"]
    atr = pd.Series(
        calc_atr(high.values, df["low"].values, close.values, 14), index=df.index
    )
    atr_avg = atr.rolling(lookback).mean()
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (ema_f > ema_s) & (atr > atr_avg), 1, 0)


def strat_obv_trend(df, lookback=48):
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    obv = (np.sign(close.diff()) * vol).cumsum()
    obv_ma = obv.rolling(lookback).mean()
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (obv > obv_ma) & (ema_f > ema_s), 1, 0)


def strat_mfi_breakout(df, lookback=48, mfi_thresh=60):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    tp = (high + low + close) / 3
    mf = tp * vol
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp <= tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (mfi > mfi_thresh) & (ema_f > ema_s), 1, 0)


def strat_vwap_breakout(df, lookback=48):
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (close > vwap * 1.02) & (ema_f > ema_s), 1, 0)


def strat_momentum_filter(df, lookback=48):
    close = df["close"]
    high = df["high"]
    ret = close.pct_change(lookback)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (ret > 0.05) & (ema_f > ema_s), 1, 0)


def strat_dm_wide_exit(df, lookback=48):
    """Dual MA with wide exit - signal only, exit handled separately"""
    return strat_dual_ma(df, lookback)


# ============================================================
# SIMULATION
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


def run_portfolio_with_regime(
    all_data,
    btc_df,
    strat_fn,
    min_oos_bars=16000,
    max_positions=10,
    test_bars=720,
    position_scale=5.0,
    train_bars=4320,
    exit_params=None,
):
    """OOS 포트폴리오 + 윈도우별 BTC 레짐 태깅"""
    if exit_params is None:
        exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > min_oos_bars:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    # BTC OOS for regime classification
    btc_split = int(len(btc_df) * 0.6)
    btc_oos = btc_df.iloc[btc_split:].copy().reset_index(drop=True)

    min_len = min(len(d) for d in oos_data.values())

    windows = []
    i = train_bars
    while i + test_bars <= min_len:
        # BTC 30-day return for regime
        if i + test_bars <= len(btc_oos):
            btc_start = btc_oos["close"].iloc[i]
            btc_end = btc_oos["close"].iloc[min(i + test_bars - 1, len(btc_oos) - 1)]
            btc_ret = (btc_end - btc_start) / btc_start
            if btc_ret > 0.10:
                regime = "BULL"
            elif btc_ret < -0.10:
                regime = "BEAR"
            else:
                regime = "SIDEWAYS"
        else:
            regime = "UNKNOWN"
            btc_ret = 0

        # Portfolio simulation for this window
        period_pnl = 0
        period_trades = []
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
                0.0003,
            )
            period_pnl += r["total_return"]
            period_trades.extend(r["trades"])

        windows.append(
            {
                "window_idx": len(windows),
                "regime": regime,
                "btc_return": btc_ret,
                "pnl": period_pnl,
                "trades": len(period_trades),
                "trade_pnls": period_trades,
            }
        )
        i += test_bars

    return windows


def main():
    print("=" * 80)
    print("PHASE 18d: REGIME ANALYSIS - WHICH STRATEGY SURVIVES ALL CONDITIONS?")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Loaded {len(all_data)} symbols")

    btc_df = all_data.get("BTCUSDT")
    if btc_df is None:
        print("ERROR: BTCUSDT not found")
        return

    strategies = {
        "dual_ma": (
            strat_dual_ma,
            {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        ),
        "vol_profile": (
            strat_vol_profile,
            {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        ),
        "obv_trend": (
            strat_obv_trend,
            {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        ),
        "mfi_60": (
            lambda df: strat_mfi_breakout(df, mfi_thresh=60),
            {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        ),
        "vwap_bkout": (
            strat_vwap_breakout,
            {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        ),
        "momentum": (
            strat_momentum_filter,
            {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        ),
        "dm_wide": (
            strat_dual_ma,
            {"max_bars": 96, "atr_stop": 4.0, "profit_target_atr": 10.0},
        ),
    }

    all_results = {}
    for name, (fn, exit_p) in strategies.items():
        windows = run_portfolio_with_regime(
            all_data, btc_df, fn, min_oos_bars=16000, exit_params=exit_p
        )
        all_results[name] = windows

    # 레짐 분포 출력
    if all_results:
        sample = list(all_results.values())[0]
        print(f"\nTotal windows: {len(sample)}")
        regimes = [w["regime"] for w in sample]
        for r in ["BULL", "BEAR", "SIDEWAYS"]:
            cnt = regimes.count(r)
            print(f"  {r}: {cnt} windows ({cnt/len(regimes)*100:.0f}%)")

        # 윈도우 상세
        print("\n  Window BTC returns:")
        for w in sample:
            print(
                f"    W{w['window_idx']:>2}: BTC {w['btc_return']*100:>+6.1f}% → {w['regime']}"
            )

    # 레짐별 전략 성과
    print(f"\n{'=' * 80}")
    print("REGIME-SPECIFIC PERFORMANCE (5x scale)")
    print("=" * 80)

    for regime in ["BULL", "BEAR", "SIDEWAYS", "ALL"]:
        print(f"\n--- {regime} ---")
        print(
            f"  {'Strategy':<15} {'Return':>8} {'AvgPnL':>8} {'WinRate':>8} {'Windows':>8} {'MaxLoss':>8}"
        )

        regime_scores = []
        for name, windows in all_results.items():
            if regime == "ALL":
                rw = windows
            else:
                rw = [w for w in windows if w["regime"] == regime]

            if not rw:
                print(f"  {name:<15} {'N/A':>8}")
                continue

            pnls = [w["pnl"] for w in rw]
            total_ret = 1.0
            for p in pnls:
                total_ret *= 1 + p
            total_ret -= 1
            avg_pnl = np.mean(pnls)
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            max_loss = min(pnls)

            print(
                f"  {name:<15} {total_ret*100:>+7.1f}% {avg_pnl*100:>+7.2f}% "
                f"{win_rate:>7.0f}% {len(rw):>8} {max_loss*100:>+7.2f}%"
            )

            regime_scores.append(
                (name, total_ret, avg_pnl, win_rate, max_loss, len(rw))
            )

        # Rank
        if regime_scores and regime != "ALL":
            best = sorted(regime_scores, key=lambda x: -x[1])
            print(f"  → Best in {regime}: {best[0][0]} ({best[0][1]*100:+.1f}%)")

    # 종합 점수: 레짐별 순위 합산
    print(f"\n{'=' * 80}")
    print("COMPOSITE SCORE (lower = better)")
    print("  Score = rank in BULL + rank in BEAR + rank in SIDEWAYS")
    print("  Penalize: negative return in any regime → +10")
    print("=" * 80)

    strat_names = list(all_results.keys())
    regime_ranks = {}

    for regime in ["BULL", "BEAR", "SIDEWAYS"]:
        scores = []
        for name, windows in all_results.items():
            rw = [w for w in windows if w["regime"] == regime]
            if not rw:
                scores.append((name, -999))
                continue
            pnls = [w["pnl"] for w in rw]
            total_ret = 1.0
            for p in pnls:
                total_ret *= 1 + p
            scores.append((name, total_ret - 1))

        scores.sort(key=lambda x: -x[1])
        for rank, (name, ret) in enumerate(scores):
            if name not in regime_ranks:
                regime_ranks[name] = {"ranks": {}, "returns": {}}
            regime_ranks[name]["ranks"][regime] = rank + 1
            regime_ranks[name]["returns"][regime] = ret

    print(
        f"\n  {'Strategy':<15} {'BULL':>6} {'BEAR':>6} {'SIDE':>6} {'Score':>6} {'Bull%':>8} {'Bear%':>8} {'Side%':>8}"
    )
    composite = []
    for name in strat_names:
        info = regime_ranks[name]
        r = info["ranks"]
        ret = info["returns"]
        score = r.get("BULL", 8) + r.get("BEAR", 8) + r.get("SIDEWAYS", 8)
        # Penalty for negative return in any regime
        for regime in ["BULL", "BEAR", "SIDEWAYS"]:
            if ret.get(regime, 0) < 0:
                score += 10
        composite.append((name, score, r, ret))
        print(
            f"  {name:<15} {r.get('BULL','?'):>6} {r.get('BEAR','?'):>6} "
            f"{r.get('SIDEWAYS','?'):>6} {score:>6} "
            f"{ret.get('BULL',0)*100:>+7.1f}% {ret.get('BEAR',0)*100:>+7.1f}% "
            f"{ret.get('SIDEWAYS',0)*100:>+7.1f}%"
        )

    composite.sort(key=lambda x: x[1])
    print("\n  RANKING (best to worst):")
    for i, (name, score, r, ret) in enumerate(composite):
        neg_regimes = [
            reg for reg in ["BULL", "BEAR", "SIDEWAYS"] if ret.get(reg, 0) < 0
        ]
        neg_str = (
            f" WARNING NEGATIVE in: {', '.join(neg_regimes)}" if neg_regimes else ""
        )
        print(f"  {i+1}. {name} (score={score}){neg_str}")

    # 최종: 상위 3개의 상관관계
    print(f"\n{'=' * 80}")
    print("CORRELATION BETWEEN TOP 3 STRATEGIES")
    print("=" * 80)

    top3 = [c[0] for c in composite[:3]]
    for i, n1 in enumerate(top3):
        for n2 in top3[i + 1 :]:
            w1 = [w["pnl"] for w in all_results[n1]]
            w2 = [w["pnl"] for w in all_results[n2]]
            if len(w1) == len(w2) and len(w1) > 2:
                corr = np.corrcoef(w1, w2)[0, 1]
                print(f"  {n1} vs {n2}: correlation = {corr:.3f}")

    # 앙상블 테스트 (상위 2-3개 평균)
    print(f"\n{'=' * 80}")
    print("ENSEMBLE TEST (equal-weight average of top strategies)")
    print("=" * 80)

    for n_top in [2, 3]:
        top_n = [c[0] for c in composite[:n_top]]
        ensemble_pnls = []
        for w_idx in range(len(all_results[top_n[0]])):
            avg_pnl = np.mean([all_results[name][w_idx]["pnl"] for name in top_n])
            ensemble_pnls.append(avg_pnl)

        eq = [1.0]
        for p in ensemble_pnls:
            eq.append(eq[-1] * (1 + p))
        eq_arr = np.array(eq)
        peak = np.maximum.accumulate(eq_arr)
        dd = ((eq_arr - peak) / peak).min()
        sharpe = np.mean(ensemble_pnls) / (np.std(ensemble_pnls) + 1e-10) * np.sqrt(12)
        total_ret = eq_arr[-1] - 1
        wfa = sum(1 for p in ensemble_pnls if p > 0) / len(ensemble_pnls) * 100

        # Regime breakdown
        print(f"\n  Ensemble Top-{n_top} ({', '.join(top_n)}):")
        print(
            f"    Overall: Sharpe={sharpe:.2f} Ret={total_ret*100:+.1f}% DD={dd*100:.1f}% WFA={wfa:.0f}%"
        )

        for regime in ["BULL", "BEAR", "SIDEWAYS"]:
            regime_pnls = []
            for w_idx in range(len(all_results[top_n[0]])):
                if all_results[top_n[0]][w_idx]["regime"] == regime:
                    avg_pnl = np.mean(
                        [all_results[name][w_idx]["pnl"] for name in top_n]
                    )
                    regime_pnls.append(avg_pnl)
            if regime_pnls:
                r_total = 1.0
                for p in regime_pnls:
                    r_total *= 1 + p
                r_wr = sum(1 for p in regime_pnls if p > 0) / len(regime_pnls) * 100
                print(
                    f"    {regime}: Ret={(r_total-1)*100:+.1f}% WinRate={r_wr:.0f}% Windows={len(regime_pnls)}"
                )

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
