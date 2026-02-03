#!/usr/bin/env python3
"""
Phase 18: OOS 재검증 (min_len 함정 수정)

기존 문제: min_len = 최단 종목 기준 → OOS가 3개월로 축소
수정: 각 윈도우마다 해당 시점에 데이터가 있는 종목만 참여
     → 초기 윈도우: 장기 종목만 (BTC, ETH 등)
     → 후기 윈도우: 신규 종목도 합류
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
    all_data, max_positions=10, test_bars=720, position_scale=5.0, train_bars=4320
):
    """
    수정된 OOS 포트폴리오 테스트
    - 각 종목을 개별적으로 60/40 시간 분할
    - 각 윈도우에서 해당 시점에 OOS 데이터가 있는 종목만 참여
    - 글로벌 타임라인 기반으로 윈도우 진행
    """
    exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

    # 각 종목별 OOS 시작 인덱스를 datetime 기준으로 계산
    symbol_oos = {}
    for symbol, df in all_data.items():
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) < train_bars + test_bars:
            continue
        oos_start_dt = oos_df["datetime"].iloc[0]
        symbol_oos[symbol] = {
            "df": oos_df,
            "start_dt": oos_start_dt,
            "length": len(oos_df),
        }

    if not symbol_oos:
        return None

    # 글로벌 타임라인: 가장 이른 OOS 시작부터 가장 늦은 OOS 끝까지
    all_starts = [v["start_dt"] for v in symbol_oos.values()]
    earliest_start = min(all_starts)
    latest_end = max(v["df"]["datetime"].iloc[-1] for v in symbol_oos.values())

    print(f"  OOS range: {earliest_start.date()} ~ {latest_end.date()}")
    print(f"  Symbols with OOS data: {len(symbol_oos)}")

    # 각 종목의 OOS 내 로컬 인덱스 관리
    # 윈도우를 글로벌 datetime 기준으로 진행
    # 하지만 종목마다 OOS 시작이 다르므로, 각 종목의 로컬 인덱스를 사용

    # 가장 긴 OOS를 가진 종목 기준으로 윈도우 수 결정
    max_oos_len = max(v["length"] for v in symbol_oos.values())

    equity = [1.0]
    period_returns = []
    all_trades = []
    window_details = []

    # 각 종목의 로컬 포인터: train_bars부터 시작
    # 종목마다 OOS 길이가 다르므로, 글로벌 윈도우 번호로 관리
    # 대신: 각 종목을 독립적으로 OOS 내에서 walk-forward 하고,
    # 같은 "시간대"에 해당하는 윈도우끼리 묶기

    # 방법: datetime 기반 윈도우
    # 첫 윈도우 시작: 가장 이른 OOS 시작 + train_bars (시간)
    # → 시간 기반이 복잡하므로, 종목별 인덱스 기반으로 하되
    #   "데이터가 충분한 종목만 참여" 방식 사용

    # 더 단순한 접근: 각 종목을 독립적으로 WFA 후 결과 합산
    # 이게 가장 공정한 방법

    # 종목별 독립 WFA
    symbol_results = {}
    all_period_pnls = {}  # {window_idx: [pnl1, pnl2, ...]}

    for symbol, info in symbol_oos.items():
        df = info["df"]
        length = info["length"]

        i = train_bars
        window_idx = 0
        while i + test_bars <= length:
            # 종목 선정: 이 종목의 변동성 계산
            vol = df["close"].iloc[:i].pct_change().rolling(168).std().iloc[-1]
            if np.isnan(vol) or vol == 0:
                vol = 0.01

            full = df.iloc[: i + test_bars]
            sigs = strat_vol_profile(full)
            test_sigs = sigs[i : i + test_bars]
            test_df = df.iloc[i : i + test_bars].copy().reset_index(drop=True)

            ann_vol = vol * np.sqrt(24 * 365)
            # 포지션 크기는 나중에 정규화하므로 일단 단일 종목 기준
            position_pct = min(0.10 / (ann_vol + 1e-10) / max_positions, 0.05)
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

            if window_idx not in all_period_pnls:
                all_period_pnls[window_idx] = []
            all_period_pnls[window_idx].append(
                {
                    "symbol": symbol,
                    "pnl": r["total_return"],
                    "trades": r["trades"],
                    "trade_count": r["trade_count"],
                }
            )

            i += test_bars
            window_idx += 1

    # 각 윈도우에서 변동성 낮은 상위 10개만 선택하여 합산
    # → 더 현실적: 각 윈도우별로 참여 종목 중 10개만 사용
    # 하지만 이미 위에서 종목별로 시뮬 완료. 윈도우별로 상위 10개만 합산.

    # 재계산: 윈도우별로 종목 선정 후 합산
    # 문제: 종목별 window_idx가 같은 시간대를 의미하지 않음
    # (종목마다 OOS 시작이 다르므로 window_idx=0이 다른 날짜)

    # 더 정확한 방법: datetime 기반 글로벌 윈도우
    print("\n  Recalculating with datetime-aligned windows...")

    # 글로벌 윈도우: 30일(720h) 간격으로 슬라이스
    # 시작: 가장 이른 OOS + train_bars 시간 이후
    window_start_dates = []
    # train_bars = 4320h = 180일
    first_window_start = earliest_start + pd.Timedelta(hours=train_bars)
    current = first_window_start
    while current + pd.Timedelta(hours=test_bars) <= latest_end:
        window_start_dates.append(current)
        current += pd.Timedelta(hours=test_bars)

    print(
        f"  Global windows: {len(window_start_dates)} "
        f"({window_start_dates[0].date()} ~ {window_start_dates[-1].date()})"
    )

    equity = [1.0]
    period_returns = []
    all_trades = []
    window_details = []

    for w_idx, w_start in enumerate(window_start_dates):
        w_end = w_start + pd.Timedelta(hours=test_bars)

        # 이 윈도우에 참여 가능한 종목 선정
        candidates = []
        for symbol, info in symbol_oos.items():
            df = info["df"]
            # 이 종목의 OOS 데이터가 이 윈도우를 커버하는지 확인
            # 최소: w_start 이전에 train_bars 이상의 데이터 필요
            sym_start = df["datetime"].iloc[0]
            sym_end = df["datetime"].iloc[-1]

            if sym_start > w_start - pd.Timedelta(hours=train_bars):
                continue  # 학습 데이터 부족
            if sym_end < w_end:
                continue  # 테스트 기간 미달

            # 변동성 계산 (w_start 이전 데이터로)
            mask_before = df["datetime"] < w_start
            pre_data = df[mask_before]
            if len(pre_data) < 200:
                continue
            vol = pre_data["close"].pct_change().rolling(168).std().iloc[-1]
            if np.isnan(vol) or vol == 0:
                vol = 0.01
            candidates.append((symbol, vol))

        if not candidates:
            continue

        # 변동성 낮은 순 10개 선정
        candidates.sort(key=lambda x: x[1])
        selected = candidates[:max_positions]

        period_pnl = 0
        period_trades = []
        for symbol, vol in selected:
            df = symbol_oos[symbol]["df"]
            # 윈도우 구간의 인덱스 찾기
            mask_window = (df["datetime"] >= w_start) & (df["datetime"] < w_end)
            test_indices = df.index[mask_window]
            if len(test_indices) < 100:
                continue

            # 시그널 생성: w_start 이전 데이터 + 테스트 구간
            mask_full = df["datetime"] < w_end
            full_df = df[mask_full].copy().reset_index(drop=True)
            sigs = strat_vol_profile(full_df)

            # 테스트 구간만 추출
            test_start = len(full_df) - len(test_indices)
            test_sigs = sigs[test_start:]
            test_df = full_df.iloc[test_start:].copy().reset_index(drop=True)

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

        period_returns.append(period_pnl)
        all_trades.extend(period_trades)
        equity.append(equity[-1] * (1 + period_pnl))
        window_details.append(
            {
                "window": w_idx,
                "start": str(w_start.date()),
                "end": str(w_end.date()),
                "symbols": len(selected),
                "candidates": len(candidates),
                "pnl": period_pnl,
                "trades": len(period_trades),
            }
        )

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

    result = {
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
        "window_details": window_details,
    }
    return result


def main():
    print("=" * 70)
    print("PHASE 18: OOS REVALIDATION (min_len FIX)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load all data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    data_lengths = []
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
            oos_len = int(len(df) * 0.4)
            data_lengths.append(
                (
                    symbol,
                    len(df),
                    oos_len,
                    df["datetime"].iloc[0].date(),
                    df["datetime"].iloc[-1].date(),
                )
            )

    print(f"Loaded {len(all_data)} symbols\n")

    # 데이터 커버리지 분석
    data_lengths.sort(key=lambda x: -x[1])
    print("Data coverage (top 10 longest):")
    for sym, total, oos, start, end in data_lengths[:10]:
        print(f"  {sym:<15} {total:>7} bars  OOS={oos:>6}  {start} ~ {end}")
    print(f"\nShortest 5:")
    data_lengths.sort(key=lambda x: x[1])
    for sym, total, oos, start, end in data_lengths[:5]:
        print(f"  {sym:<15} {total:>7} bars  OOS={oos:>6}  {start} ~ {end}")

    print(f"\n{'=' * 70}")
    print("TEST 1: FULL REVALIDATION (datetime-aligned, all scales)")
    print("=" * 70)

    for scale_name, scale in [("1x", 1.0), ("3x", 3.0), ("5x", 5.0)]:
        print(f"\n--- Scale {scale_name} ---")
        r = run_portfolio_oos_fixed(
            all_data, max_positions=10, test_bars=720, position_scale=scale
        )
        if r:
            c, p = check_criteria(r)
            fails = [k for k, v in c.items() if not v]
            print(
                f"\n  RESULT [{p}/6]: Sharpe={r['sharpe']:.2f} "
                f"Ret={r['total_return']*100:+.1f}% "
                f"DD={r['max_drawdown']*100:.1f}% "
                f"WR={r['win_rate']*100:.0f}% "
                f"PF={r['profit_factor']:.2f} "
                f"WFA={r['wfa_efficiency']:.0f}% "
                f"T={r['trade_count']} "
                f"Periods={r['periods']}"
            )
            if fails:
                print(f"  FAILS: {', '.join(fails)}")

            # 윈도우별 상세
            print(f"\n  Window details:")
            print(
                f"  {'#':>3} {'Start':>12} {'End':>12} {'Syms':>5} {'Cands':>6} {'PnL':>8} {'Trades':>6}"
            )
            for w in r["window_details"]:
                print(
                    f"  {w['window']:>3} {w['start']:>12} {w['end']:>12} "
                    f"{w['symbols']:>5} {w['candidates']:>6} "
                    f"{w['pnl']*100:>+7.2f}% {w['trades']:>6}"
                )
        else:
            print("  NO RESULT")

    # TEST 2: 장기 종목만 (2020년 시작) 으로 재검증
    print(f"\n{'=' * 70}")
    print("TEST 2: LONG-HISTORY ONLY (started before 2021)")
    print("=" * 70)

    long_data = {}
    for sym, df in all_data.items():
        if df["datetime"].iloc[0] < pd.Timestamp("2021-01-01"):
            long_data[sym] = df
    print(f"\nSymbols with data before 2021: {len(long_data)}")

    print(f"\n--- Scale 5x ---")
    r = run_portfolio_oos_fixed(
        long_data, max_positions=10, test_bars=720, position_scale=5.0
    )
    if r:
        c, p = check_criteria(r)
        fails = [k for k, v in c.items() if not v]
        print(
            f"\n  RESULT [{p}/6]: Sharpe={r['sharpe']:.2f} "
            f"Ret={r['total_return']*100:+.1f}% "
            f"DD={r['max_drawdown']*100:.1f}% "
            f"WR={r['win_rate']*100:.0f}% "
            f"PF={r['profit_factor']:.2f} "
            f"WFA={r['wfa_efficiency']:.0f}% "
            f"T={r['trade_count']} "
            f"Periods={r['periods']}"
        )
        if fails:
            print(f"  FAILS: {', '.join(fails)}")

        print(f"\n  Window details:")
        print(
            f"  {'#':>3} {'Start':>12} {'End':>12} {'Syms':>5} {'Cands':>6} {'PnL':>8} {'Trades':>6}"
        )
        for w in r["window_details"]:
            print(
                f"  {w['window']:>3} {w['start']:>12} {w['end']:>12} "
                f"{w['symbols']:>5} {w['candidates']:>6} "
                f"{w['pnl']*100:>+7.2f}% {w['trades']:>6}"
            )

    # TEST 3: 연도별 성과 분석 (장기 종목 사용)
    print(f"\n{'=' * 70}")
    print("TEST 3: YEARLY PERFORMANCE (long-history, 5x)")
    print("=" * 70)

    if r and r.get("window_details"):
        from collections import defaultdict

        yearly = defaultdict(lambda: {"pnl": [], "trades": 0})
        for w in r["window_details"]:
            year = w["start"][:4]
            yearly[year]["pnl"].append(w["pnl"])
            yearly[year]["trades"] += w["trades"]

        print(
            f"\n  {'Year':>6} {'Windows':>8} {'Return':>8} {'Avg/Win':>8} {'Win%':>6} {'Trades':>7}"
        )
        for year in sorted(yearly.keys()):
            y = yearly[year]
            pnls = y["pnl"]
            total_ret = 1.0
            for p in pnls:
                total_ret *= 1 + p
            total_ret -= 1
            avg = np.mean(pnls)
            win_pct = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(
                f"  {year:>6} {len(pnls):>8} {total_ret*100:>+7.1f}% "
                f"{avg*100:>+7.2f}% {win_pct:>5.0f}% {y['trades']:>7}"
            )

    print(f"\n{'=' * 70}")
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
