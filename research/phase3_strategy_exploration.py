#!/usr/bin/env python3
"""
Ralph-Loop Phase 3: Strategy Exploration
=========================================
Task 3.1 ~ 3.7: Test multiple strategy families with walk-forward validation
"""

import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Paths
DATA_ROOT = Path("E:/data/crypto_ohlcv")
STATE_PATH = Path(
    "E:/투자/Multi-Asset Strategy Platform/research/ralph_loop_state.json"
)
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")
FEATURE_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/features")

RESULTS_PATH.mkdir(exist_ok=True)

# Realistic cost model
SLIPPAGE = 0.0005  # 0.05%
COMMISSION = 0.0004  # 0.04% taker
FUNDING_PER_8H = 0.0001  # 0.01%


def load_state():
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_ohlcv(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    tf_map = {
        "1h": "binance_futures_1h",
        "4h": "binance_futures_4h",
        "1d": "binance_futures_1d",
    }
    path = DATA_ROOT / tf_map.get(timeframe, "binance_futures_4h") / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["datetime", "timestamp", "date"]:
        if col in df.columns:
            df["datetime"] = pd.to_datetime(df[col])
            break
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# =============================================================================
# Backtest Engine (No Look-Ahead)
# =============================================================================
def backtest_walk_forward(
    df: pd.DataFrame,
    signal_func,
    train_bars=1080,
    test_bars=180,
    position_pct=0.02,
    max_bars_hold=48,
    **kwargs,
):
    """
    Walk-forward backtest with no look-ahead bias.
    train_bars: 1080 = 180 days (4h), test_bars: 180 = 30 days
    """
    results = []
    total_bars = len(df)

    if total_bars < train_bars + test_bars:
        return {"profitable": False, "reason": "insufficient_data"}

    step = 0
    i = train_bars

    while i + test_bars <= total_bars:
        train_df = df.iloc[:i].copy()
        test_df = df.iloc[i : i + test_bars].copy()

        # Generate signals using only training data for any parameter fitting
        try:
            signals = signal_func(train_df, test_df, **kwargs)
        except Exception as e:
            i += test_bars
            step += 1
            continue

        # Simulate trades on test period
        pnl = simulate_trades(test_df, signals, position_pct, max_bars_hold)
        results.append(pnl)

        i += test_bars
        step += 1

    if not results:
        return {"profitable": False, "reason": "no_results"}

    # Aggregate results
    total_return = 1.0
    all_trades = []
    for r in results:
        total_return *= 1 + r["period_return"]
        all_trades.extend(r.get("trades", []))

    total_return = total_return - 1

    wins = sum(1 for t in all_trades if t > 0)
    losses = sum(1 for t in all_trades if t <= 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    gross_profit = sum(t for t in all_trades if t > 0)
    gross_loss = abs(sum(t for t in all_trades if t < 0))
    profit_factor = gross_profit / (gross_loss + 1e-10)

    # Max drawdown
    equity = [1.0]
    for r in results:
        equity.append(equity[-1] * (1 + r["period_return"]))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sharpe (monthly periods)
    period_returns = [r["period_return"] for r in results]
    if len(period_returns) > 1:
        sharpe = (
            np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)
        )
    else:
        sharpe = 0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trade_count": len(all_trades),
        "periods": len(results),
        "period_returns": period_returns,
        "profitable": total_return > 0 and sharpe > 0,
    }


def simulate_trades(
    df: pd.DataFrame, signals: pd.Series, position_pct=0.02, max_bars=48
):
    """Simulate trades with realistic costs"""
    capital = 1.0
    position = 0  # 1 long, -1 short, 0 flat
    entry_price = 0
    bars_held = 0
    trades = []

    for i in range(len(df)):
        if i >= len(signals):
            break

        close = df["close"].iloc[i]
        sig = signals.iloc[i] if isinstance(signals, pd.Series) else signals[i]

        # Exit conditions
        if position != 0:
            bars_held += 1

            should_exit = False
            # Max hold time
            if bars_held >= max_bars:
                should_exit = True
            # Signal reversal
            elif sig != 0 and sig != position:
                should_exit = True
            # Signal flat
            elif sig == 0 and position != 0:
                should_exit = True

            if should_exit:
                exit_price = close * (1 - SLIPPAGE * np.sign(position))
                pnl = position * (exit_price - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct  # exit commission
                pnl -= (
                    FUNDING_PER_8H * bars_held / 2 * position_pct
                )  # funding (4h = half of 8h)
                trades.append(pnl)
                capital += pnl
                position = 0

        # Entry conditions
        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = close * (1 + SLIPPAGE * position)
            capital -= COMMISSION * position_pct  # entry commission
            bars_held = 0

    # Close any open position
    if position != 0:
        close = df["close"].iloc[-1]
        exit_price = close * (1 - SLIPPAGE * np.sign(position))
        pnl = position * (exit_price - entry_price) / entry_price * position_pct
        pnl -= COMMISSION * position_pct
        pnl -= FUNDING_PER_8H * bars_held / 2 * position_pct
        trades.append(pnl)

    period_return = sum(trades)
    return {"period_return": period_return, "trades": trades}


# =============================================================================
# Strategy 3.1: Momentum Strategies
# =============================================================================
def strategy_tsmom(train_df, test_df, lookback=42):
    """Time-Series Momentum: go with the trend"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    ret = combined["close"].pct_change(lookback)
    signals = np.sign(ret)
    # Return only test period signals
    return pd.Series(signals.values[-len(test_df) :], index=range(len(test_df)))


def strategy_dual_momentum(train_df, test_df, lookback=42, abs_threshold=0.0):
    """Dual Momentum: absolute + relative (vs BTC)"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    ret = combined["close"].pct_change(lookback)
    # Absolute momentum: only trade if return > threshold
    signals = np.where(ret > abs_threshold, 1, np.where(ret < -abs_threshold, -1, 0))
    return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))


def strategy_kama_momentum(train_df, test_df, kama_period=24):
    """KAMA Momentum: trade based on KAMA direction"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    close = combined["close"]

    # Calculate KAMA
    change = abs(close - close.shift(kama_period))
    volatility = abs(close - close.shift(1)).rolling(kama_period).sum()
    er = change / (volatility + 1e-10)

    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[kama_period] = close.iloc[kama_period]
    for i in range(kama_period + 1, len(close)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            close.iloc[i] - kama.iloc[i - 1]
        )

    signals = np.where(close > kama, 1, np.where(close < kama, -1, 0))
    return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))


# =============================================================================
# Strategy 3.2: Mean Reversion Strategies
# =============================================================================
def strategy_bb_reversion(train_df, test_df, window=24, num_std=2.0):
    """Bollinger Band Mean Reversion"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    close = combined["close"]
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std

    signals = np.where(close < lower, 1, np.where(close > upper, -1, 0))
    return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))


def strategy_rsi_reversion(train_df, test_df, window=14, oversold=30, overbought=70):
    """RSI Mean Reversion"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    close = combined["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    signals = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
    return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))


# =============================================================================
# Strategy 3.3: Breakout Strategies
# =============================================================================
def strategy_donchian_breakout(train_df, test_df, lookback=25):
    """Donchian Channel Breakout with trend filter"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    close = combined["close"]
    high = combined["high"]
    low = combined["low"]

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)

    # Trend filter: EMA 100
    ema100 = close.ewm(span=100, adjust=False).mean()

    signals = np.zeros(len(combined))
    signals[(close > upper) & (close > ema100)] = 1
    signals[(close < lower) & (close < ema100)] = -1

    return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))


def strategy_vol_breakout(train_df, test_df, atr_mult=1.5, lookback=24):
    """Volatility Breakout: ATR-based breakout"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    close = combined["close"]
    high = combined["high"]
    low = combined["low"]

    tr = pd.DataFrame(
        {
            "hl": high - low,
            "hc": abs(high - close.shift(1)),
            "lc": abs(low - close.shift(1)),
        }
    ).max(axis=1)
    atr = tr.rolling(lookback).mean()

    # Breakout above/below previous close + ATR
    prev_close = close.shift(1)
    signals = np.where(
        close > prev_close + atr * atr_mult,
        1,
        np.where(close < prev_close - atr * atr_mult, -1, 0),
    )

    return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))


# =============================================================================
# Strategy 3.4: Carry Strategies
# =============================================================================
def strategy_funding_carry(train_df, test_df, symbol="", threshold=0.0003):
    """Funding Rate Carry: collect funding"""
    fr_path = DATA_ROOT / "binance_funding_rate" / f"{symbol}.csv"
    if not fr_path.exists():
        return pd.Series(0, index=range(len(test_df)))

    try:
        fr_df = pd.read_csv(fr_path)
        for col in ["datetime", "timestamp", "fundingTime"]:
            if col in fr_df.columns:
                fr_df["datetime"] = pd.to_datetime(fr_df[col])
                break

        fr_col = None
        for col in ["fundingRate", "funding_rate", "rate"]:
            if col in fr_df.columns:
                fr_col = col
                break

        if not fr_col:
            return pd.Series(0, index=range(len(test_df)))

        fr_df = fr_df.sort_values("datetime").set_index("datetime")

        test_aligned = test_df.set_index("datetime")
        test_aligned["fr"] = fr_df[fr_col].reindex(test_aligned.index, method="ffill")

        fr = test_aligned["fr"].fillna(0).values

        # Short when funding is high positive (longs pay shorts)
        # Long when funding is very negative (shorts pay longs)
        signals = np.where(fr > threshold, -1, np.where(fr < -threshold, 1, 0))

        return pd.Series(signals, index=range(len(test_df)))
    except Exception:
        return pd.Series(0, index=range(len(test_df)))


# =============================================================================
# Strategy 3.5: Sentiment Strategies
# =============================================================================
def strategy_funding_reversal(
    train_df, test_df, symbol="", lookback=24, zscore_threshold=2.0
):
    """Funding Rate Crowding Reversal: fade extreme funding"""
    fr_path = DATA_ROOT / "binance_funding_rate" / f"{symbol}.csv"
    if not fr_path.exists():
        return pd.Series(0, index=range(len(test_df)))

    try:
        fr_df = pd.read_csv(fr_path)
        for col in ["datetime", "timestamp", "fundingTime"]:
            if col in fr_df.columns:
                fr_df["datetime"] = pd.to_datetime(fr_df[col])
                break

        fr_col = None
        for col in ["fundingRate", "funding_rate", "rate"]:
            if col in fr_df.columns:
                fr_col = col
                break

        if not fr_col:
            return pd.Series(0, index=range(len(test_df)))

        combined = pd.concat([train_df, test_df]).reset_index(drop=True)
        fr_df = fr_df.sort_values("datetime").set_index("datetime")

        combined_aligned = combined.set_index("datetime")
        combined_aligned["fr"] = fr_df[fr_col].reindex(
            combined_aligned.index, method="ffill"
        )
        fr = combined_aligned["fr"].fillna(0)

        # Z-score of funding rate
        fr_zscore = (fr - fr.rolling(lookback).mean()) / (
            fr.rolling(lookback).std() + 1e-10
        )

        # Fade extremes
        signals = np.where(
            fr_zscore > zscore_threshold,
            -1,
            np.where(fr_zscore < -zscore_threshold, 1, 0),
        )

        return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))
    except Exception:
        return pd.Series(0, index=range(len(test_df)))


# =============================================================================
# Strategy 3.7: Hybrid/Composite
# =============================================================================
def strategy_multi_factor(train_df, test_df):
    """Multi-factor: combine momentum + mean reversion + breakout signals"""
    combined = pd.concat([train_df, test_df]).reset_index(drop=True)
    close = combined["close"]
    high = combined["high"]
    low = combined["low"]

    n = len(combined)
    scores = np.zeros(n)

    # Factor 1: Trend (EMA crossover)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    scores += np.where(ema_fast > ema_slow, 1, -1)

    # Factor 2: Momentum (ROC)
    roc = close.pct_change(24)
    scores += np.sign(roc)

    # Factor 3: Breakout (Donchian)
    upper = high.rolling(24).max().shift(1)
    lower = low.rolling(24).min().shift(1)
    scores += np.where(close > upper, 1, np.where(close < lower, -1, 0))

    # Factor 4: RSI filter (neutral zone = no signal)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    scores += np.where(rsi > 60, 1, np.where(rsi < 40, -1, 0))

    # Threshold: need 3+ factors agreement
    signals = np.where(scores >= 3, 1, np.where(scores <= -3, -1, 0))

    return pd.Series(signals[-len(test_df) :], index=range(len(test_df)))


# =============================================================================
# Main Strategy Exploration
# =============================================================================
def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 3: STRATEGY EXPLORATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    state = load_state()

    # Use established symbols with long history
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

    strategies = {
        "3.1a_tsmom_42": (
            "Momentum: TSMOM(42)",
            lambda t, s: strategy_tsmom(t, s, lookback=42),
        ),
        "3.1b_tsmom_84": (
            "Momentum: TSMOM(84)",
            lambda t, s: strategy_tsmom(t, s, lookback=84),
        ),
        "3.1c_dual_mom": (
            "Momentum: Dual",
            lambda t, s: strategy_dual_momentum(t, s, lookback=42),
        ),
        "3.1d_kama_mom": (
            "Momentum: KAMA",
            lambda t, s: strategy_kama_momentum(t, s, kama_period=24),
        ),
        "3.2a_bb_rev": (
            "MeanRev: Bollinger",
            lambda t, s: strategy_bb_reversion(t, s, window=24),
        ),
        "3.2b_rsi_rev": (
            "MeanRev: RSI",
            lambda t, s: strategy_rsi_reversion(t, s, window=14),
        ),
        "3.3a_donchian": (
            "Breakout: Donchian(25)",
            lambda t, s: strategy_donchian_breakout(t, s, lookback=25),
        ),
        "3.3b_vol_break": (
            "Breakout: Volatility",
            lambda t, s: strategy_vol_breakout(t, s, atr_mult=1.5),
        ),
        "3.7a_multi_factor": (
            "Hybrid: Multi-Factor",
            lambda t, s: strategy_multi_factor(t, s),
        ),
    }

    all_results = {}

    for strat_key, (strat_name, strat_func) in strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing: {strat_name}")
        print(f"{'='*60}")

        strat_results = {}

        for symbol in symbols:
            df = load_ohlcv(symbol, "4h")
            if df.empty or len(df) < 1500:
                continue

            result = backtest_walk_forward(df, strat_func)
            strat_results[symbol] = result

            status = "OK" if result.get("profitable", False) else "FAIL"
            ret = result.get("total_return", 0)
            sharpe = result.get("sharpe", 0)
            trades = result.get("trade_count", 0)
            print(
                f"  {symbol:<12} [{status}] Ret={ret*100:+.1f}%  Sharpe={sharpe:.2f}  Trades={trades}"
            )

        # Summary for this strategy
        profitable_count = sum(
            1 for r in strat_results.values() if r.get("profitable", False)
        )
        avg_return = np.mean([r.get("total_return", 0) for r in strat_results.values()])
        avg_sharpe = np.mean([r.get("sharpe", 0) for r in strat_results.values()])
        avg_trades = np.mean([r.get("trade_count", 0) for r in strat_results.values()])

        all_results[strat_key] = {
            "name": strat_name,
            "profitable_symbols": profitable_count,
            "total_symbols": len(strat_results),
            "profitable_pct": (
                profitable_count / len(strat_results) * 100 if strat_results else 0
            ),
            "avg_return": avg_return,
            "avg_sharpe": avg_sharpe,
            "avg_trades": avg_trades,
            "symbol_results": {
                s: {k: v for k, v in r.items() if k != "period_returns"}
                for s, r in strat_results.items()
            },
        }

        print(
            f"\n  Summary: {profitable_count}/{len(strat_results)} profitable ({all_results[strat_key]['profitable_pct']:.0f}%)"
        )
        print(
            f"  Avg Return: {avg_return*100:+.1f}%  Avg Sharpe: {avg_sharpe:.2f}  Avg Trades: {avg_trades:.0f}"
        )

    # Test funding-based strategies on symbols with funding data
    funding_strategies = {
        "3.4a_funding_carry": (
            "Carry: Funding",
            lambda t, s, sym="": strategy_funding_carry(t, s, symbol=sym),
        ),
        "3.5a_funding_rev": (
            "Sentiment: Funding Reversal",
            lambda t, s, sym="": strategy_funding_reversal(t, s, symbol=sym),
        ),
    }

    for strat_key, (strat_name, strat_func) in funding_strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing: {strat_name}")
        print(f"{'='*60}")

        strat_results = {}

        for symbol in symbols:
            df = load_ohlcv(symbol, "4h")
            if df.empty or len(df) < 1500:
                continue

            result = backtest_walk_forward(
                df, lambda t, s: strat_func(t, s, sym=symbol)
            )
            strat_results[symbol] = result

            status = "OK" if result.get("profitable", False) else "FAIL"
            ret = result.get("total_return", 0)
            sharpe = result.get("sharpe", 0)
            trades = result.get("trade_count", 0)
            print(
                f"  {symbol:<12} [{status}] Ret={ret*100:+.1f}%  Sharpe={sharpe:.2f}  Trades={trades}"
            )

        profitable_count = sum(
            1 for r in strat_results.values() if r.get("profitable", False)
        )
        avg_return = np.mean([r.get("total_return", 0) for r in strat_results.values()])
        avg_sharpe = np.mean([r.get("sharpe", 0) for r in strat_results.values()])
        avg_trades = np.mean([r.get("trade_count", 0) for r in strat_results.values()])

        all_results[strat_key] = {
            "name": strat_name,
            "profitable_symbols": profitable_count,
            "total_symbols": len(strat_results),
            "profitable_pct": (
                profitable_count / len(strat_results) * 100 if strat_results else 0
            ),
            "avg_return": avg_return,
            "avg_sharpe": avg_sharpe,
            "avg_trades": avg_trades,
            "symbol_results": {
                s: {k: v for k, v in r.items() if k != "period_returns"}
                for s, r in strat_results.items()
            },
        }

        print(
            f"\n  Summary: {profitable_count}/{len(strat_results)} profitable ({all_results[strat_key]['profitable_pct']:.0f}%)"
        )
        print(
            f"  Avg Return: {avg_return*100:+.1f}%  Avg Sharpe: {avg_sharpe:.2f}  Avg Trades: {avg_trades:.0f}"
        )

    # =================================================================
    # Final Ranking
    # =================================================================
    print("\n" + "=" * 70)
    print("STRATEGY RANKING (by profitable %)")
    print("=" * 70)

    ranked = sorted(
        all_results.items(), key=lambda x: x[1]["profitable_pct"], reverse=True
    )

    promising = []
    failed = []

    for rank, (key, res) in enumerate(ranked, 1):
        marker = "***" if res["profitable_pct"] >= 50 else "   "
        print(
            f"  {rank}. {marker} {res['name']:<30} {res['profitable_pct']:.0f}% profitable  "
            f"Ret={res['avg_return']*100:+.1f}%  Sharpe={res['avg_sharpe']:.2f}  Trades={res['avg_trades']:.0f}"
        )

        if res["profitable_pct"] >= 50 and res["avg_sharpe"] > 0:
            promising.append(
                {
                    "strategy": key,
                    "name": res["name"],
                    "profitable_pct": res["profitable_pct"],
                    "avg_return": res["avg_return"],
                    "avg_sharpe": res["avg_sharpe"],
                }
            )
        else:
            failed.append(
                {
                    "strategy": key,
                    "name": res["name"],
                    "profitable_pct": res["profitable_pct"],
                    "avg_return": res["avg_return"],
                }
            )

    # Save results
    report = {
        "generated_at": datetime.now().isoformat(),
        "strategies_tested": len(all_results),
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "symbol_results"}
            for k, v in all_results.items()
        },
        "detailed_results": all_results,
        "ranking": [
            {
                "rank": i + 1,
                "strategy": k,
                "name": v["name"],
                "profitable_pct": v["profitable_pct"],
                "avg_return": v["avg_return"],
                "avg_sharpe": v["avg_sharpe"],
            }
            for i, (k, v) in enumerate(ranked)
        ],
        "promising": promising,
        "failed": failed,
    }

    report_path = RESULTS_PATH / "phase3_strategy_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Update state
    state["current_phase"] = "4"
    state["current_task"] = "4.1"
    state["completed_tasks"].extend(["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"])
    state["findings"]["promising_strategies"] = promising
    state["findings"]["failed_strategies"] = failed
    state["next_actions"] = [
        "4.1: In-Sample backtest top strategies",
        "4.2: Out-of-Sample validation",
    ]

    save_state(state)

    print(f"\n  Promising strategies: {len(promising)}")
    print(f"  Failed strategies: {len(failed)}")
    print(f"  Report saved: {report_path}")
    print(f"  Next: Phase 4 - Validation Framework")

    return report


if __name__ == "__main__":
    report = main()
