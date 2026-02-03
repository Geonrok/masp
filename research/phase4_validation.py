#!/usr/bin/env python3
"""
Ralph-Loop Phase 4: Validation Framework
==========================================
Task 4.1~4.5: Rigorous validation of top strategies from Phase 3
Top strategies: TSMOM(84), TSMOM(42), Multi-Factor Hybrid
"""

import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

DATA_ROOT = Path("E:/data/crypto_ohlcv")
STATE_PATH = Path(
    "E:/투자/Multi-Asset Strategy Platform/research/ralph_loop_state.json"
)
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")

SLIPPAGE = 0.0005
COMMISSION = 0.0004
FUNDING_PER_8H = 0.0001


def load_state():
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_ohlcv(symbol, timeframe="4h"):
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
    return df.sort_values("datetime").reset_index(drop=True)


def generate_tsmom_signals(df, lookback=84):
    """TSMOM: go with the trend based on past returns"""
    ret = df["close"].pct_change(lookback)
    return np.sign(ret).fillna(0).astype(int)


def generate_multifactor_signals(df):
    """Multi-Factor: combine trend + momentum + breakout + RSI"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    scores = np.zeros(len(df))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    scores += np.where(ema_fast > ema_slow, 1, -1)

    roc = close.pct_change(24)
    scores += np.sign(roc).fillna(0)

    upper = high.rolling(24).max().shift(1)
    lower = low.rolling(24).min().shift(1)
    scores += np.where(close > upper, 1, np.where(close < lower, -1, 0))

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    scores += np.where(rsi > 60, 1, np.where(rsi < 40, -1, 0))

    return pd.Series(
        np.where(scores >= 3, 1, np.where(scores <= -3, -1, 0)), index=df.index
    )


def simulate_full(df, signals, position_pct=0.02, max_bars=48):
    """Full simulation returning equity curve and trade list"""
    equity = [1.0]
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []
    capital = 1.0

    for i in range(len(df)):
        close = df["close"].iloc[i]
        sig = signals.iloc[i] if isinstance(signals, pd.Series) else signals[i]

        if position != 0:
            bars_held += 1
            should_exit = (
                (bars_held >= max_bars) or (sig != 0 and sig != position) or (sig == 0)
            )

            if should_exit:
                exit_price = close * (1 - SLIPPAGE * np.sign(position))
                pnl = position * (exit_price - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 2 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0

        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = close * (1 + SLIPPAGE * position)
            capital -= COMMISSION * position_pct
            bars_held = 0

        equity.append(capital)

    if position != 0:
        close = df["close"].iloc[-1]
        exit_price = close * (1 - SLIPPAGE * np.sign(position))
        pnl = position * (exit_price - entry_price) / entry_price * position_pct
        pnl -= COMMISSION * position_pct
        pnl -= FUNDING_PER_8H * bars_held / 2 * position_pct
        trades.append(pnl)
        capital += pnl
        equity.append(capital)

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    wins = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t <= 0)
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))

    return {
        "equity": equity,
        "drawdown": dd,
        "trades": trades,
        "total_return": capital - 1,
        "max_drawdown": dd.min(),
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "profit_factor": gross_profit / (gross_loss + 1e-10),
        "trade_count": len(trades),
    }


# =============================================================================
# Task 4.1: In-Sample Backtest (first 60%)
# =============================================================================
def task_4_1_in_sample(df, signal_func, name):
    n = len(df)
    is_end = int(n * 0.6)
    is_df = df.iloc[:is_end].copy()
    signals = signal_func(is_df)
    result = simulate_full(is_df, signals)
    result["period"] = "IS"
    result["name"] = name
    return result


# =============================================================================
# Task 4.2: Out-of-Sample Test (next 20%)
# =============================================================================
def task_4_2_out_of_sample(df, signal_func, name):
    n = len(df)
    is_end = int(n * 0.6)
    oos_end = int(n * 0.8)
    # Use all data up to OOS start for signal calculation (train portion known)
    oos_df = df.iloc[:oos_end].copy()
    signals = signal_func(oos_df)
    # Only evaluate OOS portion
    oos_signals = signals.iloc[is_end:oos_end]
    oos_prices = df.iloc[is_end:oos_end].copy().reset_index(drop=True)
    oos_signals = oos_signals.reset_index(drop=True)
    result = simulate_full(oos_prices, oos_signals)
    result["period"] = "OOS"
    result["name"] = name
    return result


# =============================================================================
# Task 4.3: Walk-Forward Analysis
# =============================================================================
def task_4_3_walk_forward(df, signal_func, name, train_bars=1080, test_bars=180):
    """Walk-forward with 180-day train, 30-day test"""
    n = len(df)
    results = []
    i = train_bars

    while i + test_bars <= n:
        train = df.iloc[:i].copy()
        test = df.iloc[i : i + test_bars].copy().reset_index(drop=True)

        # Signals from full history up to test start
        all_signals = signal_func(train)
        # For test: extend signals using available data
        combined = pd.concat([train, test]).reset_index(drop=True)
        full_signals = signal_func(combined)
        test_signals = full_signals.iloc[i : i + test_bars].reset_index(drop=True)

        r = simulate_full(test, test_signals)
        results.append(r["total_return"])
        i += test_bars

    if not results:
        return {"wfa_efficiency": 0, "periods": 0, "name": name}

    # WFA efficiency
    profitable_periods = sum(1 for r in results if r > 0)
    wfa_efficiency = profitable_periods / len(results) * 100

    # Sharpe
    sharpe = (
        np.mean(results) / (np.std(results) + 1e-10) * np.sqrt(12)
        if len(results) > 1
        else 0
    )

    return {
        "wfa_efficiency": wfa_efficiency,
        "periods": len(results),
        "profitable_periods": profitable_periods,
        "avg_period_return": np.mean(results),
        "sharpe": sharpe,
        "period_returns": results,
        "name": name,
    }


# =============================================================================
# Task 4.4: Stress Testing
# =============================================================================
def task_4_4_stress_test(df, signal_func, name):
    """Test on specific market regimes"""
    dt = df["datetime"]

    stress_periods = {
        "COVID_crash": ("2020-02-15", "2020-04-15"),
        "May2021_crash": ("2021-04-15", "2021-06-30"),
        "FTX_collapse": ("2022-10-15", "2022-12-31"),
        "ETF_rally_2024": ("2024-01-01", "2024-03-31"),
        "Recent_2025": ("2025-06-01", "2025-12-31"),
    }

    results = {}
    for period_name, (start, end) in stress_periods.items():
        mask = (dt >= start) & (dt <= end)
        if mask.sum() < 50:
            results[period_name] = {"skipped": True, "reason": "insufficient_data"}
            continue

        period_df = df[mask].copy().reset_index(drop=True)
        # Use all available history for signals
        all_before = df[df["datetime"] <= end].copy()
        signals = signal_func(all_before)
        period_signals = signals.iloc[-len(period_df) :].reset_index(drop=True)

        r = simulate_full(period_df, period_signals)
        results[period_name] = {
            "return": r["total_return"],
            "max_dd": r["max_drawdown"],
            "trades": r["trade_count"],
            "win_rate": r["win_rate"],
        }

    return {"stress_results": results, "name": name}


# =============================================================================
# Task 4.5: Statistical Tests
# =============================================================================
def task_4_5_statistical_tests(wfa_result, name):
    """Monte Carlo + t-test + bootstrap"""
    returns = wfa_result.get("period_returns", [])

    if len(returns) < 5:
        return {"name": name, "insufficient_data": True}

    # t-test: is mean return > 0?
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    n = len(returns)
    t_stat = mean_ret / (std_ret / np.sqrt(n)) if std_ret > 0 else 0
    # Approximate p-value (one-sided)
    from scipy import stats as scipy_stats

    p_value = 1 - scipy_stats.t.cdf(t_stat, df=n - 1)

    # Bootstrap confidence interval (95%)
    np.random.seed(42)
    n_bootstrap = 5000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    # Monte Carlo permutation test
    np.random.seed(42)
    n_mc = 5000
    mc_returns = []
    for _ in range(n_mc):
        shuffled = np.random.permutation(returns)
        signs = np.random.choice([-1, 1], size=len(returns))
        mc_returns.append(np.mean(shuffled * signs))
    mc_p_value = np.mean(np.array(mc_returns) >= mean_ret)

    return {
        "name": name,
        "mean_return": mean_ret,
        "t_stat": t_stat,
        "p_value": p_value,
        "bootstrap_ci_95": (ci_lower, ci_upper),
        "mc_p_value": mc_p_value,
        "significant_5pct": p_value < 0.05,
        "significant_10pct": p_value < 0.10,
    }


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 4: VALIDATION FRAMEWORK")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    state = load_state()

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
        "tsmom_84": ("TSMOM(84)", lambda df: generate_tsmom_signals(df, 84)),
        "tsmom_42": ("TSMOM(42)", lambda df: generate_tsmom_signals(df, 42)),
        "multi_factor": ("Multi-Factor", generate_multifactor_signals),
    }

    all_results = {}

    for strat_key, (strat_name, signal_func) in strategies.items():
        print(f"\n{'='*70}")
        print(f"VALIDATING: {strat_name}")
        print(f"{'='*70}")

        strat_results = {
            "is": {},
            "oos": {},
            "wfa": {},
            "stress": {},
            "stats": {},
        }

        for symbol in symbols:
            df = load_ohlcv(symbol)
            if df.empty or len(df) < 2000:
                continue

            print(f"\n  {symbol} ({len(df)} bars)")

            # 4.1 IS
            is_r = task_4_1_in_sample(df, signal_func, strat_name)
            strat_results["is"][symbol] = is_r
            print(
                f"    IS:  Ret={is_r['total_return']*100:+.1f}%  DD={is_r['max_drawdown']*100:.1f}%  WR={is_r['win_rate']*100:.0f}%  PF={is_r['profit_factor']:.2f}  T={is_r['trade_count']}"
            )

            # 4.2 OOS
            oos_r = task_4_2_out_of_sample(df, signal_func, strat_name)
            strat_results["oos"][symbol] = oos_r
            print(
                f"    OOS: Ret={oos_r['total_return']*100:+.1f}%  DD={oos_r['max_drawdown']*100:.1f}%  WR={oos_r['win_rate']*100:.0f}%  PF={oos_r['profit_factor']:.2f}  T={oos_r['trade_count']}"
            )

            # Degradation ratio
            if is_r["total_return"] != 0:
                deg = oos_r["total_return"] / is_r["total_return"]
                print(f"    Degradation ratio: {deg:.2f}")

            # 4.3 WFA
            wfa_r = task_4_3_walk_forward(df, signal_func, strat_name)
            strat_results["wfa"][symbol] = wfa_r
            print(
                f"    WFA: {wfa_r['wfa_efficiency']:.0f}% efficient ({wfa_r.get('profitable_periods',0)}/{wfa_r['periods']} profitable)  Sharpe={wfa_r.get('sharpe',0):.2f}"
            )

            # 4.4 Stress
            stress_r = task_4_4_stress_test(df, signal_func, strat_name)
            strat_results["stress"][symbol] = stress_r
            for period, res in stress_r["stress_results"].items():
                if "skipped" not in res:
                    print(
                        f"    Stress {period}: Ret={res['return']*100:+.1f}%  DD={res['max_dd']*100:.1f}%"
                    )

            # 4.5 Stats
            stats_r = task_4_5_statistical_tests(wfa_r, strat_name)
            strat_results["stats"][symbol] = stats_r
            if not stats_r.get("insufficient_data"):
                sig_marker = (
                    "***"
                    if stats_r["significant_5pct"]
                    else ("*" if stats_r["significant_10pct"] else "")
                )
                print(
                    f"    Stats: t={stats_r['t_stat']:.2f}  p={stats_r['p_value']:.3f}  CI=[{stats_r['bootstrap_ci_95'][0]*100:.2f}%, {stats_r['bootstrap_ci_95'][1]*100:.2f}%] {sig_marker}"
                )

        # Strategy summary
        print(f"\n  {'='*60}")
        print(f"  SUMMARY: {strat_name}")
        print(f"  {'='*60}")

        oos_returns = [r["total_return"] for r in strat_results["oos"].values()]
        wfa_effs = [
            r["wfa_efficiency"]
            for r in strat_results["wfa"].values()
            if r["periods"] > 0
        ]
        wfa_sharpes = [
            r.get("sharpe", 0)
            for r in strat_results["wfa"].values()
            if r["periods"] > 0
        ]
        p_values = [
            r.get("p_value", 1)
            for r in strat_results["stats"].values()
            if not r.get("insufficient_data")
        ]

        oos_profitable = sum(1 for r in oos_returns if r > 0)
        avg_wfa = np.mean(wfa_effs) if wfa_effs else 0
        avg_sharpe = np.mean(wfa_sharpes) if wfa_sharpes else 0
        sig_count = sum(1 for p in p_values if p < 0.05)

        print(
            f"  OOS Profitable: {oos_profitable}/{len(oos_returns)} ({oos_profitable/len(oos_returns)*100:.0f}%)"
        )
        print(f"  Avg WFA Efficiency: {avg_wfa:.0f}%")
        print(f"  Avg WFA Sharpe: {avg_sharpe:.2f}")
        print(f"  Statistically Significant (5%): {sig_count}/{len(p_values)}")

        # Check success criteria
        criteria = {
            "oos_sharpe_gt_1": avg_sharpe > 1.0,
            "max_dd_lt_25": all(
                r["max_drawdown"] > -0.25 for r in strat_results["oos"].values()
            ),
            "win_rate_gt_45": np.mean(
                [r["win_rate"] for r in strat_results["oos"].values()]
            )
            > 0.45,
            "profit_factor_gt_1_5": np.mean(
                [r["profit_factor"] for r in strat_results["oos"].values()]
            )
            > 1.5,
            "wfa_efficiency_gt_50": avg_wfa > 50,
            "trade_count_gt_100": all(
                r["trade_count"] > 100 for r in strat_results["oos"].values()
            ),
        }

        print(f"\n  SUCCESS CRITERIA:")
        passed = 0
        for c, v in criteria.items():
            status = "PASS" if v else "FAIL"
            if v:
                passed += 1
            print(f"    {c}: {status}")
        print(f"  TOTAL: {passed}/{len(criteria)}")

        all_results[strat_key] = {
            "name": strat_name,
            "oos_profitable_pct": oos_profitable / len(oos_returns) * 100,
            "avg_wfa_efficiency": avg_wfa,
            "avg_wfa_sharpe": avg_sharpe,
            "significant_count": sig_count,
            "criteria_passed": passed,
            "criteria_total": len(criteria),
            "criteria": criteria,
            "details": {
                "is": {
                    s: {
                        k: v
                        for k, v in r.items()
                        if k not in ("equity", "drawdown", "trades")
                    }
                    for s, r in strat_results["is"].items()
                },
                "oos": {
                    s: {
                        k: v
                        for k, v in r.items()
                        if k not in ("equity", "drawdown", "trades")
                    }
                    for s, r in strat_results["oos"].items()
                },
                "wfa": {
                    s: {k: v for k, v in r.items() if k != "period_returns"}
                    for s, r in strat_results["wfa"].items()
                },
                "stress": strat_results["stress"],
                "stats": {
                    s: {
                        k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in r.items()
                    }
                    for s, r in strat_results["stats"].items()
                },
            },
        }

    # Final ranking
    print("\n" + "=" * 70)
    print("PHASE 4 FINAL RANKING")
    print("=" * 70)

    best = None
    for key, res in sorted(
        all_results.items(), key=lambda x: x[1]["criteria_passed"], reverse=True
    ):
        marker = "*** BEST ***" if best is None else ""
        if best is None:
            best = key
        print(
            f"  {res['name']}: {res['criteria_passed']}/{res['criteria_total']} criteria passed  "
            f"WFA Sharpe={res['avg_wfa_sharpe']:.2f}  WFA Eff={res['avg_wfa_efficiency']:.0f}%  {marker}"
        )

    # Identify best performers per symbol
    best_performers = []
    best_strat = all_results.get(best, {})
    if "details" in best_strat:
        for symbol, oos_r in best_strat["details"]["oos"].items():
            if oos_r.get("total_return", 0) > 0:
                best_performers.append(
                    {
                        "symbol": symbol,
                        "strategy": best_strat["name"],
                        "oos_return": oos_r["total_return"],
                        "win_rate": oos_r["win_rate"],
                    }
                )

    # Save results
    report = {
        "generated_at": datetime.now().isoformat(),
        "strategies_validated": len(all_results),
        "best_strategy": best,
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "details"}
            for k, v in all_results.items()
        },
        "detailed_results": all_results,
        "best_performers": best_performers,
    }

    report_path = RESULTS_PATH / "phase4_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Update state
    state["current_phase"] = "5"
    state["current_task"] = "5.1"
    state["completed_tasks"].extend(["4.1", "4.2", "4.3", "4.4", "4.5"])
    state["findings"]["best_performers"] = best_performers
    state["findings"]["validation"] = {
        "best_strategy": best,
        "best_strategy_name": all_results[best]["name"] if best else "None",
        "criteria_passed": all_results[best]["criteria_passed"] if best else 0,
    }
    state["next_actions"] = ["5.1: Position Sizing", "5.2: Risk Management"]
    save_state(state)

    print(f"\n  Best strategy: {all_results[best]['name'] if best else 'None'}")
    print(f"  Best performers: {len(best_performers)} symbols")
    print(f"  Report saved: {report_path}")
    print(f"  Next: Phase 5 - Production Readiness")

    return report


if __name__ == "__main__":
    report = main()
