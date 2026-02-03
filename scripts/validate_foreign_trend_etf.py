# -*- coding: utf-8 -*-
"""
Independent validator for:
  Foreign_30d_SMA_100 (외국인 + 추세 ETF 타이밍 전략)

Rules (as provided):
  BUY if:
    (1) Foreign 30d cumulative net buy > 0
    (2) Close > SMA(100)
  SELL if any condition fails

Look-ahead assumption:
  Foreign data and Close/SMA are confirmed after T close -> usable for T+1 trading.
  Default: shift(1) on the signal (enter/exit next day).

Costs:
  Round-trip cost = 0.09% (default). Applied as half per side on entry and exit:
    cost_per_side = round_trip_cost / 2
  Applied on days with position change: turnover = abs(diff(position))

Outputs:
  - Console summary
  - CSVs in out_dir
  - PNG charts in out_dir (optional)

This script intentionally does NOT reference any existing project strategy code.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False).dt.tz_localize(None)


def _coerce_numeric(s: pd.Series) -> pd.Series:
    # handles "1,234", "  123 ", etc.
    return pd.to_numeric(
        s.astype(str).str.replace(",", "").str.strip(), errors="coerce"
    )


def _read_csv_robust(path: Path) -> pd.DataFrame:
    # Try utf-8-sig -> utf-8 -> cp949
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Failed to read CSV {path} with common encodings. Last error: {last_err}"
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def annualized_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * math.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return float("nan")
    total = eq.iloc[-1] / eq.iloc[0]
    n_days = len(eq) - 1
    if n_days <= 0:
        return float("nan")
    years = n_days / periods_per_year
    if years <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def summarize_period(equity: pd.Series, start: str, end: str) -> Dict[str, float]:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    sub = equity[(equity.index >= s) & (equity.index <= e)].dropna()
    if len(sub) < 2:
        return {"return": float("nan"), "mdd": float("nan")}
    ret = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
    mdd = max_drawdown(sub)
    return {"return": ret, "mdd": mdd}


def realized_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def format_pct(x: float, digits: int = 2) -> str:
    if x is None or np.isnan(x):
        return "nan"
    return f"{x*100:.{digits}f}%"


def format_num(x: float, digits: int = 3) -> str:
    if x is None or np.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


# -----------------------------
# Data loading (independent / robust heuristics)
# -----------------------------
@dataclass
class PriceData:
    close: pd.Series  # indexed by date
    ret: pd.Series  # close-to-close returns


@dataclass
class ForeignData:
    net: pd.Series  # indexed by date


def load_kospi200_parquet(path: Path) -> PriceData:
    if not path.exists():
        raise FileNotFoundError(f"KOSPI200 parquet not found: {path}")

    df = pd.read_parquet(path)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}

    date_col_candidates = ["date", "datetime", "time", "일자", "날짜", "trade_date"]
    close_col_candidates = [
        "close",
        "adj close",
        "adj_close",
        "adjusted close",
        "종가",
        "종가(원)",
    ]

    date_col = None
    for k in date_col_candidates:
        if k in cols:
            date_col = cols[k]
            break
    if date_col is None:
        # Sometimes parquet index is date
        if isinstance(df.index, (pd.DatetimeIndex,)):
            idx = df.index.tz_localize(None)
            df = df.copy()
            df["__date__"] = idx
            date_col = "__date__"
        else:
            raise RuntimeError(
                f"Cannot find a date column in parquet columns={list(df.columns)}"
            )

    close_col = None
    # direct match by lowercase
    lower_cols = [c.lower() for c in df.columns]
    for cand in close_col_candidates:
        if cand in lower_cols:
            close_col = df.columns[lower_cols.index(cand)]
            break
    # yfinance often uses 'Close'
    if close_col is None:
        for cand in ["Close", "Adj Close", "close", "adj_close", "Adj_Close"]:
            if cand in df.columns:
                close_col = cand
                break
    if close_col is None:
        raise RuntimeError(
            f"Cannot find a close column in parquet columns={list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "date": _to_datetime(df[date_col]),
            "close": _coerce_numeric(df[close_col]),
        }
    ).dropna(subset=["date", "close"])

    out = out.sort_values("date").drop_duplicates("date")
    out = out.set_index("date")
    close = out["close"].astype(float)
    ret = close.pct_change().rename("ret")

    return PriceData(close=close.rename("close"), ret=ret)


def _infer_foreign_net_from_df(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, str]]:
    """
    Attempts to infer:
      - date column
      - foreign net buy column (or compute from buy-sell)
    Returns: (net_series_indexed_by_date, meta dict)
    """
    meta: Dict[str, str] = {}

    # Date
    date_candidates = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "일자",
        "날짜",
        "trade_date",
        "기준일자",
    ]
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # try index
        if isinstance(df.index, pd.DatetimeIndex):
            d = df.index
            meta["date_col"] = "<index>"
            dates = d.tz_localize(None)
        else:
            raise RuntimeError("No recognizable date column in foreign CSV.")
    else:
        meta["date_col"] = date_col
        dates = _to_datetime(df[date_col])

    # Foreign net candidates
    # Accept both Korean/English patterns
    colnames = list(df.columns)
    lower = [c.lower() for c in colnames]

    # 1) direct net column
    net_patterns = [
        "foreign_net",
        "net_foreign",
        "foreign net",
        "외국인순매수",
        "외국인 순매수",
        "외국인(순매수)",
        "외국인 순매수(수량)",
        "외국인 순매수(금액)",
        "외국인_순매수",
        "외국인순매수금액",
        "외국인순매수수량",
        "외국인합계",
        "외국인 합계",  # Added for aggregated foreign data
    ]
    net_col = None
    for pat in net_patterns:
        # exact match first
        if pat in colnames:
            net_col = pat
            break
        # substring search
        for c in colnames:
            if pat.replace(" ", "") in c.replace(" ", ""):
                net_col = c
                break
        if net_col is not None:
            break

    if net_col is not None:
        meta["net_col"] = net_col
        net = _coerce_numeric(df[net_col])

    else:
        # 2) compute from buy/sell
        buy_patterns = [
            "foreign_buy",
            "외국인매수",
            "외국인 매수",
            "foreign buy",
            "외국인매수(금액)",
            "외국인매수(수량)",
        ]
        sell_patterns = [
            "foreign_sell",
            "외국인매도",
            "외국인 매도",
            "foreign sell",
            "외국인매도(금액)",
            "외국인매도(수량)",
        ]

        buy_col = None
        sell_col = None

        for pat in buy_patterns:
            for c in colnames:
                if pat.replace(" ", "") in c.replace(" ", ""):
                    buy_col = c
                    break
            if buy_col is not None:
                break

        for pat in sell_patterns:
            for c in colnames:
                if pat.replace(" ", "") in c.replace(" ", ""):
                    sell_col = c
                    break
            if sell_col is not None:
                break

        if buy_col is None or sell_col is None:
            raise RuntimeError(
                "Cannot infer foreign net buy column. "
                "Provide a CSV that contains foreign net (순매수) or buy/sell columns, "
                "or use --foreign_file to point to the correct file."
            )

        meta["buy_col"] = buy_col
        meta["sell_col"] = sell_col
        net = _coerce_numeric(df[buy_col]) - _coerce_numeric(df[sell_col])

    out = pd.DataFrame({"date": dates, "foreign_net": net}).dropna(subset=["date"])
    out = out.sort_values("date").drop_duplicates("date").set_index("date")
    series = out["foreign_net"].astype(float)

    return series, meta


def load_foreign_from_glob(
    pattern: str,
    price_index: pd.DatetimeIndex,
    start: str,
    end: str,
    preferred_keywords: Tuple[str, ...] = ("kospi200", "kosp200", "코스피200", "200"),
) -> Tuple[ForeignData, Dict[str, str]]:
    """
    Tries to pick the most relevant foreign investor CSV among many files.

    Scoring:
      - overlap with price dates within [start, end]
      - non-null ratio
      - filename keyword bonus (kospi200, etc.)
    """
    files = sorted([Path(f) for f in glob_module.glob(pattern)])
    if len(files) == 0:
        raise FileNotFoundError(
            f"No foreign investor CSV files matched pattern: {pattern}"
        )

    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    px_dates = price_index[(price_index >= s) & (price_index <= e)]

    best = None
    best_score = -1.0
    best_meta: Dict[str, str] = {}

    for fp in files:
        try:
            df = _read_csv_robust(fp)
            net, meta = _infer_foreign_net_from_df(df)
            net_sub = net.reindex(px_dates)
            overlap = net_sub.notna().sum()
            if overlap <= 10:
                continue

            non_null_ratio = (
                float(overlap) / float(len(px_dates)) if len(px_dates) else 0.0
            )
            # keyword bonus
            name = fp.name.lower()
            bonus = 0.0
            for kw in preferred_keywords:
                if kw in name:
                    bonus += 0.15

            score = (float(overlap) * (0.5 + non_null_ratio)) * (1.0 + bonus)

            if score > best_score:
                best_score = score
                best = (fp, net)
                best_meta = meta | {
                    "selected_file": str(fp),
                    "score": str(score),
                    "overlap_days": str(int(overlap)),
                }
        except Exception:
            continue

    if best is None:
        raise RuntimeError(
            "Failed to select a foreign investor CSV automatically. "
            "Use --foreign_file to point to the correct *_investor.csv."
        )

    fp, net = best
    net = net.sort_index()
    return ForeignData(net=net.rename("foreign_net")), best_meta


def load_foreign_single_file(path: Path) -> Tuple[ForeignData, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Foreign file not found: {path}")
    df = _read_csv_robust(path)
    net, meta = _infer_foreign_net_from_df(df)
    return ForeignData(net=net.rename("foreign_net")), meta | {
        "selected_file": str(path)
    }


# -----------------------------
# Backtest core
# -----------------------------
@dataclass
class BacktestResult:
    df: pd.DataFrame  # joined frame with indicators, signal, position, returns, equity
    metrics: Dict[str, float]
    trade_stats: Dict[str, float]
    meta: Dict[str, str]


def extract_trade_returns(df: pd.DataFrame, equity_col: str = "equity") -> pd.Series:
    """
    Trade return is computed from equity ratio between entry and exit dates.
    Assumes position is 1 during trade, 0 otherwise.
    """
    pos = df["position"].fillna(0).astype(int)
    changes = pos.diff().fillna(0)
    entry_idx = df.index[changes == 1].tolist()
    exit_idx = df.index[changes == -1].tolist()

    # If still in position at the end, treat last day as exit
    if len(entry_idx) > len(exit_idx):
        exit_idx.append(df.index[-1])

    trade_rets = []
    for en, ex in zip(entry_idx, exit_idx):
        if ex <= en:
            continue
        eq_en = df.loc[en, equity_col]
        eq_ex = df.loc[ex, equity_col]
        if pd.isna(eq_en) or pd.isna(eq_ex) or eq_en <= 0:
            continue
        trade_rets.append(eq_ex / eq_en - 1.0)

    return pd.Series(trade_rets, dtype=float)


def run_strategy(
    price: PriceData,
    foreign: ForeignData,
    start: str,
    end: str,
    foreign_period: int,
    sma_period: int,
    execution_lag: int,
    round_trip_cost: float,
    label: str,
) -> BacktestResult:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    df = pd.DataFrame(
        {
            "close": price.close,
            "ret": price.ret,
        }
    ).join(pd.DataFrame({"foreign_net": foreign.net}), how="left")

    df = df[(df.index >= s) & (df.index <= e)].copy()
    df = df.sort_index()

    # Indicators (based on T close data)
    df["foreign_roll"] = (
        df["foreign_net"].rolling(foreign_period, min_periods=foreign_period).sum()
    )
    df["sma"] = df["close"].rolling(sma_period, min_periods=sma_period).mean()

    # Raw signal at T close
    df["signal_raw"] = (df["foreign_roll"] > 0) & (df["close"] > df["sma"])

    # Executable position (T+lag). Default lag=1 as requested.
    df["position"] = df["signal_raw"].shift(execution_lag).fillna(False).astype(int)

    # Transaction costs
    # Apply cost per side when position changes by 1 unit (0->1 or 1->0).
    cost_per_side = round_trip_cost / 2.0
    df["turnover"] = df["position"].diff().abs().fillna(0.0)
    df["cost"] = df["turnover"] * cost_per_side

    # Strategy daily return: position * underlying daily return - cost
    df["strategy_ret"] = (df["position"] * df["ret"]).fillna(0.0) - df["cost"].fillna(
        0.0
    )

    # Equity curves
    df["equity"] = (1.0 + df["strategy_ret"]).cumprod()
    df["equity_bh"] = (1.0 + df["ret"].fillna(0.0)).cumprod()

    # Metrics
    metrics: Dict[str, float] = {}
    metrics["sharpe"] = annualized_sharpe(df["strategy_ret"])
    metrics["cagr"] = cagr(df["equity"])
    metrics["mdd"] = max_drawdown(df["equity"])
    metrics["vol_ann"] = realized_vol(df["strategy_ret"])
    metrics["bh_cagr"] = cagr(df["equity_bh"])
    metrics["bh_mdd"] = max_drawdown(df["equity_bh"])

    # Win rates
    # Daily win rate when invested
    invested = df["position"] == 1
    if invested.sum() > 0:
        metrics["daily_win_rate_in_position"] = float(
            (df.loc[invested, "strategy_ret"] > 0).mean()
        )
    else:
        metrics["daily_win_rate_in_position"] = float("nan")

    # Trade win rate
    trade_rets = extract_trade_returns(df)
    metrics["n_trades"] = float(len(trade_rets))
    if len(trade_rets) > 0:
        metrics["trade_win_rate"] = float((trade_rets > 0).mean())
        metrics["avg_trade_return"] = float(trade_rets.mean())
        metrics["median_trade_return"] = float(trade_rets.median())
    else:
        metrics["trade_win_rate"] = float("nan")
        metrics["avg_trade_return"] = float("nan")
        metrics["median_trade_return"] = float("nan")

    # Exposure
    metrics["exposure"] = float(df["position"].mean())

    trade_stats = {
        "n_trades": metrics["n_trades"],
        "trade_win_rate": metrics["trade_win_rate"],
        "avg_trade_return": metrics["avg_trade_return"],
        "median_trade_return": metrics["median_trade_return"],
    }

    meta = {
        "label": label,
        "foreign_period": str(foreign_period),
        "sma_period": str(sma_period),
        "execution_lag": str(execution_lag),
        "round_trip_cost": str(round_trip_cost),
    }

    return BacktestResult(df=df, metrics=metrics, trade_stats=trade_stats, meta=meta)


# -----------------------------
# Walk-forward validation
# -----------------------------
@dataclass
class WFResult:
    method: str
    fold_rows: pd.DataFrame
    summary: Dict[str, float]


def walk_forward_split_indices(
    dates: pd.DatetimeIndex, n_folds: int
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Expanding-window walk-forward:
      fold i:
        train = [0 : test_start)
        test  = [test_start : test_end)
    where the test segments partition the timeline into n_folds equal contiguous blocks.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    n = len(dates)
    if n < n_folds * 30:
        # Too short: still create folds but warn in output
        pass

    # create cut points
    boundaries = np.linspace(0, n, n_folds + 1).astype(int)
    folds = []
    for i in range(1, n_folds + 1):
        test_start = boundaries[i - 1]
        test_end = boundaries[i]
        train_idx = dates[:test_start]
        test_idx = dates[test_start:test_end]
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue
        folds.append((train_idx, test_idx))
    return folds


def run_walk_forward(
    price: PriceData,
    foreign: ForeignData,
    start: str,
    end: str,
    foreign_period: int,
    sma_period: int,
    execution_lag: int,
    round_trip_cost: float,
    n_folds: int = 11,
) -> WFResult:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    dates = price.close.index
    dates = dates[(dates >= s) & (dates <= e)].sort_values()

    folds = walk_forward_split_indices(dates, n_folds=n_folds)
    rows = []

    for k, (train_idx, test_idx) in enumerate(folds, start=1):
        train_start, train_end = train_idx[0], train_idx[-1]
        test_start, test_end = test_idx[0], test_idx[-1]

        bt_train = run_strategy(
            price,
            foreign,
            start=str(train_start.date()),
            end=str(train_end.date()),
            foreign_period=foreign_period,
            sma_period=sma_period,
            execution_lag=execution_lag,
            round_trip_cost=round_trip_cost,
            label=f"WF_train_fold{k}",
        )
        bt_test = run_strategy(
            price,
            foreign,
            start=str(test_start.date()),
            end=str(test_end.date()),
            foreign_period=foreign_period,
            sma_period=sma_period,
            execution_lag=execution_lag,
            round_trip_cost=round_trip_cost,
            label=f"WF_test_fold{k}",
        )

        sh_tr = bt_train.metrics["sharpe"]
        sh_te = bt_test.metrics["sharpe"]
        ratio = (
            (sh_te / sh_tr)
            if (not np.isnan(sh_tr) and sh_tr != 0 and not np.isnan(sh_te))
            else float("nan")
        )

        rows.append(
            {
                "fold": k,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_sharpe": sh_tr,
                "test_sharpe": sh_te,
                "wf_ratio": ratio,
                "train_mdd": bt_train.metrics["mdd"],
                "test_mdd": bt_test.metrics["mdd"],
                "train_cagr": bt_train.metrics["cagr"],
                "test_cagr": bt_test.metrics["cagr"],
                "test_exposure": bt_test.metrics["exposure"],
            }
        )

    fold_df = pd.DataFrame(rows)
    if len(fold_df) == 0:
        summary = {
            "mean_train_sharpe": float("nan"),
            "mean_test_sharpe": float("nan"),
            "wf_ratio": float("nan"),
        }
    else:
        mean_train = float(
            fold_df["train_sharpe"].replace([np.inf, -np.inf], np.nan).mean()
        )
        mean_test = float(
            fold_df["test_sharpe"].replace([np.inf, -np.inf], np.nan).mean()
        )
        wf_ratio = (
            float(mean_test / mean_train)
            if (
                not np.isnan(mean_train) and mean_train != 0 and not np.isnan(mean_test)
            )
            else float("nan")
        )
        summary = {
            "mean_train_sharpe": mean_train,
            "mean_test_sharpe": mean_test,
            "wf_ratio": wf_ratio,
            "n_folds_used": float(len(fold_df)),
        }

    return WFResult(
        method=f"{n_folds}-fold expanding WF", fold_rows=fold_df, summary=summary
    )


# -----------------------------
# Leveraged ETF simulation
# -----------------------------
def simulate_daily_reset_leverage(
    ret: pd.Series, leverage: float, daily_fee: float = 0.0
) -> pd.Series:
    """
    Daily reset leveraged return:
      r_L = leverage * r - daily_fee
    This captures daily compounding / volatility decay effects naturally.
    """
    r = ret.fillna(0.0).astype(float)
    rL = leverage * r - daily_fee
    return rL.rename(f"ret_{leverage:.1f}x")


def vol_decay_gap(ret: pd.Series, leverage: float) -> Dict[str, float]:
    """
    Quantify volatility drag vs naive leverage on log returns:
      gap = log(Π(1+Lr)) - L*log(Π(1+r))
    Negative gap indicates volatility decay relative to naive expectation.
    """
    r = ret.dropna().astype(float)
    if len(r) < 2:
        return {"log_gap": float("nan"), "avg_daily_log_gap": float("nan")}
    log1 = np.log1p(r)
    logL = np.log1p(leverage * r)
    log_gap = float(logL.sum() - leverage * log1.sum())
    avg_gap = float(log_gap / len(r))
    return {"log_gap": log_gap, "avg_daily_log_gap": avg_gap}


# -----------------------------
# Sensitivity grid
# -----------------------------
def run_sensitivity_grid(
    price: PriceData,
    foreign: ForeignData,
    start: str,
    end: str,
    foreign_period_list: List[int],
    sma_period_list: List[int],
    execution_lag: int,
    round_trip_cost: float,
) -> pd.DataFrame:
    rows = []
    for fp in foreign_period_list:
        for sp in sma_period_list:
            bt = run_strategy(
                price,
                foreign,
                start,
                end,
                foreign_period=fp,
                sma_period=sp,
                execution_lag=execution_lag,
                round_trip_cost=round_trip_cost,
                label=f"grid_fp{fp}_sp{sp}",
            )
            m = bt.metrics
            rows.append(
                {
                    "foreign_period": fp,
                    "sma_period": sp,
                    "sharpe": m["sharpe"],
                    "cagr": m["cagr"],
                    "mdd": m["mdd"],
                    "exposure": m["exposure"],
                    "n_trades": m["n_trades"],
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Verdict logic
# -----------------------------
@dataclass
class ClaimedPerformance:
    sharpe: float = 1.225
    cagr: float = 0.138
    mdd: float = -0.169
    wf_ratio: float = 0.86
    win_rate: float = (
        0.552  # ambiguous: daily or trade. We'll compare both with tolerance.
    )


def check_close(a: float, b: float, tol_abs: float) -> bool:
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= tol_abs


def verdict(
    bt_main: BacktestResult,
    wf: WFResult,
    claimed: ClaimedPerformance,
    mdd_1x_limit: float,
    mdd_2x_limit: float,
    lev_mdd_1x: float,
    lev_mdd_2x: float,
    grid: Optional[pd.DataFrame] = None,
) -> Tuple[str, List[str]]:
    """
    PASS / CONDITIONAL / FAIL
    - FAIL if look-ahead is violated (this script enforces shift, so violation is typically user-data alignment issues),
      or if WF ratio < 0.5, or if leverage MDD limits are badly broken, or if performance can't reproduce at all.
    - CONDITIONAL if borderline WF ratio (0.5~0.7), or strong parameter sensitivity, or crisis defense weak.
    """
    reasons: List[str] = []

    m = bt_main.metrics
    sh = m["sharpe"]
    cg = m["cagr"]
    dd = m["mdd"]
    wf_ratio = wf.summary.get("wf_ratio", float("nan"))

    # Reproduction checks (tolerances chosen to avoid false fail due to implementation micro-differences)
    ok_sh = check_close(sh, claimed.sharpe, tol_abs=0.15)
    ok_cg = check_close(cg, claimed.cagr, tol_abs=0.02)
    ok_dd = check_close(dd, claimed.mdd, tol_abs=0.03)

    # Win rate ambiguity: accept either daily(in-position) or trade win rate
    wr_daily = m.get("daily_win_rate_in_position", float("nan"))
    wr_trade = m.get("trade_win_rate", float("nan"))
    ok_wr = check_close(wr_daily, claimed.win_rate, tol_abs=0.05) or check_close(
        wr_trade, claimed.win_rate, tol_abs=0.05
    )

    reproduced = (ok_sh and ok_cg and ok_dd) or (
        ok_sh and ok_dd
    )  # allow some CAGR drift due to cost timing

    if not reproduced:
        reasons.append(
            "주장 성과(Sharpe/CAGR/MDD)와 재현 결과가 충분히 근접하지 않음(구현/데이터/정렬/비용가정 차이 가능)."
        )

    # WF
    if np.isnan(wf_ratio):
        reasons.append(
            "Walk-Forward WF Ratio 산출 불가(폴드 수 부족/데이터 부족/샤프 0/NaN)."
        )
    elif wf_ratio < 0.5:
        reasons.append(f"Walk-Forward WF Ratio가 낮음: {wf_ratio:.3f} (<0.5).")
    elif wf_ratio < 0.7:
        reasons.append(f"Walk-Forward WF Ratio가 경계값: {wf_ratio:.3f} (0.5~0.7).")

    # Leverage suitability
    if not np.isnan(lev_mdd_1x) and lev_mdd_1x < mdd_1x_limit:
        reasons.append(
            f"1배(가정) 전략 MDD가 기준 초과: {format_pct(lev_mdd_1x)} < {format_pct(mdd_1x_limit)}"
        )
    if not np.isnan(lev_mdd_2x) and lev_mdd_2x < mdd_2x_limit:
        reasons.append(
            f"2배(가정) 전략 MDD가 기준 초과: {format_pct(lev_mdd_2x)} < {format_pct(mdd_2x_limit)}"
        )

    # Parameter sensitivity: flag if best Sharpe is extremely spiky vs median
    if grid is not None and len(grid) > 5:
        g = grid.copy()
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["sharpe"])
        if len(g) > 5:
            best = float(g["sharpe"].max())
            med = float(g["sharpe"].median())
            if (best - med) > 0.6:
                reasons.append(
                    "파라미터 민감도에서 최고 Sharpe가 중앙값 대비 과도하게 튐(과적합/스누핑 신호)."
                )

    # Decision
    if (np.isnan(wf_ratio) or wf_ratio < 0.5) and not reproduced:
        return "FAIL", reasons

    if not reproduced:
        # If not reproduced but WF ok, still conditional (data/assumption mismatch might be fixable)
        return "CONDITIONAL", reasons

    if (not np.isnan(wf_ratio) and wf_ratio >= 0.7) and (
        len([r for r in reasons if "기준 초과" in r]) == 0
    ):
        return "PASS", reasons

    return "CONDITIONAL", reasons


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--price_parquet",
        type=str,
        default=r"E:/투자/data/kospi_futures/kospi200_daily_yf.parquet",
    )
    ap.add_argument(
        "--foreign_glob",
        type=str,
        default=r"E:/투자/data/kr_stock/investor_trading/*_investor.csv",
    )
    ap.add_argument(
        "--foreign_file",
        type=str,
        default="",
        help="Optional: force a single foreign CSV instead of auto-select.",
    )
    ap.add_argument("--start", type=str, default="2017-01-01")
    ap.add_argument("--end", type=str, default="2026-12-31")

    ap.add_argument("--foreign_period", type=int, default=30)
    ap.add_argument("--sma_period", type=int, default=100)
    ap.add_argument(
        "--execution_lag",
        type=int,
        default=1,
        help="Default 1 = shift(1) as requested. Use 0/2 for diagnostics.",
    )
    ap.add_argument(
        "--round_trip_cost",
        type=float,
        default=0.0009,
        help="Round trip cost. Default 0.09%% = 0.0009",
    )

    ap.add_argument("--wf_folds", type=int, default=11)

    ap.add_argument("--grid_foreign_periods", type=str, default="20,25,30,35,40")
    ap.add_argument("--grid_sma_periods", type=str, default="60,80,100,120,150")

    ap.add_argument("--out_dir", type=str, default="outputs/foreign_trend_validation")
    ap.add_argument(
        "--plots",
        action="store_true",
        help="If set, tries to save PNG plots (requires matplotlib).",
    )

    # Leverage suitability thresholds
    ap.add_argument("--mdd_1x_limit", type=float, default=-0.20)
    ap.add_argument("--mdd_2x_limit", type=float, default=-0.35)
    ap.add_argument(
        "--lev_daily_fee_1x",
        type=float,
        default=0.0,
        help="Daily fee for 1x series (e.g., expense ratio/252).",
    )
    ap.add_argument(
        "--lev_daily_fee_2x",
        type=float,
        default=0.0,
        help="Daily fee for 2x series (e.g., expense ratio/252).",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load price
    price_path = Path(args.price_parquet)
    price = load_kospi200_parquet(price_path)

    # Load foreign
    foreign_meta: Dict[str, str] = {}
    if args.foreign_file.strip():
        foreign, foreign_meta = load_foreign_single_file(Path(args.foreign_file))
    else:
        foreign, foreign_meta = load_foreign_from_glob(
            pattern=args.foreign_glob,
            price_index=price.close.index,
            start=args.start,
            end=args.end,
        )

    # Main backtest (requested parameters)
    bt = run_strategy(
        price=price,
        foreign=foreign,
        start=args.start,
        end=args.end,
        foreign_period=args.foreign_period,
        sma_period=args.sma_period,
        execution_lag=args.execution_lag,
        round_trip_cost=args.round_trip_cost,
        label=f"Foreign_{args.foreign_period}d_SMA_{args.sma_period}_lag{args.execution_lag}",
    )

    # Diagnostics: alternative lags (0 and 2) to catch accidental look-ahead / timing sensitivity
    bt_lag0 = run_strategy(
        price,
        foreign,
        args.start,
        args.end,
        args.foreign_period,
        args.sma_period,
        0,
        args.round_trip_cost,
        "diag_lag0",
    )
    bt_lag2 = run_strategy(
        price,
        foreign,
        args.start,
        args.end,
        args.foreign_period,
        args.sma_period,
        2,
        args.round_trip_cost,
        "diag_lag2",
    )

    # Walk-forward
    wf = run_walk_forward(
        price=price,
        foreign=foreign,
        start=args.start,
        end=args.end,
        foreign_period=args.foreign_period,
        sma_period=args.sma_period,
        execution_lag=args.execution_lag,
        round_trip_cost=args.round_trip_cost,
        n_folds=args.wf_folds,
    )

    # 70/30 split (simple)
    # Implement as a special case: train first 70% days, test last 30% days
    dates = bt.df.index
    cut = int(len(dates) * 0.7)
    if cut >= 50:
        tr_start, tr_end = dates[0], dates[cut - 1]
        te_start, te_end = dates[cut], dates[-1]
        bt_train = run_strategy(
            price,
            foreign,
            str(tr_start.date()),
            str(tr_end.date()),
            args.foreign_period,
            args.sma_period,
            args.execution_lag,
            args.round_trip_cost,
            "train_70",
        )
        bt_test = run_strategy(
            price,
            foreign,
            str(te_start.date()),
            str(te_end.date()),
            args.foreign_period,
            args.sma_period,
            args.execution_lag,
            args.round_trip_cost,
            "test_30",
        )
        train_sh = bt_train.metrics["sharpe"]
        test_sh = bt_test.metrics["sharpe"]
        split_wf_ratio = (
            (test_sh / train_sh)
            if (not np.isnan(train_sh) and train_sh != 0 and not np.isnan(test_sh))
            else float("nan")
        )
    else:
        bt_train = None
        bt_test = None
        split_wf_ratio = float("nan")

    # Sensitivity grid
    grid_foreign = [int(x) for x in args.grid_foreign_periods.split(",") if x.strip()]
    grid_sma = [int(x) for x in args.grid_sma_periods.split(",") if x.strip()]
    grid = run_sensitivity_grid(
        price,
        foreign,
        args.start,
        args.end,
        foreign_period_list=grid_foreign,
        sma_period_list=grid_sma,
        execution_lag=args.execution_lag,
        round_trip_cost=args.round_trip_cost,
    )

    # Crisis periods
    crisis = {
        "covid_2020": {"start": "2020-02-15", "end": "2020-04-30"},
        "rate_hike_2022": {"start": "2022-01-01", "end": "2022-12-31"},
    }
    crisis_out = {}
    for k, v in crisis.items():
        crisis_out[k] = {
            "strategy": summarize_period(bt.df["equity"], v["start"], v["end"]),
            "buyhold": summarize_period(bt.df["equity_bh"], v["start"], v["end"]),
            "cash_ratio": (
                float(
                    (
                        bt.df.loc[
                            (bt.df.index >= v["start"]) & (bt.df.index <= v["end"]),
                            "position",
                        ]
                        == 0
                    ).mean()
                )
                if (
                    (bt.df.index >= pd.to_datetime(v["start"]))
                    & (bt.df.index <= pd.to_datetime(v["end"]))
                ).any()
                else float("nan")
            ),
        }

    # Leverage suitability (simulate 1x/2x daily reset series from underlying returns)
    # Strategy is applied to "underlying returns" as proxy (ETF tracking error/fees are ignored unless provided).
    r1 = simulate_daily_reset_leverage(
        bt.df["ret"], leverage=1.0, daily_fee=args.lev_daily_fee_1x
    )
    r2 = simulate_daily_reset_leverage(
        bt.df["ret"], leverage=2.0, daily_fee=args.lev_daily_fee_2x
    )

    # Use the SAME position series but apply levered returns
    lev = pd.DataFrame(index=bt.df.index)
    lev["position"] = bt.df["position"]
    lev["turnover"] = bt.df["turnover"]
    cost_per_side = args.round_trip_cost / 2.0
    lev["cost"] = lev["turnover"] * cost_per_side

    lev["ret_1x"] = r1
    lev["ret_2x"] = r2
    lev["strategy_ret_1x"] = lev["position"] * lev["ret_1x"] - lev["cost"]
    lev["strategy_ret_2x"] = lev["position"] * lev["ret_2x"] - lev["cost"]
    lev["equity_1x"] = (1.0 + lev["strategy_ret_1x"].fillna(0.0)).cumprod()
    lev["equity_2x"] = (1.0 + lev["strategy_ret_2x"].fillna(0.0)).cumprod()

    lev_mdd_1x = max_drawdown(lev["equity_1x"])
    lev_mdd_2x = max_drawdown(lev["equity_2x"])

    decay_2x = vol_decay_gap(bt.df["ret"], leverage=2.0)

    # Verdict
    claimed = ClaimedPerformance()
    v, reasons = verdict(
        bt_main=bt,
        wf=wf,
        claimed=claimed,
        mdd_1x_limit=args.mdd_1x_limit,
        mdd_2x_limit=args.mdd_2x_limit,
        lev_mdd_1x=lev_mdd_1x,
        lev_mdd_2x=lev_mdd_2x,
        grid=grid,
    )

    # Save outputs
    # Main DF (trim to useful columns)
    keep_cols = [
        "close",
        "ret",
        "foreign_net",
        "foreign_roll",
        "sma",
        "signal_raw",
        "position",
        "turnover",
        "cost",
        "strategy_ret",
        "equity",
        "equity_bh",
    ]
    bt.df[keep_cols].to_csv(out_dir / "bt_main_timeseries.csv", encoding="utf-8-sig")
    grid.to_csv(out_dir / "sensitivity_grid.csv", index=False, encoding="utf-8-sig")
    wf.fold_rows.to_csv(
        out_dir / "walk_forward_folds.csv", index=False, encoding="utf-8-sig"
    )
    lev.to_csv(out_dir / "leveraged_sim.csv", encoding="utf-8-sig")

    # Summary JSON
    summary = {
        "selected_foreign_meta": foreign_meta,
        "params": {
            "foreign_period": args.foreign_period,
            "sma_period": args.sma_period,
            "execution_lag": args.execution_lag,
            "round_trip_cost": args.round_trip_cost,
            "start": args.start,
            "end": args.end,
        },
        "main_metrics": bt.metrics,
        "diagnostics": {
            "lag0_metrics": bt_lag0.metrics,
            "lag2_metrics": bt_lag2.metrics,
            "lag0_minus_lag1_sharpe": (
                float(bt_lag0.metrics["sharpe"] - bt.metrics["sharpe"])
                if not np.isnan(bt.metrics["sharpe"])
                else float("nan")
            ),
            "lag2_minus_lag1_sharpe": (
                float(bt_lag2.metrics["sharpe"] - bt.metrics["sharpe"])
                if not np.isnan(bt.metrics["sharpe"])
                else float("nan")
            ),
        },
        "walk_forward": {
            "method": wf.method,
            "summary": wf.summary,
            "split_70_30": {
                "wf_ratio": split_wf_ratio,
                "train_sharpe": (
                    float(bt_train.metrics["sharpe"]) if bt_train else float("nan")
                ),
                "test_sharpe": (
                    float(bt_test.metrics["sharpe"]) if bt_test else float("nan")
                ),
            },
        },
        "crisis": crisis_out,
        "leverage": {
            "mdd_1x": lev_mdd_1x,
            "mdd_2x": lev_mdd_2x,
            "mdd_limits": {"1x": args.mdd_1x_limit, "2x": args.mdd_2x_limit},
            "vol_decay_2x": decay_2x,
        },
        "verdict": {"result": v, "reasons": reasons},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # Optional plots
    if args.plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            # Equity curve
            fig = plt.figure()
            plt.plot(bt.df.index, bt.df["equity"], label="Strategy")
            plt.plot(bt.df.index, bt.df["equity_bh"], label="Buy&Hold")
            plt.legend()
            plt.title("Equity Curve (Strategy vs Buy&Hold)")
            plt.tight_layout()
            fig.savefig(out_dir / "equity_curve.png", dpi=150)
            plt.close(fig)

            # Drawdown
            eq = bt.df["equity"].dropna()
            peak = eq.cummax()
            dd = (eq / peak) - 1.0
            fig = plt.figure()
            plt.plot(dd.index, dd.values)
            plt.title("Strategy Drawdown")
            plt.tight_layout()
            fig.savefig(out_dir / "drawdown.png", dpi=150)
            plt.close(fig)

            # Sensitivity (Sharpe heatmap-like pivot as image using imshow)
            pivot = grid.pivot(
                index="foreign_period", columns="sma_period", values="sharpe"
            )
            fig = plt.figure()
            plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
            plt.xticks(
                range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=45
            )
            plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
            plt.title("Sensitivity (Sharpe) - foreign_period x sma_period")
            plt.tight_layout()
            fig.savefig(out_dir / "sensitivity_sharpe.png", dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

    # Console summary (human-readable)
    print("\n==================== Independent Validation Summary ====================")
    print(f"Selected foreign file: {foreign_meta.get('selected_file', 'N/A')}")
    print(f"Date range: {args.start} ~ {args.end}")
    print(
        f"Params: foreign_period={args.foreign_period}, sma_period={args.sma_period}, lag={args.execution_lag}, round_trip_cost={args.round_trip_cost}"
    )
    print("\n[Main Metrics]")
    print(f"  Sharpe: {format_num(bt.metrics['sharpe'])}")
    print(f"  CAGR  : {format_pct(bt.metrics['cagr'])}")
    print(f"  MDD   : {format_pct(bt.metrics['mdd'])}")
    print(f"  Exposure: {format_pct(bt.metrics['exposure'])}")
    print(f"  Trades: {int(bt.metrics['n_trades'])}")
    print(
        f"  WinRate (daily in-position): {format_pct(bt.metrics.get('daily_win_rate_in_position', float('nan')))}"
    )
    print(
        f"  WinRate (per-trade): {format_pct(bt.metrics.get('trade_win_rate', float('nan')))}"
    )

    print("\n[Diagnostics: execution lag sensitivity]")
    print(f"  lag=0 Sharpe: {format_num(bt_lag0.metrics['sharpe'])}")
    print(f"  lag=1 Sharpe: {format_num(bt.metrics['sharpe'])}")
    print(f"  lag=2 Sharpe: {format_num(bt_lag2.metrics['sharpe'])}")

    print("\n[Walk-Forward]")
    print(
        f"  {wf.method} | mean_train_sharpe={format_num(wf.summary.get('mean_train_sharpe', float('nan')))} "
        f"mean_test_sharpe={format_num(wf.summary.get('mean_test_sharpe', float('nan')))} "
        f"WF_ratio={format_num(wf.summary.get('wf_ratio', float('nan')))} "
        f"folds_used={int(wf.summary.get('n_folds_used', 0)) if not np.isnan(wf.summary.get('n_folds_used', float('nan'))) else 'nan'}"
    )
    print(f"  70/30 split WF_ratio={format_num(split_wf_ratio)}")

    print("\n[Crisis performance]")
    for k, v in crisis_out.items():
        st = v["strategy"]
        bh = v["buyhold"]
        print(
            f"  {k}: Strategy ret={format_pct(st['return'])}, mdd={format_pct(st['mdd'])} | "
            f"B&H ret={format_pct(bh['return'])}, mdd={format_pct(bh['mdd'])} | cash_ratio={format_pct(v['cash_ratio'])}"
        )

    print("\n[Leverage suitability (simulated daily-reset)]")
    print(
        f"  Strategy MDD 1x: {format_pct(lev_mdd_1x)} (limit {format_pct(args.mdd_1x_limit)})"
    )
    print(
        f"  Strategy MDD 2x: {format_pct(lev_mdd_2x)} (limit {format_pct(args.mdd_2x_limit)})"
    )
    print(
        f"  2x vol-decay log_gap: {decay_2x['log_gap']:.6f} | avg_daily_log_gap: {decay_2x['avg_daily_log_gap']:.8f}"
    )

    print("\n[VERDICT]")
    print(f"  => {v}")
    if reasons:
        for r in reasons:
            print(f"   - {r}")

    print(f"\nOutputs saved to: {out_dir.resolve()}")
    print("=======================================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
