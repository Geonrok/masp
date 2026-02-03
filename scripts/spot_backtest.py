"""
현물 전략 백테스트

현물 vs 선물 차이점:
- 숏 포지션 불가 (롱만 가능)
- 수수료 낮음 (0.1% vs 0.04%)
- 레버리지 없음
- 펀딩비 없음
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("E:/data/crypto_ohlcv")

print("=" * 70)
print("현물 전략 백테스트")
print("=" * 70)


def calc_kama(prices, period=10, fast=2, slow=30):
    n = len(prices)
    kama = np.full(n, np.nan)
    if n < period + 1:
        return kama
    kama[period - 1] = np.mean(prices[:period])
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    for i in range(period, n):
        change = abs(prices[i] - prices[i - period])
        volatility = np.sum(np.abs(np.diff(prices[i - period : i + 1])))
        er = change / volatility if volatility > 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])
    return kama


def calc_sma(prices, period):
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


def calc_tsmom(prices, period=60):
    n = len(prices)
    signal = np.zeros(n, dtype=bool)
    for i in range(period, n):
        signal[i] = prices[i] > prices[i - period]
    return signal


def load_ohlcv(exchange, timeframe="1d", min_days=100):
    folder = DATA_ROOT / f"{exchange}_{timeframe}"
    if not folder.exists():
        return {}
    data = {}
    for f in folder.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            date_col = [
                c for c in df.columns if "date" in c.lower() or "time" in c.lower()
            ]
            if not date_col:
                continue
            df["date"] = pd.to_datetime(df[date_col[0]]).dt.normalize()
            df = df.set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            required = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                continue
            df = df[required]
            if len(df) >= min_days:
                data[f.stem] = df
        except:
            continue
    return data


def backtest(data, btc_data, strategy, kama_p=10, tsmom_p=60, btc_ma_p=30, max_pos=10):
    if not data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0}

    btc_gate = None
    if btc_data is not None:
        btc_prices = btc_data["close"].values
        btc_ma = calc_sma(btc_prices, btc_ma_p)
        btc_gate = pd.Series(btc_prices > btc_ma, index=btc_data.index)

    signal_data = {}
    for symbol, df in data.items():
        if len(df) < max(kama_p, tsmom_p, 100):
            continue
        prices = df["close"].values

        if strategy == "buy_hold":
            signal = np.ones(len(prices), dtype=bool)
        elif strategy == "kama_only":
            kama = calc_kama(prices, kama_p)
            signal = prices > kama
        elif strategy == "tsmom_only":
            signal = calc_tsmom(prices, tsmom_p)
        elif strategy == "or_loose":
            kama = calc_kama(prices, kama_p)
            signal = (prices > kama) | calc_tsmom(prices, tsmom_p)
            if btc_gate is not None:
                aligned = btc_gate.reindex(df.index).fillna(False)
                signal = signal & aligned.values
        elif strategy == "or_loose_no_gate":
            kama = calc_kama(prices, kama_p)
            signal = (prices > kama) | calc_tsmom(prices, tsmom_p)
        else:
            kama = calc_kama(prices, kama_p)
            signal = (prices > kama) | calc_tsmom(prices, tsmom_p)

        df = df.copy()
        df["signal"] = signal
        df["dvol"] = df["close"] * df["volume"]
        signal_data[symbol] = df

    if not signal_data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0}

    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))
    capital = 10000.0
    cash = capital
    positions = {}
    values = [capital]
    returns = []
    trades = 0

    for i, date in enumerate(all_dates):
        prices_today = {}
        signals_today = {}
        vols_today = {}

        for sym, df in signal_data.items():
            if date in df.index:
                prices_today[sym] = df.loc[date, "close"]
                signals_today[sym] = df.loc[date, "signal"]
                vols_today[sym] = df.loc[date, "dvol"]

        pos_value = sum(
            shares * prices_today.get(sym, cost)
            for sym, (shares, cost) in positions.items()
        )
        port_value = cash + pos_value

        if i > 0:
            ret = (port_value - values[-1]) / values[-1] if values[-1] > 0 else 0
            returns.append(ret)
        values.append(port_value)

        active = [(s, vols_today.get(s, 0)) for s, sig in signals_today.items() if sig]
        active.sort(key=lambda x: x[1], reverse=True)
        targets = set(s for s, _ in active[:max_pos])

        current_syms = set(positions.keys())
        exits = current_syms - targets
        new_entries = targets - current_syms

        # 현물 수수료 (업비트 0.05%, 바이낸스 0.1%)
        for sym in exits:
            if sym in positions and sym in prices_today:
                shares, _ = positions[sym]
                sell_price = prices_today[sym] * 0.998  # 0.1% 슬리피지 + 0.1% 수수료
                cash += shares * sell_price
                del positions[sym]
                trades += 1

        if targets:
            curr_val = cash + sum(
                s * prices_today.get(sym, 0) for sym, (s, _) in positions.items()
            )
            per_pos = curr_val / len(targets)

            for sym in new_entries:
                if sym in prices_today:
                    buy_price = prices_today[sym] * 1.002
                    cost = per_pos
                    if cost <= cash:
                        shares = cost / buy_price
                        cash -= cost
                        positions[sym] = (shares, buy_price)
                        trades += 1

    final = values[-1]
    total_ret = (final - capital) / capital
    rets = np.array(returns)
    sharpe = (
        np.mean(rets) / np.std(rets) * np.sqrt(252)
        if len(rets) > 1 and np.std(rets) > 0
        else 0
    )
    vals = np.array(values)
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    mdd = np.min(dd)

    return {"return": total_ret, "sharpe": sharpe, "mdd": mdd, "trades": trades}


def main():
    # 거래소별 테스트
    exchanges = ["binance_spot", "upbit", "bithumb"]
    strategies = ["buy_hold", "kama_only", "tsmom_only", "or_loose", "or_loose_no_gate"]
    universes = ["btc_only", "top5", "top10", "top20"]
    params = [(10, 60, 30), (20, 30, 30), (5, 90, 30)]

    results = []

    for exchange in exchanges:
        print(f"\n{exchange.upper()} 테스트 중...")
        data = load_ohlcv(exchange, "1d", 100)
        if not data:
            print(f"  데이터 없음")
            continue
        print(f"  심볼: {len(data)}개")

        btc_data = None
        for k, v in data.items():
            if "BTC" in k.upper() and "DOWN" not in k.upper():
                btc_data = v
                break

        for universe in universes:
            if universe == "btc_only":
                filtered = {
                    k: v
                    for k, v in data.items()
                    if "BTC" in k.upper() and "DOWN" not in k.upper()
                }
                if not filtered:
                    continue
                filtered = dict(list(filtered.items())[:1])
            elif universe == "top5":
                vols = [
                    (s, (df["close"] * df["volume"]).mean()) for s, df in data.items()
                ]
                vols.sort(key=lambda x: x[1], reverse=True)
                filtered = {s: data[s] for s, _ in vols[:5]}
            elif universe == "top10":
                vols = [
                    (s, (df["close"] * df["volume"]).mean()) for s, df in data.items()
                ]
                vols.sort(key=lambda x: x[1], reverse=True)
                filtered = {s: data[s] for s, _ in vols[:10]}
            else:
                vols = [
                    (s, (df["close"] * df["volume"]).mean()) for s, df in data.items()
                ]
                vols.sort(key=lambda x: x[1], reverse=True)
                filtered = {s: data[s] for s, _ in vols[:20]}

            for strategy in strategies:
                for kama_p, tsmom_p, btc_ma_p in params:
                    result = backtest(
                        filtered,
                        btc_data,
                        strategy,
                        kama_p,
                        tsmom_p,
                        btc_ma_p,
                        max_pos=min(10, len(filtered)),
                    )
                    results.append(
                        {
                            "exchange": exchange,
                            "universe": universe,
                            "strategy": strategy,
                            "kama_p": kama_p,
                            "tsmom_p": tsmom_p,
                            **result,
                        }
                    )

    df = pd.DataFrame(results)
    output_file = Path(
        "E:/투자/Multi-Asset Strategy Platform/outputs/spot_backtest_results.csv"
    )
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("현물 백테스트 결과")
    print("=" * 70)

    # 거래소별 최고
    print("\n[거래소별 최고 전략]")
    for ex in exchanges:
        ex_df = df[df["exchange"] == ex]
        if len(ex_df) == 0:
            continue
        best = ex_df.loc[ex_df["sharpe"].idxmax()]
        print(
            f"  {ex}: {best['strategy']} ({best['universe']}) | "
            f"수익률: {best['return']*100:.1f}% | 샤프: {best['sharpe']:.2f} | MDD: {best['mdd']*100:.1f}%"
        )

    # 전략별 평균
    print("\n[전략별 평균 성과]")
    strat_summary = (
        df.groupby("strategy")
        .agg({"return": "mean", "sharpe": "mean", "mdd": "mean"})
        .round(3)
    )
    strat_summary = strat_summary.sort_values("sharpe", ascending=False)
    for idx, row in strat_summary.iterrows():
        print(
            f"  {idx}: 수익률 {row['return']*100:.1f}% | 샤프 {row['sharpe']:.2f} | MDD {row['mdd']*100:.1f}%"
        )

    # 유니버스별 평균
    print("\n[유니버스별 평균 성과]")
    univ_summary = (
        df.groupby("universe")
        .agg({"return": "mean", "sharpe": "mean", "mdd": "mean"})
        .round(3)
    )
    univ_summary = univ_summary.sort_values("sharpe", ascending=False)
    for idx, row in univ_summary.iterrows():
        print(
            f"  {idx}: 수익률 {row['return']*100:.1f}% | 샤프 {row['sharpe']:.2f} | MDD {row['mdd']*100:.1f}%"
        )

    # 상위 15개
    print("\n[상위 15개 전략]")
    top15 = df.nlargest(15, "sharpe")
    for i, (_, row) in enumerate(top15.iterrows(), 1):
        print(
            f"  #{i}: {row['exchange']} | {row['strategy']} | {row['universe']} | KAMA{row['kama_p']}/TSMOM{row['tsmom_p']}"
        )
        print(
            f"       수익률: {row['return']*100:.1f}% | 샤프: {row['sharpe']:.2f} | MDD: {row['mdd']*100:.1f}%"
        )

    print(f"\n결과 저장: {output_file}")
    return df


if __name__ == "__main__":
    results = main()
