"""
전체 조합 백테스트 - 4시간봉 버전
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
import warnings

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")

print("=" * 80)
print("전체 조합 백테스트 - 4시간봉")
print(f'시작 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print("=" * 80)


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


def load_ohlcv_4h(exchange="binance_futures"):
    folder = DATA_ROOT / f"{exchange}_4h"
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
            df["datetime"] = pd.to_datetime(df[date_col[0]])
            df = df.set_index("datetime").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            required = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                continue
            df = df[required]
            if len(df) >= 500:  # 4h 기준 최소 데이터
                symbol = f.stem
                data[symbol] = df
        except:
            continue
    return data


def backtest_4h(data, btc_data, kama_p=40, tsmom_p=240, btc_ma_p=120, max_pos=10):
    """4시간봉 백테스트 (파라미터는 4h 캔들 수 기준)"""
    if not data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0, "win_rate": 0}

    # BTC Gate
    btc_gate = None
    if btc_data is not None:
        btc_prices = btc_data["close"].values
        btc_ma = calc_sma(btc_prices, btc_ma_p)
        btc_gate = pd.Series(btc_prices > btc_ma, index=btc_data.index)

    signal_data = {}
    for symbol, df in data.items():
        if len(df) < max(kama_p, tsmom_p, 500):
            continue

        prices = df["close"].values
        kama = calc_kama(prices, kama_p)
        kama_signal = prices > kama
        tsmom_signal = calc_tsmom(prices, tsmom_p)
        signal = kama_signal | tsmom_signal

        if btc_gate is not None:
            aligned = btc_gate.reindex(df.index, method="ffill").fillna(False)
            signal = signal & aligned.values

        df = df.copy()
        df["signal"] = signal
        df["dvol"] = df["close"] * df["volume"]
        signal_data[symbol] = df

    if not signal_data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0, "win_rate": 0}

    all_times = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))

    capital = 10000.0
    cash = capital
    positions = {}
    values = [capital]
    returns = []
    trades = 0
    winning_trades = 0

    slippage = 0.005
    commission = 0.001
    cost_factor = 1 + slippage + commission
    sell_factor = 1 - slippage - commission

    for i, dt in enumerate(all_times):
        prices_now = {}
        signals_now = {}
        vols_now = {}

        for sym, df in signal_data.items():
            if dt in df.index:
                prices_now[sym] = df.loc[dt, "close"]
                signals_now[sym] = df.loc[dt, "signal"]
                vols_now[sym] = df.loc[dt, "dvol"]

        pos_value = sum(
            shares * prices_now.get(sym, cost)
            for sym, (shares, cost) in positions.items()
        )
        port_value = cash + pos_value

        if i > 0:
            ret = (port_value - values[-1]) / values[-1] if values[-1] > 0 else 0
            returns.append(ret)

        values.append(port_value)

        active = [(s, vols_now.get(s, 0)) for s, sig in signals_now.items() if sig]
        active.sort(key=lambda x: x[1], reverse=True)
        targets = set(s for s, _ in active[:max_pos])

        current_syms = set(positions.keys())
        exits = current_syms - targets
        new_entries = targets - current_syms

        for sym in exits:
            if sym in positions and sym in prices_now:
                shares, entry_price = positions[sym]
                sell_price = prices_now[sym] * sell_factor
                cash += shares * sell_price
                if sell_price > entry_price:
                    winning_trades += 1
                del positions[sym]
                trades += 1

        if targets:
            curr_val = cash + sum(
                s * prices_now.get(sym, 0) for sym, (s, _) in positions.items()
            )
            per_pos = curr_val / len(targets) if targets else 0

            for sym in new_entries:
                if sym in prices_now:
                    buy_price = prices_now[sym] * cost_factor
                    cost = per_pos
                    if cost <= cash and buy_price > 0:
                        shares = cost / buy_price
                        cash -= cost
                        positions[sym] = (shares, buy_price)
                        trades += 1

    final = values[-1]
    total_ret = (final - capital) / capital
    rets = np.array(returns)

    # 4시간봉은 하루 6개 캔들 → 연 252*6 = 1512
    sharpe = (
        np.mean(rets) / np.std(rets) * np.sqrt(1512)
        if len(rets) > 1 and np.std(rets) > 0
        else 0
    )

    vals = np.array(values)
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    mdd = np.min(dd)
    win_rate = winning_trades / (trades / 2) if trades > 0 else 0

    return {
        "return": total_ret,
        "sharpe": sharpe,
        "mdd": mdd,
        "trades": trades,
        "win_rate": win_rate,
    }


def main():
    print("\n데이터 로드 중...")
    ohlcv_4h = load_ohlcv_4h("binance_futures")
    print(f"  Binance Futures 4h: {len(ohlcv_4h)}개 심볼")

    btc_data = ohlcv_4h.get("BTCUSDT")

    def get_universe(data, universe_type):
        if universe_type == "btc_only":
            return {k: v for k, v in data.items() if "BTC" in k.upper()}
        elif universe_type == "top5":
            vols = [(s, (df["close"] * df["volume"]).mean()) for s, df in data.items()]
            vols.sort(key=lambda x: x[1], reverse=True)
            return {s: data[s] for s, _ in vols[:5]}
        elif universe_type == "top10":
            vols = [(s, (df["close"] * df["volume"]).mean()) for s, df in data.items()]
            vols.sort(key=lambda x: x[1], reverse=True)
            return {s: data[s] for s, _ in vols[:10]}
        return data

    # 4h 파라미터 (일봉 대비 6배)
    param_sets = [
        {"kama_p": 30, "tsmom_p": 360, "btc_ma_p": 180},  # KAMA5/TSMOM60일 → 4h
        {"kama_p": 60, "tsmom_p": 360, "btc_ma_p": 180},  # KAMA10/TSMOM60일
        {"kama_p": 120, "tsmom_p": 180, "btc_ma_p": 300},  # KAMA20/TSMOM30일
    ]

    universes = ["btc_only", "top5", "top10"]

    results = []
    total = len(universes) * len(param_sets)
    count = 0

    print(f"\n총 테스트: {total}개")
    print("\n백테스트 실행 중...")

    for universe in universes:
        universe_data = get_universe(ohlcv_4h, universe)
        if not universe_data:
            continue

        for params in param_sets:
            count += 1
            print(
                f'  {count}/{total}: {universe} | KAMA{params["kama_p"]}/TSMOM{params["tsmom_p"]}'
            )

            result = backtest_4h(
                universe_data,
                btc_data,
                kama_p=params["kama_p"],
                tsmom_p=params["tsmom_p"],
                btc_ma_p=params["btc_ma_p"],
                max_pos=min(10, len(universe_data)),
            )

            results.append(
                {
                    "timeframe": "4h",
                    "universe": universe,
                    "kama_p": params["kama_p"],
                    "tsmom_p": params["tsmom_p"],
                    "btc_ma_p": params["btc_ma_p"],
                    **result,
                }
            )

    df = pd.DataFrame(results)
    output_file = Path(
        "E:/투자/Multi-Asset Strategy Platform/outputs/backtest_4h_results.csv"
    )
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("4시간봉 백테스트 결과")
    print("=" * 80)

    for _, row in df.iterrows():
        print(f"  {row['universe']} | KAMA{row['kama_p']}/TSMOM{row['tsmom_p']}")
        print(
            f"    수익률: {row['return']*100:.1f}% | 샤프: {row['sharpe']:.2f} | MDD: {row['mdd']*100:.1f}%"
        )

    print(f"\n결과 저장: {output_file}")
    return df


if __name__ == "__main__":
    results = main()
