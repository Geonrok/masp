"""
파라미터 안정성 테스트

목적: 기존 전략(KAMA5/TSMOM90/MA30) 주변 파라미터들의 성과를 확인하여
과적합 여부를 판단
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("E:/data/crypto_ohlcv")


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


def calc_tsmom(prices, period):
    n = len(prices)
    signal = np.zeros(n, dtype=bool)
    for i in range(period, n):
        signal[i] = prices[i] > prices[i - period]
    return signal


def load_ohlcv(exchange, min_days=100):
    folder = DATA_ROOT / f"{exchange}_1d"
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


def backtest(data, btc_data, kama_p, tsmom_p, btc_ma_p, max_pos=10):
    if not data or btc_data is None:
        return {"return": 0, "sharpe": 0, "mdd": 0}

    btc_prices = btc_data["close"].values
    btc_ma = calc_sma(btc_prices, btc_ma_p)
    btc_gate = pd.Series(btc_prices > btc_ma, index=btc_data.index)

    signal_data = {}
    for symbol, df in data.items():
        if len(df) < max(kama_p, tsmom_p, 100):
            continue
        prices = df["close"].values
        kama = calc_kama(prices, kama_p)
        signal = (prices > kama) | calc_tsmom(prices, tsmom_p)
        aligned = btc_gate.reindex(df.index).fillna(False)
        signal = signal & aligned.values
        df = df.copy()
        df["signal"] = signal
        df["dvol"] = df["close"] * df["volume"]
        signal_data[symbol] = df

    if not signal_data:
        return {"return": 0, "sharpe": 0, "mdd": 0}

    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))
    capital = 10000.0
    cash = capital
    positions = {}
    values = [capital]
    returns = []

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

        exits = set(positions.keys()) - targets
        new_entries = targets - set(positions.keys())

        for sym in exits:
            if sym in positions and sym in prices_today:
                shares, _ = positions[sym]
                cash += shares * prices_today[sym] * 0.998
                del positions[sym]

        if targets:
            per_pos = (
                cash
                + sum(s * prices_today.get(sym, 0) for sym, (s, _) in positions.items())
            ) / len(targets)
            for sym in new_entries:
                if sym in prices_today:
                    buy_price = prices_today[sym] * 1.002
                    if per_pos <= cash:
                        shares = per_pos / buy_price
                        cash -= per_pos
                        positions[sym] = (shares, buy_price)

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
    mdd = np.min((vals - peak) / peak)

    return {"return": total_ret, "sharpe": sharpe, "mdd": mdd}


def main():
    print("파라미터 안정성 테스트")
    print("=" * 70)

    # 데이터 로드
    data = load_ohlcv("upbit", 100)
    print(f"Upbit 심볼: {len(data)}개")

    # Top20 유니버스
    vols = [(s, (df["close"] * df["volume"]).mean()) for s, df in data.items()]
    vols.sort(key=lambda x: x[1], reverse=True)
    filtered = {s: data[s] for s, _ in vols[:20]}

    btc_data = None
    for k, v in data.items():
        if "BTC" in k.upper():
            btc_data = v
            break

    # 파라미터 그리드 (세밀하게)
    kama_range = [3, 5, 7, 10, 15, 20, 30]
    tsmom_range = [30, 60, 90, 120, 180]
    btc_ma_range = [20, 30, 50]

    results = []
    total = len(kama_range) * len(tsmom_range) * len(btc_ma_range)

    print(f"\n테스트 조합: {total}개")
    print("테스트 중...")

    count = 0
    for kama_p in kama_range:
        for tsmom_p in tsmom_range:
            for btc_ma_p in btc_ma_range:
                count += 1
                if count % 20 == 0:
                    print(f"  {count}/{total}")
                result = backtest(filtered, btc_data, kama_p, tsmom_p, btc_ma_p)
                results.append(
                    {
                        "kama": kama_p,
                        "tsmom": tsmom_p,
                        "btc_ma": btc_ma_p,
                        "sharpe": result["sharpe"],
                        "return": result["return"] * 100,
                        "mdd": result["mdd"] * 100,
                    }
                )

    df = pd.DataFrame(results)

    # 기존 전략 찾기
    original = df[(df["kama"] == 5) & (df["tsmom"] == 90) & (df["btc_ma"] == 30)]

    print("\n" + "=" * 70)
    print("기존 전략 (KAMA5/TSMOM90/MA30)")
    print("=" * 70)
    if len(original) > 0:
        o = original.iloc[0]
        print(f'  샤프: {o["sharpe"]:.2f}')
        print(f'  수익률: {o["return"]:.1f}%')
        print(f'  MDD: {o["mdd"]:.1f}%')
        original_sharpe = o["sharpe"]
    else:
        original_sharpe = 0

    # 상위 10개
    print("\n" + "=" * 70)
    print("상위 10개 파라미터 조합")
    print("=" * 70)
    top10 = df.nlargest(10, "sharpe")
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        marker = (
            " ★ 기존"
            if r["kama"] == 5 and r["tsmom"] == 90 and r["btc_ma"] == 30
            else ""
        )
        print(
            f'  #{i}: KAMA{r["kama"]}/TSMOM{r["tsmom"]}/MA{r["btc_ma"]} | '
            f'샤프: {r["sharpe"]:.2f} | 수익률: {r["return"]:.1f}% | MDD: {r["mdd"]:.1f}%{marker}'
        )

    # 파라미터별 평균 샤프
    print("\n" + "=" * 70)
    print("파라미터별 평균 샤프비율 (안정성 확인)")
    print("=" * 70)

    print("\n[KAMA 기간별]")
    kama_avg = df.groupby("kama")["sharpe"].agg(["mean", "std"]).round(2)
    for k, row in kama_avg.iterrows():
        marker = " ← 기존" if k == 5 else ""
        print(
            f'  KAMA {k:>2}: 평균 {row["mean"]:.2f} (표준편차 {row["std"]:.2f}){marker}'
        )

    print("\n[TSMOM 기간별]")
    tsmom_avg = df.groupby("tsmom")["sharpe"].agg(["mean", "std"]).round(2)
    for k, row in tsmom_avg.iterrows():
        marker = " ← 기존" if k == 90 else ""
        print(
            f'  TSMOM {k:>3}: 평균 {row["mean"]:.2f} (표준편차 {row["std"]:.2f}){marker}'
        )

    print("\n[BTC MA 기간별]")
    ma_avg = df.groupby("btc_ma")["sharpe"].agg(["mean", "std"]).round(2)
    for k, row in ma_avg.iterrows():
        marker = " ← 기존" if k == 30 else ""
        print(
            f'  MA {k:>2}: 평균 {row["mean"]:.2f} (표준편차 {row["std"]:.2f}){marker}'
        )

    # 전체 분포
    print("\n" + "=" * 70)
    print("샤프비율 분포")
    print("=" * 70)
    print(f'  최소: {df["sharpe"].min():.2f}')
    print(f'  최대: {df["sharpe"].max():.2f}')
    print(f'  평균: {df["sharpe"].mean():.2f}')
    print(f'  표준편차: {df["sharpe"].std():.2f}')

    rank = len(df[df["sharpe"] > original_sharpe]) + 1
    print(f"  기존 전략 순위: {rank} / {len(df)}")

    # 인접 파라미터 성과 (과적합 판단)
    print("\n" + "=" * 70)
    print("인접 파라미터 성과 (과적합 판단)")
    print("=" * 70)

    # KAMA 3~7, TSMOM 60~120, MA 20~50 범위
    neighbors = df[
        (df["kama"].isin([3, 5, 7]))
        & (df["tsmom"].isin([60, 90, 120]))
        & (df["btc_ma"].isin([20, 30, 50]))
    ]
    print(f"  인접 조합 수: {len(neighbors)}개")
    print(f'  인접 평균 샤프: {neighbors["sharpe"].mean():.2f}')
    print(f'  인접 최소 샤프: {neighbors["sharpe"].min():.2f}')
    print(f'  인접 최대 샤프: {neighbors["sharpe"].max():.2f}')

    # 저장
    output_file = Path(
        "E:/투자/Multi-Asset Strategy Platform/outputs/parameter_stability_test.csv"
    )
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {output_file}")

    return df


if __name__ == "__main__":
    results = main()
