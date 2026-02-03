"""
빠른 종합 백테스트 - 주요 조합만 테스트
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("종합 백테스트 - 모든 거래소, 모든 전략")
print("=" * 80)


# 지표 함수들
def calc_kama(prices, period=5, fast=2, slow=30):
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


def calc_tsmom(prices, period=90):
    n = len(prices)
    signal = np.zeros(n, dtype=bool)
    for i in range(period, n):
        signal[i] = prices[i] > prices[i - period]
    return signal


# 데이터 로드
DATA_ROOT = Path("E:/data/crypto_ohlcv")


def load_data(exchange, min_days=100):
    folder = DATA_ROOT / f"{exchange}_1d"
    if not folder.exists():
        return {}, None

    data = {}
    btc_data = None

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
                symbol = f.stem
                data[symbol] = df
                if symbol.upper() in ["BTC", "BTCUSDT"]:
                    btc_data = df
        except:
            continue

    if btc_data is None:
        for k, v in data.items():
            if "BTC" in k.upper() and "DOWN" not in k.upper():
                btc_data = v
                break

    return data, btc_data


def backtest(
    data,
    btc_data,
    strategy,
    kama_p=5,
    tsmom_p=90,
    btc_ma_p=30,
    use_gate=True,
    max_pos=20,
):
    """바이어스 제거 백테스트"""
    if not data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0}

    # BTC Gate 계산
    btc_gate = None
    if use_gate and btc_data is not None:
        btc_prices = btc_data["close"].values
        btc_ma = calc_sma(btc_prices, btc_ma_p)
        btc_gate = pd.Series(btc_prices > btc_ma, index=btc_data.index)

    # 시그널 데이터 준비
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
        elif strategy == "kama_or_tsmom":
            kama = calc_kama(prices, kama_p)
            signal = (prices > kama) | calc_tsmom(prices, tsmom_p)
        elif strategy == "kama_and_tsmom":
            kama = calc_kama(prices, kama_p)
            signal = (prices > kama) & calc_tsmom(prices, tsmom_p)
        else:  # or_loose
            kama = calc_kama(prices, kama_p)
            signal = (prices > kama) | calc_tsmom(prices, tsmom_p)

        # BTC Gate 적용
        if use_gate and btc_gate is not None:
            aligned = btc_gate.reindex(df.index).fillna(False)
            signal = signal & aligned.values

        df = df.copy()
        df["signal"] = signal
        df["dvol"] = df["close"] * df["volume"]
        signal_data[symbol] = df

    if not signal_data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0}

    # 시뮬레이션 (바이어스 제거: Day T signal -> Day T+1 execution)
    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))

    capital = 10000.0
    cash = capital
    positions = {}  # symbol -> (shares, cost)
    prev_prices = {}
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

        # 포트폴리오 가치 계산 (mark-to-market)
        pos_value = 0
        prev_pos_value = 0
        for sym, (shares, cost) in positions.items():
            if sym in prices_today:
                pos_value += shares * prices_today[sym]
                if sym in prev_prices:
                    prev_pos_value += shares * prev_prices[sym]
                else:
                    prev_pos_value += shares * cost

        port_value = cash + pos_value
        prev_port_value = cash + prev_pos_value

        # 일간 수익률 계산
        if i > 0:
            ret = (port_value - values[-1]) / values[-1] if values[-1] > 0 else 0
            returns.append(ret)

        values.append(port_value)

        # 타겟 포지션 결정 (오늘 시그널 기반, 내일 실행)
        active = [(s, vols_today.get(s, 0)) for s, sig in signals_today.items() if sig]
        active.sort(key=lambda x: x[1], reverse=True)
        targets = set(s for s, _ in active[:max_pos])

        current_syms = set(positions.keys())
        new_entries = targets - current_syms
        exits = current_syms - targets

        # 매도 (슬리피지 + 수수료 적용)
        for sym in exits:
            if sym in positions and sym in prices_today:
                shares, _ = positions[sym]
                sell_price = prices_today[sym] * 0.994  # 0.5% 슬리피지 + 0.1% 수수료
                cash += shares * sell_price
                del positions[sym]
                trades += 1

        # 매수
        if targets:
            curr_val = cash + sum(
                s * prices_today.get(sym, 0) for sym, (s, _) in positions.items()
            )
            per_pos = curr_val / len(targets)

            for sym in new_entries:
                if sym in prices_today:
                    buy_price = prices_today[sym] * 1.006  # 0.5% 슬리피지 + 0.1% 수수료
                    cost = per_pos
                    if cost <= cash:
                        shares = cost / buy_price
                        cash -= cost
                        positions[sym] = (shares, buy_price)
                        trades += 1

        prev_prices = prices_today.copy()

    # 결과 계산
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
    # 테스트 설정
    exchanges = ["upbit", "bithumb", "binance_spot", "binance_futures"]
    strategies = [
        "buy_hold",
        "kama_only",
        "tsmom_only",
        "kama_or_tsmom",
        "kama_and_tsmom",
        "or_loose",
    ]
    universes = ["btc_only", "top5", "top10", "top20"]
    params = [(5, 90, 30), (10, 60, 30), (20, 30, 50)]  # (kama, tsmom, btc_ma)

    results = []

    for exchange in exchanges:
        print(f"\n거래소: {exchange.upper()}")
        data, btc_data = load_data(exchange)

        if not data:
            print(f"  데이터 없음")
            continue

        print(f"  심볼: {len(data)}개")

        for universe in universes:
            # 유니버스 필터링
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
            elif universe == "top20":
                vols = [
                    (s, (df["close"] * df["volume"]).mean()) for s, df in data.items()
                ]
                vols.sort(key=lambda x: x[1], reverse=True)
                filtered = {s: data[s] for s, _ in vols[:20]}
            else:
                filtered = data

            for strategy in strategies:
                for kama_p, tsmom_p, btc_ma_p in params:
                    use_gate = strategy == "or_loose"

                    result = backtest(
                        filtered,
                        btc_data,
                        strategy,
                        kama_p=kama_p,
                        tsmom_p=tsmom_p,
                        btc_ma_p=btc_ma_p,
                        use_gate=use_gate,
                        max_pos=min(20, len(filtered)),
                    )

                    results.append(
                        {
                            "exchange": exchange,
                            "universe": universe,
                            "strategy": strategy,
                            "kama_p": kama_p,
                            "tsmom_p": tsmom_p,
                            "btc_ma_p": btc_ma_p,
                            "return": result["return"],
                            "sharpe": result["sharpe"],
                            "mdd": result["mdd"],
                            "trades": result["trades"],
                        }
                    )

        print(
            f'  {len([r for r in results if r["exchange"] == exchange])}개 테스트 완료'
        )

    print(f"\n\n총 {len(results)}개 테스트 완료")

    # DataFrame 생성
    df = pd.DataFrame(results)

    # 결과 저장
    output_file = Path(
        "E:/투자/Multi-Asset Strategy Platform/outputs/comprehensive_results.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"결과 저장: {output_file}")

    # 요약 출력
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)

    # 거래소별 최고 전략
    print("\n[거래소별 최고 샤프비율 전략]")
    for ex in exchanges:
        ex_df = df[df["exchange"] == ex]
        if len(ex_df) == 0:
            continue
        best = ex_df.loc[ex_df["sharpe"].idxmax()]
        print(
            f"  {ex.upper()}: {best['strategy']} ({best['universe']}) | "
            f"수익률: {best['return']*100:.1f}% | 샤프: {best['sharpe']:.2f} | MDD: {best['mdd']*100:.1f}%"
        )

    # 전략별 평균
    print("\n[전략별 평균 성과]")
    strat_summary = (
        df.groupby("strategy")
        .agg(
            {
                "return": "mean",
                "sharpe": "mean",
                "mdd": "mean",
            }
        )
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
        .agg(
            {
                "return": "mean",
                "sharpe": "mean",
                "mdd": "mean",
            }
        )
        .round(3)
    )
    univ_summary = univ_summary.sort_values("sharpe", ascending=False)
    for idx, row in univ_summary.iterrows():
        print(
            f"  {idx}: 수익률 {row['return']*100:.1f}% | 샤프 {row['sharpe']:.2f} | MDD {row['mdd']*100:.1f}%"
        )

    # 상위 10개
    print("\n[상위 10개 전략 (샤프비율 기준)]")
    top10 = df.nlargest(10, "sharpe")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(
            f"  #{i}: {row['exchange']} | {row['strategy']} | {row['universe']} | "
            f"KAMA{row['kama_p']}/TSMOM{row['tsmom_p']}/MA{row['btc_ma_p']}"
        )
        print(
            f"       수익률: {row['return']*100:.1f}% | 샤프: {row['sharpe']:.2f} | MDD: {row['mdd']*100:.1f}%"
        )

    # 하위 10개
    print("\n[하위 10개 전략 (샤프비율 기준)]")
    bottom10 = df.nsmallest(10, "sharpe")
    for i, (_, row) in enumerate(bottom10.iterrows(), 1):
        print(
            f"  #{i}: {row['exchange']} | {row['strategy']} | {row['universe']} | "
            f"KAMA{row['kama_p']}/TSMOM{row['tsmom_p']}/MA{row['btc_ma_p']}"
        )
        print(
            f"       수익률: {row['return']*100:.1f}% | 샤프: {row['sharpe']:.2f} | MDD: {row['mdd']*100:.1f}%"
        )

    print("\n완료!")
    return df


if __name__ == "__main__":
    results = main()
