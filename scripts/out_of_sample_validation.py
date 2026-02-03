"""
Out-of-Sample (OOS) 검증

검증 방법:
1. 단순 분할: 2017-2023 (훈련) vs 2024-2026 (테스트)
2. Walk-Forward: 롤링 윈도우 방식
3. 연도별 성과 분석

목적: KAMA5/TSMOM90/MA30 전략이 과적합인지 검증
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("E:/data/crypto_ohlcv")

print("=" * 70)
print("Out-of-Sample 검증")
print("전략: OR_LOOSE (KAMA5/TSMOM90/MA30)")
print("=" * 70)


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


def backtest_period(
    data, btc_data, start_date, end_date, kama_p=5, tsmom_p=90, btc_ma_p=30, max_pos=10
):
    """특정 기간에 대한 백테스트"""
    if not data or btc_data is None:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0, "days": 0}

    # 기간 필터링
    btc_filtered = btc_data[
        (btc_data.index >= start_date) & (btc_data.index <= end_date)
    ]
    if len(btc_filtered) < btc_ma_p + 10:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0, "days": 0}

    btc_prices = btc_filtered["close"].values
    btc_ma = calc_sma(btc_prices, btc_ma_p)
    btc_gate = pd.Series(btc_prices > btc_ma, index=btc_filtered.index)

    signal_data = {}
    for symbol, df in data.items():
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(df_filtered) < max(kama_p, tsmom_p, 100):
            continue

        prices = df_filtered["close"].values
        kama = calc_kama(prices, kama_p)
        signal = (prices > kama) | calc_tsmom(prices, tsmom_p)

        aligned = btc_gate.reindex(df_filtered.index).fillna(False)
        signal = signal & aligned.values

        df_filtered = df_filtered.copy()
        df_filtered["signal"] = signal
        df_filtered["dvol"] = df_filtered["close"] * df_filtered["volume"]
        signal_data[symbol] = df_filtered

    if not signal_data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0, "days": 0}

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

        exits = set(positions.keys()) - targets
        new_entries = targets - set(positions.keys())

        for sym in exits:
            if sym in positions and sym in prices_today:
                shares, _ = positions[sym]
                cash += shares * prices_today[sym] * 0.998
                del positions[sym]
                trades += 1

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
    mdd = np.min((vals - peak) / peak)

    return {
        "return": total_ret,
        "sharpe": sharpe,
        "mdd": mdd,
        "trades": trades,
        "days": len(all_dates),
    }


def main():
    # 데이터 로드
    print("\n데이터 로드 중...")
    data = load_ohlcv("upbit", 100)
    print(f"Upbit 심볼: {len(data)}개")

    # Top20 유니버스
    vols = [(s, (df["close"] * df["volume"]).mean()) for s, df in data.items()]
    vols.sort(key=lambda x: x[1], reverse=True)
    filtered = {s: data[s] for s, _ in vols[:20]}
    print(f"Top20 유니버스: {list(filtered.keys())[:5]}...")

    btc_data = None
    for k, v in data.items():
        if "BTC" in k.upper():
            btc_data = v
            break

    # ============================================================
    # 1. 단순 분할 검증 (In-Sample vs Out-of-Sample)
    # ============================================================
    print("\n" + "=" * 70)
    print("1. 단순 분할 검증")
    print("   In-Sample: 2017-01-01 ~ 2023-12-31 (7년)")
    print("   Out-of-Sample: 2024-01-01 ~ 2026-01-24 (2년)")
    print("=" * 70)

    is_result = backtest_period(
        filtered, btc_data, pd.Timestamp("2017-01-01"), pd.Timestamp("2023-12-31")
    )

    oos_result = backtest_period(
        filtered, btc_data, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31")
    )

    print("\n[In-Sample 결과 (2017-2023)]")
    print(f'  수익률: {is_result["return"]*100:.1f}%')
    print(f'  샤프비율: {is_result["sharpe"]:.2f}')
    print(f'  MDD: {is_result["mdd"]*100:.1f}%')
    print(f'  거래일: {is_result["days"]}일')

    print("\n[Out-of-Sample 결과 (2024-2026)]")
    print(f'  수익률: {oos_result["return"]*100:.1f}%')
    print(f'  샤프비율: {oos_result["sharpe"]:.2f}')
    print(f'  MDD: {oos_result["mdd"]*100:.1f}%')
    print(f'  거래일: {oos_result["days"]}일')

    # OOS 효율성
    if is_result["sharpe"] > 0:
        oos_efficiency = oos_result["sharpe"] / is_result["sharpe"]
        print("\n[OOS 효율성]")
        print(f"  OOS Sharpe / IS Sharpe = {oos_efficiency:.2f}")
        if oos_efficiency >= 0.5:
            print("  -> [PASS] 양호 (0.5 이상이면 과적합 아님)")
        else:
            print("  -> [WARN] 주의 필요 (0.5 미만)")

    # ============================================================
    # 2. 연도별 성과 분석
    # ============================================================
    print("\n" + "=" * 70)
    print("2. 연도별 성과 분석")
    print("=" * 70)

    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    yearly_results = []

    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")

        result = backtest_period(filtered, btc_data, start, end)

        if result["days"] > 30:  # 최소 30일 이상
            yearly_results.append(
                {
                    "year": year,
                    "return": result["return"] * 100,
                    "sharpe": result["sharpe"],
                    "mdd": result["mdd"] * 100,
                    "days": result["days"],
                }
            )

    print(f'\n{"연도":<6} {"수익률":>10} {"샤프":>8} {"MDD":>10} {"거래일":>8}')
    print("-" * 50)
    for r in yearly_results:
        print(
            f'{r["year"]:<6} {r["return"]:>9.1f}% {r["sharpe"]:>8.2f} {r["mdd"]:>9.1f}% {r["days"]:>8}'
        )

    # 통계
    df_yearly = pd.DataFrame(yearly_results)
    print("-" * 50)
    print(
        f'{"평균":<6} {df_yearly["return"].mean():>9.1f}% {df_yearly["sharpe"].mean():>8.2f} {df_yearly["mdd"].mean():>9.1f}%'
    )
    print(
        f'{"표준편차":<6} {df_yearly["return"].std():>9.1f}% {df_yearly["sharpe"].std():>8.2f}'
    )

    # 양수 연도 비율
    positive_years = len(df_yearly[df_yearly["sharpe"] > 0])
    print(
        f"\n양수 샤프 연도: {positive_years}/{len(df_yearly)} ({positive_years/len(df_yearly)*100:.0f}%)"
    )

    # ============================================================
    # 3. Walk-Forward 분석
    # ============================================================
    print("\n" + "=" * 70)
    print("3. Walk-Forward 분석")
    print("   훈련: 2년 / 테스트: 1년 / 스텝: 1년")
    print("=" * 70)

    wf_results = []

    # 2019년부터 테스트 (2017-2018 훈련 필요)
    test_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    for test_year in test_years:
        train_start = pd.Timestamp(f"{test_year-2}-01-01")
        train_end = pd.Timestamp(f"{test_year-1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")

        train_result = backtest_period(filtered, btc_data, train_start, train_end)
        test_result = backtest_period(filtered, btc_data, test_start, test_end)

        if train_result["days"] > 100 and test_result["days"] > 30:
            wf_results.append(
                {
                    "test_year": test_year,
                    "train_sharpe": train_result["sharpe"],
                    "test_sharpe": test_result["sharpe"],
                    "train_return": train_result["return"] * 100,
                    "test_return": test_result["return"] * 100,
                }
            )

    print(
        f'\n{"테스트연도":<10} {"훈련샤프":>10} {"테스트샤프":>10} {"훈련수익률":>12} {"테스트수익률":>12}'
    )
    print("-" * 60)
    for r in wf_results:
        print(
            f'{r["test_year"]:<10} {r["train_sharpe"]:>10.2f} {r["test_sharpe"]:>10.2f} {r["train_return"]:>11.1f}% {r["test_return"]:>11.1f}%'
        )

    # Walk-Forward 효율성
    df_wf = pd.DataFrame(wf_results)
    if len(df_wf) > 0:
        avg_train = df_wf["train_sharpe"].mean()
        avg_test = df_wf["test_sharpe"].mean()
        wf_efficiency = avg_test / avg_train if avg_train > 0 else 0

        print("-" * 60)
        print(f'{"평균":<10} {avg_train:>10.2f} {avg_test:>10.2f}')
        print(f"\nWalk-Forward 효율성: {wf_efficiency:.2f}")
        if wf_efficiency >= 0.5:
            print("→ [PASS] 양호 (0.5 이상)")
        else:
            print("→ [WARN] 과적합 의심")

    # ============================================================
    # 4. Buy & Hold 비교
    # ============================================================
    print("\n" + "=" * 70)
    print("4. Buy & Hold (BTC) 대비 성과")
    print("=" * 70)

    # BTC Buy & Hold 계산
    if btc_data is not None:
        btc_2024 = btc_data[btc_data.index >= "2024-01-01"]
        if len(btc_2024) > 0:
            btc_return = (
                btc_2024["close"].iloc[-1] / btc_2024["close"].iloc[0] - 1
            ) * 100
            print("\n[2024년 이후]")
            print(f"  BTC Buy & Hold: {btc_return:.1f}%")
            print(f'  OR_LOOSE 전략: {oos_result["return"]*100:.1f}%')
            print(f'  초과 수익: {oos_result["return"]*100 - btc_return:.1f}%')

    # ============================================================
    # 최종 판정
    # ============================================================
    print("\n" + "=" * 70)
    print("최종 판정")
    print("=" * 70)

    # 판정 기준
    checks = []

    # 1. OOS 효율성
    if is_result["sharpe"] > 0 and oos_result["sharpe"] / is_result["sharpe"] >= 0.5:
        checks.append(("OOS 효율성 >= 0.5", True))
    else:
        checks.append(("OOS 효율성 >= 0.5", False))

    # 2. 양수 연도 비율
    if positive_years / len(df_yearly) >= 0.7:
        checks.append(("양수 샤프 연도 >= 70%", True))
    else:
        checks.append(("양수 샤프 연도 >= 70%", False))

    # 3. Walk-Forward 효율성
    if len(df_wf) > 0 and wf_efficiency >= 0.5:
        checks.append(("Walk-Forward 효율성 >= 0.5", True))
    else:
        checks.append(("Walk-Forward 효율성 >= 0.5", False))

    # 4. OOS 샤프 > 0
    if oos_result["sharpe"] > 0:
        checks.append(("OOS 샤프 > 0", True))
    else:
        checks.append(("OOS 샤프 > 0", False))

    print()
    passed = 0
    for check, result in checks:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"  {status}: {check}")
        if result:
            passed += 1

    print(f"\n결과: {passed}/{len(checks)} 통과")

    if passed >= 3:
        print("\n[OK] 전략 유효성 확인됨 - 과적합 아님")
    elif passed >= 2:
        print("\n[CAUTION] 부분적 유효 - 주의 필요")
    else:
        print("\n[ALERT] 과적합 의심 - 전략 재검토 필요")

    return {
        "is_result": is_result,
        "oos_result": oos_result,
        "yearly": yearly_results,
        "wf": wf_results,
    }


if __name__ == "__main__":
    results = main()
