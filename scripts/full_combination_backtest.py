"""
전체 조합 백테스트 - 실제 매매 가정

조합 요소:
1. 기본 전략: OR_LOOSE (KAMA OR TSMOM) AND BTC_GATE
2. 파생상품 필터: Funding Rate, Open Interest, Long/Short Ratio
3. 심리 필터: Fear & Greed Index
4. 거시 필터: VIX, DXY
5. 시간프레임: 1d, 4h

실제 매매 조건:
- 슬리피지: 0.5%
- 수수료: 0.1%
- Day T 시그널 → Day T+1 실행 (바이어스 프리)
"""

import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")

print("=" * 80)
print("전체 조합 백테스트 - 실제 매매 가정")
print(f'시작 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print("=" * 80)


# ============================================================
# 지표 계산 함수들
# ============================================================
def calc_kama(prices, period=10, fast=2, slow=30):
    """KAMA (Kaufman Adaptive Moving Average)"""
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
    """Simple Moving Average"""
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


def calc_tsmom(prices, period=60):
    """Time-Series Momentum"""
    n = len(prices)
    signal = np.zeros(n, dtype=bool)
    for i in range(period, n):
        signal[i] = prices[i] > prices[i - period]
    return signal


def calc_rsi(prices, period=14):
    """RSI (Relative Strength Index)"""
    n = len(prices)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# 데이터 로드 함수들
# ============================================================
def load_ohlcv(
    exchange: str, timeframe: str = "1d", min_days: int = 100
) -> Dict[str, pd.DataFrame]:
    """OHLCV 데이터 로드"""
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
                symbol = f.stem
                data[symbol] = df
        except:
            continue

    return data


def load_funding_rate() -> Dict[str, pd.DataFrame]:
    """Funding Rate 데이터 로드"""
    folder = DATA_ROOT / "binance_funding_rate"
    if not folder.exists():
        return {}

    data = {}
    for f in folder.glob("*_funding.csv"):
        try:
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
            # 일별로 평균
            df = df.groupby("date")["fundingRate"].mean().reset_index()
            df = df.set_index("date").sort_index()
            symbol = f.stem.replace("_funding", "")
            data[symbol] = df
        except:
            continue

    return data


def load_open_interest() -> Dict[str, pd.DataFrame]:
    """Open Interest 데이터 로드"""
    folder = DATA_ROOT / "binance_open_interest"
    if not folder.exists():
        return {}

    data = {}
    for f in folder.glob("*_oi.csv"):
        try:
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
            df = df.set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            symbol = f.stem.replace("_oi", "")
            data[symbol] = df
        except:
            continue

    return data


def load_long_short_ratio() -> Dict[str, pd.DataFrame]:
    """Long/Short Ratio 데이터 로드"""
    folder = DATA_ROOT / "binance_long_short_ratio"
    if not folder.exists():
        return {}

    data = {}
    for f in folder.glob("*_lsratio.csv"):
        try:
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
            df = df.set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            symbol = f.stem.replace("_lsratio", "")
            data[symbol] = df
        except:
            continue

    return data


def load_fear_greed() -> Optional[pd.DataFrame]:
    """Fear & Greed Index 로드"""
    path = DATA_ROOT / "sentiment" / "fear_greed_index.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df = df.set_index("date").sort_index()
        return df
    except:
        return None


def load_macro(name: str) -> Optional[pd.DataFrame]:
    """거시경제 데이터 로드"""
    path = DATA_ROOT / "macro" / f"{name}.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df = df.set_index("date").sort_index()
        return df
    except:
        return None


# ============================================================
# 필터 함수들
# ============================================================
def apply_funding_filter(
    signal: np.ndarray,
    dates: pd.DatetimeIndex,
    funding_data: pd.DataFrame,
    threshold: float = 0.0005,
) -> np.ndarray:
    """
    Funding Rate 필터
    - FR > threshold: 롱 과열 → 롱 시그널 제거
    - FR < -threshold: 숏 과열 → 숏에 유리하지만 여기선 롱만
    """
    if funding_data is None or len(funding_data) == 0:
        return signal

    result = signal.copy()
    for i, date in enumerate(dates):
        if date in funding_data.index:
            fr = funding_data.loc[date, "fundingRate"]
            if fr > threshold:  # 롱 과열
                result[i] = False

    return result


def apply_oi_filter(
    signal: np.ndarray,
    dates: pd.DatetimeIndex,
    oi_data: pd.DataFrame,
    lookback: int = 7,
) -> np.ndarray:
    """
    Open Interest 변화율 필터
    - OI 급증 + 가격 상승: 추세 지속 (유지)
    - OI 급감: 추세 약화 가능 (제거)
    """
    if oi_data is None or len(oi_data) < lookback:
        return signal

    result = signal.copy()
    oi_values = oi_data["sumOpenInterestValue"].values
    oi_dates = oi_data.index

    # OI 변화율 계산
    oi_change = np.zeros(len(oi_values))
    for i in range(lookback, len(oi_values)):
        if oi_values[i - lookback] > 0:
            oi_change[i] = (oi_values[i] - oi_values[i - lookback]) / oi_values[
                i - lookback
            ]

    oi_change_series = pd.Series(oi_change, index=oi_dates)

    for i, date in enumerate(dates):
        if date in oi_change_series.index:
            change = oi_change_series.loc[date]
            if change < -0.2:  # OI 20% 이상 급감
                result[i] = False

    return result


def apply_lsratio_filter(
    signal: np.ndarray,
    dates: pd.DatetimeIndex,
    ls_data: pd.DataFrame,
    threshold: float = 2.0,
) -> np.ndarray:
    """
    Long/Short Ratio 필터
    - LSR > threshold: 롱 과밀 → 롱 시그널 제거 (역추세)
    """
    if ls_data is None or len(ls_data) == 0:
        return signal

    result = signal.copy()
    for i, date in enumerate(dates):
        if date in ls_data.index:
            lsr = ls_data.loc[date, "longShortRatio"]
            if lsr > threshold:  # 롱 과밀
                result[i] = False

    return result


def apply_fear_greed_filter(
    signal: np.ndarray,
    dates: pd.DatetimeIndex,
    fg_data: pd.DataFrame,
    extreme_fear: int = 20,
    extreme_greed: int = 80,
) -> np.ndarray:
    """
    Fear & Greed Index 필터
    - 극단적 공포(< extreme_fear): 매수 기회 (유지)
    - 극단적 탐욕(> extreme_greed): 과열 (제거)
    """
    if fg_data is None or len(fg_data) == 0:
        return signal

    result = signal.copy()
    for i, date in enumerate(dates):
        if date in fg_data.index:
            fg = fg_data.loc[date, "fear_greed_value"]
            if fg > extreme_greed:  # 극단적 탐욕
                result[i] = False

    return result


def apply_vix_filter(
    signal: np.ndarray,
    dates: pd.DatetimeIndex,
    vix_data: pd.DataFrame,
    threshold: float = 30,
) -> np.ndarray:
    """
    VIX 필터
    - VIX > threshold: 공포 장세 → 위험자산 회피 (제거)
    """
    if vix_data is None or len(vix_data) == 0:
        return signal

    result = signal.copy()
    for i, date in enumerate(dates):
        if date in vix_data.index:
            vix = vix_data.loc[date, "close"]
            if vix > threshold:
                result[i] = False

    return result


def apply_dxy_filter(
    signal: np.ndarray,
    dates: pd.DatetimeIndex,
    dxy_data: pd.DataFrame,
    ma_period: int = 20,
) -> np.ndarray:
    """
    DXY (달러 인덱스) 필터
    - DXY > MA: 달러 강세 → 암호화폐 약세 (제거)
    """
    if dxy_data is None or len(dxy_data) < ma_period:
        return signal

    dxy_close = dxy_data["close"].values
    dxy_ma = calc_sma(dxy_close, ma_period)
    dxy_signal = pd.Series(dxy_close > dxy_ma, index=dxy_data.index)

    result = signal.copy()
    for i, date in enumerate(dates):
        if date in dxy_signal.index:
            if dxy_signal.loc[date]:  # 달러 강세
                result[i] = False

    return result


# ============================================================
# 백테스트 엔진
# ============================================================
def backtest(
    data: Dict[str, pd.DataFrame],
    btc_data: pd.DataFrame,
    filters: Dict[str, any],
    kama_p: int = 10,
    tsmom_p: int = 60,
    btc_ma_p: int = 30,
    max_pos: int = 10,
    slippage: float = 0.005,
    commission: float = 0.001,
) -> Dict:
    """
    바이어스 프리 백테스트

    Args:
        data: OHLCV 데이터
        btc_data: BTC 데이터 (BTC Gate용)
        filters: 적용할 필터들
        kama_p: KAMA 기간
        tsmom_p: TSMOM 기간
        btc_ma_p: BTC MA 기간
        max_pos: 최대 포지션 수
        slippage: 슬리피지
        commission: 수수료
    """
    if not data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0, "win_rate": 0}

    # BTC Gate 계산
    btc_gate = None
    if btc_data is not None:
        btc_prices = btc_data["close"].values
        btc_ma = calc_sma(btc_prices, btc_ma_p)
        btc_gate = pd.Series(btc_prices > btc_ma, index=btc_data.index)

    # 시그널 계산
    signal_data = {}
    for symbol, df in data.items():
        if len(df) < max(kama_p, tsmom_p, 100):
            continue

        prices = df["close"].values
        dates = df.index

        # 기본 시그널: KAMA OR TSMOM (OR_LOOSE 기반)
        kama = calc_kama(prices, kama_p)
        kama_signal = prices > kama
        tsmom_signal = calc_tsmom(prices, tsmom_p)
        signal = kama_signal | tsmom_signal

        # BTC Gate 적용
        if btc_gate is not None:
            aligned = btc_gate.reindex(dates).fillna(False)
            signal = signal & aligned.values

        # 추가 필터 적용
        if filters.get("funding") and symbol in filters.get("funding_data", {}):
            signal = apply_funding_filter(
                signal,
                dates,
                filters["funding_data"].get(symbol),
                filters.get("funding_threshold", 0.0005),
            )

        if filters.get("oi") and symbol in filters.get("oi_data", {}):
            signal = apply_oi_filter(signal, dates, filters["oi_data"].get(symbol))

        if filters.get("lsratio") and symbol in filters.get("lsratio_data", {}):
            signal = apply_lsratio_filter(
                signal,
                dates,
                filters["lsratio_data"].get(symbol),
                filters.get("lsratio_threshold", 2.0),
            )

        if filters.get("fear_greed") and filters.get("fg_data") is not None:
            signal = apply_fear_greed_filter(
                signal,
                dates,
                filters["fg_data"],
                filters.get("fg_fear", 20),
                filters.get("fg_greed", 80),
            )

        if filters.get("vix") and filters.get("vix_data") is not None:
            signal = apply_vix_filter(
                signal, dates, filters["vix_data"], filters.get("vix_threshold", 30)
            )

        if filters.get("dxy") and filters.get("dxy_data") is not None:
            signal = apply_dxy_filter(
                signal, dates, filters["dxy_data"], filters.get("dxy_ma", 20)
            )

        df = df.copy()
        df["signal"] = signal
        df["dvol"] = df["close"] * df["volume"]
        signal_data[symbol] = df

    if not signal_data:
        return {"return": 0, "sharpe": 0, "mdd": 0, "trades": 0, "win_rate": 0}

    # 시뮬레이션
    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))

    capital = 10000.0
    cash = capital
    positions = {}
    values = [capital]
    returns = []
    trades = 0
    winning_trades = 0

    cost_factor = 1 + slippage + commission
    sell_factor = 1 - slippage - commission

    for i, date in enumerate(all_dates):
        prices_today = {}
        signals_today = {}
        vols_today = {}

        for sym, df in signal_data.items():
            if date in df.index:
                prices_today[sym] = df.loc[date, "close"]
                signals_today[sym] = df.loc[date, "signal"]
                vols_today[sym] = df.loc[date, "dvol"]

        # 포트폴리오 가치 계산
        pos_value = sum(
            shares * prices_today.get(sym, cost)
            for sym, (shares, cost) in positions.items()
        )
        port_value = cash + pos_value

        if i > 0:
            ret = (port_value - values[-1]) / values[-1] if values[-1] > 0 else 0
            returns.append(ret)

        values.append(port_value)

        # 타겟 포지션 결정
        active = [(s, vols_today.get(s, 0)) for s, sig in signals_today.items() if sig]
        active.sort(key=lambda x: x[1], reverse=True)
        targets = set(s for s, _ in active[:max_pos])

        current_syms = set(positions.keys())
        exits = current_syms - targets
        new_entries = targets - current_syms

        # 매도
        for sym in exits:
            if sym in positions and sym in prices_today:
                shares, entry_price = positions[sym]
                sell_price = prices_today[sym] * sell_factor
                proceeds = shares * sell_price
                cash += proceeds

                # 승률 계산
                if sell_price > entry_price:
                    winning_trades += 1

                del positions[sym]
                trades += 1

        # 매수
        if targets:
            curr_val = cash + sum(
                s * prices_today.get(sym, 0) for sym, (s, _) in positions.items()
            )
            per_pos = curr_val / len(targets) if targets else 0

            for sym in new_entries:
                if sym in prices_today:
                    buy_price = prices_today[sym] * cost_factor
                    cost = per_pos
                    if cost <= cash and buy_price > 0:
                        shares = cost / buy_price
                        cash -= cost
                        positions[sym] = (shares, buy_price)
                        trades += 1

        prices_today.copy()

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

    win_rate = (
        winning_trades / (trades / 2) if trades > 0 else 0
    )  # trades/2 = 매도 횟수

    return {
        "return": total_ret,
        "sharpe": sharpe,
        "mdd": mdd,
        "trades": trades,
        "win_rate": win_rate,
        "final_value": final,
    }


# ============================================================
# 메인 실행
# ============================================================
def main():
    # 데이터 로드
    print("\n데이터 로드 중...")

    # OHLCV
    ohlcv_futures = load_ohlcv("binance_futures", "1d", min_days=100)
    print(f"  Binance Futures 1d: {len(ohlcv_futures)}개 심볼")

    # BTC 데이터
    btc_data = ohlcv_futures.get("BTCUSDT")
    if btc_data is None:
        for k, v in ohlcv_futures.items():
            if "BTC" in k.upper():
                btc_data = v
                break

    # 파생상품 데이터
    funding_data = load_funding_rate()
    print(f"  Funding Rate: {len(funding_data)}개 심볼")

    oi_data = load_open_interest()
    print(f"  Open Interest: {len(oi_data)}개 심볼")

    lsratio_data = load_long_short_ratio()
    print(f"  Long/Short Ratio: {len(lsratio_data)}개 심볼")

    # 심리/거시 데이터
    fg_data = load_fear_greed()
    print(f'  Fear & Greed: {"로드됨" if fg_data is not None else "없음"}')

    vix_data = load_macro("VIX")
    print(f'  VIX: {"로드됨" if vix_data is not None else "없음"}')

    dxy_data = load_macro("DXY")
    print(f'  DXY: {"로드됨" if dxy_data is not None else "없음"}')

    # 유니버스 정의
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
        elif universe_type == "top20":
            vols = [(s, (df["close"] * df["volume"]).mean()) for s, df in data.items()]
            vols.sort(key=lambda x: x[1], reverse=True)
            return {s: data[s] for s, _ in vols[:20]}
        return data

    # 필터 조합 정의
    filter_options = {
        "funding": [False, True],
        "oi": [False, True],
        "lsratio": [False, True],
        "fear_greed": [False, True],
        "vix": [False, True],
        "dxy": [False, True],
    }

    # 파라미터 조합
    param_sets = [
        {"kama_p": 5, "tsmom_p": 90, "btc_ma_p": 30},
        {"kama_p": 10, "tsmom_p": 60, "btc_ma_p": 30},
        {"kama_p": 20, "tsmom_p": 30, "btc_ma_p": 50},
    ]

    universes = ["btc_only", "top5", "top10", "top20"]

    # 모든 필터 조합 생성
    filter_combinations = []
    filter_names = list(filter_options.keys())

    for values in product(*filter_options.values()):
        combo = dict(zip(filter_names, values))
        filter_combinations.append(combo)

    print(f"\n총 필터 조합: {len(filter_combinations)}개")
    print(f"총 파라미터 세트: {len(param_sets)}개")
    print(f"총 유니버스: {len(universes)}개")
    print(
        f"총 테스트 수: {len(filter_combinations) * len(param_sets) * len(universes)}개"
    )

    # 백테스트 실행
    print("\n백테스트 실행 중...")
    results = []
    total_tests = len(filter_combinations) * len(param_sets) * len(universes)
    test_count = 0

    for universe in universes:
        universe_data = get_universe(ohlcv_futures, universe)
        if not universe_data:
            continue

        for params in param_sets:
            for filter_combo in filter_combinations:
                test_count += 1

                if test_count % 50 == 0:
                    print(
                        f"  진행: {test_count}/{total_tests} ({test_count/total_tests*100:.1f}%)"
                    )

                # 필터 설정
                filters = {
                    **filter_combo,
                    "funding_data": funding_data,
                    "funding_threshold": 0.0005,
                    "oi_data": oi_data,
                    "lsratio_data": lsratio_data,
                    "lsratio_threshold": 2.0,
                    "fg_data": fg_data,
                    "fg_fear": 20,
                    "fg_greed": 80,
                    "vix_data": vix_data,
                    "vix_threshold": 30,
                    "dxy_data": dxy_data,
                    "dxy_ma": 20,
                }

                # 백테스트
                result = backtest(
                    universe_data,
                    btc_data,
                    filters,
                    kama_p=params["kama_p"],
                    tsmom_p=params["tsmom_p"],
                    btc_ma_p=params["btc_ma_p"],
                    max_pos=min(10, len(universe_data)),
                )

                # 결과 저장
                filter_str = (
                    "+".join([k for k, v in filter_combo.items() if v]) or "base"
                )

                results.append(
                    {
                        "universe": universe,
                        "kama_p": params["kama_p"],
                        "tsmom_p": params["tsmom_p"],
                        "btc_ma_p": params["btc_ma_p"],
                        "filters": filter_str,
                        "funding": filter_combo["funding"],
                        "oi": filter_combo["oi"],
                        "lsratio": filter_combo["lsratio"],
                        "fear_greed": filter_combo["fear_greed"],
                        "vix": filter_combo["vix"],
                        "dxy": filter_combo["dxy"],
                        "return": result["return"],
                        "sharpe": result["sharpe"],
                        "mdd": result["mdd"],
                        "trades": result["trades"],
                        "win_rate": result["win_rate"],
                    }
                )

    # 결과 저장
    print("\n결과 저장 중...")
    df = pd.DataFrame(results)

    output_dir = Path("E:/투자/Multi-Asset Strategy Platform/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "full_combination_results.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_file}")

    # 결과 분석
    print("\n" + "=" * 80)
    print("결과 분석")
    print("=" * 80)

    # 상위 20개
    print("\n[상위 20개 전략 (샤프비율 기준)]")
    top20 = df.nlargest(20, "sharpe")
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        print(
            f"  #{i}: {row['universe']} | KAMA{row['kama_p']}/TSMOM{row['tsmom_p']} | 필터: {row['filters']}"
        )
        print(
            f"       수익률: {row['return']*100:.1f}% | 샤프: {row['sharpe']:.2f} | MDD: {row['mdd']*100:.1f}% | 승률: {row['win_rate']*100:.1f}%"
        )

    # 필터별 평균
    print("\n[필터별 평균 샤프비율]")
    for filter_name in filter_names:
        on_avg = df[df[filter_name]]["sharpe"].mean()
        off_avg = df[not df[filter_name]]["sharpe"].mean()
        diff = on_avg - off_avg
        print(f"  {filter_name}: ON={on_avg:.3f}, OFF={off_avg:.3f}, 차이={diff:+.3f}")

    # 유니버스별 평균
    print("\n[유니버스별 평균 성과]")
    univ_summary = (
        df.groupby("universe")
        .agg(
            {
                "return": "mean",
                "sharpe": "mean",
                "mdd": "mean",
                "win_rate": "mean",
            }
        )
        .round(3)
    )
    for idx, row in univ_summary.iterrows():
        print(
            f"  {idx}: 수익률 {row['return']*100:.1f}% | 샤프 {row['sharpe']:.2f} | MDD {row['mdd']*100:.1f}% | 승률 {row['win_rate']*100:.1f}%"
        )

    # 최고 필터 조합
    print("\n[최고 필터 조합 (유니버스별)]")
    for univ in universes:
        univ_df = df[df["universe"] == univ]
        if len(univ_df) > 0:
            best = univ_df.loc[univ_df["sharpe"].idxmax()]
            print(
                f"  {univ}: {best['filters']} | 샤프 {best['sharpe']:.2f} | 수익률 {best['return']*100:.1f}%"
            )

    print("\n완료!")
    print(f"총 테스트: {len(results)}개")
    print(f"결과 파일: {output_file}")

    return df


if __name__ == "__main__":
    results = main()
