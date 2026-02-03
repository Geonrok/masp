"""
종합 백테스트 - 모든 거래소, 모든 전략, 모든 조합

테스트 범위:
1. 거래소: Upbit, Bithumb, Binance Spot, Binance Futures
2. 전략: Buy&Hold, KAMA, TSMOM, KAMA+TSMOM, OR_LOOSE (with BTC Gate)
3. 유니버스: BTC Only, Top 5, Top 10, Top 20, All
4. 필터: ADV Filter, BTC Gate, Volume Filter
5. 파라미터: KAMA(5,10,20), TSMOM(30,60,90), BTC_MA(20,30,50)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from itertools import product
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("E:/data/crypto_ohlcv")

# ============================================================
# 지표 계산 함수들
# ============================================================


def calc_kama(
    prices: np.ndarray, period: int = 5, fast: int = 2, slow: int = 30
) -> np.ndarray:
    """Kaufman Adaptive Moving Average"""
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


def calc_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average"""
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


def calc_tsmom(prices: np.ndarray, period: int = 90) -> np.ndarray:
    """Time-Series Momentum: 현재가 > period일 전 가격"""
    n = len(prices)
    signal = np.zeros(n, dtype=bool)
    for i in range(period, n):
        signal[i] = prices[i] > prices[i - period]
    return signal


def calc_adv(volumes: np.ndarray, prices: np.ndarray, period: int = 20) -> np.ndarray:
    """Average Daily Volume (in currency value)"""
    dollar_volume = volumes * prices
    adv = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        adv[i] = np.mean(dollar_volume[i - period + 1 : i + 1])
    return adv


# ============================================================
# 전략 시그널 생성
# ============================================================


@dataclass
class StrategyConfig:
    """전략 설정"""

    name: str = "or_loose"
    kama_period: int = 5
    tsmom_period: int = 90
    btc_ma_period: int = 30
    use_btc_gate: bool = True
    use_kama: bool = True
    use_tsmom: bool = True
    adv_threshold: float = 0  # 0이면 필터 비활성화
    max_positions: int = 20
    slippage_pct: float = 0.005
    commission_pct: float = 0.001


def generate_signals(
    df: pd.DataFrame, config: StrategyConfig, btc_gate: Optional[pd.Series] = None
) -> pd.Series:
    """전략별 시그널 생성"""
    prices = df["close"].values
    n = len(prices)

    if config.name == "buy_hold":
        return pd.Series(True, index=df.index)

    # KAMA 시그널
    kama_signal = np.ones(n, dtype=bool)
    if config.use_kama:
        kama = calc_kama(prices, config.kama_period)
        kama_signal = prices > kama

    # TSMOM 시그널
    tsmom_signal = np.ones(n, dtype=bool)
    if config.use_tsmom:
        tsmom_signal = calc_tsmom(prices, config.tsmom_period)

    # 전략별 조합
    if config.name == "kama_only":
        entry_signal = kama_signal
    elif config.name == "tsmom_only":
        entry_signal = tsmom_signal
    elif config.name == "kama_and_tsmom":  # AND 조합
        entry_signal = kama_signal & tsmom_signal
    elif config.name in ["kama_or_tsmom", "or_loose"]:  # OR 조합
        entry_signal = kama_signal | tsmom_signal
    else:
        entry_signal = kama_signal | tsmom_signal

    # BTC Gate 적용
    if config.use_btc_gate and btc_gate is not None:
        # btc_gate를 df.index에 맞게 정렬
        aligned_gate = btc_gate.reindex(df.index).fillna(False)
        entry_signal = entry_signal & aligned_gate.values

    return pd.Series(entry_signal, index=df.index)


# ============================================================
# 백테스터
# ============================================================


@dataclass
class BacktestResult:
    """백테스트 결과"""

    strategy: str
    exchange: str
    universe: str
    params: Dict[str, Any]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    invested_days: int
    total_days: int
    final_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "exchange": self.exchange,
            "universe": self.universe,
            "params": str(self.params),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "invested_days": self.invested_days,
            "total_days": self.total_days,
            "final_value": self.final_value,
        }


def run_backtest(
    data: Dict[str, pd.DataFrame],
    config: StrategyConfig,
    initial_capital: float = 10000.0,
    btc_data: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    바이어스 제거 백테스트 실행

    Day T signal -> Day T+1 execution
    """
    if not data:
        return BacktestResult(
            strategy=config.name,
            exchange="",
            universe="",
            params={},
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            total_trades=0,
            invested_days=0,
            total_days=0,
            final_value=initial_capital,
        )

    # BTC Gate 계산
    btc_gate = None
    if config.use_btc_gate and btc_data is not None:
        btc_prices = btc_data["close"].values
        btc_ma = calc_sma(btc_prices, config.btc_ma_period)
        btc_gate = pd.Series(btc_prices > btc_ma, index=btc_data.index)

    # 각 심볼에 대해 시그널 생성
    signal_data = {}
    for symbol, df in data.items():
        if len(df) < max(config.kama_period, config.tsmom_period, 100):
            continue

        # ADV 필터
        if config.adv_threshold > 0:
            adv = calc_adv(df["volume"].values, df["close"].values)
            if np.nanmean(adv) < config.adv_threshold:
                continue

        signal = generate_signals(df, config, btc_gate)
        df = df.copy()
        df["signal"] = signal
        df["dollar_volume"] = df["close"] * df["volume"]
        signal_data[symbol] = df

    if not signal_data:
        return BacktestResult(
            strategy=config.name,
            exchange="",
            universe="",
            params={},
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            total_trades=0,
            invested_days=0,
            total_days=0,
            final_value=initial_capital,
        )

    # 모든 날짜 수집
    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))

    # 시뮬레이션
    cash = initial_capital
    positions = {}  # symbol -> (shares, cost_basis)
    prev_prices = {}

    portfolio_values = [initial_capital]
    daily_returns = []
    trade_count = 0
    invested_days = 0

    for i, date in enumerate(all_dates):
        # 오늘 데이터 수집
        prices_today = {}
        signals_today = {}
        volumes_today = {}

        for symbol, df in signal_data.items():
            if date in df.index:
                prices_today[symbol] = df.loc[date, "close"]
                signals_today[symbol] = df.loc[date, "signal"]
                volumes_today[symbol] = df.loc[date, "dollar_volume"]

        # 포트폴리오 가치 계산
        position_value = sum(
            shares * prices_today.get(sym, prev_prices.get(sym, 0))
            for sym, (shares, _) in positions.items()
        )
        portfolio_value = cash + position_value

        # 일간 수익률
        if i > 0:
            prev_value = portfolio_values[-1]
            daily_ret = (
                (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            )
            daily_returns.append(daily_ret)
            if positions:
                invested_days += 1

        portfolio_values.append(portfolio_value)

        # 타겟 포지션 결정 (Day T 시그널 -> Day T+1 실행)
        active = [
            (sym, volumes_today.get(sym, 0))
            for sym, sig in signals_today.items()
            if sig
        ]
        active_sorted = sorted(active, key=lambda x: x[1], reverse=True)
        target_symbols = set(s for s, _ in active_sorted[: config.max_positions])

        # 매도 (타겟에 없는 포지션)
        for sym in list(positions.keys()):
            if sym not in target_symbols and sym in prices_today:
                shares, _ = positions[sym]
                sell_price = prices_today[sym] * (1 - config.slippage_pct)
                proceeds = shares * sell_price * (1 - config.commission_pct)
                cash += proceeds
                del positions[sym]
                trade_count += 1

        # 매수 (새 포지션)
        if target_symbols:
            current_value = cash + sum(
                s * prices_today.get(sym, 0) for sym, (s, _) in positions.items()
            )
            target_per_position = current_value / len(target_symbols)

            for sym in target_symbols:
                if sym not in positions and sym in prices_today:
                    buy_price = prices_today[sym] * (1 + config.slippage_pct)
                    cost = target_per_position
                    shares = cost / buy_price * (1 - config.commission_pct)

                    if cost <= cash:
                        cash -= cost
                        positions[sym] = (shares, buy_price)
                        trade_count += 1

        prev_prices = prices_today.copy()

    # 최종 결과 계산
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital

    n_years = len(all_dates) / 252 if len(all_dates) > 0 else 1
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    daily_rets = np.array(daily_returns)
    sharpe = (
        np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        if len(daily_rets) > 1 and np.std(daily_rets) > 0
        else 0
    )

    portfolio_arr = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_arr)
    drawdown = (portfolio_arr - peak) / peak
    max_dd = np.min(drawdown)

    win_rate = np.mean(daily_rets > 0) if len(daily_rets) > 0 else 0

    return BacktestResult(
        strategy=config.name,
        exchange="",
        universe="",
        params={
            "kama": config.kama_period,
            "tsmom": config.tsmom_period,
            "btc_ma": config.btc_ma_period,
            "btc_gate": config.use_btc_gate,
            "max_pos": config.max_positions,
        },
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=trade_count,
        invested_days=invested_days,
        total_days=len(all_dates),
        final_value=final_value,
    )


# ============================================================
# 데이터 로딩
# ============================================================


def load_exchange_data(
    exchange: str, min_days: int = 100
) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """거래소 데이터 로드"""
    folder = DATA_ROOT / f"{exchange}_1d"

    if not folder.exists():
        logger.warning(f"폴더 없음: {folder}")
        return {}, None

    data = {}
    btc_data = None

    for csv_file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)

            # 날짜 컬럼 찾기
            date_col = None
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_col = col
                    break

            if date_col is None:
                continue

            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.set_index("date")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # OHLCV 컬럼 확인
            required = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                continue

            df = df[required]

            if len(df) >= min_days:
                symbol = csv_file.stem
                data[symbol] = df

                # BTC 데이터 식별
                if symbol.upper() in ["BTC", "BTCUSDT", "BTC-KRW"]:
                    btc_data = df

        except Exception as e:
            continue

    # BTC 찾기 (없으면 이름으로 검색)
    if btc_data is None:
        for key, df in data.items():
            if (
                "BTC" in key.upper()
                and "DOWN" not in key.upper()
                and "UP" not in key.upper()
            ):
                btc_data = df
                break

    return data, btc_data


def filter_universe(
    data: Dict[str, pd.DataFrame],
    universe_type: str,
    btc_data: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """유니버스 필터링"""
    if universe_type == "btc_only":
        for key, df in data.items():
            if "BTC" in key.upper() and "DOWN" not in key.upper():
                return {key: df}
        return {}

    if universe_type == "all":
        return data

    # 거래량 기준 상위 N개
    n_map = {
        "top5": 5,
        "top10": 10,
        "top20": 20,
        "top50": 50,
    }

    if universe_type not in n_map:
        return data

    n = n_map[universe_type]

    # 평균 거래량으로 정렬
    volumes = []
    for symbol, df in data.items():
        avg_vol = (df["close"] * df["volume"]).mean()
        volumes.append((symbol, avg_vol))

    volumes.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in volumes[:n]]

    return {s: data[s] for s in top_symbols if s in data}


# ============================================================
# 메인 실행
# ============================================================


def run_comprehensive_backtest():
    """종합 백테스트 실행"""

    print("\n" + "=" * 80)
    print("종합 백테스트 - 모든 거래소, 모든 전략, 모든 조합")
    print("=" * 80 + "\n")

    # 테스트 설정
    exchanges = ["upbit", "bithumb", "binance_spot", "binance_futures"]
    universes = ["btc_only", "top5", "top10", "top20"]

    strategies = [
        ("buy_hold", False, False, False),  # (name, use_kama, use_tsmom, use_gate)
        ("kama_only", True, False, False),
        ("tsmom_only", False, True, False),
        ("kama_or_tsmom", True, True, False),
        ("or_loose", True, True, True),
        ("kama_and_tsmom", True, True, False),  # AND 조합
        ("or_loose_strict", True, True, True),  # AND with gate
    ]

    # 파라미터 조합
    kama_periods = [5, 10, 20]
    tsmom_periods = [30, 60, 90]
    btc_ma_periods = [20, 30, 50]

    results = []
    total_tests = 0

    for exchange in exchanges:
        print(f"\n{'='*60}")
        print(f"거래소: {exchange.upper()}")
        print(f"{'='*60}")

        # 데이터 로드
        data, btc_data = load_exchange_data(exchange)

        if not data:
            print(f"  데이터 없음, 건너뜀")
            continue

        print(f"  로드된 심볼: {len(data)}개")

        for universe in universes:
            filtered_data = filter_universe(data, universe, btc_data)

            if not filtered_data:
                continue

            print(f"\n  유니버스: {universe} ({len(filtered_data)}개 심볼)")

            for strat_name, use_kama, use_tsmom, use_gate in strategies:
                # AND 조합 처리
                is_and_combo = (
                    "and" in strat_name.lower() or "strict" in strat_name.lower()
                )

                # 파라미터 조합 테스트
                if strat_name == "buy_hold":
                    # Buy & Hold는 파라미터 불필요
                    param_combos = [(5, 90, 30)]
                else:
                    param_combos = list(
                        product(kama_periods, tsmom_periods, btc_ma_periods)
                    )

                for kama_p, tsmom_p, btc_ma_p in param_combos:
                    config = StrategyConfig(
                        name=strat_name,
                        kama_period=kama_p,
                        tsmom_period=tsmom_p,
                        btc_ma_period=btc_ma_p,
                        use_btc_gate=use_gate,
                        use_kama=use_kama,
                        use_tsmom=use_tsmom,
                        max_positions=min(20, len(filtered_data)),
                    )

                    # AND 조합 처리
                    if is_and_combo:
                        config.name = strat_name

                    result = run_backtest(filtered_data, config, btc_data=btc_data)
                    result.exchange = exchange
                    result.universe = universe

                    results.append(result)
                    total_tests += 1

            # 진행상황 출력
            if len(results) % 50 == 0:
                print(f"    진행: {len(results)} 테스트 완료...")

    print(f"\n\n총 {total_tests}개 테스트 완료")

    # 결과 DataFrame 생성
    df_results = pd.DataFrame([r.to_dict() for r in results])

    # 결과 저장
    output_file = (
        PROJECT_ROOT
        / "outputs"
        / f"comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {output_file}")

    # 요약 출력
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)

    # 거래소별 최고 전략
    print("\n[거래소별 최고 수익률 전략]")
    for exchange in exchanges:
        ex_results = df_results[df_results["exchange"] == exchange]
        if len(ex_results) == 0:
            continue

        best = ex_results.loc[ex_results["total_return"].idxmax()]
        print(f"\n  {exchange.upper()}:")
        print(f"    전략: {best['strategy']}")
        print(f"    유니버스: {best['universe']}")
        print(f"    파라미터: {best['params']}")
        print(f"    총 수익률: {best['total_return']*100:.1f}%")
        print(f"    샤프비율: {best['sharpe_ratio']:.2f}")
        print(f"    MDD: {best['max_drawdown']*100:.1f}%")

    # 전략별 평균 성과
    print("\n\n[전략별 평균 성과]")
    strategy_summary = (
        df_results.groupby("strategy")
        .agg(
            {
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean",
            }
        )
        .round(3)
    )

    strategy_summary = strategy_summary.sort_values("sharpe_ratio", ascending=False)
    print(strategy_summary.to_string())

    # 유니버스별 평균 성과
    print("\n\n[유니버스별 평균 성과]")
    universe_summary = (
        df_results.groupby("universe")
        .agg(
            {
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
            }
        )
        .round(3)
    )

    universe_summary = universe_summary.sort_values("sharpe_ratio", ascending=False)
    print(universe_summary.to_string())

    # 상위 10개 전략
    print("\n\n[상위 10개 전략 (샤프비율 기준)]")
    top10 = df_results.nlargest(10, "sharpe_ratio")[
        [
            "strategy",
            "exchange",
            "universe",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "params",
        ]
    ]

    for i, row in top10.iterrows():
        print(f"\n  #{top10.index.get_loc(i)+1}:")
        print(f"    {row['strategy']} | {row['exchange']} | {row['universe']}")
        print(
            f"    수익률: {row['total_return']*100:.1f}% | 샤프: {row['sharpe_ratio']:.2f} | MDD: {row['max_drawdown']*100:.1f}%"
        )

    return df_results


if __name__ == "__main__":
    results = run_comprehensive_backtest()
