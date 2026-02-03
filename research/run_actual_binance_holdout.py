"""
KAMA=10, TSMOM=60 Binance 2025 Holdout - 실제 백테스트
"""

import sys

sys.path.insert(0, "E:/투자/Multi-Asset Strategy Platform")

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 70)
print("BINANCE 2025 HOLDOUT - 실제 백테스트")
print("Strategy: KAMA=10, TSMOM=60, Gate=30")
print("=" * 70)

# ============================================================
# 1. Binance API로 데이터 가져오기
# ============================================================
print("\n[1] Binance API 데이터 로드")
print("-" * 50)

from libs.adapters.real_binance_spot import BinanceSpotMarketData

adapter = BinanceSpotMarketData()


def fetch_ohlcv(symbol: str, days: int = 500) -> pd.DataFrame:
    try:
        ohlcv = adapter.get_ohlcv(symbol, interval="1d", limit=days)
        if not ohlcv:
            return pd.DataFrame()

        data = []
        for candle in ohlcv:
            data.append(
                {
                    "date": candle.timestamp[:10],
                    "close": candle.close,
                }
            )
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()


print("Fetching symbols...")
all_symbols = adapter.get_all_symbols()
print(f"Total USDT pairs: {len(all_symbols)}")

major_symbols = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "SOL/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "SHIB/USDT",
    "LTC/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "ATOM/USDT",
    "UNI/USDT",
    "ETC/USDT",
    "XLM/USDT",
    "BCH/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "FIL/USDT",
    "LDO/USDT",
    "ARB/USDT",
    "OP/USDT",
    "AAVE/USDT",
    "MKR/USDT",
    "GRT/USDT",
    "STX/USDT",
    "IMX/USDT",
]

valid_symbols = [s for s in major_symbols if s in all_symbols]
print(f"Using {len(valid_symbols)} symbols")

# ============================================================
# 2. 가격 데이터 로드
# ============================================================
print("\n[2] 데이터 로드 중...")
print("-" * 50)

price_data = {}

for i, symbol in enumerate(valid_symbols):
    df = fetch_ohlcv(symbol, days=500)
    if not df.empty and len(df) > 100:
        price_data[symbol] = df
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(valid_symbols)}] loaded...")

print(f"Total: {len(price_data)} symbols")

btc_df = price_data.get("BTC/USDT")
if btc_df is None:
    print("[ERROR] BTC required!")
    sys.exit(1)


# ============================================================
# 3. 지표 계산
# ============================================================
def calc_kama(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """KAMA 계산"""
    n = len(prices)
    kama = np.zeros(n)
    kama[:period] = np.nan

    if n < period + 1:
        return kama

    # 초기값
    kama[period - 1] = np.mean(prices[:period])

    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)

    for i in range(period, n):
        change = abs(prices[i] - prices[i - period])
        volatility = sum(
            abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1)
        )
        er = change / volatility if volatility > 0 else 0
        sc = (er * (fast - slow) + slow) ** 2
        kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])

    return kama


def calc_ma(prices: np.ndarray, period: int) -> np.ndarray:
    result = np.zeros(len(prices))
    result[: period - 1] = np.nan
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


# ============================================================
# 4. 신호 계산
# ============================================================
print("\n[3] 신호 계산")
print("-" * 50)

KAMA_PERIOD = 10
TSMOM_LOOKBACK = 60
GATE_MA_PERIOD = 30

# BTC Gate
btc_prices = btc_df["close"].values
btc_ma30 = calc_ma(btc_prices, GATE_MA_PERIOD)
btc_gate = btc_prices > btc_ma30

# Gate 신호를 DataFrame으로
btc_df = btc_df.copy()
btc_df["gate"] = btc_gate

# 2025년 필터
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-12-31")

# 각 심볼별 신호 계산
signal_data = {}

for symbol, df in price_data.items():
    df = df.copy()
    prices = df["close"].values
    n = len(prices)

    # KAMA
    kama = calc_kama(prices, KAMA_PERIOD)
    kama_signal = prices > kama

    # TSMOM
    tsmom_signal = np.zeros(n, dtype=bool)
    for i in range(TSMOM_LOOKBACK, n):
        tsmom_signal[i] = prices[i] > prices[i - TSMOM_LOOKBACK]

    # Entry signal (KAMA OR TSMOM)
    entry_signal = kama_signal | tsmom_signal

    df["kama"] = kama
    df["kama_signal"] = kama_signal
    df["tsmom_signal"] = tsmom_signal
    df["entry_signal"] = entry_signal

    # Gate 병합
    df = df.merge(btc_df[["date", "gate"]], on="date", how="left")
    df["gate"] = df["gate"].fillna(False)
    df["final_signal"] = df["gate"] & df["entry_signal"]

    # 2025년 필터
    df_2025 = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if len(df_2025) > 0:
        signal_data[symbol] = df_2025.set_index("date")

print(f"Symbols with 2025 data: {len(signal_data)}")

# 공통 날짜 찾기
all_dates = set()
for symbol, df in signal_data.items():
    all_dates.update(df.index.tolist())

common_dates = sorted(list(all_dates))
print(f"Trading days: {len(common_dates)}")
print(f"Period: {common_dates[0]} ~ {common_dates[-1]}")

# 신호 확인
total_signals = 0
for date in common_dates[:30]:  # 처음 30일만 확인
    cnt = 0
    for symbol, df in signal_data.items():
        if date in df.index and df.loc[date, "final_signal"]:
            cnt += 1
    total_signals += cnt

print(f"Average signals per day (first 30 days): {total_signals/30:.1f}")

# ============================================================
# 5. 백테스트
# ============================================================
print("\n[4] 백테스트 실행")
print("-" * 50)

INITIAL_CAPITAL = 10000
MAX_POSITIONS = 20

portfolio_values = []
daily_returns = []
position_counts = []

current_value = INITIAL_CAPITAL
prev_positions = {}

for i, date in enumerate(common_dates):
    # 현재 신호
    signals_today = {}
    prices_today = {}

    for symbol, df in signal_data.items():
        if date in df.index:
            signals_today[symbol] = df.loc[date, "final_signal"]
            prices_today[symbol] = df.loc[date, "close"]

    # 활성 신호
    active = [s for s, sig in signals_today.items() if sig]
    selected = active[:MAX_POSITIONS]
    position_counts.append(len(selected))

    # 일일 수익 계산
    if i > 0 and len(prev_positions) > 0:
        daily_pnl = 0
        for symbol, (weight, prev_price) in prev_positions.items():
            if symbol in prices_today:
                curr_price = prices_today[symbol]
                ret = (curr_price - prev_price) / prev_price
                daily_pnl += current_value * weight * ret

        current_value += daily_pnl
        if portfolio_values:
            daily_ret = daily_pnl / portfolio_values[-1]
        else:
            daily_ret = 0
        daily_returns.append(daily_ret)
    else:
        daily_returns.append(0)

    portfolio_values.append(current_value)

    # 다음날 포지션 설정
    if len(selected) > 0:
        weight = 1.0 / len(selected)
        prev_positions = {
            s: (weight, prices_today[s]) for s in selected if s in prices_today
        }
    else:
        prev_positions = {}

# ============================================================
# 6. 성과 지표
# ============================================================
print("\n[5] 성과 지표")
print("-" * 50)

portfolio_values = np.array(portfolio_values)
daily_returns = np.array(daily_returns)

total_return = (portfolio_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL

if len(daily_returns) > 1 and np.std(daily_returns) > 0:
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
else:
    sharpe = 0

peak = np.maximum.accumulate(portfolio_values)
drawdown = (portfolio_values - peak) / peak
max_drawdown = np.min(drawdown)

n_days = len(portfolio_values)
cagr = (
    (portfolio_values[-1] / INITIAL_CAPITAL) ** (365 / n_days) - 1 if n_days > 0 else 0
)

avg_positions = np.mean(position_counts)

print(f"Initial: ${INITIAL_CAPITAL:,.0f}")
print(f"Final:   ${portfolio_values[-1]:,.0f}")
print(f"Return:  {total_return*100:.1f}%")
print(f"CAGR:    {cagr*100:.1f}%")
print(f"Sharpe:  {sharpe:.3f}")
print(f"MDD:     {max_drawdown*100:.1f}%")
print(f"Days:    {n_days}")
print(f"Avg Pos: {avg_positions:.1f}")

# ============================================================
# 7. 비교
# ============================================================
print("\n" + "=" * 70)
print("결과 비교")
print("=" * 70)

print(f"""
+-------------------------------------------------------------------+
| Strategy          | Market  | Sharpe  | MDD     | Return | Source |
+-------------------------------------------------------------------+
| KAMA=10, TSMOM=60 | Upbit   | 3.162   | -17.3%  | 203%   | actual |
| KAMA=10, TSMOM=60 | Bithumb | 2.581   | -18.9%  | 163%   | actual |
| KAMA=10, TSMOM=60 | Binance | {sharpe:.3f}   | {max_drawdown*100:.1f}%  | {total_return*100:.0f}%   | NEW    |
+-------------------------------------------------------------------+
| KAMA=5, TSMOM=90  | Upbit   | 2.350   | -21.8%  | 133%   | actual |
| KAMA=5, TSMOM=90  | Bithumb | 1.810   | -26.2%  | 109%   | actual |
| KAMA=5, TSMOM=90  | Binance | 2.540   | -18.4%  | 176%   | actual |
+-------------------------------------------------------------------+
""")

# 저장
results_df = pd.DataFrame(
    [
        {
            "strategy": "KAMA=10, TSMOM=60",
            "market": "binance",
            "sharpe": sharpe,
            "mdd": max_drawdown,
            "total_return": total_return,
            "avg_positions": avg_positions,
        }
    ]
)
output_path = (
    "E:/투자/Multi-Asset Strategy Platform/research/binance_holdout_kama10_actual.csv"
)
results_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print("=" * 70)
