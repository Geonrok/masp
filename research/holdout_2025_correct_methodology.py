"""
2025 Holdout Test - CORRECT Methodology
No look-ahead bias: signal at Day N close -> trade at Day N+1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = Path("E:/data/crypto_ohlcv")
MARKETS = {'Binance': 'binance_spot_1d', 'Upbit': 'upbit_1d', 'Bithumb': 'bithumb_1d'}
HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")

def load_market(folder):
    data = {}
    for csv_file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
            df['date'] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.sort_values('date').drop_duplicates('date', keep='last')
            df = df[['date', 'close', 'volume']].set_index('date')
            if len(df[(df.index >= HOLDOUT_START) & (df.index <= HOLDOUT_END)]) >= 30:
                data[csv_file.stem] = df
        except:
            continue
    return data

def calc_kama(prices, period):
    n = len(prices)
    kama = np.full(n, np.nan)
    if n < period + 1: return kama
    kama[period-1] = np.mean(prices[:period])
    fast, slow = 2/(2+1), 2/(30+1)
    for i in range(period, n):
        change = abs(prices[i] - prices[i-period])
        volatility = sum(abs(prices[j] - prices[j-1]) for j in range(i-period+1, i+1))
        er = change / volatility if volatility > 0 else 0
        sc = (er * (fast - slow) + slow) ** 2
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
    return kama

def calc_ma(prices, period):
    result = np.full(len(prices), np.nan)
    for i in range(period-1, len(prices)):
        result[i] = np.mean(prices[i-period+1:i+1])
    return result

def run_correct_backtest(data, kama_p, tsmom_p):
    """
    올바른 방법론:
    1. Day N 종가에서 시그널 확인
    2. Day N+1에 매수/매도 실행 (종가 기준)
    3. Day N의 수익은 Day N-1에 보유한 포지션 기준
    """
    btc_key = None
    for k in data.keys():
        if k.upper() in ['BTC', 'BTCUSDT']:
            btc_key = k
            break
    if not btc_key:
        for k in data.keys():
            if 'BTC' in k.upper() and 'DOWN' not in k.upper():
                btc_key = k
                break
    if not btc_key: return None

    btc_df = data[btc_key]
    btc_prices = btc_df['close'].values
    btc_ma30 = calc_ma(btc_prices, 30)
    btc_gate = pd.Series(btc_prices > btc_ma30, index=btc_df.index)

    btc_2025 = btc_df[(btc_df.index >= HOLDOUT_START) & (btc_df.index <= HOLDOUT_END)]
    if len(btc_2025) < 2: return None
    btc_ret = (btc_2025['close'].iloc[-1] - btc_2025['close'].iloc[0]) / btc_2025['close'].iloc[0]

    # 시그널 생성
    signal_data = {}
    for symbol, df in data.items():
        prices = df['close'].values
        n = len(prices)
        if n < 100: continue
        kama = calc_kama(prices, kama_p)
        kama_sig = prices > kama
        tsmom_sig = np.array([prices[i] > prices[i-tsmom_p] if i >= tsmom_p else False for i in range(n)])
        df = df.copy()
        df['entry'] = kama_sig | tsmom_sig
        df = df.join(pd.DataFrame({'gate': btc_gate}), how='left')
        df['gate'] = df['gate'].fillna(False)
        df['signal'] = df['gate'] & df['entry']
        df_2025 = df[(df.index >= HOLDOUT_START) & (df.index <= HOLDOUT_END)]
        if len(df_2025) > 0:
            signal_data[symbol] = df_2025

    if not signal_data: return None
    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))
    dates_2025 = [d for d in all_dates if HOLDOUT_START <= d <= HOLDOUT_END]

    # 올바른 백테스트
    portfolio = 10000
    daily_rets = []
    prev_positions = {}
    entry_count = 0
    exit_count = 0

    for i, date in enumerate(dates_2025):
        prices_today, signals_today, volumes_today = {}, {}, {}
        for symbol, df in signal_data.items():
            if date in df.index:
                signals_today[symbol] = df.loc[date, 'signal']
                prices_today[symbol] = df.loc[date, 'close']
                volumes_today[symbol] = df.loc[date, 'volume'] * df.loc[date, 'close']

        # 오늘의 수익: 전일 포지션 기준
        if prev_positions:
            pnl = 0
            for sym, (w, prev_p) in prev_positions.items():
                if sym in prices_today:
                    ret = (prices_today[sym] - prev_p) / prev_p
                    pnl += portfolio * w * ret
            portfolio += pnl
            daily_rets.append(pnl / (portfolio - pnl) if portfolio != pnl else 0)
        else:
            daily_rets.append(0)

        # 시그널 기반 포지션 결정
        active = [(s, volumes_today.get(s, 0)) for s, sig in signals_today.items() if sig]
        selected = [s for s, v in sorted(active, key=lambda x: x[1], reverse=True)[:20]]

        was_invested = len(prev_positions) > 0
        is_invested = len(selected) > 0
        if not was_invested and is_invested:
            entry_count += 1
        elif was_invested and not is_invested:
            exit_count += 1

        if selected:
            w = 1.0 / len(selected)
            prev_positions = {s: (w, prices_today[s]) for s in selected if s in prices_today}
        else:
            prev_positions = {}

    daily_rets = np.array(daily_rets)
    total_ret = (portfolio - 10000) / 10000
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    pv = np.cumprod(1 + daily_rets) * 10000
    mdd = np.min((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv))

    winning_days = np.sum(daily_rets > 0)
    losing_days = np.sum(daily_rets < 0)
    win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

    return {
        'return': total_ret, 'sharpe': sharpe, 'mdd': mdd,
        'btc_ret': btc_ret, 'n_symbols': len(signal_data),
        'entry_count': entry_count, 'exit_count': exit_count,
        'win_rate': win_rate, 'trading_days': len(dates_2025)
    }

# Main
print("=" * 85)
print("CORRECT METHODOLOGY BACKTEST - 2025 HOLDOUT")
print("=" * 85)
print()
print("Method: Signal at Day N close -> Position for Day N+1 -> Return attributed correctly")
print()

results = []
for market_name, folder_name in MARKETS.items():
    folder = DATA_ROOT / folder_name
    if not folder.exists(): continue
    data = load_market(folder)
    print(f"Loading {market_name}: {len(data)} symbols")

    for strat_name, kama, tsmom in [('KAMA=5, TSMOM=90', 5, 90), ('KAMA=10, TSMOM=60', 10, 60)]:
        r = run_correct_backtest(data, kama, tsmom)
        if r:
            results.append((market_name, strat_name, r))

print()
print("=" * 85)
print("DETAILED RESULTS")
print("=" * 85)

for market, strat, r in results:
    print(f"\n{market} - {strat}:")
    print(f"  Total Return:  {r['return']*100:+.1f}%")
    print(f"  Sharpe Ratio:  {r['sharpe']:.3f}")
    print(f"  Max Drawdown:  {r['mdd']*100:.1f}%")
    print(f"  Win Rate:      {r['win_rate']*100:.1f}%")
    print(f"  Entry/Exit:    {r['entry_count']} / {r['exit_count']}")
    print(f"  BTC B&H:       {r['btc_ret']*100:+.1f}%")
    print(f"  vs BTC:        {(r['return']-r['btc_ret'])*100:+.1f}%p")

print()
print("=" * 85)
print("SUMMARY TABLE")
print("=" * 85)
print(f"{'Market':<10} {'Strategy':<20} {'Return':>12} {'Sharpe':>10} {'MDD':>10} {'vs BTC':>12}")
print("-" * 85)
for market, strat, r in results:
    print(f"{market:<10} {strat:<20} {r['return']*100:>+11.1f}% {r['sharpe']:>10.2f} {r['mdd']*100:>9.1f}% {(r['return']-r['btc_ret'])*100:>+11.1f}%p")

print()
print("=" * 85)
print("COMPARISON: CORRECT vs BIASED METHODOLOGY")
print("=" * 85)

original_biased = {
    ('Binance', 'KAMA=5, TSMOM=90'): +176,
    ('Binance', 'KAMA=10, TSMOM=60'): +150,
    ('Upbit', 'KAMA=5, TSMOM=90'): +133,
    ('Upbit', 'KAMA=10, TSMOM=60'): +110,
    ('Bithumb', 'KAMA=5, TSMOM=90'): +109,
    ('Bithumb', 'KAMA=10, TSMOM=60'): +95,
}

print(f"\n{'Market':<10} {'Strategy':<20} {'Correct':>12} {'Biased':>12} {'Difference':>12}")
print("-" * 70)
for market, strat, r in results:
    biased = original_biased.get((market, strat), 0)
    correct = r['return'] * 100
    diff = correct - biased
    print(f"{market:<10} {strat:<20} {correct:>+11.1f}% {biased:>+11.0f}% {diff:>+11.0f}%p")

print()
print("=" * 85)
avg_ret = np.mean([r['return'] for _, _, r in results])
avg_sharpe = np.mean([r['sharpe'] for _, _, r in results])
print(f"Average Return (Correct): {avg_ret*100:+.1f}%")
print(f"Average Sharpe (Correct): {avg_sharpe:.2f}")
print("=" * 85)
