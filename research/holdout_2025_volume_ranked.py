"""
2025 Holdout Test - Volume Ranked Selection
Original methodology likely ranked symbols by volume/liquidity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("2025 HOLDOUT TEST - VOLUME RANKED SELECTION")
print("=" * 80)

DATA_ROOT = Path("E:/data/crypto_ohlcv")
MARKETS = {
    'binance': 'binance_spot_1d',
    'upbit': 'upbit_1d',
    'bithumb': 'bithumb_1d'
}

STRATEGIES = [
    {'name': 'KAMA=10, TSMOM=60', 'kama': 10, 'tsmom': 60, 'gate': 30},
    {'name': 'KAMA=5, TSMOM=90', 'kama': 5, 'tsmom': 90, 'gate': 30},
]

HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")
INITIAL_CAPITAL = 10000
MAX_POSITIONS = 20

def load_market_data(market_folder: Path) -> dict:
    data = {}
    for csv_file in market_folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
            df['date'] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.sort_values('date').reset_index(drop=True)
            df = df.drop_duplicates(subset=['date'], keep='last')
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            symbol = csv_file.stem
            if symbol:
                data[symbol] = df
        except:
            continue
    return data

def calc_kama(prices, period):
    n = len(prices)
    kama = np.full(n, np.nan)
    if n < period + 1:
        return kama
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

def run_backtest_volume_ranked(signal_data: dict, common_dates: list, volume_data: dict,
                                initial_capital: float = 10000, max_positions: int = 20) -> dict:
    """
    Volume-ranked backtest:
    - Among symbols with signals, select top N by trading volume
    """
    portfolio_values = []
    daily_returns = []
    position_counts = []

    current_value = initial_capital
    prev_positions = {}

    for i, date in enumerate(common_dates):
        signals_today = {}
        prices_today = {}
        volumes_today = {}

        for symbol, df in signal_data.items():
            if date in df.index:
                signals_today[symbol] = df.loc[date, 'final_signal']
                prices_today[symbol] = df.loc[date, 'close']
                # Get recent average volume for ranking
                idx = df.index.get_loc(date)
                if idx >= 20:
                    recent_vol = df.iloc[idx-20:idx]['volume'].mean() * df.iloc[idx-20:idx]['close'].mean()
                else:
                    recent_vol = df.iloc[:idx+1]['volume'].mean() * df.iloc[:idx+1]['close'].mean() if idx > 0 else 0
                volumes_today[symbol] = recent_vol

        # Filter to symbols with signals
        active = [(s, volumes_today.get(s, 0)) for s, sig in signals_today.items() if sig]

        # Sort by volume (descending) and take top N
        active_sorted = sorted(active, key=lambda x: x[1], reverse=True)
        selected = [s for s, v in active_sorted[:max_positions]]

        position_counts.append(len(selected))

        # Calculate daily P&L
        if i > 0 and len(prev_positions) > 0:
            daily_pnl = 0
            for symbol, (weight, prev_price) in prev_positions.items():
                if symbol in prices_today:
                    curr_price = prices_today[symbol]
                    ret = (curr_price - prev_price) / prev_price
                    daily_pnl += current_value * weight * ret

            current_value += daily_pnl
            daily_ret = daily_pnl / portfolio_values[-1] if portfolio_values else 0
            daily_returns.append(daily_ret)
        else:
            daily_returns.append(0)

        portfolio_values.append(current_value)

        # Set positions for next day
        if len(selected) > 0:
            weight = 1.0 / len(selected)
            prev_positions = {s: (weight, prices_today[s]) for s in selected if s in prices_today}
        else:
            prev_positions = {}

    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)

    total_return = (portfolio_values[-1] - initial_capital) / initial_capital

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)

    return {
        'sharpe': sharpe,
        'mdd': max_drawdown,
        'total_return': total_return,
        'avg_positions': np.mean(position_counts),
        'final_value': portfolio_values[-1],
        'n_days': len(portfolio_values),
    }

def run_strategy_on_market(price_data: dict, strategy: dict, market_name: str) -> dict:
    kama_period = strategy['kama']
    tsmom_period = strategy['tsmom']
    gate_period = strategy['gate']

    btc_key = None
    for key in price_data.keys():
        if key.upper() == 'BTC' or key.upper() == 'BTCUSDT':
            btc_key = key
            break
    if btc_key is None:
        for key in price_data.keys():
            if 'BTC' in key.upper() and 'DOWN' not in key.upper() and 'UP' not in key.upper():
                btc_key = key
                break
    if btc_key is None:
        return None

    btc_df = price_data[btc_key].copy()
    btc_prices = btc_df['close'].values
    btc_ma = calc_ma(btc_prices, gate_period)
    btc_gate = btc_prices > btc_ma
    btc_df['gate'] = btc_gate
    btc_df = btc_df.set_index('date')

    signal_data = {}
    volume_data = {}

    for symbol, df in price_data.items():
        df = df.copy()
        prices = df['close'].values
        n = len(prices)
        if n < max(kama_period, tsmom_period, gate_period) + 10:
            continue

        kama = calc_kama(prices, kama_period)
        kama_signal = prices > kama
        tsmom_signal = np.zeros(n, dtype=bool)
        for i in range(tsmom_period, n):
            tsmom_signal[i] = prices[i] > prices[i - tsmom_period]

        df['entry_signal'] = kama_signal | tsmom_signal
        df['volume'] = df['volume']
        df = df.set_index('date')
        df = df.join(btc_df[['gate']], how='left')
        df['gate'] = df['gate'].fillna(False).astype(bool)
        df['final_signal'] = df['gate'] & df['entry_signal']

        df_2025 = df[(df.index >= HOLDOUT_START) & (df.index <= HOLDOUT_END)]
        if len(df_2025) > 0:
            signal_data[symbol] = df_2025
            volume_data[symbol] = df_2025

    if len(signal_data) == 0:
        return None

    all_dates = set()
    for df in signal_data.values():
        all_dates.update(df.index.tolist())
    common_dates = sorted(list(all_dates))

    if len(common_dates) < 10:
        return None

    result = run_backtest_volume_ranked(signal_data, common_dates, volume_data, INITIAL_CAPITAL, MAX_POSITIONS)
    result['n_symbols'] = len(signal_data)
    result['period'] = f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}"

    return result

# ============================================================
# Main
# ============================================================
print("\n[1] Loading Data")
print("-" * 60)

market_data = {}
for market_name, folder_name in MARKETS.items():
    folder_path = DATA_ROOT / folder_name
    if folder_path.exists():
        data = load_market_data(folder_path)
        market_data[market_name] = data
        print(f"  {market_name}: {len(data)} symbols loaded")

print("\n[2] Running Backtests (Volume-Ranked Selection)")
print("-" * 60)

results = []

for strategy in STRATEGIES:
    print(f"\nStrategy: {strategy['name']}")

    for market_name in MARKETS.keys():
        if market_name not in market_data:
            continue

        data = market_data[market_name]
        result = run_strategy_on_market(data, strategy, market_name)

        if result:
            results.append({
                'strategy': strategy['name'],
                'market': market_name,
                'sharpe': result['sharpe'],
                'mdd': result['mdd'],
                'return': result['total_return'],
                'avg_pos': result['avg_positions'],
            })
            print(f"  {market_name}: Sharpe={result['sharpe']:.3f}, MDD={result['mdd']*100:.1f}%, Return={result['total_return']*100:.1f}%")

print("\n" + "=" * 80)
print("RESULTS SUMMARY (Volume-Ranked)")
print("=" * 80)

results_df = pd.DataFrame(results)

print(f"\n{'Strategy':<20} {'Market':<10} {'Sharpe':>8} {'MDD':>8} {'Return':>10}")
print("-" * 60)
for _, row in results_df.iterrows():
    print(f"{row['strategy']:<20} {row['market']:<10} {row['sharpe']:>8.3f} {row['mdd']*100:>7.1f}% {row['return']*100:>9.1f}%")

print("\n" + "-" * 60)
print("COMPARISON WITH ORIGINAL REPORT")
print("-" * 60)
print("""
Original (KAMA=5, TSMOM=90):
  Upbit:   Sharpe 2.35, MDD -21.8%, Return +133%
  Bithumb: Sharpe 1.81, MDD -26.2%, Return +109%
  Binance: Sharpe 2.54, MDD -18.4%, Return +176%
""")

output_path = "E:/data/holdout_2025_volume_ranked_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
print("=" * 80)
