"""
Generate Final Summary - All KOSPI and KOSDAQ Results
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path('E:/투자/data/kr_stock/backtest_results')

print('=' * 80)
print('FINAL COMPREHENSIVE STRATEGY SUMMARY')
print('KOSPI + KOSDAQ Markets')
print('=' * 80)

# Load all results
kospi_files = [
    'final_all_results_20260129_162537.csv',
    'investor_flow_20260129_152812.csv',
]

kosdaq_files = [
    'kosdaq_fast_20260129_213920.csv',
    'kosdaq_advanced_20260129_225023.csv',
]

# Load KOSPI results
kospi_dfs = []
for f in kospi_files:
    try:
        df = pd.read_csv(RESULTS_DIR / f)
        df['market'] = 'KOSPI'
        kospi_dfs.append(df)
        print(f'Loaded KOSPI: {f} - {len(df)} rows')
    except Exception as e:
        print(f'Error loading {f}: {e}')

# Load KOSDAQ results
kosdaq_dfs = []
for f in kosdaq_files:
    try:
        df = pd.read_csv(RESULTS_DIR / f)
        df['market'] = 'KOSDAQ'
        kosdaq_dfs.append(df)
        print(f'Loaded KOSDAQ: {f} - {len(df)} rows')
    except Exception as e:
        print(f'Error loading {f}: {e}')

# Combine all
all_dfs = kospi_dfs + kosdaq_dfs
if len(all_dfs) > 0:
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f'\nTotal combined results: {len(combined)}')

    # Remove duplicates if any
    combined = combined.drop_duplicates(subset=['ticker', 'strategy', 'direction', 'market'], keep='first')
    print(f'After deduplication: {len(combined)}')

    print()
    print('=' * 80)
    print('OVERALL STATISTICS')
    print('=' * 80)
    print(f"Total combinations: {len(combined):,}")
    print(f"Unique strategies: {combined['strategy'].nunique()}")
    print(f"Unique tickers: {combined['ticker'].nunique()}")
    print(f"  - KOSPI tickers: {combined[combined['market']=='KOSPI']['ticker'].nunique()}")
    print(f"  - KOSDAQ tickers: {combined[combined['market']=='KOSDAQ']['ticker'].nunique()}")

    print()
    print('BY MARKET:')
    for market in ['KOSPI', 'KOSDAQ']:
        mkt_df = combined[combined['market'] == market]
        if len(mkt_df) > 0:
            print(f"\n{market}:")
            print(f"  Combinations: {len(mkt_df):,}")
            print(f"  Avg CAGR: {mkt_df['cagr'].mean():.2%}")
            print(f"  Avg Sharpe: {mkt_df['sharpe'].mean():.2f}")
            print(f"  % Sharpe > 0.5: {(mkt_df['sharpe'] > 0.5).mean():.2%}")
            print(f"  % Sharpe > 1.0: {(mkt_df['sharpe'] > 1.0).mean():.2%}")

    print()
    print('BY DIRECTION:')
    for direction in ['LONG', 'SHORT']:
        dir_df = combined[combined['direction'] == direction]
        if len(dir_df) > 0:
            print(f"\n{direction}:")
            print(f"  Combinations: {len(dir_df):,}")
            print(f"  Avg CAGR: {dir_df['cagr'].mean():.2%}")
            print(f"  Avg Sharpe: {dir_df['sharpe'].mean():.2f}")
            print(f"  % Sharpe > 0.5: {(dir_df['sharpe'] > 0.5).mean():.2%}")

    # Tradable combinations
    tradable = combined[(combined['sharpe'] > 0.5) & (combined['max_dd'] > -0.5)]
    print()
    print('=' * 80)
    print('TRADABLE COMBINATIONS (Sharpe > 0.5, MDD > -50%)')
    print('=' * 80)
    print(f"Total tradable: {len(tradable):,}")
    print(f"  - KOSPI: {len(tradable[tradable['market']=='KOSPI']):,}")
    print(f"  - KOSDAQ: {len(tradable[tradable['market']=='KOSDAQ']):,}")
    print(f"  - LONG: {len(tradable[tradable['direction']=='LONG']):,}")
    print(f"  - SHORT: {len(tradable[tradable['direction']=='SHORT']):,}")

    # Best strategies overall
    print()
    print('=' * 80)
    print('TOP 20 STRATEGIES BY MEAN SHARPE')
    print('=' * 80)
    strat_summary = combined.groupby(['strategy', 'direction']).agg({
        'cagr': 'mean',
        'sharpe': 'mean',
        'max_dd': 'mean',
        'win_rate': 'mean',
        'ticker': 'count'
    }).round(4)
    strat_summary.columns = ['avg_cagr', 'avg_sharpe', 'avg_mdd', 'avg_winrate', 'count']
    strat_summary = strat_summary.sort_values('avg_sharpe', ascending=False)
    print(strat_summary.head(20).to_string())

    # Top 50 individual combinations
    print()
    print('=' * 80)
    print('TOP 50 INDIVIDUAL COMBINATIONS (ALL MARKETS)')
    print('=' * 80)
    top50 = combined.nlargest(50, 'sharpe')[['market', 'ticker', 'strategy', 'direction', 'cagr', 'sharpe', 'max_dd']]
    for _, row in top50.iterrows():
        print(f"{row['market']:>6} | {row['ticker']:>8} | {row['strategy']:<18} | {row['direction']:>5} | CAGR: {row['cagr']:>7.2%} | Sharpe: {row['sharpe']:>5.2f} | MDD: {row['max_dd']:>7.2%}")

    # Best by market
    print()
    print('=' * 80)
    print('TOP 30 KOSPI COMBINATIONS')
    print('=' * 80)
    kospi_top = combined[combined['market'] == 'KOSPI'].nlargest(30, 'sharpe')[['ticker', 'strategy', 'direction', 'cagr', 'sharpe', 'max_dd']]
    for _, row in kospi_top.iterrows():
        print(f"{row['ticker']:>8} | {row['strategy']:<18} | {row['direction']:>5} | CAGR: {row['cagr']:>7.2%} | Sharpe: {row['sharpe']:>5.2f} | MDD: {row['max_dd']:>7.2%}")

    print()
    print('=' * 80)
    print('TOP 30 KOSDAQ COMBINATIONS')
    print('=' * 80)
    kosdaq_top = combined[combined['market'] == 'KOSDAQ'].nlargest(30, 'sharpe')[['ticker', 'strategy', 'direction', 'cagr', 'sharpe', 'max_dd']]
    for _, row in kosdaq_top.iterrows():
        print(f"{row['ticker']:>8} | {row['strategy']:<18} | {row['direction']:>5} | CAGR: {row['cagr']:>7.2%} | Sharpe: {row['sharpe']:>5.2f} | MDD: {row['max_dd']:>7.2%}")

    # Save final results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined.to_csv(RESULTS_DIR / f'FINAL_ALL_MARKETS_{timestamp}.csv', index=False, encoding='utf-8-sig')
    tradable.to_csv(RESULTS_DIR / f'FINAL_TRADABLE_{timestamp}.csv', index=False, encoding='utf-8-sig')
    strat_summary.to_csv(RESULTS_DIR / f'FINAL_STRATEGY_SUMMARY_{timestamp}.csv', encoding='utf-8-sig')

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f"Total strategies tested: {combined['strategy'].nunique()}")
    print(f"Total tickers tested: {combined['ticker'].nunique()}")
    print(f"Total combinations: {len(combined):,}")
    print(f"Tradable combinations: {len(tradable):,}")
    print(f"\nFiles saved to: {RESULTS_DIR}")

else:
    print('No data to combine')
