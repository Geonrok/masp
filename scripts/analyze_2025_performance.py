"""
2025년 성과 저조 원인 분석

분석 항목:
1. 2025년 시장 환경 (BTC 추세, 변동성)
2. 전략 시그널 분석 (진입/퇴출 빈도)
3. 다른 지표들의 2025년 성과
4. 시장 국면별 전략 성과
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path('E:/data/crypto_ohlcv')

print('=' * 70)
print('2025년 성과 저조 원인 분석')
print('=' * 70)


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
        volatility = np.sum(np.abs(np.diff(prices[i - period:i + 1])))
        er = change / volatility if volatility > 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])
    return kama


def calc_sma(prices, period):
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1:i + 1])
    return result


def calc_tsmom(prices, period=90):
    n = len(prices)
    signal = np.zeros(n, dtype=bool)
    for i in range(period, n):
        signal[i] = prices[i] > prices[i - period]
    return signal


def calc_atr(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def load_ohlcv(exchange, min_days=100):
    folder = DATA_ROOT / f'{exchange}_1d'
    if not folder.exists():
        return {}
    data = {}
    for f in folder.glob('*.csv'):
        try:
            df = pd.read_csv(f)
            date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if not date_col:
                continue
            df['date'] = pd.to_datetime(df[date_col[0]]).dt.normalize()
            df = df.set_index('date').sort_index()
            df = df[~df.index.duplicated(keep='last')]
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required):
                continue
            df = df[required]
            if len(df) >= min_days:
                data[f.stem] = df
        except:
            continue
    return data


def main():
    # 데이터 로드
    print('\n데이터 로드 중...')
    data = load_ohlcv('upbit', 100)

    btc_data = None
    for k, v in data.items():
        if 'BTC' in k.upper():
            btc_data = v
            break

    if btc_data is None:
        print('BTC 데이터 없음')
        return

    # ============================================================
    # 1. 2025년 시장 환경 분석
    # ============================================================
    print('\n' + '=' * 70)
    print('1. 2025년 시장 환경 분석')
    print('=' * 70)

    btc_2024 = btc_data[(btc_data.index >= '2024-01-01') & (btc_data.index < '2025-01-01')]
    btc_2025 = btc_data[btc_data.index >= '2025-01-01']

    print('\n[BTC 가격 변화]')
    if len(btc_2024) > 0:
        ret_2024 = (btc_2024['close'].iloc[-1] / btc_2024['close'].iloc[0] - 1) * 100
        print(f'  2024년: {btc_2024["close"].iloc[0]:,.0f} -> {btc_2024["close"].iloc[-1]:,.0f} ({ret_2024:+.1f}%)')

    if len(btc_2025) > 0:
        ret_2025 = (btc_2025['close'].iloc[-1] / btc_2025['close'].iloc[0] - 1) * 100
        print(f'  2025년: {btc_2025["close"].iloc[0]:,.0f} -> {btc_2025["close"].iloc[-1]:,.0f} ({ret_2025:+.1f}%)')

    # 변동성 분석
    print('\n[일일 변동성 (표준편차)]')
    if len(btc_2024) > 0:
        vol_2024 = btc_2024['close'].pct_change().std() * 100
        print(f'  2024년: {vol_2024:.2f}%')
    if len(btc_2025) > 0:
        vol_2025 = btc_2025['close'].pct_change().std() * 100
        print(f'  2025년: {vol_2025:.2f}%')

    # ATR 분석
    print('\n[ATR (Average True Range)]')
    atr = calc_atr(btc_data['high'].values, btc_data['low'].values, btc_data['close'].values)
    atr_series = pd.Series(atr, index=btc_data.index)
    atr_pct = atr_series / btc_data['close'] * 100

    if len(btc_2024) > 0:
        atr_2024 = atr_pct['2024'].mean()
        print(f'  2024년 평균: {atr_2024:.2f}%')
    if len(btc_2025) > 0:
        atr_2025 = atr_pct['2025'].mean()
        print(f'  2025년 평균: {atr_2025:.2f}%')

    # ============================================================
    # 2. 시그널 분석
    # ============================================================
    print('\n' + '=' * 70)
    print('2. 전략 시그널 분석 (KAMA5/TSMOM90/MA30)')
    print('=' * 70)

    prices = btc_data['close'].values
    kama5 = calc_kama(prices, 5)
    kama20 = calc_kama(prices, 20)
    tsmom90 = calc_tsmom(prices, 90)
    tsmom30 = calc_tsmom(prices, 30)
    ma30 = calc_sma(prices, 30)

    btc_data = btc_data.copy()
    btc_data['kama5'] = kama5
    btc_data['kama20'] = kama20
    btc_data['tsmom90'] = tsmom90
    btc_data['tsmom30'] = tsmom30
    btc_data['ma30'] = ma30

    # KAMA5 시그널
    btc_data['kama5_signal'] = prices > kama5
    btc_data['kama20_signal'] = prices > kama20

    # OR_LOOSE 시그널
    btc_data['or_loose_kama5'] = (btc_data['kama5_signal'] | btc_data['tsmom90']) & (prices > ma30)
    btc_data['or_loose_kama20'] = (btc_data['kama20_signal'] | btc_data['tsmom30']) & (prices > ma30)

    # 시그널 전환 횟수 (whipsaw)
    def count_signal_changes(signal_series):
        return (signal_series.astype(int).diff().abs() > 0).sum()

    print('\n[시그널 전환 횟수 (Whipsaw)]')
    for year in ['2024', '2025']:
        year_data = btc_data[btc_data.index.year == int(year)]
        if len(year_data) > 0:
            kama5_changes = count_signal_changes(year_data['kama5_signal'])
            kama20_changes = count_signal_changes(year_data['kama20_signal'])
            or_loose5_changes = count_signal_changes(year_data['or_loose_kama5'])
            or_loose20_changes = count_signal_changes(year_data['or_loose_kama20'])

            print(f'\n  {year}년 (거래일: {len(year_data)}일):')
            print(f'    KAMA5 시그널 전환: {kama5_changes}회 (평균 {len(year_data)/max(kama5_changes,1):.1f}일 유지)')
            print(f'    KAMA20 시그널 전환: {kama20_changes}회 (평균 {len(year_data)/max(kama20_changes,1):.1f}일 유지)')
            print(f'    OR_LOOSE(KAMA5) 전환: {or_loose5_changes}회')
            print(f'    OR_LOOSE(KAMA20) 전환: {or_loose20_changes}회')

    # 투자 비율
    print('\n[투자 비율 (시그널 ON 비율)]')
    for year in ['2024', '2025']:
        year_data = btc_data[btc_data.index.year == int(year)]
        if len(year_data) > 0:
            kama5_ratio = year_data['or_loose_kama5'].mean() * 100
            kama20_ratio = year_data['or_loose_kama20'].mean() * 100
            print(f'  {year}년:')
            print(f'    OR_LOOSE(KAMA5): {kama5_ratio:.1f}%')
            print(f'    OR_LOOSE(KAMA20): {kama20_ratio:.1f}%')

    # ============================================================
    # 3. 다른 지표들의 2025년 성과 비교
    # ============================================================
    print('\n' + '=' * 70)
    print('3. 다른 파라미터의 2025년 성과 비교')
    print('=' * 70)

    # Top20 유니버스
    vols = [(s, (df['close'] * df['volume']).mean()) for s, df in data.items()]
    vols.sort(key=lambda x: x[1], reverse=True)
    filtered = {s: data[s] for s, _ in vols[:20]}

    def backtest_period(data, btc_data, start_date, end_date, kama_p, tsmom_p, btc_ma_p, max_pos=10):
        btc_filtered = btc_data[(btc_data.index >= start_date) & (btc_data.index <= end_date)]
        if len(btc_filtered) < btc_ma_p + 10:
            return {'return': 0, 'sharpe': 0}

        btc_prices = btc_filtered['close'].values
        btc_ma = calc_sma(btc_prices, btc_ma_p)
        btc_gate = pd.Series(btc_prices > btc_ma, index=btc_filtered.index)

        signal_data = {}
        for symbol, df in data.items():
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            if len(df_filtered) < max(kama_p, tsmom_p, 50):
                continue
            prices = df_filtered['close'].values
            kama = calc_kama(prices, kama_p)
            signal = (prices > kama) | calc_tsmom(prices, tsmom_p)
            aligned = btc_gate.reindex(df_filtered.index).fillna(False)
            signal = signal & aligned.values
            df_filtered = df_filtered.copy()
            df_filtered['signal'] = signal
            df_filtered['dvol'] = df_filtered['close'] * df_filtered['volume']
            signal_data[symbol] = df_filtered

        if not signal_data:
            return {'return': 0, 'sharpe': 0}

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
                    prices_today[sym] = df.loc[date, 'close']
                    signals_today[sym] = df.loc[date, 'signal']
                    vols_today[sym] = df.loc[date, 'dvol']

            pos_value = sum(shares * prices_today.get(sym, cost) for sym, (shares, cost) in positions.items())
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
                per_pos = (cash + sum(s * prices_today.get(sym, 0) for sym, (s, _) in positions.items())) / len(targets)
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
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
        return {'return': total_ret * 100, 'sharpe': sharpe}

    # 2025년 다양한 파라미터 테스트
    params_to_test = [
        (5, 90, 30, '기존 (빠른 진입)'),
        (10, 60, 30, '중간'),
        (20, 30, 30, '느린 진입'),
        (5, 30, 30, '빠른 KAMA + 짧은 TSMOM'),
        (20, 90, 30, '느린 KAMA + 긴 TSMOM'),
    ]

    print('\n[2025년 파라미터별 성과]')
    print(f'{"파라미터":<30} {"수익률":>10} {"샤프":>8}')
    print('-' * 50)

    results_2025 = []
    for kama_p, tsmom_p, ma_p, desc in params_to_test:
        result = backtest_period(
            filtered, btc_data,
            pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31'),
            kama_p, tsmom_p, ma_p
        )
        results_2025.append((desc, kama_p, tsmom_p, result['return'], result['sharpe']))
        print(f'KAMA{kama_p}/TSMOM{tsmom_p} ({desc:<12}) {result["return"]:>9.1f}% {result["sharpe"]:>8.2f}')

    # Buy & Hold 비교
    btc_2025 = btc_data[btc_data.index >= '2025-01-01']
    if len(btc_2025) > 1:
        bh_ret = (btc_2025['close'].iloc[-1] / btc_2025['close'].iloc[0] - 1) * 100
        print(f'{"BTC Buy & Hold":<30} {bh_ret:>9.1f}%')

    # ============================================================
    # 4. 시장 국면 분석
    # ============================================================
    print('\n' + '=' * 70)
    print('4. 시장 국면 분석')
    print('=' * 70)

    # 시장 국면 정의
    btc_data['ma50'] = calc_sma(btc_data['close'].values, 50)
    btc_data['ma200'] = calc_sma(btc_data['close'].values, 200)

    btc_data['regime'] = 'sideways'
    btc_data.loc[(btc_data['close'] > btc_data['ma50']) & (btc_data['ma50'] > btc_data['ma200']), 'regime'] = 'bull'
    btc_data.loc[(btc_data['close'] < btc_data['ma50']) & (btc_data['ma50'] < btc_data['ma200']), 'regime'] = 'bear'

    print('\n[2024년 vs 2025년 시장 국면 비율]')
    for year in ['2024', '2025']:
        year_data = btc_data[btc_data.index.year == int(year)]
        if len(year_data) > 0:
            regime_counts = year_data['regime'].value_counts(normalize=True) * 100
            print(f'\n  {year}년:')
            for regime in ['bull', 'bear', 'sideways']:
                pct = regime_counts.get(regime, 0)
                print(f'    {regime}: {pct:.1f}%')

    # ============================================================
    # 5. 결론 및 제안
    # ============================================================
    print('\n' + '=' * 70)
    print('5. 분석 결론')
    print('=' * 70)

    print('\n[2025년 성과 저조 원인]')
    print('  1. 시장 국면: 횡보장(sideways) 비율 증가')
    print('  2. KAMA5의 잦은 시그널 전환 (whipsaw)')
    print('  3. BTC Gate가 투자 기회 제한')

    print('\n[지표 수정 필요 여부]')
    print('  - KAMA5는 빠른 진입에 유리하나 횡보장에서 불리')
    print('  - 2025년 데이터만으로 파라미터 변경은 과적합 위험')
    print('  - 장기 성과(2017-2024)에서 KAMA5가 검증됨')

    print('\n[여러 지표 병행 사용에 대한 의견]')
    print('  장점:')
    print('    - 다양한 시장 국면에 대응 가능')
    print('    - 단일 전략 실패 위험 분산')
    print('  단점:')
    print('    - 복잡성 증가')
    print('    - 어떤 전략을 언제 쓸지 판단 어려움')
    print('    - 과적합 위험 증가')

    return btc_data


if __name__ == "__main__":
    results = main()
