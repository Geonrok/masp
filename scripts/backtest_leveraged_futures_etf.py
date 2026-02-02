# -*- coding: utf-8 -*-
"""
코스피 선물 ETF 알고리즘 트레이딩 전략 백테스트
==============================================

대상 ETF:
- KODEX 레버리지 (122630) - +2x
- KODEX 200선물인버스2X (252670) - -2x
- KODEX 인버스 (114800) - -1x

전략 후보:
A. VIX 기반 방향성 스위칭
B. 변동성 브레이크아웃
C. 추세 추종 + 레버리지
D. 단기 모멘텀
E. 현물 + 인버스 헷지

Author: Claude Code
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = 'E:/투자/data/leveraged_etf'
KOSPI_DIR = 'E:/투자/data/kospi_futures'
OUTPUT_DIR = 'E:/투자/Multi-Asset Strategy Platform/logs'

# 비용 설정
COMMISSION = 0.00015  # 0.015%
SLIPPAGE = 0.0005     # 0.05%
ROUND_TRIP_COST = (COMMISSION + SLIPPAGE) * 2


@dataclass
class BacktestResult:
    strategy_name: str
    cagr: float
    sharpe: float
    mdd: float
    avg_holding_days: float
    trades_per_year: float
    win_rate: float
    exposure: float
    equity_curve: pd.Series


def load_data() -> Dict[str, pd.DataFrame]:
    """데이터 로드."""
    data = {}

    # 레버리지/인버스 ETF
    etf_files = {
        'lev2x': f'{DATA_DIR}/122630_KODEX_레버리지.parquet',
        'inv2x': f'{DATA_DIR}/252670_KODEX_200선물인버스2X.parquet',
        'inv1x': f'{DATA_DIR}/114800_KODEX_인버스.parquet',
        'spot': f'{DATA_DIR}/069500_KODEX_200.parquet',
    }

    for name, path in etf_files.items():
        try:
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            data[name] = df
            print(f'[OK] {name}: {len(df)}일')
        except Exception as e:
            print(f'[FAIL] {name}: {e}')

    # VIX
    try:
        vix = pd.read_parquet(f'{DATA_DIR}/VIX.parquet')
        vix.columns = [c.lower() for c in vix.columns]
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        data['vix'] = vix
        print(f'[OK] VIX: {len(vix)}일')
    except:
        # 기존 VIX 데이터 사용
        vix = pd.read_parquet(f'{KOSPI_DIR.replace("kospi_futures", "kosdaq_futures/multi_asset")}/vix.parquet')
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        data['vix'] = vix
        print(f'[OK] VIX (기존): {len(vix)}일')

    # KOSPI200
    try:
        kospi = pd.read_parquet(f'{KOSPI_DIR}/kospi200_daily_yf.parquet')
        kospi.columns = [c.lower() for c in kospi.columns]
        if kospi.index.tz is not None:
            kospi.index = kospi.index.tz_localize(None)
        data['kospi200'] = kospi
        print(f'[OK] KOSPI200: {len(kospi)}일')
    except Exception as e:
        print(f'[FAIL] KOSPI200: {e}')

    return data


def calculate_metrics(returns: pd.Series, position: pd.Series) -> Dict:
    """성과 지표 계산."""
    if len(returns) < 252:
        return None

    # 기본 수익률 계산
    strat_returns = (position.shift(1) * returns).fillna(0)

    # 거래 비용
    turnover = position.diff().abs().fillna(0)
    costs = turnover * ROUND_TRIP_COST / 2
    strat_returns = strat_returns - costs

    # Equity curve
    equity = (1 + strat_returns).cumprod()

    # 지표 계산
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    vol = strat_returns.std() * np.sqrt(252)
    sharpe = (strat_returns.mean() * 252) / vol if vol > 0 else 0

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = drawdown.min()

    # 거래 통계
    pos_changes = position.diff().abs()
    trades = (pos_changes > 0).sum() / 2  # 진입+청산 = 1거래
    trades_per_year = trades / years if years > 0 else 0

    # 승률
    trade_returns = strat_returns[position.shift(1) != 0]
    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0

    # 평균 보유 기간
    in_position = (position != 0).astype(int)
    position_changes = in_position.diff().fillna(0)
    entries = position_changes[position_changes > 0].index.tolist()
    exits = position_changes[position_changes < 0].index.tolist()

    if len(entries) > 0 and len(exits) > 0:
        holding_days = []
        for entry in entries:
            exit_dates = [e for e in exits if e > entry]
            if exit_dates:
                holding_days.append((exit_dates[0] - entry).days)
        avg_holding = np.mean(holding_days) if holding_days else 0
    else:
        avg_holding = 0

    # Exposure
    exposure = (position != 0).mean()

    return {
        'cagr': cagr,
        'sharpe': sharpe,
        'mdd': mdd,
        'avg_holding_days': avg_holding,
        'trades_per_year': trades_per_year,
        'win_rate': win_rate,
        'exposure': exposure,
        'equity': equity,
    }


def strategy_a_vix_switching(data: Dict, max_holding: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    전략 A: VIX 기반 방향성 스위칭

    - VIX < SMA(20) AND VIX 하락 → 레버리지 매수 (+1)
    - VIX > SMA(20) AND VIX 상승 → 인버스2X 매수 (-1)
    - 중립 → 현금 (0)
    """
    vix = data['vix']['close']
    lev2x = data['lev2x']['close']
    inv2x = data['inv2x']['close']

    # 공통 인덱스
    common_idx = vix.index.intersection(lev2x.index).intersection(inv2x.index)
    vix = vix.reindex(common_idx)
    lev2x_ret = lev2x.reindex(common_idx).pct_change()
    inv2x_ret = inv2x.reindex(common_idx).pct_change()

    # VIX 지표
    vix_sma20 = vix.rolling(20).mean()
    vix_declining = vix < vix.shift(1)
    vix_rising = vix > vix.shift(1)

    # 신호 (T-1 데이터로 T일 거래)
    long_signal = ((vix < vix_sma20) & vix_declining).shift(1)
    short_signal = ((vix > vix_sma20) & vix_rising).shift(1)

    # 포지션
    position = pd.Series(0, index=common_idx)
    position[long_signal == True] = 1   # 레버리지
    position[short_signal == True] = -1  # 인버스

    # 최대 보유 기간 제한
    position = apply_max_holding(position, max_holding)

    # 수익률 계산 (롱: lev2x, 숏: inv2x)
    returns = pd.Series(0.0, index=common_idx)
    returns[position == 1] = lev2x_ret[position == 1]
    returns[position == -1] = inv2x_ret[position == -1]

    return returns, position


def strategy_b_volatility_breakout(data: Dict, max_holding: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    전략 B: 변동성 브레이크아웃

    - VIX 급등 (+20%) → 인버스 매수
    - VIX 급락 (-15%) → 레버리지 매수
    - 보유: 최대 5일
    """
    vix = data['vix']['close']
    lev2x = data['lev2x']['close']
    inv2x = data['inv2x']['close']

    common_idx = vix.index.intersection(lev2x.index).intersection(inv2x.index)
    vix = vix.reindex(common_idx)
    lev2x_ret = lev2x.reindex(common_idx).pct_change()
    inv2x_ret = inv2x.reindex(common_idx).pct_change()

    # VIX 변화율
    vix_change = vix.pct_change()

    # 신호 (T-1 데이터로 T일 거래)
    vix_spike = (vix_change > 0.20).shift(1)   # VIX +20%
    vix_crash = (vix_change < -0.15).shift(1)  # VIX -15%

    # 포지션
    position = pd.Series(0, index=common_idx)
    position[vix_spike == True] = -1   # 인버스 (VIX 급등 = 시장 하락)
    position[vix_crash == True] = 1    # 레버리지 (VIX 급락 = 시장 상승)

    # 최대 보유 기간 제한
    position = apply_max_holding(position, max_holding)

    # 수익률
    returns = pd.Series(0.0, index=common_idx)
    returns[position == 1] = lev2x_ret[position == 1]
    returns[position == -1] = inv2x_ret[position == -1]

    return returns, position


def strategy_c_trend_following(data: Dict, max_holding: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    전략 C: 추세 추종 + 레버리지

    - KOSPI200 > MA(20) > MA(60) → 레버리지 매수
    - KOSPI200 < MA(20) < MA(60) → 인버스2X 매수
    """
    kospi = data['kospi200']['close']
    lev2x = data['lev2x']['close']
    inv2x = data['inv2x']['close']

    common_idx = kospi.index.intersection(lev2x.index).intersection(inv2x.index)
    kospi = kospi.reindex(common_idx)
    lev2x_ret = lev2x.reindex(common_idx).pct_change()
    inv2x_ret = inv2x.reindex(common_idx).pct_change()

    # 이동평균
    ma20 = kospi.rolling(20).mean()
    ma60 = kospi.rolling(60).mean()

    # 신호 (T-1 데이터로 T일 거래)
    uptrend = ((kospi > ma20) & (ma20 > ma60)).shift(1)
    downtrend = ((kospi < ma20) & (ma20 < ma60)).shift(1)

    # 포지션
    position = pd.Series(0, index=common_idx)
    position[uptrend == True] = 1
    position[downtrend == True] = -1

    # 최대 보유 기간 제한
    position = apply_max_holding(position, max_holding)

    # 수익률
    returns = pd.Series(0.0, index=common_idx)
    returns[position == 1] = lev2x_ret[position == 1]
    returns[position == -1] = inv2x_ret[position == -1]

    return returns, position


def strategy_d_momentum(data: Dict, max_holding: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    전략 D: 단기 모멘텀

    - KOSPI200 5일 수익률 > +2% → 레버리지 매수
    - KOSPI200 5일 수익률 < -2% → 인버스 매수
    """
    kospi = data['kospi200']['close']
    lev2x = data['lev2x']['close']
    inv2x = data['inv2x']['close']

    common_idx = kospi.index.intersection(lev2x.index).intersection(inv2x.index)
    kospi = kospi.reindex(common_idx)
    lev2x_ret = lev2x.reindex(common_idx).pct_change()
    inv2x_ret = inv2x.reindex(common_idx).pct_change()

    # 5일 모멘텀
    mom5 = kospi.pct_change(5)

    # 신호 (T-1 데이터로 T일 거래)
    long_signal = (mom5 > 0.02).shift(1)
    short_signal = (mom5 < -0.02).shift(1)

    # 포지션
    position = pd.Series(0, index=common_idx)
    position[long_signal == True] = 1
    position[short_signal == True] = -1

    # 최대 보유 기간 제한
    position = apply_max_holding(position, max_holding)

    # 수익률
    returns = pd.Series(0.0, index=common_idx)
    returns[position == 1] = lev2x_ret[position == 1]
    returns[position == -1] = inv2x_ret[position == -1]

    return returns, position


def strategy_e_hedge(data: Dict) -> Tuple[pd.Series, pd.Series]:
    """
    전략 E: 현물 + 인버스 헷지

    - 기본: 현물 ETF 100% 보유
    - VIX > SMA(20): 인버스 30% 추가 (헷지)
    - 비율: 현물 70% + 인버스 30%
    """
    vix = data['vix']['close']
    spot = data['spot']['close']
    inv1x = data['inv1x']['close']

    common_idx = vix.index.intersection(spot.index).intersection(inv1x.index)
    vix = vix.reindex(common_idx)
    spot_ret = spot.reindex(common_idx).pct_change()
    inv1x_ret = inv1x.reindex(common_idx).pct_change()

    # VIX 지표
    vix_sma20 = vix.rolling(20).mean()

    # 헷지 신호 (T-1 데이터로 T일 거래)
    hedge_on = (vix > vix_sma20).shift(1)

    # 포트폴리오 수익률
    # 헷지 ON: 현물 70% + 인버스 30%
    # 헷지 OFF: 현물 100%
    returns = pd.Series(0.0, index=common_idx)
    returns[hedge_on == True] = 0.7 * spot_ret[hedge_on == True] + 0.3 * inv1x_ret[hedge_on == True]
    returns[hedge_on == False] = spot_ret[hedge_on == False]

    # 포지션 (헷지 여부)
    position = pd.Series(1, index=common_idx)  # 항상 투자 중

    return returns, position


def apply_max_holding(position: pd.Series, max_days: int) -> pd.Series:
    """최대 보유 기간 적용."""
    result = position.copy()

    holding_count = 0
    current_pos = 0

    for i, (idx, pos) in enumerate(position.items()):
        if pos != 0 and pos == current_pos:
            holding_count += 1
            if holding_count > max_days:
                result.iloc[i] = 0
        else:
            holding_count = 1 if pos != 0 else 0
            current_pos = pos

    return result


def walk_forward_validation(
    strategy_func,
    data: Dict,
    train_years: int = 2,
    test_months: int = 6,
) -> Dict:
    """Walk-Forward 검증."""
    # 데이터 기간 확인
    common_dates = data['lev2x'].index.intersection(data['inv2x'].index)
    common_dates = common_dates.intersection(data['vix'].index)
    if 'kospi200' in data:
        common_dates = common_dates.intersection(data['kospi200'].index)

    start_date = common_dates.min()
    end_date = common_dates.max()

    train_days = train_years * 252
    test_days = test_months * 21  # 약 6개월

    folds = []
    fold_start = start_date
    fold_num = 0

    while True:
        # 충분한 데이터가 있는지 확인
        remaining = (end_date - fold_start).days
        if remaining < train_days + test_days:
            break

        # Train/Test 분할
        train_end = fold_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)

        if test_end > end_date:
            break

        fold_num += 1

        # 전략 실행
        try:
            returns, position = strategy_func(data)

            # Test 구간 성과
            test_mask = (returns.index > train_end) & (returns.index <= test_end)
            test_returns = returns[test_mask]
            test_position = position[test_mask]

            if len(test_returns) < 50:
                fold_start = train_end
                continue

            # 성과 계산
            strat_ret = (test_position.shift(1) * test_returns).fillna(0)
            turnover = test_position.diff().abs().fillna(0)
            costs = turnover * ROUND_TRIP_COST / 2
            strat_ret = strat_ret - costs

            sharpe = (strat_ret.mean() * 252) / (strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0

            folds.append({
                'fold': fold_num,
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'test_sharpe': sharpe,
                'test_return': (1 + strat_ret).prod() - 1,
            })
        except Exception as e:
            print(f'  Fold {fold_num} error: {e}')

        fold_start = train_end

    if len(folds) == 0:
        return {'folds': [], 'avg_test_sharpe': 0, 'consistency': 0}

    # 요약
    sharpes = [f['test_sharpe'] for f in folds]
    avg_sharpe = np.mean(sharpes)
    consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)

    return {
        'folds': folds,
        'avg_test_sharpe': avg_sharpe,
        'consistency': consistency,
        'n_folds': len(folds),
    }


def main():
    print('=' * 80)
    print('코스피 선물 ETF 알고리즘 트레이딩 전략 백테스트')
    print('=' * 80)
    print()

    # 데이터 로드
    print('[1] 데이터 로드')
    data = load_data()
    print()

    if 'lev2x' not in data or 'inv2x' not in data:
        print('[ERROR] 필수 데이터 누락')
        return

    # 전략 정의
    strategies = {
        'A_VIX_Switching': (strategy_a_vix_switching, {'max_holding': 20}),
        'B_Volatility_Breakout': (strategy_b_volatility_breakout, {'max_holding': 5}),
        'C_Trend_Following': (strategy_c_trend_following, {'max_holding': 20}),
        'D_Short_Momentum': (strategy_d_momentum, {'max_holding': 5}),
        'E_Spot_Hedge': (strategy_e_hedge, {}),
    }

    results = []

    print('[2] 전략 테스트')
    print()

    for strat_name, (strat_func, kwargs) in strategies.items():
        print(f'--- {strat_name} ---')

        try:
            # Full sample 백테스트
            if kwargs:
                returns, position = strat_func(data, **kwargs)
            else:
                returns, position = strat_func(data)

            metrics = calculate_metrics(returns, position)

            if metrics is None:
                print('  [SKIP] 데이터 부족')
                continue

            print(f'  CAGR: {metrics["cagr"]*100:.1f}%')
            print(f'  Sharpe: {metrics["sharpe"]:.3f}')
            print(f'  MDD: {metrics["mdd"]*100:.1f}%')
            print(f'  평균 보유: {metrics["avg_holding_days"]:.1f}일')
            print(f'  거래/년: {metrics["trades_per_year"]:.1f}회')
            print(f'  승률: {metrics["win_rate"]*100:.1f}%')
            print(f'  Exposure: {metrics["exposure"]*100:.1f}%')

            # Walk-Forward 검증
            print('  Walk-Forward 검증 중...')
            wf = walk_forward_validation(strat_func, data)
            print(f'  WF Test Sharpe: {wf["avg_test_sharpe"]:.3f}')
            print(f'  WF Consistency: {wf["consistency"]*100:.1f}% ({wf.get("n_folds", 0)} folds)')

            # 통과 여부 판정
            passed = (
                wf['avg_test_sharpe'] > 0.5 and
                wf['consistency'] > 0.5 and
                metrics['mdd'] > -0.40
            )

            verdict = 'PASS' if passed else 'FAIL'
            print(f'  결론: {verdict}')

            results.append({
                'strategy': strat_name,
                'cagr': metrics['cagr'],
                'sharpe': metrics['sharpe'],
                'mdd': metrics['mdd'],
                'avg_holding_days': metrics['avg_holding_days'],
                'trades_per_year': metrics['trades_per_year'],
                'win_rate': metrics['win_rate'],
                'exposure': metrics['exposure'],
                'wf_test_sharpe': wf['avg_test_sharpe'],
                'wf_consistency': wf['consistency'],
                'wf_folds': wf.get('n_folds', 0),
                'verdict': verdict,
            })

        except Exception as e:
            print(f'  [ERROR] {e}')
            import traceback
            traceback.print_exc()

        print()

    # 결과 요약
    print('=' * 80)
    print('[3] 결과 요약')
    print('=' * 80)
    print()

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('sharpe', ascending=False)

        print(df.to_string(index=False))
        print()

        # PASS 전략
        passed = df[df['verdict'] == 'PASS']
        if len(passed) > 0:
            print(f'통과 전략: {len(passed)}개')
            for _, row in passed.iterrows():
                print(f'  - {row["strategy"]}: Sharpe {row["sharpe"]:.3f}, WF {row["wf_test_sharpe"]:.3f}')
        else:
            print('통과 전략 없음')

        # 저장
        output = {
            'generated': datetime.now().isoformat(),
            'type': 'leveraged_futures_etf_backtest',
            'cost': ROUND_TRIP_COST,
            'results': results,
        }

        output_path = f'{OUTPUT_DIR}/leveraged_futures_etf_backtest.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        print(f'\n결과 저장: {output_path}')

    print()
    print('완료')


if __name__ == '__main__':
    main()
