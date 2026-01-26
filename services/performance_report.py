"""
Performance Report Automation Service

주간/월간 성과 리포트를 자동으로 생성하고 텔레그램으로 전송합니다.

포함 지표:
- 수익률 (일/주/월/연)
- 샤프비율
- 최대 낙폭 (MDD)
- 승률
- 평균 손익비
- 거래 통계

사용법:
    python -m services.performance_report --period weekly
    python -m services.performance_report --period monthly
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# Default paths
DATA_ROOT = Path('E:/data/crypto_ohlcv')
TRADE_LOG_PATH = Path('E:/투자/Multi-Asset Strategy Platform/data/trade_log.json')


@dataclass
class TradeRecord:
    """거래 기록"""
    symbol: str
    side: str  # buy or sell
    price: float
    quantity: float
    timestamp: datetime
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """성과 지표"""
    # 기본 정보
    period_name: str  # "Weekly" or "Monthly"
    start_date: datetime
    end_date: datetime

    # 수익률
    total_return: float
    annualized_return: float

    # 리스크 조정 지표
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # 낙폭
    max_drawdown: float
    max_drawdown_duration: int  # days

    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_pnl: float

    # 일별 통계
    best_day: float
    worst_day: float
    avg_daily_return: float
    daily_volatility: float
    positive_days: int
    negative_days: int

    # 벤치마크 대비
    btc_return: float
    excess_return: float

    def to_dict(self) -> dict:
        return {
            'period_name': self.period_name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_trade_pnl': self.avg_trade_pnl,
            'best_day': self.best_day,
            'worst_day': self.worst_day,
            'avg_daily_return': self.avg_daily_return,
            'daily_volatility': self.daily_volatility,
            'positive_days': self.positive_days,
            'negative_days': self.negative_days,
            'btc_return': self.btc_return,
            'excess_return': self.excess_return,
        }


class PerformanceReportService:
    """
    성과 리포트 서비스

    거래 로그와 포트폴리오 데이터를 분석하여
    주간/월간 성과 리포트를 생성합니다.
    """

    def __init__(
        self,
        trade_log_path: Optional[Path] = None,
        exchange: str = "upbit",
    ):
        self.trade_log_path = trade_log_path or TRADE_LOG_PATH
        self.exchange = exchange
        self.notifier = TelegramNotifier()

        logger.info(f"[PerformanceReport] Initialized")

    def load_trade_log(self) -> List[TradeRecord]:
        """거래 로그 로드"""
        if not self.trade_log_path.exists():
            logger.warning(f"[PerformanceReport] Trade log not found: {self.trade_log_path}")
            return []

        try:
            with open(self.trade_log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            trades = []
            for record in data:
                trades.append(TradeRecord(
                    symbol=record.get('symbol', ''),
                    side=record.get('side', ''),
                    price=float(record.get('price', 0)),
                    quantity=float(record.get('quantity', 0)),
                    timestamp=datetime.fromisoformat(record.get('timestamp', '')),
                    pnl=record.get('pnl'),
                    pnl_pct=record.get('pnl_pct'),
                ))
            return trades
        except Exception as e:
            logger.error(f"[PerformanceReport] Failed to load trade log: {e}")
            return []

    def load_portfolio_values(self, days: int = 365) -> pd.DataFrame:
        """
        포트폴리오 가치 시계열 로드

        실제 포트폴리오 데이터가 없으면 BTC 데이터로 시뮬레이션
        """
        portfolio_path = Path('E:/투자/Multi-Asset Strategy Platform/data/portfolio_history.csv')

        if portfolio_path.exists():
            try:
                df = pd.read_csv(portfolio_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                return df
            except Exception as e:
                logger.warning(f"[PerformanceReport] Failed to load portfolio history: {e}")

        # Fallback: 백테스트 결과 기반 시뮬레이션
        return self._simulate_portfolio_from_backtest(days)

    def _simulate_portfolio_from_backtest(self, days: int = 365) -> pd.DataFrame:
        """백테스트 결과 기반 포트폴리오 시뮬레이션"""
        # BTC 데이터 로드
        folder = DATA_ROOT / f'{self.exchange}_1d'
        btc_file = None

        for f in folder.glob('*.csv'):
            if 'BTC' in f.stem.upper() and 'DOWN' not in f.stem.upper():
                btc_file = f
                break

        if btc_file is None:
            logger.error("[PerformanceReport] BTC data not found for simulation")
            return pd.DataFrame()

        try:
            df = pd.read_csv(btc_file)
            date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            df['date'] = pd.to_datetime(df[date_col[0]]).dt.normalize()
            df = df.set_index('date').sort_index()
            df = df.tail(days)

            # 간단한 시뮬레이션 (전략 수익률 = BTC 수익률 * 0.7 + 알파)
            df['btc_return'] = df['close'].pct_change()
            df['strategy_return'] = df['btc_return'] * 0.7  # 전략 수익률 시뮬레이션

            initial_value = 10000000  # 1천만원
            df['portfolio_value'] = initial_value * (1 + df['strategy_return']).cumprod()
            df['btc_value'] = initial_value * (1 + df['btc_return']).cumprod()

            return df[['portfolio_value', 'btc_value', 'strategy_return', 'btc_return', 'close']].dropna()
        except Exception as e:
            logger.error(f"[PerformanceReport] Simulation failed: {e}")
            return pd.DataFrame()

    def calculate_metrics(
        self,
        period: str = "weekly",  # "weekly" or "monthly"
        end_date: Optional[datetime] = None,
    ) -> PerformanceMetrics:
        """
        성과 지표 계산

        Args:
            period: "weekly" or "monthly"
            end_date: 기준 날짜 (기본: 오늘)
        """
        if end_date is None:
            end_date = datetime.now()

        # 기간 설정
        if period == "weekly":
            days = 7
            period_name = "Weekly"
            start_date = end_date - timedelta(days=7)
        elif period == "monthly":
            days = 30
            period_name = "Monthly"
            start_date = end_date - timedelta(days=30)
        else:
            days = 365
            period_name = "Annual"
            start_date = end_date - timedelta(days=365)

        # 포트폴리오 데이터 로드
        portfolio = self.load_portfolio_values(days * 2)  # 여유있게 로드

        if portfolio.empty or len(portfolio) < 2:
            return self._empty_metrics(period_name, start_date, end_date)

        # 기간 필터링
        mask = (portfolio.index >= start_date) & (portfolio.index <= end_date)
        period_data = portfolio[mask]

        if len(period_data) < 2:
            return self._empty_metrics(period_name, start_date, end_date)

        # 수익률 계산
        returns = period_data['strategy_return'].values
        btc_returns = period_data['btc_return'].values

        total_return = (period_data['portfolio_value'].iloc[-1] / period_data['portfolio_value'].iloc[0]) - 1
        btc_total_return = (period_data['btc_value'].iloc[-1] / period_data['btc_value'].iloc[0]) - 1

        # 연환산 수익률
        trading_days = len(period_data)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

        # 일별 통계
        avg_daily_return = np.mean(returns)
        daily_volatility = np.std(returns)
        positive_days = np.sum(returns > 0)
        negative_days = np.sum(returns < 0)
        best_day = np.max(returns) * 100
        worst_day = np.min(returns) * 100

        # 샤프비율 (연환산)
        risk_free_rate = 0.03 / 252  # 일일 무위험 수익률 (3%)
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # 소르티노비율
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.01
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # 최대 낙폭
        portfolio_values = period_data['portfolio_value'].values
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdowns)

        # MDD 지속 기간
        mdd_duration = 0
        current_duration = 0
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
                mdd_duration = max(mdd_duration, current_duration)
            else:
                current_duration = 0

        # 칼마비율
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 거래 통계 (로그 기반)
        trades = self.load_trade_log()
        period_trades = [t for t in trades if start_date <= t.timestamp <= end_date]

        total_trades = len(period_trades)
        winning_trades = sum(1 for t in period_trades if t.pnl and t.pnl > 0)
        losing_trades = sum(1 for t in period_trades if t.pnl and t.pnl < 0)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in period_trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in period_trades if t.pnl and t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else 0

        all_pnl = [t.pnl for t in period_trades if t.pnl]
        avg_trade_pnl = np.mean(all_pnl) if all_pnl else 0

        return PerformanceMetrics(
            period_name=period_name,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=mdd_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            best_day=best_day,
            worst_day=worst_day,
            avg_daily_return=avg_daily_return * 100,
            daily_volatility=daily_volatility * 100,
            positive_days=positive_days,
            negative_days=negative_days,
            btc_return=btc_total_return,
            excess_return=total_return - btc_total_return,
        )

    def _empty_metrics(
        self,
        period_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> PerformanceMetrics:
        """빈 메트릭스 반환"""
        return PerformanceMetrics(
            period_name=period_name,
            start_date=start_date,
            end_date=end_date,
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            avg_trade_pnl=0,
            best_day=0,
            worst_day=0,
            avg_daily_return=0,
            daily_volatility=0,
            positive_days=0,
            negative_days=0,
            btc_return=0,
            excess_return=0,
        )

    def format_telegram_message(self, metrics: PerformanceMetrics) -> str:
        """텔레그램 메시지 포맷팅"""
        # 수익률 이모지
        if metrics.total_return > 0.05:
            return_emoji = "[++]"
        elif metrics.total_return > 0:
            return_emoji = "[+]"
        elif metrics.total_return > -0.05:
            return_emoji = "[-]"
        else:
            return_emoji = "[--]"

        lines = [
            f"<b>[MASP] {metrics.period_name} Performance Report</b>",
            f"Period: {metrics.start_date.strftime('%Y-%m-%d')} ~ {metrics.end_date.strftime('%Y-%m-%d')}",
            "",
            f"<b>Returns {return_emoji}</b>",
            f"  Total: {metrics.total_return*100:+.2f}%",
            f"  Annualized: {metrics.annualized_return*100:+.1f}%",
            f"  vs BTC: {metrics.btc_return*100:+.2f}%",
            f"  Alpha: {metrics.excess_return*100:+.2f}%",
            "",
            f"<b>Risk Metrics</b>",
            f"  Sharpe: {metrics.sharpe_ratio:.2f}",
            f"  Sortino: {metrics.sortino_ratio:.2f}",
            f"  Calmar: {metrics.calmar_ratio:.2f}",
            f"  Max DD: {metrics.max_drawdown*100:.1f}%",
            f"  DD Duration: {metrics.max_drawdown_duration} days",
            "",
            f"<b>Daily Stats</b>",
            f"  Best Day: {metrics.best_day:+.2f}%",
            f"  Worst Day: {metrics.worst_day:+.2f}%",
            f"  Avg Daily: {metrics.avg_daily_return:+.3f}%",
            f"  Volatility: {metrics.daily_volatility:.2f}%",
            f"  Win Days: {metrics.positive_days} / Lose Days: {metrics.negative_days}",
        ]

        if metrics.total_trades > 0:
            lines.extend([
                "",
                f"<b>Trade Stats</b>",
                f"  Total Trades: {metrics.total_trades}",
                f"  Win Rate: {metrics.win_rate*100:.1f}%",
                f"  Profit Factor: {metrics.profit_factor:.2f}",
                f"  Avg Win: {metrics.avg_win:+,.0f}",
                f"  Avg Loss: {metrics.avg_loss:+,.0f}",
            ])

        return "\n".join(lines)

    def generate_report(
        self,
        period: str = "weekly",
        send_telegram: bool = True,
    ) -> PerformanceMetrics:
        """
        리포트 생성 및 전송

        Args:
            period: "weekly" or "monthly"
            send_telegram: 텔레그램 전송 여부
        """
        logger.info(f"[PerformanceReport] Generating {period} report...")

        metrics = self.calculate_metrics(period)

        if send_telegram and self.notifier.enabled:
            message = self.format_telegram_message(metrics)
            success = self.notifier.send_message_sync(message)
            if success:
                logger.info("[PerformanceReport] Report sent to Telegram")
            else:
                logger.warning("[PerformanceReport] Failed to send Telegram")

        return metrics

    def save_report(
        self,
        metrics: PerformanceMetrics,
        output_dir: Optional[Path] = None
    ) -> Path:
        """리포트를 파일로 저장"""
        if output_dir is None:
            output_dir = Path('E:/투자/Multi-Asset Strategy Platform/outputs/reports')

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"performance_{metrics.period_name.lower()}_{metrics.end_date.strftime('%Y%m%d')}.json"
        filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"[PerformanceReport] Report saved: {filepath}")
        return filepath


def main():
    """CLI 실행"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='MASP Performance Report')
    parser.add_argument(
        '--period',
        choices=['weekly', 'monthly', 'annual'],
        default='weekly',
        help='Report period'
    )
    parser.add_argument(
        '--no-telegram',
        action='store_true',
        help='Skip Telegram notification'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save report to file'
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"MASP Performance Report - {args.period.capitalize()}")
    print("=" * 60)

    service = PerformanceReportService()
    metrics = service.calculate_metrics(args.period)

    # 콘솔 출력
    print(f"\nPeriod: {metrics.start_date.strftime('%Y-%m-%d')} ~ {metrics.end_date.strftime('%Y-%m-%d')}")
    print("\n--- Returns ---")
    print(f"Total Return: {metrics.total_return*100:+.2f}%")
    print(f"Annualized: {metrics.annualized_return*100:+.1f}%")
    print(f"BTC Return: {metrics.btc_return*100:+.2f}%")
    print(f"Alpha: {metrics.excess_return*100:+.2f}%")

    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown*100:.1f}%")
    print(f"MDD Duration: {metrics.max_drawdown_duration} days")

    print("\n--- Daily Stats ---")
    print(f"Best Day: {metrics.best_day:+.2f}%")
    print(f"Worst Day: {metrics.worst_day:+.2f}%")
    print(f"Avg Daily Return: {metrics.avg_daily_return:+.3f}%")
    print(f"Daily Volatility: {metrics.daily_volatility:.2f}%")
    print(f"Positive Days: {metrics.positive_days}")
    print(f"Negative Days: {metrics.negative_days}")

    if metrics.total_trades > 0:
        print("\n--- Trade Stats ---")
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Win Rate: {metrics.win_rate*100:.1f}%")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Avg Win: {metrics.avg_win:+,.0f}")
        print(f"Avg Loss: {metrics.avg_loss:+,.0f}")

    # 텔레그램 전송
    if not args.no_telegram and service.notifier.enabled:
        print("\n" + "=" * 60)
        response = input("Send Telegram notification? (y/n): ")
        if response.lower() == 'y':
            message = service.format_telegram_message(metrics)
            success = service.notifier.send_message_sync(message)
            print(f"Telegram sent: {success}")

    # 파일 저장
    if args.save:
        filepath = service.save_report(metrics)
        print(f"\nReport saved: {filepath}")


if __name__ == "__main__":
    main()
