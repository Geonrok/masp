"""
Daily Report 자동 생성 스크립트
- 모든 거래소의 일일 리포트 생성
- Cron/Task Scheduler로 자동화 가능
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_reports(report_date: date = None) -> dict:
    """모든 거래소 리포트 생성"""

    from libs.adapters.trade_logger import TradeLogger
    from libs.analytics.daily_report import DailyReportGenerator
    from libs.analytics.strategy_health import StrategyHealthMonitor

    d = report_date or date.today()
    results = {}

    # 거래소별 로그 디렉토리
    exchanges = {
        "paper": "logs/paper_trades",
        "upbit": "logs/upbit_trades",
        "bithumb": "logs/bithumb_trades",
        "live": "logs/live_trades",
    }

    for exchange, base_dir in exchanges.items():
        trades_dir = Path(base_dir) / "trades"
        reports_dir = Path(base_dir) / "reports"

        if not trades_dir.exists():
            continue

        try:
            logger = TradeLogger(log_dir=str(trades_dir))
            health = StrategyHealthMonitor()
            reporter = DailyReportGenerator(logger, health, report_dir=str(reports_dir))

            # 거래 확인
            trade_count = logger.get_trade_count(d)

            if trade_count > 0:
                reporter.generate(d)
                results[exchange] = {
                    "status": "generated",
                    "trades": trade_count,
                    "path": str(reports_dir / f"daily_{d.strftime('%Y-%m-%d')}.md"),
                }
            else:
                results[exchange] = {
                    "status": "skipped",
                    "trades": 0,
                    "reason": "No trades",
                }

        except Exception as e:
            results[exchange] = {"status": "error", "error": str(e)}

    return results


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Daily Reports")
    parser.add_argument("--date", help="Report date (YYYY-MM-DD)")
    parser.add_argument(
        "--yesterday", action="store_true", help="Generate for yesterday"
    )

    args = parser.parse_args()

    if args.yesterday:
        report_date = date.today() - timedelta(days=1)
    elif args.date:
        report_date = date.fromisoformat(args.date)
    else:
        report_date = date.today()

    print("=" * 60)
    print("Daily Report Generator")
    print(f"Date: {report_date}")
    print("=" * 60)

    results = generate_reports(report_date)

    for exchange, result in results.items():
        status = result.get("status")
        if status == "generated":
            print(f"\n✅ {exchange}: {result['trades']} trades")
            print(f"   → {result['path']}")
        elif status == "skipped":
            print(f"\n⏭️  {exchange}: {result['reason']}")
        else:
            print(f"\n❌ {exchange}: {result.get('error', 'Unknown error')}")

    print(f"\n{'=' * 60}")
    print("Complete!")


if __name__ == "__main__":
    main()
