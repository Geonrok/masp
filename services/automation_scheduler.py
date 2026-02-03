"""
Automation Scheduler

일간/주간 자동화 작업 스케줄러:
- 일간 시그널 알림 (매일 오전 9시)
- 시장 국면 분석 (매일 오전 9시)
- 주간 성과 리포트 (매주 일요일 오전 10시)
- 월간 성과 리포트 (매월 1일 오전 10시)
- 리스크 모니터링 (매 시간)

사용법:
    python -m services.automation_scheduler
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.daily_signal_alert import DailySignalAlertService
from services.performance_report import PerformanceReportService
from services.risk_management_service import RiskManagementService, RiskConfig
from libs.analysis.market_regime import MarketRegimeDetector
from libs.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)


class AutomationScheduler:
    """
    MASP 자동화 스케줄러

    APScheduler를 사용하여 일간/주간 작업을 스케줄링합니다.
    """

    def __init__(
        self,
        timezone: str = "Asia/Seoul",
        initial_capital: float = 10_000_000,
    ):
        self.timezone = timezone
        self.scheduler = AsyncIOScheduler(timezone=timezone)

        # 서비스 초기화
        self.signal_service = DailySignalAlertService()
        self.report_service = PerformanceReportService()
        self.risk_service = RiskManagementService(
            config=RiskConfig(),
            initial_capital=initial_capital,
        )
        self.regime_detector = MarketRegimeDetector()
        self.notifier = TelegramNotifier()

        self._running = False
        self._setup_jobs()

        logger.info("[AutomationScheduler] Initialized")

    def _setup_jobs(self):
        """스케줄 작업 설정"""
        # 일간 시그널 알림 (매일 오전 9:00)
        self.scheduler.add_job(
            self._job_daily_signal,
            trigger=CronTrigger(hour=9, minute=0, timezone=self.timezone),
            id="daily_signal",
            name="Daily Signal Alert",
            max_instances=1,
            coalesce=True,
        )

        # 시장 국면 분석 (매일 오전 9:05)
        self.scheduler.add_job(
            self._job_market_regime,
            trigger=CronTrigger(hour=9, minute=5, timezone=self.timezone),
            id="market_regime",
            name="Market Regime Analysis",
            max_instances=1,
            coalesce=True,
        )

        # 리스크 모니터링 (매 시간 정각)
        self.scheduler.add_job(
            self._job_risk_monitor,
            trigger=CronTrigger(minute=0, timezone=self.timezone),
            id="risk_monitor",
            name="Risk Monitor",
            max_instances=1,
            coalesce=True,
        )

        # 주간 리포트 (매주 일요일 오전 10:00)
        self.scheduler.add_job(
            self._job_weekly_report,
            trigger=CronTrigger(
                day_of_week=6, hour=10, minute=0, timezone=self.timezone
            ),
            id="weekly_report",
            name="Weekly Performance Report",
            max_instances=1,
            coalesce=True,
        )

        # 월간 리포트 (매월 1일 오전 10:00)
        self.scheduler.add_job(
            self._job_monthly_report,
            trigger=CronTrigger(day=1, hour=10, minute=0, timezone=self.timezone),
            id="monthly_report",
            name="Monthly Performance Report",
            max_instances=1,
            coalesce=True,
        )

        logger.info("[AutomationScheduler] Jobs configured")

    async def _job_daily_signal(self):
        """일간 시그널 알림 작업"""
        logger.info("[Job] Running daily signal alert")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.signal_service.send_daily_alert)
            logger.info("[Job] Daily signal alert completed")
        except Exception as e:
            logger.error(f"[Job] Daily signal alert failed: {e}")

    async def _job_market_regime(self):
        """시장 국면 분석 작업"""
        logger.info("[Job] Running market regime analysis")
        try:
            loop = asyncio.get_running_loop()
            analysis = await loop.run_in_executor(None, self.regime_detector.analyze)

            if self.notifier.enabled:
                message = self.regime_detector.format_telegram_message(analysis)
                await loop.run_in_executor(
                    None, self.notifier.send_message_sync, message
                )
            logger.info(f"[Job] Market regime: {analysis.regime.value}")
        except Exception as e:
            logger.error(f"[Job] Market regime analysis failed: {e}")

    async def _job_risk_monitor(self):
        """리스크 모니터링 작업"""
        logger.info("[Job] Running risk monitor")
        try:
            loop = asyncio.get_running_loop()
            alert = await loop.run_in_executor(None, self.risk_service.monitor)

            if alert:
                logger.warning(f"[Job] Risk alert: {alert.level.value}")
            else:
                logger.debug("[Job] Risk monitor: No alerts")
        except Exception as e:
            logger.error(f"[Job] Risk monitor failed: {e}")

    async def _job_weekly_report(self):
        """주간 리포트 작업"""
        logger.info("[Job] Running weekly report")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self.report_service.generate_report, "weekly", True
            )
            logger.info("[Job] Weekly report completed")
        except Exception as e:
            logger.error(f"[Job] Weekly report failed: {e}")

    async def _job_monthly_report(self):
        """월간 리포트 작업"""
        logger.info("[Job] Running monthly report")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self.report_service.generate_report, "monthly", True
            )
            logger.info("[Job] Monthly report completed")
        except Exception as e:
            logger.error(f"[Job] Monthly report failed: {e}")

    def run_now(self, job_name: str):
        """특정 작업 즉시 실행"""
        job_map = {
            "daily_signal": self.signal_service.send_daily_alert,
            "market_regime": lambda: self.regime_detector.analyze(),
            "risk_monitor": self.risk_service.monitor,
            "weekly_report": lambda: self.report_service.generate_report(
                "weekly", True
            ),
            "monthly_report": lambda: self.report_service.generate_report(
                "monthly", True
            ),
        }

        if job_name not in job_map:
            logger.error(f"[Scheduler] Unknown job: {job_name}")
            return False

        logger.info(f"[Scheduler] Running job now: {job_name}")
        try:
            result = job_map[job_name]()
            logger.info(f"[Scheduler] Job {job_name} completed")
            return result
        except Exception as e:
            logger.error(f"[Scheduler] Job {job_name} failed: {e}")
            return None

    async def run_forever(self):
        """스케줄러 실행"""
        self._running = True

        # 시그널 핸들러 설정
        def _handler(signum, frame):
            logger.info(f"[Scheduler] Received signal {signum}, stopping...")
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

        self.scheduler.start()
        logger.info("[Scheduler] Started")

        # 시작 알림 전송
        if self.notifier.enabled:
            self.notifier.send_message_sync(
                "<b>[MASP] Automation Scheduler Started</b>\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                "Scheduled Jobs:\n"
                "- Daily Signal: 09:00\n"
                "- Market Regime: 09:05\n"
                "- Risk Monitor: Every hour\n"
                "- Weekly Report: Sunday 10:00\n"
                "- Monthly Report: 1st day 10:00"
            )

        try:
            while self._running:
                await asyncio.sleep(1)
        finally:
            self.scheduler.shutdown(wait=True)
            logger.info("[Scheduler] Stopped")

    def stop(self):
        """스케줄러 중지"""
        self._running = False

    def get_job_status(self) -> list:
        """작업 상태 조회"""
        jobs = []
        for job in self.scheduler.get_jobs():
            try:
                next_run = job.next_run_time.isoformat() if job.next_run_time else None
            except (AttributeError, TypeError):
                next_run = None
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": next_run,
                    "trigger": str(job.trigger),
                }
            )
        return jobs


def main():
    """CLI 실행"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="MASP Automation Scheduler")
    parser.add_argument(
        "--run-now",
        choices=[
            "daily_signal",
            "market_regime",
            "risk_monitor",
            "weekly_report",
            "monthly_report",
        ],
        help="Run specific job immediately",
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run as daemon (scheduler mode)"
    )

    args = parser.parse_args()

    scheduler = AutomationScheduler()

    if args.run_now:
        print(f"Running job: {args.run_now}")
        result = scheduler.run_now(args.run_now)
        print(f"Result: {result}")
    elif args.daemon:
        print("=" * 60)
        print("MASP Automation Scheduler")
        print("=" * 60)
        print("\nScheduled Jobs:")
        for job in scheduler.get_job_status():
            print(f"  - {job['name']}: {job['trigger']}")
        print("\nPress Ctrl+C to stop")
        asyncio.run(scheduler.run_forever())
    else:
        print("=" * 60)
        print("MASP Automation Scheduler")
        print("=" * 60)
        print("\nUsage:")
        print("  --daemon           Run as background scheduler")
        print("  --run-now <job>    Run specific job immediately")
        print("\nAvailable jobs:")
        print("  daily_signal       Daily signal alert")
        print("  market_regime      Market regime analysis")
        print("  risk_monitor       Risk monitoring")
        print("  weekly_report      Weekly performance report")
        print("  monthly_report     Monthly performance report")


if __name__ == "__main__":
    main()
