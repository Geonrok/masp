"""
MultiExchangeScheduler - 다중 거래소 스케줄러
MASP Phase 3A - Dual Exchange Support (Upbit 09:00 + Bithumb 00:00)
"""
from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from apscheduler.events import EVENT_JOB_MISSED, EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.strategy_runner import StrategyRunner

logger = logging.getLogger(__name__)


class MultiExchangeScheduler:
    """
    Multi-exchange scheduler for simultaneous Upbit/Bithumb execution.
    
    Features:
        - Upbit: 09:00 KST (일봉 리셋)
        - Bithumb: 00:00 KST (자정 리셋)
        - Independent job management per exchange
        - Graceful shutdown handling
    
    Usage:
        scheduler = MultiExchangeScheduler()
        await scheduler.run_forever()
    """

    def __init__(
        self,
        config_path: str = "config/schedule_config.json",
    ) -> None:
        self._config_path = Path(config_path)
        self._config = self._load_config()
        self._lock = asyncio.Lock()
        self._running = False
        self._signal_handlers_registered = False
        self._listener_added = False
        
        # Exchange runners and jobs
        self._runners: Dict[str, StrategyRunner] = {}
        self._jobs: Dict[str, Any] = {}
        self._triggers: Dict[str, CronTrigger] = {}
        
        # Single scheduler for all jobs
        self._scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
        
        # Initialize exchange configurations
        self._init_exchanges()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML/JSON config file."""
        if not self._config_path.exists():
            logger.warning(f"[MultiExchangeScheduler] Config not found: {self._config_path}")
            return self._default_config()

        raw = self._config_path.read_text(encoding="utf-8")

        try:
            import json
            data = json.loads(raw)
            if isinstance(data, dict):
                if "exchanges" in data:
                    return data
                return self._convert_legacy_config(data)
        except json.JSONDecodeError:
            pass

        if HAS_YAML:
            try:
                data = yaml.safe_load(raw)
                if isinstance(data, dict):
                    if "exchanges" in data:
                        return data
                    return self._convert_legacy_config(data)
            except Exception as exc:
                logger.warning(f"[MultiExchangeScheduler] YAML parse failed: {exc}")

        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default dual-exchange configuration."""
        return {
            "exchanges": {
                "upbit": {
                    "enabled": True,
                    "strategy": "KAMA-TSMOM-Gate",
                    "symbols": ["BTC/KRW"],
                    "position_size_krw": 10000,
                    "schedule": {
                        "hour": 9,
                        "minute": 0,
                        "timezone": "Asia/Seoul",
                        "jitter": 30,
                    },
                },
                "bithumb": {
                    "enabled": True,
                    "strategy": "KAMA-TSMOM-Gate",
                    "symbols": ["BTC/KRW"],
                    "position_size_krw": 10000,
                    "schedule": {
                        "hour": 0,
                        "minute": 0,
                        "timezone": "Asia/Seoul",
                        "jitter": 30,
                    },
                },
            }
        }

    def _convert_legacy_config(self, data: Dict) -> Dict[str, Any]:
        """Convert legacy single-exchange config to new format."""
        if not isinstance(data, dict):
            return self._default_config()
        
        schedule = data.get("schedule", {})
        cron = schedule.get("cron", {})
        
        return {
            "exchanges": {
                "upbit": {
                    "enabled": True,
                    "strategy": "KAMA-TSMOM-Gate",
                    "symbols": ["BTC/KRW"],
                    "position_size_krw": 10000,
                    "schedule": {
                        "hour": cron.get("hour", 9),
                        "minute": cron.get("minute", 0),
                        "timezone": cron.get("timezone", "Asia/Seoul"),
                        "jitter": schedule.get("jitter", 60),
                    },
                },
                "bithumb": {
                    "enabled": True,
                    "strategy": "KAMA-TSMOM-Gate",
                    "symbols": ["BTC/KRW"],
                    "position_size_krw": 10000,
                    "schedule": {
                        "hour": 0,
                        "minute": 0,
                        "timezone": "Asia/Seoul",
                        "jitter": 30,
                    },
                },
            }
        }

    def _init_exchanges(self) -> None:
        """Initialize runners and triggers for each enabled exchange."""
        exchanges = self._config.get("exchanges", {})
        
        for exchange_name, cfg in exchanges.items():
            if not cfg.get("enabled", False):
                logger.info(f"[MultiExchangeScheduler] {exchange_name} disabled, skipping")
                continue
            
            # Create StrategyRunner
            runner = StrategyRunner(
                strategy_name=cfg.get("strategy", "KAMA-TSMOM-Gate"),
                exchange=exchange_name,
                symbols=cfg.get("symbols", ["BTC/KRW"]),
                position_size_krw=cfg.get("position_size_krw", 10000),
            )
            self._runners[exchange_name] = runner
            
            # Create CronTrigger
            sched = cfg.get("schedule", {})
            trigger = CronTrigger(
                hour=int(sched.get("hour", 9)),
                minute=int(sched.get("minute", 0)),
                timezone=sched.get("timezone", "Asia/Seoul"),
            )
            self._triggers[exchange_name] = trigger
            init_hour = int(sched.get("hour", 9))
            init_minute = int(sched.get("minute", 0))
            logger.info(
                f"[MultiExchangeScheduler] {exchange_name.upper()} initialized: "
                f"{init_hour:02d}:{init_minute:02d} KST"
            )

    def _register_signal_handlers(self) -> None:
        """Register graceful shutdown handlers."""
        if self._signal_handlers_registered:
            return

        def _handler(signum, frame):
            logger.info(f"[MultiExchangeScheduler] Signal {signum} received, shutting down")
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError) as exc:
                logger.warning(f"[MultiExchangeScheduler] Signal {sig} registration failed: {exc}")

        if hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, _handler)
            except (ValueError, OSError) as exc:
                logger.warning(f"[MultiExchangeScheduler] SIGBREAK registration failed: {exc}")

        self._signal_handlers_registered = True

    def _configure_scheduler(self) -> None:
        """Configure scheduler with all exchange jobs."""
        if not self._listener_added:
            self._scheduler.add_listener(self._on_job_event, 
                EVENT_JOB_MISSED | EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
            self._listener_added = True

        exchanges = self._config.get("exchanges", {})
        
        for exchange_name, cfg in exchanges.items():
            if exchange_name not in self._runners:
                continue
            
            sched = cfg.get("schedule", {})
            jitter = int(sched.get("jitter", 30))
            if jitter < 0:
                jitter = 0

            job = self._scheduler.add_job(
                self._run_job,
                args=[exchange_name],
                trigger=self._triggers[exchange_name],
                id=f"job_{exchange_name}",
                name=f"{exchange_name.upper()} Daily Run",
                jitter=jitter,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300,
            )
            self._jobs[exchange_name] = job
            
            logger.info(
                f"[MultiExchangeScheduler] Job scheduled: {exchange_name.upper()} "
                f"(jitter={jitter}s)"
            )

    def _on_job_event(self, event) -> None:
        """Handle job events (missed, executed, error)."""
        job_id = getattr(event, "job_id", "unknown")
        
        if hasattr(event, "exception") and event.exception:
            logger.error(
                f"[MultiExchangeScheduler] Job {job_id} ERROR: {event.exception}"
            )
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(
                f"[MultiExchangeScheduler] Job {job_id} MISSED at "
                f"{getattr(event, 'scheduled_run_time', 'N/A')}"
            )
        else:
            logger.info(f"[MultiExchangeScheduler] Job {job_id} executed successfully")

    async def _run_job(self, exchange_name: str) -> None:
        """Execute strategy for specific exchange."""
        async with self._lock:
            runner = self._runners.get(exchange_name)
            if runner is None:
                logger.error(f"[MultiExchangeScheduler] No runner for {exchange_name}")
                return

            logger.info(f"[MultiExchangeScheduler] ▶ Running {exchange_name.upper()} strategy")
            
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, runner.run_once)
                logger.info(f"[MultiExchangeScheduler] ✅ {exchange_name.upper()} result: {result}")
            except Exception as exc:
                logger.error(f"[MultiExchangeScheduler] ❌ {exchange_name.upper()} failed: {exc}")
                raise

    def run_once(self, exchange_name: str = None) -> Dict[str, Any]:
        """
        Run strategy once (manual trigger).
        
        Args:
            exchange_name: Specific exchange or None for all enabled
            
        Returns:
            Dict of results per exchange
        """
        results = {}
        
        try:
            asyncio.get_running_loop()
            logger.error("[MultiExchangeScheduler] run_once called from running event loop")
            return {"error": "Cannot run from async context"}
        except RuntimeError:
            pass
        
        targets = [exchange_name] if exchange_name else list(self._runners.keys())
        
        for name in targets:
            runner = self._runners.get(name)
            if runner is None:
                results[name] = {"error": f"No runner for {name}"}
                continue
            
            try:
                result = runner.run_once()
                results[name] = result
            except Exception as exc:
                results[name] = {"error": str(exc)}
        
        return results

    async def run_forever(self) -> None:
        """Start scheduler and run until shutdown signal."""
        self._register_signal_handlers()
        self._configure_scheduler()
        self._running = True

        logger.info("[MultiExchangeScheduler] 🚀 Starting multi-exchange scheduler")
        logger.info(f"[MultiExchangeScheduler] Active exchanges: {list(self._runners.keys())}")

        self._scheduler.start()
        try:
            while self._running:
                await asyncio.sleep(0.5)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("[MultiExchangeScheduler] Shutting down...")
        try:
            self._scheduler.shutdown(wait=True)
        except Exception as exc:
            logger.debug(f"[MultiExchangeScheduler] Shutdown note: {exc}")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        self._shutdown()

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        status = {
            "running": self._running,
            "config_path": str(self._config_path),
            "exchanges": {},
        }

        exchanges_config = self._config.get("exchanges", {})

        for name, runner in self._runners.items():
            job = self._jobs.get(name)
            cfg = exchanges_config.get(name, {})
            sched = cfg.get("schedule", {})

            raw_hour = sched.get("hour", 9)
            raw_minute = sched.get("minute", 0)
            try:
                hour = int(raw_hour)
                minute = int(raw_minute)
                schedule_config = f"{hour:02d}:{minute:02d} KST"
            except (ValueError, TypeError):
                schedule_config = f"{raw_hour}:{raw_minute} KST"

            trigger = self._triggers.get(name)
            if trigger:
                try:
                    runtime_hour = int(str(trigger.fields[5]))
                    runtime_minute = int(str(trigger.fields[6]))
                    schedule_runtime = f"{runtime_hour:02d}:{runtime_minute:02d} KST"
                except (ValueError, TypeError, IndexError):
                    schedule_runtime = "N/A"
            else:
                schedule_runtime = "N/A"

            status["exchanges"][name] = {
                "strategy": runner.strategy_name,
                "symbols": runner.symbols,
                "position_size_krw": runner.position_size_krw,
                "schedule": schedule_config,
                "schedule_runtime": schedule_runtime,
                "next_run": str(job.next_run_time) if job else "N/A",
            }

        return status


# CLI Entry Point
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Multi-Exchange Scheduler")
    parser.add_argument("--config", default="config/schedule_config.json", help="Config path")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--exchange", default=None, help="Specific exchange for --once")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    
    args = parser.parse_args()
    
    scheduler = MultiExchangeScheduler(config_path=args.config)
    
    if args.status:
        import json
        print(json.dumps(scheduler.get_status(), indent=2, default=str))
    elif args.once:
        result = scheduler.run_once(exchange_name=args.exchange)
        print(f"Result: {result}")
    else:
        asyncio.run(scheduler.run_forever())
