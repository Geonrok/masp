"""
DailyScheduler - APScheduler-based daily runner.
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
from pathlib import Path
from typing import Any, Dict, Optional

from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.strategy_runner import StrategyRunner
from libs.core.startup_validator import validate_startup, validate_api_keys

logger = logging.getLogger(__name__)


class DailyScheduler:
    """
    Daily scheduler for StrategyRunner.run_once execution.
    """

    def __init__(
        self,
        runner: StrategyRunner,
        config_path: str = "config/schedule_config.yaml",
    ) -> None:
        self._runner = runner
        self._config_path = Path(config_path)
        self._lock = asyncio.Lock()
        self._running = False
        self._signal_handlers_registered = False
        self._listener_added = False

        config = self._load_config()
        schedule = config.get("schedule", {}) if isinstance(config, dict) else {}
        cron_cfg = schedule.get("cron", {}) if isinstance(schedule, dict) else {}

        hour = int(cron_cfg.get("hour", 9))
        minute = int(cron_cfg.get("minute", 0))
        timezone = cron_cfg.get("timezone", "Asia/Seoul")
        self._jitter = int(schedule.get("jitter", 60))
        if self._jitter < 0:
            self._jitter = 0

        self._trigger = CronTrigger(hour=hour, minute=minute, timezone=timezone)
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._job = None

    @property
    def trigger(self) -> CronTrigger:
        return self._trigger

    @property
    def jitter(self) -> int:
        return self._jitter

    def _load_config(self) -> Dict[str, Any]:
        if not self._config_path.exists():
            return {}

        raw = self._config_path.read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
            return data if isinstance(data, dict) else {}
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("[DailyScheduler] Failed to parse YAML: %s", exc)

        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as exc:
            logger.warning("[DailyScheduler] Failed to parse JSON: %s", exc)
            return {}

    def _register_signal_handlers(self) -> None:
        if self._signal_handlers_registered:
            return

        def _handler(signum, frame):
            logger.info("[DailyScheduler] Received signal %s, shutting down", signum)
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "[DailyScheduler] Signal %s registration failed: %s",
                    sig,
                    exc,
                )

        if hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, _handler)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "[DailyScheduler] SIGBREAK registration failed: %s",
                    exc,
                )

        self._signal_handlers_registered = True

    def _configure_scheduler(self) -> None:
        if not self._listener_added:
            self._scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
            self._listener_added = True

        if self._job is None:
            self._job = self._scheduler.add_job(
                self._run_job,
                trigger=self._trigger,
                jitter=self._jitter,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300,
            )

    def _on_job_missed(self, event) -> None:
        logger.warning(
            "[DailyScheduler] Job missed: %s at %s",
            event.job_id,
            getattr(event, "scheduled_run_time", None),
        )

    async def _run_job(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._runner.run_once)

    def run_once(self) -> bool:
        """Run a single scheduled job synchronously."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._run_job())
            return True
        logger.error("[DailyScheduler] run_once called from a running event loop")
        return False

    async def run_forever(self, validate_keys: bool = True, strict: bool = False) -> None:
        """
        Run the scheduler forever.

        Args:
            validate_keys: If True, validate API keys for enabled exchanges at startup
            strict: If True, raise exception if validation fails
        """
        # Validate API keys at startup
        if validate_keys:
            logger.info("[DailyScheduler] Validating API keys at startup...")
            validation_result = validate_api_keys(
                schedule_config_path=str(self._config_path),
                raise_on_error=strict,
            )
            if not validation_result.is_valid:
                logger.error(
                    "[DailyScheduler] API key validation failed: %d errors",
                    len(validation_result.errors),
                )
                for err in validation_result.errors:
                    logger.error("  - %s", err)
                if strict:
                    return

            for warn in validation_result.warnings:
                logger.warning("[DailyScheduler] %s", warn)

        self._register_signal_handlers()
        self._configure_scheduler()
        self._running = True

        self._scheduler.start()
        try:
            while self._running:
                await asyncio.sleep(0.5)
        finally:
            try:
                self._scheduler.shutdown(wait=True)
            except Exception as exc:
                logger.debug("[DailyScheduler] Shutdown skipped: %s", exc)

    def stop(self) -> None:
        self._running = False
        try:
            self._scheduler.shutdown(wait=True)
        except Exception as exc:
            logger.debug("[DailyScheduler] Shutdown skipped: %s", exc)
