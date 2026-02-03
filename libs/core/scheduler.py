"""
Scheduler - Handles timed execution of strategy runs.
Supports interval-based and cron-based scheduling with KST timezone.
"""

import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pytz
from croniter import croniter
from filelock import FileLock, Timeout

from libs.core.config import Config


class Scheduler:
    """
    Local scheduler for strategy execution.

    Features:
    - Interval-based scheduling (every N seconds)
    - Cron-based scheduling
    - KST timezone support
    - File-based locking to prevent concurrent execution
    - Graceful shutdown handling
    """

    def __init__(
        self,
        config: Config,
        run_callback: Callable[[], None],
        heartbeat_callback: Optional[Callable[[], None]] = None,
        heartbeat_interval: int = 30,
    ):
        """
        Initialize scheduler.

        Args:
            config: Service configuration
            run_callback: Function to call for each run
            heartbeat_callback: Function to call for heartbeats
            heartbeat_interval: Seconds between heartbeats
        """
        self.config = config
        self.run_callback = run_callback
        self.heartbeat_callback = heartbeat_callback
        self.heartbeat_interval = heartbeat_interval

        self._running = False
        self._shutdown_requested = False
        self._kst = pytz.timezone(config.schedule.timezone)

        # Lock file path
        lock_dir = Path("storage/locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = lock_dir / f"{config.asset_class.value}.lock"
        self._lock: Optional[FileLock] = None

        # Start time for uptime tracking
        self._start_time: Optional[float] = None

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""

        def handler(signum, frame):
            print(f"\n[Scheduler] Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True

        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        # On Windows, also handle SIGBREAK if available
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handler)

    def _acquire_lock(self) -> bool:
        """
        Acquire run lock to prevent concurrent execution.
        Returns True if lock acquired, False otherwise.
        """
        if not self.config.run_lock:
            return True

        try:
            self._lock = FileLock(str(self._lock_path), timeout=0)
            self._lock.acquire()
            return True
        except Timeout:
            print(
                f"[Scheduler] Another instance is running (lock file: {self._lock_path})"
            )
            return False

    def _release_lock(self) -> None:
        """Release run lock."""
        if self._lock and self._lock.is_locked:
            self._lock.release()

    def _now_kst(self) -> datetime:
        """Get current time in KST."""
        return datetime.now(self._kst)

    def _get_next_cron_time(self, cron_expr: str) -> datetime:
        """Get next scheduled time based on cron expression."""
        now = self._now_kst()
        cron = croniter(cron_expr, now)
        return cron.get_next(datetime)

    def _safe_run(self) -> None:
        """Execute run callback with exception handling."""
        try:
            self.run_callback()
        except Exception as e:
            print(f"[Scheduler] Run failed with error: {e}")
            traceback.print_exc()

    def _safe_heartbeat(self) -> None:
        """Execute heartbeat callback with exception handling."""
        if self.heartbeat_callback:
            try:
                self.heartbeat_callback()
            except Exception as e:
                print(f"[Scheduler] Heartbeat failed with error: {e}")

    @property
    def uptime_seconds(self) -> Optional[float]:
        """Get uptime in seconds since scheduler started."""
        if self._start_time:
            return time.time() - self._start_time
        return None

    def run_once(self) -> bool:
        """
        Run the strategy once and exit.

        Returns:
            True if run completed, False if lock acquisition failed
        """
        if not self._acquire_lock():
            return False

        try:
            self._start_time = time.time()

            # Emit initial heartbeat
            self._safe_heartbeat()

            # Execute run
            print(f"[Scheduler] Starting single run at {self._now_kst().isoformat()}")
            self._safe_run()
            print(f"[Scheduler] Run completed at {self._now_kst().isoformat()}")

            return True
        finally:
            self._release_lock()

    def run_daemon(self) -> None:
        """
        Run the scheduler in daemon mode.
        Continues until shutdown signal received.
        """
        if not self._acquire_lock():
            sys.exit(1)

        try:
            self._setup_signal_handlers()
            self._running = True
            self._start_time = time.time()

            print(
                f"[Scheduler] Starting daemon mode for {self.config.asset_class.value}"
            )
            print(f"[Scheduler] Schedule: {self.config.schedule.mode}")

            if self.config.schedule.mode == "cron":
                print(
                    f"[Scheduler] Cron expression: {self.config.schedule.cron_expression}"
                )
            else:
                print(
                    f"[Scheduler] Interval: {self.config.schedule.interval_seconds} seconds"
                )

            # Emit initial heartbeat
            self._safe_heartbeat()

            last_heartbeat = time.time()
            last_run: Optional[float] = None

            while not self._shutdown_requested:
                now = time.time()

                # Check if heartbeat is due
                if now - last_heartbeat >= self.heartbeat_interval:
                    self._safe_heartbeat()
                    last_heartbeat = now

                # Check if run is due
                should_run = False

                if self.config.schedule.mode == "cron":
                    # Cron-based scheduling
                    if self.config.schedule.cron_expression:
                        next_time = self._get_next_cron_time(
                            self.config.schedule.cron_expression
                        )
                        # Run if we're within 1 second of scheduled time
                        now_kst = self._now_kst()
                        if abs((next_time - now_kst).total_seconds()) < 1:
                            should_run = True
                else:
                    # Interval-based scheduling
                    if last_run is None:
                        should_run = True
                    elif now - last_run >= self.config.schedule.interval_seconds:
                        should_run = True

                if should_run:
                    print(
                        f"\n[Scheduler] Starting scheduled run at {self._now_kst().isoformat()}"
                    )
                    self._safe_run()
                    last_run = time.time()
                    print(f"[Scheduler] Run completed at {self._now_kst().isoformat()}")

                # Sleep briefly to avoid busy waiting
                time.sleep(0.5)

            print("\n[Scheduler] Shutdown complete")

        finally:
            self._running = False
            self._release_lock()

    def stop(self) -> None:
        """Request scheduler shutdown."""
        self._shutdown_requested = True
