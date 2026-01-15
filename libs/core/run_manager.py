"""
Run Manager - Handles run lifecycle and ID generation.
"""

from datetime import datetime, timezone
from typing import Optional
import secrets
import time

import pytz


class RunManager:
    """
    Manages run lifecycle for strategy execution.
    
    Lifecycle:
        RUN_STARTED → (0..n events) → RUN_FINISHED
    """
    
    def __init__(self, asset_class: str):
        self.asset_class = asset_class
        self._kst = pytz.timezone("Asia/Seoul")
    
    def generate_run_id(self) -> str:
        """
        Generate a stable, unique run ID.
        Format: {asset_class}_{YYYYMMDD}_{HHMMSS}_{random_suffix}
        
        This format is:
        - Human readable (includes date/time)
        - Sortable chronologically
        - Unique (random suffix prevents collisions)
        """
        now = datetime.now(self._kst)
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        suffix = secrets.token_hex(4)  # 8 hex chars
        
        return f"{self.asset_class}_{date_part}_{time_part}_{suffix}"
    
    def parse_run_id(self, run_id: str) -> dict:
        """
        Parse a run ID into its components.
        Returns dict with asset_class, date, time, suffix.
        """
        parts = run_id.split("_")
        if len(parts) >= 4:
            # Handle asset classes with underscores
            suffix = parts[-1]
            time_part = parts[-2]
            date_part = parts[-3]
            asset_class = "_".join(parts[:-3])
            
            return {
                "asset_class": asset_class,
                "date": date_part,
                "time": time_part,
                "suffix": suffix,
            }
        return {"raw": run_id}


class RunContext:
    """
    Context for a single run execution.
    Tracks timing and counters.
    """
    
    def __init__(self, run_id: str, asset_class: str):
        self.run_id = run_id
        self.asset_class = asset_class
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.decision_count: int = 0
        self.error_count: int = 0
        self.status: str = "pending"
    
    def start(self) -> None:
        """Mark run as started."""
        self.start_time = time.time()
        self.status = "running"
    
    def finish(self, success: bool = True) -> None:
        """Mark run as finished."""
        self.end_time = time.time()
        self.status = "success" if success else "fail"
    
    def increment_decisions(self, count: int = 1) -> None:
        """Increment decision counter."""
        self.decision_count += count
    
    def increment_errors(self, count: int = 1) -> None:
        """Increment error counter."""
        self.error_count += count
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get run duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None
    
    def to_dict(self) -> dict:
        """Convert context to dictionary."""
        return {
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "status": self.status,
            "decision_count": self.decision_count,
            "error_count": self.error_count,
            "duration_seconds": self.duration_seconds,
        }


