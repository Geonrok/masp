"""Alert history provider - connects Telegram/notification logs to alert_history component."""

from __future__ import annotations

import csv
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Alert type constants (matching component)
ALERT_TYPE_TRADE = "TRADE"
ALERT_TYPE_SIGNAL = "SIGNAL"
ALERT_TYPE_ERROR = "ERROR"
ALERT_TYPE_DAILY = "DAILY"
ALERT_TYPE_SYSTEM = "SYSTEM"

# Status constants
STATUS_SENT = "SENT"
STATUS_FAILED = "FAILED"

# Default alert log directory
DEFAULT_ALERT_LOG_DIR = "logs/alerts"


def _get_alert_file_path(
    alert_date: Optional[date] = None, log_dir: str = DEFAULT_ALERT_LOG_DIR
) -> Path:
    """Get alert log file path for a specific date.

    Args:
        alert_date: Date to get file for (default today)
        log_dir: Log directory path

    Returns:
        Path to alert log file
    """
    d = alert_date or date.today()
    log_path = Path(log_dir)
    month_dir = log_path / d.strftime("%Y-%m")
    return month_dir / f"alerts_{d.strftime('%Y-%m-%d')}.csv"


def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime.

    Args:
        timestamp_str: Timestamp string

    Returns:
        Parsed datetime
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    return datetime.now()


def _parse_alert_type(type_str: str) -> str:
    """Parse and normalize alert type.

    Args:
        type_str: Alert type string

    Returns:
        Normalized alert type
    """
    type_upper = type_str.upper().strip()
    type_map = {
        "TRADE": ALERT_TYPE_TRADE,
        "SIGNAL": ALERT_TYPE_SIGNAL,
        "ERROR": ALERT_TYPE_ERROR,
        "DAILY": ALERT_TYPE_DAILY,
        "SYSTEM": ALERT_TYPE_SYSTEM,
        "INFO": ALERT_TYPE_SYSTEM,
        "WARNING": ALERT_TYPE_ERROR,
    }
    return type_map.get(type_upper, ALERT_TYPE_SYSTEM)


def _read_alert_file(file_path: Path) -> List[Dict[str, Any]]:
    """Read alerts from a CSV file.

    Args:
        file_path: Path to alert CSV file

    Returns:
        List of alert dicts
    """
    alerts: List[Dict[str, Any]] = []

    if not file_path.exists():
        return alerts

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    alert = {
                        "id": row.get("id", row.get("alert_id", "")),
                        "timestamp": _parse_timestamp(row.get("timestamp", "")),
                        "alert_type": _parse_alert_type(
                            row.get("alert_type", row.get("type", "SYSTEM"))
                        ),
                        "exchange": row.get("exchange", ""),
                        "message": row.get("message", ""),
                        "status": row.get("status", STATUS_SENT).upper(),
                    }
                    alerts.append(alert)
                except Exception as e:
                    logger.debug("Failed to parse alert row: %s", e)
    except Exception as e:
        logger.debug("Failed to read alert file %s: %s", file_path, e)

    return alerts


def get_alert_history(
    days: int = 7,
    log_dir: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Get alert history from log files.

    Args:
        days: Number of days to look back (default 7)
        log_dir: Alert log directory
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        List of alert dicts sorted by timestamp (newest first)
    """
    if log_dir is None:
        log_dir = os.getenv("MASP_ALERT_LOG_DIR", DEFAULT_ALERT_LOG_DIR)

    # Determine date range
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    all_alerts: List[Dict[str, Any]] = []

    # Collect alerts from each day
    current_date = start_date
    while current_date <= end_date:
        file_path = _get_alert_file_path(current_date, log_dir)
        day_alerts = _read_alert_file(file_path)
        all_alerts.extend(day_alerts)
        current_date += timedelta(days=1)

    # Sort by timestamp (newest first)
    all_alerts.sort(key=lambda a: a.get("timestamp", datetime.min), reverse=True)

    return all_alerts


def get_alert_store(
    days: int = 7,
    log_dir: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Get alert store for alert_history component.

    Returns None if no alerts found (triggers demo mode).

    Args:
        days: Number of days to look back
        log_dir: Alert log directory

    Returns:
        List of alerts or None for demo mode
    """
    alerts = get_alert_history(days=days, log_dir=log_dir)

    # Return None to trigger demo mode if no alerts
    if not alerts:
        return None

    return alerts
