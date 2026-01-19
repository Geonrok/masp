"""Log provider - connects application logs to log_viewer component."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from services.dashboard.components.log_viewer import LogEntry, LogLevel

logger = logging.getLogger(__name__)

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Log file patterns
LOG_FILE_PATTERNS = [
    "masp*.log",
    "*.log",
    "app*.log",
    "error*.log",
]

# Regex patterns for log parsing
LOG_LINE_PATTERNS = [
    # Standard Python logging: 2024-01-01 12:00:00,000 - module - LEVEL - message
    re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)\s*"
        r"[-–]\s*(?P<source>[\w.]+)\s*"
        r"[-–]\s*(?P<level>\w+)\s*"
        r"[-–]\s*(?P<message>.*)$"
    ),
    # Bracketed format: [2024-01-01 12:00:00] [LEVEL] [source] message
    re.compile(
        r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*"
        r"\[(?P<level>\w+)\]\s*"
        r"\[(?P<source>[\w.]+)\]\s*"
        r"(?P<message>.*)$"
    ),
    # Simple format: LEVEL - source - message
    re.compile(
        r"^(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*"
        r"[-–]\s*(?P<source>[\w.]+)\s*"
        r"[-–]\s*(?P<message>.*)$"
    ),
]


def _parse_log_level(level_str: str) -> LogLevel:
    """Parse log level string to LogLevel enum.

    Args:
        level_str: Log level string

    Returns:
        LogLevel enum value
    """
    level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARN": LogLevel.WARNING,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
        "FATAL": LogLevel.CRITICAL,
    }
    return level_map.get(level_str.upper(), LogLevel.INFO)


def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime.

    Args:
        timestamp_str: Timestamp string

    Returns:
        Parsed datetime or current time if parsing fails
    """
    formats = [
        "%Y-%m-%d %H:%M:%S,%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    return datetime.now()


def _parse_log_line(line: str) -> Optional[LogEntry]:
    """Parse a single log line to LogEntry.

    Args:
        line: Log line string

    Returns:
        LogEntry if parsing succeeds, None otherwise
    """
    line = line.strip()
    if not line:
        return None

    for pattern in LOG_LINE_PATTERNS:
        match = pattern.match(line)
        if match:
            groups = match.groupdict()
            timestamp_str = groups.get("timestamp", "")
            timestamp = _parse_timestamp(timestamp_str) if timestamp_str else datetime.now()

            return LogEntry(
                timestamp=timestamp,
                level=_parse_log_level(groups.get("level", "INFO")),
                source=groups.get("source", "unknown"),
                message=groups.get("message", line),
            )

    # Fallback: treat entire line as message
    return LogEntry(
        timestamp=datetime.now(),
        level=LogLevel.INFO,
        source="unknown",
        message=line,
    )


def _find_log_files(log_dir: str, max_files: int = 10) -> List[Path]:
    """Find log files in directory.

    Args:
        log_dir: Directory to search
        max_files: Maximum number of files to return

    Returns:
        List of log file paths, sorted by modification time (newest first)
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    files: List[Path] = []

    for pattern in LOG_FILE_PATTERNS:
        files.extend(log_path.glob(pattern))

    # Deduplicate and sort by modification time
    unique_files = list(set(files))
    unique_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    return unique_files[:max_files]


def _read_log_file(
    file_path: Path, max_lines: int = 1000
) -> List[LogEntry]:
    """Read log entries from a file.

    Args:
        file_path: Path to log file
        max_lines: Maximum lines to read (from end of file)

    Returns:
        List of LogEntry objects
    """
    entries: List[LogEntry] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # Read all lines and take last max_lines
            lines = f.readlines()
            lines = lines[-max_lines:] if len(lines) > max_lines else lines

            for line in lines:
                entry = _parse_log_line(line)
                if entry:
                    entries.append(entry)

    except Exception as e:
        logger.debug("Failed to read log file %s: %s", file_path, e)

    return entries


def get_log_entries(
    log_dir: Optional[str] = None,
    max_entries: int = 500,
    max_files: int = 5,
) -> List[LogEntry]:
    """Get log entries from log files.

    Args:
        log_dir: Log directory path (default: logs/)
        max_entries: Maximum entries to return
        max_files: Maximum files to read

    Returns:
        List of LogEntry objects, sorted by timestamp (newest first)
    """
    if log_dir is None:
        log_dir = os.getenv("MASP_LOG_DIR", DEFAULT_LOG_DIR)

    files = _find_log_files(log_dir, max_files)
    if not files:
        logger.debug("No log files found in %s", log_dir)
        return []

    all_entries: List[LogEntry] = []
    entries_per_file = max(max_entries // len(files), 100)

    for file_path in files:
        entries = _read_log_file(file_path, entries_per_file)
        all_entries.extend(entries)

    # Sort by timestamp (newest first) and limit
    all_entries.sort(key=lambda e: e.timestamp, reverse=True)
    return all_entries[:max_entries]


def get_log_provider(
    log_dir: Optional[str] = None,
    max_entries: int = 500,
) -> Callable[[], List[LogEntry]]:
    """Get log provider function for log_viewer component.

    Args:
        log_dir: Log directory path
        max_entries: Maximum entries to return

    Returns:
        Function that returns List[LogEntry]
    """

    def provider() -> List[LogEntry]:
        return get_log_entries(log_dir=log_dir, max_entries=max_entries)

    return provider
