"""
Backup and Restore Manager for MASP

Provides comprehensive backup/restore functionality for:
- SQLite database (event store)
- JSON configuration files
- Trade logs (CSV)
- Backtest results
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Configuration for backup operations."""

    # Base paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    backup_dir: Path = field(default_factory=lambda: Path.cwd() / "backups")

    # What to backup
    include_database: bool = True
    include_config: bool = True
    include_trade_logs: bool = True
    include_backtests: bool = True

    # Retention settings
    max_backups: int = 30
    compress: bool = True

    # Paths relative to project root
    database_path: str = "storage/local.db"
    config_paths: List[str] = field(
        default_factory=lambda: [
            "storage/runtime_config.json",
            "config/schedule_config.json",
        ]
    )
    trade_log_dir: str = "logs"
    backtest_dir: str = "data/backtests"


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    backup_id: str
    backup_path: Optional[Path] = None
    size_bytes: int = 0
    items_backed_up: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    checksum: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "backup_id": self.backup_id,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "size_bytes": self.size_bytes,
            "items_backed_up": self.items_backed_up,
            "duration_seconds": self.duration_seconds,
            "checksum": self.checksum,
            "error": self.error,
        }


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    success: bool
    backup_id: str
    items_restored: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "backup_id": self.backup_id,
            "items_restored": self.items_restored,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "warnings": self.warnings,
        }


class BackupManager:
    """
    Manages backup and restore operations for MASP.

    Supports:
    - Full backups (all data)
    - Selective backups (specific components)
    - Incremental backups (only changed files)
    - Automatic rotation (keep N most recent)

    Usage:
        manager = BackupManager()
        result = manager.create_backup()

        # Restore from specific backup
        manager.restore_backup("backup_20240115_120000")

        # List available backups
        backups = manager.list_backups()
    """

    BACKUP_PREFIX = "masp_backup"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, config: Optional[BackupConfig] = None):
        """
        Initialize backup manager.

        Args:
            config: Backup configuration (uses defaults if None)
        """
        self.config = config or BackupConfig()
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[BackupManager] Initialized with backup_dir={self.config.backup_dir}"
        )

    def create_backup(
        self,
        backup_id: Optional[str] = None,
        description: str = "",
    ) -> BackupResult:
        """
        Create a full backup.

        Args:
            backup_id: Custom backup ID (auto-generated if None)
            description: Optional description for the backup

        Returns:
            BackupResult with details
        """
        start_time = datetime.now(timezone.utc)

        if not backup_id:
            backup_id = f"{self.BACKUP_PREFIX}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"[BackupManager] Creating backup: {backup_id}")

        # Create temporary backup directory
        temp_dir = self.config.backup_dir / f".{backup_id}_temp"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)

            items_backed_up = {}

            # 1. Backup database
            if self.config.include_database:
                count = self._backup_database(temp_dir)
                items_backed_up["database"] = count

            # 2. Backup config files
            if self.config.include_config:
                count = self._backup_configs(temp_dir)
                items_backed_up["config_files"] = count

            # 3. Backup trade logs
            if self.config.include_trade_logs:
                count = self._backup_trade_logs(temp_dir)
                items_backed_up["trade_logs"] = count

            # 4. Backup backtest results
            if self.config.include_backtests:
                count = self._backup_backtests(temp_dir)
                items_backed_up["backtests"] = count

            # Create manifest
            manifest = {
                "backup_id": backup_id,
                "created_at": start_time.isoformat(),
                "description": description,
                "items": items_backed_up,
                "config": {
                    "include_database": self.config.include_database,
                    "include_config": self.config.include_config,
                    "include_trade_logs": self.config.include_trade_logs,
                    "include_backtests": self.config.include_backtests,
                },
            }

            manifest_path = temp_dir / self.MANIFEST_FILE
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            # Create final backup (compressed or directory)
            if self.config.compress:
                backup_path = self.config.backup_dir / f"{backup_id}.tar.gz"
                self._compress_directory(temp_dir, backup_path)
                size_bytes = backup_path.stat().st_size
                checksum = self._calculate_checksum(backup_path)
            else:
                backup_path = self.config.backup_dir / backup_id
                shutil.move(str(temp_dir), str(backup_path))
                size_bytes = self._get_directory_size(backup_path)
                checksum = None

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Cleanup old backups
            self._rotate_backups()

            logger.info(
                f"[BackupManager] Backup complete: {backup_id} "
                f"({size_bytes / 1024 / 1024:.2f} MB, {duration:.2f}s)"
            )

            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_path=backup_path,
                size_bytes=size_bytes,
                items_backed_up=items_backed_up,
                duration_seconds=duration,
                checksum=checksum,
            )

        except Exception as e:
            logger.error(f"[BackupManager] Backup failed: {e}")
            return BackupResult(
                success=False,
                backup_id=backup_id,
                error=str(e),
                duration_seconds=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
            )
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def restore_backup(
        self,
        backup_id: str,
        restore_database: bool = True,
        restore_config: bool = True,
        restore_trade_logs: bool = False,  # Default False to avoid overwriting
        restore_backtests: bool = False,
    ) -> RestoreResult:
        """
        Restore from a backup.

        Args:
            backup_id: ID of the backup to restore
            restore_database: Whether to restore database
            restore_config: Whether to restore config files
            restore_trade_logs: Whether to restore trade logs
            restore_backtests: Whether to restore backtest results

        Returns:
            RestoreResult with details
        """
        start_time = datetime.now(timezone.utc)
        warnings = []

        logger.info(f"[BackupManager] Restoring backup: {backup_id}")

        # Find backup
        backup_path = self._find_backup(backup_id)
        if not backup_path:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                error=f"Backup not found: {backup_id}",
            )

        # Extract to temp directory if compressed
        temp_dir = self.config.backup_dir / f".{backup_id}_restore_temp"
        try:
            if backup_path.suffix == ".gz":
                self._extract_archive(backup_path, temp_dir)
                source_dir = temp_dir
            else:
                source_dir = backup_path

            # Read manifest
            manifest_path = source_dir / self.MANIFEST_FILE
            if not manifest_path.exists():
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    error="Manifest file not found in backup",
                )

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            items_restored = {}

            # 1. Restore database
            if restore_database and manifest.get("config", {}).get("include_database"):
                count = self._restore_database(source_dir)
                items_restored["database"] = count
            elif restore_database:
                warnings.append("Database was not included in this backup")

            # 2. Restore config files
            if restore_config and manifest.get("config", {}).get("include_config"):
                count = self._restore_configs(source_dir)
                items_restored["config_files"] = count
            elif restore_config:
                warnings.append("Config files were not included in this backup")

            # 3. Restore trade logs
            if restore_trade_logs and manifest.get("config", {}).get(
                "include_trade_logs"
            ):
                count = self._restore_trade_logs(source_dir)
                items_restored["trade_logs"] = count
            elif restore_trade_logs:
                warnings.append("Trade logs were not included in this backup")

            # 4. Restore backtests
            if restore_backtests and manifest.get("config", {}).get(
                "include_backtests"
            ):
                count = self._restore_backtests(source_dir)
                items_restored["backtests"] = count
            elif restore_backtests:
                warnings.append("Backtests were not included in this backup")

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.info(
                f"[BackupManager] Restore complete: {backup_id} ({duration:.2f}s)"
            )

            return RestoreResult(
                success=True,
                backup_id=backup_id,
                items_restored=items_restored,
                duration_seconds=duration,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"[BackupManager] Restore failed: {e}")
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                error=str(e),
                duration_seconds=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
            )
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.

        Returns:
            List of backup info dictionaries
        """
        backups = []

        for item in self.config.backup_dir.iterdir():
            if item.name.startswith("."):
                continue

            if item.is_file() and item.suffix == ".gz":
                # Compressed backup
                backup_id = item.stem.replace(".tar", "")
                try:
                    with tarfile.open(item, "r:gz") as tar:
                        manifest_member = tar.getmember(f"{self.MANIFEST_FILE}")
                        manifest_file = tar.extractfile(manifest_member)
                        if manifest_file:
                            manifest = json.load(manifest_file)
                            backups.append(
                                {
                                    "backup_id": backup_id,
                                    "path": str(item),
                                    "size_bytes": item.stat().st_size,
                                    "created_at": manifest.get("created_at"),
                                    "description": manifest.get("description", ""),
                                    "items": manifest.get("items", {}),
                                    "compressed": True,
                                }
                            )
                except Exception as e:
                    logger.warning(f"[BackupManager] Failed to read backup {item}: {e}")

            elif item.is_dir() and item.name.startswith(self.BACKUP_PREFIX):
                # Uncompressed backup
                manifest_path = item / self.MANIFEST_FILE
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        backups.append(
                            {
                                "backup_id": item.name,
                                "path": str(item),
                                "size_bytes": self._get_directory_size(item),
                                "created_at": manifest.get("created_at"),
                                "description": manifest.get("description", ""),
                                "items": manifest.get("items", {}),
                                "compressed": False,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"[BackupManager] Failed to read backup {item}: {e}"
                        )

        # Sort by created_at descending
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backups

    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a specific backup.

        Args:
            backup_id: ID of the backup to delete

        Returns:
            True if deleted successfully
        """
        backup_path = self._find_backup(backup_id)
        if not backup_path:
            logger.warning(f"[BackupManager] Backup not found: {backup_id}")
            return False

        try:
            if backup_path.is_file():
                backup_path.unlink()
            else:
                shutil.rmtree(backup_path)
            logger.info(f"[BackupManager] Deleted backup: {backup_id}")
            return True
        except Exception as e:
            logger.error(f"[BackupManager] Failed to delete backup {backup_id}: {e}")
            return False

    def _backup_database(self, dest_dir: Path) -> int:
        """Backup SQLite database."""
        db_path = self.config.project_root / self.config.database_path
        if not db_path.exists():
            logger.warning(f"[BackupManager] Database not found: {db_path}")
            return 0

        # Use SQLite backup API for consistency
        dest_path = dest_dir / "database" / db_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            source_conn = sqlite3.connect(str(db_path))
            dest_conn = sqlite3.connect(str(dest_path))
            source_conn.backup(dest_conn)
            source_conn.close()
            dest_conn.close()
            logger.debug(f"[BackupManager] Database backed up: {db_path}")
            return 1
        except Exception as e:
            logger.error(f"[BackupManager] Database backup failed: {e}")
            # Fallback to file copy
            shutil.copy2(db_path, dest_path)
            return 1

    def _backup_configs(self, dest_dir: Path) -> int:
        """Backup configuration files."""
        count = 0
        config_dest = dest_dir / "config"
        config_dest.mkdir(parents=True, exist_ok=True)

        for config_path_str in self.config.config_paths:
            config_path = self.config.project_root / config_path_str
            if config_path.exists():
                dest_path = config_dest / config_path.name
                shutil.copy2(config_path, dest_path)
                count += 1
                logger.debug(f"[BackupManager] Config backed up: {config_path}")

        return count

    def _backup_trade_logs(self, dest_dir: Path) -> int:
        """Backup trade log files."""
        logs_path = self.config.project_root / self.config.trade_log_dir
        if not logs_path.exists():
            return 0

        count = 0
        logs_dest = dest_dir / "trade_logs"

        # Copy entire logs directory structure
        for log_dir in logs_path.iterdir():
            if log_dir.is_dir() and "trades" in log_dir.name:
                dest_subdir = logs_dest / log_dir.name
                shutil.copytree(log_dir, dest_subdir, dirs_exist_ok=True)
                count += sum(1 for _ in dest_subdir.rglob("*.csv"))

        logger.debug(f"[BackupManager] Trade logs backed up: {count} files")
        return count

    def _backup_backtests(self, dest_dir: Path) -> int:
        """Backup backtest results."""
        bt_path = self.config.project_root / self.config.backtest_dir
        if not bt_path.exists():
            return 0

        bt_dest = dest_dir / "backtests"
        shutil.copytree(bt_path, bt_dest, dirs_exist_ok=True)

        count = sum(1 for _ in bt_dest.rglob("*.json"))
        logger.debug(f"[BackupManager] Backtests backed up: {count} files")
        return count

    def _restore_database(self, source_dir: Path) -> int:
        """Restore SQLite database."""
        db_backup = source_dir / "database"
        if not db_backup.exists():
            return 0

        db_files = list(db_backup.glob("*.db"))
        if not db_files:
            return 0

        backup_file = db_files[0]
        dest_path = self.config.project_root / self.config.database_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup of current database before restore
        if dest_path.exists():
            backup_current = dest_path.with_suffix(".db.pre_restore")
            shutil.copy2(dest_path, backup_current)

        shutil.copy2(backup_file, dest_path)
        logger.debug(f"[BackupManager] Database restored: {dest_path}")
        return 1

    def _restore_configs(self, source_dir: Path) -> int:
        """Restore configuration files."""
        config_backup = source_dir / "config"
        if not config_backup.exists():
            return 0

        count = 0
        for config_file in config_backup.iterdir():
            # Find matching config path
            for config_path_str in self.config.config_paths:
                config_path = self.config.project_root / config_path_str
                if config_path.name == config_file.name:
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_file, config_path)
                    count += 1
                    logger.debug(f"[BackupManager] Config restored: {config_path}")
                    break

        return count

    def _restore_trade_logs(self, source_dir: Path) -> int:
        """Restore trade log files."""
        logs_backup = source_dir / "trade_logs"
        if not logs_backup.exists():
            return 0

        logs_dest = self.config.project_root / self.config.trade_log_dir
        logs_dest.mkdir(parents=True, exist_ok=True)

        count = 0
        for log_dir in logs_backup.iterdir():
            if log_dir.is_dir():
                dest_subdir = logs_dest / log_dir.name
                shutil.copytree(log_dir, dest_subdir, dirs_exist_ok=True)
                count += sum(1 for _ in dest_subdir.rglob("*.csv"))

        logger.debug(f"[BackupManager] Trade logs restored: {count} files")
        return count

    def _restore_backtests(self, source_dir: Path) -> int:
        """Restore backtest results."""
        bt_backup = source_dir / "backtests"
        if not bt_backup.exists():
            return 0

        bt_dest = self.config.project_root / self.config.backtest_dir
        bt_dest.mkdir(parents=True, exist_ok=True)

        shutil.copytree(bt_backup, bt_dest, dirs_exist_ok=True)

        count = sum(1 for _ in bt_dest.rglob("*.json"))
        logger.debug(f"[BackupManager] Backtests restored: {count} files")
        return count

    def _find_backup(self, backup_id: str) -> Optional[Path]:
        """Find backup by ID."""
        # Check compressed
        compressed = self.config.backup_dir / f"{backup_id}.tar.gz"
        if compressed.exists():
            return compressed

        # Check directory
        directory = self.config.backup_dir / backup_id
        if directory.exists():
            return directory

        return None

    def _compress_directory(self, source_dir: Path, dest_path: Path) -> None:
        """Compress directory to tar.gz."""
        with tarfile.open(dest_path, "w:gz") as tar:
            for item in source_dir.iterdir():
                tar.add(item, arcname=item.name)

    def _extract_archive(self, archive_path: Path, dest_dir: Path) -> None:
        """Extract tar.gz archive."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory."""
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    def _rotate_backups(self) -> None:
        """Remove old backups beyond max_backups limit."""
        backups = self.list_backups()

        if len(backups) <= self.config.max_backups:
            return

        # Delete oldest backups
        for backup in backups[self.config.max_backups :]:
            self.delete_backup(backup["backup_id"])


# Convenience functions
def create_backup(
    backup_dir: Optional[Path] = None,
    compress: bool = True,
    description: str = "",
) -> BackupResult:
    """
    Create a backup with default settings.

    Args:
        backup_dir: Backup destination (default: ./backups)
        compress: Whether to compress the backup
        description: Optional description

    Returns:
        BackupResult
    """
    config = BackupConfig()
    if backup_dir:
        config.backup_dir = backup_dir
    config.compress = compress

    manager = BackupManager(config)
    return manager.create_backup(description=description)


def restore_backup(
    backup_id: str,
    backup_dir: Optional[Path] = None,
) -> RestoreResult:
    """
    Restore from a backup.

    Args:
        backup_id: ID of the backup to restore
        backup_dir: Backup directory (default: ./backups)

    Returns:
        RestoreResult
    """
    config = BackupConfig()
    if backup_dir:
        config.backup_dir = backup_dir

    manager = BackupManager(config)
    return manager.restore_backup(backup_id)


def list_backups(backup_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List all available backups.

    Args:
        backup_dir: Backup directory (default: ./backups)

    Returns:
        List of backup info dictionaries
    """
    config = BackupConfig()
    if backup_dir:
        config.backup_dir = backup_dir

    manager = BackupManager(config)
    return manager.list_backups()
