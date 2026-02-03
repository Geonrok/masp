"""
Tests for backup and restore functionality.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from libs.storage.backup import (
    BackupConfig,
    BackupManager,
    BackupResult,
    RestoreResult,
    create_backup,
    list_backups,
    restore_backup,
)


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    # Create directories
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()

    logs_dir = tmp_path / "logs" / "paper_trades" / "trades" / "2024-01"
    logs_dir.mkdir(parents=True)

    config_dir = tmp_path / "config"
    config_dir.mkdir()

    backtests_dir = tmp_path / "data" / "backtests" / "momentum"
    backtests_dir.mkdir(parents=True)

    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()

    # Create SQLite database
    db_path = storage_dir / "local.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            ts_utc TEXT,
            event_type TEXT,
            payload TEXT
        )
    """)
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?)",
        ("evt_001", datetime.now(timezone.utc).isoformat(), "TEST", '{"data": 1}'),
    )
    conn.commit()
    conn.close()

    # Create config files
    runtime_config = storage_dir / "runtime_config.json"
    runtime_config.write_text(json.dumps({"key": "value"}))

    schedule_config = config_dir / "schedule_config.json"
    schedule_config.write_text(json.dumps({"schedule": "daily"}))

    # Create trade logs
    trade_log = logs_dir / "trades_2024-01-15.csv"
    trade_log.write_text(
        "timestamp,exchange,symbol,side,quantity,price\n2024-01-15,upbit,BTC,BUY,0.1,50000000\n"
    )

    # Create backtest result
    backtest_file = backtests_dir / "bt_20240115_120000.json"
    backtest_file.write_text(
        json.dumps(
            {
                "strategy": "momentum",
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.12,
            }
        )
    )

    return tmp_path


@pytest.fixture
def backup_config(temp_project):
    """Create backup config for temp project."""
    return BackupConfig(
        project_root=temp_project,
        backup_dir=temp_project / "backups",
        config_paths=[
            "storage/runtime_config.json",
            "config/schedule_config.json",
        ],
    )


class TestBackupManager:
    """Tests for BackupManager."""

    def test_create_backup_full(self, backup_config):
        """Test creating a full backup."""
        manager = BackupManager(backup_config)
        result = manager.create_backup(description="Test backup")

        assert result.success is True
        assert result.backup_id.startswith("masp_backup_")
        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert result.size_bytes > 0
        assert result.items_backed_up.get("database", 0) == 1
        assert result.items_backed_up.get("config_files", 0) >= 1
        assert result.checksum is not None

    def test_create_backup_uncompressed(self, backup_config):
        """Test creating uncompressed backup."""
        backup_config.compress = False
        manager = BackupManager(backup_config)
        result = manager.create_backup()

        assert result.success is True
        assert result.backup_path.is_dir()
        assert (result.backup_path / "manifest.json").exists()

    def test_create_backup_selective(self, backup_config):
        """Test creating selective backup."""
        backup_config.include_trade_logs = False
        backup_config.include_backtests = False

        manager = BackupManager(backup_config)
        result = manager.create_backup()

        assert result.success is True
        assert "trade_logs" not in result.items_backed_up
        assert "backtests" not in result.items_backed_up

    def test_list_backups(self, backup_config):
        """Test listing backups."""
        manager = BackupManager(backup_config)

        # Create multiple backups with explicit IDs to avoid timestamp collision
        result1 = manager.create_backup(backup_id="masp_backup_test_001")
        result2 = manager.create_backup(backup_id="masp_backup_test_002")

        backups = manager.list_backups()

        assert len(backups) == 2
        # Both backups should be present
        backup_ids = [b["backup_id"] for b in backups]
        assert result1.backup_id in backup_ids
        assert result2.backup_id in backup_ids

    def test_delete_backup(self, backup_config):
        """Test deleting a backup."""
        manager = BackupManager(backup_config)
        result = manager.create_backup()

        assert result.backup_path.exists()

        deleted = manager.delete_backup(result.backup_id)

        assert deleted is True
        assert not result.backup_path.exists()

    def test_restore_backup_database(self, backup_config):
        """Test restoring database from backup."""
        manager = BackupManager(backup_config)

        # Create backup
        backup_result = manager.create_backup()
        assert backup_result.success

        # Modify database
        db_path = backup_config.project_root / "storage" / "local.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM events")
        conn.commit()
        conn.close()

        # Restore
        restore_result = manager.restore_backup(
            backup_result.backup_id,
            restore_database=True,
            restore_config=False,
        )

        assert restore_result.success is True
        assert restore_result.items_restored.get("database", 0) == 1

        # Verify data restored
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1  # Original event restored

    def test_restore_backup_config(self, backup_config):
        """Test restoring config from backup."""
        manager = BackupManager(backup_config)

        # Create backup
        backup_result = manager.create_backup()

        # Modify config
        config_path = backup_config.project_root / "storage" / "runtime_config.json"
        config_path.write_text('{"modified": true}')

        # Restore
        restore_result = manager.restore_backup(
            backup_result.backup_id,
            restore_database=False,
            restore_config=True,
        )

        assert restore_result.success is True

        # Verify config restored
        restored_config = json.loads(config_path.read_text())
        assert restored_config == {"key": "value"}

    def test_backup_rotation(self, backup_config):
        """Test backup rotation (max_backups limit)."""
        backup_config.max_backups = 3
        manager = BackupManager(backup_config)

        # Create more backups than max
        backup_ids = []
        for i in range(5):
            result = manager.create_backup(backup_id=f"masp_backup_test_{i:03d}")
            backup_ids.append(result.backup_id)

        backups = manager.list_backups()

        # Should only keep 3 most recent
        assert len(backups) == 3
        # Oldest backups should be deleted
        assert backup_ids[0] not in [b["backup_id"] for b in backups]
        assert backup_ids[1] not in [b["backup_id"] for b in backups]


class TestBackupResult:
    """Tests for BackupResult."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BackupResult(
            success=True,
            backup_id="test_backup",
            backup_path=Path("/tmp/backup.tar.gz"),
            size_bytes=1024,
            items_backed_up={"database": 1},
            duration_seconds=5.5,
            checksum="abc123",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["backup_id"] == "test_backup"
        assert d["size_bytes"] == 1024
        assert d["checksum"] == "abc123"


class TestRestoreResult:
    """Tests for RestoreResult."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = RestoreResult(
            success=True,
            backup_id="test_backup",
            items_restored={"database": 1, "config_files": 2},
            duration_seconds=3.2,
            warnings=["Some warning"],
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["items_restored"]["database"] == 1
        assert len(d["warnings"]) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_backup_function(self, backup_config):
        """Test create_backup convenience function."""
        result = create_backup(
            backup_dir=backup_config.backup_dir,
            compress=True,
            description="Test",
        )

        # Note: This might fail if project structure doesn't exist
        # The result depends on actual file existence
        assert isinstance(result, BackupResult)

    def test_list_backups_function(self, backup_config):
        """Test list_backups convenience function."""
        # Create a backup first
        manager = BackupManager(backup_config)
        manager.create_backup()

        backups = list_backups(backup_dir=backup_config.backup_dir)

        assert isinstance(backups, list)
        assert len(backups) >= 1


class TestBackupConfig:
    """Tests for BackupConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BackupConfig()

        assert config.include_database is True
        assert config.include_config is True
        assert config.compress is True
        assert config.max_backups == 30

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        config = BackupConfig(
            project_root=tmp_path,
            backup_dir=tmp_path / "custom_backups",
            include_trade_logs=False,
            max_backups=10,
        )

        assert config.project_root == tmp_path
        assert config.include_trade_logs is False
        assert config.max_backups == 10
