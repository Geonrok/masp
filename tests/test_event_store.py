"""
Tests for EventStore module.
"""

import os
import tempfile
import pytest
from datetime import datetime, timezone
from pathlib import Path

from libs.core.event_store import EventStore
from libs.core.event_logger import Event, EventType, Severity


class TestEventStore:
    """Tests for EventStore."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_events.db")
            yield db_path

    @pytest.fixture
    def store(self, temp_db):
        """Create EventStore instance."""
        return EventStore(dsn=temp_db)

    def test_initialization(self, temp_db):
        """Test store initialization creates database."""
        store = EventStore(dsn=temp_db)

        assert os.path.exists(temp_db)

    def test_schema_creation(self, store):
        """Test schema tables are created."""
        with store._get_connection() as conn:
            cursor = conn.cursor()

            # Check events table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
            )
            assert cursor.fetchone() is not None

            # Check runs table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
            )
            assert cursor.fetchone() is not None

    def test_write_event(self, store):
        """Test writing a single event."""
        event = Event(
            event_id="evt_001",
            ts_utc=datetime.now(timezone.utc),
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_001",
            symbol="BTC/USDT",
            severity=Severity.INFO,
            event_type=EventType.SIGNAL_DECISION,
            payload={"signal": "BUY", "strength": 0.8},
        )

        store.write_event(event)

        events = store.list_events(run_id="run_001")
        assert len(events) == 1
        assert events[0]["event_id"] == "evt_001"
        assert events[0]["payload"]["signal"] == "BUY"

    def test_run_lifecycle(self, store):
        """Test run start and finish events."""
        # Start run
        start_event = Event(
            event_id="evt_start",
            ts_utc=datetime.now(timezone.utc),
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_002",
            symbol=None,
            severity=Severity.INFO,
            event_type=EventType.RUN_STARTED,
            payload={"config": "test"},
        )
        store.write_event(start_event)

        # Check run is created
        run = store.get_run("run_002")
        assert run is not None
        assert run["status"] == "running"

        # Finish run
        finish_event = Event(
            event_id="evt_finish",
            ts_utc=datetime.now(timezone.utc),
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_002",
            symbol=None,
            severity=Severity.INFO,
            event_type=EventType.RUN_FINISHED,
            payload={
                "status": "completed",
                "decision_count": 5,
                "error_count": 0,
                "duration_seconds": 10.5,
            },
        )
        store.write_event(finish_event)

        # Check run is updated
        run = store.get_run("run_002")
        assert run["status"] == "completed"
        assert run["decision_count"] == 5
        assert run["error_count"] == 0

    def test_list_runs(self, store):
        """Test listing runs."""
        # Create multiple runs
        for i in range(5):
            event = Event(
                event_id=f"evt_start_{i}",
                ts_utc=datetime.now(timezone.utc),
                ts_kst=datetime.now(),
                asset_class="crypto" if i % 2 == 0 else "futures",
                strategy_id="KAMA",
                run_id=f"run_{i:03d}",
                symbol=None,
                severity=Severity.INFO,
                event_type=EventType.RUN_STARTED,
                payload={},
            )
            store.write_event(event)

        # List all runs
        runs = store.list_runs()
        assert len(runs) == 5

        # List by asset class
        crypto_runs = store.list_runs(asset_class="crypto")
        assert len(crypto_runs) == 3

        # Test pagination
        paginated = store.list_runs(limit=2, offset=1)
        assert len(paginated) == 2

    def test_list_events_by_run(self, store):
        """Test listing events for a specific run."""
        run_id = "run_specific"

        # Create multiple events
        for i in range(10):
            event = Event(
                event_id=f"evt_{i:03d}",
                ts_utc=datetime.now(timezone.utc),
                ts_kst=datetime.now(),
                asset_class="crypto",
                strategy_id="KAMA",
                run_id=run_id,
                symbol="BTC/USDT",
                severity=Severity.INFO,
                event_type=EventType.SIGNAL_DECISION,
                payload={"index": i},
            )
            store.write_event(event)

        events = store.list_events_by_run(run_id)
        assert len(events) == 10

    def test_list_events_filtering(self, store):
        """Test event filtering."""
        # Create events with different types
        for event_type in [
            EventType.SIGNAL_DECISION,
            EventType.ERROR,
            EventType.HEARTBEAT,
        ]:
            event = Event(
                event_id=f"evt_{event_type.value}",
                ts_utc=datetime.now(timezone.utc),
                ts_kst=datetime.now(),
                asset_class="crypto",
                strategy_id="KAMA",
                run_id="run_filter",
                symbol="BTC/USDT",
                severity=Severity.INFO,
                event_type=event_type,
                payload={},
            )
            store.write_event(event)

        # Filter by event type
        decisions = store.list_events(event_type=EventType.SIGNAL_DECISION.value)
        assert len(decisions) >= 1

        errors = store.list_events(event_type=EventType.ERROR.value)
        assert len(errors) >= 1

    def test_list_decisions(self, store):
        """Test listing decision events."""
        event = Event(
            event_id="evt_decision",
            ts_utc=datetime.now(timezone.utc),
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_dec",
            symbol="ETH/USDT",
            severity=Severity.INFO,
            event_type=EventType.SIGNAL_DECISION,
            payload={"signal": "SELL"},
        )
        store.write_event(event)

        decisions = store.list_decisions()
        assert any(d["payload"]["signal"] == "SELL" for d in decisions)

    def test_list_errors(self, store):
        """Test listing error events."""
        event = Event(
            event_id="evt_error",
            ts_utc=datetime.now(timezone.utc),
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_err",
            symbol="BTC/USDT",
            severity=Severity.ERROR,
            event_type=EventType.ERROR,
            payload={"error": "Connection failed"},
        )
        store.write_event(event)

        errors = store.list_errors()
        assert any(e["payload"]["error"] == "Connection failed" for e in errors)

    def test_get_last_heartbeat(self, store):
        """Test getting last heartbeat."""
        event = Event(
            event_id="evt_heartbeat",
            ts_utc=datetime.now(timezone.utc),
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_hb",
            symbol=None,
            severity=Severity.INFO,
            event_type=EventType.HEARTBEAT,
            payload={"status": "healthy"},
        )
        store.write_event(event)

        heartbeat = store.get_last_heartbeat()
        assert heartbeat is not None
        assert heartbeat["payload"]["status"] == "healthy"

    def test_get_asset_classes(self, store):
        """Test getting distinct asset classes."""
        for asset_class in ["crypto", "futures", "stocks"]:
            event = Event(
                event_id=f"evt_{asset_class}",
                ts_utc=datetime.now(timezone.utc),
                ts_kst=datetime.now(),
                asset_class=asset_class,
                strategy_id="KAMA",
                run_id=f"run_{asset_class}",
                symbol=None,
                severity=Severity.INFO,
                event_type=EventType.RUN_STARTED,
                payload={},
            )
            store.write_event(event)

        asset_classes = store.get_asset_classes()
        assert "crypto" in asset_classes
        assert "futures" in asset_classes
        assert "stocks" in asset_classes

    def test_get_stats(self, store):
        """Test getting statistics."""
        # Create some runs and events
        for i in range(3):
            start_event = Event(
                event_id=f"evt_stat_start_{i}",
                ts_utc=datetime.now(timezone.utc),
                ts_kst=datetime.now(),
                asset_class="crypto",
                strategy_id="KAMA",
                run_id=f"run_stat_{i}",
                symbol=None,
                severity=Severity.INFO,
                event_type=EventType.RUN_STARTED,
                payload={},
            )
            store.write_event(start_event)

        stats = store.get_stats()
        assert stats["total_runs"] >= 3
        assert stats["total_events"] >= 3

        # Filter by asset class
        stats_crypto = store.get_stats(asset_class="crypto")
        assert stats_crypto["total_runs"] >= 3

    def test_event_upsert(self, store):
        """Test event INSERT OR REPLACE behavior."""
        event = Event(
            event_id="evt_upsert",
            ts_utc=datetime.now(timezone.utc),
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_upsert",
            symbol="BTC/USDT",
            severity=Severity.INFO,
            event_type=EventType.SIGNAL_DECISION,
            payload={"version": 1},
        )
        store.write_event(event)

        # Update same event
        event.payload = {"version": 2}
        store.write_event(event)

        events = store.list_events(run_id="run_upsert")
        assert len(events) == 1
        assert events[0]["payload"]["version"] == 2

    def test_timestamp_filtering(self, store):
        """Test filtering by timestamp."""
        import time

        # Create event with specific timestamp
        now = datetime.now(timezone.utc)

        event = Event(
            event_id="evt_time",
            ts_utc=now,
            ts_kst=datetime.now(),
            asset_class="crypto",
            strategy_id="KAMA",
            run_id="run_time",
            symbol="BTC/USDT",
            severity=Severity.INFO,
            event_type=EventType.SIGNAL_DECISION,
            payload={},
        )
        store.write_event(event)

        # Query with since filter
        events = store.list_events(since=now.isoformat())
        assert len(events) >= 1


class TestEventStoreConnection:
    """Tests for database connection handling."""

    def test_connection_context_manager(self):
        """Test connection is properly closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = EventStore(dsn=db_path)

            with store._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1

    def test_directory_creation(self):
        """Test database directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "dir", "test.db")
            store = EventStore(dsn=nested_path)

            assert os.path.exists(os.path.dirname(nested_path))
