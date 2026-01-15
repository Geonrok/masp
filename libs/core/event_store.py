"""
Event Store - SQLite-based persistent storage for events.
Phase 0 implementation with local SQLite.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from libs.core.event_logger import Event, EventType


class EventStore:
    """
    SQLite-based event store for Phase 0.
    Provides write/read operations with query support.
    """
    
    def __init__(self, dsn: str = "storage/local.db"):
        """
        Initialize event store.
        
        Args:
            dsn: Path to SQLite database file
        """
        self.dsn = dsn
        self._ensure_db_exists()
        self._init_schema()
    
    def _ensure_db_exists(self) -> None:
        """Ensure database directory exists."""
        db_path = Path(self.dsn)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.dsn)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_schema(self) -> None:
        """Initialize database schema with indices."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    ts_kst TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    strategy_id TEXT,
                    run_id TEXT NOT NULL,
                    symbol TEXT,
                    severity TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            
            # Runs summary table (for faster queries)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    asset_class TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT,
                    decision_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    duration_seconds REAL
                )
            """)
            
            # Indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_ts_utc 
                ON events(ts_utc)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_asset_class 
                ON events(asset_class)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_run_id 
                ON events(run_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_event_type 
                ON events(event_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_asset_class 
                ON runs(asset_class)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_start_time 
                ON runs(start_time)
            """)
            
            conn.commit()
    
    def write_event(self, event: Event) -> None:
        """Write a single event to the store."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO events 
                (event_id, ts_utc, ts_kst, asset_class, strategy_id, 
                 run_id, symbol, severity, event_type, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.ts_utc.isoformat(),
                event.ts_kst.isoformat(),
                event.asset_class,
                event.strategy_id,
                event.run_id,
                event.symbol,
                event.severity.value,
                event.event_type.value,
                json.dumps(event.payload),
            ))
            
            # Update runs table for RUN_STARTED/RUN_FINISHED
            if event.event_type == EventType.RUN_STARTED:
                cursor.execute("""
                    INSERT OR REPLACE INTO runs 
                    (run_id, asset_class, start_time, status)
                    VALUES (?, ?, ?, 'running')
                """, (event.run_id, event.asset_class, event.ts_utc.isoformat()))
            
            elif event.event_type == EventType.RUN_FINISHED:
                payload = event.payload
                cursor.execute("""
                    UPDATE runs SET 
                        end_time = ?,
                        status = ?,
                        decision_count = ?,
                        error_count = ?,
                        duration_seconds = ?
                    WHERE run_id = ?
                """, (
                    event.ts_utc.isoformat(),
                    payload.get("status", "unknown"),
                    payload.get("decision_count", 0),
                    payload.get("error_count", 0),
                    payload.get("duration_seconds"),
                    event.run_id,
                ))
            
            conn.commit()
    
    def _row_to_event(self, row: sqlite3.Row) -> dict:
        """Convert database row to event dictionary."""
        return {
            "event_id": row["event_id"],
            "ts_utc": row["ts_utc"],
            "ts_kst": row["ts_kst"],
            "asset_class": row["asset_class"],
            "strategy_id": row["strategy_id"],
            "run_id": row["run_id"],
            "symbol": row["symbol"],
            "severity": row["severity"],
            "event_type": row["event_type"],
            "payload": json.loads(row["payload"]),
        }
    
    def _row_to_run(self, row: sqlite3.Row) -> dict:
        """Convert database row to run dictionary."""
        return {
            "run_id": row["run_id"],
            "asset_class": row["asset_class"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "status": row["status"],
            "decision_count": row["decision_count"],
            "error_count": row["error_count"],
            "duration_seconds": row["duration_seconds"],
        }
    
    def list_runs(
        self,
        asset_class: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """
        List runs with optional filtering.
        
        Args:
            asset_class: Filter by asset class
            limit: Maximum number of runs to return
            offset: Offset for pagination
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM runs"
            params: list[Any] = []
            
            if asset_class:
                query += " WHERE asset_class = ?"
                params.append(asset_class)
            
            query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            return [self._row_to_run(row) for row in cursor.fetchall()]
    
    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a single run by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return self._row_to_run(row) if row else None
    
    def list_events_by_run(
        self,
        run_id: str,
        limit: int = 1000,
    ) -> list[dict]:
        """List all events for a specific run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM events 
                WHERE run_id = ? 
                ORDER BY ts_utc ASC
                LIMIT ?
            """, (run_id, limit))
            return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def list_events(
        self,
        run_id: Optional[str] = None,
        asset_class: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        List events with flexible filtering.
        
        Args:
            run_id: Filter by run ID
            asset_class: Filter by asset class
            event_type: Filter by event type
            since: Filter events after this ISO timestamp
            limit: Maximum number of events
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params: list[Any] = []
            
            if run_id:
                query += " AND run_id = ?"
                params.append(run_id)
            
            if asset_class:
                query += " AND asset_class = ?"
                params.append(asset_class)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if since:
                query += " AND ts_utc >= ?"
                params.append(since)
            
            query += " ORDER BY ts_utc DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def list_decisions(
        self,
        asset_class: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List SIGNAL_DECISION events."""
        return self.list_events(
            asset_class=asset_class,
            event_type=EventType.SIGNAL_DECISION.value,
            since=since,
            limit=limit,
        )
    
    def list_errors(
        self,
        asset_class: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List ERROR events."""
        return self.list_events(
            asset_class=asset_class,
            event_type=EventType.ERROR.value,
            since=since,
            limit=limit,
        )
    
    def get_last_heartbeat(
        self,
        asset_class: Optional[str] = None,
    ) -> Optional[dict]:
        """Get the most recent heartbeat event."""
        events = self.list_events(
            asset_class=asset_class,
            event_type=EventType.HEARTBEAT.value,
            limit=1,
        )
        return events[0] if events else None
    
    def get_asset_classes(self) -> list[str]:
        """Get list of all asset classes with data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT asset_class FROM runs
                ORDER BY asset_class
            """)
            return [row["asset_class"] for row in cursor.fetchall()]
    
    def get_stats(self, asset_class: Optional[str] = None) -> dict:
        """Get summary statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total runs
            if asset_class:
                cursor.execute(
                    "SELECT COUNT(*) as count FROM runs WHERE asset_class = ?",
                    (asset_class,)
                )
            else:
                cursor.execute("SELECT COUNT(*) as count FROM runs")
            total_runs = cursor.fetchone()["count"]
            
            # Total events
            if asset_class:
                cursor.execute(
                    "SELECT COUNT(*) as count FROM events WHERE asset_class = ?",
                    (asset_class,)
                )
            else:
                cursor.execute("SELECT COUNT(*) as count FROM events")
            total_events = cursor.fetchone()["count"]
            
            return {
                "total_runs": total_runs,
                "total_events": total_events,
            }


