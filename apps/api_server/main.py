"""
API Server - Read-only REST API for the dashboard.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from libs.core.event_store import EventStore
from libs.core.config import AssetClass
from libs.core.paths import find_repo_root

REPO_ROOT = find_repo_root(Path(__file__))
STORAGE_DIR = REPO_ROOT / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_PATH = STORAGE_DIR / "local.db"
DASHBOARD_INDEX = REPO_ROOT / "web" / "dashboard" / "index.html"

app = FastAPI(
    title="Multi-Asset Strategy Platform API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

event_store = EventStore(dsn=str(STORAGE_PATH))


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class AssetClassesResponse(BaseModel):
    asset_classes: list[str]
    available: list[str]


class StatsResponse(BaseModel):
    total_runs: int
    total_events: int
    last_heartbeat: Optional[dict]
    last_run: Optional[dict]


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    if DASHBOARD_INDEX.exists():
        return FileResponse(str(DASHBOARD_INDEX), media_type="text/html")
    return HTMLResponse(
        content=f"<h1>Dashboard not found</h1><p>{DASHBOARD_INDEX}</p>", status_code=404
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        version="0.1.0",
    )


@app.get("/asset-classes", response_model=AssetClassesResponse)
async def get_asset_classes():
    return AssetClassesResponse(
        asset_classes=[ac.value for ac in AssetClass],
        available=event_store.get_asset_classes(),
    )


@app.get("/runs")
async def list_runs(
    asset_class: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    runs = event_store.list_runs(asset_class=asset_class, limit=limit, offset=offset)
    return {"runs": runs, "count": len(runs)}


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = event_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    events = event_store.list_events_by_run(run_id)
    return {"run": run, "events": events, "event_count": len(events)}


@app.get("/events")
async def list_events(
    run_id: Optional[str] = Query(None),
    asset_class: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    events = event_store.list_events(
        run_id=run_id,
        asset_class=asset_class,
        event_type=event_type,
        since=since,
        limit=limit,
    )
    return {"events": events, "count": len(events)}


@app.get("/decisions")
async def list_decisions(
    asset_class: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    decisions = event_store.list_decisions(
        asset_class=asset_class, since=since, limit=limit
    )
    return {"decisions": decisions, "count": len(decisions)}


@app.get("/errors")
async def list_errors(
    asset_class: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    errors = event_store.list_errors(asset_class=asset_class, since=since, limit=limit)
    return {"errors": errors, "count": len(errors)}


@app.get("/stats")
async def get_stats(asset_class: Optional[str] = Query(None)):
    stats = event_store.get_stats(asset_class=asset_class)
    last_heartbeat = event_store.get_last_heartbeat(asset_class=asset_class)
    runs = event_store.list_runs(asset_class=asset_class, limit=1)
    return StatsResponse(
        total_runs=stats["total_runs"],
        total_events=stats["total_events"],
        last_heartbeat=last_heartbeat,
        last_run=runs[0] if runs else None,
    )
