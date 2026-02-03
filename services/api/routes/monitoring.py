"""
Monitoring API routes.

Provides endpoints for:
- System metrics (CPU, Memory, Disk)
- Trading metrics (orders, latency)
- Alert management
- Health dashboard data
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from services.api.middleware.auth import verify_admin_token

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class SystemMetricsResponse(BaseModel):
    """System metrics response."""

    status: str
    metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    timestamp: str


class TradingMetricsResponse(BaseModel):
    """Trading metrics response."""

    total_orders: int
    success_rate: float
    exchanges: Dict[str, Any]
    timestamp: str


class AlertResponse(BaseModel):
    """Single alert response."""

    id: str
    category: str
    severity: str
    title: str
    message: str
    timestamp: str
    acknowledged: bool
    resolved: bool


class AlertsListResponse(BaseModel):
    """Alerts list response."""

    alerts: List[AlertResponse]
    total: int
    active_count: int


class AlertAckRequest(BaseModel):
    """Alert acknowledgment request."""

    alert_id: str
    acknowledged_by: Optional[str] = "api"


class MonitoringSummaryResponse(BaseModel):
    """Overall monitoring summary."""

    system_status: str
    trading_summary: Dict[str, Any]
    active_alerts_count: int
    timestamp: str


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/system", response_model=SystemMetricsResponse)
async def get_system_metrics(
    _: str = Depends(verify_admin_token),
):
    """
    Get current system resource metrics.

    Returns CPU, memory, disk usage and system alerts.
    """
    try:
        from libs.monitoring.system_monitor import get_system_monitor

        monitor = get_system_monitor()
        status_data = monitor.get_status()

        return SystemMetricsResponse(
            status=status_data.get("status", "unknown"),
            metrics=status_data.get("metrics", {}),
            alerts=status_data.get("alerts", []),
            timestamp=datetime.now().isoformat(),
        )
    except ImportError:
        return SystemMetricsResponse(
            status="unavailable",
            metrics={},
            alerts=[],
            timestamp=datetime.now().isoformat(),
        )


@router.get("/system/history")
async def get_system_history(
    minutes: int = Query(default=30, ge=1, le=1440),
    _: str = Depends(verify_admin_token),
):
    """
    Get system metrics history.

    Args:
        minutes: Number of minutes of history (1-1440)
    """
    try:
        from libs.monitoring.system_monitor import get_system_monitor

        monitor = get_system_monitor()
        history = monitor.get_history(minutes=minutes)

        return {
            "history": history,
            "period_minutes": minutes,
            "count": len(history),
        }
    except ImportError:
        return {"history": [], "period_minutes": minutes, "count": 0}


@router.get("/trading", response_model=TradingMetricsResponse)
async def get_trading_metrics(
    minutes: int = Query(default=60, ge=1, le=1440),
    _: str = Depends(verify_admin_token),
):
    """
    Get aggregated trading metrics.

    Args:
        minutes: Lookback period in minutes
    """
    try:
        from libs.monitoring.trading_metrics import get_trading_metrics

        aggregator = get_trading_metrics()
        summary = aggregator.get_summary()

        return TradingMetricsResponse(
            total_orders=summary.get("total_orders", 0),
            success_rate=summary.get("overall_success_rate", 0.0),
            exchanges=summary.get("by_exchange", {}),
            timestamp=summary.get("timestamp", datetime.now().isoformat()),
        )
    except ImportError:
        return TradingMetricsResponse(
            total_orders=0,
            success_rate=0.0,
            exchanges={},
            timestamp=datetime.now().isoformat(),
        )


@router.get("/trading/{exchange}")
async def get_exchange_metrics(
    exchange: str,
    minutes: int = Query(default=60, ge=1, le=1440),
    _: str = Depends(verify_admin_token),
):
    """
    Get trading metrics for a specific exchange.

    Args:
        exchange: Exchange name
        minutes: Lookback period in minutes
    """
    try:
        from libs.monitoring.trading_metrics import get_trading_metrics

        aggregator = get_trading_metrics()
        stats = aggregator.get_exchange_stats(exchange, minutes)

        if stats.get("total_orders", 0) == 0:
            return {
                "exchange": exchange,
                "message": "No orders recorded",
                "period_minutes": minutes,
            }

        return stats
    except ImportError:
        raise HTTPException(status_code=501, detail="Trading metrics not available")


@router.get("/alerts", response_model=AlertsListResponse)
async def get_alerts(
    active_only: bool = Query(default=True),
    hours: int = Query(default=24, ge=1, le=168),
    category: Optional[str] = None,
    severity: Optional[str] = None,
    _: str = Depends(verify_admin_token),
):
    """
    Get alerts list.

    Args:
        active_only: Only show active (unresolved) alerts
        hours: Hours of history to include
        category: Filter by category (system, trading, risk, connectivity, security)
        severity: Filter by minimum severity (info, warning, error, critical)
    """
    try:
        from libs.monitoring.alert_manager import (
            AlertCategory,
            AlertSeverity,
            get_alert_manager,
        )

        manager = get_alert_manager()

        # Parse filters
        cat_filter = None
        if category:
            try:
                cat_filter = AlertCategory(category.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid category: {category}"
                )

        sev_filter = None
        if severity:
            try:
                sev_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid severity: {severity}"
                )

        # Get alerts
        if active_only:
            alerts = manager.get_active_alerts(
                category=cat_filter, min_severity=sev_filter
            )
        else:
            alerts = manager.get_alert_history(hours=hours, category=cat_filter)
            if sev_filter:
                severity_order = list(AlertSeverity)
                min_idx = severity_order.index(sev_filter)
                alerts = [
                    a for a in alerts if severity_order.index(a.severity) >= min_idx
                ]

        # Format response
        alert_responses = [
            AlertResponse(
                id=a.id,
                category=a.category.value,
                severity=a.severity.value,
                title=a.title,
                message=a.message,
                timestamp=a.timestamp.isoformat(),
                acknowledged=a.acknowledged,
                resolved=a.resolved,
            )
            for a in alerts
        ]

        active_count = sum(1 for a in alerts if not a.resolved)

        return AlertsListResponse(
            alerts=alert_responses,
            total=len(alert_responses),
            active_count=active_count,
        )
    except ImportError:
        return AlertsListResponse(alerts=[], total=0, active_count=0)


@router.post("/alerts/acknowledge")
async def acknowledge_alert(
    request: AlertAckRequest,
    _: str = Depends(verify_admin_token),
):
    """
    Acknowledge an alert.

    Args:
        request: Alert acknowledgment request
    """
    try:
        from libs.monitoring.alert_manager import get_alert_manager

        manager = get_alert_manager()
        success = manager.acknowledge(
            request.alert_id, by=request.acknowledged_by or "api"
        )

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Alert not found: {request.alert_id}"
            )

        return {"success": True, "alert_id": request.alert_id}
    except ImportError:
        raise HTTPException(status_code=501, detail="Alert manager not available")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    _: str = Depends(verify_admin_token),
):
    """
    Resolve an alert.

    Args:
        alert_id: Alert ID to resolve
    """
    try:
        from libs.monitoring.alert_manager import get_alert_manager

        manager = get_alert_manager()
        success = manager.resolve(alert_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")

        return {"success": True, "alert_id": alert_id}
    except ImportError:
        raise HTTPException(status_code=501, detail="Alert manager not available")


@router.get("/alerts/stats")
async def get_alert_stats(
    _: str = Depends(verify_admin_token),
):
    """Get alert statistics."""
    try:
        from libs.monitoring.alert_manager import get_alert_manager

        manager = get_alert_manager()
        stats = manager.get_stats()

        return stats
    except ImportError:
        return {"error": "Alert manager not available"}


@router.get("/summary", response_model=MonitoringSummaryResponse)
async def get_monitoring_summary(
    _: str = Depends(verify_admin_token),
):
    """
    Get overall monitoring summary.

    Returns system status, trading summary, and active alerts count.
    """
    system_status = "unknown"
    trading_summary = {}
    active_alerts_count = 0

    # System status
    try:
        from libs.monitoring.system_monitor import get_system_monitor

        monitor = get_system_monitor()
        system_status = monitor.get_status().get("status", "unknown")
    except ImportError:
        pass

    # Trading summary
    try:
        from libs.monitoring.trading_metrics import get_trading_metrics

        aggregator = get_trading_metrics()
        summary = aggregator.get_summary()
        trading_summary = {
            "total_orders": summary.get("total_orders", 0),
            "success_rate": summary.get("overall_success_rate", 0.0),
            "exchanges": summary.get("exchanges", []),
        }
    except ImportError:
        pass

    # Alerts
    try:
        from libs.monitoring.alert_manager import get_alert_manager

        manager = get_alert_manager()
        active_alerts_count = manager.get_stats().get("active_alerts", 0)
    except ImportError:
        pass

    return MonitoringSummaryResponse(
        system_status=system_status,
        trading_summary=trading_summary,
        active_alerts_count=active_alerts_count,
        timestamp=datetime.now().isoformat(),
    )


@router.get("/circuit-breaker")
async def get_circuit_breaker_status(
    _: str = Depends(verify_admin_token),
):
    """Get circuit breaker status if available."""
    try:
        # This would integrate with the circuit breaker module
        return {
            "status": "unavailable",
            "message": "Circuit breaker not configured in this context",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
