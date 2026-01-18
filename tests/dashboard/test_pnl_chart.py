"""Tests for PnL chart component."""
from __future__ import annotations


def test_import_pnl_chart():
    from services.dashboard.components import pnl_chart

    assert hasattr(pnl_chart, "render_pnl_chart")
    assert hasattr(pnl_chart, "render_pnl_summary")
