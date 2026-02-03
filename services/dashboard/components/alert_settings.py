"""Alert settings component for managing notification rules."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Session state key prefix
_KEY_PREFIX = "alert_settings."


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def _get_demo_rules() -> List[Dict[str, Any]]:
    """Generate demo alert rules."""
    return [
        {
            "name": "default",
            "enabled": True,
            "alert_types": [],
            "min_priority": "NORMAL",
            "exchanges": [],
            "symbols": [],
            "cooldown_seconds": 0,
            "aggregate_count": 0,
        },
        {
            "name": "critical_only",
            "enabled": True,
            "alert_types": ["ERROR", "ANOMALY"],
            "min_priority": "HIGH",
            "exchanges": [],
            "symbols": [],
            "cooldown_seconds": 60,
            "aggregate_count": 0,
        },
        {
            "name": "trade_aggregator",
            "enabled": True,
            "alert_types": ["TRADE"],
            "min_priority": "LOW",
            "exchanges": [],
            "symbols": [],
            "cooldown_seconds": 0,
            "aggregate_count": 5,
        },
    ]


def _get_priority_options() -> List[str]:
    """Return priority options."""
    return ["LOW", "NORMAL", "HIGH", "CRITICAL"]


def _get_alert_type_options() -> List[str]:
    """Return alert type options."""
    return ["TRADE", "SIGNAL", "ERROR", "SYSTEM", "DAILY", "ANOMALY"]


def render_alert_rule_card(
    rule: Dict[str, Any],
    on_toggle: Optional[Callable[[str, bool], None]] = None,
    on_delete: Optional[Callable[[str], None]] = None,
) -> None:
    """Render a single alert rule card.

    Args:
        rule: Rule data dictionary
        on_toggle: Callback for enable/disable toggle
        on_delete: Callback for delete action
    """
    name = rule.get("name", "")
    enabled = rule.get("enabled", True)

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            status = "Enabled" if enabled else "Disabled"
            st.markdown(f"**{name}** ({status})")

        with col2:
            if on_toggle:
                new_state = st.toggle(
                    "Enable",
                    value=enabled,
                    key=_key(f"toggle_{name}"),
                    label_visibility="collapsed",
                )
                if new_state != enabled:
                    on_toggle(name, new_state)

        with col3:
            if on_delete and name != "default":
                if st.button("Delete", key=_key(f"delete_{name}"), type="secondary"):
                    on_delete(name)

        # Rule details
        with st.expander("Details", expanded=False):
            col_a, col_b = st.columns(2)

            with col_a:
                types = rule.get("alert_types", [])
                st.text(f"Types: {', '.join(types) if types else 'All'}")

                priority = rule.get("min_priority", "LOW")
                st.text(f"Min Priority: {priority}")

            with col_b:
                cooldown = rule.get("cooldown_seconds", 0)
                st.text(f"Cooldown: {cooldown}s")

                aggregate = rule.get("aggregate_count", 0)
                st.text(f"Aggregate: {aggregate if aggregate > 0 else 'Off'}")

            exchanges = rule.get("exchanges", [])
            if exchanges:
                st.text(f"Exchanges: {', '.join(exchanges)}")

            symbols = rule.get("symbols", [])
            if symbols:
                st.text(f"Symbols: {', '.join(symbols)}")

        st.divider()


def render_new_rule_form(
    on_create: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Render form for creating new alert rule.

    Args:
        on_create: Callback when new rule is created
    """
    with st.expander("Create New Rule", expanded=False):
        name = st.text_input("Rule Name", key=_key("new_name"))

        col1, col2 = st.columns(2)

        with col1:
            types = st.multiselect(
                "Alert Types (empty = all)",
                options=_get_alert_type_options(),
                key=_key("new_types"),
            )

            priority = st.selectbox(
                "Minimum Priority",
                options=_get_priority_options(),
                index=1,  # NORMAL
                key=_key("new_priority"),
            )

        with col2:
            cooldown = st.number_input(
                "Cooldown (seconds)",
                min_value=0,
                max_value=3600,
                value=0,
                key=_key("new_cooldown"),
            )

            aggregate = st.number_input(
                "Aggregate Count (0 = off)",
                min_value=0,
                max_value=100,
                value=0,
                key=_key("new_aggregate"),
            )

        exchanges_str = st.text_input(
            "Exchanges (comma-separated, empty = all)",
            key=_key("new_exchanges"),
        )

        symbols_str = st.text_input(
            "Symbols (comma-separated, empty = all)",
            key=_key("new_symbols"),
        )

        if st.button("Create Rule", key=_key("create_btn"), type="primary"):
            if not name:
                st.error("Rule name is required")
            elif on_create:
                new_rule = {
                    "name": name,
                    "enabled": True,
                    "alert_types": types,
                    "min_priority": priority,
                    "exchanges": [
                        e.strip() for e in exchanges_str.split(",") if e.strip()
                    ],
                    "symbols": [s.strip() for s in symbols_str.split(",") if s.strip()],
                    "cooldown_seconds": cooldown,
                    "aggregate_count": aggregate,
                }
                on_create(new_rule)
                st.success(f"Rule '{name}' created")
                st.rerun()


def render_anomaly_thresholds(
    thresholds: Optional[Dict[str, float]] = None,
    on_save: Optional[Callable[[Dict[str, float]], None]] = None,
) -> None:
    """Render anomaly detection threshold settings.

    Args:
        thresholds: Current threshold values
        on_save: Callback when thresholds are saved
    """
    if thresholds is None:
        thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "disk_percent": 95.0,
            "api_error_rate": 0.1,
            "response_time_ms": 5000.0,
        }

    st.subheader("Anomaly Detection Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        cpu = st.slider(
            "CPU Usage (%)",
            min_value=50.0,
            max_value=100.0,
            value=thresholds.get("cpu_percent", 90.0),
            step=5.0,
            key=_key("thresh_cpu"),
        )

        memory = st.slider(
            "Memory Usage (%)",
            min_value=50.0,
            max_value=100.0,
            value=thresholds.get("memory_percent", 90.0),
            step=5.0,
            key=_key("thresh_memory"),
        )

        disk = st.slider(
            "Disk Usage (%)",
            min_value=50.0,
            max_value=100.0,
            value=thresholds.get("disk_percent", 95.0),
            step=5.0,
            key=_key("thresh_disk"),
        )

    with col2:
        error_rate = st.slider(
            "API Error Rate (%)",
            min_value=1.0,
            max_value=50.0,
            value=thresholds.get("api_error_rate", 0.1) * 100,
            step=1.0,
            key=_key("thresh_error"),
        )

        response_time = st.slider(
            "Response Time (ms)",
            min_value=1000,
            max_value=30000,
            value=int(thresholds.get("response_time_ms", 5000)),
            step=500,
            key=_key("thresh_response"),
        )

    if on_save:
        if st.button("Save Thresholds", key=_key("save_thresh"), type="primary"):
            new_thresholds = {
                "cpu_percent": cpu,
                "memory_percent": memory,
                "disk_percent": disk,
                "api_error_rate": error_rate / 100,
                "response_time_ms": float(response_time),
            }
            on_save(new_thresholds)
            st.success("Thresholds saved")


def render_alert_settings(
    rules: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    on_rule_toggle: Optional[Callable[[str, bool], None]] = None,
    on_rule_delete: Optional[Callable[[str], None]] = None,
    on_rule_create: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_thresholds_save: Optional[Callable[[Dict[str, float]], None]] = None,
) -> None:
    """Render complete alert settings panel.

    Args:
        rules: List of alert rules
        thresholds: Anomaly detection thresholds
        on_rule_toggle: Callback for rule enable/disable
        on_rule_delete: Callback for rule deletion
        on_rule_create: Callback for rule creation
        on_thresholds_save: Callback for threshold updates
    """
    st.subheader("Alert Settings")

    if rules is None:
        st.caption("Demo Data - Connect to AlertManager for real settings")
        rules = _get_demo_rules()

    # Tabs for rules and thresholds
    tab1, tab2 = st.tabs(["Alert Rules", "Anomaly Thresholds"])

    with tab1:
        st.caption(f"{len(rules)} rules configured")

        # Existing rules
        for rule in rules:
            render_alert_rule_card(
                rule=rule,
                on_toggle=on_rule_toggle,
                on_delete=on_rule_delete,
            )

        # New rule form
        render_new_rule_form(on_create=on_rule_create)

    with tab2:
        render_anomaly_thresholds(
            thresholds=thresholds,
            on_save=on_thresholds_save,
        )
