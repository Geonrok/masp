"""
AI Code Review Dashboard Component for Streamlit.

Displays review status and history in the MASP dashboard.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = PROJECT_ROOT / ".ai-review" / "reviews"
CONFIG_FILE = PROJECT_ROOT / ".ai-review" / "review-config.json"


def get_recent_commits(count: int = 10) -> List[Dict]:
    """Get recent git commits."""
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--format=%h|%s|%ai|%an"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "date": parts[2] if len(parts) > 2 else "",
                    "author": parts[3] if len(parts) > 3 else "",
                })
        return commits
    except Exception:
        return []


def get_review_summary(commit_hash: str) -> Optional[Dict]:
    """Get review summary for a commit."""
    summary_file = REVIEW_DIR / f"{commit_hash}-summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_review_config() -> Optional[Dict]:
    """Get review configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def status_badge(status: str) -> str:
    """Return status badge HTML."""
    colors = {
        "PASS": ("#28a745", "white"),
        "CHANGES_REQUESTED": ("#ffc107", "black"),
        "BLOCKED": ("#dc3545", "white"),
        "NOT_REVIEWED": ("#6c757d", "white"),
    }
    bg, fg = colors.get(status, ("#6c757d", "white"))
    return f'<span style="background-color:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-size:12px;">{status}</span>'


def reviewer_status_icon(status: Optional[str]) -> str:
    """Return reviewer status icon."""
    icons = {
        "PASS": "✅",
        "CHANGES_REQUESTED": "⚠️",
        "BLOCKED": "❌",
        None: "⬜",
    }
    return icons.get(status, "⬜")


def render_review_dashboard():
    """Render the review dashboard component."""
    st.header("🔍 AI Code Review Dashboard")

    # Get config
    config = get_review_config()
    if not config:
        st.warning("Review configuration not found. Run setup first.")
        return

    # Overview metrics
    commits = get_recent_commits(10)

    col1, col2, col3, col4 = st.columns(4)

    reviewed = sum(1 for c in commits if get_review_summary(c["hash"]))
    passed = sum(
        1 for c in commits
        if get_review_summary(c["hash"]) and get_review_summary(c["hash"]).get("overall_status") == "PASS"
    )
    blocked = sum(
        1 for c in commits
        if get_review_summary(c["hash"]) and get_review_summary(c["hash"]).get("overall_status") == "BLOCKED"
    )

    with col1:
        st.metric("Total Commits", len(commits))
    with col2:
        st.metric("Reviewed", f"{reviewed}/{len(commits)}")
    with col3:
        st.metric("Passed", passed, delta=f"{passed/max(reviewed,1)*100:.0f}%")
    with col4:
        st.metric("Blocked", blocked, delta=None if blocked == 0 else f"-{blocked}")

    st.divider()

    # Reviewer legend
    st.subheader("검수자 현황")

    reviewers = config.get("reviewers", {})
    cols = st.columns(len(reviewers))

    for i, (name, info) in enumerate(reviewers.items()):
        with cols[i]:
            st.markdown(f"**{info.get('name', name)}**")
            st.caption(info.get("model", ""))
            focus = info.get("focus", [])
            st.caption(", ".join(focus[:3]))

    st.divider()

    # Commit review table
    st.subheader("최근 커밋 검수 현황")

    for commit in commits:
        summary = get_review_summary(commit["hash"])

        with st.container():
            col1, col2, col3 = st.columns([1, 4, 2])

            with col1:
                st.code(commit["hash"], language=None)

            with col2:
                st.markdown(f"**{commit['message'][:50]}{'...' if len(commit['message']) > 50 else ''}**")

                if summary:
                    # Reviewer icons
                    reviewer_statuses = summary.get("reviewers", {})
                    icons = []
                    for reviewer_name in ["codex", "gemini", "opencode", "copilot"]:
                        status = reviewer_statuses.get(reviewer_name)
                        icon = reviewer_status_icon(status)
                        icons.append(f"{icon} {reviewer_name}")
                    st.caption(" | ".join(icons))

            with col3:
                if summary:
                    status = summary.get("overall_status", "NOT_REVIEWED")
                    st.markdown(status_badge(status), unsafe_allow_html=True)

                    issues = summary.get("total_issues", {})
                    st.caption(f"P0:{issues.get('p0', 0)} P1:{issues.get('p1', 0)} P2:{issues.get('p2', 0)}")
                else:
                    st.markdown(status_badge("NOT_REVIEWED"), unsafe_allow_html=True)
                    st.caption("검수 대기중")

            st.divider()

    # Quick actions
    st.subheader("빠른 작업")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 자동 라우팅 실행", use_container_width=True):
            with st.spinner("분석 중..."):
                result = subprocess.run(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                     str(PROJECT_ROOT / ".ai-review" / "auto-review.ps1"), "-DryRun"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                st.code(result.stdout or result.stderr)

    with col2:
        if st.button("📊 요약 생성", use_container_width=True):
            with st.spinner("요약 생성 중..."):
                result = subprocess.run(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                     str(PROJECT_ROOT / ".ai-review" / "review-collector.ps1"),
                     "-Mode", "summarize"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                st.success("요약이 생성되었습니다.")
                st.rerun()

    with col3:
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()


def render_review_detail(commit_hash: str):
    """Render detailed review for a specific commit."""
    st.subheader(f"검수 상세: {commit_hash}")

    summary = get_review_summary(commit_hash)
    if not summary:
        st.warning("이 커밋에 대한 검수 결과가 없습니다.")
        return

    # Overall status
    status = summary.get("overall_status", "NOT_REVIEWED")
    st.markdown(f"### 전체 상태: {status_badge(status)}", unsafe_allow_html=True)

    # Issues breakdown
    issues = summary.get("total_issues", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P0 (Blocking)", issues.get("p0", 0))
    with col2:
        st.metric("P1 (Important)", issues.get("p1", 0))
    with col3:
        st.metric("P2 (Suggestions)", issues.get("p2", 0))

    # Individual reviews
    st.divider()
    st.subheader("개별 검수 결과")

    for reviewer, status in summary.get("reviewers", {}).items():
        review_file = REVIEW_DIR / f"{commit_hash}-{reviewer}.md"
        if review_file.exists():
            with st.expander(f"{reviewer_status_icon(status)} {reviewer.upper()}: {status}"):
                with open(review_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())


if __name__ == "__main__":
    # Standalone test
    st.set_page_config(page_title="Review Dashboard", layout="wide")
    render_review_dashboard()
