from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path, marker: str = "pyproject.toml", max_up: int = 6) -> Path:
    """
    Find repository root by searching upward for a marker file (default: pyproject.toml).

    Args:
        start: starting path (file or directory)
        marker: marker file name
        max_up: maximum levels to traverse upward

    Returns:
        Path to repository root directory
    """
    p = start.resolve()
    if p.is_file():
        p = p.parent

    for _ in range(max_up + 1):
        if (p / marker).exists():
            return p
        if p.parent == p:
            break
        p = p.parent

    # Deterministic fallback (should not happen in normal repo layouts)
    return start.resolve() if start.is_dir() else start.resolve().parent
