from __future__ import annotations

from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    print(f"[FORMAT_AUDIT][FAIL] {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"[FORMAT_AUDIT][OK] {msg}")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    # 1) requirements.txt: line count check (non-empty lines)
    req = REPO / "requirements.txt"
    if not req.exists():
        fail("requirements.txt missing")

    txt = req.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in txt.splitlines() if ln.strip()]  # Filter out empty lines
    if len(lines) != 17:
        fail(f"requirements.txt must be exactly 17 non-empty lines, got {len(lines)}")

    # Check raw text for concatenation patterns (not joined lines)
    if "pydantic>=2.0.0fastapi" in txt:
        fail("requirements.txt looks concatenated (pydantic...fastapi)")
    if any(" " in ln for ln in lines):
        fail("requirements.txt contains spaces (unexpected)")
    ok("requirements.txt looks OK (17 lines)")

    # 2) pyproject.toml sanity: must contain key section headers
    pyproj = REPO / "pyproject.toml"
    if not pyproj.exists():
        fail("pyproject.toml missing")

    t = read_text(pyproj)
    for header in ("[build-system]", "[project]", "[tool.setuptools.packages.find]"):
        if header not in t:
            fail(f"pyproject.toml missing section header: {header}")
    ok("pyproject.toml section headers OK")

    # 3) CMD scripts: must have line breaks
    cmd_files = [
        REPO / "scripts" / "install.cmd",
        REPO / "scripts" / "smoke_test.cmd",
        REPO / "scripts" / "start_api.cmd",
    ]
    for p in cmd_files:
        if not p.exists():
            fail(f"{p} missing")
        s = read_text(p)

        # Check for specific concatenation patterns (same line only)
        if "echo.echo" in s:
            fail(f"{p.name}: contains 'echo.echo' (likely missing newlines)")

        # Check each line for concatenation patterns
        for line in s.splitlines():
            # pushd and echo on same line
            if re.search(r"pushd.*echo", line, flags=re.IGNORECASE):
                fail(f"{p.name}: 'pushd ... echo' on same line")
            # %CD% and echo concatenated
            if "%CD%echo" in line:
                fail(f"{p.name}: '%CD%echo' pattern suggests concatenation")

        # must start with 3 specific lines
        head = s.splitlines()[:3]
        if len(head) < 3:
            fail(f"{p.name}: too few lines (likely concatenated)")
        if head[0].strip().lower() != "@echo off":
            fail(f"{p.name}: first line must be '@echo off'")
        if head[1].strip().lower() != "setlocal":
            fail(f"{p.name}: second line must be 'setlocal'")
        if not head[2].strip().lower().startswith('pushd "%~dp0'):
            fail(f"{p.name}: third line must start with pushd")

        ok(f"{p.name} basic newline patterns OK")

    ok("FORMAT_AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
