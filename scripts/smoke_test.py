#!/usr/bin/env python
"""
Smoke Test for Multi-Asset Strategy Platform.
Exit: 0=PASS, 1=FAIL
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.core.paths import find_repo_root

REPO_ROOT = find_repo_root(Path(__file__))
STORAGE_PATH = REPO_ROOT / "storage" / "local.db"
os.chdir(REPO_ROOT)


def log(msg):
    print(f">> {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def step1_compile():
    log("Step 1: Compile check")
    r = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "libs", "apps"], capture_output=True
    )
    if r.returncode != 0:
        fail("Compile failed")
        return False
    ok("Compile")
    return True


def step2_import():
    log("Step 2: Import check")
    code = (
        "from libs.core import Config, EventStore; "
        "from libs.core.paths import find_repo_root; "
        "from libs.strategies import BaseStrategy; "
        "from apps.api_server.main import app; "
        "print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"Import failed: {r.stderr[:200]}")
        return False
    ok("Import")
    return True


def step3_services():
    log("Step 3: Run 4 services")
    services = [
        "crypto_spot_service",
        "crypto_futures_service",
        "kr_stock_spot_service",
        "kr_stock_futures_service",
    ]
    for svc in services:
        r = subprocess.run(
            [sys.executable, "-m", f"apps.{svc}", "--once"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            fail(f"{svc} failed: {r.stderr[:100]}")
            return False
        ok(svc)
    return True


def step4_database():
    log("Step 4: Database check")
    if not STORAGE_PATH.exists():
        fail(f"DB not found: {STORAGE_PATH}")
        return False
    ok(f"DB exists ({STORAGE_PATH.stat().st_size} bytes)")
    return True


def step5_api():
    log("Step 5: API server test")
    proc = subprocess.Popen(
        [sys.executable, "-m", "apps.api_server", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(10):
            time.sleep(1)
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/health")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "healthy":
                        ok(f"/health: {data}")
                        return True
            except urllib.error.URLError:
                continue
        fail("/health not responding")
        return False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def step6_dashboard_file():
    log("Step 6: Dashboard file check")
    f = REPO_ROOT / "web" / "dashboard" / "index.html"
    if not f.exists():
        fail(f"Dashboard not found: {f}")
        return False
    ok(f"Dashboard exists ({f.stat().st_size} bytes)")
    return True


def step7_dashboard_route():
    log("Step 7: Dashboard route test")
    proc = subprocess.Popen(
        [sys.executable, "-m", "apps.api_server", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(2)
        try:
            req = urllib.request.Request("http://127.0.0.1:8001/")
            with urllib.request.urlopen(req, timeout=5) as resp:
                content = resp.read().decode()
                if "<!DOCTYPE html>" in content:
                    ok("/ returns HTML")
                    return True
                fail("/ not HTML")
                return False
        except urllib.error.URLError as e:
            fail(f"/ failed: {e}")
            return False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main():
    print("=" * 50)
    print("Multi-Asset Strategy Platform - Smoke Test")
    print("=" * 50)
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"STORAGE_PATH: {STORAGE_PATH}")
    print(f"Python: {sys.executable}")
    print()

    results = [
        ("Compile", step1_compile()),
        ("Import", step2_import()),
        ("Services", step3_services()),
        ("Database", step4_database()),
        ("API", step5_api()),
        ("Dashboard File", step6_dashboard_file()),
        ("Dashboard Route", step7_dashboard_route()),
    ]

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_pass = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("SMOKE TEST PASSED")
        return 0
    else:
        print("SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
