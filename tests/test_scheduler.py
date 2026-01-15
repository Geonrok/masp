"""
DailyScheduler tests.
"""
from __future__ import annotations

import asyncio
import signal
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.triggers.cron import CronTrigger

from services.scheduler import DailyScheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "schedule_config.yaml"


class DummyRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run_once(self) -> None:
        self.calls += 1


def test_cron_trigger_defaults():
    scheduler = DailyScheduler(DummyRunner(), config_path=str(CONFIG_PATH))
    expected = CronTrigger(hour=9, minute=0, timezone="Asia/Seoul")
    assert str(scheduler.trigger) == str(expected)


def test_jitter_from_config():
    scheduler = DailyScheduler(DummyRunner(), config_path=str(CONFIG_PATH))
    assert scheduler.jitter == 60


def test_job_missed_listener_registered():
    scheduler = DailyScheduler(DummyRunner(), config_path=str(CONFIG_PATH))
    scheduler._configure_scheduler()
    listeners = scheduler._scheduler._listeners
    assert any(mask & EVENT_JOB_MISSED for (_cb, mask) in listeners)


@pytest.mark.asyncio
async def test_run_job_uses_executor(monkeypatch):
    runner = DummyRunner()
    scheduler = DailyScheduler(runner, config_path=str(CONFIG_PATH))
    called: dict = {}

    async def fake_run_in_executor(executor, func, *args):
        called["executor"] = executor
        called["func"] = func
        called["args"] = args
        return None

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "run_in_executor", fake_run_in_executor)

    await scheduler._run_job()

    assert called["executor"] is None
    assert called["func"] == runner.run_once


@pytest.mark.asyncio
async def test_lock_prevents_overlap():
    calls: list[str] = []
    started = threading.Event()
    block = threading.Event()

    def run_once():
        calls.append("start")
        started.set()
        block.wait(timeout=1)
        calls.append("end")

    runner = SimpleNamespace(run_once=run_once)
    scheduler = DailyScheduler(runner, config_path=str(CONFIG_PATH))

    task1 = asyncio.create_task(scheduler._run_job())
    await asyncio.to_thread(started.wait, 1)

    task2 = asyncio.create_task(scheduler._run_job())
    await asyncio.sleep(0.1)

    assert calls.count("start") == 1

    block.set()
    await asyncio.gather(task1, task2)

    assert calls.count("start") == 2


def test_stop_calls_shutdown_wait_true(monkeypatch):
    scheduler = DailyScheduler(DummyRunner(), config_path=str(CONFIG_PATH))
    scheduler._scheduler.shutdown = MagicMock()

    scheduler.stop()

    scheduler._scheduler.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_run_forever_registers_signal_handlers(monkeypatch):
    scheduler = DailyScheduler(DummyRunner(), config_path=str(CONFIG_PATH))
    registered = []

    def fake_signal(sig, handler):
        registered.append(sig)

    monkeypatch.setattr(signal, "signal", fake_signal)
    scheduler._scheduler.start = MagicMock()
    scheduler._scheduler.shutdown = MagicMock()

    task = asyncio.create_task(scheduler.run_forever())
    await asyncio.sleep(0.1)
    scheduler.stop()
    await task

    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered


@pytest.mark.asyncio
async def test_run_forever_shutdowns_wait_true(monkeypatch):
    scheduler = DailyScheduler(DummyRunner(), config_path=str(CONFIG_PATH))
    scheduler._scheduler.start = MagicMock()
    scheduler._scheduler.shutdown = MagicMock()

    task = asyncio.create_task(scheduler.run_forever())
    await asyncio.sleep(0.1)
    scheduler._running = False
    await task

    scheduler._scheduler.shutdown.assert_called_once_with(wait=True)
