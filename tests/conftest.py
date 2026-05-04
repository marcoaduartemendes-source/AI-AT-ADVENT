"""Pytest setup for the AI-AT-ADVENT trading system.

Mirrors how `python src/run_orchestrator.py` resolves imports: the
src/ directory is prepended to sys.path so `from brokers.base import …`
and `from strategy_engine.orchestrator import …` work without us
having to ship a setup.py just to please pytest.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tests must NEVER hit the real trading DB. Override the env var so
# every PerformanceTracker() opens a tmpdir copy.
os.environ.setdefault("PYTEST_RUNNING", "1")


# Auto-isolate the alert-dedup DB for every test. Without this, the
# new common.alerts dedup cache (which persists in
# data/alert_dedup.db by default) makes tests order-dependent: if a
# message appears in test A then again in test B, B's call gets
# suppressed and assertions about delivery break.
import pytest


@pytest.fixture(autouse=True)
def _isolate_alert_dedup(tmp_path, monkeypatch):
    """Each test gets a fresh dedup DB in its own tmp_path. Removes
    cross-test leakage entirely.

    Cooldown defaults to 0 (dedup disabled) so existing alert tests
    that send the same message multiple times to exercise the rate
    limiter still work. The dedup-specific tests in
    tests/test_alert_dedup.py override this with their own cooldown."""
    monkeypatch.setenv(
        "ALERT_DEDUP_DB", str(tmp_path / "alert_dedup.db"),
    )
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "0")
    yield
