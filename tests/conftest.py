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
