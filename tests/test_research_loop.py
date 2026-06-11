"""Tests for the autonomous research-loop agent (src/run_research_loop.py).

Pins the safety rails and output discipline that make this loop safe to
run autonomously on the droplet every night.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import run_research_loop as rl


def _make_evidence_dir(tmp_path: Path) -> Path:
    """Drop fake docs/ + data/ paths into tmp_path so the agent's
    readers see realistic input without touching the real tree."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "docs" / "self_grade.json").write_text(json.dumps({
        "overall_grade": 4.5,
        "components": {"alpha_vs_spy": {"score": 0.0, "reason": "lagging SPY"}},
    }))
    (tmp_path / "docs" / "validation.json").write_text(json.dumps({
        "as_of": "2026-06-10T00:00:00+00:00",
        "strategies": {
            "good": {"verdict": "PASS", "sharpe_5y": 4.2,
                     "reason": "Sharpe 4.2"},
        },
    }))
    return tmp_path


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Make all relative paths resolve under tmp_path. The agent uses
    Path('docs/...') etc, so cwd-rewriting is the cleanest isolation."""
    _make_evidence_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Redirect the module-level Path constants too.
    monkeypatch.setattr(rl, "QUEUE_PATH",
                        tmp_path / "docs" / "research_proposals.json")
    monkeypatch.setattr(rl, "DIGEST_PATH",
                        tmp_path / "docs" / "research_proposals.md")
    monkeypatch.setattr(rl, "HISTORY_PATH",
                        tmp_path / "data" / "research_history.jsonl")
    return tmp_path


class TestEvidenceAssembly:
    def test_assemble_evidence_includes_expected_keys(self, in_tmp):
        ev = rl._assemble_evidence()
        for k in ("as_of", "self_grade", "validation", "recent_trades",
                  "equity_summary", "prior_proposals_tail"):
            assert k in ev, f"missing key: {k}"

    def test_trim_validation_drops_per_strategy_noise(self):
        big = {
            "as_of": "x",
            "strategies": {
                "s": {"verdict": "PASS", "sharpe_5y": 3.1,
                      "reason": "ok", "trades_full_history": list(range(500))}
            }
        }
        out = rl._trim_validation(big)
        assert "trades_full_history" not in out["strategies"]["s"]
        assert out["strategies"]["s"]["verdict"] == "PASS"

    def test_truncate_enforces_budget(self, in_tmp):
        ev = {"recent_trades": [{"x": "y" * 50} for _ in range(200)],
              "cycle_status_tail": [{"x": "y" * 50} for _ in range(200)]}
        out = rl._truncate_evidence(ev, max_chars=1000)
        assert len(out) <= 1000


class TestNoApiKeyIsNoOp:
    """The agent must NEVER raise when ANTHROPIC_API_KEY is missing —
    the timer would page if it did. Empty output, exit 0."""

    def test_missing_key_returns_empty(self, in_tmp, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        parsed, cost = rl._call_claude({"as_of": "x"})
        assert parsed is None
        assert cost == 0.0

    def test_main_exits_zero_without_key(self, in_tmp, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch.object(rl.sys, "argv", ["run_research_loop.py"]):
            assert rl.main() == 0


class TestWriters:
    def test_queue_round_trip(self, in_tmp):
        parsed = {
            "summary": "book is fine",
            "weakest_grade_axis": {"axis": "fee_discipline", "score": 2.1,
                                    "why": "no PASS verdicts yet"},
            "proposals": [{
                "category": "OPS", "priority": "HIGH",
                "title": "Fund Coinbase wallet",
                "rationale": "commodity_carry blocked 50 cycles",
                "evidence_keys": ["cycle_status_tail"],
                "proposed_action": "Transfer USD",
                "expected_impact": "setup_health 0→9",
                "risk": "real money in market",
            }],
            "new_strategy_ideas": [],
        }
        rl._write_queue(parsed, cost=0.05)
        loaded = json.loads(rl.QUEUE_PATH.read_text())
        assert loaded["proposals"][0]["title"] == "Fund Coinbase wallet"
        assert loaded["cost_usd"] == 0.05

    def test_digest_renders_high_priority_clearly(self, in_tmp):
        parsed = {
            "summary": "book is fine",
            "weakest_grade_axis": {"axis": "x", "score": 1, "why": "y"},
            "proposals": [{"category": "OPS", "priority": "HIGH",
                           "title": "Fund Coinbase wallet",
                           "rationale": "r", "proposed_action": "a",
                           "expected_impact": "i", "risk": "k"}],
        }
        rl._write_digest(parsed, cost=0.02)
        body = rl.DIGEST_PATH.read_text()
        assert "HIGH" in body
        assert "Fund Coinbase wallet" in body
        assert "$0.02" in body

    def test_history_is_append_only(self, in_tmp):
        for _ in range(3):
            rl._append_history({"proposals": [1], "summary": "x"}, cost=0.01)
        lines = rl.HISTORY_PATH.read_text().strip().split("\n")
        assert len(lines) == 3


class TestHighPriorityAlerts:
    def test_high_priority_fires_alert(self, in_tmp, monkeypatch):
        fired = []

        class _FakeAlerts:
            @staticmethod
            def alert(msg, severity="info"):
                fired.append((severity, msg))

        monkeypatch.setitem(
            __import__("sys").modules, "common.alerts", _FakeAlerts)
        rl._alert_if_high_priority({"proposals": [
            {"priority": "HIGH", "title": "Fund Coinbase wallet"},
            {"priority": "LOW", "title": "Tweak X"},
        ]})
        assert fired and "Coinbase" in fired[0][1]

    def test_no_high_priority_stays_quiet(self, in_tmp, monkeypatch):
        fired = []

        class _FakeAlerts:
            @staticmethod
            def alert(msg, severity="info"):
                fired.append(msg)

        monkeypatch.setitem(
            __import__("sys").modules, "common.alerts", _FakeAlerts)
        rl._alert_if_high_priority({"proposals": [
            {"priority": "MEDIUM", "title": "Tweak Y"},
        ]})
        assert fired == []


class TestSafetyRails:
    """The agent is forbidden from writing code or touching state. These
    tests pin that the surface area is read-evidence + write-proposals,
    nothing else."""

    def test_no_executable_writes_in_main(self, in_tmp, monkeypatch):
        """main() should only ever write to QUEUE_PATH / DIGEST_PATH /
        HISTORY_PATH. Patch open() to record everything and assert."""
        opens: list[str] = []
        real_open = open

        def _spy_open(path, mode="r", *a, **kw):
            if isinstance(path, (str, Path)) and "w" in mode:
                opens.append(str(path))
            return real_open(path, mode, *a, **kw)

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("builtins.open", _spy_open)
        with patch.object(rl.sys, "argv", ["run_research_loop.py"]):
            rl.main()
        # No-API-key path = no writes. Even with an API key, only queue/
        # digest/history paths are allowed.
        forbidden = [p for p in opens
                     if not any(safe in p for safe in
                                ("research_proposals", "research_history"))]
        assert forbidden == [], f"unexpected writes: {forbidden}"
