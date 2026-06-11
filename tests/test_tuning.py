"""Tests for the weekly parameter-tuning loop (src/run_tuning.py).

Pins the invariants that make autonomous sweeping safe: production
constants are ALWAYS restored (even on crash), recommendations require a
meaningful Sharpe margin (anti-churn), and the output is proposals-only.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import run_tuning as rt
from backtests.runner import BacktestSummary
from strategies import high_vol_trend as hvt


def _summary(strategy: str, days: int, sharpe: float) -> BacktestSummary:
    return BacktestSummary(strategy=strategy, window_days=days,
                           n_trades=30, total_pnl_usd=1000.0, sharpe=sharpe)


_SWEEP = {
    "strategy": "high_vol_trend",
    "module": "strategies.high_vol_trend",
    "param": "VOL_TARGET_ANN",
    "candidates": [0.12, 0.17, 0.22],
}


class TestParamRestoration:
    def test_constant_restored_after_sweep(self):
        original = hvt.VOL_TARGET_ANN
        with patch.object(rt, "_run_backtest",
                          side_effect=lambda s, d: _summary(s, d, 1.0)):
            rt.run_sweep(dict(_SWEEP))
        assert hvt.VOL_TARGET_ANN == original

    def test_constant_restored_even_when_backtest_explodes(self):
        original = hvt.VOL_TARGET_ANN

        def _boom(s, d):
            raise RuntimeError("yahoo 403")
        with patch.object(rt, "_run_backtest", side_effect=_boom):
            rt.run_sweep(dict(_SWEEP))   # per-window errors are captured
        assert hvt.VOL_TARGET_ANN == original


class TestRecommendationDiscipline:
    def test_clear_winner_recommended(self):
        # 0.22 scores Sharpe 3.0 across windows; current 0.17 scores 1.0.
        def _fake(strategy, days):
            val = hvt.VOL_TARGET_ANN
            return _summary(strategy, days, 3.0 if val == 0.22 else 1.0)
        with patch.object(rt, "_run_backtest", side_effect=_fake):
            out = rt.run_sweep(dict(_SWEEP))
        assert out["best_value"] == "0.22"
        assert out["recommendation"].startswith("CONSIDER")

    def test_marginal_winner_keeps_current(self):
        # Best beats current by only 0.2 Sharpe — below the 0.5 margin;
        # recommending a change on noise is how curve-fitting happens.
        def _fake(strategy, days):
            val = hvt.VOL_TARGET_ANN
            return _summary(strategy, days, 1.2 if val == 0.22 else 1.0)
        with patch.object(rt, "_run_backtest", side_effect=_fake):
            out = rt.run_sweep(dict(_SWEEP))
        assert out["recommendation"] == "KEEP_CURRENT"

    def test_current_value_always_in_comparison(self):
        with patch.object(rt, "_run_backtest",
                          side_effect=lambda s, d: _summary(s, d, 1.0)):
            out = rt.run_sweep(dict(_SWEEP))
        assert str(out["current_value"]) in out["results"]


class TestOutput:
    def test_main_writes_tuning_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rt, "TUNING_PATH", tmp_path / "tuning.json")
        with patch.object(rt, "_run_backtest",
                          side_effect=lambda s, d: _summary(s, d, 1.5)):
            with patch.object(rt.sys, "argv", ["run_tuning.py"]):
                assert rt.main() == 0
        data = json.loads((tmp_path / "tuning.json").read_text())
        assert data["sweeps"], "expected at least one sweep result"
        for s in data["sweeps"]:
            assert "recommendation" in s or "error" in s

    def test_registry_entries_use_call_time_imports(self):
        """Every registered module constant must actually exist — a typo
        here would silently sweep nothing."""
        import importlib
        for sweep in rt.SWEEPS:
            mod = importlib.import_module(sweep["module"])
            assert hasattr(mod, sweep["param"]), (
                f"{sweep['module']}.{sweep['param']} does not exist")
