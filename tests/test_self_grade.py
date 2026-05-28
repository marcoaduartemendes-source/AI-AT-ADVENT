"""Tests for self_grade axis computations that had subtle bugs."""
from __future__ import annotations

from common.self_grade import _grade_execution


def _cyc(proposed, submitted, dry):
    return {"timestamp": "2026-05-22T20:00:00+00:00",
            "proposals_total": proposed, "proposals_submitted": submitted,
            "proposals_dry": dry}


class TestExecutionExcludesDry:
    """execution_quality must measure submitted / live-submittable, NOT
    submitted / all-proposed. DRY-logged proposals are intentional
    non-submissions; counting them dragged the score to ~0 in a
    mostly-DRY book even when the live strategy submitted everything."""

    def test_dry_proposals_excluded_from_denominator(self):
        # 5 live proposals all submitted, 45 DRY-logged → 100%, not 10%.
        g, reason = _grade_execution([_cyc(50, 5, 45)])
        assert g == 10.0
        assert "excl. 45 DRY" in reason

    def test_all_dry_is_neutral_not_zero(self):
        g, reason = _grade_execution([_cyc(40, 0, 40)])
        assert g == 5.0
        assert "no live-submittable" in reason

    def test_real_misses_still_score_low(self):
        # 10 live-submittable, only 2 placed (8 genuine failures), 0 DRY.
        g, _ = _grade_execution([_cyc(10, 2, 0)])
        assert g == 2.0

    def test_missing_dry_field_falls_back_gracefully(self):
        # Older cycles without proposals_dry → treated as 0 DRY.
        g, _ = _grade_execution([{"timestamp": "t", "proposals_total": 10,
                                  "proposals_submitted": 8}])
        assert g == 8.0


class TestEconomicLeverage:
    """Gross-leverage stat must count leveraged-ETF positions at their
    ECONOMIC exposure (TQQQ ×3), not face notional — otherwise the 3x
    sleeves read as far less levered than they really are."""

    def test_3x_etf_counts_triple(self):
        import build_dashboard as bd
        assert bd._LEVERAGED_ETF_FACTOR["TQQQ"] == 3.0
        assert bd._LEVERAGED_ETF_FACTOR.get("SPY", 1.0) == 1.0
