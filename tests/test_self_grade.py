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

    def test_all_dry_is_zero_not_neutral(self):
        # 2026-06-05: an all-DRY book IS a failure (zero throughput is
        # not "neutral"). The previous 5/10 fallback was masking the
        # real silence as a passing grade — see the cap-starvation
        # diagnosis in AUDIT.md. The metric now correctly scores it 0.
        g, reason = _grade_execution([_cyc(40, 0, 40)])
        assert g == 0.0
        assert "silent" in reason.lower() or "zero" in reason.lower()

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


class TestAlphaTrackAuditFix:
    """2026-06-11 audit: alpha_track scored 10.0/10 on a book that had
    earned $0.50, because canceled/pending orders were counted as 0.0
    P&L observations. Replays the exact May-2026 ledger shape from
    docs/AUDIT_INCEPTION.md §4 and pins that it can never score high
    again."""

    def _historical_ledger(self):
        """The real shape: 50 rows = 38 CANCELED + 2 PENDING + 10 FILLED,
        of which only 5 carry P&L, totalling $0.50."""
        from datetime import UTC, datetime, timedelta
        now = datetime.now(UTC)
        rows = []
        for _i in range(38):
            rows.append({"timestamp": (now - timedelta(days=1)).isoformat(),
                         "fill_status": "CANCELED", "pnl_usd": None})
        for _i in range(2):
            rows.append({"timestamp": (now - timedelta(days=1)).isoformat(),
                         "fill_status": "PENDING", "pnl_usd": None})
        # 10 filled; only 5 have realized P&L, summing to $0.50
        for _i in range(5):
            rows.append({"timestamp": (now - timedelta(days=2)).isoformat(),
                         "fill_status": "FILLED", "pnl_usd": 0.10})
        for _i in range(5):
            rows.append({"timestamp": (now - timedelta(days=2)).isoformat(),
                         "fill_status": "FILLED", "pnl_usd": None})
        return rows

    def test_historical_fifty_cents_cannot_score_high(self):
        from common.self_grade import _grade_alpha
        g, reason = _grade_alpha(self._historical_ledger())
        assert g == 0.0, (
            f"a $0.50 book with 5 closed trips must not score {g}: {reason}")
        assert "closed round trips" in reason

    def test_canceled_orders_are_not_zero_returns(self):
        """The core defect: an unfilled order is the ABSENCE of a return,
        not a zero one. 100 canceled rows must not manufacture a sample."""
        from datetime import UTC, datetime, timedelta
        from common.self_grade import _grade_alpha
        now = datetime.now(UTC)
        rows = [{"timestamp": (now - timedelta(days=1)).isoformat(),
                 "fill_status": "CANCELED", "pnl_usd": None}
                for _ in range(100)]
        g, reason = _grade_alpha(rows)
        assert g == 0.0
        assert "0 closed round trips" in reason

    def test_real_track_record_still_scores(self):
        """A genuine record — 30 closed round trips, consistently
        profitable — must still grade well, or the fix is too blunt."""
        from datetime import UTC, datetime, timedelta
        from common.self_grade import _grade_alpha
        now = datetime.now(UTC)
        rows = [{"timestamp": (now - timedelta(days=3)).isoformat(),
                 "fill_status": "FILLED",
                 "pnl_usd": 100.0 + (i % 5) * 10}
                for i in range(30)]
        g, reason = _grade_alpha(rows)
        assert g >= 7.0, f"genuine record should grade well, got {g}: {reason}"
