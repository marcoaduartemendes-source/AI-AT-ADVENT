"""Strategic reviewer — calls Claude Opus 4.7 with all relevant context,
parses recommendations, persists them.

Why direct Anthropic SDK and not the Agent tool: this runs on a GitHub
Actions schedule (weekly), not in a Claude Code session. We invoke the
API directly from CI.

Prompt caching is used aggressively because the system prompt
(framing + instructions + schema) is constant week-to-week; only the
data changes. That keeps the cost per review small even on Opus.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

logger = logging.getLogger(__name__)


# Constant system prompt — cached. Keep stable so cache hits across runs.
SYSTEM_PROMPT = """You are a senior portfolio manager reviewing a multi-asset
systematic trading book that runs across Coinbase (crypto + commodity futures),
Alpaca (US equities + ETFs), and Kalshi (prediction markets).

Your job each week is to produce a concise, actionable recommendation document
in strict JSON format. You DO NOT execute trades. You produce a structured
recommendation that the user reviews and applies manually.

The book targets ~15% net annualized at ~12% portfolio vol with hard kill
switches at 5/10/15% drawdown. The 10 strategies span carry (funding-rate /
basis / commodity term-structure), momentum (TSMOM ETF / cross-sectional
crypto), mean-reversion (Kalshi calibration / mean-reversion), event-driven
(PEAD / macro Kalshi), and a vol-managed overlay.

Hard rules:
  1. Never recommend leverage > 2x book equity.
  2. Never recommend allocation to a single strategy > 30%.
  3. Recommend FREEZE for any strategy with rolling 30-day Sharpe < -1.0
     OR drawdown > 20%.
  4. Recommend WATCH for any strategy with rolling 30-day Sharpe < 0
     OR drawdown > 8%.
  5. Recommend UNFREEZE only if the strategy has been frozen ≥ 30 days
     AND demonstrated paper-mode improvement.
  6. Be specific. "Reduce crypto exposure" is useless. "Reduce
     crypto_funding_carry from 15% to 8% allocation due to negative
     funding regime" is useful.

You MUST respond with VALID JSON ONLY (no prose, no markdown fences) matching
exactly this schema:

{
  "summary": "<2-3 sentence executive summary of book health>",
  "overall_health": "GREEN" | "YELLOW" | "RED",
  "risk_multiplier_recommendation": <float between 0.5 and 2.0>,
  "risk_multiplier_reason": "<one sentence>",
  "strategy_actions": [
    {
      "strategy": "<exact strategy name>",
      "action": "ACTIVATE" | "WATCH" | "FREEZE" | "RETIRE" | "INCREASE" | "DECREASE" | "MAINTAIN",
      "target_alloc_pct": <float, optional, only for INCREASE/DECREASE>,
      "reason": "<one or two sentences citing specific metrics>",
      "confidence": <float 0.0 to 1.0>
    }
  ],
  "investigate": [
    "<specific data anomaly or hypothesis worth a deeper look>"
  ],
  "next_review_horizon_days": <int>
}

If a strategy is fine, include it as MAINTAIN. Cover all strategies in the
book even if their action is MAINTAIN.
"""


@dataclass
class ReviewResult:
    timestamp: datetime
    summary: str
    overall_health: str        # GREEN / YELLOW / RED
    risk_multiplier_rec: float
    risk_multiplier_reason: str
    strategy_actions: list[dict[str, Any]] = field(default_factory=list)
    investigate: list[str] = field(default_factory=list)
    next_review_horizon_days: int = 7
    raw_payload: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    model_used: str = ""


# ─── Persistence ──────────────────────────────────────────────────────────


class ReviewDB:
    def __init__(self, db_path: str | None = None):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("REVIEW_DB_PATH", "data/strategic_review.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    overall_health  TEXT NOT NULL,
                    summary         TEXT NOT NULL,
                    risk_mult_rec   REAL,
                    risk_mult_reason TEXT,
                    payload_json    TEXT NOT NULL,
                    model_used      TEXT,
                    cost_usd        REAL
                )
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, r: ReviewResult) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO reviews (timestamp, overall_health, summary, "
                "risk_mult_rec, risk_mult_reason, payload_json, model_used, cost_usd) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (r.timestamp.isoformat(), r.overall_health, r.summary,
                 r.risk_multiplier_rec, r.risk_multiplier_reason,
                 json.dumps(r.raw_payload), r.model_used, r.cost_usd),
            )
            return cur.lastrowid

    def latest(self) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM reviews ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def history(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, timestamp, overall_health, summary, risk_mult_rec "
                "FROM reviews ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


# ─── Reviewer ────────────────────────────────────────────────────────────


class StrategicReviewer:
    """Pulls all the data, calls Opus, persists the structured result."""

    def __init__(self, model: str = "claude-opus-4-7", db: ReviewDB | None = None):
        self.model = model
        self.db = db or ReviewDB()

    def gather_context(self) -> dict:
        """Collect everything the reviewer needs in one dict."""
        ctx: dict = {"as_of": datetime.now(UTC).isoformat()}

        # Risk state + recent equity
        try:
            from risk.manager import EquitySnapshotDB
            risk_db = EquitySnapshotDB()
            with risk_db._conn() as c:
                rows = c.execute(
                    "SELECT timestamp, equity_usd FROM equity_snapshots "
                    "ORDER BY id DESC LIMIT 200"
                ).fetchall()
                ks_rows = c.execute(
                    "SELECT timestamp, state, drawdown_pct, note "
                    "FROM kill_switch_events ORDER BY id DESC LIMIT 20"
                ).fetchall()
            ctx["equity_history"] = [
                {"t": r["timestamp"], "eq": float(r["equity_usd"])} for r in rows
            ]
            ctx["kill_switch_events"] = [dict(r) for r in ks_rows]
        except Exception as e:
            logger.warning(f"Could not load risk context: {e}")
            ctx["equity_history"] = []
            ctx["kill_switch_events"] = []

        # Allocator history + lifecycle events
        try:
            from allocator.lifecycle import StrategyRegistry
            reg = StrategyRegistry()
            ctx["latest_allocations"] = list(reg.latest_allocations().values())
            ctx["lifecycle_events"] = reg.lifecycle_events(limit=30)
        except Exception as e:
            logger.warning(f"Could not load allocator context: {e}")
            ctx["latest_allocations"] = []
            ctx["lifecycle_events"] = []

        # Per-strategy 30d + 7d metrics
        try:
            from allocator.metrics import StrategyPerformance
            perf = StrategyPerformance()
            strategies = [a["name"] for a in ctx["latest_allocations"]] or [
                "crypto_funding_carry", "risk_parity_etf",
                "kalshi_calibration_arb", "crypto_basis_trade",
                "tsmom_etf", "commodity_carry", "pead", "macro_kalshi",
                "crypto_xsmom", "vol_managed_overlay",
            ]
            ctx["metrics_30d"] = {}
            ctx["metrics_7d"] = {}
            for s in strategies:
                m30 = perf.metrics_for(s, window_days=30)
                m7 = perf.metrics_for(s, window_days=7)
                ctx["metrics_30d"][s] = {
                    "n_trades": m30.n_trades,
                    "win_rate": round(m30.win_rate, 3),
                    "total_pnl_usd": round(m30.total_pnl_usd, 2),
                    "shrunk_sharpe": round(m30.shrunk_sharpe, 3),
                    "drawdown_pct": round(m30.drawdown_pct, 4),
                }
                ctx["metrics_7d"][s] = {
                    "n_trades": m7.n_trades,
                    "win_rate": round(m7.win_rate, 3),
                    "total_pnl_usd": round(m7.total_pnl_usd, 2),
                    "shrunk_sharpe": round(m7.shrunk_sharpe, 3),
                }
        except Exception as e:
            logger.warning(f"Could not load metrics context: {e}")
            ctx["metrics_30d"] = {}
            ctx["metrics_7d"] = {}

        # Scout signal volume
        try:
            from scouts.signal_bus import SignalBus
            bus = SignalBus()
            recent = bus.latest(limit=500)
            counts: dict[str, int] = {}
            for r in recent:
                key = f"{r.venue}:{r.signal_type}"
                counts[key] = counts.get(key, 0) + 1
            ctx["scout_signal_counts_recent"] = counts
        except Exception as e:
            logger.warning(f"Could not load scout context: {e}")
            ctx["scout_signal_counts_recent"] = {}

        return ctx

    def review(self) -> ReviewResult:
        ctx = self.gather_context()
        return self._call_llm(ctx)

    def _call_llm(self, ctx: dict) -> ReviewResult:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed; add to requirements.txt"
            ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — strategic review needs an "
                "Anthropic API key (https://console.anthropic.com/) stored "
                "as a GitHub repo secret."
            )

        client = anthropic.Anthropic(api_key=api_key)
        # Build user prompt from the gathered context
        user_prompt = (
            "Here is this week's data. Produce the JSON recommendation per "
            "the schema in the system prompt. Respond with JSON only.\n\n"
            f"```json\n{json.dumps(ctx, default=str, indent=2)}\n```"
        )

        # Use prompt caching on the system prompt (it's stable)
        message = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(b.text for b in message.content if hasattr(b, "text"))
        # Strip code fences if Opus added them despite instructions
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json\n"):
                text = text[5:]
            text = text.rsplit("```", 1)[0].strip()

        try:
            payload = json.loads(text)
        except Exception as e:
            logger.error(f"Could not parse Opus response as JSON: {e}\nText was:\n{text[:500]}")
            raise

        # Approximate cost: Opus 4.x at ~$15/M input + $75/M output, with
        # cache discount on the system prompt
        usage = getattr(message, "usage", None)
        cost = None
        if usage:
            in_tok = getattr(usage, "input_tokens", 0)
            out_tok = getattr(usage, "output_tokens", 0)
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            # Rough numbers — Anthropic price page is the source of truth
            cost = (
                (in_tok - cache_read - cache_write) * 15 / 1_000_000
                + cache_write * 18.75 / 1_000_000        # 1.25x write rate
                + cache_read * 1.5 / 1_000_000           # 0.1x read rate
                + out_tok * 75 / 1_000_000
            )

        result = ReviewResult(
            timestamp=datetime.now(UTC),
            summary=payload.get("summary", ""),
            overall_health=payload.get("overall_health", "YELLOW"),
            risk_multiplier_rec=float(payload.get("risk_multiplier_recommendation", 1.0)),
            risk_multiplier_reason=payload.get("risk_multiplier_reason", ""),
            strategy_actions=payload.get("strategy_actions", []),
            investigate=payload.get("investigate", []),
            next_review_horizon_days=int(payload.get("next_review_horizon_days", 7)),
            raw_payload=payload,
            cost_usd=cost,
            model_used=self.model,
        )
        self.db.save(result)
        return result
