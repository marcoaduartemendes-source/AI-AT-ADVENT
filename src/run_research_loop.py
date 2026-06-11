"""Research-loop agent — autonomous nightly review of the bot.

Reads the bot's own evidence files (self_grade, validation, walk_forward,
cycle_status, recent trades, watchdog) and asks Claude to produce a
RANKED QUEUE of concrete, actionable improvements. The agent NEVER
writes code and NEVER executes trades — every output is a proposal
in `docs/research_proposals.json` for human (or human-supervised
Claude Code) review.

Cadence (managed by deploy/systemd/research-loop.timer):
  • Daily at 07:00 UTC (after the 06:30 research backtests refresh
    validation.json) — Sonnet, ≤6K input + 2K output ≈ $0.05.

Safety rails:
  1. NEVER emits code patches. Proposals are plain English; concrete
     enough to act on, not so concrete they bypass human judgement.
  2. NEVER touches LIVE_STRATEGIES, allocator state, or any database.
  3. Hard token budget per run; missing ANTHROPIC_API_KEY → emit nothing
     instead of failing (timer just produces a no-op).
  4. All output goes to docs/research_proposals.json (queue) and
     docs/research_proposals.md (human digest). Both are rotated so
     the operator sees what changed week-over-week.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("research_loop")

DAILY_MODEL = "claude-sonnet-4-6"
DAILY_MAX_TOKENS = 2400
DAILY_INPUT_CAP_CHARS = 24_000   # ≈ 6K input tokens

QUEUE_PATH = Path("docs/research_proposals.json")
DIGEST_PATH = Path("docs/research_proposals.md")
HISTORY_PATH = Path("data/research_history.jsonl")


# ── Evidence assembly ────────────────────────────────────────────────


def _load_json(path: str) -> dict | list | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _recent_trades(db_path: str, limit: int = 40) -> list[dict]:
    if not Path(db_path).exists():
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, strategy, side, product_id, amount_usd, "
                "pnl_usd, fill_status FROM trades "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"recent_trades read failed: {e}")
        return []


def _equity_summary(risk_db: str) -> dict:
    if not Path(risk_db).exists():
        return {}
    try:
        with sqlite3.connect(risk_db) as conn:
            conn.row_factory = sqlite3.Row
            first = conn.execute(
                "SELECT timestamp, equity_usd FROM equity_snapshots "
                "ORDER BY id ASC LIMIT 1"
            ).fetchone()
            last = conn.execute(
                "SELECT timestamp, equity_usd FROM equity_snapshots "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            peak = conn.execute(
                "SELECT MAX(equity_usd) FROM equity_snapshots"
            ).fetchone()
        if not first or not last:
            return {}
        return {
            "inception_ts": first["timestamp"],
            "inception_equity": float(first["equity_usd"]),
            "current_ts": last["timestamp"],
            "current_equity": float(last["equity_usd"]),
            "peak_equity": float(peak[0]) if peak and peak[0] else None,
        }
    except Exception as e:
        logger.debug(f"equity summary read failed: {e}")
        return {}


def _read_prior_proposals(n: int = 10) -> list[dict]:
    """Last N proposal batches — lets the agent avoid repeating itself
    and notice which prior ideas were implemented."""
    if not HISTORY_PATH.exists():
        return []
    out: list[dict] = []
    try:
        with HISTORY_PATH.open() as f:
            for line in f.readlines()[-n:]:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return out


def _assemble_evidence() -> dict:
    """Bundle every signal the agent will reason over. Trimmed to fit
    within DAILY_INPUT_CAP_CHARS — older trades are dropped first."""
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "self_grade": _load_json("docs/self_grade.json"),
        "validation": _trim_validation(_load_json("docs/validation.json")),
        "walk_forward": _load_json("docs/walk_forward.json"),
        "benchmark": _load_json("docs/benchmark.json"),
        "watchdog": _load_json("docs/watchdog.json"),
        "cycle_status_tail": _tail_cycles(
            _load_json("docs/cycle_status.json"), n=10),
        "recent_trades": _recent_trades(
            os.environ.get("TRADING_DB_PATH",
                           "data/trading_performance.db"),
            limit=40),
        "equity_summary": _equity_summary(
            os.environ.get("RISK_DB_PATH", "data/risk_state.db")),
        "prior_proposals_tail": _read_prior_proposals(n=5),
    }


def _trim_validation(data):
    """validation.json includes a per-strategy detail block; keep verdict
    + sharpe + walk-forward only to stay within token budget."""
    if not isinstance(data, dict):
        return data
    strategies = data.get("strategies") or {}
    trimmed = {
        nm: {k: v for k, v in (s or {}).items()
             if k in ("verdict", "sharpe_5y", "n_trades_5y",
                      "return_on_volume_pct", "reason")}
        for nm, s in strategies.items()
    }
    return {**{k: v for k, v in data.items() if k != "strategies"},
            "strategies": trimmed}


def _tail_cycles(data, n=10):
    if isinstance(data, list):
        return data[-n:]
    return data


# ── Claude call ──────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are the autonomous research analyst for a live \
multi-asset systematic trading bot. You run nightly. Your job is to \
turn the bot's own telemetry into a SHORT ranked queue of concrete, \
actionable proposals a human operator can review the next morning.

HARD CONSTRAINTS:
  • You NEVER write code. You write proposals.
  • You NEVER touch live money, allocations, or risk state directly.
  • You output ONLY a JSON object matching the schema below.
  • You produce at most 5 proposals per run. Quality over volume.
  • Every proposal must cite the SPECIFIC evidence row that motivated \
    it (a validation verdict, a self_grade axis score, a cycle error \
    pattern, an equity move). No hand-waving.
  • If the evidence shows no clear action, say so honestly — emit \
    fewer proposals or zero. Padding is forbidden.
  • Categories: STRATEGY (add/remove/modify a sleeve), ALLOCATION \
    (target_alloc_pct change), INFRA (deploy/monitoring/data feed), \
    RISK (cap/stop/threshold), OPS (operator action like fund-wallet).
  • Priorities: HIGH (clear evidence of harm or missed alpha), \
    MEDIUM (likely improvement), LOW (worth tracking).

OUTPUT SCHEMA (return EXACTLY this JSON object, no other text):
{
  "summary": "<one-paragraph state of the book — what is working, \
what is failing, the single most important thing to fix>",
  "weakest_grade_axis": {"axis": "<name>", "score": <0-10>, \
    "why": "<one sentence root cause>"},
  "proposals": [
    {
      "category": "STRATEGY|ALLOCATION|INFRA|RISK|OPS",
      "priority": "HIGH|MEDIUM|LOW",
      "title": "<≤60 chars, imperative voice: 'Retire X', 'Bump Y from A to B'>",
      "rationale": "<2-4 sentences. Cite specific numbers from evidence.>",
      "evidence_keys": ["<top-level key from evidence used>", "..."],
      "proposed_action": "<one paragraph the operator could turn into a commit>",
      "expected_impact": "<which self-grade axis or P&L line improves and \
roughly by how much>",
      "risk": "<what could go wrong>"
    }
  ],
  "new_strategy_ideas": [
    {
      "name": "<snake_case>",
      "thesis": "<1-2 sentences>",
      "documented_edge": "<paper / firm / academic citation if any>",
      "data_required": "<what feed is needed>",
      "implementability": "SIMPLE|MEDIUM|HARD"
    }
  ]
}

Be skeptical. The bot has historically suffered from: closet-index \
roster, false kill latches, stale validation, ETF cap starvation, \
unfunded venue wallets. Look for the next problem of that family."""


def _truncate_evidence(ev: dict, max_chars: int) -> str:
    """Serialize evidence with a hard char budget; drop oldest cycles
    and trades first if over."""
    raw = json.dumps(ev, default=str, indent=1)
    if len(raw) <= max_chars:
        return raw
    # Aggressive trim: halve recent_trades and cycle_status_tail.
    ev2 = dict(ev)
    if isinstance(ev2.get("recent_trades"), list):
        ev2["recent_trades"] = ev2["recent_trades"][:20]
    if isinstance(ev2.get("cycle_status_tail"), list):
        ev2["cycle_status_tail"] = ev2["cycle_status_tail"][-5:]
    raw = json.dumps(ev2, default=str, indent=1)
    return raw[:max_chars]


def _call_claude(evidence: dict) -> tuple[dict | None, float]:
    """Returns (parsed_dict, estimated_usd_cost). Returns (None, 0) on
    any failure — the timer just produces a no-op."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — skipping research loop")
        return None, 0.0
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package missing — skipping research loop")
        return None, 0.0

    payload = _truncate_evidence(evidence, DAILY_INPUT_CAP_CHARS)
    user_msg = (
        "Below is the bot's current telemetry as JSON. Produce the queue.\n\n"
        f"```json\n{payload}\n```"
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=DAILY_MODEL,
            max_tokens=DAILY_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        logger.warning(f"Claude call failed: {e}")
        return None, 0.0

    text = "".join(
        b.text for b in msg.content
        if getattr(b, "type", "") == "text"
    )
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        logger.warning("Claude response missing JSON object")
        return None, 0.0
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"Claude response JSON parse failed: {e}")
        return None, 0.0

    # Sonnet 4.5 (approx 2026 pricing): ~$3/Mtok in, ~$15/Mtok out.
    cost = (
        msg.usage.input_tokens * 3.0 / 1_000_000
        + msg.usage.output_tokens * 15.0 / 1_000_000
    )
    return parsed, cost


# ── Output writers ───────────────────────────────────────────────────


def _write_queue(parsed: dict, cost: float) -> None:
    """Newest-on-top JSON queue the dashboard renders."""
    enriched = {
        "as_of": datetime.now(UTC).isoformat(),
        "model": DAILY_MODEL,
        "cost_usd": round(cost, 4),
        **parsed,
    }
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(
        json.dumps(enriched, indent=2), encoding="utf-8")


def _write_digest(parsed: dict, cost: float) -> None:
    """Human-friendly markdown the operator can open on a phone."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Research-loop digest — {now_str}", ""]
    if (s := parsed.get("summary")):
        lines += [f"**State of the book.** {s}", ""]
    if (a := parsed.get("weakest_grade_axis") or {}):
        lines += [
            f"**Weakest grade axis:** `{a.get('axis')}` "
            f"({a.get('score')}/10) — {a.get('why')}",
            ""
        ]
    props = parsed.get("proposals") or []
    if props:
        lines.append("## Proposals (ranked)")
        for i, p in enumerate(props, 1):
            lines += [
                f"### {i}. [{p.get('priority', '?')}] "
                f"[{p.get('category', '?')}] {p.get('title', '?')}",
                f"**Rationale.** {p.get('rationale', '')}",
                f"**Action.** {p.get('proposed_action', '')}",
                f"**Expected impact.** {p.get('expected_impact', '')}",
                f"**Risk.** {p.get('risk', '')}",
                "",
            ]
    if (ideas := parsed.get("new_strategy_ideas") or []):
        lines.append("## New strategy ideas")
        for idea in ideas:
            lines += [
                f"- **{idea.get('name', '?')}** "
                f"({idea.get('implementability', '?')}): "
                f"{idea.get('thesis', '')}  ",
                f"  Edge: {idea.get('documented_edge', '—')}; "
                f"data: {idea.get('data_required', '—')}",
            ]
        lines.append("")
    lines.append(
        f"_LLM cost this run: ${round(cost, 4)} "
        f"(model: {DAILY_MODEL})_")
    DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIGEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def _append_history(parsed: dict, cost: float) -> None:
    """JSONL append-only ledger. Lets future runs see what was proposed
    week-over-week and notice repeats / convergence."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "cost_usd": round(cost, 4),
        "n_proposals": len(parsed.get("proposals") or []),
        "n_ideas": len(parsed.get("new_strategy_ideas") or []),
        "summary": parsed.get("summary", ""),
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _alert_if_high_priority(parsed: dict) -> None:
    """Any HIGH-priority proposal pages the operator (deduped daily)."""
    high = [p for p in (parsed.get("proposals") or [])
            if (p.get("priority") or "").upper() == "HIGH"]
    if not high:
        return
    try:
        from common.alerts import alert
        titles = "; ".join(p.get("title", "?") for p in high[:3])
        alert(
            f"🧠 Research loop: {len(high)} HIGH-priority proposal(s) — "
            f"{titles}",
            severity="warning",
        )
    except Exception as e:  # noqa: BLE001 — alert failure must not break run
        logger.debug(f"alert dispatch failed: {e}")


# ── Entry point ──────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Nightly research-loop agent (proposes; never writes code)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Assemble evidence but don't call Claude or write")
    args = ap.parse_args()

    evidence = _assemble_evidence()
    if args.dry_run:
        print(json.dumps(evidence, default=str, indent=2)[:4000])
        return 0

    parsed, cost = _call_claude(evidence)
    if parsed is None:
        logger.info("research_loop: no output this cycle")
        return 0

    _write_queue(parsed, cost)
    _write_digest(parsed, cost)
    _append_history(parsed, cost)
    _alert_if_high_priority(parsed)
    n = len(parsed.get("proposals") or [])
    logger.info(
        f"research_loop: wrote {n} proposals, cost ${round(cost, 4)} "
        f"(queue: {QUEUE_PATH}, digest: {DIGEST_PATH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
