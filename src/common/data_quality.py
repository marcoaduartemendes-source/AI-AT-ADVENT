"""Data-quality agent — continuous audit of dashboard/JSON integrity.

USER MANDATE (2026-05-20)
"Build an agent that constantly checks for data accuracy and
quality of the dashboard."

The dashboard is only as good as the JSONs it reads. This agent
audits every artifact the orchestrator + dashboard produce, on
every cycle, and writes docs/data_quality.json. Issues:
  • STALE       — file older than its expected refresh cadence
  • MISSING     — file the dashboard expects doesn't exist
  • INVALID     — file exists but is unparseable / malformed
  • INCONSISTENT— file refers to entities that don't exist elsewhere
  • EMPTY       — file is parseable but has no actionable rows

Each issue surfaces in the self-grade's data_quality axis and the
dashboard's data-quality panel. The agent is intentionally LOUD
about its findings — every issue is a failure mode the user
would otherwise discover by squinting at the dashboard.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# Expected refresh cadence per file. If older than this, STALE.
EXPECTED_AGE_HOURS = {
    "docs/cycle_status.json":   1,    # every 5 min normally
    "docs/trades_recent.json":  1,    # rewritten every cycle
    "docs/benchmark.json":      6,    # several times a day
    "docs/validation.json":     30,   # daily backtest harness
    "docs/walk_forward.json":   30,   # daily
    "docs/improvements.json":   1,    # every cycle (cheap)
    "docs/self_grade.json":     1,    # every cycle
    "docs/hedge_fund_13f.json": 30,   # daily scout
    "docs/index.html":          1,    # dashboard cron 15 min
}


def _age_hours(p: Path) -> float | None:
    if not p.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
        return (datetime.now(UTC) - mtime).total_seconds() / 3600.0
    except Exception:
        return None


def _audit_file(rel: str, max_age_h: float) -> dict:
    p = Path(rel)
    if not p.exists():
        return {"file": rel, "status": "MISSING",
                "reason": "file does not exist"}
    age = _age_hours(p)
    if age is None:
        return {"file": rel, "status": "INVALID",
                "reason": "could not read mtime"}
    if rel.endswith(".json"):
        try:
            parsed = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return {"file": rel, "status": "INVALID",
                    "reason": f"JSON parse error: {e}",
                    "age_hours": round(age, 1)}
        # Emptiness check (list with 0 rows or dict with no top-level keys)
        if isinstance(parsed, list) and len(parsed) == 0:
            return {"file": rel, "status": "EMPTY",
                    "reason": "list with 0 entries",
                    "age_hours": round(age, 1)}
        if isinstance(parsed, dict) and not parsed:
            return {"file": rel, "status": "EMPTY",
                    "reason": "empty dict",
                    "age_hours": round(age, 1)}
    if age > max_age_h:
        return {"file": rel, "status": "STALE",
                "reason": (f"{age:.1f}h old (expected refresh every "
                           f"{max_age_h}h)"),
                "age_hours": round(age, 1)}
    return {"file": rel, "status": "OK",
            "age_hours": round(age, 1)}


def _cross_check() -> list[dict]:
    """Internal-consistency checks across artifacts.

    Catches the failure mode where one JSON references a strategy
    name that doesn't exist in another (eg. trades_recent has
    'foo_strategy' but validation has no entry for it = something
    is mis-registered)."""
    issues: list[dict] = []
    try:
        trades = json.loads(
            Path("docs/trades_recent.json").read_text(encoding="utf-8"))
        validation = json.loads(
            Path("docs/validation.json").read_text(encoding="utf-8"))
    except Exception:
        return issues

    val_strats = set((validation.get("strategies") or {}).keys())
    seen_in_trades = {t.get("strategy") for t in (trades or [])
                       if t.get("strategy")}
    orphans = seen_in_trades - val_strats
    if orphans:
        issues.append({"check": "trade-orphans",
                        "status": "INCONSISTENT",
                        "reason": (f"trades exist for strategies not in "
                                   f"validation: {sorted(orphans)[:5]}")})

    # Cancel-rate sanity — if >50% of recent trades are CANCELED that's
    # the loop bug from 2026-05-20 (RTH gate). Flag it explicitly.
    if trades:
        recent = trades[:50]
        cancels = sum(1 for t in recent
                       if t.get("fill_status") == "CANCELED")
        if recent and cancels / len(recent) > 0.5:
            issues.append({
                "check": "cancel-rate",
                "status": "INCONSISTENT",
                "reason": (f"{cancels}/{len(recent)} recent trades "
                           f"CANCELED ({cancels/len(recent):.0%}) — "
                           f"likely out-of-RTH submission loop"),
            })
    return issues


def run_data_quality(out_path: str = "docs/data_quality.json") -> dict:
    """Audit every expected file + cross-check consistency.
    Returns the payload + writes to docs/data_quality.json."""
    rows: list[dict] = []
    for rel, max_age in EXPECTED_AGE_HOURS.items():
        rows.append(_audit_file(rel, max_age))
    rows.extend(_cross_check())

    counts: dict[str, int] = {}
    for r in rows:
        s = r.get("status", "?")
        counts[s] = counts.get(s, 0) + 1

    # Score: 10 = all OK; subtract per failure category.
    n_total = len(rows)
    n_ok = counts.get("OK", 0)
    score = round(10.0 * (n_ok / n_total) if n_total else 0.0, 1)

    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "score": score,
        "counts": counts,
        "n_checks": n_total,
        "rows": rows,
    }
    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info(f"data_quality: score {score}/10  counts={counts}")
    except Exception as e:
        logger.warning(f"data_quality: write failed: {e}")
    return payload
