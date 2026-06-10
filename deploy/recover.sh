#!/usr/bin/env bash
# ONE-COMMAND RECOVERY — deploy latest code, refresh research verdicts,
# clear stale strategy freezes, and print the book's vital signs.
#
#   ssh root@<IP> 'bash /opt/ai-at-advent/deploy/recover.sh'
#
# Created 2026-06-10: the operator was being handed 3-4 step command
# sequences (update.sh → run_research --force → --unfreeze all →
# status) and steps kept being skipped, so fixes sat undeployed and
# state cleanup never happened. One script, idempotent, safe to run
# any time.

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/ai-at-advent}"
SERVICE_USER="${SERVICE_USER:-aaa}"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root" >&2
    exit 1
fi

echo "═══ [1/4] Deploying latest code ═══"
bash "$INSTALL_DIR/deploy/update.sh"

_run_as_aaa() {
    sudo -u "$SERVICE_USER" bash -c \
        "cd $INSTALL_DIR && set -a; . /etc/aaa.env; set +a; $1"
}

echo
echo "═══ [2/4] Research verdicts (only if stale >24h) ═══"
# Freshness check inline — research takes ~4 min, skip when fresh.
_needs_research=$(_run_as_aaa '.venv/bin/python - <<PY
import json, sys
from datetime import UTC, datetime
try:
    d = json.load(open("docs/validation.json"))
    age_h = (datetime.now(UTC) - datetime.fromisoformat(
        d["as_of"].replace("Z", "+00:00"))).total_seconds() / 3600
    print("YES" if age_h > 24 else "NO")
except Exception:
    print("YES")
PY')
if [[ "$_needs_research" == "YES" ]]; then
    echo "  validation stale — running full research (≈4 min)…"
    _run_as_aaa '.venv/bin/python src/run_research.py --force' || \
        echo "  ⚠ research run FAILED — check: journalctl -u research.service"
else
    echo "  validation fresh (<24h) — skipping"
fi

echo
echo "═══ [3/4] Clearing stale strategy freezes ═══"
# Safe: anything that genuinely deserves freezing gets re-frozen by the
# allocator on the next cycle using the FRESH verdicts from step 2.
_run_as_aaa '.venv/bin/python src/run_orchestrator.py --unfreeze all' || true

echo
echo "═══ [4/4] Book vitals ═══"
_run_as_aaa '.venv/bin/python src/run_orchestrator.py --status' \
    | head -30 || true

echo
echo "Done. Dashboard refreshes within ~1 min: check kill switch state,"
echo "venue cash (Coinbase available \$), and per-strategy Status column."
