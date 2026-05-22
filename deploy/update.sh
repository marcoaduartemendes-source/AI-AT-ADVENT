#!/usr/bin/env bash
# Pull the latest commit and refresh dependencies.
#
# Run on the VPS as root, e.g. via SSH:
#     ssh root@<IP> 'bash /opt/ai-at-advent/deploy/update.sh'
#
# Or wire it to a cron / webhook listener for push-on-merge deploys.

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/ai-at-advent}"
SERVICE_USER="${SERVICE_USER:-aaa}"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root" >&2
    exit 1
fi

cd "$INSTALL_DIR"

# Ensure runtime dirs exist (systemd ProtectSystem=strict makes
# everything outside ReadWritePaths read-only, so /opt/.../data
# must exist before the service starts or it fails with
# "Failed to set up mount namespacing").
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" \
    "$INSTALL_DIR/data" "$INSTALL_DIR/docs"

# Sprint D — Quiesce: stop the orchestrator timer before pulling so
# we don't deploy mid-cycle (a HH:04 deploy used to race the HH:05
# fire and run mixed code). Re-enable at the end. The timer is the
# only one we quiesce — scouts and dashboard are stateless reads
# and tolerate a half-deployed state.
echo "[1/6] Quiescing orchestrator timer"
systemctl stop orchestrator.timer 2>/dev/null || true

echo "[2/6] Fetching latest"
sudo -u "$SERVICE_USER" git fetch --all --quiet

CURRENT="$(sudo -u "$SERVICE_USER" git rev-parse HEAD)"
# Pin the deploy branch explicitly (overridable via DEPLOY_BRANCH env)
# instead of trusting whatever happens to be checked out on the box.
# A debug checkout left over on the VPS would otherwise become an
# accidental deploy of an unrelated branch's tip.
BRANCH="${DEPLOY_BRANCH:-claude/daily-ai-news-digest-vcaC1}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "ERROR: HEAD is on '$CURRENT_BRANCH' but deploy expects '$BRANCH'." >&2
    echo "       Switch back with: git checkout $BRANCH" >&2
    echo "       (or override with DEPLOY_BRANCH=...)" >&2
    systemctl start orchestrator.timer
    exit 1
fi

# Sprint D rollback: tag the current commit as last-good BEFORE we
# update. If the deploy breaks and rollback.sh is invoked, this is
# the commit we revert to.
sudo -u "$SERVICE_USER" git tag --force "deploy/last-good" "$CURRENT" 2>/dev/null || true

# 2026-05-22: preserve runtime-written docs across the reset. The
# orchestrator writes live state into docs/*.json (cycle_status,
# trades_recent, self_grade, …) which are ALSO git-tracked (for the
# GH-Pages path). A bare `git reset --hard` clobbers the live data
# back to whatever stale snapshot is committed — observed reverting
# the droplet's fresh cycles to a 32h-old state on every deploy.
# Snapshot the live docs, reset code, then restore the live docs so
# the running system's data always wins over the committed copy.
_docs_bak="$(mktemp -d)"
cp -a "$INSTALL_DIR/docs/." "$_docs_bak/" 2>/dev/null || true

sudo -u "$SERVICE_USER" git reset --hard "origin/$BRANCH" --quiet
NEW="$(sudo -u "$SERVICE_USER" git rev-parse HEAD)"

# Restore live runtime JSON (NOT index.html — that's rebuilt fresh
# by dashboard.service from these inputs).
for f in cycle_status trades_recent benchmark validation walk_forward \
         improvements self_grade data_quality auto_overrides \
         hedge_fund_13f; do
    if [[ -f "$_docs_bak/$f.json" ]]; then
        cp -a "$_docs_bak/$f.json" "$INSTALL_DIR/docs/$f.json" 2>/dev/null || true
        chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/docs/$f.json" 2>/dev/null || true
    fi
done
rm -rf "$_docs_bak"

if [[ "$CURRENT" == "$NEW" ]]; then
    echo "  already at $CURRENT — nothing to do"
    systemctl start orchestrator.timer
    exit 0
fi
echo "  $CURRENT → $NEW"

echo "[3/6] Refreshing Python deps (only if requirements.txt changed)"
if sudo -u "$SERVICE_USER" git diff "$CURRENT" "$NEW" --name-only | grep -q "^requirements.txt$"; then
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet -r requirements.txt
else
    echo "  requirements unchanged"
fi

echo "[4/6] Refreshing systemd unit files (if changed)"
# 2026-05-22: glob ALL unit files instead of a hardcoded list. The
# old list (orchestrator scouts dashboard db-backup daily-digest)
# silently skipped dashboard-http.service when it was added later —
# the box ran fine but never served the live dashboard until a
# manual cp. Globbing means any NEW unit auto-installs on deploy.
for src in "$INSTALL_DIR"/deploy/systemd/*.service \
           "$INSTALL_DIR"/deploy/systemd/*.timer; do
    [[ -f "$src" ]] || continue
    dst="/etc/systemd/system/$(basename "$src")"
    if ! cmp -s "$src" "$dst"; then
        cp "$src" "$dst"
        echo "  updated $(basename "$src")"
    fi
done
systemctl daemon-reload

# Long-running services need an explicit restart to pick up new code
# (timers re-exec on each fire; daemons don't). Best-effort.
if [[ -f "/etc/systemd/system/dashboard-http.service" ]]; then
    systemctl enable --now dashboard-http.service 2>/dev/null || true
    systemctl try-restart dashboard-http.service 2>/dev/null || true
fi

# Sprint A2 — Enable db-backup timer on first deploy that includes it.
if [[ -f "/etc/systemd/system/db-backup.timer" ]]; then
    systemctl enable --now db-backup.timer 2>/dev/null || true
fi

# Sprint E2 — Enable daily-digest timer on first deploy that includes it.
if [[ -f "/etc/systemd/system/daily-digest.timer" ]]; then
    systemctl enable --now daily-digest.timer 2>/dev/null || true
fi

# 2026-05-22 — Enable research timer so the droplet refreshes the heavy
# validation + walk-forward backtests itself (GH Actions is throttled).
# Without this, research_freshness / overfit_resistance stay stale and
# new strategies never get a verdict.
if [[ -f "/etc/systemd/system/research.timer" ]]; then
    systemctl enable --now research.timer 2>/dev/null || true
fi

echo "[5/6] Restarting orchestrator timer"
systemctl start orchestrator.timer

echo "[6/6] Deploy complete (was $CURRENT, now $NEW). Roll back via:"
echo "      bash /opt/ai-at-advent/deploy/rollback.sh"
