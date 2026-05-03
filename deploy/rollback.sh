#!/usr/bin/env bash
# Roll back to the last-known-good deploy.
#
# Sprint D audit fix: pre-fix, the only way to recover from a bad
# deploy was to revert the offending commit upstream and wait for
# the next CI run. That's slow when the orchestrator is mis-trading
# in the meantime. update.sh now tags every successful pre-update
# HEAD as `deploy/last-good`. This script resets to that tag,
# refreshes deps, and restarts the timer.
#
# Run on the VPS as root:
#     bash /opt/ai-at-advent/deploy/rollback.sh

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/ai-at-advent}"
SERVICE_USER="${SERVICE_USER:-aaa}"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root" >&2
    exit 1
fi

cd "$INSTALL_DIR"

# Resolve the rollback target. If the tag doesn't exist (e.g.
# brand-new install), bail with a clear error rather than guessing.
if ! sudo -u "$SERVICE_USER" git rev-parse "deploy/last-good" &>/dev/null; then
    echo "ERROR: 'deploy/last-good' tag missing — no known-good commit." >&2
    echo "  Set it manually with: git tag deploy/last-good <sha>" >&2
    exit 1
fi

CURRENT="$(sudo -u "$SERVICE_USER" git rev-parse HEAD)"
TARGET="$(sudo -u "$SERVICE_USER" git rev-parse deploy/last-good)"

if [[ "$CURRENT" == "$TARGET" ]]; then
    echo "Already at last-good ($CURRENT) — nothing to do."
    exit 0
fi

echo "Rolling back $CURRENT → $TARGET"

systemctl stop orchestrator.timer 2>/dev/null || true

# Tag the bad commit so we can investigate after recovery.
sudo -u "$SERVICE_USER" git tag --force "deploy/rolled-back-from" "$CURRENT" 2>/dev/null || true

sudo -u "$SERVICE_USER" git reset --hard "$TARGET" --quiet

# Re-install deps in case the rollback target needed older versions.
if [[ -f requirements.txt ]]; then
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet -r requirements.txt
fi

# Refresh systemd files in case the bad deploy changed them.
for unit in orchestrator scouts dashboard db-backup; do
    src="$INSTALL_DIR/deploy/systemd/$unit.service"
    [[ -f "$src" ]] || continue
    cp "$src" "/etc/systemd/system/$unit.service"
    if [[ -f "$INSTALL_DIR/deploy/systemd/$unit.timer" ]]; then
        cp "$INSTALL_DIR/deploy/systemd/$unit.timer" "/etc/systemd/system/$unit.timer"
    fi
done
systemctl daemon-reload

systemctl start orchestrator.timer
echo "Rollback complete. Now at $TARGET. Bad commit tagged as deploy/rolled-back-from."
