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
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Sprint D rollback: tag the current commit as last-good BEFORE we
# update. If the deploy breaks and rollback.sh is invoked, this is
# the commit we revert to.
sudo -u "$SERVICE_USER" git tag --force "deploy/last-good" "$CURRENT" 2>/dev/null || true

sudo -u "$SERVICE_USER" git reset --hard "origin/$BRANCH" --quiet
NEW="$(sudo -u "$SERVICE_USER" git rev-parse HEAD)"

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
for unit in orchestrator scouts dashboard db-backup daily-digest; do
    src="$INSTALL_DIR/deploy/systemd/$unit.service"
    dst="/etc/systemd/system/$unit.service"
    [[ -f "$src" ]] || continue    # db-backup is new; tolerate older deploys
    if ! cmp -s "$src" "$dst"; then
        cp "$src" "$dst"
        if [[ -f "$INSTALL_DIR/deploy/systemd/$unit.timer" ]]; then
            cp "$INSTALL_DIR/deploy/systemd/$unit.timer" "/etc/systemd/system/$unit.timer"
        fi
        echo "  updated $unit"
    fi
done
systemctl daemon-reload

# Sprint A2 — Enable db-backup timer on first deploy that includes it.
if [[ -f "/etc/systemd/system/db-backup.timer" ]]; then
    systemctl enable --now db-backup.timer 2>/dev/null || true
fi

# Sprint E2 — Enable daily-digest timer on first deploy that includes it.
if [[ -f "/etc/systemd/system/daily-digest.timer" ]]; then
    systemctl enable --now daily-digest.timer 2>/dev/null || true
fi

echo "[5/6] Restarting orchestrator timer"
systemctl start orchestrator.timer

echo "[6/6] Deploy complete (was $CURRENT, now $NEW). Roll back via:"
echo "      bash /opt/ai-at-advent/deploy/rollback.sh"
